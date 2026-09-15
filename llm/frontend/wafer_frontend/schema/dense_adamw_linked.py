"""Strict two-step 1x1 Dense AdamW linked source; timing-only numerics."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .artifact_manifest import (
    LinkedProgramManifest, ManifestInputKind, RecordOpcode,
)
from .common import stable_artifact_id
from .flexible_dense_backward import FlexibleDenseBackwardLinkedProgram
from .persistent_state import StateKind
from .serde import canonical_digest
from .workload_materialization import WorkloadMaterializationManifest
from .workload_run import WorkloadFamily, WorkloadOptimizerKind


_SCHEMA = "wafer_frontend.dense_adamw_linked_program/v1alpha1"


@dataclass(frozen=True, slots=True)
class DenseAdamwLinkedProgram:
    schema_version: str
    producer_pass: str
    id: str
    materialization: WorkloadMaterializationManifest
    backward_source: FlexibleDenseBackwardLinkedProgram
    manifest: LinkedProgramManifest
    step_index: int

    @classmethod
    def create(cls, **semantic: object) -> "DenseAdamwLinkedProgram":
        result = cls(
            _SCHEMA,
            "dense_adamw_linker",
            stable_artifact_id(
                "dense_adamw_linked_program", semantic, schema_version=_SCHEMA,
            ),
            **semantic,
        )
        result.validate()
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self)

    def validate(self, path: str = "dense_adamw_linked_program") -> None:
        if self.schema_version != _SCHEMA or self.producer_pass != "dense_adamw_linker":
            raise SchemaError("unsupported AdamW linked source", path=path)
        self.materialization.validate(f"{path}.materialization")
        self.backward_source.validate(f"{path}.backward_source")
        self.manifest.validate(f"{path}.manifest")
        request = self.materialization.request
        if (
            request.family is not WorkloadFamily.DENSE_TRAINING
            or request.optimizer is None
            or request.optimizer.kind is not WorkloadOptimizerKind.ADAMW
            or (request.mesh.columns, request.mesh.rows) != (1, 1)
            or request.steps.training is None
            or request.steps.training.step_count != 2
            or type(self.step_index) is not int
            or self.step_index not in (0, 1)
            or request.model.num_layers != 2
            or request.model.hidden_size != 16
            or request.model.intermediate_size != 1
        ):
            raise SchemaError(
                "AdamW executable subset requires two H16/I1 layers, two steps, and 1x1",
                path=path,
            )
        old = self.backward_source.manifest
        if (
            self.manifest.producer_pass != self.producer_pass
            or self.manifest.source_ir1_id != old.source_ir1_id
            or self.manifest.source_projection_id != old.source_projection_id
            or self.manifest.source_schedule_set_id != old.source_schedule_set_id
            or self.manifest.source_global_dag_id != old.source_global_dag_id
        ):
            raise SchemaError("AdamW must preserve the WGRAD source lineage", path=path)
        source_digest = tuple(
            digest for digest in self.manifest.input_digests
            if digest.kind is ManifestInputKind.DENSE_ADAMW_SOURCE
        )
        if (
            len(source_digest) != 1
            or source_digest[0].artifact_id != self.materialization.id
            or source_digest[0].digest != canonical_digest(self.materialization)
        ):
            raise SchemaError("AdamW manifest must sign the real P3 source", path=path)
        if len(self.manifest.fragments) != 1:
            raise SchemaError("1x1 AdamW requires one production fragment", path=path)
        fragment = self.manifest.fragments[0]
        if len(fragment.core_streams) != 1:
            raise SchemaError("1x1 AdamW requires one production command stream", path=path)
        records = fragment.core_streams[0].records
        operations = {
            op.id: op for op in self.materialization.logical_graph.operations
            if op.kind.value == "adamw_update" and op.step == self.step_index
        }
        updates = tuple(r for r in records if r.opcode is RecordOpcode.ADAMW_UPDATE)
        if (
            len(operations) != 17
            or len(updates) != 17
            or {r.source_global_action_id for r in updates} != set(operations)
            or any(r.opcode is RecordOpcode.SGD_UPDATE for r in records)
        ):
            raise SchemaError("17 real AdamW operations must replace all SGD actions", path=path)
        state = fragment.state_abi
        if (
            len(state) != 83
            or sum(a.kind is StateKind.TRAINABLE_PARAMETER for a in state) != 15
            or len({a.state_ref for a in state}) != 83
        ):
            raise SchemaError("15 packed weights and 68 independent optimizer states required", path=path)
        logical_optimizer = {
            item.logical_name: item
            for item in self.materialization.logical_graph.state_versions
            if item.version == 0 and item.kind.value.startswith("optimizer_")
        }
        physical_optimizer = {
            item.state_ref: item for item in state
            if item.kind is not StateKind.TRAINABLE_PARAMETER
        }
        if set(physical_optimizer) != set(logical_optimizer) or any(
            physical_optimizer[name].kind.value != logical.kind.value
            for name, logical in logical_optimizer.items()
        ):
            raise SchemaError("StateABI must close all E2E AdamW version-0 states", path=path)
        physical_parameter_bytes = sum(
            item.size_bytes for item in state
            if item.kind is StateKind.TRAINABLE_PARAMETER
        )
        logical_parameter_bytes = sum(
            item.size_bytes for item in self.materialization.state_inventory
            if item.object_kind.value == "parameter"
        )
        if physical_parameter_bytes != logical_parameter_bytes:
            raise SchemaError("packed physical weight ABI differs from logical P3 parameters", path=path)


__all__ = ["DenseAdamwLinkedProgram"]
