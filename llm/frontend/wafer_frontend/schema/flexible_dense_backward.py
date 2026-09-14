"""Real command-manifest carrier for flexible Dense backward training."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ._validation_session import mark_validation_complete, validation_seen
from .artifact_manifest import (
    LinkedProgramManifest,
    ManifestInputKind,
    RecordOpcode,
)
from .common import stable_artifact_id, validate_uint64
from .flexible_dense_train import (
    FlexibleDenseTrainActionKind,
    FlexibleDenseTrainGradientSyncRole,
    FlexibleDenseTrainPlan,
)
from .flexible_dense_backward_ir import (
    FlexibleDenseBackwardGlobalDAG,
    FlexibleDenseBackwardIR,
    FlexibleDenseBackwardProjection,
    FlexibleDenseBackwardSchedule,
)
from .ir1 import PhysicalFabric
from .persistent_state import (
    HbmAddressSpace,
    PersistentStateAccess,
    StateKind,
)
from .serde import canonical_digest
from .train_n6 import TrainLinkedProgram


FLEXIBLE_DENSE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.flexible_dense_backward_linked_program/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class FlexibleDenseBackwardLinkedProgram:
    """One real symbolic ProgramArtifact manifest; runtime evidence is separate."""

    schema_version: str
    producer_pass: str
    id: str
    plan: FlexibleDenseTrainPlan
    forward_lineage: TrainLinkedProgram
    backward_ir: FlexibleDenseBackwardIR
    backward_projection: FlexibleDenseBackwardProjection
    backward_schedule: FlexibleDenseBackwardSchedule
    backward_global_dag: FlexibleDenseBackwardGlobalDAG
    fabric: PhysicalFabric
    hbm_address_spaces: tuple[HbmAddressSpace, ...]
    manifest: LinkedProgramManifest
    record_count: int
    runtime_verified: bool

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleDenseBackwardLinkedProgram":
        result = cls(
            FLEXIBLE_DENSE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION,
            "flexible_dense_backward_linker",
            stable_artifact_id(
                "flexible_dense_backward_linked_program",
                semantic,
                schema_version=(
                    FLEXIBLE_DENSE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION
                ),
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self)

    def validate(
        self, path: str = "flexible_dense_backward_linked_program"
    ) -> None:
        if validation_seen(self, "flexible_dense_backward_linked_program"):
            return
        if (
            self.schema_version
            != FLEXIBLE_DENSE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION
            or self.producer_pass != "flexible_dense_backward_linker"
        ):
            raise SchemaError("unsupported linked program", path=path)
        self.plan.validate(f"{path}.plan")
        self.forward_lineage.validate(f"{path}.forward_lineage")
        if (
            self.forward_lineage.source.dp_degree != self.plan.spec.dp_degree
            or any(
                replica.lowering_context.ir1.source_ir0_id
                != self.plan.forward_graph.id
                for replica in self.forward_lineage.source.replicas
            )
        ):
            raise SchemaError(
                "forward lineage does not derive from the exact plan",
                path=f"{path}.forward_lineage",
            )
        self.backward_ir.validate(f"{path}.backward_ir")
        self.backward_projection.validate(f"{path}.backward_projection")
        self.backward_schedule.validate(f"{path}.backward_schedule")
        self.backward_global_dag.validate(f"{path}.backward_global_dag")
        if (
            self.backward_ir.source_plan_id != self.plan.id
            or self.backward_projection.source_ir_id != self.backward_ir.id
            or self.backward_schedule.source_projection_id
            != self.backward_projection.id
            or self.backward_global_dag.source_schedule_id
            != self.backward_schedule.id
        ):
            raise SchemaError(
                "backward four-stage lineage is not an exact chain",
                path=f"{path}.backward_ir",
            )
        self.fabric.validate(f"{path}.fabric")
        if (
            self.fabric.die_grid != self.plan.spec.mesh.physical_shape
            or len(self.fabric.dies) != self.plan.spec.mesh.rank_count
        ):
            raise SchemaError("fabric does not match Mesh", path=f"{path}.fabric")
        if tuple(item.die_id for item in self.hbm_address_spaces) != tuple(
            range(self.plan.spec.mesh.rank_count)
        ):
            raise SchemaError(
                "HBM spaces must exactly cover rank/die ids",
                path=f"{path}.hbm_address_spaces",
            )
        for index, space in enumerate(self.hbm_address_spaces):
            space.validate(f"{path}.hbm_address_spaces[{index}]")
        self.manifest.validate(f"{path}.manifest")
        if (
            self.manifest.producer_pass != "flexible_dense_backward_linker"
            or self.manifest.source_ir1_id != self.backward_ir.id
            or self.manifest.source_projection_id
            != self.backward_projection.id
            or self.manifest.source_schedule_set_id
            != self.backward_schedule.id
            or self.manifest.source_global_dag_id
            != self.backward_global_dag.id
        ):
            raise SchemaError(
                "manifest provenance does not identify the exact plan",
                path=f"{path}.manifest",
            )
        expected_lineage = tuple(sorted((
            (ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_IR,
             self.backward_ir.id, self.backward_ir.schema_version,
             self.backward_ir.digest),
            (ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_PROJECTION,
             self.backward_projection.id, self.backward_projection.schema_version,
             self.backward_projection.digest),
            (ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_SCHEDULE,
             self.backward_schedule.id, self.backward_schedule.schema_version,
             self.backward_schedule.digest),
            (ManifestInputKind.FLEXIBLE_DENSE_BACKWARD_GLOBAL,
             self.backward_global_dag.id, self.backward_global_dag.schema_version,
             self.backward_global_dag.digest),
        ), key=lambda item: (item[0].value, item[1])))
        actual_lineage = tuple(
            (digest.kind, digest.artifact_id, digest.schema_version, digest.digest)
            for digest in self.manifest.input_digests
            if digest.kind is not ManifestInputKind.COMMAND_FRAGMENT
        )
        if actual_lineage != expected_lineage:
            raise SchemaError(
                "manifest must preserve the exact production Train forward lineage",
                path=f"{path}.manifest.input_digests",
            )
        records = tuple(
            record
            for fragment in self.manifest.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        validate_uint64(self.record_count, f"{path}.record_count")
        if self.record_count != len(records) or not records:
            raise SchemaError("record count does not close", path=path)
        actions = {
            item.id: item
            for item in self.plan.rank_actions
            if item.kind is not FlexibleDenseTrainActionKind.FORWARD
        }
        records_by_action = {
            action_id: tuple(
                record
                for record in records
                if record.source_global_action_id == action_id
            )
            for action_id in actions
        }
        if any(not values for values in records_by_action.values()) or {
            record.source_global_action_id for record in records
        } != set(actions):
            raise SchemaError(
                "manifest must cover every non-forward plan action",
                path=f"{path}.manifest.fragments",
            )
        expected_opcodes = {
            FlexibleDenseTrainActionKind.PARAMETER_LOAD: (
                RecordOpcode.SRAM_ALLOC_AT,
                RecordOpcode.SRAM_ALLOC_AT,
                RecordOpcode.LSU_LOAD,
            ),
            FlexibleDenseTrainActionKind.BACKWARD: (
                RecordOpcode.SRAM_BIND,
                RecordOpcode.MATMUL,
            ),
            FlexibleDenseTrainActionKind.WEIGHT_GRADIENT: (
                RecordOpcode.SRAM_BIND,
                RecordOpcode.MATMUL,
            ),
            FlexibleDenseTrainActionKind.SGD_UPDATE: (
                RecordOpcode.SRAM_BIND,
                RecordOpcode.SGD_UPDATE,
            ),
            FlexibleDenseTrainActionKind.PARAMETER_STORE: (
                RecordOpcode.LSU_STORE,
                RecordOpcode.SRAM_FREE,
                RecordOpcode.SRAM_FREE,
            ),
        }
        for action_id, action in actions.items():
            actual = tuple(item.opcode for item in records_by_action[action_id])
            if action.kind is FlexibleDenseTrainActionKind.GRADIENT_SYNC:
                expected_sync = {
                    FlexibleDenseTrainGradientSyncRole.REDUCE_SEND: (
                        RecordOpcode.DTE_SEND,
                    ),
                    FlexibleDenseTrainGradientSyncRole.REDUCE_RECEIVE: (
                        RecordOpcode.DTE_RECV, RecordOpcode.DTE_WAIT,
                        RecordOpcode.LOCAL_REDUCE,
                    ),
                    FlexibleDenseTrainGradientSyncRole.BROADCAST_SEND: (
                        RecordOpcode.DTE_SEND,
                    ),
                    FlexibleDenseTrainGradientSyncRole.BROADCAST_RECEIVE: (
                        RecordOpcode.DTE_RECV, RecordOpcode.DTE_WAIT,
                    ),
                }
                if self.plan.spec.dp_degree == 1 or actual != expected_sync.get(
                    action.gradient_sync_role
                ):
                    raise SchemaError(
                        "DP tree sync opcode quotient drifted",
                        path=f"{path}.manifest.fragments",
                    )
            elif actual != expected_opcodes[action.kind]:
                raise SchemaError(
                    "action opcode quotient drifted",
                    path=f"{path}.manifest.fragments",
                )
        state_abis = tuple(
            abi
            for fragment in self.manifest.fragments
            for abi in fragment.state_abi
        )
        expected_state_owners = {
            (template.state_ref, rank)
            for template in self.plan.parameter_templates
            for rank in template.owner_ranks
        }
        if {(item.state_ref, item.die_id) for item in state_abis} != expected_state_owners:
            raise SchemaError(
                "StateABI must exactly cover every physical parameter owner",
                path=f"{path}.manifest.fragments.state_abi",
            )
        template_by_state = {
            item.state_ref: item for item in self.plan.parameter_templates
        }
        for abi in state_abis:
            template = template_by_state[abi.state_ref]
            space = self.hbm_address_spaces[abi.die_id]
            if (
                abi.kind is not StateKind.TRAINABLE_PARAMETER
                or abi.access is not PersistentStateAccess.READ_WRITE
                or abi.size_bytes != template.weight_bytes
                or abi.address < space.base_address
                or abi.address + abi.size_bytes
                > space.base_address + space.size_bytes
            ):
                raise SchemaError(
                    "trainable StateABI ownership/span drifted",
                    path=f"{path}.manifest.fragments.state_abi",
                )
        for action_id, action in actions.items():
            if action.kind is not FlexibleDenseTrainActionKind.SGD_UPDATE:
                continue
            record = next(
                item
                for item in records_by_action[action_id]
                if item.opcode is RecordOpcode.SGD_UPDATE
            )
            operands = {item.name: item for item in record.operands}
            if (
                operands["weight_address"].symbol_ref
                != operands["updated_weight_address"].symbol_ref
            ):
                raise SchemaError(
                    "SGD must update the exact owned weight buffer in place",
                    path=f"{path}.manifest.fragments",
                )
        if self.runtime_verified is not False:
            raise SchemaError(
                "linked manifest alone is not runtime evidence",
                path=f"{path}.runtime_verified",
            )
        expected_id = stable_artifact_id(
            "flexible_dense_backward_linked_program",
            self._semantic(),
            schema_version=FLEXIBLE_DENSE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable linked program id; expected {expected_id!r}",
                path=f"{path}.id",
            )
        mark_validation_complete(self, "flexible_dense_backward_linked_program")


__all__ = [
    "FLEXIBLE_DENSE_BACKWARD_LINKED_PROGRAM_SCHEMA_VERSION",
    "FlexibleDenseBackwardLinkedProgram",
]
