"""Typed full-model MoE compile sequence assembled from production components."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .artifact_manifest import LinkedProgramManifest, RecordOpcode
from .dense_compile_sequence import expected_segment_profile
from .e2e_workload_graph import E2EOperationKind
from .flexible_moe import MoeRectActionKind, MoeRectFlowStage
from .ir0 import StateAccessMode
from .moe_compile_sequence import MoeCompileSequence
from .n6 import LinkedProgramProfile
from .persistent_state import StateKind
from .serde import canonical_digest
from .workload_run import WorkloadFamily


MOE_FULL_MODEL_SEQUENCE_SCHEMA_VERSION = (
    "wafer_frontend.moe_full_model_compile_sequence/v1alpha1"
)
MOE_FULL_MODEL_SEGMENT_SCHEMA_VERSION = (
    "wafer_frontend.moe_full_model_compile_segment/v1alpha1"
)
MOE_FULL_MODEL_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.moe_full_model_operation_binding/v1alpha1"
)


class MoeFullModelCoverage(str, Enum):
    FULL_MODEL = "full_model"


class MoeFullModelCompileStatus(str, Enum):
    FULL_MODEL_RUNTIME_LINKED = "full_model_runtime_linked"


class MoeFullModelRuntimeStatus(str, Enum):
    EXECUTABLE_MANIFEST_MATERIALIZED = "executable_manifest_materialized"


class MoeFullModelLowering(str, Enum):
    SHARED_SPINE_NODE = "shared_spine_node"
    SHARED_SPINE_STATE_ACCESS = "shared_spine_state_access"
    SHARED_SPINE_VALUE = "shared_spine_value"
    FLEXIBLE_MOE_ACTION = "flexible_moe_action"
    FLEXIBLE_MOE_TRACE = "flexible_moe_trace"


@dataclass(frozen=True, slots=True)
class MoeFullModelOperationBinding:
    id: str
    operation_ref: str
    kind: E2EOperationKind
    step: int
    layer: int | None
    lowering: MoeFullModelLowering
    production_refs: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "MoeFullModelOperationBinding":
        result = cls(
            id=stable_artifact_id(
                "moe_full_model_operation_binding",
                semantic,
                schema_version=MOE_FULL_MODEL_BINDING_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "moe_full_model_operation_binding") -> None:
        validate_nonempty(self.operation_ref, f"{path}.operation_ref")
        if type(self.kind) is not E2EOperationKind:
            raise SchemaError("must use a typed operation kind", path=f"{path}.kind")
        validate_uint64(self.step, f"{path}.step")
        if self.layer is not None:
            validate_uint64(self.layer, f"{path}.layer")
        if type(self.lowering) is not MoeFullModelLowering:
            raise SchemaError("must use a typed lowering", path=f"{path}.lowering")
        if not self.production_refs or len(set(self.production_refs)) != len(
            self.production_refs
        ):
            raise SchemaError(
                "production refs must be non-empty and unique",
                path=f"{path}.production_refs",
            )
        for index, ref in enumerate(self.production_refs):
            validate_nonempty(ref, f"{path}.production_refs[{index}]")
        expected = stable_artifact_id(
            "moe_full_model_operation_binding",
            self._semantic(),
            schema_version=MOE_FULL_MODEL_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable binding id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeFullModelCompileSegment:
    id: str
    phase: str
    step: int
    shared_spine_profile: LinkedProgramProfile
    shared_spine_digest: str
    replica_die_ids: tuple[int, ...]
    replaced_dense_mlp_node_refs: tuple[str, ...]
    operation_bindings: tuple[MoeFullModelOperationBinding, ...]
    moe_unit_refs: tuple[str, ...]
    executable_manifest: LinkedProgramManifest
    executable_manifest_digest: str

    @classmethod
    def create(cls, **semantic: object) -> "MoeFullModelCompileSegment":
        result = cls(
            id=stable_artifact_id(
                "moe_full_model_compile_segment",
                semantic,
                schema_version=MOE_FULL_MODEL_SEGMENT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "moe_full_model_compile_segment") -> None:
        if self.phase not in ("prefill", "decode"):
            raise SchemaError("unsupported inference phase", path=f"{path}.phase")
        validate_uint64(self.step, f"{path}.step")
        self.shared_spine_profile.validate(f"{path}.shared_spine_profile")
        if self.shared_spine_digest != canonical_digest(self.shared_spine_profile):
            raise SchemaError("shared-spine digest mismatch", path=f"{path}.shared_spine_digest")
        if (
            not self.replica_die_ids
            or self.replica_die_ids != tuple(sorted(set(self.replica_die_ids)))
        ):
            raise SchemaError("replica dies must be canonical", path=f"{path}.replica_die_ids")
        for die in self.replica_die_ids:
            validate_uint64(die, f"{path}.replica_die_ids")
        if (
            not self.replaced_dense_mlp_node_refs
            or len(set(self.replaced_dense_mlp_node_refs))
            != len(self.replaced_dense_mlp_node_refs)
        ):
            raise SchemaError(
                "must explicitly identify every replaced Dense MLP node",
                path=f"{path}.replaced_dense_mlp_node_refs",
            )
        for index, ref in enumerate(self.replaced_dense_mlp_node_refs):
            validate_nonempty(ref, f"{path}.replaced_dense_mlp_node_refs[{index}]")
        if not self.operation_bindings:
            raise SchemaError("must bind model operations", path=f"{path}.operation_bindings")
        refs = []
        for index, binding in enumerate(self.operation_bindings):
            if type(binding) is not MoeFullModelOperationBinding:
                raise SchemaError("must carry typed bindings", path=f"{path}.operation_bindings[{index}]")
            binding.validate(f"{path}.operation_bindings[{index}]")
            if binding.step != self.step:
                raise SchemaError("binding belongs to another step", path=f"{path}.operation_bindings[{index}]")
            refs.append(binding.operation_ref)
        if len(set(refs)) != len(refs):
            raise SchemaError("operation is lowered more than once", path=f"{path}.operation_bindings")
        if not self.moe_unit_refs or len(set(self.moe_unit_refs)) != len(self.moe_unit_refs):
            raise SchemaError("MoE unit refs must be non-empty and unique", path=f"{path}.moe_unit_refs")
        self.executable_manifest.validate(f"{path}.executable_manifest")
        if self.executable_manifest_digest != canonical_digest(self.executable_manifest):
            raise SchemaError("executable manifest digest mismatch", path=f"{path}.executable_manifest_digest")
        if tuple(item.logical_core.die_id for item in self.executable_manifest.core_streams) != self.replica_die_ids:
            raise SchemaError("executable manifest must cover every EP die", path=f"{path}.executable_manifest")
        expected = stable_artifact_id(
            "moe_full_model_compile_segment",
            self._semantic(),
            schema_version=MOE_FULL_MODEL_SEGMENT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable segment id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeFullModelCompileSequence:
    schema_version: str
    producer_pass: str
    id: str
    moe_blocks: MoeCompileSequence
    moe_blocks_digest: str
    segments: tuple[MoeFullModelCompileSegment, ...]
    coverage: MoeFullModelCoverage
    compile_status: MoeFullModelCompileStatus
    runtime_status: MoeFullModelRuntimeStatus

    @classmethod
    def create(
        cls,
        *,
        moe_blocks: MoeCompileSequence,
        segments: tuple[MoeFullModelCompileSegment, ...],
    ) -> "MoeFullModelCompileSequence":
        semantic = {
            "moe_blocks": moe_blocks,
            "moe_blocks_digest": canonical_digest(moe_blocks),
            "segments": segments,
            "coverage": MoeFullModelCoverage.FULL_MODEL,
            "compile_status": MoeFullModelCompileStatus.FULL_MODEL_RUNTIME_LINKED,
            "runtime_status": MoeFullModelRuntimeStatus.EXECUTABLE_MANIFEST_MATERIALIZED,
        }
        result = cls(
            schema_version=MOE_FULL_MODEL_SEQUENCE_SCHEMA_VERSION,
            producer_pass="compile_moe_full_model_inference_sequence",
            id=stable_artifact_id(
                "moe_full_model_compile_sequence",
                semantic,
                schema_version=MOE_FULL_MODEL_SEQUENCE_SCHEMA_VERSION,
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
    def materialization(self):
        return self.moe_blocks.materialization

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "moe_full_model_compile_sequence") -> None:
        if self.schema_version != MOE_FULL_MODEL_SEQUENCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "compile_moe_full_model_inference_sequence":
            raise SchemaError("unexpected producer", path=f"{path}.producer_pass")
        self.moe_blocks.validate(f"{path}.moe_blocks")
        request = self.materialization.request
        if request.family is not WorkloadFamily.MOE_INFERENCE:
            raise SchemaError("requires MoE inference", path=f"{path}.moe_blocks")
        if self.moe_blocks_digest != canonical_digest(self.moe_blocks):
            raise SchemaError("MoE block digest mismatch", path=f"{path}.moe_blocks_digest")
        if (
            self.coverage is not MoeFullModelCoverage.FULL_MODEL
            or self.compile_status is not MoeFullModelCompileStatus.FULL_MODEL_RUNTIME_LINKED
            or self.runtime_status is not MoeFullModelRuntimeStatus.EXECUTABLE_MANIFEST_MATERIALIZED
        ):
            raise SchemaError("coverage/runtime claim drifted", path=path)
        inference = request.steps.inference
        assert inference is not None
        expected_steps = tuple(range(inference.decode_steps + 1))
        if tuple(segment.step for segment in self.segments) != expected_steps:
            raise SchemaError("segments must cover every inference step", path=f"{path}.segments")
        operations = {item.id: item for item in self.materialization.logical_graph.operations}
        moe_units = {item.id: item for item in self.moe_blocks.units}
        all_bound: list[str] = []
        for index, segment in enumerate(self.segments):
            segment_path = f"{path}.segments[{index}]"
            segment.validate(segment_path)
            expected_phase = "prefill" if index == 0 else "decode"
            if segment.phase != expected_phase:
                raise SchemaError("segment phase drifted", path=segment_path)
            if segment.replica_die_ids != self.materialization.placement.active_die_ids:
                raise SchemaError("shared spine must cover every active EP rank", path=f"{segment_path}.replica_die_ids")
            expected_unit_refs = tuple(
                item.id for item in self.moe_blocks.units if item.step == segment.step
            )
            if segment.moe_unit_refs != expected_unit_refs or any(
                ref not in moe_units for ref in segment.moe_unit_refs
            ):
                raise SchemaError("MoE unit closure differs", path=f"{segment_path}.moe_unit_refs")
            expected_replaced = tuple(
                f"{segment.shared_spine_profile.lowering_context.ir1.instances[0].origin_instance_id}.layer{layer}.{name}"
                for layer in range(request.model.num_layers)
                for name in ("gate_up", "swiglu", "down")
            )
            if segment.replaced_dense_mlp_node_refs != expected_replaced:
                raise SchemaError("Dense MLP replacement set is not exact", path=f"{segment_path}.replaced_dense_mlp_node_refs")
            dense_actions = {
                action.id: getattr(action.origin_ref, "op_id", getattr(action.origin_ref, "node_ref", None))
                for action in segment.shared_spine_profile.lowering_context.global_dag.actions
            }
            executable_actions = {
                record.source_global_action_id
                for fragment in segment.executable_manifest.fragments
                for stream in fragment.core_streams
                for record in stream.records
            }
            replaced_actions = {
                action_id for action_id, node_ref in dense_actions.items()
                if node_ref in set(expected_replaced)
            }
            retained_actions = set(dense_actions) - replaced_actions
            expected_moe_actions = {
                stable_artifact_id(
                    "moe_full_model_action",
                    {
                        "source": segment.executable_manifest.source_global_dag_id,
                        "layer": moe_units[ref].layer,
                        "action": action_id,
                    },
                    schema_version="wafer_frontend.moe_full_model_region_linker/v1alpha1",
                )
                for ref in segment.moe_unit_refs
                for action_id in {
                    record.source_global_action_id
                    for fragment in moe_units[ref].linked_manifest.fragments
                    for stream in fragment.core_streams
                    for record in stream.records
                }
            }
            if (
                not replaced_actions
                or executable_actions & replaced_actions
                or not retained_actions.issubset(executable_actions)
                or not expected_moe_actions.issubset(executable_actions)
            ):
                raise SchemaError("executable manifest is not an exact Dense-MLP region replacement", path=f"{segment_path}.executable_manifest")
            opcodes = {
                record.opcode
                for fragment in segment.executable_manifest.fragments
                for stream in fragment.core_streams
                for record in stream.records
            }
            required_opcodes = {
                RecordOpcode.EMBEDDING_LOOKUP,
                RecordOpcode.ATTENTION_EXACT,
                RecordOpcode.LOCAL_REDUCE,
                RecordOpcode.MATMUL,
            }
            has_remote_flow = any(
                moe_units[ref].plan.flows for ref in segment.moe_unit_refs
            )
            if has_remote_flow:
                required_opcodes.update({
                    RecordOpcode.DTE_SEND,
                    RecordOpcode.DTE_RECV,
                })
            if not required_opcodes.issubset(opcodes):
                raise SchemaError("executable manifest lacks shared-spine/MoE/head records", path=f"{segment_path}.executable_manifest")
            ir1 = segment.shared_spine_profile.lowering_context.ir1
            if (
                ir1.fabric.die_grid != (1, 1)
                or ir1.profile != expected_segment_profile(self.materialization, index)
                or segment.shared_spine_profile.profile_id
                != ir1.profile.stable_id()
            ):
                raise SchemaError(
                    "shared spine must be an exact TP1 rank-local profile",
                    path=f"{segment_path}.shared_spine_profile",
                )
            ir1_nodes = {item.origin_node_id for item in ir1.nodes}
            ir1_values = {item.id for item in ir1.values}
            ir1_accesses = {item.id for item in ir1.state_accesses}
            if not set(expected_replaced).issubset(ir1_nodes):
                raise SchemaError("replacement node is absent from shared spine", path=segment_path)
            observed_step = []
            for binding in segment.operation_bindings:
                operation = operations.get(binding.operation_ref)
                if operation is None or (
                    operation.kind,
                    operation.step,
                    operation.layer,
                ) != (binding.kind, binding.step, binding.layer):
                    raise SchemaError("P3 operation binding drifted", path=f"{segment_path}.operation_bindings")
                if binding.lowering is MoeFullModelLowering.SHARED_SPINE_NODE:
                    valid = ir1_nodes
                elif binding.lowering is MoeFullModelLowering.SHARED_SPINE_VALUE:
                    valid = ir1_values
                elif binding.lowering is MoeFullModelLowering.SHARED_SPINE_STATE_ACCESS:
                    valid = ir1_accesses
                else:
                    valid = None
                if valid is not None and not set(binding.production_refs).issubset(valid):
                    raise SchemaError("shared-spine production ref is unknown", path=f"{segment_path}.operation_bindings")
                expected_lowering, expected_refs = self._expected_production_binding(
                    operation,
                    segment,
                    moe_units,
                    tuple(
                        item
                        for item in self.materialization.logical_graph.operations
                        if item.step == segment.step
                    ),
                )
                if (
                    binding.lowering is not expected_lowering
                    or binding.production_refs != expected_refs
                ):
                    raise SchemaError(
                        f"production lineage does not exactly lower the P3 operation {operation.kind.value}: {binding.production_refs!r} != {expected_refs!r}",
                        path=f"{segment_path}.operation_bindings",
                    )
                observed_step.append(binding.operation_ref)
            expected_step = tuple(
                item.id
                for item in self.materialization.logical_graph.operations
                if item.step == segment.step
            )
            if tuple(observed_step) != expected_step:
                raise SchemaError("segment does not exactly cover P3 operations in order", path=f"{segment_path}.operation_bindings")
            all_bound.extend(observed_step)
        if tuple(all_bound) != tuple(item.id for item in self.materialization.logical_graph.operations):
            raise SchemaError("sequence is not a full graph cover", path=f"{path}.segments")
        expected = stable_artifact_id(
            "moe_full_model_compile_sequence",
            self._semantic(),
            schema_version=MOE_FULL_MODEL_SEQUENCE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable sequence id", path=f"{path}.id")

    @staticmethod
    def _expected_production_binding(
        operation,
        segment: MoeFullModelCompileSegment,
        moe_units,
        step_operations,
    ) -> tuple[MoeFullModelLowering, tuple[str, ...]]:
        kind = operation.kind
        ir1 = segment.shared_spine_profile.lowering_context.ir1
        origin = ir1.instances[0].origin_instance_id
        if kind in (
            E2EOperationKind.ROUTER,
            E2EOperationKind.ROUTE_FREEZE,
            E2EOperationKind.DISPATCH,
            E2EOperationKind.EXPERT_FORWARD,
            E2EOperationKind.COMBINE,
        ):
            units = tuple(
                moe_units[ref]
                for ref in segment.moe_unit_refs
                if moe_units[ref].layer == operation.layer
            )
            if len(units) != 1:
                raise SchemaError("requires one exact layer MoE unit", path="moe_unit_refs")
            unit = units[0]
            if kind is E2EOperationKind.ROUTE_FREEZE:
                return MoeFullModelLowering.FLEXIBLE_MOE_TRACE, (unit.spec.trace.id,)
            if kind is E2EOperationKind.ROUTER:
                refs = tuple(
                    item.id
                    for item in unit.plan.actions
                    if item.kind is MoeRectActionKind.GATE
                )
            elif kind is E2EOperationKind.EXPERT_FORWARD:
                refs = tuple(
                    item.id
                    for item in unit.plan.actions
                    if item.kind is MoeRectActionKind.EXPERT_FORWARD
                    and item.rank == operation.expert
                )
            else:
                primary = (
                    MoeRectActionKind.PACK
                    if kind is E2EOperationKind.DISPATCH
                    else MoeRectActionKind.WEIGHTED_COMBINE
                )
                stage = (
                    MoeRectFlowStage.DISPATCH
                    if kind is E2EOperationKind.DISPATCH
                    else MoeRectFlowStage.COMBINE
                )
                flow_refs = {item.id for item in unit.plan.flows if item.stage is stage}
                refs = tuple(dict.fromkeys(
                    item.id
                    for item in unit.plan.actions
                    if item.kind is primary or item.flow_ref in flow_refs
                ))
            if not refs:
                raise SchemaError("MoE operation has no production action", path="moe_unit_refs")
            return MoeFullModelLowering.FLEXIBLE_MOE_ACTION, refs
        if kind in (E2EOperationKind.KV_LOAD, E2EOperationKind.KV_APPEND):
            attention_ref = f"{origin}.layer{operation.layer}.attention"
            physical_refs = {
                item.id for item in ir1.nodes if item.origin_node_id == attention_ref
            }
            manifest = ir1.persistent_state_manifest
            if manifest is None:
                raise SchemaError("shared spine lacks persistent state", path="shared_spine_profile")
            states = {item.id: item for item in manifest.declarations}
            refs = tuple(
                item.id
                for item in ir1.state_accesses
                if item.node_ref in physical_refs
                and states[item.state_ref].identity.kind
                in (StateKind.KV_KEY, StateKind.KV_VALUE)
                and item.mode in (StateAccessMode.WRITE, StateAccessMode.READ_WRITE)
            )
            return MoeFullModelLowering.SHARED_SPINE_STATE_ACCESS, refs
        if kind is E2EOperationKind.LOGITS:
            return MoeFullModelLowering.SHARED_SPINE_VALUE, (f"{origin}.logits",)
        suffix = {
            E2EOperationKind.EMBEDDING: "embedding",
            E2EOperationKind.INPUT_NORM: f"layer{operation.layer}.norm1",
            E2EOperationKind.QKV: f"layer{operation.layer}.qkv",
            E2EOperationKind.ROPE: f"layer{operation.layer}.rope",
            E2EOperationKind.ATTENTION: f"layer{operation.layer}.attention",
            E2EOperationKind.ATTENTION_OUT: f"layer{operation.layer}.o",
            E2EOperationKind.POST_NORM: f"layer{operation.layer}.norm2",
            E2EOperationKind.FINAL_NORM: "final_norm",
            E2EOperationKind.LM_HEAD: "lm_head",
        }.get(kind)
        if kind is E2EOperationKind.RESIDUAL:
            preceding = tuple(
                item
                for item in step_operations
                if item.layer == operation.layer
                and item.kind is E2EOperationKind.RESIDUAL
                and item.sequence_index <= operation.sequence_index
            )
            suffix = f"layer{operation.layer}.residual{len(preceding)}"
        if suffix is None:
            raise SchemaError("P3 operation lacks a shared-spine lowering", path="operation.kind")
        return MoeFullModelLowering.SHARED_SPINE_NODE, (f"{origin}.{suffix}",)


__all__ = [
    "MOE_FULL_MODEL_SEQUENCE_SCHEMA_VERSION",
    "MoeFullModelCompileSegment",
    "MoeFullModelCompileSequence",
    "MoeFullModelCompileStatus",
    "MoeFullModelCoverage",
    "MoeFullModelLowering",
    "MoeFullModelOperationBinding",
    "MoeFullModelRuntimeStatus",
]
