"""Two-step P3 Dense training sequence bound to a real one-step lowerer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .e2e_workload_graph import E2EOperationKind, E2EStateKind
from .flexible_dense_backward import FlexibleDenseBackwardLinkedProgram
from .flexible_dense_train import FlexibleDenseTrainActionKind
from .serde import canonical_digest
from .workload_materialization import (
    WorkloadMaterializationManifest,
    WorkloadMaterializationStatus,
)
from .workload_run import (
    WorkloadFamily,
    WorkloadMemoryMode,
    WorkloadOptimizerKind,
)


DENSE_TRAINING_COMPILE_SEQUENCE_SCHEMA_VERSION = (
    "wafer_frontend.dense_training_compile_sequence/v1alpha1"
)
DENSE_TRAINING_COMPILE_SEGMENT_SCHEMA_VERSION = (
    "wafer_frontend.dense_training_compile_segment/v1alpha1"
)
DENSE_TRAINING_PARAMETER_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.dense_training_parameter_binding/v1alpha1"
)
DENSE_TRAINING_LEGACY_SHARD_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.dense_training_legacy_shard_binding/v1alpha1"
)


class DenseTrainingCompileStatus(str, Enum):
    REUSED_REAL_ONE_STEP_LINKED_PROGRAM = "reused_real_one_step_linked_program"


class DenseTrainingRuntimeStatus(str, Enum):
    RUNTIME_NOT_MATERIALIZED = "runtime_not_materialized"


class DenseTrainingSyncLowering(str, Enum):
    SINGLETON_NOOP = "singleton_noop"
    DP_TREE = "dp_tree"


@dataclass(frozen=True, slots=True)
class DenseTrainingLegacyShardBinding:
    id: str
    tp_shard: int
    owner_ranks: tuple[int, ...]
    legacy_tensor_ref: str
    legacy_state_ref: str
    legacy_byte_offset: int
    logical_bytes: int
    load_action_refs: tuple[str, ...]
    wgrad_action_refs: tuple[str, ...]
    sync_action_refs: tuple[str, ...]
    sgd_action_refs: tuple[str, ...]
    store_action_refs: tuple[str, ...]
    state_abi_refs: tuple[str, ...]
    hbm_addresses: tuple[int, ...]
    sync_lowering: DenseTrainingSyncLowering

    @classmethod
    def create(cls, **semantic: object) -> "DenseTrainingLegacyShardBinding":
        result = cls(
            id=stable_artifact_id(
                "dense_training_legacy_shard_binding",
                semantic,
                schema_version=(
                    DENSE_TRAINING_LEGACY_SHARD_BINDING_SCHEMA_VERSION
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
            if name != "id"
        }

    def validate(self, path: str = "dense_training_legacy_shard_binding") -> None:
        for name in ("tp_shard", "legacy_byte_offset", "logical_bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.logical_bytes == 0:
            raise SchemaError("must be positive", path=f"{path}.logical_bytes")
        for name in ("legacy_tensor_ref", "legacy_state_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if not self.owner_ranks or tuple(sorted(self.owner_ranks)) != self.owner_ranks:
            raise SchemaError(
                "owner ranks must be sorted and non-empty",
                path=f"{path}.owner_ranks",
            )
        if len(set(self.owner_ranks)) != len(self.owner_ranks):
            raise SchemaError("owner ranks must be unique", path=f"{path}.owner_ranks")
        for rank in self.owner_ranks:
            validate_uint64(rank, f"{path}.owner_ranks")
        exact_owner_arity = (
            "load_action_refs",
            "wgrad_action_refs",
            "sgd_action_refs",
            "store_action_refs",
            "state_abi_refs",
            "hbm_addresses",
        )
        for name in exact_owner_arity:
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != len(self.owner_ranks):
                raise SchemaError("must contain one item per owner", path=f"{path}.{name}")
            if len(set(values)) != len(values):
                raise SchemaError("must contain unique items", path=f"{path}.{name}")
        for name in (
            "load_action_refs",
            "wgrad_action_refs",
            "sync_action_refs",
            "sgd_action_refs",
            "store_action_refs",
            "state_abi_refs",
        ):
            for index, value in enumerate(getattr(self, name)):
                validate_nonempty(value, f"{path}.{name}[{index}]")
        for index, address in enumerate(self.hbm_addresses):
            validate_uint64(address, f"{path}.hbm_addresses[{index}]")
        if type(self.sync_lowering) is not DenseTrainingSyncLowering:
            raise SchemaError("must be a sync lowering", path=f"{path}.sync_lowering")
        if self.sync_lowering is DenseTrainingSyncLowering.SINGLETON_NOOP:
            if len(self.owner_ranks) != 1 or self.sync_action_refs:
                raise SchemaError("singleton sync must not emit transport actions", path=path)
        elif len(self.owner_ranks) <= 1 or not self.sync_action_refs:
            raise SchemaError("DP tree requires multiple owners and sync actions", path=path)
        expected = stable_artifact_id(
            "dense_training_legacy_shard_binding",
            self._semantic(),
            schema_version=DENSE_TRAINING_LEGACY_SHARD_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable legacy shard binding id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class DenseTrainingParameterBinding:
    id: str
    step: int
    parameter_ref: str
    layer: int | None
    load_operation_ref: str
    wgrad_operation_ref: str
    sync_operation_ref: str
    sgd_operation_ref: str
    store_operation_ref: str
    input_parameter_version: int
    input_parameter_state_ref: str
    raw_gradient_state_ref: str
    synced_gradient_state_ref: str
    output_parameter_version: int
    output_parameter_state_ref: str
    legacy_shards: tuple[DenseTrainingLegacyShardBinding, ...]

    @classmethod
    def create(cls, **semantic: object) -> "DenseTrainingParameterBinding":
        result = cls(
            id=stable_artifact_id(
                "dense_training_parameter_binding",
                semantic,
                schema_version=DENSE_TRAINING_PARAMETER_BINDING_SCHEMA_VERSION,
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

    def validate(self, path: str = "dense_training_parameter_binding") -> None:
        validate_uint64(self.step, f"{path}.step")
        validate_nonempty(self.parameter_ref, f"{path}.parameter_ref")
        if self.layer is not None:
            validate_uint64(self.layer, f"{path}.layer")
        for name in (
            "load_operation_ref",
            "wgrad_operation_ref",
            "sync_operation_ref",
            "sgd_operation_ref",
            "store_operation_ref",
            "input_parameter_state_ref",
            "raw_gradient_state_ref",
            "synced_gradient_state_ref",
            "output_parameter_state_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if self.input_parameter_version != self.step:
            raise SchemaError("input version must equal step", path=path)
        if self.output_parameter_version != self.step + 1:
            raise SchemaError("output version must advance once", path=path)
        if not self.legacy_shards:
            raise SchemaError("must bind legacy TP shards", path=f"{path}.legacy_shards")
        for index, shard in enumerate(self.legacy_shards):
            if type(shard) is not DenseTrainingLegacyShardBinding:
                raise SchemaError(
                    "must be a legacy shard binding",
                    path=f"{path}.legacy_shards[{index}]",
                )
            shard.validate(f"{path}.legacy_shards[{index}]")
        if tuple(item.tp_shard for item in self.legacy_shards) != tuple(
            range(len(self.legacy_shards))
        ):
            raise SchemaError("must cover contiguous TP shards", path=f"{path}.legacy_shards")
        expected = stable_artifact_id(
            "dense_training_parameter_binding",
            self._semantic(),
            schema_version=DENSE_TRAINING_PARAMETER_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable parameter binding id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class DenseTrainingCompileSegment:
    id: str
    step: int
    parameter_bindings: tuple[DenseTrainingParameterBinding, ...]
    linked_program: FlexibleDenseBackwardLinkedProgram
    linked_manifest_digest: str
    one_shot_workload_end: bool

    @classmethod
    def create(cls, **semantic: object) -> "DenseTrainingCompileSegment":
        result = cls(
            id=stable_artifact_id(
                "dense_training_compile_segment",
                semantic,
                schema_version=DENSE_TRAINING_COMPILE_SEGMENT_SCHEMA_VERSION,
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

    def validate(self, path: str = "dense_training_compile_segment") -> None:
        validate_uint64(self.step, f"{path}.step")
        if not self.parameter_bindings:
            raise SchemaError("must bind trainable parameters", path=f"{path}.parameter_bindings")
        keys = []
        for index, binding in enumerate(self.parameter_bindings):
            if type(binding) is not DenseTrainingParameterBinding:
                raise SchemaError(
                    "must be a parameter binding",
                    path=f"{path}.parameter_bindings[{index}]",
                )
            binding.validate(f"{path}.parameter_bindings[{index}]")
            if binding.step != self.step:
                raise SchemaError(
                    "binding belongs to another step",
                    path=f"{path}.parameter_bindings[{index}]",
                )
            keys.append(binding.parameter_ref)
        if keys != sorted(set(keys)):
            raise SchemaError(
                "parameter bindings must be unique and sorted",
                path=f"{path}.parameter_bindings",
            )
        if type(self.linked_program) is not FlexibleDenseBackwardLinkedProgram:
            raise SchemaError("must embed a real linked program", path=f"{path}.linked_program")
        self.linked_program.validate(f"{path}.linked_program")
        if not self.linked_program.manifest.fragments:
            raise SchemaError("linked program must be non-empty", path=f"{path}.linked_program")
        if self.linked_manifest_digest != canonical_digest(self.linked_program.manifest):
            raise SchemaError(
                "linked manifest digest mismatch",
                path=f"{path}.linked_manifest_digest",
            )
        if type(self.one_shot_workload_end) is not bool:
            raise SchemaError("must be bool", path=f"{path}.one_shot_workload_end")
        expected = stable_artifact_id(
            "dense_training_compile_segment",
            self._semantic(),
            schema_version=DENSE_TRAINING_COMPILE_SEGMENT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable segment id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class DenseTrainingCompileSequence:
    schema_version: str
    producer_pass: str
    id: str
    materialization: WorkloadMaterializationManifest
    materialization_digest: str
    logical_graph_digest: str
    segments: tuple[DenseTrainingCompileSegment, ...]
    compile_status: DenseTrainingCompileStatus
    runtime_status: DenseTrainingRuntimeStatus

    @classmethod
    def create(
        cls,
        *,
        materialization: WorkloadMaterializationManifest,
        segments: tuple[DenseTrainingCompileSegment, ...],
    ) -> "DenseTrainingCompileSequence":
        semantic = {
            "materialization": materialization,
            "materialization_digest": materialization.digest,
            "logical_graph_digest": materialization.logical_graph_digest,
            "segments": segments,
            "compile_status": (
                DenseTrainingCompileStatus.REUSED_REAL_ONE_STEP_LINKED_PROGRAM
            ),
            "runtime_status": DenseTrainingRuntimeStatus.RUNTIME_NOT_MATERIALIZED,
        }
        result = cls(
            schema_version=DENSE_TRAINING_COMPILE_SEQUENCE_SCHEMA_VERSION,
            producer_pass="compile_dense_training_sequence",
            id=stable_artifact_id(
                "dense_training_compile_sequence",
                semantic,
                schema_version=DENSE_TRAINING_COMPILE_SEQUENCE_SCHEMA_VERSION,
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
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "dense_training_compile_sequence") -> None:
        if self.schema_version != DENSE_TRAINING_COMPILE_SEQUENCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "compile_dense_training_sequence":
            raise SchemaError("unexpected producer", path=f"{path}.producer_pass")
        self.materialization.validate(f"{path}.materialization")
        request = self.materialization.request
        if (
            self.materialization.status is not WorkloadMaterializationStatus.PARTIAL
            or request.family is not WorkloadFamily.DENSE_TRAINING
            or request.steps.training is None
            or request.steps.training.step_count != 2
            or request.optimizer is None
            or request.optimizer.kind is not WorkloadOptimizerKind.SGD
            or request.memory.mode is not WorkloadMemoryMode.RESIDENT_HBM
        ):
            raise SchemaError(
                "requires a two-step resident Dense SGD manifest",
                path=f"{path}.materialization",
            )
        if self.materialization_digest != self.materialization.digest:
            raise SchemaError(
                "materialization digest mismatch",
                path=f"{path}.materialization_digest",
            )
        if self.logical_graph_digest != self.materialization.logical_graph_digest:
            raise SchemaError("logical graph digest mismatch", path=f"{path}.logical_graph_digest")
        if tuple(segment.step for segment in self.segments) != (0, 1):
            raise SchemaError("requires exactly training steps 0 and 1", path=f"{path}.segments")
        for index, segment in enumerate(self.segments):
            if type(segment) is not DenseTrainingCompileSegment:
                raise SchemaError("must be a compile segment", path=f"{path}.segments[{index}]")
            segment.validate(f"{path}.segments[{index}]")
        if tuple(item.one_shot_workload_end for item in self.segments) != (False, True):
            raise SchemaError("only final training step may end workload", path=f"{path}.segments")
        if self.segments[0].linked_program != self.segments[1].linked_program:
            raise SchemaError(
                "two steps must reuse one exact compiled template",
                path=f"{path}.segments",
            )
        if (
            self.compile_status
            is not DenseTrainingCompileStatus.REUSED_REAL_ONE_STEP_LINKED_PROGRAM
            or self.runtime_status
            is not DenseTrainingRuntimeStatus.RUNTIME_NOT_MATERIALIZED
        ):
            raise SchemaError("compile/runtime status overclaims support", path=path)
        self._validate_graph_bindings(path)
        expected = stable_artifact_id(
            "dense_training_compile_sequence",
            self._semantic(),
            schema_version=DENSE_TRAINING_COMPILE_SEQUENCE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable sequence id", path=f"{path}.id")

    def _validate_graph_bindings(self, path: str) -> None:
        graph = self.materialization.logical_graph
        operations = {item.id: item for item in graph.operations}
        states = {item.id: item for item in graph.state_versions}
        parameters = tuple(sorted({
            item.logical_name
            for item in graph.state_versions
            if item.kind is E2EStateKind.PARAMETER
        }))
        previous = {}
        for segment in self.segments:
            if tuple(item.parameter_ref for item in segment.parameter_bindings) != parameters:
                raise SchemaError("must bind every P3 trainable parameter", path=f"{path}.segments")
            for binding in segment.parameter_bindings:
                refs = (
                    (binding.load_operation_ref, E2EOperationKind.PARAMETER_LOAD),
                    (binding.wgrad_operation_ref, E2EOperationKind.WGRAD),
                    (binding.sync_operation_ref, E2EOperationKind.GRADIENT_SYNC),
                    (binding.sgd_operation_ref, E2EOperationKind.SGD_UPDATE),
                    (binding.store_operation_ref, E2EOperationKind.PARAMETER_STORE),
                )
                selected = []
                for operation_ref, kind in refs:
                    operation = operations.get(operation_ref)
                    if operation is None or (
                        operation.kind,
                        operation.step,
                        operation.parameter_ref,
                    ) != (kind, binding.step, binding.parameter_ref):
                        raise SchemaError("P3 operation binding drifted", path=f"{path}.segments")
                    selected.append(operation)
                load, wgrad, sync, sgd, store = selected
                expected_edges = (
                    (load.reads, (binding.input_parameter_state_ref,)),
                    (wgrad.writes, (binding.raw_gradient_state_ref,)),
                    (sync.reads, (binding.raw_gradient_state_ref,)),
                    (sync.writes, (binding.synced_gradient_state_ref,)),
                    (sgd.reads, (
                        binding.input_parameter_state_ref,
                        binding.synced_gradient_state_ref,
                    )),
                    (sgd.writes, (binding.output_parameter_state_ref,)),
                    (store.reads, (binding.output_parameter_state_ref,)),
                )
                if any(actual != expected for actual, expected in expected_edges):
                    raise SchemaError("P3 parameter state lineage drifted", path=f"{path}.segments")
                input_state = states.get(binding.input_parameter_state_ref)
                output_state = states.get(binding.output_parameter_state_ref)
                if input_state is None or output_state is None or (
                    input_state.kind,
                    output_state.kind,
                    input_state.logical_name,
                    output_state.logical_name,
                    input_state.version,
                    output_state.version,
                ) != (
                    E2EStateKind.PARAMETER,
                    E2EStateKind.PARAMETER,
                    binding.parameter_ref,
                    binding.parameter_ref,
                    binding.input_parameter_version,
                    binding.output_parameter_version,
                ):
                    raise SchemaError(
                        "P3 parameter version binding drifted",
                        path=f"{path}.segments",
                    )
                if (
                    binding.step
                    and previous.get(binding.parameter_ref)
                    != binding.input_parameter_state_ref
                ):
                    raise SchemaError(
                        "parameter version is not continuous across steps",
                        path=f"{path}.segments",
                    )
                previous[binding.parameter_ref] = binding.output_parameter_state_ref
                self._validate_legacy_binding(binding, segment, path)

    def _validate_legacy_binding(
        self,
        binding: DenseTrainingParameterBinding,
        segment: DenseTrainingCompileSegment,
        path: str,
    ) -> None:
        program = segment.linked_program
        plan = program.plan
        templates = {item.state_ref: item for item in plan.parameter_templates}
        actions = {item.id: item for item in plan.rank_actions}
        state_abis = {
            item.id: item
            for fragment in program.manifest.fragments
            for item in fragment.state_abi
        }
        values = tuple(
            item
            for item in self.materialization.logical_graph.tensor_values
            if item.state_ref == binding.input_parameter_state_ref
        )
        for shard in binding.legacy_shards:
            template = templates.get(shard.legacy_state_ref)
            if template is None or template.tensor_ref != shard.legacy_tensor_ref:
                raise SchemaError("legacy parameter template drifted", path=f"{path}.segments")
            shard_values = tuple(sorted(
                (item for item in values if item.tp_shard == shard.tp_shard),
                key=lambda item: item.logical_rank,
            ))
            if (
                tuple(item.logical_rank for item in shard_values) != shard.owner_ranks
                or {item.size_bytes for item in shard_values} != {shard.logical_bytes}
                or shard.legacy_byte_offset + shard.logical_bytes > template.weight_bytes
            ):
                raise SchemaError("P3 value/legacy byte slice drifted", path=f"{path}.segments")
            fused_gate = binding.parameter_ref.endswith(".mlp_gate.weight")
            fused_up = binding.parameter_ref.endswith(".mlp_up.weight")
            if fused_gate or fused_up:
                if (
                    not shard.legacy_tensor_ref.endswith(".w_gate_up")
                    or template.weight_bytes != 2 * shard.logical_bytes
                    or shard.legacy_byte_offset
                    != (shard.logical_bytes if fused_up else 0)
                ):
                    raise SchemaError(
                        "P3 gate/up must exactly partition legacy fused storage",
                        path=f"{path}.segments",
                    )
            else:
                exact_shard = (
                    template.weight_bytes == shard.logical_bytes
                    and shard.legacy_byte_offset == 0
                )
                replicated_slice = (
                    template.weight_bytes
                    == shard.logical_bytes * plan.spec.tp_degree
                    and shard.legacy_byte_offset
                    == shard.tp_shard * shard.logical_bytes
                )
                if not exact_shard and not replicated_slice:
                    raise SchemaError(
                        "P3 parameter must bind an exact legacy TP slice",
                        path=f"{path}.segments",
                    )
            expected_actions = {}
            for kind, field in (
                (FlexibleDenseTrainActionKind.PARAMETER_LOAD, "load_action_refs"),
                (FlexibleDenseTrainActionKind.WEIGHT_GRADIENT, "wgrad_action_refs"),
                (FlexibleDenseTrainActionKind.GRADIENT_SYNC, "sync_action_refs"),
                (FlexibleDenseTrainActionKind.SGD_UPDATE, "sgd_action_refs"),
                (FlexibleDenseTrainActionKind.PARAMETER_STORE, "store_action_refs"),
            ):
                expected_actions[field] = tuple(sorted(
                    item.id
                    for item in actions.values()
                    if item.kind is kind and item.state_ref == shard.legacy_state_ref
                ))
            if any(getattr(shard, name) != refs for name, refs in expected_actions.items()):
                raise SchemaError("legacy action coverage drifted", path=f"{path}.segments")
            abis = tuple(sorted(
                (
                    item
                    for item in state_abis.values()
                    if item.state_ref == shard.legacy_state_ref
                ),
                key=lambda item: item.die_id,
            ))
            if (
                tuple(item.id for item in abis) != shard.state_abi_refs
                or tuple(item.address for item in abis) != shard.hbm_addresses
                or tuple(item.die_id for item in abis) != shard.owner_ranks
            ):
                raise SchemaError("legacy StateABI binding drifted", path=f"{path}.segments")


__all__ = [
    "DENSE_TRAINING_COMPILE_SEQUENCE_SCHEMA_VERSION",
    "DenseTrainingCompileSegment",
    "DenseTrainingCompileSequence",
    "DenseTrainingCompileStatus",
    "DenseTrainingLegacyShardBinding",
    "DenseTrainingParameterBinding",
    "DenseTrainingRuntimeStatus",
    "DenseTrainingSyncLowering",
]
