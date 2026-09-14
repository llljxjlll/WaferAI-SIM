"""Compile-only Dense Prefill/Decode sequence with explicit KV continuity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .artifact_manifest import LinkedProgramManifest
from .common import ProfileKey, stable_artifact_id, validate_nonempty, validate_uint64
from .e2e_workload_graph import E2EStateKind
from .experiment import ExperimentSpec, InferSource, InstanceRole, WorkloadMode
from .memory_plan import MemoryObjectKind, MemoryTier
from .persistent_state import StateKind
from .rect_mesh_compile import RectMeshCompileCapabilityReport, RectMeshCompileChain
from .serde import canonical_digest
from .workload_materialization import (
    WorkloadMaterializationManifest,
    WorkloadMaterializationStatus,
)
from .workload_run import WorkloadFamily


DENSE_COMPILE_SEQUENCE_SCHEMA_VERSION = (
    "wafer_frontend.dense_compile_sequence/v1alpha1"
)
DENSE_COMPILE_SEGMENT_SCHEMA_VERSION = (
    "wafer_frontend.dense_compile_segment/v1alpha1"
)
DENSE_KV_SEGMENT_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.dense_kv_segment_binding/v1alpha1"
)


class DenseCompileSequenceStatus(str, Enum):
    INDEPENDENT_LINKED_PROGRAMS = "independent_linked_programs"


class DenseCompileRuntimeStatus(str, Enum):
    RUNTIME_NOT_MATERIALIZED = "runtime_not_materialized"


def expected_segment_profile(
    manifest: WorkloadMaterializationManifest,
    segment_index: int,
) -> ProfileKey:
    request = manifest.request
    inference = request.steps.inference
    assert inference is not None
    if segment_index == 0:
        context = inference.prefill_tokens
        return ProfileKey(
            prefill_tokens=inference.request_count * inference.prefill_tokens,
            decode_tokens=0,
            num_seqs=inference.request_count,
            context_sum=inference.request_count * context,
            context_max=context,
            kv_pages=inference.request_count * ((context + 15) // 16),
            expert_load=None,
        )
    context = inference.prefill_tokens + segment_index
    return ProfileKey(
        prefill_tokens=0,
        decode_tokens=inference.request_count,
        num_seqs=inference.request_count,
        context_sum=inference.request_count * context,
        context_max=context,
        kv_pages=inference.request_count * ((context + 15) // 16),
        expert_load=None,
    )


def _kv_state_ref(
    manifest: WorkloadMaterializationManifest,
    layer: int,
    version: int,
) -> str:
    matches = tuple(
        state.id
        for state in manifest.logical_graph.state_versions
        if state.kind is E2EStateKind.KV
        and state.layer == layer
        and state.version == version
    )
    if len(matches) != 1:
        raise SchemaError(
            "requires exactly one KV state for layer/version",
            path="dense_compile_sequence.materialization.logical_graph.state_versions",
        )
    return matches[0]


def _kv_slot(
    manifest: WorkloadMaterializationManifest,
    layer: int,
    logical_rank: int,
) -> tuple[int, int]:
    inventory = tuple(
        item
        for item in manifest.state_inventory
        if item.object_kind is MemoryObjectKind.KV
        and item.logical_rank == logical_rank
    )
    if len(inventory) != 1:
        raise SchemaError(
            "requires one KV inventory allocation per rank",
            path="dense_compile_sequence.materialization.state_inventory",
        )
    memory_states = tuple(
        item
        for item in manifest.memory_plan.state_versions
        if item.state_ref == inventory[0].id and item.generation == 0
    )
    if len(memory_states) != 1:
        raise SchemaError(
            "requires one initial memory state for KV inventory",
            path="dense_compile_sequence.materialization.memory_plan.state_versions",
        )
    requests = tuple(
        item
        for item in manifest.memory_plan.requests
        if item.state_version_ref == memory_states[0].id
    )
    if len(requests) != 1 or requests[0].tier is not MemoryTier.HBM:
        raise SchemaError(
            "KV sequence requires a resident HBM allocation",
            path="dense_compile_sequence.materialization.memory_plan.requests",
        )
    allocations = tuple(
        item
        for item in manifest.memory_plan.allocations
        if item.request_ref == requests[0].id
    )
    layer_count = manifest.request.model.num_layers
    if (
        len(allocations) != 1
        or requests[0].size_bytes % layer_count
        or allocations[0].reserved_bytes < requests[0].size_bytes
    ):
        raise SchemaError(
            "KV allocation cannot be partitioned into stable layer slots",
            path="dense_compile_sequence.materialization.memory_plan.allocations",
        )
    stride = requests[0].size_bytes // layer_count
    return allocations[0].address + layer * stride, stride


@dataclass(frozen=True, slots=True)
class DenseKvSegmentBinding:
    id: str
    segment_index: int
    layer: int
    logical_rank: int
    address: int
    reserved_bytes: int
    input_version: int
    input_state_ref: str
    output_version: int
    output_state_ref: str

    @classmethod
    def create(cls, **semantic: object) -> "DenseKvSegmentBinding":
        result = cls(
            id=stable_artifact_id(
                "dense_kv_segment_binding",
                semantic,
                schema_version=DENSE_KV_SEGMENT_BINDING_SCHEMA_VERSION,
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

    def validate(self, path: str = "dense_kv_segment_binding") -> None:
        for name in (
            "segment_index",
            "layer",
            "logical_rank",
            "address",
            "reserved_bytes",
            "input_version",
            "output_version",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.reserved_bytes == 0:
            raise SchemaError("must be positive", path=f"{path}.reserved_bytes")
        if self.input_version != self.segment_index:
            raise SchemaError("input version must equal segment index", path=path)
        if self.output_version != self.input_version + 1:
            raise SchemaError("output version must advance exactly once", path=path)
        for name in ("input_state_ref", "output_state_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        expected = stable_artifact_id(
            "dense_kv_segment_binding",
            self._semantic(),
            schema_version=DENSE_KV_SEGMENT_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable KV binding id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class DenseCompileSegment:
    id: str
    segment_index: int
    phase: str
    step: int
    legacy_spec: ExperimentSpec
    operation_refs: tuple[str, ...]
    layer_bindings: tuple[int, ...]
    kv_bindings: tuple[DenseKvSegmentBinding, ...]
    compilation_id: str
    capability_report: RectMeshCompileCapabilityReport
    linked_profile_id: str
    linked_manifest: LinkedProgramManifest
    linked_manifest_digest: str
    one_shot_workload_end: bool

    @classmethod
    def create(cls, **semantic: object) -> "DenseCompileSegment":
        result = cls(
            id=stable_artifact_id(
                "dense_compile_segment",
                semantic,
                schema_version=DENSE_COMPILE_SEGMENT_SCHEMA_VERSION,
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

    def validate(self, path: str = "dense_compile_segment") -> None:
        validate_uint64(self.segment_index, f"{path}.segment_index")
        validate_uint64(self.step, f"{path}.step")
        if self.step != self.segment_index:
            raise SchemaError("step must equal segment index", path=f"{path}.step")
        expected_phase = "prefill" if self.segment_index == 0 else "decode"
        if self.phase != expected_phase:
            raise SchemaError("phase disagrees with segment index", path=f"{path}.phase")
        self.legacy_spec.validate(f"{path}.legacy_spec")
        infer = self.legacy_spec.workload.infer
        if (
            self.legacy_spec.workload.mode is not WorkloadMode.INFER
            or infer is None
            or infer.source is not InferSource.STATIC_PROFILE
            or infer.profile is None
        ):
            raise SchemaError("requires one static inference profile", path=f"{path}.legacy_spec")
        if not self.operation_refs or len(set(self.operation_refs)) != len(self.operation_refs):
            raise SchemaError("must contain unique graph operations", path=f"{path}.operation_refs")
        if not self.layer_bindings or self.layer_bindings != tuple(range(len(self.layer_bindings))):
            raise SchemaError("must bind contiguous model layers", path=f"{path}.layer_bindings")
        if not self.kv_bindings:
            raise SchemaError("must bind KV continuity", path=f"{path}.kv_bindings")
        for index, binding in enumerate(self.kv_bindings):
            binding.validate(f"{path}.kv_bindings[{index}]")
            if binding.segment_index != self.segment_index:
                raise SchemaError("KV binding belongs to another segment", path=f"{path}.kv_bindings[{index}]")
        validate_nonempty(self.compilation_id, f"{path}.compilation_id")
        self.capability_report.validate(f"{path}.capability_report")
        if (
            self.capability_report.selected_chain is not RectMeshCompileChain.NAIVE_FIXED_V1
            or not self.capability_report.executable_baseline
            or self.capability_report.standard_chain_complete
            or self.capability_report.dense_workload_complete
        ):
            raise SchemaError("segment must remain legacy motif scope", path=f"{path}.capability_report")
        validate_nonempty(self.linked_profile_id, f"{path}.linked_profile_id")
        self.linked_manifest.validate(f"{path}.linked_manifest")
        if not self.linked_manifest.fragments or not self.linked_manifest.core_streams:
            raise SchemaError("linked program must be non-empty", path=f"{path}.linked_manifest")
        if self.linked_manifest_digest != canonical_digest(self.linked_manifest):
            raise SchemaError("linked manifest digest mismatch", path=f"{path}.linked_manifest_digest")
        if type(self.one_shot_workload_end) is not bool:
            raise SchemaError("must be bool", path=f"{path}.one_shot_workload_end")
        expected = stable_artifact_id(
            "dense_compile_segment",
            self._semantic(),
            schema_version=DENSE_COMPILE_SEGMENT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable segment id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class DenseCompileSequence:
    schema_version: str
    producer_pass: str
    id: str
    materialization: WorkloadMaterializationManifest
    materialization_digest: str
    logical_graph_digest: str
    segments: tuple[DenseCompileSegment, ...]
    compile_status: DenseCompileSequenceStatus
    runtime_status: DenseCompileRuntimeStatus

    @classmethod
    def create(
        cls,
        *,
        materialization: WorkloadMaterializationManifest,
        segments: tuple[DenseCompileSegment, ...],
    ) -> "DenseCompileSequence":
        semantic = {
            "materialization": materialization,
            "materialization_digest": materialization.digest,
            "logical_graph_digest": materialization.logical_graph_digest,
            "segments": segments,
            "compile_status": DenseCompileSequenceStatus.INDEPENDENT_LINKED_PROGRAMS,
            "runtime_status": DenseCompileRuntimeStatus.RUNTIME_NOT_MATERIALIZED,
        }
        result = cls(
            schema_version=DENSE_COMPILE_SEQUENCE_SCHEMA_VERSION,
            producer_pass="compile_dense_e2e_sequence",
            id=stable_artifact_id(
                "dense_compile_sequence",
                semantic,
                schema_version=DENSE_COMPILE_SEQUENCE_SCHEMA_VERSION,
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

    def validate(self, path: str = "dense_compile_sequence") -> None:
        if self.schema_version != DENSE_COMPILE_SEQUENCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "compile_dense_e2e_sequence":
            raise SchemaError("unexpected producer", path=f"{path}.producer_pass")
        self.materialization.validate(f"{path}.materialization")
        request = self.materialization.request
        if (
            self.materialization.status is not WorkloadMaterializationStatus.PARTIAL
            or request.family is not WorkloadFamily.DENSE_INFERENCE
            or request.steps.inference is None
            or request.steps.inference.decode_steps != 2
            or request.model.num_layers < 2
        ):
            raise SchemaError("requires a P3 Dense Prefill+2 Decode materialization", path=f"{path}.materialization")
        if self.materialization_digest != self.materialization.digest:
            raise SchemaError("materialization digest mismatch", path=f"{path}.materialization_digest")
        if self.logical_graph_digest != self.materialization.logical_graph_digest:
            raise SchemaError("logical graph digest mismatch", path=f"{path}.logical_graph_digest")
        if self.compile_status is not DenseCompileSequenceStatus.INDEPENDENT_LINKED_PROGRAMS:
            raise SchemaError("invalid compile status", path=f"{path}.compile_status")
        if self.runtime_status is not DenseCompileRuntimeStatus.RUNTIME_NOT_MATERIALIZED:
            raise SchemaError("runtime cannot be claimed by compile-only sequence", path=f"{path}.runtime_status")
        if len(self.segments) != 3:
            raise SchemaError("requires Prefill, Decode, Decode segments", path=f"{path}.segments")
        graph = self.materialization.logical_graph
        ranks = tuple(range(request.parallel.logical_rank_count))
        previous: dict[tuple[int, int], DenseKvSegmentBinding] = {}
        previous_physical_layout: tuple[tuple[StateKind, int, int], ...] | None = None
        for index, segment in enumerate(self.segments):
            segment_path = f"{path}.segments[{index}]"
            segment.validate(segment_path)
            if segment.segment_index != index:
                raise SchemaError("segment indices must be contiguous", path=segment_path)
            infer = segment.legacy_spec.workload.infer
            assert infer is not None and infer.profile is not None
            if infer.profile != expected_segment_profile(self.materialization, index):
                raise SchemaError("legacy profile does not match graph segment", path=f"{segment_path}.legacy_spec")
            expected_role = InstanceRole.PREFILL if index == 0 else InstanceRole.DECODE
            instances = segment.legacy_spec.parallel.instances
            if len(instances) != 1 or instances[0].role is not expected_role:
                raise SchemaError("legacy instance role does not match segment", path=f"{segment_path}.legacy_spec.parallel")
            legacy_model = segment.legacy_spec.model
            target_model = request.model
            if (
                legacy_model.V,
                legacy_model.H,
                legacy_model.I,
                legacy_model.L,
                legacy_model.NH,
                legacy_model.KVH,
                legacy_model.DH,
                legacy_model.max_position_embeddings,
                legacy_model.dtype,
                instances[0].tp,
                instances[0].dp,
                instances[0].ep,
                instances[0].pp,
            ) != (
                target_model.vocabulary_size,
                target_model.hidden_size,
                target_model.intermediate_size,
                target_model.num_layers,
                target_model.num_attention_heads,
                target_model.num_kv_heads,
                target_model.head_dim,
                target_model.max_sequence_length,
                target_model.dtype,
                request.parallel.tp,
                1,
                1,
                1,
            ):
                raise SchemaError("legacy model/parallel binding drifted", path=f"{segment_path}.legacy_spec")
            expected_ops = tuple(op.id for op in graph.operations if op.step == index)
            if segment.operation_refs != expected_ops:
                raise SchemaError("segment operations do not exactly cover graph step", path=f"{segment_path}.operation_refs")
            if segment.layer_bindings != tuple(range(request.model.num_layers)):
                raise SchemaError("segment does not cover every model layer", path=f"{segment_path}.layer_bindings")
            state_abis = {
                abi.id: abi
                for fragment in segment.linked_manifest.fragments
                for abi in fragment.state_abi
            }
            physical_kv = tuple(
                sorted(
                    (
                        abi
                        for abi in state_abis.values()
                        if abi.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
                    ),
                    key=lambda abi: (abi.kind.value, abi.die_id, abi.address),
                )
            )
            expected_state_count = 2 * request.model.num_layers * len(ranks)
            if len(physical_kv) != expected_state_count:
                raise SchemaError(
                    "linked program must expose every physical KV StateABI",
                    path=f"{segment_path}.linked_manifest.fragments",
                )
            dtype_bytes = 2 if request.model.dtype.value == "fp16" else 4
            expected_state_bytes = (
                infer.profile.context_sum
                * (request.model.num_kv_heads // request.parallel.tp)
                * request.model.head_dim
                * dtype_bytes
            )
            if any(abi.size_bytes != expected_state_bytes for abi in physical_kv):
                raise SchemaError(
                    "physical KV StateABI must retain the segment's true extent",
                    path=f"{segment_path}.linked_manifest.fragments",
                )
            physical_layout = tuple(
                (abi.kind, abi.die_id, abi.address) for abi in physical_kv
            )
            if previous_physical_layout is not None and physical_layout != previous_physical_layout:
                raise SchemaError(
                    "physical KV StateABI address changed between segments",
                    path=f"{segment_path}.linked_manifest.fragments",
                )
            previous_physical_layout = physical_layout
            expected_keys = tuple(
                (layer, rank)
                for layer in range(request.model.num_layers)
                for rank in ranks
            )
            actual_keys = tuple((item.layer, item.logical_rank) for item in segment.kv_bindings)
            if actual_keys != expected_keys:
                raise SchemaError("KV bindings must cover every layer/rank", path=f"{segment_path}.kv_bindings")
            for binding in segment.kv_bindings:
                expected_address, expected_bytes = _kv_slot(
                    self.materialization, binding.layer, binding.logical_rank
                )
                if (
                    binding.input_state_ref != _kv_state_ref(self.materialization, binding.layer, index)
                    or binding.output_state_ref != _kv_state_ref(self.materialization, binding.layer, index + 1)
                    or binding.address != expected_address
                    or binding.reserved_bytes != expected_bytes
                ):
                    raise SchemaError("KV state/version/address binding drifted", path=f"{segment_path}.kv_bindings")
                key = (binding.layer, binding.logical_rank)
                if index and (
                    previous[key].output_state_ref != binding.input_state_ref
                    or previous[key].output_version != binding.input_version
                    or previous[key].address != binding.address
                    or previous[key].reserved_bytes != binding.reserved_bytes
                ):
                    raise SchemaError("KV continuity was reset between segments", path=f"{segment_path}.kv_bindings")
                previous[key] = binding
            if segment.one_shot_workload_end != (index == 2):
                raise SchemaError("only the final segment may end the one-shot workload", path=f"{segment_path}.one_shot_workload_end")
        expected = stable_artifact_id(
            "dense_compile_sequence",
            self._semantic(),
            schema_version=DENSE_COMPILE_SEQUENCE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable sequence id", path=f"{path}.id")


def build_kv_segment_bindings(
    manifest: WorkloadMaterializationManifest,
    segment_index: int,
) -> tuple[DenseKvSegmentBinding, ...]:
    request = manifest.request
    return tuple(
        DenseKvSegmentBinding.create(
            segment_index=segment_index,
            layer=layer,
            logical_rank=rank,
            address=_kv_slot(manifest, layer, rank)[0],
            reserved_bytes=_kv_slot(manifest, layer, rank)[1],
            input_version=segment_index,
            input_state_ref=_kv_state_ref(manifest, layer, segment_index),
            output_version=segment_index + 1,
            output_state_ref=_kv_state_ref(manifest, layer, segment_index + 1),
        )
        for layer in range(request.model.num_layers)
        for rank in range(request.parallel.logical_rank_count)
    )


__all__ = [
    "DENSE_COMPILE_SEGMENT_SCHEMA_VERSION",
    "DENSE_COMPILE_SEQUENCE_SCHEMA_VERSION",
    "DENSE_KV_SEGMENT_BINDING_SCHEMA_VERSION",
    "DenseCompileRuntimeStatus",
    "DenseCompileSegment",
    "DenseCompileSequence",
    "DenseCompileSequenceStatus",
    "DenseKvSegmentBinding",
    "build_kv_segment_bindings",
    "expected_segment_profile",
]
