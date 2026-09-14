"""Compile a P3 Dense Prefill/Decode/Decode graph as three legacy programs."""

from __future__ import annotations

from dataclasses import replace

from ..compiler import compile_rect_mesh
from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import DType
from ..schema.dense_compile_sequence import (
    DenseCompileSegment,
    DenseCompileSequence,
    build_kv_segment_bindings,
    expected_segment_profile,
)
from ..schema.experiment import (
    ExperimentSpec,
    InferSource,
    InstanceRole,
    WorkloadMode,
)
from ..schema.ir1 import PhysicalFabric
from ..schema.memory_plan import MemoryPlanExecution
from ..schema.n6 import LinkedProgramProfile
from ..schema.placement import (
    PersistentStateReservationPolicy,
    PersistentStateSlotReservation,
)
from ..schema.persistent_state import HbmAddressSpace
from ..schema.persistent_state import StateKind
from ..schema.rect_mesh import RectMeshSpec
from ..schema.rect_mesh_compile import RectMeshCompileChain, RectMeshCompileMode
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import (
    WorkloadMaterializationManifest,
    WorkloadMaterializationStatus,
)
from ..schema.workload_run import (
    WorkloadExecutionStrategy,
    WorkloadFamily,
    WorkloadMemoryMode,
    WorkloadModelArchitecture,
)


def _reject(reasons: list[str]) -> None:
    if reasons:
        raise UnsupportedFeatureError(
            "request is outside Dense compile-sequence subset: "
            + ", ".join(sorted(set(reasons))),
            path="dense_compile_sequence",
            code="dense_compile_sequence_unsupported",
        )


def _validate_inputs(
    manifest: WorkloadMaterializationManifest,
    legacy_template: ExperimentSpec,
    fabric: PhysicalFabric,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> None:
    manifest.validate()
    legacy_template.validate()
    fabric.validate("fabric")
    request = manifest.request
    reasons: list[str] = []
    if manifest.status is not WorkloadMaterializationStatus.PARTIAL:
        reasons.append("manifest.partial_required")
    if request.family is not WorkloadFamily.DENSE_INFERENCE:
        reasons.append("request.dense_inference_required")
    if request.model.architecture is not WorkloadModelArchitecture.LLAMA_DENSE:
        reasons.append("request.dense_model_required")
    if request.steps.inference is None:
        reasons.append("request.inference_steps_required")
    else:
        if request.steps.inference.prefill_tokens == 0:
            reasons.append("request.prefill_required")
        if request.steps.inference.decode_steps != 2:
            reasons.append("request.exactly_two_decode_steps_required")
        if request.steps.inference.request_count % request.parallel.tp:
            reasons.append("request.decode_tokens_must_be_divisible_by_tp")
    if request.model.num_layers < 2:
        reasons.append("request.at_least_two_layers_required")
    if request.memory.mode is not WorkloadMemoryMode.RESIDENT_HBM:
        reasons.append("request.resident_hbm_required")
    if manifest.memory_plan.execution is not MemoryPlanExecution.NO_EXTERNAL_TRANSPORT_REQUIRED:
        reasons.append("manifest.external_transport_not_supported")
    if (
        not request.execution.timing
        or request.execution.functional
        or request.execution.strategy is not WorkloadExecutionStrategy.BASELINE
    ):
        reasons.append("request.baseline_timing_only_required")
    if (request.parallel.dp, request.parallel.ep, request.parallel.pp) != (1, 1, 1):
        reasons.append("request.dp_ep_pp_must_equal_one")
    if request.parallel.tp != request.mesh.rank_count:
        reasons.append("request.tp_must_cover_mesh")
    expected_dies = tuple(range(request.mesh.rank_count))
    if request.parallel.active_die_ids and request.parallel.active_die_ids != expected_dies:
        reasons.append("request.full_row_major_mesh_required")
    if manifest.placement.active_die_ids != expected_dies:
        reasons.append("manifest.full_row_major_mesh_required")
    if fabric.die_grid != (request.mesh.columns, request.mesh.rows):
        reasons.append("fabric.mesh_shape_mismatch")
    if len(fabric.dies) != request.mesh.rank_count:
        reasons.append("fabric.complete_rectangle_required")
    if legacy_template.workload.mode is not WorkloadMode.INFER:
        reasons.append("legacy.inference_template_required")
    else:
        infer = legacy_template.workload.infer
        assert infer is not None
        if infer.source is not InferSource.STATIC_PROFILE or infer.profile is None:
            reasons.append("legacy.static_profile_template_required")
    instances = legacy_template.parallel.instances
    if len(instances) != 1:
        reasons.append("legacy.single_instance_required")
    elif (
        instances[0].tp,
        instances[0].dp,
        instances[0].ep,
        instances[0].pp,
    ) != (request.parallel.tp, 1, 1, 1):
        reasons.append("legacy.parallel_mismatch")
    model = legacy_template.model
    target = request.model
    if (
        model.V,
        model.H,
        model.I,
        model.L,
        model.NH,
        model.KVH,
        model.DH,
        model.max_position_embeddings,
        model.dtype,
    ) != (
        target.vocabulary_size,
        target.hidden_size,
        target.intermediate_size,
        target.num_layers,
        target.num_attention_heads,
        target.num_kv_heads,
        target.head_dim,
        target.max_sequence_length,
        target.dtype,
    ):
        reasons.append("legacy.model_mismatch")
    if type(hbm_address_spaces) is not tuple or len(hbm_address_spaces) != request.mesh.rank_count:
        reasons.append("hbm.one_address_space_per_die_required")
    else:
        for index, address_space in enumerate(hbm_address_spaces):
            if type(address_space) is not HbmAddressSpace:
                raise SchemaError(
                    "must be an HbmAddressSpace",
                    path=f"hbm_address_spaces[{index}]",
                )
            address_space.validate(f"hbm_address_spaces[{index}]")
        if tuple(item.die_id for item in hbm_address_spaces) != expected_dies:
            reasons.append("hbm.row_major_die_order_required")
    _reject(reasons)


def _segment_spec(
    template: ExperimentSpec,
    manifest: WorkloadMaterializationManifest,
    segment_index: int,
) -> ExperimentSpec:
    infer = template.workload.infer
    assert infer is not None
    profile = expected_segment_profile(manifest, segment_index)
    segment_infer = replace(
        infer,
        source=InferSource.STATIC_PROFILE,
        profile=profile,
        shape_dist=None,
        pd_static=None,
    )
    instance = replace(
        template.parallel.instances[0],
        role=(InstanceRole.PREFILL if segment_index == 0 else InstanceRole.DECODE),
        sp=template.parallel.instances[0].tp > 1,
    )
    spec = replace(
        template,
        workload=replace(template.workload, infer=segment_infer, train=None),
        parallel=replace(template.parallel, instances=(instance,)),
    )
    spec.validate(f"dense_compile_sequence.segment[{segment_index}].legacy_spec")
    return spec


def _kv_reservation_policy(
    manifest: WorkloadMaterializationManifest,
) -> PersistentStateReservationPolicy:
    """Reserve every compiler KV tensor at the final Decode capacity."""

    request = manifest.request
    final_profile = expected_segment_profile(manifest, 2)
    element_bytes = {
        DType.FP16: 2,
        DType.FP32: 4,
    }.get(request.model.dtype)
    if element_bytes is None:
        raise UnsupportedFeatureError(
            "Dense KV fixed slots require fp16 or fp32",
            path="dense_compile_sequence.request.model.dtype",
            code="dense_compile_sequence_kv_dtype_unsupported",
        )
    raw_bytes = (
        final_profile.context_sum
        * (request.model.num_kv_heads // request.parallel.tp)
        * request.model.head_dim
        * element_bytes
    )
    slot_bytes = ((raw_bytes + 63) // 64) * 64
    slots = tuple(
        PersistentStateSlotReservation.create(
            producer_pass="compile_dense_e2e_sequence",
            kind=kind,
            slot_bytes=slot_bytes,
        )
        for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
    )
    return PersistentStateReservationPolicy.create(
        producer_pass="compile_dense_e2e_sequence",
        slots=slots,
    )


def _compile_dense_e2e_sequence_with_profiles(
    manifest: WorkloadMaterializationManifest,
    legacy_template: ExperimentSpec,
    fabric: PhysicalFabric,
    *,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> tuple[DenseCompileSequence, tuple[LinkedProgramProfile, ...]]:
    """Compile the sequence while retaining sources needed for ProgramIO."""

    if type(manifest) is not WorkloadMaterializationManifest:
        raise SchemaError("must be a WorkloadMaterializationManifest", path="manifest")
    if type(legacy_template) is not ExperimentSpec:
        raise SchemaError("must be an ExperimentSpec", path="legacy_template")
    if type(fabric) is not PhysicalFabric:
        raise SchemaError("must be a PhysicalFabric", path="fabric")
    _validate_inputs(manifest, legacy_template, fabric, hbm_address_spaces)
    mesh = RectMeshSpec(manifest.request.mesh.rows, manifest.request.mesh.columns)
    reservation_policy = _kv_reservation_policy(manifest)
    segments: list[DenseCompileSegment] = []
    linked_profiles: list[LinkedProgramProfile] = []
    for segment_index in range(3):
        spec = _segment_spec(legacy_template, manifest, segment_index)
        compilation = compile_rect_mesh(
            spec,
            fabric,
            rect_mesh=mesh,
            hbm_address_spaces=hbm_address_spaces,
            persistent_state_reservation_policy=reservation_policy,
            mode=RectMeshCompileMode.AUTO,
            producer_pass=f"dense_e2e_segment_{segment_index}",
        )
        compilation.validate()
        report = compilation.capability_report
        if (
            report.selected_chain is not RectMeshCompileChain.NAIVE_FIXED_V1
            or not report.executable_baseline
            or report.standard_chain_complete
            or report.dense_workload_complete
        ):
            raise SchemaError(
                "segment compiler escaped legacy independent-program scope",
                path=f"dense_compile_sequence.segments[{segment_index}].capability",
            )
        entries = compilation.compilation.linked.entries
        profile = expected_segment_profile(manifest, segment_index)
        if len(entries) != 1 or entries[0].profile_id != profile.stable_id():
            raise SchemaError(
                "compiler did not preserve exact segment profile",
                path=f"dense_compile_sequence.segments[{segment_index}].profile",
            )
        linked = entries[0]
        linked_profiles.append(linked)
        segment = DenseCompileSegment.create(
            segment_index=segment_index,
            phase="prefill" if segment_index == 0 else "decode",
            step=segment_index,
            legacy_spec=spec,
            operation_refs=tuple(
                operation.id
                for operation in manifest.logical_graph.operations
                if operation.step == segment_index
            ),
            layer_bindings=tuple(range(manifest.request.model.num_layers)),
            kv_bindings=build_kv_segment_bindings(manifest, segment_index),
            compilation_id=compilation.id,
            capability_report=report,
            linked_profile_id=linked.id,
            linked_manifest=linked.manifest,
            linked_manifest_digest=canonical_digest(linked.manifest),
            one_shot_workload_end=segment_index == 2,
        )
        segments.append(segment)
    sequence = DenseCompileSequence.create(
        materialization=manifest,
        segments=tuple(segments),
    )
    profiles = tuple(linked_profiles)
    if any(
        profile.id != segment.linked_profile_id
        or profile.manifest != segment.linked_manifest
        for profile, segment in zip(profiles, sequence.segments)
    ):
        raise SchemaError(
            "retained linked profile drifted from the sequence artifact",
            path="dense_compile_sequence.linked_profiles",
        )
    return sequence, profiles


def compile_dense_e2e_sequence(
    manifest: WorkloadMaterializationManifest,
    legacy_template: ExperimentSpec,
    fabric: PhysicalFabric,
    *,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> DenseCompileSequence:
    """Compile three independent linked programs; do not claim runtime splice."""

    sequence, _profiles = _compile_dense_e2e_sequence_with_profiles(
        manifest,
        legacy_template,
        fabric,
        hbm_address_spaces=hbm_address_spaces,
    )
    return sequence


def compile_dense_e2e_sequence_runtime_profiles(
    manifest: WorkloadMaterializationManifest,
    legacy_template: ExperimentSpec,
    fabric: PhysicalFabric,
    *,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> tuple[DenseCompileSequence, tuple[LinkedProgramProfile, ...]]:
    """Return exact sequence and linked sources for typed ProgramIO creation."""

    return _compile_dense_e2e_sequence_with_profiles(
        manifest,
        legacy_template,
        fabric,
        hbm_address_spaces=hbm_address_spaces,
    )


__all__ = [
    "compile_dense_e2e_sequence",
    "compile_dense_e2e_sequence_runtime_profiles",
]
