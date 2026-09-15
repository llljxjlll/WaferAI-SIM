"""Compile a full MoE inference graph from linked production components."""

from __future__ import annotations

from dataclasses import replace

from ..compiler import compile_rect_mesh
from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import DType
from ..schema.dense_compile_sequence import expected_segment_profile
from ..schema.e2e_workload_graph import E2EOperationKind
from ..schema.experiment import ExperimentSpec, InferSource, InstanceRole, WorkloadMode
from ..schema.flexible_moe import MoeRectActionKind, MoeRectFlowStage
from ..schema.ir0 import StateAccessMode
from ..schema.ir1 import PhysicalFabric
from ..schema.moe_full_model_compile_sequence import (
    MoeFullModelCompileSegment,
    MoeFullModelCompileSequence,
    MoeFullModelLowering,
    MoeFullModelOperationBinding,
)
from ..schema.n6 import LinkedProgramProfile
from ..schema.persistent_state import HbmAddressSpace, StateKind
from ..schema.rect_mesh import RectMeshSpec
from ..schema.rect_mesh_compile import RectMeshCompileMode
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import WorkloadMaterializationManifest
from ..schema.workload_run import WorkloadFamily, WorkloadModelArchitecture
from ..lowering.moe_full_model_linker import link_moe_full_model_segment
from .dense_compile_sequence import _kv_reservation_policy
from .moe_compile_sequence import compile_moe_sequence


_MOE_KINDS = {
    E2EOperationKind.ROUTER,
    E2EOperationKind.ROUTE_FREEZE,
    E2EOperationKind.DISPATCH,
    E2EOperationKind.EXPERT_FORWARD,
    E2EOperationKind.COMBINE,
}


def _reject(reasons: list[str]) -> None:
    if reasons:
        raise UnsupportedFeatureError(
            "request is outside MoE full-model inference compile subset: "
            + ", ".join(sorted(set(reasons))),
            path="moe_full_model_compile_sequence",
            code="moe_full_model_compile_sequence_unsupported",
        )


def _validate_inputs(
    manifest: WorkloadMaterializationManifest,
    legacy_template: ExperimentSpec,
    fabric: PhysicalFabric,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> None:
    manifest.validate("manifest")
    legacy_template.validate("legacy_template")
    fabric.validate("fabric")
    request = manifest.request
    reasons: list[str] = []
    if request.family is not WorkloadFamily.MOE_INFERENCE:
        reasons.append("request.moe_inference_required")
    if request.model.architecture is not WorkloadModelArchitecture.LLAMA_MOE:
        reasons.append("request.llama_moe_required")
    if request.model.dtype is not DType.FP16:
        reasons.append("request.fp16_required")
    if (request.mesh.rows, request.mesh.columns) != (1, 2):
        reasons.append("request.mesh_1x2_required")
    if (
        request.parallel.tp,
        request.parallel.dp,
        request.parallel.ep,
        request.parallel.pp,
    ) != (1, 1, 2, 1):
        reasons.append("request.tp1_dp1_ep2_pp1_required")
    if request.parallel.active_die_ids and request.parallel.active_die_ids != (0, 1):
        reasons.append("request.full_mesh_required")
    if manifest.placement.active_die_ids != (0, 1):
        reasons.append("manifest.full_mesh_required")
    if fabric.die_grid != (2, 1) or tuple(die.id for die in fabric.dies) != (0, 1):
        reasons.append("fabric.1x2_row_major_required")
    if len(hbm_address_spaces) != 2 or tuple(
        item.die_id for item in hbm_address_spaces
    ) != (0, 1):
        reasons.append("hbm.one_space_per_die_required")
    if legacy_template.workload.mode is not WorkloadMode.INFER:
        reasons.append("legacy.inference_template_required")
    if len(legacy_template.parallel.instances) != 1:
        reasons.append("legacy.single_instance_required")
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
    _reject(reasons)


def _rank_local_fabric(fabric: PhysicalFabric) -> PhysicalFabric:
    result = replace(
        fabric,
        die_grid=(1, 1),
        dies=(fabric.dies[0],),
        links=(),
    )
    result.validate("rank_local_fabric")
    return result


def _segment_spec(template, manifest, step: int):
    infer = template.workload.infer
    assert infer is not None
    instance = replace(
        template.parallel.instances[0],
        role=InstanceRole.PREFILL if step == 0 else InstanceRole.DECODE,
        tp=1,
        dp=1,
        ep=1,
        pp=1,
        replicas=1,
        sp=False,
    )
    result = replace(
        template,
        workload=replace(
            template.workload,
            infer=replace(
                infer,
                source=InferSource.STATIC_PROFILE,
                profile=expected_segment_profile(manifest, step),
                shape_dist=None,
                pd_static=None,
            ),
            train=None,
        ),
        parallel=replace(template.parallel, instances=(instance,)),
    )
    result.validate(f"shared_spine.segment[{step}].spec")
    return result


def _compile_shared_spine(
    manifest,
    template: ExperimentSpec,
    fabric: PhysicalFabric,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
    step: int,
) -> LinkedProgramProfile:
    spec = _segment_spec(template, manifest, step)
    compilation = compile_rect_mesh(
        spec,
        _rank_local_fabric(fabric),
        rect_mesh=RectMeshSpec(1, 1),
        hbm_address_spaces=(hbm_address_spaces[0],),
        persistent_state_reservation_policy=_kv_reservation_policy(manifest),
        mode=RectMeshCompileMode.AUTO,
        producer_pass=f"moe_full_model_shared_spine_{step}",
    )
    compilation.validate(f"shared_spine.segment[{step}].compilation")
    entries = compilation.compilation.linked.entries
    profile = expected_segment_profile(manifest, step)
    if len(entries) != 1 or entries[0].profile_id != profile.stable_id():
        raise SchemaError(
            "shared-spine compiler did not preserve the exact profile",
            path=f"shared_spine.segment[{step}]",
        )
    return entries[0]


def _node_ref(profile: LinkedProgramProfile, suffix: str) -> str:
    origin = profile.lowering_context.ir1.instances[0].origin_instance_id
    ref = f"{origin}.{suffix}"
    if ref not in {item.origin_node_id for item in profile.lowering_context.ir1.nodes}:
        raise SchemaError("shared-spine node is absent", path=f"shared_spine.{suffix}")
    return ref


def _attention_access_refs(profile: LinkedProgramProfile, layer: int) -> tuple[str, ...]:
    origin_ref = _node_ref(profile, f"layer{layer}.attention")
    physical_refs = {
        item.id
        for item in profile.lowering_context.ir1.nodes
        if item.origin_node_id == origin_ref
    }
    state_index = {
        item.id: item
        for item in profile.lowering_context.ir1.persistent_state_manifest.declarations
    }
    refs = tuple(
        item.id
        for item in profile.lowering_context.ir1.state_accesses
        if item.node_ref in physical_refs
        and state_index[item.state_ref].identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
    )
    if len(refs) != 2 or any(
        next(
            item for item in profile.lowering_context.ir1.state_accesses
            if item.id == ref
        ).mode not in (StateAccessMode.WRITE, StateAccessMode.READ_WRITE)
        for ref in refs
    ):
        raise SchemaError("attention must expose exact K/V state accesses", path="shared_spine.state_accesses")
    return refs


def _moe_refs(unit, kind: E2EOperationKind, expert: int | None) -> tuple[str, ...]:
    if kind is E2EOperationKind.ROUTE_FREEZE:
        return (unit.spec.trace.id,)
    if kind is E2EOperationKind.ROUTER:
        action_kinds = {MoeRectActionKind.GATE}
    elif kind is E2EOperationKind.EXPERT_FORWARD:
        refs = tuple(
            action.id
            for action in unit.plan.actions
            if action.kind is MoeRectActionKind.EXPERT_FORWARD
            and action.rank == expert
        )
        if not refs:
            raise SchemaError("expert forward action is absent", path="moe_unit.plan")
        return refs
    elif kind is E2EOperationKind.DISPATCH:
        action_kinds = {MoeRectActionKind.PACK}
    elif kind is E2EOperationKind.COMBINE:
        action_kinds = {MoeRectActionKind.WEIGHTED_COMBINE}
    else:
        raise SchemaError("not a MoE forward operation", path="operation.kind")
    if kind in (E2EOperationKind.DISPATCH, E2EOperationKind.COMBINE):
        stage = (
            MoeRectFlowStage.DISPATCH
            if kind is E2EOperationKind.DISPATCH
            else MoeRectFlowStage.COMBINE
        )
        flow_refs = {flow.id for flow in unit.plan.flows if flow.stage is stage}
        refs = [
            action.id
            for action in unit.plan.actions
            if action.kind in action_kinds or action.flow_ref in flow_refs
        ]
    else:
        refs = [
            action.id for action in unit.plan.actions if action.kind in action_kinds
        ]
    result = tuple(dict.fromkeys(refs))
    if not result:
        raise SchemaError("MoE production action closure is empty", path="moe_unit.plan")
    return result


def _binding(operation, profile: LinkedProgramProfile, unit_by_layer):
    kind = operation.kind
    layer = operation.layer
    if kind in _MOE_KINDS:
        assert layer is not None
        unit = unit_by_layer[layer]
        lowering = (
            MoeFullModelLowering.FLEXIBLE_MOE_TRACE
            if kind is E2EOperationKind.ROUTE_FREEZE
            else MoeFullModelLowering.FLEXIBLE_MOE_ACTION
        )
        refs = _moe_refs(unit, kind, operation.expert)
    elif kind in (E2EOperationKind.KV_LOAD, E2EOperationKind.KV_APPEND):
        assert layer is not None
        lowering = MoeFullModelLowering.SHARED_SPINE_STATE_ACCESS
        refs = _attention_access_refs(profile, layer)
    elif kind is E2EOperationKind.LOGITS:
        origin = profile.lowering_context.ir1.instances[0].origin_instance_id
        lowering = MoeFullModelLowering.SHARED_SPINE_VALUE
        refs = (f"{origin}.logits",)
    else:
        suffix = {
            E2EOperationKind.EMBEDDING: "embedding",
            E2EOperationKind.INPUT_NORM: f"layer{layer}.norm1",
            E2EOperationKind.QKV: f"layer{layer}.qkv",
            E2EOperationKind.ROPE: f"layer{layer}.rope",
            E2EOperationKind.ATTENTION: f"layer{layer}.attention",
            E2EOperationKind.ATTENTION_OUT: f"layer{layer}.o",
            E2EOperationKind.POST_NORM: f"layer{layer}.norm2",
            E2EOperationKind.FINAL_NORM: "final_norm",
            E2EOperationKind.LM_HEAD: "lm_head",
        }.get(kind)
        if kind is E2EOperationKind.RESIDUAL:
            assert layer is not None
            # The first residual precedes POST_NORM; the second consumes MoE combine.
            same_layer = [
                item
                for item in unit_by_layer["operations"]
                if item.layer == layer
                and item.kind is E2EOperationKind.RESIDUAL
                and item.sequence_index <= operation.sequence_index
            ]
            suffix = f"layer{layer}.residual{len(same_layer)}"
        if suffix is None:
            raise SchemaError("P3 operation lacks a full-model lowering", path=f"operation.{kind.value}")
        lowering = MoeFullModelLowering.SHARED_SPINE_NODE
        refs = (_node_ref(profile, suffix),)
    return MoeFullModelOperationBinding.create(
        operation_ref=operation.id,
        kind=kind,
        step=operation.step,
        layer=layer,
        lowering=lowering,
        production_refs=refs,
    )


def compile_moe_full_model_inference_sequence(
    manifest: WorkloadMaterializationManifest,
    legacy_template: ExperimentSpec,
    fabric: PhysicalFabric,
    *,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
) -> MoeFullModelCompileSequence:
    """Compile Prefill/Decode/Decode with Dense shared spines and MoE blocks."""

    if type(manifest) is not WorkloadMaterializationManifest:
        raise SchemaError("must be a WorkloadMaterializationManifest", path="manifest")
    _validate_inputs(manifest, legacy_template, fabric, hbm_address_spaces)
    moe_blocks = compile_moe_sequence(
        manifest, source_rank_policy="rank0_shared_spine"
    )
    operations = manifest.logical_graph.operations
    segments = []
    for step in range(3):
        profile = _compile_shared_spine(
            manifest, legacy_template, fabric, hbm_address_spaces, step
        )
        units = tuple(unit for unit in moe_blocks.units if unit.step == step)
        unit_by_layer = {unit.layer: unit for unit in units}
        unit_by_layer["operations"] = tuple(
            item for item in operations if item.step == step
        )
        origin = profile.lowering_context.ir1.instances[0].origin_instance_id
        bindings = tuple(
            _binding(operation, profile, unit_by_layer)
            for operation in operations
            if operation.step == step
        )
        replaced = tuple(
            f"{origin}.layer{layer}.{name}"
            for layer in range(manifest.request.model.num_layers)
            for name in ("gate_up", "swiglu", "down")
        )
        executable = link_moe_full_model_segment(profile, units, replaced)
        segments.append(MoeFullModelCompileSegment.create(
            phase="prefill" if step == 0 else "decode",
            step=step,
            shared_spine_profile=profile,
            shared_spine_digest=canonical_digest(profile),
            replica_die_ids=manifest.placement.active_die_ids,
            replaced_dense_mlp_node_refs=replaced,
            operation_bindings=bindings,
            moe_unit_refs=tuple(unit.id for unit in units),
            executable_manifest=executable,
            executable_manifest_digest=canonical_digest(executable),
        ))
    return MoeFullModelCompileSequence.create(
        moe_blocks=moe_blocks,
        segments=tuple(segments),
    )


__all__ = ["compile_moe_full_model_inference_sequence"]
