"""Bind selected Swizzle witnesses to exact IR-1 execution provenance."""

from __future__ import annotations

from ...errors import SchemaError
from ...schema.action import BarrierContract, BarrierScope, FusionActionKind, SyncContract
from ...schema.common import ProfileKey
from ...schema.ir0 import CollectiveWorkload, FusionImpl, FusionPattern, GemmWorkload, OpKind, ReduceOp
from ...schema.ir1 import IR1
from ...schema.swizzle import (
    SwizzleActionKind,
    SwizzleActionWitness,
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleDecision,
    SwizzlePhase,
)
from ...schema.swizzle_plan import (
    SwizzleBoundAction,
    SwizzleBoundRankProgram,
    SwizzleChunkOrigin,
    SwizzleComputeOrigin,
    SwizzleDeploymentSelection,
    SwizzleFusionPlan,
    SwizzleReductionOrigin,
    SwizzleReductionOriginKind,
    SwizzleValueOrigin,
    SwizzleValueUse,
)
from .materialize import materialize_swizzle_decision, materialize_swizzle_selection


def _member_ref(decision: SwizzleDecision, action: SwizzleActionWitness) -> str:
    problem = decision.problem
    if action.kind is SwizzleActionKind.COMP:
        return problem.gemm.node_ref
    if problem.pattern is FusionPattern.AG_GEMM and action.kind in (
        SwizzleActionKind.REDUCE,
        SwizzleActionKind.LOCAL_COPY,
    ):
        return problem.gemm.node_ref
    return problem.collective.node_ref


def _chunk_origin(
    decision: SwizzleDecision,
    candidate: SwizzleCandidate,
    chunk: int | None,
) -> SwizzleChunkOrigin | None:
    if chunk is None:
        return None
    axis = candidate.split_axis
    if axis is None or axis.extent % candidate.chunk_count:
        raise SchemaError("selected split axis is not uniformly chunkable", path="decision.selected_candidate_ref")
    views = (
        decision.problem.gemm.lhs,
        decision.problem.gemm.rhs,
        decision.problem.gemm.output,
        decision.problem.collective.input,
        decision.problem.collective.output,
    )
    view = next(
        (
            item
            for item in views
            if item.value_ref == axis.tensor_ref and item.shape[axis.index] == axis.extent
        ),
        None,
    )
    if view is None or chunk >= candidate.chunk_count:
        raise SchemaError("chunk has no exact problem tensor view", path="swizzle_action.chunk_index")
    extent = axis.extent // candidate.chunk_count
    offset = tuple(chunk * extent if index == axis.index else 0 for index in range(len(view.shape)))
    shape = tuple(extent if index == axis.index else value for index, value in enumerate(view.shape))
    return SwizzleChunkOrigin(
        source_value_ref=view.value_ref,
        axis=axis.index,
        chunk_index=chunk,
        logical_offset=offset,
        logical_shape=shape,
    )


def _sync(
    decision: SwizzleDecision,
    action: SwizzleActionWitness,
    ranks: tuple[int, ...],
) -> SyncContract:
    completion = f"event.{action.id}"
    if action.kind is SwizzleActionKind.WAIT:
        if len(action.deps) != 1:
            raise SchemaError("WAIT requires exactly one producer dependency", path="swizzle_action.deps")
        return SyncContract(completion, f"event.{action.deps[0]}", None)
    if action.kind is SwizzleActionKind.BARRIER:
        suffix = "none" if action.chunk_index is None else str(action.chunk_index)
        barrier_id = f"barrier.{decision.selected_candidate_ref}.{action.phase.value}.{suffix}"
        return SyncContract(
            completion,
            None,
            BarrierContract(barrier_id, ranks, len(ranks), BarrierScope.GROUP),
        )
    return SyncContract(completion, None, None)


def _origin_for_ref(
    ref: str,
    use: SwizzleValueUse,
    *,
    action: SwizzleActionWitness,
    logical_refs: tuple[str, ...],
    local_refs: tuple[str, ...],
    gemm_ref: str,
    producers: dict[tuple[str, int | None], str],
) -> SwizzleValueOrigin:
    if use is SwizzleValueUse.WRITE and action.kind not in (
        SwizzleActionKind.WAIT,
        SwizzleActionKind.BARRIER,
    ):
        return SwizzleValueOrigin(ref, use, None, action.id, None)
    producer = producers.get((ref, action.chunk_index))
    if producer is not None:
        return SwizzleValueOrigin(ref, use, None, producer, None)
    logical = next(
        (item for item in logical_refs if ref == item or ref.startswith(f"{item}::")),
        None,
    )
    if logical is not None:
        return SwizzleValueOrigin(ref, use, logical, None, None)
    if any(ref == item or ref.startswith(f"{item}::") for item in local_refs):
        return SwizzleValueOrigin(ref, use, None, None, gemm_ref)
    raise SchemaError(
        f"value {ref!r} has no IR-1, local-operand, or producer origin",
        path="swizzle_action.value_origins",
    )


def _index_temporary_producers(
    actions: tuple[SwizzleActionWitness, ...],
) -> dict[tuple[str, int | None], str]:
    producers: dict[tuple[str, int | None], str] = {}
    for action in actions:
        if action.kind in (SwizzleActionKind.WAIT, SwizzleActionKind.BARRIER):
            continue
        for ref in action.output_refs:
            key = (ref, action.chunk_index)
            prior = producers.get(key)
            if prior is not None and prior != action.id:
                raise SchemaError(
                    "temporary has multiple data producers in one chunk",
                    path="candidate.rank_programs",
                )
            producers[key] = action.id
    return producers



def materialize_swizzle_plan(
    ir1: IR1,
    decision: SwizzleDecision,
    profile: ProfileKey,
    *,
    deployment_selection: SwizzleDeploymentSelection | None = None,
) -> SwizzleFusionPlan:
    """Build an exact SwizzleFusionPlan without synthesizing missing semantics."""

    if type(ir1) is not IR1 or type(decision) is not SwizzleDecision or type(profile) is not ProfileKey:
        raise SchemaError("requires typed IR1, decision and profile", path="swizzle_materialize")
    ir1.validate("ir1")
    decision.validate("decision")
    profile.validate("profile")
    if deployment_selection is None:
        adapter = materialize_swizzle_decision(decision)
    else:
        if type(deployment_selection) is not SwizzleDeploymentSelection:
            raise SchemaError(
                "must be a SwizzleDeploymentSelection",
                path="deployment_selection",
            )
        deployment_selection.validate("deployment_selection")
        if deployment_selection.economic_decision != decision:
            raise SchemaError(
                "deployment selection belongs to a different economic decision",
                path="deployment_selection.economic_decision",
            )
        adapter = materialize_swizzle_selection(deployment_selection)
    candidate = adapter.candidate
    if decision.problem.source_ir1_id != ir1.id:
        raise SchemaError("decision belongs to a different IR-1", path="decision.problem.source_ir1_id")
    skeleton = next((item for item in ir1.fused_op_skeletons if item.id == decision.problem.fused_op_id), None)
    group = next((item for item in ir1.groups if item.id == decision.problem.group.group_ref), None)
    if skeleton is None or group is None:
        raise SchemaError("decision references a missing skeleton/group", path="decision.problem")
    owner_profile = ir1.profile if not ir1.instance_profiles else next(
        (item.profile for item in ir1.instance_profiles if item.instance_ref == skeleton.instance_id),
        None,
    )
    if owner_profile != profile:
        raise SchemaError("profile does not own fused skeleton", path="profile")
    nodes = {node.id: node for node in ir1.nodes}
    gemm = nodes.get(decision.problem.gemm.node_ref)
    collective = nodes.get(decision.problem.collective.node_ref)
    if gemm is None or collective is None or type(gemm.workload) is not GemmWorkload or type(collective.workload) is not CollectiveWorkload:
        raise SchemaError("problem descriptors must bind exact GEMM/collective IR-1 members", path="decision.problem")
    ranks = tuple(item.rank for item in group.placements)
    logical_refs = tuple(item.id for item in ir1.values)
    local_refs = decision.problem.gemm.local_operand_refs
    all_witnesses = tuple(action for program in candidate.rank_programs for action in program.actions)
    producers = _index_temporary_producers(all_witnesses)
    adapter_by_rank = {program.rank: program for program in adapter.rank_programs}
    bound_programs = []
    for source_program in candidate.rank_programs:
        adapter_actions = adapter_by_rank[source_program.rank].actions
        bound_actions = []
        for source, routed in zip(source_program.actions, adapter_actions, strict=True):
            member_ref = _member_ref(decision, source)
            member = nodes[member_ref]
            compute = None
            reduction = None
            if source.kind is SwizzleActionKind.COMP:
                if member.kind is not OpKind.GEMM or type(member.workload) is not GemmWorkload:
                    raise SchemaError("COMP origin must be the exact IR-1 GEMM", path="swizzle_action.member_ref")
                compute = SwizzleComputeOrigin(
                    member.id,
                    member.workload,
                    member.math,
                    member.effects,
                    member.impl_ref,
                    member.inputs,
                    member.outputs,
                )
            elif source.kind is SwizzleActionKind.REDUCE:
                if decision.problem.pattern is FusionPattern.AG_GEMM:
                    reduction = SwizzleReductionOrigin(
                        gemm.id,
                        SwizzleReductionOriginKind.GEMM_ACCUMULATION,
                        ReduceOp.SUM,
                        gemm.math,
                        None,
                    )
                else:
                    reduction = SwizzleReductionOrigin(
                        collective.id,
                        SwizzleReductionOriginKind.COLLECTIVE_REDUCTION,
                        ReduceOp.SUM,
                        collective.math,
                        collective.workload,
                    )
            value_origins = tuple(
                _origin_for_ref(
                    ref,
                    use,
                    action=source,
                    logical_refs=logical_refs,
                    local_refs=local_refs,
                    gemm_ref=gemm.id,
                    producers=producers,
                )
                for use, refs in (
                    (SwizzleValueUse.READ, source.input_refs),
                    (SwizzleValueUse.WRITE, source.output_refs),
                )
                for ref in refs
            )
            bound_actions.append(
                SwizzleBoundAction(
                    source_action=source,
                    fusion_kind=routed.fusion_kind,
                    member_ref=member_ref,
                    expected_route=routed.expected_route,
                    chunk_origin=_chunk_origin(decision, candidate, source.chunk_index),
                    value_origins=value_origins,
                    compute_origin=compute,
                    reduction_origin=reduction,
                    sync=_sync(decision, source, ranks),
                )
            )
        bound_programs.append(SwizzleBoundRankProgram(source_program.rank, tuple(bound_actions)))
    plan = SwizzleFusionPlan.create(
        producer_pass="inter_die_plan",
        source_ir1_id=ir1.id,
        fused_op_id=skeleton.id,
        group_ref=group.id,
        impl=FusionImpl.SWIZZLE_TOPO,
        profile_key=profile,
        pattern=candidate.pattern,
        algorithm=candidate.algorithm,
        deployment_selection=adapter.deployment_selection,
        rank_programs=tuple(bound_programs),
        buffer_requirements=candidate.buffer_requirements,
    )
    plan.validate_against(ir1)
    return plan


__all__ = ["materialize_swizzle_plan"]
