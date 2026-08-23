"""Executable capacity-safe UNFUSED baseline from generalized MoE execution."""

from __future__ import annotations

from ...errors import SchemaError
from ...schema.ir0 import FusionPattern
from ...schema.swizzle import SwizzleActionKind, SwizzleAlgorithm
from ...schema.swizzle_moe import (
    MoeActionWitness,
    MoeRankProgram,
    MoeSwizzleCandidate,
    MoeSwizzleProblem,
)
from ...schema.swizzle_moe_calibration import MoeSwizzleCalibrationProfile
from ...schema.swizzle_moe_execution import (
    MoeScaleExecution,
    MoeScaleExecutionActionKind,
    MoeScaleExecutionFlowRole,
)
from ...schema.swizzle_moe_scale import MoeSwizzleScaleOracle, MoeSwizzleScaleSpec
from .moe_cost import build_moe_action_owner_map, build_moe_swizzle_cost


def _endpoint_session_dependencies(
    problem, actions, base_dependencies=None, *, owners=None,
):
    owners = (
        build_moe_action_owner_map(problem, actions)
        if owners is None else owners
    )
    if set(owners) != {action.id for action in actions}:
        raise SchemaError(
            "endpoint owner coverage is not exact",
            path="moe_unfused.sessions",
        )
    by_packet = {}
    for action in actions:
        if action.packet_ref is not None:
            by_packet.setdefault((action.packet_ref, action.stage), {})[action.kind] = action
    extra = (
        {action.id: set(action.deps) for action in actions}
        if base_dependencies is None
        else {action.id: set(base_dependencies[action.id]) for action in actions}
    )
    action_index = {action.id: action for action in actions}
    action_packet = {
        action.id: (action.packet_ref, action.stage)
        for action in actions
        if action.packet_ref is not None
    }
    ancestor_cache = {}

    def ancestors(ref):
        if ref not in ancestor_cache:
            result = set(extra[ref])
            for dependency in extra[ref]:
                result.update(ancestors(dependency))
            ancestor_cache[ref] = result
        return ancestor_cache[ref]

    packet_predecessors = {key: set() for key in by_packet}
    for key, triple in by_packet.items():
        for action in triple.values():
            for dependency in ancestors(action.id):
                predecessor = action_packet.get(dependency)
                if predecessor is not None and predecessor != key:
                    packet_predecessors[key].add(predecessor)

    def packet_key(key):
        triple = by_packet[key]
        send = triple[SwizzleActionKind.SEND]
        recv = triple[SwizzleActionKind.RECV]
        return (
            send.pipeline_index if send.pipeline_index is not None else send.tile_index,
            send.stage,
            send.rank,
            recv.rank,
            key,
        )

    prior_wait = {}
    next_lane = {}
    pending_packets = set(by_packet)
    admitted_packets = set()
    while pending_packets:
        ready = sorted(
            (
                key for key in pending_packets
                if packet_predecessors[key].issubset(admitted_packets)
            ),
            key=packet_key,
        )
        if not ready:
            raise SchemaError(
                "transport packet dependency graph is cyclic",
                path="moe_unfused.sessions",
            )
        key = ready[0]
        triple = by_packet[key]
        send = triple[SwizzleActionKind.SEND]
        recv = triple[SwizzleActionKind.RECV]
        wait = triple[SwizzleActionKind.WAIT]
        endpoints = (
            (owners[send.id].runtime_core_id, send),
            (owners[recv.id].runtime_core_id, recv),
        )
        for runtime_core, endpoint_action in endpoints:
            lane = next_lane.get(runtime_core, 0)
            next_lane[runtime_core] = (lane + 1) % problem.endpoint_session_capacity
            previous = prior_wait.get((runtime_core, lane))
            if previous is not None:
                extra[endpoint_action.id].add(previous)
            prior_wait[(runtime_core, lane)] = wait.id
        admitted_packets.add(key)
        pending_packets.remove(key)
    return extra


def _apply_endpoint_session_dependencies(problem, actions):
    extra = _endpoint_session_dependencies(problem, actions)
    order = {action.id: index for index, action in enumerate(actions)}
    pending = set(order)
    rebuilt = {}
    result = []
    while pending:
        ready = sorted((ref for ref in pending if extra[ref].issubset(rebuilt)), key=order.__getitem__)
        if not ready:
            raise SchemaError("per-core session dependencies create cycle", path="moe_unfused.sessions")
        ref = ready[0]
        action = actions[order[ref]]
        semantic = {name: getattr(action, name) for name in action.__dataclass_fields__ if name not in ("schema_version", "id")}
        semantic["deps"] = tuple(rebuilt[dep].id for dep in sorted(extra[ref], key=order.__getitem__))
        rebuilt[ref] = MoeActionWitness.create(**semantic)
        result.append(rebuilt[ref])
        pending.remove(ref)
    return tuple(result)


def build_executable_moe_unfused_baseline(
    problem: MoeSwizzleProblem,
    spec: MoeSwizzleScaleSpec,
    oracle: MoeSwizzleScaleOracle,
    execution: MoeScaleExecution,
    *,
    calibration_profile: MoeSwizzleCalibrationProfile | None = None,
) -> MoeSwizzleCandidate:
    problem.validate("moe_unfused.problem")
    execution.validate_against(spec, oracle, "moe_unfused.execution")
    if (
        problem.source_execution_id != execution.id
        or problem.region.source_spec_id != spec.id
        or problem.region.source_oracle_id != oracle.id
    ):
        raise SchemaError("problem source provenance mismatch", path="moe_unfused")
    region = problem.region
    pattern = region.pattern
    originals = {
        item.id: item for item in execution.actions if item.id in set(region.member_refs)
    }
    if set(originals) != set(region.member_refs):
        raise SchemaError("region action slice is not executable", path="moe_unfused.originals")
    action_by_token_role = {
        (item.token_index, item.role): item for item in originals.values()
    }
    flow_by_ref = {item.flow_ref: item for item in execution.flows}
    generated: list[MoeActionWitness] = []
    wait_by_flow: dict[str, str] = {}
    comp_by_assignment: dict[str, MoeActionWitness] = {}

    def make_comp(assignment: object, role: str) -> MoeActionWitness:
        original = action_by_token_role.get((assignment.token_index, role))
        if (
            original is None
            or original.kind is not MoeScaleExecutionActionKind.GEMM
            or original.die_id != assignment.expert_rank
        ):
            raise SchemaError("assignment lacks exact original GEMM", path="moe_unfused.comp")
        flow_ref = (
            assignment.dispatch_flow_ref
            if pattern is FusionPattern.MOE_DISPATCH_GEMM
            else None
        )
        owns_dynamic_root = (
            assignment.dispatch_flow_ref is not None
            if pattern is FusionPattern.MOE_DISPATCH_GEMM
            else assignment.combine_flow_ref is not None
        )
        dynamic_family = (
            "dispatch_operand"
            if pattern is FusionPattern.MOE_DISPATCH_GEMM
            else "combine_output"
        )
        deps = () if flow_ref is None else (wait_by_flow[flow_ref],)
        result = MoeActionWitness.create(
            rank=original.die_id,
            kind=SwizzleActionKind.COMP,
            deps=deps,
            assignment_refs=(assignment.id,),
            expert_index=assignment.expert_index,
            tile_index=assignment.token_index,
            n_block_index=(0 if pattern is FusionPattern.MOE_GEMM_COMBINE else None),
            packet_ref=None,
            stage=None,
            pivot_rank=None,
            original_action_refs=(original.id,),
            route_ref=None,
            peer_rank=None,
            logical_bytes=0,
            flops=original.flops,
            work_role=role,
            pipeline_index=(assignment.token_index if owns_dynamic_root else None),
            buffer_slot=(0 if owns_dynamic_root else None),
            buffer_family=(dynamic_family if owns_dynamic_root else None),
        )
        generated.append(result)
        return result

    assignments = region.semantic_witness.traffic.assignments
    if pattern is FusionPattern.MOE_GEMM_COMBINE:
        for assignment in assignments:
            comp_by_assignment[assignment.id] = make_comp(assignment, "down")

    flow_role = (
        MoeScaleExecutionFlowRole.DISPATCH
        if pattern is FusionPattern.MOE_DISPATCH_GEMM
        else MoeScaleExecutionFlowRole.COMBINE
    )
    for assignment in assignments:
        flow_ref = (
            assignment.dispatch_flow_ref
            if pattern is FusionPattern.MOE_DISPATCH_GEMM
            else assignment.combine_flow_ref
        )
        route_ref = (
            assignment.dispatch_route_ref
            if pattern is FusionPattern.MOE_DISPATCH_GEMM
            else assignment.combine_route_ref
        )
        if flow_ref is None:
            continue
        flow = flow_by_ref.get(flow_ref)
        if flow is None or flow.role is not flow_role:
            raise SchemaError("assignment flow role is not exact", path="moe_unfused.flow")
        send_original = originals[flow.send_action_ref]
        recv_original = originals[flow.recv_action_ref]
        wait_original = originals[flow.wait_action_ref]
        wave_deps = ()
        send_deps = wave_deps
        if pattern is FusionPattern.MOE_GEMM_COMBINE:
            send_deps = (comp_by_assignment[assignment.id].id,) + tuple(
                ref for ref in wave_deps if ref != comp_by_assignment[assignment.id].id
            )
        send = MoeActionWitness.create(
            rank=flow.source_die_id,
            kind=SwizzleActionKind.SEND,
            deps=send_deps,
            assignment_refs=(assignment.id,),
            expert_index=assignment.expert_index,
            tile_index=assignment.token_index,
            n_block_index=(0 if pattern is FusionPattern.MOE_GEMM_COMBINE else None),
            packet_ref=flow.flow_ref,
            stage=0,
            pivot_rank=None,
            original_action_refs=(send_original.id,),
            route_ref=route_ref,
            peer_rank=flow.destination_die_id,
            logical_bytes=flow.bytes,
            flops=0,
            work_role=f"{pattern.value}.transport",
            pipeline_index=(
                assignment.token_index
                if pattern is FusionPattern.MOE_GEMM_COMBINE else None
            ),
            buffer_slot=(
                0 if pattern is FusionPattern.MOE_GEMM_COMBINE else None
            ),
            buffer_family=(
                "combine_output"
                if pattern is FusionPattern.MOE_GEMM_COMBINE else None
            ),
        )
        recv = MoeActionWitness.create(
            rank=flow.destination_die_id,
            kind=SwizzleActionKind.RECV,
            deps=wave_deps,
            assignment_refs=(assignment.id,),
            expert_index=assignment.expert_index,
            tile_index=assignment.token_index,
            n_block_index=(0 if pattern is FusionPattern.MOE_GEMM_COMBINE else None),
            packet_ref=flow.flow_ref,
            stage=0,
            pivot_rank=None,
            original_action_refs=(recv_original.id,),
            route_ref=route_ref,
            peer_rank=flow.source_die_id,
            logical_bytes=flow.bytes,
            flops=0,
            work_role=f"{pattern.value}.transport",
            pipeline_index=(
                assignment.token_index
                if pattern is FusionPattern.MOE_DISPATCH_GEMM else None
            ),
            buffer_slot=(
                0 if pattern is FusionPattern.MOE_DISPATCH_GEMM else None
            ),
            buffer_family=(
                "dispatch_operand"
                if pattern is FusionPattern.MOE_DISPATCH_GEMM else None
            ),
        )
        wait = MoeActionWitness.create(
            rank=flow.destination_die_id,
            kind=SwizzleActionKind.WAIT,
            deps=(send.id, recv.id),
            assignment_refs=(assignment.id,),
            expert_index=assignment.expert_index,
            tile_index=assignment.token_index,
            n_block_index=(0 if pattern is FusionPattern.MOE_GEMM_COMBINE else None),
            packet_ref=flow.flow_ref,
            stage=0,
            pivot_rank=None,
            original_action_refs=(wait_original.id,),
            route_ref=None,
            peer_rank=None,
            logical_bytes=0,
            flops=0,
            work_role=f"{pattern.value}.transport",
        )
        generated.extend((send, recv, wait))
        wait_by_flow[flow.flow_ref] = wait.id

    if pattern is FusionPattern.MOE_DISPATCH_GEMM:
        for assignment in assignments:
            gate = make_comp(assignment, "gate")
            up = make_comp(assignment, "up")
            original = action_by_token_role.get((assignment.token_index, "swiglu"))
            if (
                original is None
                or original.kind is not MoeScaleExecutionActionKind.SWIGLU
                or original.id != assignment.swiglu_action_ref
            ):
                raise SchemaError(
                    "assignment lacks exact original SWIGLU",
                    path="moe_unfused.swiglu",
                )
            generated.append(MoeActionWitness.create(
                rank=assignment.expert_rank,
                kind=SwizzleActionKind.SWIGLU,
                deps=(gate.id, up.id),
                assignment_refs=(assignment.id,),
                expert_index=assignment.expert_index,
                tile_index=assignment.token_index,
                n_block_index=None,
                packet_ref=None,
                stage=None,
                pivot_rank=None,
                original_action_refs=(original.id,),
                route_ref=None,
                peer_rank=None,
                logical_bytes=spec.intermediate_size * 2,
                flops=0,
                work_role="swiglu",
                pipeline_index=assignment.token_index,
                buffer_slot=0,
                buffer_family="dispatch_operand",
                packed_value_ref=f"moe.swiglu.{assignment.token_index}",
            ))
    generated = list(_apply_endpoint_session_dependencies(problem, tuple(generated)))
    original_refs = tuple(ref for item in generated for ref in item.original_action_refs)
    if len(original_refs) != len(set(original_refs)) or set(original_refs) != set(region.member_refs):
        raise SchemaError("baseline does not cover the region exactly once", path="moe_unfused.original_action_refs")
    rank_programs = tuple(
        MoeRankProgram(
            rank=rank,
            actions=tuple(item for item in generated if item.rank == rank),
        )
        for rank in range(spec.mesh_rows * spec.mesh_columns)
    )
    actions = tuple(item for program in rank_programs for item in program.actions)
    cost = build_moe_swizzle_cost(
        problem,
        SwizzleAlgorithm.UNFUSED,
        actions,
        (),
        calibration_profile,
    )
    result = MoeSwizzleCandidate.create(
        problem_ref=problem.id,
        pattern=pattern,
        algorithm=SwizzleAlgorithm.UNFUSED,
        packetization=(),
        tile_schedule=(),
        rank_programs=rank_programs,
        expert_wave_count=1,
        token_block_size=1,
        output_column_block_size=(
            spec.intermediate_size
            if pattern is FusionPattern.MOE_DISPATCH_GEMM
            else spec.hidden_size
        ),
        compute_output_block_count=1,
        transport_output_block_count=1,
        unroll_degree=1,
        double_buffer=False,
        compute_core_fraction=0.75,
        communication_core_fraction=0.25,
        original_action_refs=tuple(sorted(original_refs)),
        cost=cost,
    )
    result.validate_against(problem)
    return result


__all__ = ["build_executable_moe_unfused_baseline"]
