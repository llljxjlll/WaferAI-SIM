"""Direct-XY personalized packet and GroupGEMM candidate generation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from ...errors import SchemaError
from ...schema.ir0 import FusionPattern
from ...schema.swizzle import SwizzleActionKind, SwizzleAlgorithm
from ...schema.swizzle_moe import (
    MoeActionWitness,
    MoePacketSlice,
    MoePacketWitness,
    MoeRankProgram,
    MoeSwizzleCandidate,
    MoeSwizzleProblem,
    MoeTileWitness,
    MoeTokenAssignmentView,
)
from ...schema.swizzle_moe_calibration import MoeSwizzleCalibrationProfile
from ...schema.swizzle_moe_execution import MoeScaleExecution
from ...schema.swizzle_moe_placement import (
    build_moe_candidate_action_owner_map,
)
from ...schema.swizzle_moe_scale import MoeSwizzleScaleOracle, MoeSwizzleScaleSpec
from .moe_cost import build_moe_swizzle_cost
from .moe_unfused import _endpoint_session_dependencies


_COMBINE_TRANSPORT_BLOCK_COUNTS = (1, 2)


def _validate_sources(problem, spec, oracle, execution, path: str) -> None:
    problem.validate(f"{path}.problem")
    execution.validate_against(spec, oracle, f"{path}.execution")
    if (
        problem.source_execution_id != execution.id
        or problem.region.source_spec_id != spec.id
        or problem.region.source_oracle_id != oracle.id
    ):
        raise SchemaError("problem source provenance mismatch", path=path)


def _endpoints(assignment, pattern):
    return (
        (assignment.source_rank, assignment.expert_rank, assignment.dispatch_route_ref)
        if pattern is FusionPattern.MOE_DISPATCH_GEMM
        else (assignment.expert_rank, assignment.source_rank, assignment.combine_route_ref)
    )


def _whole_offsets(assignment, pattern):
    token = assignment.token_index * assignment.payload_bytes
    expert = assignment.contributor_ordinal * assignment.payload_bytes
    return (token, expert) if pattern is FusionPattern.MOE_DISPATCH_GEMM else (expert, token)


def _m_block_by_assignment(problem, token_block_size, algorithm):
    pattern = problem.region.pattern
    routes = {item.id: item for item in problem.topology.group.routes}
    pivots = {
        (source, destination): pivot
        for source, destination, pivot in problem.topology.pivot_by_pair
    }
    grouped = defaultdict(list)
    for item in problem.region.semantic_witness.traffic.assignments:
        source, destination, route_ref = _endpoints(item, pattern)
        if source == destination:
            arrival = 0
        elif algorithm is SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A:
            arrival = len(routes[route_ref].die_path) - 1
        else:
            arrival = 1 if pivots[(source, destination)] in (source, destination) else 2
        grouped[(arrival, item.expert_index)].append(item)
    result = {}
    next_m_block_by_expert = defaultdict(int)
    for arrival, expert in sorted(grouped):
        ordered = sorted(
            grouped[(arrival, expert)],
            key=lambda item: (item.token_index, item.source_rank, item.id),
        )
        for start in range(0, len(ordered), token_block_size):
            m_block = next_m_block_by_expert[expert]
            next_m_block_by_expert[expert] += 1
            for item in ordered[start:start + token_block_size]:
                result[item.id] = m_block
    return result


def _remote_runs(problem, token_block_size, algorithm):
    pattern = problem.region.pattern
    m_blocks = _m_block_by_assignment(problem, token_block_size, algorithm)
    grouped = defaultdict(list)
    for item in problem.region.semantic_witness.traffic.assignments:
        if item.source_rank == item.expert_rank:
            continue
        grouped[(
            *_endpoints(item, pattern)[:2], item.expert_index, m_blocks[item.id]
        )].append(item)
    return tuple(
        tuple(sorted(
            grouped[key],
            key=lambda item: (item.token_index, item.contributor_ordinal, item.id),
        ))
        for key in sorted(grouped)
    )


def _packet(
    assignments,
    pattern,
    *,
    stage,
    source_rank,
    destination_rank,
    pivot_rank,
    route_ref,
    n_block_index=None,
    slice_bytes=None,
    source_offsets: Callable | None = None,
    destination_offsets: Callable | None = None,
):
    slices = []
    for assignment in assignments:
        whole_source, whole_destination = _whole_offsets(assignment, pattern)
        extent = assignment.payload_bytes if slice_bytes is None else slice_bytes
        block_offset = 0 if n_block_index is None else n_block_index * extent
        source_offset = whole_source + block_offset
        destination_offset = whole_destination + block_offset
        slices.append(
            MoePacketSlice(
                assignment_ref=assignment.id,
                source_offset_bytes=(
                    source_offset if source_offsets is None else source_offsets(assignment)
                ),
                destination_offset_bytes=(
                    destination_offset
                    if destination_offsets is None
                    else destination_offsets(assignment)
                ),
                bytes=extent,
                n_block_index=n_block_index,
            )
        )
    return MoePacketWitness.create(
        stage=stage,
        source_rank=source_rank,
        destination_rank=destination_rank,
        pivot_rank=pivot_rank,
        route_ref=route_ref,
        slices=tuple(slices),
        logical_bytes=sum(item.bytes for item in slices),
    )


def _output_refs(region, execution, assignment):
    outputs = set(region.boundary_output_refs)
    return tuple(dict.fromkeys(
        value_ref
        for action in execution.actions
        if action.id in set(region.member_refs) and action.token_index == assignment.token_index
        for value_ref in action.write_values
        if value_ref in outputs
    ))


def _build_tiles(
    problem,
    spec,
    execution,
    algorithm,
    packets,
    final_packet_by_block,
    predecessor_by_packet,
    physical_slots,
    token_block_size,
    compute_output_block_count,
):
    region = problem.region
    pattern = region.pattern
    assignments = region.semantic_witness.traffic.assignments
    packet_index = {item.id: item for item in packets}
    route_index = {item.id: item for item in problem.topology.group.routes}
    block_extent = spec.hidden_size // compute_output_block_count
    groups = defaultdict(list)
    for assignment in assignments:
        representative_block = (
            None if pattern is FusionPattern.MOE_DISPATCH_GEMM else 0
        )
        packet_ref = final_packet_by_block.get(
            (assignment.id, representative_block)
        )
        if packet_ref is None:
            arrival = 0
        elif algorithm is SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A:
            arrival = (
                len(route_index[packet_index[packet_ref].route_ref].die_path)
                - 1
            )
        else:
            arrival = 2 if packet_ref in predecessor_by_packet else 1
        groups[(arrival, assignment.expert_index)].append(assignment)
    tiles = []
    next_m_block_by_expert = defaultdict(int)
    for arrival, expert in sorted(groups):
        ordered = sorted(
            groups[(arrival, expert)],
            key=lambda item: (item.token_index, item.source_rank, item.id),
        )
        for start in range(0, len(ordered), token_block_size):
            block = tuple(ordered[start : start + token_block_size])
            m_block_index = next_m_block_by_expert[expert]
            next_m_block_by_expert[expert] += 1
            n_blocks = (
                (None,)
                if pattern is FusionPattern.MOE_DISPATCH_GEMM
                else tuple(range(compute_output_block_count))
            )
            for n_block in n_blocks:
                required = tuple(dict.fromkeys(
                    final_packet_by_block[(item.id, n_block)]
                    for item in block
                    if (item.id, n_block) in final_packet_by_block
                ))
                output_refs = tuple(dict.fromkeys(
                    ref
                    for item in block
                    for ref in _output_refs(region, execution, item)
                ))
                tiles.append(
                    MoeTileWitness.create(
                        expert_index=expert,
                        tile_index=len(tiles),
                        m_block_index=m_block_index,
                        m_block_size=len(block),
                        split_axis=region.semantic_witness.split_axis,
                        assignment_refs=tuple(item.id for item in block),
                        arrival_class=arrival,
                        required_packet_refs=required,
                        output_value_refs=output_refs,
                        n_block_index=n_block,
                        output_column_offset=(
                            0 if n_block is None else n_block * block_extent
                        ),
                        output_column_extent=(
                            spec.intermediate_size
                            if pattern is FusionPattern.MOE_DISPATCH_GEMM
                            else block_extent
                        ),
                    )
                )
    return tuple(tiles), token_block_size


def _build_program(
    problem,
    spec,
    execution,
    algorithm,
    packets,
    final_packet_by_block,
    *,
    first_stage_packets,
    predecessor_by_packet,
    physical_slots,
    token_block_size,
    compute_output_block_count,
):
    region = problem.region
    pattern = region.pattern
    assignments = region.semantic_witness.traffic.assignments
    assignment_index = {item.id: item for item in assignments}
    flow_index = {item.flow_ref: item for item in execution.flows}
    original_index = {item.id: item for item in execution.actions}
    tiles, token_block_size = _build_tiles(
        problem, spec, execution, algorithm, packets,
        final_packet_by_block, predecessor_by_packet, physical_slots,
        token_block_size, compute_output_block_count,
    )
    tile_by_assignment_block = {
        (assignment_ref, tile.n_block_index): tile
        for tile in tiles
        for assignment_ref in tile.assignment_refs
    }
    tile_pipeline_index = {
        tile.tile_index: tile.m_block_index for tile in tiles
    }
    generated = []
    comp_by_block = {}

    def emit_comp(tile):
        roles = ("gate", "up") if pattern is FusionPattern.MOE_DISPATCH_GEMM else ("down",)
        results = []
        pipeline_index = tile_pipeline_index[tile.tile_index]
        for role in roles:
            originals = tuple(
                item for item in execution.actions
                if item.id in set(region.member_refs)
                and item.role == role
                and item.token_index in {
                    assignment_index[ref].token_index for ref in tile.assignment_refs
                }
            )
            n_blocks = (
                1
                if tile.n_block_index is None
                else compute_output_block_count
            )
            flops = sum(item.flops for item in originals) // n_blocks
            original_refs = (
                tuple(item.id for item in originals)
                if tile.n_block_index in (None, 0)
                else ()
            )
            deps = tuple(
                wait_by_packet[ref] for ref in tile.required_packet_refs
            ) if pattern is FusionPattern.MOE_DISPATCH_GEMM else ()
            action = MoeActionWitness.create(
                rank=tile.expert_index,
                kind=SwizzleActionKind.COMP,
                deps=deps,
                assignment_refs=tile.assignment_refs,
                expert_index=tile.expert_index,
                tile_index=tile.tile_index,
                n_block_index=tile.n_block_index,
                packet_ref=None,
                stage=None,
                pivot_rank=None,
                original_action_refs=original_refs,
                route_ref=None,
                peer_rank=None,
                logical_bytes=0,
                flops=flops,
                work_role=role,
                pipeline_index=pipeline_index,
                buffer_slot=pipeline_index % physical_slots,
                buffer_family=("dispatch_operand" if pattern is FusionPattern.MOE_DISPATCH_GEMM else "combine_output"),
            )
            generated.append(action)
            results.append(action)
        if pattern is FusionPattern.MOE_DISPATCH_GEMM:
            swiglu = MoeActionWitness.create(
                rank=tile.expert_index,
                kind=SwizzleActionKind.SWIGLU,
                deps=tuple(item.id for item in results),
                assignment_refs=tile.assignment_refs,
                expert_index=tile.expert_index,
                tile_index=tile.tile_index,
                n_block_index=None,
                packet_ref=None,
                stage=None,
                pivot_rank=None,
                original_action_refs=tuple(
                    assignment_index[ref].swiglu_action_ref
                    for ref in tile.assignment_refs
                ),
                route_ref=None,
                peer_rank=None,
                logical_bytes=len(tile.assignment_refs) * spec.intermediate_size * 2,
                flops=0,
                work_role="swiglu",
                pipeline_index=pipeline_index,
                buffer_slot=pipeline_index % physical_slots,
                buffer_family="dispatch_operand",
                packed_value_ref=f"moe.swiglu.{tile.tile_index}",
            )
            generated.append(swiglu)
        return tuple(results)

    if pattern is FusionPattern.MOE_GEMM_COMBINE:
        wait_by_packet = {}
        for tile in tiles:
            comp = emit_comp(tile)[0]
            for ref in tile.assignment_refs:
                comp_by_block[(ref, tile.n_block_index)] = comp.id

    wait_by_packet = {}
    for packet_index, packet in enumerate(packets):
        slices = packet.slices
        packet_assignments = tuple(assignment_index[item.assignment_ref] for item in slices)
        n_block = slices[0].n_block_index
        if any(item.n_block_index != n_block for item in slices):
            raise SchemaError("packet mixes N blocks", path="moe_candidate.packetization")
        first = packet.id in first_stage_packets
        original_send = []
        original_recv = []
        original_wait = []
        if first and n_block in (None, 0):
            for assignment in packet_assignments:
                flow_ref = (
                    assignment.dispatch_flow_ref
                    if pattern is FusionPattern.MOE_DISPATCH_GEMM
                    else assignment.combine_flow_ref
                )
                flow = flow_index[flow_ref]
                original_send.append(flow.send_action_ref)
                original_recv.append(flow.recv_action_ref)
                original_wait.append(flow.wait_action_ref)
        compute_blocks = (
            (None,)
            if pattern is FusionPattern.MOE_DISPATCH_GEMM
            else (0,)
            if compute_output_block_count == 1
            else (
                (n_block,)
                if n_block is not None
                else tuple(range(compute_output_block_count))
            )
        )
        packet_tiles = {
            tile_pipeline_index[
                tile_by_assignment_block[(item.id, block)].tile_index
            ]
            for item in packet_assignments
            for block in compute_blocks
        }
        if len(packet_tiles) != 1:
            raise SchemaError(
                "one packet cannot span multiple M-block tiles",
                path="moe_candidate.packetization",
            )
        packet_pipeline = next(iter(packet_tiles))
        deps = []
        if packet.id in predecessor_by_packet:
            deps.append(wait_by_packet[predecessor_by_packet[packet.id]])
        if pattern is FusionPattern.MOE_GEMM_COMBINE and first:
            deps.extend(
                comp_by_block[(item.id, block)]
                for item in packet_assignments
                for block in compute_blocks
            )
        deps = tuple(dict.fromkeys(deps))
        common = dict(
            assignment_refs=tuple(item.id for item in packet_assignments),
            expert_index=packet_assignments[0].expert_index,
            tile_index=packet_pipeline,
            n_block_index=n_block,
            packet_ref=packet.id,
            stage=packet.stage,
            pivot_rank=packet.pivot_rank,
            pipeline_index=packet_pipeline,
            buffer_slot=packet_pipeline % physical_slots,
            buffer_family=("dispatch_operand" if pattern is FusionPattern.MOE_DISPATCH_GEMM else "combine_output"),
            work_role=f"{pattern.value}.transport",
        )
        send = MoeActionWitness.create(
            rank=packet.source_rank, kind=SwizzleActionKind.SEND, deps=deps,
            original_action_refs=tuple(original_send), route_ref=packet.route_ref,
            peer_rank=packet.destination_rank, logical_bytes=packet.logical_bytes,
            flops=0, **common,
        )
        recv = MoeActionWitness.create(
            rank=packet.destination_rank, kind=SwizzleActionKind.RECV, deps=deps,
            original_action_refs=tuple(original_recv), route_ref=packet.route_ref,
            peer_rank=packet.source_rank, logical_bytes=packet.logical_bytes,
            flops=0, **common,
        )
        wait = MoeActionWitness.create(
            rank=packet.destination_rank, kind=SwizzleActionKind.WAIT,
            deps=(send.id, recv.id), assignment_refs=common["assignment_refs"],
            expert_index=common["expert_index"], tile_index=common["tile_index"],
            n_block_index=n_block, packet_ref=packet.id, stage=packet.stage,
            pivot_rank=packet.pivot_rank, original_action_refs=tuple(original_wait),
            route_ref=None, peer_rank=None, logical_bytes=0, flops=0,
            pipeline_index=packet_pipeline, buffer_slot=packet_pipeline % physical_slots,
            buffer_family=("dispatch_operand" if pattern is FusionPattern.MOE_DISPATCH_GEMM else "combine_output"),
            work_role=common["work_role"],
        )
        generated.extend((send, recv, wait))
        wait_by_packet[packet.id] = wait.id

    if pattern is FusionPattern.MOE_DISPATCH_GEMM:
        for tile in tiles:
            emit_comp(tile)

    old_actions = tuple(generated)
    by_packet_kind = {
        (action.packet_ref, action.kind): action
        for action in old_actions
        if action.packet_ref is not None
    }
    comps_by_tile = defaultdict(list)
    for action in old_actions:
        if action.kind is SwizzleActionKind.COMP:
            comps_by_tile[action.tile_index].append(action)
    extra_deps = {action.id: set(action.deps) for action in old_actions}
    action_owners = build_moe_candidate_action_owner_map(
        problem, old_actions
    )
    tile_domains = defaultdict(lambda: defaultdict(list))
    for tile in tiles:
        comp = comps_by_tile[tile.tile_index][0]
        tile_domains[(
            action_owners[comp.id].runtime_core_id,
            comp.buffer_family,
            comp.buffer_slot,
        )][
            comp.pipeline_index
        ].append(tile)
    for occupants in tile_domains.values():
        ordered = sorted(occupants.items())
        for (_, previous_tiles), (_, current_tiles) in zip(
            ordered, ordered[1:]
        ):
            previous_comps = tuple(
                action
                for tile in previous_tiles
                for action in comps_by_tile[tile.tile_index]
            )
            current_comps = tuple(
                action
                for tile in current_tiles
                for action in comps_by_tile[tile.tile_index]
            )
            if pattern is FusionPattern.MOE_DISPATCH_GEMM:
                readers = tuple(
                    action.id
                    for action in old_actions
                    if action.kind is SwizzleActionKind.SWIGLU
                    and action.tile_index in {
                        tile.tile_index for tile in previous_tiles
                    }
                )
                targets = tuple(dict.fromkeys(
                    by_packet_kind[(packet_ref, SwizzleActionKind.RECV)]
                    for tile in current_tiles
                    for packet_ref in tile.required_packet_refs
                )) or current_comps
                for target in targets:
                    extra_deps[target.id].update(readers)
            else:
                previous_assignment_refs = {
                    ref for tile in previous_tiles for ref in tile.assignment_refs
                }
                old_readers = tuple(
                    action.id
                    for action in old_actions
                    if action.packet_ref is not None
                    and set(action.assignment_refs).intersection(
                        previous_assignment_refs
                    )
                    and action.kind is SwizzleActionKind.SEND
                )
                for target in current_comps:
                    extra_deps[target.id].update(old_readers)

    extra_deps = _endpoint_session_dependencies(
        problem, old_actions, extra_deps
    )

    # Rebuild stable ids in the strengthened topological order.  This permits
    # a future packet writer to depend on all old readers without relying on
    # source-list order, while rejecting a cycle fail closed.
    order = {action.id: index for index, action in enumerate(old_actions)}
    pending = set(order)
    rebuilt = {}
    generated = []
    while pending:
        ready = sorted(
            (
                ref for ref in pending
                if extra_deps[ref].issubset(rebuilt)
            ),
            key=order.__getitem__,
        )
        if not ready:
            witness = tuple(
                (
                    old_actions[order[ref]].kind.value,
                    old_actions[order[ref]].pipeline_index,
                    tuple(
                        old_actions[order[dep]].kind.value
                        for dep in extra_deps[ref]
                        if dep in pending
                    ),
                )
                for ref in sorted(pending, key=order.__getitem__)[:4]
            )
            raise SchemaError(
                f"buffer slot reuse creates a dependency cycle for {pattern.value} M={token_block_size}; witness={witness}",
                path="moe_candidate.slot_reuse",
            )
        ref = ready[0]
        action = next(item for item in old_actions if item.id == ref)
        semantic = {
            name: getattr(action, name)
            for name in action.__dataclass_fields__
            if name not in ("schema_version", "id")
        }
        semantic["deps"] = tuple(
            rebuilt[dep].id for dep in sorted(extra_deps[ref], key=order.__getitem__)
        )
        current = MoeActionWitness.create(**semantic)
        rebuilt[ref] = current
        generated.append(current)
        pending.remove(ref)
    programs = tuple(
        MoeRankProgram(
            rank=rank,
            actions=tuple(item for item in generated if item.rank == rank),
        )
        for rank in range(spec.mesh_rows * spec.mesh_columns)
    )
    return programs, tiles, token_block_size


def _finish_candidate(
    problem, spec, execution, algorithm, packets, final_packet_by_block,
    first_stage_packets, predecessor_by_packet, *, token_block_size,
    compute_output_block_count, transport_output_block_count,
    unroll_degree=1, double_buffer=False,
    calibration_profile: MoeSwizzleCalibrationProfile | None = None,
):
    programs, tiles, token_block_size = _build_program(
        problem, spec, execution, algorithm, packets, final_packet_by_block,
        first_stage_packets=first_stage_packets,
        predecessor_by_packet=predecessor_by_packet,
        physical_slots=(2 if double_buffer else 1),
        token_block_size=token_block_size,
        compute_output_block_count=compute_output_block_count,
    )
    actions = tuple(item for program in programs for item in program.actions)
    cost = build_moe_swizzle_cost(problem, algorithm, actions, packets, calibration_profile)
    result = MoeSwizzleCandidate.create(
        problem_ref=problem.id,
        pattern=problem.region.pattern,
        algorithm=algorithm,
        packetization=packets,
        tile_schedule=tiles,
        rank_programs=programs,
        expert_wave_count=1,
        token_block_size=token_block_size,
        output_column_block_size=(
            spec.intermediate_size
            if problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM
            else spec.hidden_size // compute_output_block_count
        ),
        compute_output_block_count=compute_output_block_count,
        transport_output_block_count=transport_output_block_count,
        unroll_degree=unroll_degree,
        double_buffer=double_buffer,
        compute_core_fraction=0.75,
        communication_core_fraction=0.25,
        original_action_refs=tuple(sorted(problem.region.member_refs)),
        cost=cost,
    )
    result.validate_against(problem)
    return result


def _candidate_token_block_sizes(spec):
    maximum = max(1, min(8, min(spec.trace.expert_histogram)))
    return tuple(size for size in (1, 2, 4, 8) if size <= maximum)


def build_direct_xy_moe_candidates(
    problem, spec, oracle, execution, *,
    calibration_profile: MoeSwizzleCalibrationProfile | None = None,
):
    _validate_sources(problem, spec, oracle, execution, "moe_direct_xy")
    algorithm = SwizzleAlgorithm.DIRECT_XY_PERSONALIZED_A2A
    if algorithm not in problem.allowed_algorithms:
        raise SchemaError("Direct XY is not admitted", path="moe_direct_xy.algorithm")
    pattern = problem.region.pattern
    transport_output_block_counts = (
        (1,)
        if pattern is FusionPattern.MOE_DISPATCH_GEMM
        else tuple(reversed(_COMBINE_TRANSPORT_BLOCK_COUNTS))
    )
    compute_output_block_count = 1
    result = []
    for transport_output_block_count in transport_output_block_counts:
        for token_block_size in reversed(_candidate_token_block_sizes(spec)):
            packets = []
            final_packet_by_block = {}
            for assignments in _remote_runs(problem, token_block_size, algorithm):
                source, destination, route_ref = _endpoints(assignments[0], pattern)
                n_blocks = (
                    (None,)
                    if (
                        pattern is FusionPattern.MOE_DISPATCH_GEMM
                        or transport_output_block_count == 1
                    )
                    else tuple(range(transport_output_block_count))
                )
                for n_block in n_blocks:
                    packet = _packet(
                        assignments, pattern, stage=0, source_rank=source,
                        destination_rank=destination, pivot_rank=None,
                        route_ref=route_ref, n_block_index=n_block,
                        slice_bytes=(
                            None
                            if n_block is None
                            else spec.hidden_size * 2
                            // transport_output_block_count
                        ),
                    )
                    packets.append(packet)
                    for assignment in assignments:
                        if (
                            n_block is None
                            and pattern is FusionPattern.MOE_GEMM_COMBINE
                        ):
                            final_packet_by_block[(assignment.id, 0)] = packet.id
                        else:
                            final_packet_by_block[(assignment.id, n_block)] = packet.id
            packetization = tuple(packets)
            result.append(_finish_candidate(
                problem, spec, execution, algorithm, packetization,
                final_packet_by_block, {item.id for item in packetization}, {},
                token_block_size=token_block_size,
                compute_output_block_count=compute_output_block_count,
                transport_output_block_count=transport_output_block_count,
                calibration_profile=calibration_profile,
            ))
    return tuple(result)


def build_direct_xy_moe_candidate(
    problem, spec, oracle, execution, *,
    calibration_profile: MoeSwizzleCalibrationProfile | None = None,
):
    return build_direct_xy_moe_candidates(
        problem, spec, oracle, execution,
        calibration_profile=calibration_profile,
    )[0]


__all__ = [
    "build_direct_xy_moe_candidate",
    "build_direct_xy_moe_candidates",
]
