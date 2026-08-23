"""Wang-style 1D Looped CollectiveEinsum candidate generation.

The paper describes logical rings.  On the backend-v1 physical Mesh we emit a
bidirectional open-line pipeline by default and add a ring variant only when
the frozen IR-1 embedding proves a real one-hop Hamiltonian cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import SchemaError
from ...schema.ir0 import FusionPattern
from ...schema.swizzle import (
    SwizzleActionKind,
    SwizzleActionWitness,
    SwizzleAlgorithm,
    SwizzleBufferRequirement,
    SwizzleFeasibilityCheck,
    SwizzleFeasibilityWitness,
    SwizzlePhase,
    SwizzleProblem,
    SwizzleRankProgramWitness,
    SwizzleSemanticWitness,
    SwizzleTensorAxisRole,
    SwizzleTopologyKind,
    SwizzleTopologyWitness,
)
from .chunking import WangChunkSpec, legal_wang_chunk_specs
from .enumerate import SwizzleCandidateDraft


@dataclass(slots=True)
class _Programs:
    actions: dict[int, list[SwizzleActionWitness]]

    @classmethod
    def create(cls, ranks: tuple[int, ...]) -> "_Programs":
        return cls({rank: [] for rank in ranks})

    def add(self, action: SwizzleActionWitness) -> SwizzleActionWitness:
        self.actions[action.rank].append(action)
        return action

    def freeze(self) -> tuple[SwizzleRankProgramWitness, ...]:
        return tuple(
            SwizzleRankProgramWitness(rank, tuple(self.actions[rank]))
            for rank in sorted(self.actions)
        )


@dataclass(slots=True)
class _DteSlotReuse:
    occupants: dict[tuple[int, int], tuple[int, list[str], tuple[str, ...]]]

    @classmethod
    def create(cls) -> "_DteSlotReuse":
        return cls({})

    def dependencies(self, rank: int, chunk: int) -> tuple[str, ...]:
        occupant = self.occupants.get((rank, chunk % 2))
        if occupant is None:
            return ()
        occupant_chunk, readers, reuse_deps = occupant
        return reuse_deps if occupant_chunk == chunk else tuple(readers)

    def record(self, rank: int, chunk: int, reader: str) -> None:
        key = (rank, chunk % 2)
        occupant = self.occupants.get(key)
        if occupant is None:
            self.occupants[key] = (chunk, [reader], ())
            return
        occupant_chunk, readers, reuse_deps = occupant
        if occupant_chunk == chunk:
            readers.append(reader)
            return
        self.occupants[key] = (chunk, [reader], tuple(readers))


def _action(
    programs: _Programs,
    *,
    rank: int,
    kind: SwizzleActionKind,
    deps: tuple[str, ...],
    chunk: int | None,
    phase: SwizzlePhase,
    peer: int | None = None,
    route_ref: str | None = None,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    logical_bytes: int = 0,
    flops: int = 0,
) -> SwizzleActionWitness:
    return programs.add(
        SwizzleActionWitness.create(
            rank=rank,
            kind=kind,
            deps=deps,
            chunk_index=chunk,
            phase=phase,
            peer_rank=peer,
            route_ref=route_ref,
            input_refs=inputs,
            output_refs=outputs,
            logical_bytes=logical_bytes,
            flops=flops,
        )
    )


def _route_index(problem: SwizzleProblem) -> dict[tuple[int, int], object]:
    routes: dict[tuple[int, int], object] = {}
    for route in problem.group.routes:
        pair = (route.source_rank, route.destination_rank)
        if pair in routes:
            raise SchemaError("duplicate route pair", path="swizzle_problem.group.routes")
        routes[pair] = route
    ranks = tuple(item.rank for item in problem.group.placements)
    expected = {
        (source, destination)
        for source in ranks
        for destination in ranks
        if source != destination
    }
    if set(routes) != expected:
        raise SchemaError(
            "Wang planning requires every ordered rank-pair route",
            path="swizzle_problem.group.routes",
        )
    return routes


def _rank_die_map(
    problem: SwizzleProblem,
    routes: dict[tuple[int, int], object],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for (source, destination), route in routes.items():
        for rank, die in ((source, route.die_path[0]), (destination, route.die_path[-1])):
            previous = result.get(rank)
            if previous is not None and previous != die:
                raise SchemaError(
                    "route endpoints disagree on rank placement",
                    path="swizzle_problem.group.routes",
                )
            result[rank] = die
    return result


def _physical_orders(
    problem: SwizzleProblem,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
    by_coord = {(item.x, item.y): item.rank for item in problem.group.placements}
    xs = sorted({coord[0] for coord in by_coord})
    ys = sorted({coord[1] for coord in by_coord})
    rows = tuple(
        tuple(by_coord[(x, y)] for x in xs if (x, y) in by_coord)
        for y in ys
    )
    columns = tuple(
        tuple(by_coord[(x, y)] for y in ys if (x, y) in by_coord)
        for x in xs
    )
    snake = tuple(
        rank
        for index, row in enumerate(rows)
        for rank in (row if index % 2 == 0 else tuple(reversed(row)))
    )
    return rows, columns, snake


def _real_cycle(
    ranks: tuple[int, ...],
    routes: dict[tuple[int, int], object],
) -> tuple[int, ...]:
    if len(ranks) < 3:
        return ()
    direct = {
        pair
        for pair, route in routes.items()
        if len(route.die_path) == 2
    }
    start = min(ranks)

    def visit(path: tuple[int, ...], remaining: frozenset[int]) -> tuple[int, ...]:
        if not remaining:
            return path if (path[-1], start) in direct else ()
        for candidate in sorted(remaining):
            if (path[-1], candidate) not in direct:
                continue
            found = visit(path + (candidate,), remaining - {candidate})
            if found:
                return found
        return ()

    return visit((start,), frozenset(ranks) - {start})


def _line_arcs(order: tuple[int, ...], owner: int) -> tuple[tuple[int, ...], ...]:
    index = order.index(owner)
    result: list[tuple[int, ...]] = []
    left = tuple(reversed(order[:index]))
    right = order[index + 1 :]
    if left:
        result.append(left)
    if right:
        result.append(right)
    return tuple(result)


def _ring_arcs(order: tuple[int, ...], owner: int) -> tuple[tuple[int, ...], ...]:
    size = len(order)
    origin = order.index(owner)
    clockwise_count = size // 2
    counter_count = size - 1 - clockwise_count
    clockwise = tuple(order[(origin + step) % size] for step in range(1, clockwise_count + 1))
    counter = tuple(order[(origin - step) % size] for step in range(1, counter_count + 1))
    return tuple(arc for arc in (clockwise, counter) if arc)


def _transport(
    programs: _Programs,
    routes: dict[tuple[int, int], object],
    *,
    source: int,
    destination: int,
    chunk: int,
    phase: SwizzlePhase,
    payload_ref: str,
    source_dep: str | None,
    output_ref: str,
    logical_bytes: int,
    dte_slot_reuse: _DteSlotReuse | None = None,
    destination_slot_dependency: str | None = None,
) -> tuple[SwizzleActionWitness, str]:
    route = routes[(source, destination)]
    deps = () if source_dep is None else (source_dep,)
    if dte_slot_reuse is not None:
        deps = tuple(
            dict.fromkeys(deps + dte_slot_reuse.dependencies(source, chunk))
        )
    send = _action(
        programs,
        rank=source,
        kind=SwizzleActionKind.SEND,
        deps=deps,
        chunk=chunk,
        phase=phase,
        peer=destination,
        route_ref=route.id,
        inputs=(payload_ref,),
        logical_bytes=logical_bytes,
    )
    if dte_slot_reuse is not None:
        dte_slot_reuse.record(source, chunk, send.id)
    recv_deps = (send.id,)
    if destination_slot_dependency is not None:
        recv_deps += (destination_slot_dependency,)
    if dte_slot_reuse is not None:
        recv_deps += dte_slot_reuse.dependencies(destination, chunk)
    recv_deps = tuple(dict.fromkeys(recv_deps))
    recv = _action(
        programs,
        rank=destination,
        kind=SwizzleActionKind.RECV,
        deps=recv_deps,
        chunk=chunk,
        phase=phase,
        peer=source,
        route_ref=route.id,
        outputs=(output_ref,),
        logical_bytes=logical_bytes,
    )
    wait = _action(
        programs,
        rank=destination,
        kind=SwizzleActionKind.WAIT,
        deps=(recv.id,),
        chunk=chunk,
        phase=phase,
        inputs=(output_ref,),
        outputs=(output_ref,),
    )
    return wait, output_ref


def _tile_shape(
    problem: SwizzleProblem,
    witness: SwizzleSemanticWitness,
    chunk_count: int,
) -> tuple[int, int, int]:
    m, n, k = problem.gemm.m, problem.gemm.n, problem.gemm.k
    role = witness.split_axis.role
    if role is SwizzleTensorAxisRole.FREE_LHS:
        m //= chunk_count
    elif role is SwizzleTensorAxisRole.FREE_RHS:
        n //= chunk_count
    elif role is SwizzleTensorAxisRole.CONTRACT:
        k //= chunk_count
    elif role is SwizzleTensorAxisRole.BATCH:
        # V1 descriptors carry no batched GEMM production case.  Preserve the
        # tile and let feasibility reject an indivisible batch witness.
        pass
    return (m, n, k)


def _add_compute(
    programs: _Programs,
    *,
    rank: int,
    chunk: int,
    phase: SwizzlePhase,
    input_refs: tuple[str, ...],
    dependency: str | None,
    lane_tail: dict[tuple[int, int], str],
    unroll_degree: int,
    output_ref: str,
    flops_per_rank_chunk: int,
    slot_reuse_deps: tuple[str, ...] = (),
) -> SwizzleActionWitness:
    deps = (() if dependency is None else (dependency,)) + slot_reuse_deps
    deps = tuple(dict.fromkeys(deps))
    lane = chunk % unroll_degree
    previous = lane_tail.get((rank, lane))
    if previous is not None and previous not in deps:
        deps += (previous,)
    action = _action(
        programs,
        rank=rank,
        kind=SwizzleActionKind.COMP,
        deps=deps,
        chunk=chunk,
        phase=phase,
        inputs=input_refs,
        outputs=(output_ref,),
        flops=flops_per_rank_chunk,
    )
    lane_tail[(rank, lane)] = action.id
    return action


def _build_ag(
    problem: SwizzleProblem,
    semantic_witness: SwizzleSemanticWitness,
    chunk_spec: WangChunkSpec,
    order: tuple[int, ...],
    arcs_for_owner,
    routes: dict[tuple[int, int], object],
    *,
    unroll_degree: int,
) -> tuple[_Programs, dict[tuple[int, int], str]]:
    ranks = problem.collective.participant_ranks
    programs = _Programs.create(ranks)
    lane_tail: dict[tuple[int, int], str] = {}
    completion: dict[tuple[int, int], str] = {}
    partials: dict[tuple[int, int], str] = {}
    dte_slot_reuse = _DteSlotReuse.create() if unroll_degree == 2 else None
    local_operand = (
        problem.gemm.rhs.value_ref
        if problem.gemm.lhs.value_ref == problem.collective.output.value_ref
        else problem.gemm.lhs.value_ref
    )
    for chunk in range(chunk_spec.chunk_count):
        owner = ranks[chunk % len(ranks)]
        slot_reuse_deps = {
            rank: (
                ()
                if dte_slot_reuse is None
                else dte_slot_reuse.dependencies(rank, chunk)
            )
            for rank in ranks
        }
        shard_ref = f"{problem.collective.input.value_ref}::chunk{chunk}"
        local_output = f"{problem.gemm.output.value_ref}::rank{owner}::chunk{chunk}"
        local = _add_compute(
            programs,
            rank=owner,
            chunk=chunk,
            phase=SwizzlePhase.STEADY,
            input_refs=(shard_ref, local_operand),
            dependency=None,
            lane_tail=lane_tail,
            unroll_degree=unroll_degree,
            output_ref=local_output,
            flops_per_rank_chunk=chunk_spec.flops_per_rank_chunk,
            slot_reuse_deps=slot_reuse_deps[owner],
        )
        completion[(owner, chunk)] = local.id
        partials[(owner, chunk)] = local_output
        for arc in arcs_for_owner(order, owner):
            source = owner
            available_ref = shard_ref
            available_dep: str | None = None
            for hop_index, destination in enumerate(arc):
                received = f"{shard_ref}::at_rank{destination}"
                wait, available_ref = _transport(
                    programs,
                    routes,
                    source=source,
                    destination=destination,
                    chunk=chunk,
                    phase=SwizzlePhase.PROLOGUE if hop_index == 0 else SwizzlePhase.STEADY,
                    payload_ref=available_ref,
                    source_dep=available_dep,
                    output_ref=received,
                    logical_bytes=chunk_spec.logical_bytes_per_chunk,
                    dte_slot_reuse=dte_slot_reuse,
                    destination_slot_dependency=(
                        None
                        if dte_slot_reuse is None
                        else lane_tail.get((destination, chunk % unroll_degree))
                    ),
                )
                partial = f"{problem.gemm.output.value_ref}::rank{destination}::chunk{chunk}"
                comp = _add_compute(
                    programs,
                    rank=destination,
                    chunk=chunk,
                    phase=SwizzlePhase.STEADY,
                    input_refs=(received, local_operand),
                    dependency=wait.id,
                    lane_tail=lane_tail,
                    unroll_degree=unroll_degree,
                    output_ref=partial,
                    flops_per_rank_chunk=chunk_spec.flops_per_rank_chunk,
                    slot_reuse_deps=slot_reuse_deps[destination],
                )
                completion[(destination, chunk)] = comp.id
                partials[(destination, chunk)] = partial
                source = destination
                available_dep = wait.id
    for rank in ranks:
        deps = tuple(
            completion[(rank, chunk)] for chunk in range(chunk_spec.chunk_count)
        )
        if semantic_witness.split_axis.role is SwizzleTensorAxisRole.CONTRACT:
            accumulator_ref = partials[(rank, 0)]
            accumulator_dep = completion[(rank, 0)]
            for chunk in range(1, chunk_spec.chunk_count):
                output_ref = (
                    f"{problem.gemm.output.value_ref}::rank{rank}::boundary"
                    if chunk == chunk_spec.chunk_count - 1
                    else f"{problem.gemm.output.value_ref}::rank{rank}::acc{chunk}"
                )
                reduction = _action(
                    programs,
                    rank=rank,
                    kind=SwizzleActionKind.REDUCE,
                    deps=(accumulator_dep, completion[(rank, chunk)]),
                    chunk=chunk,
                    phase=SwizzlePhase.STEADY,
                    inputs=(accumulator_ref, partials[(rank, chunk)]),
                    outputs=(output_ref,),
                )
                accumulator_ref = output_ref
                accumulator_dep = reduction.id
            deps = (accumulator_dep,)
        barrier = _action(
            programs,
            rank=rank,
            kind=SwizzleActionKind.BARRIER,
            deps=deps,
            chunk=None,
            phase=SwizzlePhase.EPILOGUE,
        )
        completion[(rank, -1)] = barrier.id
    return programs, completion


def _reduction_arcs(
    order: tuple[int, ...],
    owner: int,
    outward_arcs,
) -> tuple[tuple[int, ...], ...]:
    # Outward arcs are owner-near to far.  Reverse each to obtain independent
    # far-to-owner loop-carried accumulation chains.
    return tuple(tuple(reversed(arc)) for arc in outward_arcs(order, owner))


def _build_reduce(
    problem: SwizzleProblem,
    chunk_spec: WangChunkSpec,
    order: tuple[int, ...],
    outward_arcs,
    routes: dict[tuple[int, int], object],
    *,
    unroll_degree: int,
) -> tuple[_Programs, dict[tuple[int, int], tuple[str, str]]]:
    ranks = problem.collective.participant_ranks
    programs = _Programs.create(ranks)
    lane_tail: dict[tuple[int, int], str] = {}
    dte_slot_reuse = _DteSlotReuse.create() if unroll_degree == 2 else None
    comp: dict[tuple[int, int], SwizzleActionWitness] = {}
    finals: dict[tuple[int, int], tuple[str, str]] = {}
    for chunk in range(chunk_spec.chunk_count):
        owner = ranks[chunk % len(ranks)]
        previous_comp_tail = {
            rank: (
                None
                if dte_slot_reuse is None
                else lane_tail.get((rank, chunk % unroll_degree))
            )
            for rank in ranks
        }
        slot_reuse_deps = {
            rank: (
                ()
                if dte_slot_reuse is None
                else dte_slot_reuse.dependencies(rank, chunk)
            )
            for rank in ranks
        }
        for rank in ranks:
            contribution = f"{problem.gemm.output.value_ref}::rank{rank}::chunk{chunk}::contribution"
            comp[(rank, chunk)] = _add_compute(
                programs,
                rank=rank,
                chunk=chunk,
                phase=SwizzlePhase.STEADY,
                input_refs=(problem.gemm.lhs.value_ref, problem.gemm.rhs.value_ref),
                dependency=None,
                lane_tail=lane_tail,
                unroll_degree=unroll_degree,
                output_ref=contribution,
                flops_per_rank_chunk=chunk_spec.flops_per_rank_chunk,
                slot_reuse_deps=slot_reuse_deps[rank],
            )
        owner_waits: list[SwizzleActionWitness] = []
        owner_inputs = [comp[(owner, chunk)].output_refs[0]]
        for chain_index, far_to_near in enumerate(_reduction_arcs(order, owner, outward_arcs)):
            source = far_to_near[0]
            accumulator = comp[(source, chunk)].output_refs[0]
            accumulator_dep = comp[(source, chunk)].id
            path = far_to_near[1:] + (owner,)
            for destination in path:
                received = f"wang::chunk{chunk}::chain{chain_index}::at_rank{destination}"
                wait, received = _transport(
                    programs,
                    routes,
                    source=source,
                    destination=destination,
                    chunk=chunk,
                    phase=SwizzlePhase.STEADY,
                    payload_ref=accumulator,
                    source_dep=accumulator_dep,
                    output_ref=received,
                    logical_bytes=chunk_spec.logical_bytes_per_chunk,
                    dte_slot_reuse=dte_slot_reuse,
                    destination_slot_dependency=previous_comp_tail[destination],
                )
                if destination == owner:
                    owner_waits.append(wait)
                    owner_inputs.append(received)
                    break
                reduced = f"wang::chunk{chunk}::chain{chain_index}::acc_rank{destination}"
                reduction = _action(
                    programs,
                    rank=destination,
                    kind=SwizzleActionKind.REDUCE,
                    deps=(wait.id, comp[(destination, chunk)].id),
                    chunk=chunk,
                    phase=SwizzlePhase.STEADY,
                    inputs=(comp[(destination, chunk)].output_refs[0], received),
                    outputs=(reduced,),
                )
                source = destination
                accumulator = reduced
                accumulator_dep = reduction.id
        final_ref = (
            f"{problem.collective.output.value_ref}::owner{owner}::chunk{chunk}"
            if problem.pattern is FusionPattern.GEMM_RS
            else f"{problem.collective.input.value_ref}::reduced_chunk{chunk}"
        )
        accumulator_ref = owner_inputs[0]
        accumulator_dep = comp[(owner, chunk)].id
        for input_index, (wait, received) in enumerate(
            zip(owner_waits, owner_inputs[1:], strict=True)
        ):
            output_ref = (
                final_ref
                if input_index == len(owner_waits) - 1
                else (
                    f"wang::chunk{chunk}::owner{owner}::"
                    f"acc{input_index + 1}"
                )
            )
            reduction = _action(
                programs,
                rank=owner,
                kind=SwizzleActionKind.REDUCE,
                deps=(accumulator_dep, wait.id),
                chunk=chunk,
                phase=SwizzlePhase.STEADY,
                inputs=(accumulator_ref, received),
                outputs=(output_ref,),
            )
            accumulator_ref = output_ref
            accumulator_dep = reduction.id
        finals[(owner, chunk)] = (accumulator_dep, accumulator_ref)
    return programs, finals


def _append_rs_epilogue(
    programs: _Programs,
    ranks: tuple[int, ...],
    finals: dict[tuple[int, int], tuple[str, str]],
    *,
    chunk_count: int,
) -> None:
    copies: dict[int, list[str]] = {rank: [] for rank in ranks}
    for chunk in range(chunk_count):
        owner = ranks[chunk % len(ranks)]
        final_id, final_ref = finals[(owner, chunk)]
        copy = _action(
            programs,
            rank=owner,
            kind=SwizzleActionKind.LOCAL_COPY,
            deps=(final_id,),
            chunk=chunk,
            phase=SwizzlePhase.EPILOGUE,
            inputs=(final_ref,),
            outputs=(f"{final_ref}::boundary",),
        )
        copies[owner].append(copy.id)
    if chunk_count > len(ranks):
        for rank in ranks:
            _action(
                programs,
                rank=rank,
                kind=SwizzleActionKind.BARRIER,
                deps=tuple(copies[rank]),
                chunk=None,
                phase=SwizzlePhase.EPILOGUE,
            )


def _append_ar_replication(
    problem: SwizzleProblem,
    programs: _Programs,
    order: tuple[int, ...],
    outward_arcs,
    routes: dict[tuple[int, int], object],
    finals: dict[tuple[int, int], tuple[str, str]],
    *,
    chunk_bytes: int,
    chunk_count: int,
    unroll_degree: int,
) -> None:
    ranks = problem.collective.participant_ranks
    dte_slot_reuse = _DteSlotReuse.create() if unroll_degree == 2 else None
    availability: dict[tuple[int, int], str] = {}
    for chunk in range(chunk_count):
        owner = ranks[chunk % len(ranks)]
        final_id, final_ref = finals[(owner, chunk)]
        availability[(owner, chunk)] = final_id
        for arc in outward_arcs(order, owner):
            source = owner
            source_dep = final_id
            payload = final_ref
            for destination in arc:
                received = f"{problem.collective.output.value_ref}::rank{destination}::chunk{chunk}"
                wait, payload = _transport(
                    programs,
                    routes,
                    source=source,
                    destination=destination,
                    chunk=chunk,
                    phase=SwizzlePhase.EPILOGUE,
                    payload_ref=payload,
                    source_dep=source_dep,
                    output_ref=received,
                    logical_bytes=chunk_bytes,
                    dte_slot_reuse=dte_slot_reuse,
                )
                availability[(destination, chunk)] = wait.id
                source = destination
                source_dep = wait.id
    for rank in ranks:
        _action(
            programs,
            rank=rank,
            kind=SwizzleActionKind.BARRIER,
            deps=tuple(availability[(rank, chunk)] for chunk in range(chunk_count)),
            chunk=None,
            phase=SwizzlePhase.EPILOGUE,
        )


def _buffers(
    problem: SwizzleProblem,
    programs: tuple[SwizzleRankProgramWitness, ...],
    *,
    chunk_bytes: int,
    unroll_degree: int,
) -> tuple[SwizzleBufferRequirement, ...]:
    return tuple(
        SwizzleBufferRequirement(
            rank=program.rank,
            buffer_ref=f"wang::rank{program.rank}::loop_buffer",
            size_bytes=chunk_bytes,
            double_buffered=unroll_degree == 2,
            lifetime_action_refs=tuple(action.id for action in program.actions),
        )
        for program in programs
    )


def _checks(
    problem: SwizzleProblem,
    programs: tuple[SwizzleRankProgramWitness, ...],
    buffers: tuple[SwizzleBufferRequirement, ...],
    routes: dict[tuple[int, int], object],
    *,
    chunk_spec: WangChunkSpec,
    unroll_degree: int,
) -> SwizzleFeasibilityWitness:
    ranks = problem.collective.participant_ranks
    rank_dies = _rank_die_map(problem, routes)
    group_dies = set(rank_dies.values())
    no_cross_group = all(set(route.die_path).issubset(group_dies) for route in routes.values())
    action_count = sum(len(program.actions) for program in programs)
    buffer_count = len(buffers)
    per_rank_sram = max(
        sum(item.size_bytes * (2 if item.double_buffered else 1) for item in buffers if item.rank == rank)
        for rank in ranks
    )
    # Keep names lexicographically ordered; the schema freezes this order.
    return SwizzleFeasibilityWitness(
        checks=(
            SwizzleFeasibilityCheck("action_budget", action_count <= problem.constraints.max_actions, f"{action_count}/{problem.constraints.max_actions}"),
            SwizzleFeasibilityCheck("buffer_budget", buffer_count <= problem.constraints.max_buffers, f"{buffer_count}/{problem.constraints.max_buffers}"),
            SwizzleFeasibilityCheck("chunk_count", chunk_spec.chunk_count <= problem.constraints.max_chunk_count, f"{chunk_spec.chunk_count}/{problem.constraints.max_chunk_count}"),
            SwizzleFeasibilityCheck("cross_group_routes", no_cross_group, "all route transit Dies remain in group" if no_cross_group else "a route transits a Die outside the group"),
            SwizzleFeasibilityCheck("double_buffer", unroll_degree == 1 or problem.hardware_profile.double_buffer_supported, "supported" if unroll_degree == 1 or problem.hardware_profile.double_buffer_supported else "hardware profile forbids double buffering"),
            SwizzleFeasibilityCheck("participant_count", len(ranks) >= 2, f"participants={len(ranks)}"),
            SwizzleFeasibilityCheck("route_closure", len(routes) == len(ranks) * (len(ranks) - 1), "all ordered pairs present"),
            SwizzleFeasibilityCheck("sram_budget", per_rank_sram <= problem.hardware_profile.sram_budget_bytes, f"{per_rank_sram}/{problem.hardware_profile.sram_budget_bytes}"),
            SwizzleFeasibilityCheck("transfer_size", chunk_spec.logical_bytes_per_chunk >= problem.hardware_profile.min_transfer_bytes, f"{chunk_spec.logical_bytes_per_chunk}/{problem.hardware_profile.min_transfer_bytes}"),
        )
    )


def _topology_witness(
    problem: SwizzleProblem,
    programs: tuple[SwizzleRankProgramWitness, ...],
    *,
    kind: SwizzleTopologyKind,
    order: tuple[int, ...],
    rows: tuple[tuple[int, ...], ...],
    columns: tuple[tuple[int, ...], ...],
    has_cycle: bool,
) -> SwizzleTopologyWitness:
    route_refs = tuple(
        sorted(
            {
                action.route_ref
                for program in programs
                for action in program.actions
                if action.route_ref is not None
            }
        )
    )
    width, height = problem.group.logical_shape
    coordinates = {(item.x, item.y) for item in problem.group.placements}
    xs = sorted({x for x, _ in coordinates})
    ys = sorted({y for _, y in coordinates})
    rectangle = len(coordinates) == width * height == len(xs) * len(ys)
    return SwizzleTopologyWitness(
        kind=kind,
        rank_order=order,
        row_orders=rows,
        column_orders=columns,
        route_refs=route_refs,
        is_complete_rectangle=rectangle,
        has_hamiltonian_cycle=has_cycle,
    )


def generate_wang_1d_drafts(
    problem: SwizzleProblem,
    semantic_witness: SwizzleSemanticWitness,
) -> tuple[SwizzleCandidateDraft, ...]:
    """Generate deterministic open-line and proven-ring Wang candidates."""

    problem.validate("swizzle_problem")
    semantic_witness.validate("semantic_witness")
    if semantic_witness.pattern is not problem.pattern:
        raise SchemaError("semantic witness pattern mismatch", path="semantic_witness.pattern")
    if SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL not in problem.constraints.allowed_algorithms:
        return ()
    ranks = problem.collective.participant_ranks
    if len(ranks) < 2:
        return ()
    chunk_specs = legal_wang_chunk_specs(problem, semantic_witness)
    if not chunk_specs:
        return ()
    routes = _route_index(problem)
    rows, columns, line_order = _physical_orders(problem)
    cycle_order = _real_cycle(ranks, routes)
    variants = [
        (SwizzleTopologyKind.BIDIRECTIONAL_LINE, line_order, _line_arcs),
    ]
    if cycle_order:
        variants.append((SwizzleTopologyKind.HAMILTONIAN_RING, cycle_order, _ring_arcs))
    unroll_degrees = (1, 2) if problem.constraints.allow_unroll_two else (1,)
    drafts: list[SwizzleCandidateDraft] = []
    candidate_cap = min(32, problem.constraints.max_candidates)
    for chunk_spec in chunk_specs:
        for kind, order, arc_builder in variants:
            for unroll_degree in unroll_degrees:
                if unroll_degree == 2 and (
                    not problem.hardware_profile.double_buffer_supported
                    or problem.hardware_profile.max_inflight_dte < 2
                ):
                    continue
                if problem.pattern is FusionPattern.AG_GEMM:
                    mutable_programs, _ = _build_ag(
                        problem,
                        semantic_witness,
                        chunk_spec,
                        order,
                        arc_builder,
                        routes,
                        unroll_degree=unroll_degree,
                    )
                else:
                    mutable_programs, finals = _build_reduce(
                        problem,
                        chunk_spec,
                        order,
                        arc_builder,
                        routes,
                        unroll_degree=unroll_degree,
                    )
                    if problem.pattern is FusionPattern.GEMM_RS:
                        _append_rs_epilogue(
                            mutable_programs,
                            ranks,
                            finals,
                            chunk_count=chunk_spec.chunk_count,
                        )
                    else:
                        _append_ar_replication(
                            problem,
                            mutable_programs,
                            order,
                            arc_builder,
                            routes,
                            finals,
                            chunk_bytes=chunk_spec.logical_bytes_per_chunk,
                            chunk_count=chunk_spec.chunk_count,
                            unroll_degree=unroll_degree,
                        )
                programs = mutable_programs.freeze()
                buffers = _buffers(
                    problem,
                    programs,
                    chunk_bytes=chunk_spec.logical_bytes_per_chunk,
                    unroll_degree=unroll_degree,
                )
                feasibility = _checks(
                    problem,
                    programs,
                    buffers,
                    routes,
                    chunk_spec=chunk_spec,
                    unroll_degree=unroll_degree,
                )
                if not feasibility.feasible:
                    continue
                topology = _topology_witness(
                    problem,
                    programs,
                    kind=kind,
                    order=order,
                    rows=rows,
                    columns=columns,
                    has_cycle=bool(cycle_order),
                )
                draft = SwizzleCandidateDraft(
                    problem_ref=problem.id,
                    pattern=problem.pattern,
                    algorithm=SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL,
                    split_axis=semantic_witness.split_axis,
                    chunk_count=chunk_spec.chunk_count,
                    unroll_degree=unroll_degree,
                    rank_programs=programs,
                    buffer_requirements=buffers,
                    topology_witness=topology,
                    semantic_witness=semantic_witness,
                    feasibility_witness=feasibility,
                    tile_shape=_tile_shape(
                        problem, semantic_witness, chunk_spec.chunk_count
                    ),
                )
                draft.validate_against(problem)
                drafts.append(draft)
                if len(drafts) >= candidate_cap:
                    return tuple(drafts)
    return tuple(drafts)


__all__ = ["generate_wang_1d_drafts"]
