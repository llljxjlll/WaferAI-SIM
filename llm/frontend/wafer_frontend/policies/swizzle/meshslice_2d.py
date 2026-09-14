"""MeshSlice 2D output-stationary candidate generation.

This is a fail-closed implementation of the blocked slicing described by Nam
et al.  It never infers 2D tensor partitioning from a rectangular placement:
the original tensor views must prove the OS sharding, unless the caller has a
separate boundary-reshard proof and explicitly enables that path.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
import math

from ...errors import SchemaError
from ...schema.common import DType
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
    SwizzleTopologyKind,
    SwizzleTopologyWitness,
)
from .enumerate import SwizzleCandidateDraft, legal_divisors


class MeshSliceExecutionMode(str, Enum):
    FULL_2D = "full_2d"
    ROW_ONLY = "row_only"
    COLUMN_ONLY = "column_only"
    LOCAL = "local"


def meshslice_execution_mode(
    rows: int,
    columns: int,
) -> MeshSliceExecutionMode:
    if rows <= 0 or columns <= 0:
        raise SchemaError(
            "MeshSlice dimensions must be positive",
            path="swizzle_problem.group.logical_shape",
        )
    if rows > 1 and columns > 1:
        return MeshSliceExecutionMode.FULL_2D
    if columns > 1:
        return MeshSliceExecutionMode.ROW_ONLY
    if rows > 1:
        return MeshSliceExecutionMode.COLUMN_ONLY
    return MeshSliceExecutionMode.LOCAL


def _dtype_bytes(dtype: DType) -> int:
    if dtype is DType.FP16:
        return 2
    if dtype in (DType.FP32, DType.INT32):
        return 4
    raise SchemaError("unsupported MeshSlice dtype", path="swizzle_problem.gemm.dtype")


def _physical_lines(
    problem: SwizzleProblem,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], bool]:
    by_coord = {(item.x, item.y): item.rank for item in problem.group.placements}
    xs = tuple(sorted({item.x for item in problem.group.placements}))
    ys = tuple(sorted({item.y for item in problem.group.placements}))
    rectangle = len(by_coord) == len(xs) * len(ys) and all(
        (x, y) in by_coord for x in xs for y in ys
    )
    rows = tuple(tuple(by_coord[(x, y)] for x in xs if (x, y) in by_coord) for y in ys)
    columns = tuple(tuple(by_coord[(x, y)] for y in ys if (x, y) in by_coord) for x in xs)
    return rows, columns, rectangle


def _exact_os_2d_sharding(problem: SwizzleProblem) -> bool:
    """Prove A[M,K], B[K,N], C[M,N] use the same two mesh axes."""

    lhs = problem.gemm.lhs.sharding_dim_map
    rhs = problem.gemm.rhs.sharding_dim_map
    output = problem.gemm.output.sharding_dim_map
    if min(len(lhs), len(rhs), len(output)) < 2:
        return False
    row_axis, column_axis = output[-2], output[-1]
    if row_axis is None or column_axis is None or row_axis is column_axis:
        return False
    return (
        lhs[-2:] == (row_axis, column_axis)
        and rhs[-2:] == (row_axis, column_axis)
        and not problem.gemm.lhs.partial_mesh_axes
        and not problem.gemm.rhs.partial_mesh_axes
        and not problem.gemm.output.partial_mesh_axes
    )


def _route_by_pair(problem: SwizzleProblem) -> dict[tuple[int, int], object]:
    return {(route.source_rank, route.destination_rank): route for route in problem.group.routes}


def _line_peers(
    rank: int,
    rows: tuple[tuple[int, ...], ...],
    columns: tuple[tuple[int, ...], ...],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    row = next(item for item in rows if rank in item)
    column = next(item for item in columns if rank in item)
    return (("lhs", row), ("rhs", column))


def _blocked_slice_counts(problem: SwizzleProblem, rows: int, columns: int) -> tuple[int, ...]:
    block = problem.hardware_profile.efficient_tile_floor[2]
    if problem.gemm.k % rows or problem.gemm.k % columns:
        return ()
    lhs_local_k = problem.gemm.k // columns
    rhs_local_k = problem.gemm.k // rows
    if lhs_local_k % block or rhs_local_k % block:
        return ()
    common_blocks = math.gcd(lhs_local_k // block, rhs_local_k // block)
    return legal_divisors(
        common_blocks,
        min(problem.constraints.max_chunk_count, common_blocks),
    )


def _candidate_scale(
    problem: SwizzleProblem,
    *,
    rows: int,
    columns: int,
    slice_count: int,
) -> tuple[int, int]:
    """Return exact action/buffer counts before materializing the DAG."""

    return meshslice_candidate_scale(
        problem.pattern,
        rows=rows,
        columns=columns,
        slice_count=slice_count,
    )


def meshslice_candidate_scale(
    pattern: FusionPattern,
    *,
    rows: int,
    columns: int,
    slice_count: int,
) -> tuple[int, int]:
    """Pure production estimator used by the complete shape sweep."""

    if type(pattern) is not FusionPattern:
        raise SchemaError("requires a fusion pattern", path="pattern")
    meshslice_execution_mode(rows, columns)
    if slice_count <= 0:
        raise SchemaError("slice count must be positive", path="slice_count")

    ranks = rows * columns
    peers = rows + columns - 2
    actions = slice_count * ranks * (3 * peers + 1)
    buffers_per_rank = 3
    if pattern in (FusionPattern.GEMM_RS, FusionPattern.GEMM_AR):
        actions += ranks
        buffers_per_rank += 1
    if pattern is FusionPattern.GEMM_AR:
        actions += 3 * ranks * (ranks - 1) + ranks
        buffers_per_rank += 1
    return actions, ranks * buffers_per_rank


def _make_action(**semantic: object) -> SwizzleActionWitness:
    return SwizzleActionWitness.create(**semantic)


def _round_robin_pair_rounds(
    line: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return canonical circle-method matching rounds for one line."""

    players: list[int | None] = list(line)
    if len(players) % 2:
        players.append(None)
    rounds = []
    for _round in range(len(players) - 1):
        pairs = []
        for index in range(len(players) // 2):
            left = players[index]
            right = players[-1 - index]
            if left is not None and right is not None:
                pairs.append((min(left, right), max(left, right)))
        rounds.append(tuple(sorted(pairs)))
        players = [players[0], players[-1], *players[1:-1]]
    if sum(len(item) for item in rounds) != len(line) * (len(line) - 1) // 2:
        raise SchemaError(
            "round-robin pair schedule is not exact",
            path="swizzle_candidate.rank_programs",
        )
    return tuple(rounds)


def _session_safe_gather_order(
    actions: dict[int, list[SwizzleActionWitness]],
    *,
    rows: tuple[tuple[int, ...], ...],
    columns: tuple[tuple[int, ...], ...],
    slice_count: int,
) -> None:
    """Order non-frozen gather edges in matching waves.

    SEND and RECV share one endpoint session pool.  A matching wave keeps at
    most one bidirectional peer pair live per rank, well below the platform
    capacity, while preserving every action/dependency witness and ID.
    """

    if (len(rows), len(columns)) in ((1, 1), (2, 2)):
        return
    by_id = {
        action.id: action
        for rank_actions in actions.values()
        for action in rank_actions
    }

    def operand(action: SwizzleActionWitness) -> str | None:
        refs = (
            action.input_refs
            if action.kind is SwizzleActionKind.SEND
            else action.output_refs
            if action.kind is SwizzleActionKind.RECV
            else ()
        )
        if len(refs) != 1:
            return None
        role = refs[0].rsplit(".", 1)[-1]
        return role if role in ("lhs", "rhs") else None

    sends = {}
    receives = {}
    waits = {}
    for rank, rank_actions in actions.items():
        for action in rank_actions:
            role = operand(action)
            if role is not None and action.peer_rank is not None:
                key = (rank, action.chunk_index, role, action.peer_rank)
                if action.kind is SwizzleActionKind.SEND:
                    sends[key] = action
                else:
                    receives[key] = action
            elif action.kind is SwizzleActionKind.WAIT and len(action.deps) == 1:
                receive = by_id.get(action.deps[0])
                if receive is not None and operand(receive) is not None:
                    waits[receive.id] = action

    edges = tuple(sorted({
        (chunk, role, min(rank, peer), max(rank, peer))
        for rank, chunk, role, peer in sends
    }))
    if not edges:
        return
    wave_by_edge = {}
    for chunk in range(slice_count):
        wave_base = 0
        for role, lines in (("lhs", rows), ("rhs", columns)):
            role_round_count = 0
            for line in lines:
                rounds = _round_robin_pair_rounds(line)
                role_round_count = max(role_round_count, len(rounds))
                for round_index, pairs in enumerate(rounds):
                    for left, right in pairs:
                        wave_by_edge[(chunk, role, left, right)] = (
                            wave_base + round_index
                        )
            wave_base += role_round_count
    if set(wave_by_edge) != set(edges):
        raise SchemaError(
            "round-robin waves do not exactly cover gather edges",
            path="swizzle_candidate.rank_programs",
        )

    for rank, original in actions.items():
        ordered = []
        selected = set()
        for chunk in range(slice_count):
            chunk_edges = tuple(
                edge for edge in edges
                if edge[0] == chunk and rank in edge[2:]
            )
            for edge in sorted(
                chunk_edges, key=lambda item: (wave_by_edge[item], item)
            ):
                _chunk, role, left, right = edge
                peer = right if rank == left else left
                send = sends[(rank, chunk, role, peer)]
                receive = receives[(rank, chunk, role, peer)]
                wait = waits[receive.id]
                # Complementary endpoint order avoids a mesh-wide send-first
                # admission cycle: the lower rank injects while the higher
                # rank posts and drains its receive before injecting back.
                ordered.extend(
                    (send, receive, wait)
                    if rank == left else (receive, wait, send)
                )
                selected.update((send.id, receive.id, wait.id))
            computes = tuple(
                action for action in original
                if action.kind is SwizzleActionKind.COMP
                and action.chunk_index == chunk
            )
            ordered.extend(computes)
            selected.update(action.id for action in computes)
        ordered.extend(action for action in original if action.id not in selected)
        if len(ordered) != len(original) or len({item.id for item in ordered}) != len(original):
            raise SchemaError(
                "MeshSlice session-safe order lost an action",
                path="swizzle_candidate.rank_programs",
            )
        actions[rank] = ordered


def _programs_for_slice_count(
    problem: SwizzleProblem,
    rows: tuple[tuple[int, ...], ...],
    columns: tuple[tuple[int, ...], ...],
    slice_count: int,
) -> tuple[
    tuple[SwizzleRankProgramWitness, ...],
    tuple[SwizzleBufferRequirement, ...],
    tuple[str, ...],
]:
    ranks = tuple(range(len(problem.group.placements)))
    route_by_pair = _route_by_pair(problem)
    actions: dict[int, list[SwizzleActionWitness]] = {rank: [] for rank in ranks}
    last_comp: dict[int, str] = {}
    comp_by_slice: dict[tuple[int, int], str] = {}
    used_routes: set[str] = set()
    batch = math.prod(problem.gemm.batch_shape)
    row_count, column_count = len(rows), len(columns)
    dtype_bytes = _dtype_bytes(problem.gemm.dtype)
    accum_bytes = _dtype_bytes(problem.gemm.accumulation_dtype)
    lhs_message_bytes = batch * (problem.gemm.m // row_count) * (problem.gemm.k // column_count) * dtype_bytes // slice_count
    rhs_message_bytes = batch * (problem.gemm.k // row_count) * (problem.gemm.n // column_count) * dtype_bytes // slice_count
    rank_flops_per_slice = problem.gemm.flops // (len(ranks) * slice_count)

    buffer_actions: dict[tuple[int, str], list[str]] = defaultdict(list)
    for slice_index in range(slice_count):
        phase = SwizzlePhase.PROLOGUE if slice_index == 0 else SwizzlePhase.STEADY
        sends: dict[tuple[str, int, int], SwizzleActionWitness] = {}
        for source in ranks:
            reuse_deps = ()
            if slice_index >= 2:
                reuse_deps = (comp_by_slice[(source, slice_index - 2)],)
            for operand, line in _line_peers(source, rows, columns):
                payload = lhs_message_bytes if operand == "lhs" else rhs_message_bytes
                source_buffer = f"buffer.meshslice.rank.{source}.{operand}"
                for destination in line:
                    if destination == source:
                        continue
                    route = route_by_pair.get((source, destination))
                    if route is None:
                        raise SchemaError(
                            f"missing MeshSlice route {source}->{destination}",
                            path="swizzle_problem.group.routes",
                        )
                    send = _make_action(
                        rank=source,
                        kind=SwizzleActionKind.SEND,
                        deps=reuse_deps,
                        chunk_index=slice_index,
                        phase=phase,
                        peer_rank=destination,
                        route_ref=getattr(route, "id"),
                        input_refs=(source_buffer,),
                        output_refs=(),
                        logical_bytes=payload,
                        flops=0,
                    )
                    sends[(operand, source, destination)] = send
                    actions[source].append(send)
                    used_routes.add(getattr(route, "id"))
                    buffer_actions[(source, source_buffer)].append(send.id)

        waits: dict[tuple[int, str], list[str]] = defaultdict(list)
        for destination in ranks:
            for operand, line in _line_peers(destination, rows, columns):
                destination_buffer = f"buffer.meshslice.rank.{destination}.{operand}"
                payload = lhs_message_bytes if operand == "lhs" else rhs_message_bytes
                for source in line:
                    if source == destination:
                        continue
                    send = sends[(operand, source, destination)]
                    route = route_by_pair[(source, destination)]
                    receive = _make_action(
                        rank=destination,
                        kind=SwizzleActionKind.RECV,
                        deps=(send.id,),
                        chunk_index=slice_index,
                        phase=phase,
                        peer_rank=source,
                        route_ref=getattr(route, "id"),
                        input_refs=(),
                        output_refs=(destination_buffer,),
                        logical_bytes=payload,
                        flops=0,
                    )
                    wait = _make_action(
                        rank=destination,
                        kind=SwizzleActionKind.WAIT,
                        deps=(receive.id,),
                        chunk_index=slice_index,
                        phase=phase,
                        peer_rank=None,
                        route_ref=None,
                        input_refs=(),
                        output_refs=(),
                        logical_bytes=0,
                        flops=0,
                    )
                    actions[destination].extend((receive, wait))
                    waits[(destination, operand)].append(wait.id)
                    buffer_actions[(destination, destination_buffer)].extend((receive.id, wait.id))

        for rank in ranks:
            lhs_buffer = f"buffer.meshslice.rank.{rank}.lhs"
            rhs_buffer = f"buffer.meshslice.rank.{rank}.rhs"
            output_buffer = f"buffer.meshslice.rank.{rank}.output"
            deps = tuple(waits[(rank, "lhs")] + waits[(rank, "rhs")])
            if rank in last_comp:
                deps += (last_comp[rank],)
            compute = _make_action(
                rank=rank,
                kind=SwizzleActionKind.COMP,
                deps=deps,
                chunk_index=slice_index,
                phase=(SwizzlePhase.EPILOGUE if slice_index == slice_count - 1 else SwizzlePhase.STEADY),
                peer_rank=None,
                route_ref=None,
                input_refs=(
                    (lhs_buffer, rhs_buffer)
                    if slice_index == 0
                    else (lhs_buffer, rhs_buffer, output_buffer)
                ),
                output_refs=(output_buffer,),
                logical_bytes=0,
                flops=rank_flops_per_slice,
            )
            actions[rank].append(compute)
            last_comp[rank] = compute.id
            comp_by_slice[(rank, slice_index)] = compute.id
            buffer_actions[(rank, lhs_buffer)].append(compute.id)
            buffer_actions[(rank, rhs_buffer)].append(compute.id)
            buffer_actions[(rank, output_buffer)].append(compute.id)

    output_bytes = batch * (problem.gemm.m // len(rows)) * (problem.gemm.n // len(columns)) * accum_bytes
    if problem.pattern in (FusionPattern.GEMM_RS, FusionPattern.GEMM_AR):
        reductions: dict[int, SwizzleActionWitness] = {}
        for rank in ranks:
            output_buffer = f"buffer.meshslice.rank.{rank}.output"
            reduced_buffer = f"buffer.meshslice.rank.{rank}.reduced"
            reduction = _make_action(
                rank=rank,
                kind=SwizzleActionKind.REDUCE,
                deps=(last_comp[rank],),
                chunk_index=slice_count - 1,
                phase=SwizzlePhase.EPILOGUE,
                peer_rank=None,
                route_ref=None,
                input_refs=(output_buffer,),
                output_refs=(reduced_buffer,),
                logical_bytes=output_bytes,
                flops=0,
            )
            actions[rank].append(reduction)
            reductions[rank] = reduction
            buffer_actions[(rank, output_buffer)].append(reduction.id)
            buffer_actions[(rank, reduced_buffer)].append(reduction.id)
        if problem.pattern is FusionPattern.GEMM_AR:
            replication_sends: dict[tuple[int, int], SwizzleActionWitness] = {}
            for source in ranks:
                reduced_buffer = f"buffer.meshslice.rank.{source}.reduced"
                for destination in ranks:
                    if source == destination:
                        continue
                    route = route_by_pair[(source, destination)]
                    send = _make_action(
                        rank=source,
                        kind=SwizzleActionKind.SEND,
                        deps=(reductions[source].id,),
                        chunk_index=slice_count - 1,
                        phase=SwizzlePhase.EPILOGUE,
                        peer_rank=destination,
                        route_ref=getattr(route, "id"),
                        input_refs=(reduced_buffer,),
                        output_refs=(),
                        logical_bytes=output_bytes,
                        flops=0,
                    )
                    actions[source].append(send)
                    replication_sends[(source, destination)] = send
                    used_routes.add(getattr(route, "id"))
                    buffer_actions[(source, reduced_buffer)].append(send.id)
            for destination in ranks:
                waits_for_rank: list[str] = []
                replicated_buffer = f"buffer.meshslice.rank.{destination}.replicated"
                for source in ranks:
                    if source == destination:
                        continue
                    send = replication_sends[(source, destination)]
                    route = route_by_pair[(source, destination)]
                    receive = _make_action(
                        rank=destination,
                        kind=SwizzleActionKind.RECV,
                        deps=(send.id,),
                        chunk_index=slice_count - 1,
                        phase=SwizzlePhase.EPILOGUE,
                        peer_rank=source,
                        route_ref=getattr(route, "id"),
                        input_refs=(),
                        output_refs=(replicated_buffer,),
                        logical_bytes=output_bytes,
                        flops=0,
                    )
                    wait = _make_action(
                        rank=destination,
                        kind=SwizzleActionKind.WAIT,
                        deps=(receive.id,),
                        chunk_index=slice_count - 1,
                        phase=SwizzlePhase.EPILOGUE,
                        peer_rank=None,
                        route_ref=None,
                        input_refs=(),
                        output_refs=(),
                        logical_bytes=0,
                        flops=0,
                    )
                    actions[destination].extend((receive, wait))
                    waits_for_rank.append(wait.id)
                    buffer_actions[(destination, replicated_buffer)].extend((receive.id, wait.id))
                barrier = _make_action(
                    rank=destination,
                    kind=SwizzleActionKind.BARRIER,
                    deps=tuple(waits_for_rank),
                    chunk_index=slice_count - 1,
                    phase=SwizzlePhase.EPILOGUE,
                    peer_rank=None,
                    route_ref=None,
                    input_refs=(replicated_buffer,),
                    output_refs=(replicated_buffer,),
                    logical_bytes=0,
                    flops=0,
                )
                actions[destination].append(barrier)
                buffer_actions[(destination, replicated_buffer)].append(barrier.id)

    _session_safe_gather_order(
        actions,
        rows=rows,
        columns=columns,
        slice_count=slice_count,
    )
    action_orders = {
        rank: {action.id: index for index, action in enumerate(actions[rank])}
        for rank in ranks
    }
    programs = tuple(
        SwizzleRankProgramWitness(rank, tuple(actions[rank])) for rank in ranks
    )
    lhs_buffer_bytes = batch * (problem.gemm.m // len(rows)) * (problem.gemm.k // slice_count) * dtype_bytes
    rhs_buffer_bytes = batch * (problem.gemm.k // slice_count) * (problem.gemm.n // len(columns)) * dtype_bytes
    sizes = {"lhs": lhs_buffer_bytes, "rhs": rhs_buffer_bytes, "output": output_bytes, "reduced": output_bytes, "replicated": output_bytes}
    requirements = tuple(
        SwizzleBufferRequirement(
            rank=rank,
            buffer_ref=buffer_ref,
            size_bytes=sizes[buffer_ref.rsplit(".", 1)[-1]],
            double_buffered=(buffer_ref.endswith((".lhs", ".rhs")) and slice_count > 1),
            lifetime_action_refs=tuple(sorted(
                dict.fromkeys(action_refs),
                key=action_orders[rank].__getitem__,
            )),
        )
        for (rank, buffer_ref), action_refs in sorted(buffer_actions.items())
    )
    return programs, requirements, tuple(sorted(used_routes))


def _feasibility(
    problem: SwizzleProblem,
    *,
    slice_count: int,
    programs: tuple[SwizzleRankProgramWitness, ...],
    requirements: tuple[SwizzleBufferRequirement, ...],
    rectangle: bool,
    rows: tuple[tuple[int, ...], ...],
    columns: tuple[tuple[int, ...], ...],
    boundary_closed: bool,
) -> SwizzleFeasibilityWitness:
    action_count = sum(len(program.actions) for program in programs)
    messages = tuple(
        action.logical_bytes
        for program in programs
        for action in program.actions
        if action.kind is SwizzleActionKind.SEND
    )
    per_rank_sram: dict[int, int] = defaultdict(int)
    for requirement in requirements:
        per_rank_sram[requirement.rank] += requirement.size_bytes * (2 if requirement.double_buffered else 1)
    tile = (
        problem.gemm.m // len(rows),
        problem.gemm.n // len(columns),
        problem.gemm.k // slice_count,
    )
    floor = problem.hardware_profile.efficient_tile_floor
    mode = meshslice_execution_mode(len(rows), len(columns))
    if mode is MeshSliceExecutionMode.FULL_2D:
        dte_inflight = (
            problem.hardware_profile.max_inflight_dte >= 2,
            "row and column communication can be concurrently in flight",
        )
        dimensionality = (
            True,
            "both physical dimensions contain multiple ranks",
        )
    elif mode is MeshSliceExecutionMode.LOCAL:
        dte_inflight = (True, "LOCAL mode has no remote DTE traffic")
        dimensionality = (True, "LOCAL mode is the 1x1 rectangle")
    else:
        dte_inflight = (
            problem.hardware_profile.max_inflight_dte >= 1,
            f"{mode.value} requires one remote DTE stream",
        )
        dimensionality = (
            True,
            f"{mode.value} is a complete degenerate rectangle",
        )
    checks = {
        "action_count": (action_count <= problem.constraints.max_actions, f"{action_count} actions <= limit {problem.constraints.max_actions}"),
        "boundary_sharding": (boundary_closed, "explicit 2D OS sharding or boundary reshard witness"),
        "buffer_count": (len(requirements) <= problem.constraints.max_buffers, f"{len(requirements)} buffers <= limit {problem.constraints.max_buffers}"),
        "double_buffer": (slice_count == 1 or problem.hardware_profile.double_buffer_supported, "multi-slice input double buffering is supported"),
        "dte_inflight": dte_inflight,
        "efficient_tile": (all(tile[index] >= floor[index] for index in range(3)), f"tile {tile!r} satisfies floor {floor!r}"),
        "message_payload": (not messages or min(messages) >= problem.hardware_profile.min_transfer_bytes, "all partial collectives satisfy minimum payload"),
        "rectangle": (rectangle, "placement is a complete physical rectangle"),
        "sram": (max(per_rank_sram.values(), default=0) <= problem.hardware_profile.sram_budget_bytes, f"per-rank high-water <= {problem.hardware_profile.sram_budget_bytes}"),
        "two_dimensional": dimensionality,
    }
    return SwizzleFeasibilityWitness(
        tuple(
            SwizzleFeasibilityCheck(name, passed, reason)
            for name, (passed, reason) in sorted(checks.items())
        )
    )


def generate_meshslice_2d_drafts(
    problem: SwizzleProblem,
    semantic_witness: SwizzleSemanticWitness,
    *,
    allow_boundary_reshard: bool = False,
) -> tuple[SwizzleCandidateDraft, ...]:
    """Generate Level-0/1/2 MeshSlice OS drafts in canonical slice order."""

    problem.validate("swizzle_problem")
    semantic_witness.validate("semantic_witness")
    if SwizzleAlgorithm.MESHSLICE_2D_OS not in problem.constraints.allowed_algorithms:
        return ()
    if semantic_witness.pattern is not problem.pattern:
        raise SchemaError("semantic witness pattern mismatch", path="semantic_witness.pattern")
    rows, columns, rectangle = _physical_lines(problem)
    exact_sharding = _exact_os_2d_sharding(problem)
    boundary_closed = exact_sharding or (
        allow_boundary_reshard and semantic_witness.sharding_transition_closed
    )
    if not rectangle or not rows or not columns or not boundary_closed:
        return ()
    slice_counts = _blocked_slice_counts(problem, len(rows), len(columns))
    drafts: list[SwizzleCandidateDraft] = []
    for slice_count in slice_counts:
        action_count, buffer_count = _candidate_scale(
            problem,
            rows=len(rows),
            columns=len(columns),
            slice_count=slice_count,
        )
        if (
            action_count > problem.constraints.max_actions
            or buffer_count > problem.constraints.max_buffers
        ):
            continue
        programs, requirements, route_refs = _programs_for_slice_count(
            problem, rows, columns, slice_count
        )
        feasibility = _feasibility(
            problem,
            slice_count=slice_count,
            programs=programs,
            requirements=requirements,
            rectangle=rectangle,
            rows=rows,
            columns=columns,
            boundary_closed=boundary_closed,
        )
        if not feasibility.feasible:
            continue
        rank_order = tuple(rank for row in rows for rank in row)
        draft = SwizzleCandidateDraft(
            problem_ref=problem.id,
            pattern=problem.pattern,
            algorithm=SwizzleAlgorithm.MESHSLICE_2D_OS,
            split_axis=semantic_witness.split_axis,
            chunk_count=slice_count,
            unroll_degree=1,
            rank_programs=programs,
            buffer_requirements=requirements,
            topology_witness=SwizzleTopologyWitness(
                kind=SwizzleTopologyKind.RECTANGLE_2D,
                rank_order=rank_order,
                row_orders=rows,
                column_orders=columns,
                route_refs=route_refs,
                is_complete_rectangle=True,
                has_hamiltonian_cycle=False,
            ),
            semantic_witness=semantic_witness,
            feasibility_witness=feasibility,
            tile_shape=(
                problem.gemm.m // len(rows),
                problem.gemm.n // len(columns),
                problem.gemm.k // slice_count,
            ),
        )
        draft.validate_against(problem)
        drafts.append(draft)
    return tuple(drafts[: min(32, problem.constraints.max_candidates)])


__all__ = [
    "generate_meshslice_2d_drafts",
    "meshslice_candidate_scale",
    "meshslice_execution_mode",
    "MeshSliceExecutionMode",
]
