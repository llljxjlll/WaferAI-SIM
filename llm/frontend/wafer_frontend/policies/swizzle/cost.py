"""Analytical resource-DAG cost model for Swizzle candidates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from ...errors import SchemaError
from ...schema.common import validate_dependency_dag
from ...schema.ir0 import FusionPattern
from ...schema.swizzle import (
    SwizzleActionKind,
    SwizzleActionWitness,
    SwizzleAlgorithm,
    SwizzleBufferRequirement,
    SwizzleCandidate,
    SwizzleCost,
    SwizzleFeasibilityCheck,
    SwizzleFeasibilityWitness,
    SwizzleHardwareProfile,
    SwizzlePhase,
    SwizzleProblem,
    SwizzleRankProgramWitness,
    SwizzleSemanticWitness,
    SwizzleTopologyKind,
    SwizzleTopologyWitness,
)


@dataclass(frozen=True, slots=True)
class ActionTiming:
    action_ref: str
    start_cycle: float
    finish_cycle: float
    resources: tuple[str, ...]


def interpolate_efficiency(
    profile: SwizzleHardwareProfile,
    tile_shape: tuple[int, int, int],
) -> float:
    """Interpolate calibrated efficiency in log-shape space.

    The nearest four calibration points are inverse-distance weighted.  Exact
    hits remain exact, and extrapolation is conservatively clamped to the
    observed efficiency range.
    """

    profile.validate("hardware_profile")
    if len(tile_shape) != 3 or any(type(item) is not int or item <= 0 for item in tile_shape):
        raise SchemaError("tile_shape must be a positive M/N/K triple", path="tile_shape")
    target = tuple(math.log2(item) for item in tile_shape)
    distances: list[tuple[float, int, float]] = []
    for index, point in enumerate(profile.efficiency_points):
        shape = (point.m, point.n, point.k)
        if shape == tile_shape:
            return point.efficiency
        distance = math.sqrt(
            sum((math.log2(extent) - target[axis]) ** 2 for axis, extent in enumerate(shape))
        )
        distances.append((distance, index, point.efficiency))
    nearest = sorted(distances)[: min(4, len(distances))]
    numerator = sum(efficiency / distance for distance, _, efficiency in nearest)
    denominator = sum(1.0 / distance for distance, _, _ in nearest)
    estimate = numerator / denominator
    observed = tuple(point.efficiency for point in profile.efficiency_points)
    return min(max(estimate, min(observed)), max(observed))


def _route_index(problem: SwizzleProblem) -> dict[str, object]:
    return {route.id: route for route in problem.group.routes}


def _action_duration(
    action: SwizzleActionWitness,
    problem: SwizzleProblem,
    *,
    efficiency: float,
) -> float:
    profile = problem.hardware_profile
    if action.kind is SwizzleActionKind.COMP:
        return action.flops / (profile.peak_flops_per_cycle * efficiency)
    if action.kind is SwizzleActionKind.SEND:
        routes = _route_index(problem)
        if action.route_ref not in routes:
            raise SchemaError("transport references an unknown route", path="swizzle_action.route_ref")
        route = routes[action.route_ref]
        hops = len(getattr(route, "die_path")) - 1
        return float(
            profile.dte_launch_cycles
            + profile.dte_sync_cycles
            + action.logical_bytes / profile.lane_bytes_per_cycle
            + hops * profile.hop_latency_cycles
        )
    if action.kind is SwizzleActionKind.RECV:
        return float(problem.hardware_profile.dte_sync_cycles)
    if action.kind in (SwizzleActionKind.WAIT, SwizzleActionKind.BARRIER):
        return float(problem.hardware_profile.dte_sync_cycles)
    if action.kind in (SwizzleActionKind.REDUCE, SwizzleActionKind.LOCAL_COPY):
        payload = max(1, action.logical_bytes)
        return payload / problem.hardware_profile.lane_bytes_per_cycle
    raise SchemaError("unsupported action kind", path="swizzle_action.kind")


def _base_resources(
    action: SwizzleActionWitness,
    problem: SwizzleProblem,
) -> tuple[str, ...]:
    if action.kind in (SwizzleActionKind.COMP, SwizzleActionKind.REDUCE):
        return (f"compute.rank.{action.rank}",)
    if action.kind is SwizzleActionKind.LOCAL_COPY:
        return (f"memory.rank.{action.rank}",)
    if action.kind is SwizzleActionKind.SEND:
        route = _route_index(problem).get(action.route_ref or "")
        if route is None:
            raise SchemaError("transport references an unknown route", path="swizzle_action.route_ref")
        explicit = tuple(getattr(action, "resource_refs", ()))
        resources = explicit or tuple(getattr(route, "resource_ids"))
        return tuple(sorted(set(resources)))
    return ()


def _schedule(
    problem: SwizzleProblem,
    programs: tuple[SwizzleRankProgramWitness, ...],
    *,
    efficiency: float,
) -> tuple[ActionTiming, ...]:
    actions = tuple(action for program in programs for action in program.actions)
    action_by_id = validate_dependency_dag(actions, "swizzle_rank_programs.actions")
    order = {action.id: index for index, action in enumerate(actions)}
    remaining = {action.id: len(action.deps) for action in actions}
    dependents: dict[str, list[str]] = {action.id: [] for action in actions}
    for action in actions:
        for dependency in action.deps:
            dependents[dependency].append(action.id)
    ready = sorted((ref for ref, count in remaining.items() if count == 0), key=order.__getitem__)
    availability: dict[str, float] = defaultdict(float)
    finish: dict[str, float] = {}
    timings: dict[str, ActionTiming] = {}
    while ready:
        action_ref = ready.pop(0)
        action = action_by_id[action_ref]
        earliest = max((finish[ref] for ref in action.deps), default=0.0)
        resources = list(_base_resources(action, problem))
        if action.kind is SwizzleActionKind.SEND:
            slots = tuple(
                f"dte.rank.{action.rank}.slot.{slot}"
                for slot in range(problem.hardware_profile.max_inflight_dte)
            )
            slot = min(slots, key=lambda ref: (availability[ref], ref))
            resources.append(slot)
        canonical_resources = tuple(sorted(resources))
        start = max((availability[ref] for ref in canonical_resources), default=earliest)
        start = max(start, earliest)
        end = start + _action_duration(action, problem, efficiency=efficiency)
        for resource in canonical_resources:
            availability[resource] = end
        finish[action_ref] = end
        timings[action_ref] = ActionTiming(action_ref, start, end, canonical_resources)
        for dependent in dependents[action_ref]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                ready.append(dependent)
        ready.sort(key=order.__getitem__)
    if len(timings) != len(actions):
        raise SchemaError("action schedule did not close", path="swizzle_rank_programs.actions")
    return tuple(timings[action.id] for action in actions)


def _sram_high_water(
    requirements: tuple[SwizzleBufferRequirement, ...],
    timings: dict[str, ActionTiming],
) -> int:
    by_rank: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    for requirement in requirements:
        references = tuple(timings[ref] for ref in requirement.lifetime_action_refs)
        start = min(item.start_cycle for item in references)
        finish = max(item.finish_cycle for item in references)
        size = requirement.size_bytes * (2 if requirement.double_buffered else 1)
        by_rank[requirement.rank].append((start, 1, size))
        by_rank[requirement.rank].append((finish, -1, size))
    high_water = 0
    for events in by_rank.values():
        current = 0
        for _, delta, size in sorted(events, key=lambda item: (item[0], item[1])):
            current += delta * size
            high_water = max(high_water, current)
    return high_water


def _max_inflight_sends(
    actions: tuple[SwizzleActionWitness, ...],
    timings: dict[str, ActionTiming],
) -> int:
    events: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for action in actions:
        if action.kind is not SwizzleActionKind.SEND:
            continue
        timing = timings[action.id]
        events[action.rank].append((timing.start_cycle, 1))
        events[action.rank].append((timing.finish_cycle, -1))
    maximum = 0
    for rank_events in events.values():
        current = 0
        for _, delta in sorted(rank_events, key=lambda item: (item[0], item[1])):
            current += delta
            maximum = max(maximum, current)
    return maximum


def evaluate_action_dag(
    problem: SwizzleProblem,
    programs: tuple[SwizzleRankProgramWitness, ...],
    requirements: tuple[SwizzleBufferRequirement, ...],
    *,
    tile_shape: tuple[int, int, int],
) -> SwizzleCost:
    """Level-3 deterministic earliest-start scheduling over typed witnesses."""

    problem.validate("swizzle_problem")
    if not programs:
        raise SchemaError("fused cost evaluation requires rank programs", path="rank_programs")
    efficiency = interpolate_efficiency(problem.hardware_profile, tile_shape)
    scheduled = _schedule(problem, programs, efficiency=efficiency)
    timing_by_id = {item.action_ref: item for item in scheduled}
    actions = tuple(action for program in programs for action in program.actions)
    action_by_id = {action.id: action for action in actions}
    makespan = max(item.finish_cycle for item in scheduled)
    prologue_end = max(
        (item.finish_cycle for item in scheduled if action_by_id[item.action_ref].phase is SwizzlePhase.PROLOGUE),
        default=0.0,
    )
    epilogue_start = min(
        (item.start_cycle for item in scheduled if action_by_id[item.action_ref].phase is SwizzlePhase.EPILOGUE),
        default=makespan,
    )
    prologue = min(prologue_end, makespan)
    steady = max(0.0, epilogue_start - prologue)
    epilogue = max(0.0, makespan - prologue - steady)

    route_by_id = _route_index(problem)
    sends = tuple(action for action in actions if action.kind is SwizzleActionKind.SEND)
    logical_bytes = sum(action.logical_bytes for action in sends)
    byte_hops = sum(
        action.logical_bytes * (len(getattr(route_by_id[action.route_ref or ""], "die_path")) - 1)
        for action in sends
    )
    network_busy: dict[str, float] = defaultdict(float)
    for timing in scheduled:
        action = action_by_id[timing.action_ref]
        if action.kind is not SwizzleActionKind.SEND:
            continue
        for resource in timing.resources:
            if resource.startswith("dte."):
                continue
            network_busy[resource] += timing.finish_cycle - timing.start_cycle
    bottleneck: tuple[str, ...] = ()
    if network_busy:
        maximum = max(network_busy.values())
        bottleneck = tuple(sorted(ref for ref, busy in network_busy.items() if math.isclose(busy, maximum)))
    utilization = 0.0
    if network_busy and makespan > 0.0:
        utilization = min(1.0, sum(network_busy.values()) / (makespan * len(network_busy)))
    high_water = _sram_high_water(requirements, timing_by_id)
    confidence = problem.hardware_profile.confidence_fraction
    lower = max(0.0, makespan * (1.0 - confidence))
    upper = makespan * (1.0 + confidence)
    control_count = sum(
        action.kind in (SwizzleActionKind.WAIT, SwizzleActionKind.BARRIER)
        for action in actions
    )
    return SwizzleCost.create(
        estimated_cycles=float(makespan),
        lower_cycles=float(lower),
        upper_cycles=float(upper),
        prologue_cycles=float(prologue),
        steady_cycles=float(steady),
        epilogue_cycles=float(epilogue),
        logical_bytes=logical_bytes,
        byte_hops=byte_hops,
        message_count=len(sends),
        direction_port_utilization=float(utilization),
        control_action_count=control_count,
        max_inflight=_max_inflight_sends(actions, timing_by_id),
        sram_high_water_bytes=high_water,
        bottleneck_resources=bottleneck,
    )


def _baseline_cost(problem: SwizzleProblem) -> SwizzleCost:
    profile = problem.hardware_profile
    ranks = len(problem.group.placements)
    rows, columns = problem.group.logical_shape
    tile = (
        max(1, problem.gemm.m // rows),
        max(1, problem.gemm.n // columns),
        problem.gemm.k,
    )
    efficiency = interpolate_efficiency(profile, tile)
    compute = (problem.gemm.flops / ranks) / (profile.peak_flops_per_cycle * efficiency)
    if problem.pattern is FusionPattern.AG_GEMM:
        step_bytes = problem.collective.rank_input_bytes
    elif problem.pattern is FusionPattern.GEMM_RS:
        step_bytes = problem.collective.rank_output_bytes
    else:
        step_bytes = problem.collective.logical_bytes // ranks
    one_phase = float(
        profile.dte_launch_cycles
        + (ranks - 1)
        * (
            profile.dte_sync_cycles
            + step_bytes / profile.lane_bytes_per_cycle
            + profile.hop_latency_cycles
        )
    )
    phase_count = 2 if problem.pattern is FusionPattern.GEMM_AR else 1
    communication = one_phase * phase_count
    if problem.pattern is FusionPattern.AG_GEMM:
        prologue, steady, epilogue = communication, compute, 0.0
    else:
        prologue, steady, epilogue = 0.0, compute, communication
    estimated = prologue + steady + epilogue
    confidence = profile.confidence_fraction
    if problem.pattern is FusionPattern.GEMM_AR:
        logical_bytes = 2 * (ranks - 1) * problem.collective.logical_bytes
    else:
        logical_bytes = (ranks - 1) * problem.collective.logical_bytes
    route_hops = tuple(len(route.die_path) - 1 for route in problem.group.routes)
    average_hops = sum(route_hops) / len(route_hops) if route_hops else 0.0
    return SwizzleCost.create(
        estimated_cycles=float(estimated),
        lower_cycles=float(max(0.0, estimated * (1.0 - confidence))),
        upper_cycles=float(estimated * (1.0 + confidence)),
        prologue_cycles=float(prologue),
        steady_cycles=float(steady),
        epilogue_cycles=float(epilogue),
        logical_bytes=logical_bytes,
        byte_hops=math.ceil(logical_bytes * average_hops),
        message_count=phase_count * ranks * (ranks - 1),
        direction_port_utilization=0.0,
        control_action_count=phase_count * (ranks - 1),
        max_inflight=1,
        sram_high_water_bytes=0,
        bottleneck_resources=(),
    )


def build_unfused_baseline(
    problem: SwizzleProblem,
    semantic_witness: SwizzleSemanticWitness,
) -> SwizzleCandidate:
    """Build the mandatory sequential collective/GEMM comparison point."""

    problem.validate("swizzle_problem")
    semantic_witness.validate("semantic_witness")
    return SwizzleCandidate.create(
        problem_ref=problem.id,
        pattern=problem.pattern,
        algorithm=SwizzleAlgorithm.UNFUSED,
        split_axis=None,
        chunk_count=0,
        unroll_degree=0,
        rank_programs=(),
        buffer_requirements=(),
        topology_witness=SwizzleTopologyWitness(
            kind=SwizzleTopologyKind.UNFUSED,
            rank_order=(),
            row_orders=(),
            column_orders=(),
            route_refs=(),
            is_complete_rectangle=False,
            has_hamiltonian_cycle=False,
        ),
        semantic_witness=semantic_witness,
        feasibility_witness=SwizzleFeasibilityWitness(
            (SwizzleFeasibilityCheck("baseline", True, "unfused execution is always retained"),)
        ),
        cost=_baseline_cost(problem),
    )


__all__ = [
    "ActionTiming",
    "build_unfused_baseline",
    "evaluate_action_dag",
    "interpolate_efficiency",
]
