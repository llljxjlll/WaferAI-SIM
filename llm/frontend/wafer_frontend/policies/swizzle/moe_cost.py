"""Typed personalized resource-DAG cost and deterministic MoE decisions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cmp_to_key
import math
import statistics

from ...errors import SchemaError
from ...schema.ir0 import FusionPattern
from ...schema.serde import canonical_digest
from ...schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleDecisionReason,
)
from ...schema.swizzle_moe import (
    MoeActionWitness,
    MoePacketWitness,
    MoeScenarioResourceEstimate,
    MoeSwizzleCandidate,
    MoeSwizzleCost,
    MoeSwizzleDecision,
    MoeSwizzleProblem,
    MoeSwizzleWorkloadSelection,
    MoeTrafficScenarioKind,
)
from ...schema.swizzle_moe_placement import (
    MoeCandidateCoreLifecycleFloor,
    MoeWholePairFeasibility,
    build_moe_action_dynamic_root_bindings,
    build_moe_candidate_action_owner_map,
    build_moe_candidate_core_lifecycle_floor,
)
from ...schema.swizzle_moe_calibration import (
    MoeCalibrationKind,
    MoeCalibrationStatus,
    MoeSwizzleCalibrationProfile,
)


_PROVISIONAL = {
    "dte_launch": 8.0,
    "dte_sync": 2.0,
    "dte_hop": 1.0,
    "sram_alloc": 1.0,
    "sram_bind": 1.0,
    "sram_free": 1.0,
    "event_set": 1.0,
    "event_wait": 1.0,
    "terminal_done": 1.0,
    "session_open": 1.0,
    "session_retire": 1.0,
    "local_copy": 1.0,
}


def _profile_statistics(profile):
    if profile is None:
        return _PROVISIONAL, {}, {}, 0.25
    profile.validate("moe_cost.calibration_profile")
    if profile.status is not MoeCalibrationStatus.MEASURED:
        raise SchemaError("provided calibration profile must be MEASURED", path="moe_cost.calibration_profile")
    grouped = defaultdict(list)
    repeat_spreads = []
    for sample in profile.samples:
        grouped[(sample.kind.value, sample.shape)].append(float(sample.cycles))
    fixed = {}
    shapes = {}
    swiglu_shapes = {}
    for (kind, shape), values in grouped.items():
        median = float(statistics.median(values))
        by_sample = defaultdict(list)
        for sample in profile.samples:
            if sample.kind.value == kind and sample.shape == shape:
                by_sample[sample.sample_index].append(float(sample.cycles))
        for pair in by_sample.values():
            if len(pair) == 2:
                repeat_spreads.append(abs(pair[0] - pair[1]) / max(1.0, statistics.median(pair)))
        if kind == MoeCalibrationKind.GROUP_GEMM.value:
            assert shape is not None
            shapes[shape] = median
        elif kind == MoeCalibrationKind.SWIGLU_GROUP.value:
            assert shape is not None
            swiglu_shapes[shape] = median
        else:
            fixed[kind] = median
    required = set(_PROVISIONAL)
    if set(fixed) != required:
        raise SchemaError(
            f"MEASURED fixed-kind coverage drifted; expected={sorted(required)}, actual={sorted(fixed)}",
            path="moe_cost.calibration_profile.samples",
        )
    return fixed, shapes, swiglu_shapes, max(0.01, max(repeat_spreads, default=0.0))


def _group_model(shapes):
    points = sorted(
        (2.0 * m * n * k, cycles, shape)
        for shape, cycles in shapes.items()
        for m, n, k in (shape,)
    )
    if not points:
        return 0.0, 1.0 / 256.0, (0.0, math.inf), {}
    if len(points) < 3:
        raise SchemaError("MEASURED GroupGEMM fit requires at least three shapes", path="moe_cost.calibration_profile.samples")
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((item - mean_x) ** 2 for item in xs)
    if denominator == 0.0:
        raise SchemaError("GroupGEMM shapes do not span FLOP scale", path="moe_cost.calibration_profile.samples")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator
    intercept = mean_y - slope * mean_x
    if slope < 0.0 or intercept < 0.0:
        raise SchemaError("GroupGEMM fit has negative slope/setup", path="moe_cost.calibration_profile.samples")
    return intercept, slope, (min(xs) / 2.0, max(xs) * 2.0), shapes


def _group_cycles(action, problem, model):
    intercept, slope, interval, exact = model
    gemm = problem.region.semantic_witness.traffic.expert_gemms[action.expert_index]
    n = gemm.n
    shape = (len(action.assignment_refs), n, gemm.k)
    if shape in exact:
        return exact[shape]
    if exact:
        raise SchemaError(
            f"GroupGEMM shape {shape} lacks an exact measured sample",
            path="moe_cost.actions",
        )
    flops = float(action.flops)
    if not interval[0] <= flops <= interval[1]:
        raise SchemaError(
            f"GroupGEMM shape {shape} is outside measured extrapolation interval {interval}",
            path="moe_cost.actions",
        )
    return intercept + slope * flops


def build_moe_action_owner_map(problem, actions):
    """Compatibility name for the shared canonical candidate owner map."""
    return build_moe_candidate_action_owner_map(problem, actions)


def _pivot_action_ids(actions):
    groups = defaultdict(list)
    for action in actions:
        if action.pivot_rank is not None and action.stage is not None:
            groups[(action.assignment_refs, action.expert_index, action.n_block_index, action.pivot_rank)].append(action)
    return {
        action.id for group in groups.values()
        if {item.stage for item in group} == {0, 1}
        for action in group if action.rank == action.pivot_rank
    }


def _action_resources(action, problem, owner, pivot_action_ids):
    if action.kind is SwizzleActionKind.COMP:
        return (f"compute.core.{owner.runtime_core_id}",)
    if action.kind is SwizzleActionKind.SWIGLU:
        return (f"compute.core.{owner.runtime_core_id}",)
    if action.kind is SwizzleActionKind.SEND:
        route = next(
            item for item in problem.topology.group.routes if item.id == action.route_ref
        )
        suffix = (f"dte.core.{owner.runtime_core_id}",)
        if action.id in pivot_action_ids:
            suffix += (f"pivot_sram.core.{owner.runtime_core_id}",)
        return tuple(route.resource_ids) + suffix
    if action.kind is SwizzleActionKind.RECV:
        suffix = (f"dte.core.{owner.runtime_core_id}",)
        if action.id in pivot_action_ids:
            suffix += (f"pivot_sram.core.{owner.runtime_core_id}",)
        return suffix
    if action.kind in (SwizzleActionKind.WAIT, SwizzleActionKind.BARRIER):
        return (f"control.core.{owner.runtime_core_id}",)
    if action.kind is SwizzleActionKind.LOCAL_COPY:
        return (f"sram.core.{owner.runtime_core_id}",)
    return ()


def _action_duration(action, problem, fixed, group_model, swiglu_shapes):
    if action.kind is SwizzleActionKind.COMP:
        return _group_cycles(action, problem, group_model)
    if action.kind is SwizzleActionKind.SWIGLU:
        gemm = problem.region.semantic_witness.traffic.expert_gemms[action.expert_index]
        shape = (
            len(action.assignment_refs), gemm.n,
            len(action.assignment_refs) * gemm.n,
        )
        if swiglu_shapes:
            if shape not in swiglu_shapes:
                raise SchemaError(
                    f"SWIGLU_GROUP shape {shape} lacks an exact measured sample",
                    path="moe_cost.actions",
                )
            return swiglu_shapes[shape]
        return float(max(1, shape[2] // 32))
    if action.kind is SwizzleActionKind.SEND:
        route = next(
            item for item in problem.topology.group.routes if item.id == action.route_ref
        )
        return fixed["session_open"] + fixed["dte_launch"] + (len(route.die_path) - 1) * fixed["dte_hop"]
    if action.kind is SwizzleActionKind.RECV:
        return fixed["dte_launch"]
    if action.kind is SwizzleActionKind.WAIT:
        return fixed["dte_sync"] + fixed["session_retire"]
    if action.kind is SwizzleActionKind.LOCAL_COPY:
        return fixed["local_copy"]
    if action.kind is SwizzleActionKind.BARRIER:
        return fixed["event_set"] + fixed["event_wait"]
    raise SchemaError("unsupported MoE cost action", path="moe_cost.actions")


def _earliest_schedule(actions, problem, fixed, group_model, swiglu_shapes):
    by_id = {item.id: item for item in actions}
    if len(by_id) != len(actions):
        raise SchemaError("cost actions duplicate ids", path="moe_cost.actions")
    order = {item.id: index for index, item in enumerate(actions)}
    owners = build_moe_action_owner_map(problem, actions)
    pivot_action_ids = _pivot_action_ids(actions)
    pending = set(by_id)
    finish = {}
    resource_ready = defaultdict(float)
    session_ready = {
        core.runtime_core_id: [0.0] * problem.endpoint_session_capacity
        for cores in problem.hardware_facts.ordered_cores_by_die for core in cores
    }
    timings = {}
    while pending:
        ready = tuple(
            ref for ref in pending if set(by_id[ref].deps).issubset(finish)
        )
        if not ready:
            raise SchemaError("cost action graph is cyclic/dangling", path="moe_cost.actions")

        def preview(ref):
            action = by_id[ref]
            resources = _action_resources(
                action, problem, owners[action.id], pivot_action_ids
            )
            dependency_ready = max(
                (finish[item] for item in action.deps), default=0.0
            )
            if action.kind is not SwizzleActionKind.SEND:
                start = max(
                    (dependency_ready, *(resource_ready[item] for item in resources))
                ) if resources else dependency_ready
                return start, start + _action_duration(
                    action, problem, fixed, group_model, swiglu_shapes
                )
            route = next(
                item for item in problem.topology.group.routes
                if item.id == action.route_ref
            )
            recv = next(
                item for item in actions
                if item.kind is SwizzleActionKind.RECV
                and item.packet_ref == action.packet_ref
                and item.stage == action.stage
                and item.rank == action.peer_rank
            )
            start = dependency_ready
            for runtime_core in (
                owners[action.id].runtime_core_id,
                owners[recv.id].runtime_core_id,
            ):
                start = max(start, min(session_ready[runtime_core]))
            service_start = max(
                start, resource_ready[f"dte.core.{owners[action.id].runtime_core_id}"]
            )
            pivot_resource = f"pivot_sram.core.{owners[action.id].runtime_core_id}"
            if action.id in pivot_action_ids:
                service_start = max(service_start, resource_ready[pivot_resource])
            cursor = service_start + fixed["session_open"] + fixed["dte_launch"]
            bandwidth = {
                item.resource_id: item.bytes_per_cycle
                for item in problem.hardware_facts.route_resources
            }
            for resource in route.resource_ids:
                cursor = max(cursor, resource_ready[resource])
                cursor += fixed["dte_hop"] + action.logical_bytes / bandwidth[resource]
            return service_start, cursor

        ref = min(ready, key=lambda item: (*preview(item), item))
        action = by_id[ref]
        resources = _action_resources(action, problem, owners[action.id], pivot_action_ids)
        start = max((finish[item] for item in action.deps), default=0.0)
        start = (
            max(start, *(resource_ready[item] for item in resources))
            if resources and action.kind is not SwizzleActionKind.SEND else start
        )
        chosen_sessions = []
        if action.kind is SwizzleActionKind.SEND:
            route = next(item for item in problem.topology.group.routes if item.id == action.route_ref)
            bandwidth = {item.resource_id: item.bytes_per_cycle for item in problem.hardware_facts.route_resources}
            recv = next(
                item for item in actions
                if item.kind is SwizzleActionKind.RECV
                and item.packet_ref == action.packet_ref
                and item.stage == action.stage
                and item.rank == action.peer_rank
            )
            for runtime_core in (owners[action.id].runtime_core_id, owners[recv.id].runtime_core_id):
                lane = min(range(problem.endpoint_session_capacity), key=session_ready[runtime_core].__getitem__)
                start = max(start, session_ready[runtime_core][lane])
                chosen_sessions.append((runtime_core, lane))
            cursor = max(start, resource_ready[f"dte.core.{owners[action.id].runtime_core_id}"]) + fixed["session_open"] + fixed["dte_launch"]
            pivot_resource = f"pivot_sram.core.{owners[action.id].runtime_core_id}"
            if action.id in pivot_action_ids:
                cursor = max(cursor, resource_ready[pivot_resource])
            resource_ready[f"dte.core.{owners[action.id].runtime_core_id}"] = cursor
            for resource in route.resource_ids:
                cursor = max(cursor, resource_ready[resource])
                cursor += fixed["dte_hop"] + action.logical_bytes / bandwidth[resource]
                resource_ready[resource] = cursor
            end = cursor
            if action.id in pivot_action_ids:
                resource_ready[pivot_resource] = end
            for rank, lane in chosen_sessions:
                session_ready[rank][lane] = end + fixed["session_retire"]
            timings[ref] = (start, end)
            finish[ref] = end
            pending.remove(ref)
            continue
        duration = _action_duration(action, problem, fixed, group_model, swiglu_shapes)
        end = start + duration
        for resource in resources:
            resource_ready[resource] = end
        for rank, lane in chosen_sessions:
            session_ready[rank][lane] = end + fixed["session_retire"]
        timings[ref] = (start, end)
        finish[ref] = end
        pending.remove(ref)
    return timings


def _session_peaks(problem, actions, timings):
    owners = build_moe_action_owner_map(problem, actions)
    events_by_core = defaultdict(list)
    events_by_die = defaultdict(list)
    for send in (item for item in actions if item.kind is SwizzleActionKind.SEND):
        recv = next(item for item in actions if item.kind is SwizzleActionKind.RECV and item.packet_ref == send.packet_ref and item.stage == send.stage and item.rank == send.peer_rank)
        wait = next(item for item in actions if item.kind is SwizzleActionKind.WAIT and item.packet_ref == send.packet_ref and item.stage == send.stage and item.rank == send.peer_rank)
        end = timings[wait.id][1]
        for endpoint in (send, recv):
            owner = owners[endpoint.id]
            start, _ = timings[endpoint.id]
            events_by_core[owner.runtime_core_id].extend(((start, 1), (end, -1)))
            events_by_die[owner.rank].extend(((start, 1), (end, -1)))
    def peak(events):
        active = result = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            active += delta
            result = max(result, active)
        return result
    return max((peak(items) for items in events_by_core.values()), default=0), max((peak(items) for items in events_by_die.values()), default=0)


def _actual_loads(problem, actions, packets):
    routes = {item.id: item for item in problem.topology.group.routes}
    link_bytes = defaultdict(int)
    for action in actions:
        if action.kind is SwizzleActionKind.SEND:
            for resource in routes[action.route_ref].resource_ids:
                link_bytes[resource] += action.logical_bytes
    occurrences = defaultdict(int)
    for packet in packets:
        for item in packet.slices:
            occurrences[item.assignment_ref] += 1
    pivot_bytes = [0] * len(problem.topology.group.placements)
    for packet in packets:
        if (
            packet.stage == 0
            and packet.pivot_rank is not None
            and any(occurrences[item.assignment_ref] > 1 for item in packet.slices)
        ):
            pivot_bytes[packet.pivot_rank] += packet.logical_bytes
    return tuple(sorted(link_bytes.items())), tuple(pivot_bytes)


def _scenario_network_loads(problem, algorithm, scenario):
    actual = next(item for item in problem.traffic_scenarios if item.kind is MoeTrafficScenarioKind.ACTUAL)
    token_bytes = actual.logical_payload_bytes / sum(actual.expert_token_counts)
    routes = {(item.source_rank, item.destination_rank): item for item in problem.topology.group.routes}
    pivots = {(source, destination): pivot for source, destination, pivot in problem.topology.pivot_by_pair}
    link_bytes = defaultdict(int)
    link_packets = defaultdict(int)
    pivot_bytes = [0] * len(problem.topology.group.placements)
    dispatch = problem.region.pattern is FusionPattern.MOE_DISPATCH_GEMM
    for source, row in enumerate(scenario.source_expert_token_counts):
        for expert, count in enumerate(row):
            if count == 0:
                continue
            home = problem.region.semantic_witness.traffic.expert_gemms[expert].rank
            begin, end = (source, home) if dispatch else (home, source)
            if begin == end:
                continue
            segments = ((begin, end),)
            if algorithm is SwizzleAlgorithm.COMET_MESH_PERSONALIZED_A2A:
                pivot = pivots[(begin, end)]
                if pivot not in (begin, end):
                    segments = ((begin, pivot), (pivot, end))
                    pivot_bytes[pivot] += int(count * token_bytes)
            for left, right in segments:
                for resource in routes[(left, right)].resource_ids:
                    link_bytes[resource] += int(count * token_bytes)
                    link_packets[resource] += count
    return tuple(sorted(link_bytes.items())), tuple(sorted(link_packets.items())), tuple(pivot_bytes)


def _scenario_estimates(
    problem, algorithm, actual_cycles, uncertainty, expert_cycles,
    actual_link_bytes, actual_pivot_bytes, fixed, prologue, epilogue,
    actual_session_peak,
):
    scenarios = {item.kind: item for item in problem.traffic_scenarios}
    actual = scenarios[MoeTrafficScenarioKind.ACTUAL]
    bandwidth = {item.resource_id: item.bytes_per_cycle for item in problem.hardware_facts.route_resources}
    cores_per_die = min(len(cores) for cores in problem.hardware_facts.ordered_cores_by_die)
    result = []
    for kind in sorted(MoeTrafficScenarioKind, key=lambda item: item.value):
        scenario = scenarios[kind]
        per_expert = tuple(
            expert_cycles[index] * scenario.expert_token_counts[index] / max(1, actual.expert_token_counts[index])
            for index in range(len(expert_cycles))
        )
        if kind in (MoeTrafficScenarioKind.ACTUAL, MoeTrafficScenarioKind.P95):
            cycles = actual_cycles
            link_bytes = actual_link_bytes
            pivot_bytes = actual_pivot_bytes
        else:
            link_bytes, link_packets, pivot_bytes = _scenario_network_loads(problem, algorithm, scenario)
            packets = dict(link_packets)
            compute_tail = max(per_expert, default=0.0) / cores_per_die
            link_tail = max((
                value / bandwidth[ref]
                + packets[ref] * (fixed["dte_hop"] + fixed["dte_launch"])
                for ref, value in link_bytes
            ), default=0.0)
            incident = [0] * len(problem.topology.group.placements)
            for source, row in enumerate(scenario.source_expert_token_counts):
                for expert, count in enumerate(row):
                    home = problem.region.semantic_witness.traffic.expert_gemms[expert].rank
                    if source != home:
                        incident[source] += count
                        incident[home] += count
            session_tail = max(incident, default=0) / (cores_per_die * problem.endpoint_session_capacity) * (fixed["session_open"] + fixed["session_retire"])
            cycles = prologue + compute_tail + link_tail + session_tail + epilogue
        result.append(MoeScenarioResourceEstimate(
            scenario=kind, estimated_cycles=float(cycles),
            lower_cycles=float(cycles * (1.0 - uncertainty)),
            upper_cycles=float(cycles * (1.0 + uncertainty)),
            per_expert_compute_cycles=per_expert, per_link_bytes=link_bytes,
            per_pivot_bytes=pivot_bytes,
            max_endpoint_sessions=(actual_session_peak if kind in (MoeTrafficScenarioKind.ACTUAL, MoeTrafficScenarioKind.P95) else cores_per_die * problem.endpoint_session_capacity),
        ))
    return tuple(result)


def build_moe_swizzle_cost(
    problem: MoeSwizzleProblem,
    algorithm: SwizzleAlgorithm,
    actions: tuple[MoeActionWitness, ...],
    packets: tuple[MoePacketWitness, ...],
    calibration_profile: MoeSwizzleCalibrationProfile | None = None,
) -> MoeSwizzleCost:
    problem.validate("moe_cost.problem")
    if algorithm not in problem.allowed_algorithms:
        raise SchemaError("algorithm is outside problem domain", path="moe_cost.algorithm")
    for index, action in enumerate(actions):
        action.validate(f"moe_cost.actions[{index}]")
    for index, packet in enumerate(packets):
        packet.validate(f"moe_cost.packets[{index}]")
    fixed, shapes, swiglu_shapes, uncertainty = _profile_statistics(calibration_profile)
    group_model = _group_model(shapes)
    timings = _earliest_schedule(actions, problem, fixed, group_model, swiglu_shapes)
    traffic = problem.region.semantic_witness.traffic
    flops = sum(item.flops for item in actions if item.kind is SwizzleActionKind.COMP)
    if flops != traffic.expert_gemm_flops or {
        ref for item in actions for ref in item.assignment_refs
    } != set(problem.region.assignment_refs):
        raise SchemaError("candidate changes typed work", path="moe_cost.actions")
    sends = tuple(item for item in actions if item.kind is SwizzleActionKind.SEND)
    routes = {item.id: item for item in problem.topology.group.routes}
    byte_hops = sum(
        item.logical_bytes * (len(routes[item.route_ref].die_path) - 1)
        for item in sends
    )
    root_intervals = {}
    action_index = {action.id: action for action in actions}
    assignment_index = {
        item.id: item for item in traffic.assignments
    }
    root_bindings = build_moe_action_dynamic_root_bindings(
        problem, problem.region.pattern, algorithm, actions,
    )
    for action_ref, key in root_bindings:
        action = action_index[action_ref]
        start, end = timings[action.id]
        extent = max(
            action.logical_bytes,
            sum(assignment_index[ref].payload_bytes for ref in action.assignment_refs),
        )
        prior = root_intervals.get(key)
        root_intervals[key] = (
            start if prior is None else min(start, prior[0]),
            end if prior is None else max(end, prior[1]),
            max(extent, 0 if prior is None else prior[2]),
        )
    roots = len(root_intervals)
    events_by_core = defaultdict(list)
    roots_per_core = defaultdict(int)
    for (runtime_core_id, _, _), (start, end, extent) in root_intervals.items():
        events_by_core[runtime_core_id].extend(
            ((start, extent), (end, -extent))
        )
        roots_per_core[runtime_core_id] += 1
    sram_high_water = 0
    for events in events_by_core.values():
        live = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            live += delta
            sram_high_water = max(sram_high_water, live)
    lifecycle_depth = max(roots_per_core.values(), default=0)
    prologue = lifecycle_depth * (fixed["sram_alloc"] + fixed["sram_bind"])
    steady = max((end for _, end in timings.values()), default=0.0)
    epilogue = lifecycle_depth * fixed["sram_free"] + fixed["terminal_done"]
    estimate = prologue + steady + epilogue
    comp_actions = tuple(item for item in actions if item.kind is SwizzleActionKind.COMP)
    expert_cycles = tuple(
        sum(_action_duration(item, problem, fixed, group_model, swiglu_shapes) for item in comp_actions if item.expert_index == expert)
        for expert in range(len(traffic.expert_gemms))
    )
    link_bytes, pivot_bytes = _actual_loads(problem, actions, packets)
    max_inflight_per_core, max_inflight_per_die = _session_peaks(
        problem, actions, timings
    )
    scenarios = _scenario_estimates(
        problem, algorithm, estimate, uncertainty, expert_cycles,
        link_bytes, pivot_bytes, fixed, prologue, epilogue,
        max_inflight_per_die,
    )
    actual_scenario = next(item for item in scenarios if item.scenario is MoeTrafficScenarioKind.ACTUAL)
    profile = calibration_profile
    measured = profile is not None
    provenance = (
        (None,) * 6
        if profile is None
        else (
            profile.id,
            canonical_digest(profile.samples),
            profile.tool_sha256,
            profile.hardware_sha256,
            profile.simulation_sha256,
            profile.mapping_sha256,
        )
    )
    peak_link = max((value for _, value in link_bytes), default=0)
    max_rank_send = max(
        (sum(item.logical_bytes for item in sends if item.rank == rank) for rank in range(len(expert_cycles))),
        default=0,
    )
    denominator = max(1, traffic.logical_payload_bytes)
    setup, _, _, _ = group_model
    return MoeSwizzleCost.create(
        scenario=MoeTrafficScenarioKind.ACTUAL,
        estimated_cycles=float(estimate),
        critical_path_cycles=float(estimate),
        lower_cycles=actual_scenario.lower_cycles,
        upper_cycles=actual_scenario.upper_cycles,
        prologue_cycles=float(prologue),
        steady_state_cycles=float(steady),
        epilogue_cycles=float(epilogue),
        logical_payload_bytes=traffic.logical_payload_bytes,
        transported_byte_hops=byte_hops,
        expert_gemm_flops=flops,
        region_boundary_output_bytes=traffic.region_boundary_output_bytes,
        packet_count=len(packets) if packets else len(sends),
        descriptor_count=2 * len(sends),
        event_count=2 * sum(item.kind is SwizzleActionKind.BARRIER for item in actions),
        max_inflight=max_inflight_per_core,
        max_link_utilization=float(min(1.0, peak_link / denominator)),
        max_dte_utilization=float(min(1.0, max_rank_send / denominator)),
        compute_idle_cycles=float(max(0.0, steady - sum(expert_cycles))),
        communication_idle_cycles=float(max(0.0, sum(expert_cycles) - steady)),
        sram_high_water_bytes=sram_high_water,
        scenario_estimates=scenarios,
        group_gemm_setup_cycles=float(setup * len(comp_actions)),
        swiglu_group_cycles=float(sum(
            _action_duration(item, problem, fixed, group_model, swiglu_shapes)
            for item in actions if item.kind is SwizzleActionKind.SWIGLU
        )),
        dte_launch_cycles=float(sum(item.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV) for item in actions) * fixed["dte_launch"]),
        dte_sync_cycles=float(sum(item.kind is SwizzleActionKind.WAIT for item in actions) * fixed["dte_sync"]),
        dte_hop_cycles=float(sum((len(routes[item.route_ref].die_path) - 1) * fixed["dte_hop"] for item in sends)),
        sram_lifecycle_cycles=float(prologue + lifecycle_depth * fixed["sram_free"]),
        event_control_cycles=float(sum(item.kind is SwizzleActionKind.BARRIER for item in actions) * (fixed["event_set"] + fixed["event_wait"])),
        endpoint_session_cycles=float(len(sends) * (fixed["session_open"] + fixed["session_retire"])),
        physical_root_count=roots,
        evidence_scale_name=problem.scale_name,
        evidence_scale_role=problem.scale_role,
        calibration_status=(MoeCalibrationStatus.MEASURED if measured else MoeCalibrationStatus.PROVISIONAL),
        calibrated=measured,
        calibration_profile=profile,
        calibration_profile_ref=provenance[0],
        calibration_sample_digest=provenance[1],
        calibration_tool_sha256=provenance[2],
        calibration_hardware_sha256=provenance[3],
        calibration_simulation_sha256=provenance[4],
        calibration_mapping_sha256=provenance[5],
    )


def _compare(left, right):
    for left_value, right_value in (
        (left.cost.estimated_cycles, right.cost.estimated_cycles),
        (left.cost.max_link_utilization, right.cost.max_link_utilization),
        (left.cost.descriptor_count, right.cost.descriptor_count),
        (left.cost.sram_high_water_bytes, right.cost.sram_high_water_bytes),
        (left.id, right.id),
    ):
        if left_value < right_value:
            return -1
        if left_value > right_value:
            return 1
    return 0


def decide_moe_swizzle(problem, baseline, fused_candidates):
    problem.validate("moe_decision.problem")
    candidates = (baseline,) + fused_candidates
    for index, candidate in enumerate(candidates):
        candidate.validate_against(problem, f"moe_decision.candidates[{index}]")
    calibrated_refs = {item.cost.calibration_profile_ref for item in candidates if item.cost.calibrated}
    if calibrated_refs and (
        len(calibrated_refs) != 1 or not all(item.cost.calibrated for item in candidates)
    ):
        raise SchemaError("decision candidates require one exact calibration profile", path="moe_decision.candidates")
    ranked_fused = tuple(sorted(fused_candidates, key=cmp_to_key(_compare)))
    if not fused_candidates or not all(item.cost.calibrated for item in candidates):
        ranked = (baseline,) + ranked_fused
        reason = SwizzleDecisionReason.BASELINE_ONLY if not fused_candidates else SwizzleDecisionReason.NO_PROFITABLE_FUSION
        performance_complete = False
    else:
        ranked_all = tuple(sorted(candidates, key=cmp_to_key(_compare)))
        best = ranked_all[0]
        if best.cost.estimated_cycles >= baseline.cost.estimated_cycles:
            ranked = (baseline,) + tuple(item for item in ranked_all if item.id != baseline.id)
            reason = SwizzleDecisionReason.NO_PROFITABLE_FUSION
        else:
            ranked = ranked_all
            reason = SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES
        performance_complete = True
    return MoeSwizzleDecision.create(
        problem=problem,
        baseline=baseline,
        ranked_candidates=ranked,
        selected_candidate_ref=ranked[0].id,
        decision_reason=reason,
        performance_complete=performance_complete,
    )


_CANDIDATE_LIFECYCLE_FLOOR_CACHE: dict[str, tuple[MoeCandidateCoreLifecycleFloor, ...]] = {}


def _candidate_lifecycle_floors(
    decision: MoeSwizzleDecision,
) -> tuple[MoeCandidateCoreLifecycleFloor, ...]:
    cached = _CANDIDATE_LIFECYCLE_FLOOR_CACHE.get(decision.id)
    if cached is not None:
        return cached
    result = tuple(
        item
        for candidate in decision.ranked_candidates
        for item in build_moe_candidate_core_lifecycle_floor(
            decision.problem, candidate,
        )
    )
    result = tuple(sorted(
        result,
        key=lambda item: (item.candidate_ref, item.runtime_core_id),
    ))
    _CANDIDATE_LIFECYCLE_FLOOR_CACHE[decision.id] = result
    return result


def _pair_lifecycle_floor(
    pair: tuple[str, str],
    dispatch_floors: tuple[MoeCandidateCoreLifecycleFloor, ...],
    combine_floors: tuple[MoeCandidateCoreLifecycleFloor, ...],
    alloc_cycles: float,
    bind_cycles: float,
    free_cycles: float,
) -> float:
    counts = defaultdict(lambda: [0, 0, 0])
    for item in dispatch_floors + combine_floors:
        if item.candidate_ref in pair:
            values = counts[item.runtime_core_id]
            values[0] += item.alloc_count
            values[1] += item.bind_count
            values[2] += item.free_count
    return float(max(
        values[0] * alloc_cycles
        + values[1] * bind_cycles
        + values[2] * free_cycles
        for values in counts.values()
    ))


def _compute_moe_swizzle_pair_cost_context_axes(
    dispatch_decision: MoeSwizzleDecision,
    combine_decision: MoeSwizzleDecision,
    dispatch_floors: tuple[MoeCandidateCoreLifecycleFloor, ...],
    combine_floors: tuple[MoeCandidateCoreLifecycleFloor, ...],
    fixed_cycles: tuple[float, float, float, float] | None,
) -> tuple[
    tuple[tuple[tuple[str, str], float], ...],
    tuple[tuple[tuple[str, str], float], ...],
]:
    bounds = []
    floor_cycles = []
    for left in dispatch_decision.ranked_candidates:
        for right in combine_decision.ranked_candidates:
            pair = (left.id, right.id)
            total = left.cost.estimated_cycles + right.cost.estimated_cycles
            floor = 0.0
            if fixed_cycles is not None:
                alloc, bind, free, terminal = fixed_cycles
                floor = _pair_lifecycle_floor(
                    pair, dispatch_floors, combine_floors, alloc, bind, free,
                )
                total = (
                    total - left.cost.sram_lifecycle_cycles
                    - right.cost.sram_lifecycle_cycles - terminal + floor
                )
            bounds.append((pair, float(total)))
            floor_cycles.append((pair, float(floor)))
    return (
        tuple(sorted(bounds, key=lambda item: (item[1], item[0]))),
        tuple(sorted(floor_cycles)),
    )


@dataclass(frozen=True, slots=True)
class MoeSwizzlePairCostContext:
    source_dispatch_decision_id: str
    source_combine_decision_id: str
    bounds: tuple[tuple[tuple[str, str], float], ...]
    lifecycle_floor_cycles: tuple[tuple[tuple[str, str], float], ...]
    lifecycle_fixed_cycles: tuple[float, float, float, float] | None

    def validate_against(
        self,
        dispatch_decision: MoeSwizzleDecision,
        combine_decision: MoeSwizzleDecision,
        path: str = "moe_pair_context",
    ) -> None:
        if (
            self.source_dispatch_decision_id != dispatch_decision.id
            or self.source_combine_decision_id != combine_decision.id
        ):
            raise SchemaError("pair context source decision lineage disagrees", path=path)
        if dispatch_decision.performance_complete != combine_decision.performance_complete:
            raise SchemaError("pair context mixes decision completion states", path=path)
        measured = dispatch_decision.performance_complete
        if measured:
            if (
                type(self.lifecycle_fixed_cycles) is not tuple
                or len(self.lifecycle_fixed_cycles) != 4
                or any(
                    type(value) is not float
                    or not math.isfinite(value) or value < 0.0
                    for value in self.lifecycle_fixed_cycles
                )
            ):
                raise SchemaError("measured pair context requires nonnegative fixed cycles", path=path)
        elif self.lifecycle_fixed_cycles is not None:
            raise SchemaError("provisional pair context cannot claim fixed cycles", path=path)
        dispatch_floors = _candidate_lifecycle_floors(dispatch_decision)
        combine_floors = _candidate_lifecycle_floors(combine_decision)
        expected_bounds, expected_floors = _compute_moe_swizzle_pair_cost_context_axes(
            dispatch_decision, combine_decision,
            dispatch_floors, combine_floors,
            self.lifecycle_fixed_cycles,
        )
        if self.bounds != expected_bounds:
            raise SchemaError(
                "pair context bounds disagree with typed costs/floors",
                path=f"{path}.bounds",
            )
        if self.lifecycle_floor_cycles != expected_floors:
            raise SchemaError(
                "pair context lifecycle floors disagree with typed candidates",
                path=f"{path}.lifecycle_floor_cycles",
            )


def build_moe_swizzle_pair_cost_context(
    dispatch_decision: MoeSwizzleDecision,
    combine_decision: MoeSwizzleDecision,
) -> MoeSwizzlePairCostContext:
    """Validate both decisions once and freeze full-grid cost/floor axes."""

    dispatch_decision.validate("moe_pair_context.dispatch")
    combine_decision.validate("moe_pair_context.combine")
    if dispatch_decision.performance_complete != combine_decision.performance_complete:
        raise SchemaError(
            "pair context mixes decision completion states",
            path="moe_pair_context",
        )
    fixed_cycles = None
    if dispatch_decision.performance_complete:
        profiles = {
            item.cost.calibration_profile
            for decision in (dispatch_decision, combine_decision)
            for item in decision.ranked_candidates
        }
        if len(profiles) != 1 or None in profiles:
            raise SchemaError(
                "pair context requires one measured profile",
                path="moe_pair_context",
            )
        fixed, _, _, _ = _profile_statistics(next(iter(profiles)))
        fixed_cycles = (
            float(fixed["sram_alloc"]), float(fixed["sram_bind"]),
            float(fixed["sram_free"]), float(fixed["terminal_done"]),
        )
    dispatch_floors = _candidate_lifecycle_floors(dispatch_decision)
    combine_floors = _candidate_lifecycle_floors(combine_decision)
    bounds, floor_cycles = _compute_moe_swizzle_pair_cost_context_axes(
        dispatch_decision, combine_decision,
        dispatch_floors, combine_floors, fixed_cycles,
    )
    result = MoeSwizzlePairCostContext(
        dispatch_decision.id, combine_decision.id,
        bounds, floor_cycles, fixed_cycles,
    )
    result.validate_against(dispatch_decision, combine_decision)
    return result


def build_moe_swizzle_pair_cost_lower_bounds(
    dispatch_decision: MoeSwizzleDecision,
    combine_decision: MoeSwizzleDecision,
) -> tuple[tuple[tuple[str, str], float], ...]:
    """Return the full-grid lower bounds from one validated typed context."""

    return build_moe_swizzle_pair_cost_context(
        dispatch_decision, combine_decision,
    ).bounds


def estimate_moe_swizzle_materialized_pair_cycles(
    dispatch_decision: MoeSwizzleDecision,
    combine_decision: MoeSwizzleDecision,
    witness: MoeWholePairFeasibility,
    *,
    context: MoeSwizzlePairCostContext | None = None,
) -> float:
    """Replace the typed lifecycle floor with exact materialized per-core cost."""

    witness.validate("moe_pair_cycles.witness")
    if context is None:
        context = build_moe_swizzle_pair_cost_context(
            dispatch_decision, combine_decision,
        )
    if type(context) is not MoeSwizzlePairCostContext:
        raise SchemaError("requires exact pair cost context", path="moe_pair_cycles.context")
    context.validate_against(dispatch_decision, combine_decision)
    bounds = dict(context.bounds)
    floors = dict(context.lifecycle_floor_cycles)
    if witness.candidate_refs not in bounds or witness.candidate_refs not in floors:
        raise SchemaError("pair witness is outside cost axes", path="moe_pair_cycles")
    if not witness.feasible:
        return math.inf
    if context.lifecycle_fixed_cycles is None:
        return bounds[witness.candidate_refs]
    alloc, bind, free, _ = context.lifecycle_fixed_cycles
    lifecycle = max(
        item.alloc_count * alloc
        + item.bind_count * bind
        + item.free_count * free
        for item in witness.core_lifecycle_counts
    )
    return float(
        bounds[witness.candidate_refs]
        - floors[witness.candidate_refs] + lifecycle
    )


def select_moe_swizzle_workload_deployment(
    dispatch_decision: MoeSwizzleDecision,
    combine_decision: MoeSwizzleDecision,
    pair_feasibilities: tuple[MoeWholePairFeasibility, ...],
) -> MoeSwizzleWorkloadSelection:
    """Choose the lowest-cost pair admitted by exact whole-resource witnesses."""

    dispatch_decision.validate("moe_joint_selection.dispatch_decision")
    combine_decision.validate("moe_joint_selection.combine_decision")
    if (
        dispatch_decision.problem.region.pattern is not FusionPattern.MOE_DISPATCH_GEMM
        or combine_decision.problem.region.pattern is not FusionPattern.MOE_GEMM_COMBINE
        or dispatch_decision.problem.source_execution_id
        != combine_decision.problem.source_execution_id
    ):
        raise SchemaError("joint decisions do not form one typed workload", path="moe_joint_selection")
    dispatch_candidates = {
        item.id: item for item in dispatch_decision.ranked_candidates
    }
    combine_candidates = {
        item.id: item for item in combine_decision.ranked_candidates
    }
    if type(pair_feasibilities) is not tuple or not pair_feasibilities:
        raise SchemaError(
            "joint selection requires whole-projection pair witnesses",
            path="moe_joint_selection.pair_feasibilities",
        )
    witnesses = tuple(sorted(
        pair_feasibilities,
        key=lambda item: item.candidate_refs,
    ))
    actual_pairs = set()
    witness_by_pair = {}
    for index, witness in enumerate(witnesses):
        if type(witness) is not MoeWholePairFeasibility:
            raise SchemaError(
                "requires exact whole pair witness",
                path=f"moe_joint_selection.pair_feasibilities[{index}]",
            )
        witness.validate(f"moe_joint_selection.pair_feasibilities[{index}]")
        pair = witness.candidate_refs
        actual_pairs.add(pair)
        dispatch = dispatch_candidates.get(pair[0])
        combine = combine_candidates.get(pair[1])
        if (
            dispatch is None or combine is None
            or witness.dynamic_sram_capacity_bytes
            != dispatch_decision.problem.sram_capacity_bytes
            or (
                witness.placement.feasible
                and witness.endpoint.capacity_per_core
                != dispatch_decision.problem.endpoint_session_capacity
            )
        ):
            raise SchemaError(
                "whole pair witness disagrees with candidate cost/hardware",
                path=f"moe_joint_selection.pair_feasibilities[{index}]",
            )
        witness_by_pair[pair] = witness
    if len(actual_pairs) != len(witnesses):
        raise SchemaError(
            "whole pair witnesses must have unique candidate keys",
            path="moe_joint_selection.pair_feasibilities",
        )
    dispatch_costs = tuple(sorted(
        (ref, float(candidate.cost.estimated_cycles))
        for ref, candidate in dispatch_candidates.items()
    ))
    combine_costs = tuple(sorted(
        (ref, float(candidate.cost.estimated_cycles))
        for ref, candidate in combine_candidates.items()
    ))
    dispatch_cycles = dict(dispatch_costs)
    combine_cycles = dict(combine_costs)
    dispatch_lifecycle_costs = tuple(sorted(
        (ref, float(candidate.cost.sram_lifecycle_cycles))
        for ref, candidate in dispatch_candidates.items()
    ))
    combine_lifecycle_costs = tuple(sorted(
        (ref, float(candidate.cost.sram_lifecycle_cycles))
        for ref, candidate in combine_candidates.items()
    ))
    dispatch_lifecycle = dict(dispatch_lifecycle_costs)
    combine_lifecycle = dict(combine_lifecycle_costs)
    dispatch_lifecycle_floors = _candidate_lifecycle_floors(dispatch_decision)
    combine_lifecycle_floors = _candidate_lifecycle_floors(combine_decision)
    performance_complete = (
        dispatch_decision.performance_complete
        and combine_decision.performance_complete
    )
    lifecycle_fixed_cycles = None
    calibration_profile_ref = None
    calibration_sample_digest = None
    if performance_complete:
        profiles = {
            candidate.cost.calibration_profile
            for candidate in tuple(dispatch_candidates.values())
            + tuple(combine_candidates.values())
        }
        if len(profiles) != 1 or None in profiles:
            raise SchemaError(
                "joint decisions require one exact measured calibration profile",
                path="moe_joint_selection",
            )
        profile = next(iter(profiles))
        fixed, _, _, _ = _profile_statistics(profile)
        lifecycle_fixed_cycles = (
            float(fixed["sram_alloc"]), float(fixed["sram_bind"]),
            float(fixed["sram_free"]), float(fixed["terminal_done"]),
        )
        calibration_profile_ref = profile.id
        calibration_sample_digest = canonical_digest(profile.samples)
    def pair_lower(pair):
        total = dispatch_cycles[pair[0]] + combine_cycles[pair[1]]
        if lifecycle_fixed_cycles is None:
            return float(total)
        alloc, bind, free, terminal = lifecycle_fixed_cycles
        return float(
            total - dispatch_lifecycle[pair[0]]
            - combine_lifecycle[pair[1]] - terminal
            + _pair_lifecycle_floor(
                pair, dispatch_lifecycle_floors,
                combine_lifecycle_floors, alloc, bind, free,
            )
        )
    def pair_actual(pair):
        witness = witness_by_pair[pair]
        if not witness.feasible:
            return math.inf
        if lifecycle_fixed_cycles is None:
            return pair_lower(pair)
        alloc, bind, free, _ = lifecycle_fixed_cycles
        floor = _pair_lifecycle_floor(
            pair, dispatch_lifecycle_floors,
            combine_lifecycle_floors, alloc, bind, free,
        )
        lifecycle = max(
            item.alloc_count * alloc
            + item.bind_count * bind
            + item.free_count * free
            for item in witness.core_lifecycle_counts
        )
        return float(pair_lower(pair) - floor + lifecycle)
    pair_order = tuple(sorted(
        (
            (dispatch_ref, combine_ref)
            for dispatch_ref in dispatch_candidates
            for combine_ref in combine_candidates
        ),
        key=lambda pair: (
            pair_lower(pair), pair,
        ),
    ))
    baseline_key = (
        dispatch_decision.baseline.id, combine_decision.baseline.id,
    )
    lower_bound_key = pair_order[0]
    dispatch_fused_ref = min(
        (ref for ref in dispatch_candidates if ref != baseline_key[0]),
        key=lambda ref: (dispatch_cycles[ref], ref),
        default=baseline_key[0],
    )
    combine_fused_ref = min(
        (ref for ref in combine_candidates if ref != baseline_key[1]),
        key=lambda ref: (combine_cycles[ref], ref),
        default=baseline_key[1],
    )
    mode_pairs = {
        (dispatch_ref, combine_ref)
        for dispatch_ref in (baseline_key[0], dispatch_fused_ref)
        for combine_ref in (baseline_key[1], combine_fused_ref)
    }
    def pair_cycles(pair):
        return pair_actual(pair)
    if not performance_complete:
        frontier = ()
        required_pairs = mode_pairs
        if actual_pairs != required_pairs:
            raise SchemaError(
                "provisional witnesses must exactly cover baseline/argmin modes",
                path="moe_joint_selection.pair_feasibilities",
            )
        baseline = witness_by_pair[baseline_key]
        if not baseline.feasible:
            raise SchemaError(
                "provisional whole baseline is not deployable",
                path="moe_joint_selection",
            )
        selected = baseline
        reason = SwizzleDecisionReason.NO_PROFITABLE_FUSION
    else:
        frontier_items = []
        best = None
        best_cycles = math.inf
        for pair in pair_order:
            if pair_lower(pair) >= best_cycles:
                break
            witness = witness_by_pair.get(pair)
            if witness is None:
                raise SchemaError(
                    "missing exact witness in cost-ordered feasibility frontier",
                    path="moe_joint_selection.pair_feasibilities",
                )
            frontier_items.append(witness)
            if witness.feasible:
                actual = pair_actual(pair)
                if (actual, pair) < (
                    best_cycles,
                    ("\uffff", "\uffff") if best is None else best.candidate_refs,
                ):
                    best = witness
                    best_cycles = actual
        if best is None:
            raise SchemaError(
                "no whole-workload candidate pair is deployable",
                path="moe_joint_selection",
            )
        frontier = tuple(item.candidate_refs for item in frontier_items)
        frontier_stop_lower_bound = (
            None
            if len(frontier) == len(pair_order)
            else pair_lower(pair_order[len(frontier)])
        )
        required_pairs = mode_pairs | set(frontier)
        if actual_pairs != required_pairs:
            raise SchemaError(
                "measured witnesses must exactly cover mode fallbacks and ordered frontier",
                path="moe_joint_selection.pair_feasibilities",
            )
        baseline = witness_by_pair[baseline_key]
        if pair_cycles(best.candidate_refs) < pair_cycles(baseline_key):
            selected = best
            reason = SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES
        elif baseline.feasible:
            selected = baseline
            reason = SwizzleDecisionReason.NO_PROFITABLE_FUSION
        else:
            raise SchemaError(
                "no feasible profitable whole-workload pair",
                path="moe_joint_selection",
            )
    selected_key = selected.candidate_refs
    ordered = tuple(sorted(
        witnesses,
        key=lambda item: (
            not item.feasible, pair_cycles(item.candidate_refs), item.candidate_refs,
        ),
    ))
    ranked = (selected_key,) + tuple(
        item.candidate_refs
        for item in ordered
        if item.candidate_refs != selected_key
    )
    return MoeSwizzleWorkloadSelection.create(
        source_dispatch_decision_id=dispatch_decision.id,
        source_combine_decision_id=combine_decision.id,
        dispatch_candidate_costs=dispatch_costs,
        combine_candidate_costs=combine_costs,
        dispatch_candidate_lifecycle_costs=dispatch_lifecycle_costs,
        combine_candidate_lifecycle_costs=combine_lifecycle_costs,
        dispatch_candidate_lifecycle_floors=dispatch_lifecycle_floors,
        combine_candidate_lifecycle_floors=combine_lifecycle_floors,
        lifecycle_fixed_cycles=lifecycle_fixed_cycles,
        calibration_profile_ref=calibration_profile_ref,
        calibration_sample_digest=calibration_sample_digest,
        baseline_pair_ref=baseline_key,
        lower_bound_pair_ref=lower_bound_key,
        frontier_pair_refs=frontier,
        frontier_stop_lower_bound=(
            frontier_stop_lower_bound if performance_complete else None
        ),
        pair_feasibilities=witnesses,
        ranked_pair_refs=ranked,
        selected_dispatch_candidate_ref=selected.candidate_refs[0],
        selected_combine_candidate_ref=selected.candidate_refs[1],
        baseline_estimated_cycles=pair_cycles(baseline_key),
        selected_estimated_cycles=pair_cycles(selected_key),
        decision_reason=reason,
        performance_complete=performance_complete,
    )


__all__ = [
    "build_moe_action_owner_map", "build_moe_swizzle_cost",
    "MoeSwizzlePairCostContext", "build_moe_swizzle_pair_cost_context",
    "build_moe_swizzle_pair_cost_lower_bounds",
    "estimate_moe_swizzle_materialized_pair_cycles",
    "decide_moe_swizzle", "select_moe_swizzle_workload_deployment",
]
