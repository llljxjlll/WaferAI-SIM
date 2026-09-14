#!/usr/bin/env python3
"""Resource-explicit end-to-end analytical replay for exp2-1.

The module intentionally keeps calibration provenance separate from the
analytical model.  With no validated target-binding anchors it reports
``analytical_resource_dag_extrapolation``; it never upgrades itself merely
because an old simulator result exists.

Base, forward-only overlap and full-train overlap are built from identical
actions. Only dependency edges differ. The full-train state adds legal backward
dX/WGRAD/gradient-collective overlap while preserving the optimizer's
all-gradients-ready barrier.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import heapq
import json
import math
from typing import Iterable, Mapping, Sequence

from placements import (
    HBM_STACKS,
    INFERENCE_INSTANCES,
    PD_HANDOFFS,
    TRAINING_GROUPS,
    Placement,
    directed_route_resources,
    placement_by_name,
)


CLOCK_HZ = 5.0e8
TENSOR_FLOPS_PER_DIE = 128.0e12
TENSOR_EFFICIENCY_PRIOR = 0.72
VECTOR_FLOPS_PER_DIE_PRIOR = 8.0e12
VECTOR_EFFICIENCY_PRIOR = 0.60
D2D_LINK_BPS = 1.0e12
LOCAL_NOC_BPS = 256.0e9
DTE_CHANNEL_BPS = 128.0e9
HBM_STACK_BPS = 256.0e9
CONTROL_ISSUE_CYCLES = 12.0
REDUCER_BYTES_PER_SECOND = 256.0e9
DTE_GAMMA_NS = 2.0
DTE_TAU_LAUNCH_NS = 1.0
DTE_LAUNCH_CYCLES = math.ceil(DTE_GAMMA_NS / 2.0) + math.ceil(DTE_TAU_LAUNCH_NS / 2.0)
D2D_HOP_LATENCY_CYCLES = 1.0
HBM_FIRST_BYTE_NS = 20.0
HBM_FIRST_BYTE_CYCLES = HBM_FIRST_BYTE_NS / 2.0
DTYPE_BYTES = 2
FP32_BYTES = 4
E2E_DAG_VERSION = "exp2_resource_explicit_e2e_v4_decode_fidelity"
FORWARD_WINDOWS = 8
TENSOR_TILE_M = 128
DECODE_UNROLL_STEPS = 2


def _cycles_for_work(work: float, rate_per_second: float, setup: float = 0.0) -> float:
    if work < 0 or rate_per_second <= 0:
        raise ValueError("work must be non-negative and rate must be positive")
    return setup + work / rate_per_second * CLOCK_HZ


def _manifest_value(manifest: Mapping[str, object], name: str, default: object | None = None) -> object:
    if name in manifest:
        return manifest[name]
    if default is not None:
        return default
    raise ValueError(f"model manifest is missing required field: {name}")


def _integer(manifest: Mapping[str, object], name: str, default: int | None = None) -> int:
    value = _manifest_value(manifest, name, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Action:
    action_id: str
    phase: str
    layer: int
    operator: str
    tile_id: int
    deps: tuple[str, ...]
    logical_work: float
    runtime_work: float
    bytes: float
    route: tuple[str, ...]
    resource_set: tuple[str, ...]
    duration_cycles: float
    duration_source: str
    evidence_signature: str

    def invariant_tuple(self) -> tuple[object, ...]:
        """Identity used to prove base/overlap have exactly the same work."""

        return (
            self.action_id,
            self.phase,
            self.layer,
            self.operator,
            self.tile_id,
            self.logical_work,
            self.runtime_work,
            self.bytes,
            self.route,
            self.resource_set,
            self.duration_cycles,
            self.duration_source,
            self.evidence_signature,
        )

    def manifest_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "phase": self.phase,
            "layer": self.layer,
            "operator": self.operator,
            "tile_id": self.tile_id,
            "deps": list(self.deps),
            "logical_work": self.logical_work,
            "runtime_work": self.runtime_work,
            "bytes": self.bytes,
            "route": list(self.route),
            "resource_set": list(self.resource_set),
            "duration_cycles": self.duration_cycles,
            "duration_source": self.duration_source,
            "evidence_signature": self.evidence_signature,
        }


@dataclass(frozen=True, slots=True)
class ReplayPair:
    base_actions: tuple[Action, ...]
    overlap_actions: tuple[Action, ...]
    workload: str
    full_train_actions: tuple[Action, ...] | None = None

    def __post_init__(self) -> None:
        assert_same_work(self.base_actions, self.overlap_actions)
        if self.full_train_actions is not None:
            assert_same_work(self.base_actions, self.full_train_actions)


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    makespan_cycles: float
    starts: Mapping[str, float]
    finishes: Mapping[str, float]
    phase_cycles: Mapping[str, float]
    resource_service_cycles: Mapping[str, float]
    longest_dependency_path_cycles: float
    theory_lower_cycles: float
    phase_theory_lower_cycles: Mapping[str, float]


def assert_same_work(base_actions: Sequence[Action], overlap_actions: Sequence[Action]) -> None:
    if len(base_actions) != len(overlap_actions):
        raise AssertionError("base/overlap action count differs")
    base = {action.action_id: action for action in base_actions}
    overlap = {action.action_id: action for action in overlap_actions}
    if base.keys() != overlap.keys():
        raise AssertionError("base/overlap action identity differs")
    for action_id in base:
        if base[action_id].invariant_tuple() != overlap[action_id].invariant_tuple():
            raise AssertionError(f"base/overlap work differs at {action_id}")


def replay(actions: Sequence[Action]) -> ScheduleResult:
    """Earliest-resource-ready deterministic discrete-event replay."""

    action_by_id = {action.action_id: action for action in actions}
    if len(action_by_id) != len(actions):
        raise ValueError("action ids must be unique")
    order = {action.action_id: index for index, action in enumerate(actions)}
    indegree = {action.action_id: len(action.deps) for action in actions}
    dependents: dict[str, list[str]] = defaultdict(list)
    for action in actions:
        for dependency in action.deps:
            if dependency not in action_by_id:
                raise ValueError(f"unknown dependency: {dependency}")
            dependents[dependency].append(action.action_id)
    ready = [(order[action_id], action_id) for action_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    resource_ready: dict[str, float] = defaultdict(float)
    resource_service: dict[str, float] = defaultdict(float)
    starts: dict[str, float] = {}
    finishes: dict[str, float] = {}
    dependency_finish: dict[str, float] = {}
    phase_dependency_finish: dict[str, float] = {}
    phase_resource_service: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    while ready:
        _, action_id = heapq.heappop(ready)
        action = action_by_id[action_id]
        dep_ready = max((finishes[item] for item in action.deps), default=0.0)
        resource_time = max((resource_ready[item] for item in action.resource_set), default=0.0)
        start = max(dep_ready, resource_time)
        finish = start + action.duration_cycles
        starts[action_id] = start
        finishes[action_id] = finish
        dependency_finish[action_id] = max(
            (dependency_finish[item] for item in action.deps), default=0.0
        ) + action.duration_cycles
        phase_dependency_finish[action_id] = max(
            (
                phase_dependency_finish[item]
                for item in action.deps
                if action_by_id[item].phase == action.phase
            ),
            default=0.0,
        ) + action.duration_cycles
        for resource in action.resource_set:
            resource_ready[resource] = finish
            resource_service[resource] += action.duration_cycles
            phase_resource_service[action.phase][resource] += action.duration_cycles
        for dependent in dependents[action_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, (order[dependent], dependent))
    if len(finishes) != len(actions):
        raise ValueError("action graph contains a dependency cycle")

    phase_bounds: dict[str, list[float]] = {}
    for action in actions:
        bounds = phase_bounds.setdefault(action.phase, [math.inf, 0.0])
        bounds[0] = min(bounds[0], starts[action.action_id])
        bounds[1] = max(bounds[1], finishes[action.action_id])
    phase_cycles = {phase: end - start for phase, (start, end) in phase_bounds.items()}
    dependency_lower = max(dependency_finish.values(), default=0.0)
    resource_lower = max(resource_service.values(), default=0.0)
    phase_lower = {
        phase: max(
            max(
                (phase_dependency_finish[action.action_id] for action in actions if action.phase == phase),
                default=0.0,
            ),
            max(phase_resource_service[phase].values(), default=0.0),
        )
        for phase in phase_bounds
    }
    return ScheduleResult(
        makespan_cycles=max(finishes.values(), default=0.0),
        starts=starts,
        finishes=finishes,
        phase_cycles=phase_cycles,
        resource_service_cycles=dict(resource_service),
        longest_dependency_path_cycles=dependency_lower,
        theory_lower_cycles=max(dependency_lower, resource_lower),
        phase_theory_lower_cycles=phase_lower,
    )


class _PairBuilder:
    def __init__(self) -> None:
        self.base: list[Action] = []
        self.overlap: list[Action] = []
        self.full_train: list[Action] = []

    def add(
        self,
        *,
        action_id: str,
        phase: str,
        layer: int,
        operator: str,
        tile_id: int,
        base_deps: Iterable[str],
        overlap_deps: Iterable[str],
        full_train_deps: Iterable[str] | None = None,
        logical_work: float = 0.0,
        runtime_work: float | None = None,
        bytes: float = 0.0,
        route: Iterable[str] = (),
        resources: Iterable[str],
        duration_cycles: float,
        duration_source: str = "analytical_target_rate_prior",
        evidence_signature: str = "unanchored",
    ) -> str:
        resources_tuple = tuple(dict.fromkeys(resources))
        if not resources_tuple:
            raise ValueError(f"action {action_id} has no resource")
        common = Action(
            action_id=action_id,
            phase=phase,
            layer=layer,
            operator=operator,
            tile_id=tile_id,
            deps=(),
            logical_work=float(logical_work),
            runtime_work=float(logical_work if runtime_work is None else runtime_work),
            bytes=float(bytes),
            route=tuple(route),
            resource_set=resources_tuple,
            duration_cycles=max(1.0, float(duration_cycles)),
            duration_source=duration_source,
            evidence_signature=evidence_signature,
        )
        base_tuple = tuple(dict.fromkeys(base_deps))
        overlap_tuple = tuple(dict.fromkeys(overlap_deps))
        full_tuple = (
            overlap_tuple
            if full_train_deps is None
            else tuple(dict.fromkeys(full_train_deps))
        )
        self.base.append(replace(common, deps=base_tuple))
        self.overlap.append(replace(common, deps=overlap_tuple))
        self.full_train.append(replace(common, deps=full_tuple))
        return action_id

    def pair(self, workload: str, *, include_full_train: bool = False) -> ReplayPair:
        return ReplayPair(
            tuple(self.base),
            tuple(self.overlap),
            workload,
            tuple(self.full_train) if include_full_train else None,
        )


def _attention_is_mla(manifest: Mapping[str, object]) -> bool:
    return "mla" in str(manifest.get("attention_type", "")).lower()


def _is_moe_layer(manifest: Mapping[str, object], layer: int) -> bool:
    mlp_type = str(manifest.get("mlp_type", "dense")).lower()
    if "moe" not in mlp_type and int(manifest.get("routed_expert_count", 0) or 0) == 0:
        return False
    frequency = max(1, int(manifest.get("moe_layer_frequency", 1) or 1))
    default_dense_prefix = 3 if "deepseek" in str(manifest.get("model_id", "")).lower() else 0
    dense_prefix = max(0, int(
        manifest.get("dense_layer_prefix", manifest.get("first_k_dense_replace", default_dense_prefix))
        or 0
    ))
    return layer >= dense_prefix and (layer - dense_prefix) % frequency == 0


def _kv_cache_width(manifest: Mapping[str, object]) -> int:
    """Per-token K/V state width; MLA uses its compressed cache format."""

    if _attention_is_mla(manifest):
        dimensions = manifest.get("mla_dimensions", {})
        if not isinstance(dimensions, Mapping):
            dimensions = {}
        explicit = manifest.get("mla_cache_width")
        if explicit is not None:
            width = int(explicit)
        else:
            width = int(manifest.get("kv_lora_rank", dimensions.get("kv_lora_rank", 0)) or 0) + int(
                manifest.get("qk_rope_head_dim", dimensions.get("qk_rope_head_dim", 0)) or 0
            )
        if width <= 0:
            raise ValueError(
                "MLA manifest requires mla_cache_width or "
                "kv_lora_rank + qk_rope_head_dim"
            )
        return width
    heads = _integer(manifest, "num_attention_heads", 1)
    return _integer(manifest, "num_kv_heads", heads) * _integer(manifest, "head_dim")


def _layer_work(
    manifest: Mapping[str, object],
    tokens: int,
    layer: int,
    *,
    decode_kv: int = 0,
    include_shared_experts: bool = False,
) -> dict[str, float]:
    hidden = _integer(manifest, "hidden_size")
    heads = _integer(manifest, "num_attention_heads", 1)
    kv_heads = _integer(manifest, "num_kv_heads", heads)
    if _attention_is_mla(manifest):
        dimensions = manifest.get("mla_dimensions", {})
        if not isinstance(dimensions, Mapping):
            dimensions = {}
        qk_nope = int(manifest.get("qk_nope_head_dim", dimensions.get("qk_nope_head_dim", 0)) or 0)
        qk_rope = int(manifest.get("qk_rope_head_dim", dimensions.get("qk_rope_head_dim", 0)) or 0)
        head_dim = qk_nope + qk_rope
        if head_dim <= 0:
            raise ValueError("MLA manifest lacks qk_nope/qk_rope head dimensions")
    else:
        head_dim = _integer(manifest, "head_dim", hidden // heads)
    top_k = max(1, int(manifest.get("top_k", 1) or 1))
    shared = (
        max(0, int(manifest.get("shared_expert_count", 0) or 0))
        if include_shared_experts
        else 0
    )
    routed_experts = max(0, int(manifest.get("routed_expert_count", 0) or 0))
    moe = _is_moe_layer(manifest, layer)
    if moe:
        intermediate = int(
            manifest.get("expert_intermediate_size", manifest.get("moe_intermediate_size", manifest.get("intermediate_size", 0)))
            or 0
        )
    else:
        intermediate = int(manifest.get("dense_intermediate_size", manifest.get("intermediate_size", 0)) or 0)
    if intermediate <= 0:
        raise ValueError("manifest lacks the dense/expert intermediate size for this layer")

    kv_width = kv_heads * head_dim
    qkv = 2.0 * tokens * hidden * (hidden + 2 * kv_width)
    out = 2.0 * tokens * hidden * hidden
    if decode_kv:
        attention = 4.0 * heads * tokens * decode_kv * head_dim
    else:
        attention = 4.0 * heads * tokens * tokens * head_dim
    if _attention_is_mla(manifest):
        # Separate analytical MLA path; never relabelled as standard MHA/GQA.
        latent = int(manifest.get("kv_lora_rank", dimensions.get("kv_lora_rank", max(head_dim, hidden // 8))) or max(head_dim, hidden // 8))
        q_rank = int(manifest.get("q_lora_rank", dimensions.get("q_lora_rank", latent)) or latent)
        qkv = 2.0 * tokens * hidden * (q_rank + latent) + 2.0 * tokens * q_rank * heads * head_dim
        attention = 4.0 * tokens * max(tokens, decode_kv) * heads * head_dim
    mlp_type = str(manifest.get("mlp_type", "")).lower()
    matrix_count = 2 if (not moe and mlp_type == "dense_gelu") else 3
    flop_factor = 2 * matrix_count
    routed_mlp = (
        float(flop_factor) * tokens * hidden * intermediate * top_k
        if moe
        else 0.0
    )
    shared_mlp = (
        float(flop_factor) * tokens * hidden * intermediate * shared
        if moe
        else 0.0
    )
    mlp = routed_mlp + shared_mlp if moe else (
        float(flop_factor) * tokens * hidden * intermediate
    )
    vector = float(tokens * hidden * (16 + 3 * top_k if moe else 16))
    attention_weight_bytes = (
        hidden * (hidden + 2 * kv_width) + hidden * hidden
    ) * DTYPE_BYTES
    if moe:
        touched_routed_experts = min(routed_experts, tokens * top_k)
        routed_weight_bytes = (
            touched_routed_experts * 3 * hidden * intermediate * DTYPE_BYTES
        )
        shared_weight_bytes = shared * 3 * hidden * intermediate * DTYPE_BYTES
        non_routed_weight_bytes = attention_weight_bytes + shared_weight_bytes
        parameter_bytes = non_routed_weight_bytes + routed_weight_bytes
    else:
        touched_routed_experts = 0
        routed_weight_bytes = 0
        non_routed_weight_bytes = attention_weight_bytes + (
            matrix_count * hidden * intermediate * DTYPE_BYTES
        )
        parameter_bytes = non_routed_weight_bytes
    activation_bytes = tokens * hidden * DTYPE_BYTES
    dense_tensor_flops = qkv + out + attention + (
        shared_mlp if moe else mlp
    )
    expert_tensor_flops = routed_mlp if moe else 0.0
    dense_tensor_m = float(tokens)
    expert_tensor_m = (
        float(tokens * top_k) / touched_routed_experts
        if touched_routed_experts
        else 0.0
    )
    return {
        "tensor_flops": qkv + out + attention + mlp,
        "dense_tensor_flops": float(dense_tensor_flops),
        "expert_tensor_flops": float(expert_tensor_flops),
        "dense_tensor_m": dense_tensor_m,
        "expert_tensor_m": expert_tensor_m,
        "linear_flops": qkv + out + mlp,
        "attention_flops": attention,
        "vector_flops": vector,
        "weight_bytes": float(parameter_bytes),
        "non_routed_weight_bytes": float(non_routed_weight_bytes),
        "routed_weight_bytes": float(routed_weight_bytes),
        "touched_routed_experts": float(touched_routed_experts),
        "activation_bytes": float(activation_bytes),
        "hbm_extra_bytes": 0.0,
        "collective_bytes": float(2 * activation_bytes),
        "moe": float(moe),
        "top_k": float(top_k),
        "shared_experts_included": float(shared),
        "mlp_intermediate_size": float(intermediate),
        "mlp_matrix_count": float(matrix_count),
    }


def _partition_moe_weight_service(
    work: Mapping[str, float], ep_partitions: int
) -> dict[str, float]:
    """Return per-EP-rank HBM service without changing logical operator work."""

    if ep_partitions <= 0:
        raise ValueError("ep_partitions must be positive")
    result = dict(work)
    if work["moe"]:
        result["weight_bytes"] = (
            work["non_routed_weight_bytes"]
            + work["routed_weight_bytes"] / ep_partitions
        )
    return result


def _resource_group(prefix: str, placement: Placement) -> tuple[str, ...]:
    return tuple(f"{prefix}.die{die}" for die in placement.die_ids)


def _decode_tensor_spatial_utilization(work: Mapping[str, float]) -> float:
    """FLOP-weighted harmonic utilization from physical M/128 occupancy."""

    dense_flops = float(work.get("dense_tensor_flops", 0.0))
    expert_flops = float(work.get("expert_tensor_flops", 0.0))
    total = dense_flops + expert_flops
    if total <= 0:
        return 1.0
    dense_util = min(1.0, max(1.0 / TENSOR_TILE_M, float(work["dense_tensor_m"]) / TENSOR_TILE_M))
    service = dense_flops / dense_util
    if expert_flops:
        expert_util = min(1.0, max(1.0 / TENSOR_TILE_M, float(work["expert_tensor_m"]) / TENSOR_TILE_M))
        service += expert_flops / expert_util
    return total / service


def _decode_window_work(
    work: Mapping[str, float], tile: int, window_count: int
) -> tuple[dict[str, float], float]:
    """Slice linear decode work and physical M for one M128 window."""
    total_m = float(work["dense_tensor_m"])
    tile_m = min(float(TENSOR_TILE_M), max(0.0, total_m - tile * TENSOR_TILE_M))
    if total_m <= 0 or tile_m <= 0 or tile >= window_count:
        raise ValueError("decode window has no token rows")
    fraction = tile_m / total_m
    window = dict(work)
    window["dense_tensor_flops"] = work["dense_tensor_flops"] * fraction
    window["expert_tensor_flops"] = work["expert_tensor_flops"] * fraction
    window["dense_tensor_m"] = tile_m
    window["expert_tensor_m"] = work["expert_tensor_m"] * fraction
    return window, fraction



def _add_control(
    builder: _PairBuilder,
    prefix: str,
    phase: str,
    layer: int,
    tile: int,
    placement: Placement,
    base_deps: Iterable[str],
    overlap_deps: Iterable[str],
    full_train_deps: Iterable[str] | None = None,
) -> str:
    return builder.add(
        action_id=f"{prefix}.control",
        phase=phase,
        layer=layer,
        operator="CONTROL_ISSUE",
        tile_id=tile,
        base_deps=base_deps,
        overlap_deps=overlap_deps,
        full_train_deps=full_train_deps,
        resources=(f"control.die{placement.anchor_die}.issue_queue",),
        duration_cycles=CONTROL_ISSUE_CYCLES,
    )


def _add_hbm_streams(
    builder: _PairBuilder,
    prefix: str,
    phase: str,
    layer: int,
    tile: int,
    placement: Placement,
    total_bytes: float,
    base_deps: Iterable[str],
    overlap_deps: Iterable[str],
    full_train_deps: Iterable[str] | None = None,
) -> list[str]:
    action_ids: list[str] = []
    per_stack = total_bytes / len(HBM_STACKS)
    for stack in HBM_STACKS:
        route = directed_route_resources(stack.home_die, placement.anchor_die)
        local_noc = f"noc.die{placement.anchor_die}.hbm_ingress.to.sram"
        action_ids.append(
            builder.add(
                action_id=f"{prefix}.hbm{stack.stack_id}",
                phase=phase,
                layer=layer,
                operator="HBM_STREAM",
                tile_id=tile,
                base_deps=base_deps,
                overlap_deps=overlap_deps,
                full_train_deps=full_train_deps,



                bytes=per_stack,
                route=route,
                resources=(
                    stack.resource_id,
                    f"hbm.ingress.stack{stack.stack_id}.to.die{placement.anchor_die}",
                    f"dte.die{placement.anchor_die}.channel0",
                    local_noc,
                    f"sram.die{placement.anchor_die}.write",
                    *route,
                ),
                duration_cycles=(
                    HBM_FIRST_BYTE_CYCLES
                    + _cycles_for_work(per_stack, min(HBM_STACK_BPS, DTE_CHANNEL_BPS))
                ),
                duration_source="target_config_hbm_20ns_first_byte_unvalidated",
            )
        )
    return action_ids


def _add_kv_append_hbm(
    builder: _PairBuilder,
    prefix: str,
    phase: str,
    layer: int,
    tile: int,
    placement: Placement,
    total_bytes: float,
    base_deps: Iterable[str],
    overlap_deps: Iterable[str],
) -> list[str]:
    """Write one layer's newly generated token KV state back to HBM."""
    action_ids: list[str] = []
    per_stack = total_bytes / len(HBM_STACKS)
    for stack in HBM_STACKS:
        route = directed_route_resources(placement.anchor_die, stack.home_die)
        action_ids.append(
            builder.add(
                action_id=f"{prefix}.hbm{stack.stack_id}",
                phase=phase,
                layer=layer,
                operator="KV_APPEND_HBM_WRITE",
                tile_id=tile,
                base_deps=base_deps,
                overlap_deps=overlap_deps,
                bytes=per_stack,
                route=route,
                resources=(
                    stack.resource_id,
                    f"hbm.append.die{placement.anchor_die}.to.stack{stack.stack_id}",
                    f"dte.die{placement.anchor_die}.channel0",
                    f"noc.die{placement.anchor_die}.sram.to.hbm",
                    f"sram.die{placement.anchor_die}.read",
                    *route,
                ),
                duration_cycles=(
                    HBM_FIRST_BYTE_CYCLES
                    + _cycles_for_work(per_stack, min(HBM_STACK_BPS, DTE_CHANNEL_BPS))
                ),
                duration_source="target_config_hbm_20ns_first_byte_unvalidated",
            )
        )
    return action_ids


def _add_collective(
    builder: _PairBuilder,
    prefix: str,
    phase: str,
    layer: int,
    tile: int,
    placement: Placement,
    total_bytes: float,
    base_deps: Iterable[str],
    overlap_deps: Iterable[str],
    operator: str,
    runtime_scale: float = 1.0,
    full_train_deps: Iterable[str] | None = None,
) -> list[str]:
    # Bidirectional end-to-end flows ensure opposite directions are independent.
    source, destination = placement.die_ids[0], placement.die_ids[-1]
    pairs = ((source, destination), (destination, source))
    action_ids: list[str] = []
    for flow_index, (flow_source, flow_destination) in enumerate(pairs):
        route = directed_route_resources(flow_source, flow_destination)
        flow_bytes = total_bytes / len(pairs)
        action_ids.append(
            builder.add(
                action_id=f"{prefix}.flow{flow_index}",
                phase=phase,
                layer=layer,
                operator=operator,
                tile_id=tile,
                base_deps=base_deps,
                overlap_deps=overlap_deps,
                full_train_deps=full_train_deps,
                bytes=flow_bytes,
                runtime_work=flow_bytes * runtime_scale,
                route=route,
                resources=(
                    f"dte.die{flow_source}.channel1",
                    f"noc.die{flow_source}.sram.to.dte",
                    f"noc.die{flow_destination}.dte.to.sram",
                    *route,
                ),
                duration_cycles=(

                    DTE_LAUNCH_CYCLES
                    + len(route) * D2D_HOP_LATENCY_CYCLES
                    + _cycles_for_work(
                        flow_bytes * runtime_scale,
                        min(D2D_LINK_BPS, DTE_CHANNEL_BPS),
                    )
                ),
                duration_source=(
                    "target_config_dte_2cycle_launch_plus_1cycle_per_hop_unvalidated"
                ),
            )
        )
    return action_ids


def _add_tensor(
    builder: _PairBuilder,
    prefix: str,
    phase: str,
    layer: int,
    tile: int,
    placement: Placement,
    flops: float,
    base_deps: Iterable[str],
    overlap_deps: Iterable[str],
    operator: str,
    runtime_scale: float = 1.0,
    full_train_deps: Iterable[str] | None = None,
    spatial_utilization: float = 1.0,
) -> str:
    if not 0.0 < spatial_utilization <= 1.0:
        raise ValueError("spatial_utilization must lie in (0, 1]")
    rate = len(placement.die_ids) * TENSOR_FLOPS_PER_DIE * TENSOR_EFFICIENCY_PRIOR
    runtime_flops = flops * runtime_scale / spatial_utilization
    return builder.add(
        action_id=f"{prefix}.tensor",
        phase=phase,
        layer=layer,
        operator=operator,
        tile_id=tile,
        base_deps=base_deps,
        overlap_deps=overlap_deps,
        full_train_deps=full_train_deps,
        logical_work=flops,
        runtime_work=runtime_flops,
        resources=(*_resource_group("tensor", placement), *_resource_group("sram.read", placement)),
        duration_cycles=_cycles_for_work(runtime_flops, rate),
        duration_source=(
            "decode_M128_spatial_weighted_tensor_prior"
            if spatial_utilization < 1.0
            else "analytical_target_rate_prior"
        ),
    )


def _add_reduce_vector(
    builder: _PairBuilder,
    prefix: str,
    phase: str,
    layer: int,
    tile: int,
    placement: Placement,
    vector_flops: float,
    reduce_bytes: float,
    base_deps: Iterable[str],
    overlap_deps: Iterable[str],
    operator: str,
    runtime_scale: float = 1.0,
    full_train_deps: Iterable[str] | None = None,
) -> str:
    rate = len(placement.die_ids) * VECTOR_FLOPS_PER_DIE_PRIOR * VECTOR_EFFICIENCY_PRIOR
    duration = max(
        _cycles_for_work(vector_flops * runtime_scale, rate),
        _cycles_for_work(reduce_bytes * runtime_scale, REDUCER_BYTES_PER_SECOND),
    )
    return builder.add(
        action_id=f"{prefix}.reduce",
        phase=phase,
        layer=layer,
        operator=operator,
        tile_id=tile,
        base_deps=base_deps,
        overlap_deps=overlap_deps,
        full_train_deps=full_train_deps,
        logical_work=vector_flops,
        runtime_work=vector_flops * runtime_scale,
        bytes=reduce_bytes,
        resources=(
            *_resource_group("vector", placement),
            *_resource_group("reducer", placement),
            *_resource_group("sram.write", placement),
        ),
        duration_cycles=duration,
    )


def _add_operator_windows(
    builder: _PairBuilder,
    *,
    phase: str,
    layer: int,
    placement: Placement,
    work: Mapping[str, float],
    entry_base: Sequence[str],
    entry_overlap: Sequence[str],
    entry_full_train: Sequence[str],
    overlap_enabled: bool,
    full_train_overlap_enabled: bool,
    operator_prefix: str,
    runtime_scale: float = 1.0,
    window_count: int = FORWARD_WINDOWS,
    shape_aware_tensor: bool = False,
) -> tuple[str, str, str]:
    if type(window_count) is not int or not 1 <= window_count <= FORWARD_WINDOWS:
        raise ValueError(f"window_count must lie in [1, {FORWARD_WINDOWS}]")
    base_tail = list(entry_base)
    overlap_tail = list(entry_overlap)
    full_tail = list(entry_full_train)
    overlap_control_tail = list(entry_overlap)
    full_control_tail = list(entry_full_train)
    for tile in range(window_count):
        prefix = f"{phase}.l{layer}.{placement.name}.q{tile}.{operator_prefix}"
        control = _add_control(
            builder, prefix, phase, layer, tile, placement,
            base_tail, overlap_control_tail if overlap_enabled else overlap_tail,
            full_control_tail if full_train_overlap_enabled else full_tail,
        )
        if shape_aware_tensor:
            window_work, work_fraction = _decode_window_work(
                work, tile, window_count
            )
            spatial_utilization = _decode_tensor_spatial_utilization(window_work)
        else:
            work_fraction = 1.0 / window_count
            spatial_utilization = 1.0
        hbm_base_deps = (control,)
        hbm_overlap_deps = (control,)
        hbm = _add_hbm_streams(
            builder, prefix, phase, layer, tile, placement,
            (
                work["weight_bytes"]
                + work["activation_bytes"]
                + work.get("hbm_extra_bytes", 0.0)
            ) * work_fraction,
            hbm_base_deps, hbm_overlap_deps,
            (control,),
        )
        collective_operator = "DISPATCH" if work["moe"] else "ALL_GATHER"
        comm = _add_collective(
            builder, prefix, phase, layer, tile, placement,
            work["collective_bytes"] * work_fraction,
            hbm, hbm,
            collective_operator,
            runtime_scale,
            hbm,
        )
        tensor = _add_tensor(
            builder, prefix, phase, layer, tile, placement,
            work["tensor_flops"] * work_fraction,
            comm, (*hbm, *comm) if overlap_enabled else comm,
            "EXPERT_GEMM" if work["moe"] else "DENSE_ATTN_MLP_GEMM",
            runtime_scale,
            (*hbm, *comm) if full_train_overlap_enabled else comm,
            spatial_utilization,
        )
        post = _add_collective(
            builder, f"{prefix}.post", phase, layer, tile, placement,
            work["collective_bytes"] * work_fraction,
            (tensor,), (tensor,),
            "COMBINE" if work["moe"] else "REDUCE_SCATTER",
            runtime_scale,
            (tensor,),
        )
        reduce = _add_reduce_vector(
            builder, prefix, phase, layer, tile, placement,
            work["vector_flops"] * work_fraction,
            work["activation_bytes"] * work_fraction,
            post, post,
            "TOPK_WEIGHTED_REDUCE" if work["moe"] else "NORM_RESIDUAL",
            runtime_scale,
            post,
        )
        base_tail = [reduce]
        overlap_tail = [reduce]
        full_tail = [reduce]
        overlap_control_tail = [control] if overlap_enabled else [reduce]
        full_control_tail = [control] if full_train_overlap_enabled else [reduce]
    return base_tail[0], overlap_tail[0], full_tail[0]


def _add_dp_gradient_sync(
    builder: _PairBuilder,
    layer: int,
    bytes_per_group: float,
    base_deps: Sequence[str],
    overlap_deps: Sequence[str],
    full_train_deps: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = []
    # Corresponding TP-rank anchors across the four 3x3 DP groups.
    ring = (TRAINING_GROUPS[0].anchor_die, TRAINING_GROUPS[1].anchor_die,
            TRAINING_GROUPS[3].anchor_die, TRAINING_GROUPS[2].anchor_die)
    for index, source in enumerate(ring):
        destination = ring[(index + 1) % len(ring)]
        route = directed_route_resources(source, destination)
        ids.append(
            builder.add(
                action_id=f"gradient_sync.l{layer}.flow{index}",
                phase="wgrad",
                layer=layer,
                operator="DP_GRADIENT_RING",
                tile_id=index,
                base_deps=base_deps,
                overlap_deps=overlap_deps,
                full_train_deps=full_train_deps,
                bytes=bytes_per_group,
                route=route,
                resources=(
                    f"dte.die{source}.channel1",
                    f"noc.die{source}.reducer.to.dte",
                    f"reducer.die{destination}",
                    *route,
                ),
                duration_cycles=_cycles_for_work(bytes_per_group, min(D2D_LINK_BPS, DTE_CHANNEL_BPS)),
            )
        )
    return ids, ids, ids


def build_training_replay(
    manifest: Mapping[str, object],
    seq_len: int,
    *,
    routing_skew: float = 1.0,
    include_shared_experts: bool = False,
) -> ReplayPair:
    """Build one complete forward/loss/backward/WGRAD/AdamW step.

    ``batch_size=1`` is per DP rank.  The four 3x3 groups execute concurrently;
    work counters therefore include all four replicas while latency naturally
    follows shared physical resources.
    """

    if seq_len <= 0 or routing_skew < 1.0:
        raise ValueError("seq_len must be positive and routing_skew must be >= 1")
    layers = _integer(manifest, "num_layers")
    builder = _PairBuilder()
    base_tails: dict[str, list[str]] = {group.name: [] for group in TRAINING_GROUPS}
    overlap_tails: dict[str, list[str]] = {group.name: [] for group in TRAINING_GROUPS}
    full_backward_tails: dict[str, list[str]] = {
        group.name: [] for group in TRAINING_GROUPS
    }

    # Forward is the only phase whose dependency graph changes.
    for layer in range(layers):
        work = _layer_work(
            manifest,
            seq_len,
            layer,
            include_shared_experts=include_shared_experts,
        )
        runtime_scale = routing_skew if work["moe"] else 1.0
        group_work = _partition_moe_weight_service(work, len(TRAINING_GROUPS))
        for group in TRAINING_GROUPS:
            base_tail, overlap_tail, full_tail = _add_operator_windows(
                builder,
                phase="forward",
                layer=layer,
                placement=group,
                work=group_work,
                entry_base=base_tails[group.name],
                entry_overlap=overlap_tails[group.name],
                entry_full_train=full_backward_tails[group.name],
                overlap_enabled=True,
                full_train_overlap_enabled=True,
                operator_prefix="moe" if work["moe"] else "dense",
                runtime_scale=runtime_scale,
            )
            base_tails[group.name] = [base_tail]
            overlap_tails[group.name] = [overlap_tail]
            full_backward_tails[group.name] = [full_tail]

    # Loss / cross entropy uses vector + reducer and is identical in both states.
    hidden = _integer(manifest, "hidden_size")
    for group in TRAINING_GROUPS:
        loss = _add_reduce_vector(
            builder,
            f"loss.{group.name}", "forward", layers, 0, group,
            seq_len * hidden * 8.0, seq_len * FP32_BYTES,
            base_tails[group.name], overlap_tails[group.name], "LOSS_CROSS_ENTROPY",
            full_train_deps=full_backward_tails[group.name],
        )
        base_tails[group.name] = [loss]
        overlap_tails[group.name] = [loss]
        full_backward_tails[group.name] = [loss]

    # Backward and WGRAD use explicit operator work, not a fixed E2E ratio.
    full_pending_gradient_sync: list[str] = []
    for reverse_index, layer in enumerate(reversed(range(layers))):
        work = _layer_work(
            manifest,
            seq_len,
            layer,
            include_shared_experts=include_shared_experts,
        )
        runtime_scale = routing_skew if work["moe"] else 1.0
        backward_work = dict(work)
        backward_work["tensor_flops"] = work["linear_flops"] + 2.0 * work["attention_flops"]
        backward_work["weight_bytes"] = work["weight_bytes"]
        backward_work["collective_bytes"] = work["collective_bytes"]
        backward_group_work = _partition_moe_weight_service(
            backward_work, len(TRAINING_GROUPS)
        )
        for group in TRAINING_GROUPS:
            base_tail, overlap_tail, full_tail = _add_operator_windows(
                builder,
                phase="backward",
                layer=layer,
                placement=group,
                work=backward_group_work,
                entry_base=base_tails[group.name],
                entry_overlap=overlap_tails[group.name],
                entry_full_train=full_backward_tails[group.name],
                overlap_enabled=False,
                full_train_overlap_enabled=True,
                operator_prefix="backward",
                runtime_scale=runtime_scale,
            )
            base_tails[group.name] = [base_tail]
            overlap_tails[group.name] = [overlap_tail]
            # The next layer dX may advance before this layer WGRAD/sync.
            full_backward_tails[group.name] = [full_tail]

        wgrad_work = dict(work)
        wgrad_work["tensor_flops"] = work["linear_flops"]
        wgrad_work["vector_flops"] = work["vector_flops"] * 0.5
        wgrad_group_work = _partition_moe_weight_service(
            wgrad_work, len(TRAINING_GROUPS)
        )
        full_wgrad_tails: dict[str, list[str]] = {}
        for group in TRAINING_GROUPS:
            base_tail, overlap_tail, full_tail = _add_operator_windows(
                builder,
                phase="wgrad",
                layer=layer,
                placement=group,
                work=wgrad_group_work,
                entry_base=base_tails[group.name],
                entry_overlap=overlap_tails[group.name],
                entry_full_train=full_backward_tails[group.name],
                overlap_enabled=False,
                full_train_overlap_enabled=True,
                operator_prefix="wgrad",
                runtime_scale=runtime_scale,
            )
            base_tails[group.name] = [base_tail]
            overlap_tails[group.name] = [overlap_tail]
            full_wgrad_tails[group.name] = [full_tail]
        sync_base, sync_overlap, sync_full = _add_dp_gradient_sync(
            builder,
            layer,
            (
                work["non_routed_weight_bytes"]
                if work["moe"]
                else work["weight_bytes"] / len(TRAINING_GROUPS)
            ),
            [item for values in base_tails.values() for item in values],
            [item for values in overlap_tails.values() for item in values],
            [item for values in full_wgrad_tails.values() for item in values],
        )
        full_pending_gradient_sync.extend(sync_full)
        for group in TRAINING_GROUPS:
            base_tails[group.name] = list(sync_base)
            overlap_tails[group.name] = list(sync_overlap)

    # AdamW is derived from parameter count/state traffic, never SGD-scaled.
    parameter_count = float(_manifest_value(manifest, "parameter_count"))
    parameter_breakdown = manifest.get("parameter_breakdown", {})
    routed_parameters = 0.0
    if isinstance(parameter_breakdown, Mapping):
        routed_parameters = float(parameter_breakdown.get("routed_experts", 0) or 0)
    shared_parameters = 0.0
    if isinstance(parameter_breakdown, Mapping) and not include_shared_experts:
        shared_parameters = float(parameter_breakdown.get("shared_experts", 0) or 0)
    # Dense DP replicas own a complete model.  MoE EP ranks replicate only the
    # non-routed tensors and partition routed expert parameters four ways,
    # matching the capacity placement policy.
    parameter_count_per_group = (
        parameter_count - shared_parameters - routed_parameters
        + routed_parameters / len(TRAINING_GROUPS)
    )
    optimizer_flops = parameter_count_per_group * 10.0
    optimizer_bytes = parameter_count_per_group * (DTYPE_BYTES + FP32_BYTES * 5)
    for group in TRAINING_GROUPS:
        work = {
            "tensor_flops": 0.0,
            "linear_flops": 0.0,
            "attention_flops": 0.0,
            "vector_flops": optimizer_flops,
            "weight_bytes": optimizer_bytes,
            "activation_bytes": parameter_count_per_group * DTYPE_BYTES,
            "collective_bytes": 0.0,
            "moe": 0.0,
        }
        base_tail, overlap_tail, full_tail = _add_operator_windows(
            builder,
            phase="optimizer",
            layer=layers,
            placement=group,
            work=work,
            entry_base=base_tails[group.name],
            entry_overlap=overlap_tails[group.name],
            entry_full_train=(
                *full_backward_tails[group.name],
                *full_pending_gradient_sync,
            ),
            overlap_enabled=False,
            full_train_overlap_enabled=False,
            operator_prefix="adamw",
        )
        base_tails[group.name] = [base_tail]
        overlap_tails[group.name] = [overlap_tail]
        full_backward_tails[group.name] = [full_tail]
    return builder.pair("training", include_full_train=True)


def _add_pd_handoff(
    builder: _PairBuilder,
    manifest: Mapping[str, object],
    prefill_seq: int,
    entry_base: Sequence[str],
    entry_overlap: Sequence[str],
) -> tuple[list[str], list[str]]:
    layers = _integer(manifest, "num_layers")
    # MLA transfers its compressed cache state, not a fictitious GQA K/V pair.
    per_layer_bytes = prefill_seq * _kv_cache_width(manifest) * DTYPE_BYTES
    if not _attention_is_mla(manifest):
        per_layer_bytes *= 2.0
    base_tail = list(entry_base)
    overlap_tail = list(entry_overlap)
    for layer in range(layers):
        layer_ids: list[str] = []
        for handoff_index, (source_name, destination_name) in enumerate(PD_HANDOFFS):
            source = placement_by_name(source_name)
            destination = placement_by_name(destination_name)
            source_die, destination_die = source.anchor_die, destination.anchor_die
            route = directed_route_resources(source_die, destination_die)
            layer_ids.append(
                builder.add(
                    action_id=f"handoff.l{layer}.{source_name}.to.{destination_name}",
                    phase="handoff",
                    layer=layer,
                    operator="PD_KV_STATE_TRANSFER",
                    tile_id=handoff_index,
                    base_deps=base_tail,
                    # A layer KV state may not move before prefill is ready.
                    # The aggregate prefill model exposes a full-model marker,
                    # so both schedules conservatively retain this chain.
                    overlap_deps=overlap_tail,
                    bytes=per_layer_bytes,
                    route=route,
                    resources=(
                        f"hbm.stack{handoff_index % 4}",
                        f"dte.die{source_die}.channel0",
                        f"noc.die{source_die}.hbm.to.dte",
                        f"noc.die{destination_die}.dte.to.hbm",
                        *route,
                    ),
                    duration_cycles=_cycles_for_work(per_layer_bytes, min(D2D_LINK_BPS, DTE_CHANNEL_BPS)),
                )
            )
        base_tail = layer_ids
        overlap_tail = layer_ids
    waits: list[str] = []
    for destination_name in ("D0", "D1"):
        destination = placement_by_name(destination_name)
        waits.append(
            builder.add(
                action_id=f"handoff_wait.{destination_name}",
                phase="handoff_wait",
                layer=layers,
                operator="PD_KV_READY_WAIT",
                tile_id=0,
                base_deps=base_tail,
                overlap_deps=overlap_tail,
                resources=(f"control.die{destination.anchor_die}.issue_queue",),
                duration_cycles=CONTROL_ISSUE_CYCLES,
            )
        )
    return waits, waits


def build_inference_replay(
    manifest: Mapping[str, object],
    batch_size: int,
    *,
    prefill_seq: int = 2304,
    kv_length: int = 36864,
    routing_skew: float = 1.0,
    include_shared_experts: bool = False,
) -> ReplayPair:
    """Build four-P/two-D prefill, PD handoff and two recurrent decode steps."""

    if min(batch_size, prefill_seq, kv_length) <= 0 or routing_skew < 1.0:
        raise ValueError("batch/sequence lengths must be positive and skew >= 1")
    layers = _integer(manifest, "num_layers")
    builder = _PairBuilder()
    prefill_instances = tuple(item for item in INFERENCE_INSTANCES if item.role == "prefill")
    decode_instances = tuple(item for item in INFERENCE_INSTANCES if item.role == "decode")
    base_tails: dict[str, list[str]] = {item.name: [] for item in prefill_instances}
    overlap_tails: dict[str, list[str]] = {item.name: [] for item in prefill_instances}

    for layer in range(layers):
        work = _layer_work(
            manifest,
            prefill_seq,
            layer,
            include_shared_experts=include_shared_experts,
        )
        runtime_scale = routing_skew if work["moe"] else 1.0
        for instance in prefill_instances:
            base_tail, overlap_tail, _ = _add_operator_windows(
                builder,
                phase="prefill",
                layer=layer,
                placement=instance,
                work=work,
                entry_base=base_tails[instance.name],
                entry_overlap=overlap_tails[instance.name],
                entry_full_train=overlap_tails[instance.name],
                overlap_enabled=True,
                full_train_overlap_enabled=True,
                operator_prefix="prefill",
                runtime_scale=runtime_scale,
            )
            base_tails[instance.name] = [base_tail]
            overlap_tails[instance.name] = [overlap_tail]

    handoff_base, handoff_overlap = _add_pd_handoff(
        builder,
        manifest,
        prefill_seq,
        [item for values in base_tails.values() for item in values],
        [item for values in overlap_tails.values() for item in values],
    )
    decode_base: dict[str, list[str]] = {item.name: list(handoff_base) for item in decode_instances}
    decode_overlap: dict[str, list[str]] = {item.name: list(handoff_overlap) for item in decode_instances}
    decode_window_count = min(
        FORWARD_WINDOWS, math.ceil(batch_size / TENSOR_TILE_M)
    )
    for token_step in range(DECODE_UNROLL_STEPS):
        phase = "decode_warmup" if token_step == 0 else "decode"
        token_append_base = {item.name: [] for item in decode_instances}
        token_append_overlap = {item.name: [] for item in decode_instances}
        for layer in range(layers):
            work = _layer_work(
                manifest,
                batch_size,
                layer,
                decode_kv=kv_length + token_step,
                include_shared_experts=include_shared_experts,
            )
            # KV is an HBM boundary stream.  It must not inflate activation
            # bytes consumed by NORM_RESIDUAL/reducer actions.
            work = dict(work)
            kv_multiplier = 1 if _attention_is_mla(manifest) else 2
            work["hbm_extra_bytes"] = (
                batch_size
                * (kv_length + token_step)
                * _kv_cache_width(manifest)
                * DTYPE_BYTES
                * kv_multiplier
            )
            runtime_scale = routing_skew if work["moe"] else 1.0
            for instance in decode_instances:
                base_tail, overlap_tail, _ = _add_operator_windows(
                    builder,
                    phase=phase,
                    layer=layer,
                    placement=instance,
                    work=work,
                    entry_base=decode_base[instance.name],
                    entry_overlap=decode_overlap[instance.name],
                    entry_full_train=decode_overlap[instance.name],
                    overlap_enabled=True,
                    full_train_overlap_enabled=True,
                    operator_prefix=f"decode_t{token_step}",
                    runtime_scale=runtime_scale,
                    window_count=decode_window_count,
                    shape_aware_tensor=True,
                )
                decode_base[instance.name] = [base_tail]
                decode_overlap[instance.name] = [overlap_tail]
                append_bytes = (
                    batch_size
                    * _kv_cache_width(manifest)
                    * DTYPE_BYTES
                    * kv_multiplier
                )
                append_ids = _add_kv_append_hbm(
                    builder,
                    f"{phase}.l{layer}.{instance.name}.kv_append_t{token_step}",
                    phase,
                    layer,
                    token_step,
                    instance,
                    append_bytes,
                    (base_tail,),
                    (overlap_tail,),
                )
                token_append_base[instance.name].extend(append_ids)
                token_append_overlap[instance.name].extend(append_ids)
        for instance in decode_instances:
            commit = builder.add(
                action_id=f"decode_t{token_step}.{instance.name}.token_commit",
                phase=phase,
                layer=layers,
                operator="TOKEN_COMMIT",
                tile_id=token_step,
                base_deps=(
                    *decode_base[instance.name], *token_append_base[instance.name]
                ),
                overlap_deps=(
                    *decode_overlap[instance.name], *token_append_overlap[instance.name]
                ),
                resources=(
                    f"control.die{instance.anchor_die}.issue_queue",
                    f"vector.die{instance.anchor_die}",
                    f"reducer.die{instance.anchor_die}",
                ),
                duration_cycles=CONTROL_ISSUE_CYCLES,
                duration_source="token_sample_commit_control_prior",
            )
            # Per-instance recurrence: D0 and D1 never depend on each other.
            decode_base[instance.name] = [commit]
            decode_overlap[instance.name] = [commit]
    return builder.pair("inference")


def _digest_actions(actions: Sequence[Action]) -> str:
    payload = [action.manifest_dict() for action in actions]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _evidence_metadata(
    manifest: Mapping[str, object], evidence: Mapping[str, object] | None
) -> tuple[str, list[str], list[str], float]:
    limitations = [
        "analytical_vector_rate_prior",
        "representative_collective_route_abstraction",
        "local_noc_die_level_abstraction",
        "hbm_ingress_anchor_abstraction",
        "target_config_communication_timing_unvalidated",
        "target_config_hbm_latency_unvalidated",
    ]
    signatures: list[str] = []
    uncertainty = 0.25
    if evidence:
        signatures = [str(item) for item in evidence.get("evidence_signatures", [])]
    if _attention_is_mla(manifest):
        limitations.append("analytical_only_mla")
        return "analytical_only_mla", signatures, limitations, 0.35
    if not evidence:
        limitations.append("no_target_binding_cycle_anchor")
        return "analytical_resource_dag_extrapolation", signatures, limitations, uncertainty
    signatures = [str(item) for item in evidence.get("evidence_signatures", [])]
    calibrated = (
        evidence.get("unit_closure_passed") is True
        and evidence.get("repeatability_passed") is True
        and evidence.get("validation_passed") is True
        and float(evidence.get("p95_relative_error", 1.0)) <= 0.15
    )
    if calibrated:
        # Anchors remain context only until full TP/EP flows, per-core NoC and
        # per-destination HBM ingress replace the explicit abstractions above.
        limitations.append("validated_anchor_not_applied_to_abstract_routes")
        return "analytical_resource_dag_extrapolation", signatures, limitations, uncertainty
    limitations.append("provided_evidence_failed_release_gate")
    return "analytical_resource_dag_extrapolation", signatures, limitations, uncertainty


def _decode_commit_intervals(schedule: ScheduleResult) -> dict[str, float]:
    """Return steady-token latency from per-instance commit-to-commit edges."""
    intervals: dict[str, float] = {}
    for instance in INFERENCE_INSTANCES:
        if instance.role != "decode":
            continue
        warmup = f"decode_t0.{instance.name}.token_commit"
        steady = f"decode_t1.{instance.name}.token_commit"
        if warmup not in schedule.finishes or steady not in schedule.finishes:
            raise AssertionError(f"decode recurrence commits missing for {instance.name}")
        intervals[instance.name] = schedule.finishes[steady] - schedule.finishes[warmup]
    return intervals



def summarize_replay(
    pair: ReplayPair,
    manifest: Mapping[str, object],
    *,
    evidence: Mapping[str, object] | None = None,
    capacity_status: str = "capacity_not_audited",
) -> dict[str, object]:
    base = replay(pair.base_actions)
    overlap = replay(pair.overlap_actions)
    full_train = (
        replay(pair.full_train_actions)
        if pair.full_train_actions is not None
        else None
    )
    estimate_source, signatures, limitations, uncertainty = _evidence_metadata(manifest, evidence)
    if capacity_status == "capacity_infeasible_projection":
        limitations.append("capacity_infeasible_projection")
    lower = overlap.theory_lower_cycles
    speedup = base.makespan_cycles / overlap.makespan_cycles
    result: dict[str, object] = {
        "workload": pair.workload,
        "dag_version": E2E_DAG_VERSION,
        "action_count": len(pair.base_actions),
        "action_digest_base": _digest_actions(pair.base_actions),
        "action_digest_overlap": _digest_actions(pair.overlap_actions),
        "action_digest_full_train": (
            _digest_actions(pair.full_train_actions)
            if pair.full_train_actions is not None
            else None
        ),
        "same_work_invariant": True,
        "overlap_scope": "forward_only",
        "T_base_cycles": base.makespan_cycles,
        "T_overlap_cycles": overlap.makespan_cycles,
        "T_forward_only_overlap_cycles": overlap.makespan_cycles,
        "speedup": speedup,
        "speedup_forward_only": speedup,
        "theory_lower_cycles": lower,
        "theory_speedup": base.makespan_cycles / lower if lower else 1.0,
        "attainment": lower / overlap.makespan_cycles if overlap.makespan_cycles else 0.0,
        "uncertainty_low": (base.makespan_cycles * (1 - uncertainty)) / (overlap.makespan_cycles * (1 + uncertainty)),
        "uncertainty_high": (base.makespan_cycles * (1 + uncertainty)) / (overlap.makespan_cycles * (1 - uncertainty)),
        "base_phase_cycles": dict(base.phase_cycles),
        "overlap_phase_cycles": dict(overlap.phase_cycles),
        "phase_theory_lower_cycles": dict(overlap.phase_theory_lower_cycles),
        "base_resource_service_cycles": dict(base.resource_service_cycles),
        "overlap_resource_service_cycles": dict(overlap.resource_service_cycles),
        "estimate_source": estimate_source,
        "evidence_signatures": signatures,
        "capacity_status": capacity_status,
        "status": "projection" if capacity_status == "capacity_infeasible_projection" else "estimated",
        "limitation_tags": sorted(set(limitations)),
        "hardware_assumptions": {
            "wafer": "6x6",
            "compute_cores_per_die": 16,
            "tensor_TFLOPS_per_die": 128.0,
            "hbm_stack_count": 4,
            "hbm_stack_bandwidth_Bps": HBM_STACK_BPS,
            "d2d_directed_link_Bps": D2D_LINK_BPS,
            "dte_channels_per_core": 2,
            "dte_channel_Bps": DTE_CHANNEL_BPS,
            "dte_aggregate_injection_Bps_per_core": 2 * DTE_CHANNEL_BPS,
            "dte_launch_cycles": DTE_LAUNCH_CYCLES,
            "dte_gamma_ns": DTE_GAMMA_NS,
            "dte_tau_launch_ns": DTE_TAU_LAUNCH_NS,
            "d2d_hop_latency_cycles": D2D_HOP_LATENCY_CYCLES,
            "hbm_first_byte_ns": HBM_FIRST_BYTE_NS,
            "timing_parameter_source": "configs/target_hardware.json_unvalidated",
            "route_policy": "XY_x_first",
        },
    }
    if pair.workload == "training":
        for phase in ("forward", "backward", "wgrad", "optimizer"):
            result[f"{phase}_cycles"] = overlap.phase_cycles.get(phase, 0.0)
        if full_train is None:
            raise AssertionError("training replay is missing full_train actions")
        tolerance = max(1e-6, base.makespan_cycles * 1e-12)
        if not (
            full_train.makespan_cycles <= overlap.makespan_cycles + tolerance
            and overlap.makespan_cycles <= base.makespan_cycles + tolerance
        ):
            raise AssertionError(
                "training schedule ordering violated: "
                f"full={full_train.makespan_cycles}, "
                f"forward_only={overlap.makespan_cycles}, "
                f"base={base.makespan_cycles}"
            )
        result.update(
            {
                "T_full_train_overlap_cycles": full_train.makespan_cycles,
                "speedup_full_train": (
                    base.makespan_cycles / full_train.makespan_cycles
                ),
                "full_train_phase_cycles": dict(full_train.phase_cycles),
                "full_train_resource_service_cycles": dict(
                    full_train.resource_service_cycles
                ),
                "full_train_phase_theory_lower_cycles": dict(
                    full_train.phase_theory_lower_cycles
                ),
                "full_train_theory_lower_cycles": full_train.theory_lower_cycles,
                "full_train_theory_speedup": (
                    base.makespan_cycles / full_train.theory_lower_cycles
                    if full_train.theory_lower_cycles
                    else 1.0
                ),
                "full_train_attainment": (
                    full_train.theory_lower_cycles / full_train.makespan_cycles
                    if full_train.makespan_cycles
                    else 0.0
                ),
                "full_train_phase_strictly_shorter_than_forward_only": {
                    phase: (
                        full_train.phase_cycles.get(phase, 0.0)
                        < overlap.phase_cycles.get(phase, 0.0) - tolerance
                    )
                    for phase in ("forward", "backward", "wgrad")
                },
                "training_schedule_ordering_passed": True,
            }
        )
    else:
        base_decode_intervals = _decode_commit_intervals(base)
        overlap_decode_intervals = _decode_commit_intervals(overlap)
        base_steady_decode = max(base_decode_intervals.values(), default=0.0)
        overlap_steady_decode = max(overlap_decode_intervals.values(), default=0.0)
        base_raw_decode_span = base.phase_cycles.get("decode", 0.0)
        overlap_raw_decode_span = overlap.phase_cycles.get("decode", 0.0)
        # Raw phase spans can include cross-instance start skew; recurrent
        # steady-state latency is each instance's own commit interval.
        result["base_phase_cycles"]["decode"] = base_steady_decode
        result["overlap_phase_cycles"]["decode"] = overlap_steady_decode
        result["base_decode_commit_intervals_cycles"] = base_decode_intervals
        result["overlap_decode_commit_intervals_cycles"] = overlap_decode_intervals
        result["base_raw_decode_phase_span_cycles"] = base_raw_decode_span
        result["overlap_raw_decode_phase_span_cycles"] = overlap_raw_decode_span
        result["base_decode_cycles"] = base_steady_decode
        result["overlap_decode_cycles"] = overlap_steady_decode
        result["prefill_cycles"] = overlap.phase_cycles.get("prefill", 0.0)
        result["handoff_cycles"] = overlap.phase_cycles.get("handoff", 0.0)
        result["handoff_wait_cycles"] = overlap.phase_cycles.get("handoff_wait", 0.0)
        result["decode_warmup_cycles"] = overlap.phase_cycles.get("decode_warmup", 0.0)
        result["decode_cycles"] = overlap_steady_decode
        result["decode_theory_lower_cycles"] = overlap.phase_theory_lower_cycles.get("decode", 0.0)
        result["TTFT_cycles"] = (
            result["prefill_cycles"] + result["handoff_cycles"] + result["handoff_wait_cycles"]
        )
    return result


def estimate_training_case(
    manifest: Mapping[str, object],
    seq_len: int,
    *,
    routing_skew: float = 1.0,
    include_shared_experts: bool = False,
    evidence: Mapping[str, object] | None = None,
    capacity_status: str = "capacity_not_audited",
) -> dict[str, object]:
    pair = build_training_replay(
        manifest,
        seq_len,
        routing_skew=routing_skew,
        include_shared_experts=include_shared_experts,
    )
    result = summarize_replay(pair, manifest, evidence=evidence, capacity_status=capacity_status)
    result.update({
        "seq_len": seq_len,
        "batch_size_per_dp_rank": 1,
        "tp": 9,
        "dp": 4,
        "ep": 4,
        "include_shared_experts": include_shared_experts,
        "T_full_train_overlap_seconds": (
            float(result["T_full_train_overlap_cycles"]) / CLOCK_HZ
        ),
    })
    return result


def estimate_inference_case(
    manifest: Mapping[str, object],
    batch_size: int,
    *,
    prefill_seq: int = 2304,
    kv_length: int = 36864,
    routing_skew: float = 1.0,
    include_shared_experts: bool = False,
    evidence: Mapping[str, object] | None = None,
    capacity_status: str = "capacity_not_audited",
) -> dict[str, object]:
    pair = build_inference_replay(
        manifest,
        batch_size,
        prefill_seq=prefill_seq,
        kv_length=kv_length,
        routing_skew=routing_skew,
        include_shared_experts=include_shared_experts,
    )
    result = summarize_replay(pair, manifest, evidence=evidence, capacity_status=capacity_status)
    decode_cycles = float(result["decode_cycles"])
    decode_works = [
        _layer_work(
            manifest,
            batch_size,
            layer,
            decode_kv=kv_length,
            include_shared_experts=include_shared_experts,
        )
        for layer in range(_integer(manifest, "num_layers"))
    ]
    decode_window_slices = [
        _decode_window_work(work, tile, min(FORWARD_WINDOWS, math.ceil(batch_size / TENSOR_TILE_M)))[0]
        for work in decode_works
        for tile in range(min(FORWARD_WINDOWS, math.ceil(batch_size / TENSOR_TILE_M)))
    ]
    spatial = [
        _decode_tensor_spatial_utilization(work) for work in decode_window_slices
    ]
    dense_m = [work["dense_tensor_m"] for work in decode_window_slices]
    expert_m = [work["expert_tensor_m"] for work in decode_window_slices if work["moe"]]
    decode_window_count = min(
        FORWARD_WINDOWS, math.ceil(batch_size / TENSOR_TILE_M)
    )
    kv_multiplier = 1 if _attention_is_mla(manifest) else 2
    kv_hbm_bytes_per_layer = (
        batch_size
        * kv_length
        * _kv_cache_width(manifest)
        * DTYPE_BYTES
        * kv_multiplier
    )
    result["limitation_tags"] = sorted(set(result["limitation_tags"]) | {
        "decode_shape_efficiency_analytical",
        "decode_two_token_recurrence",
        "decode_two_token_local_interval",
        "decode_steady_state_convergence_not_checked",
        "kv_append_hbm_write_analytical",
    })
    result.update(
        {
            "seq_len": prefill_seq,
            "batch_size": batch_size,
            "kv_length": kv_length,
            "tp": 6,
            "dp": 1,
            "ep": 6,
            "include_shared_experts": include_shared_experts,
            "decode_fidelity_diagnostics": {
                "version": "v4",
                "tensor_tile_m": TENSOR_TILE_M,
                "decode_window_count": decode_window_count,
                "decode_window_formula": "min(8,ceil(batch_size/128))",
                "decode_unroll_steps": DECODE_UNROLL_STEPS,
                "steady_phase": "two_token_local_commit_interval_at_fixed_context",
                "dense_tensor_m": batch_size,
                "dense_tensor_m_per_window": dense_m,
                "expert_tensor_m_per_window": expert_m,
                "dense_spatial_utilization_per_window": [
                    min(1.0, value / TENSOR_TILE_M) for value in dense_m
                ],
                "expert_tensor_m_min": min(expert_m) if expert_m else None,
                "expert_tensor_m_max": max(expert_m) if expert_m else None,
                "tensor_spatial_utilization_min": min(spatial),
                "tensor_spatial_utilization_max": max(spatial),
                "activation_reduce_bytes_per_layer": (
                    batch_size * _integer(manifest, "hidden_size") * DTYPE_BYTES
                ),
                "kv_hbm_bytes_per_layer_step0": kv_hbm_bytes_per_layer,
                "kv_bytes_enter_reducer": False,
                "dte_launch_cycles": DTE_LAUNCH_CYCLES,
                "d2d_hop_latency_cycles": D2D_HOP_LATENCY_CYCLES,
                "hbm_first_byte_cycles": HBM_FIRST_BYTE_CYCLES,
                "kv_append_bytes_per_layer_token": (
                    batch_size * _kv_cache_width(manifest) * DTYPE_BYTES * kv_multiplier
                ),
                "kv_append_operator": "KV_APPEND_HBM_WRITE",
                "token_commit_waits_for_all_layer_appends": True,
                "timing_parameter_source": "configs/target_hardware.json_unvalidated",
            },
            "system_decode_tokens_per_s": 2.0 * batch_size * CLOCK_HZ / decode_cycles,
            "TPOT_seconds": decode_cycles / CLOCK_HZ,
        }
    )
    return result


__all__ = [
    "Action",
    "ReplayPair",
    "ScheduleResult",
    "assert_same_work",
    "build_inference_replay",
    "build_training_replay",
    "estimate_inference_case",
    "estimate_training_case",
    "replay",
    "summarize_replay",
]
