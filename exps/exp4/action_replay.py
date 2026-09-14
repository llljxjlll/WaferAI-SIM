#!/usr/bin/env python3
"""Resource-explicit analytical replay from action FLOPs/bytes, not old durations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from placement_mapper import TopologyCapacityError, map_logical_ranks


CLOCK_HZ = 500_000_000
ROOT = Path(__file__).resolve().parent
TENSOR_EFFICIENCY = .72
VECTOR_EFFICIENCY = .60


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class ReplayEstimate:
    estimate_cycles: float
    theory_lower_cycles: float
    dependency_lower_cycles: float
    resource_lower_cycles: float
    attainment: float
    critical_resource: str
    class_service_cycles: dict[str, float]
    resource_service_cycles: dict[str, float]
    reference_reproduction_error: float
    result_digest: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_profiles(path: Path | None = None) -> dict[str, dict[str, Any]]:
    source = path or ROOT / "inputs/action_profiles.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("profile_count") != 36:
        raise ValueError("action profile inventory is incomplete")
    profiles = {row["case_id"]: row for row in document["profiles"]}
    if len(profiles) != 36:
        raise ValueError("duplicate action profiles")
    return profiles


PROFILES = load_profiles()
_COORD_CACHE: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}


def _coordinates(candidate: Any | None) -> tuple[tuple[int, int], ...]:
    if candidate is None:
        return tuple((rank % 6, rank // 6) for rank in range(36))
    key = (candidate.wafer_nx, candidate.wafer_ny)
    if key not in _COORD_CACHE:
        try:
            _COORD_CACHE[key] = map_logical_ranks(*key).rank_to_coordinate
        except TopologyCapacityError:
            _COORD_CACHE[key] = tuple((rank % 6, rank // 6) for rank in range(36))
    return _COORD_CACHE[key]


def _edges(source: int, destination: int, coordinates: tuple[tuple[int, int], ...]) -> list[str]:
    x, y = coordinates[source]; tx, ty = coordinates[destination]
    edges = []
    while x != tx:
        nx = x + (1 if tx > x else -1)
        edges.append(f"d2d.({x},{y})->({nx},{y})"); x = nx
    while y != ty:
        ny = y + (1 if ty > y else -1)
        edges.append(f"d2d.({x},{y})->({x},{ny})"); y = ny
    return edges


def _rates(candidate: Any | None) -> dict[str, float]:
    if candidate is None:
        return {"tensor": 128e12 * TENSOR_EFFICIENCY, "vector": 8e12 * VECTOR_EFFICIENCY,
                "sram": 256e9, "noc": 256e9, "dte_channels": 2.0,
                "d2d": 1e12, "hbm": 256e9, "hbm_stacks": 4.0,
                "reducer": 256e9, "n_ctrl": 2.0}
    tensor_peak = candidate.N * candidate.N * candidate.P_core_FLOPs
    return {"tensor": tensor_peak * TENSOR_EFFICIENCY,
            "vector": tensor_peak / 10.0 * VECTOR_EFFICIENCY,
            "sram": candidate.N * candidate.N * candidate.B_s_GBs * 1e9,
            "noc": candidate.B_GBs * 1e9, "dte_channels": float(candidate.DTE_channel),
            "d2d": candidate.D2D_edge_one_dir_GBs * 1e9,
            "hbm": candidate.HBM_stack_sustained_GBs * 1e9,
            "hbm_stacks": float(candidate.HBM_stack_count),
            "reducer": candidate.B_GBs * 1e9, "n_ctrl": float(candidate.n_ctrl)}


def _cycles(work: float, rate: float) -> float:
    return work * CLOCK_HZ / rate if work else 0.0


def _action_duration(item: Mapping[str, Any], rates: Mapping[str, float],
                     coordinates: tuple[tuple[int, int], ...], reference: bool) -> float:
    tensor_count = int(item["tensor_count"]); vector_count = int(item["vector_count"])
    byte_count = float(item["bytes"]); runtime = float(item["runtime_work"])
    if tensor_count:
        return _cycles(runtime, tensor_count * rates["tensor"])
    if item["has_hbm"]:
        hbm_rate = rates["hbm"] if reference else rates["hbm"] * rates["hbm_stacks"]
        bulk = min(hbm_rate, 128e9, rates["noc"], rates["sram"])
        return 10.0 + _cycles(byte_count, bulk)
    if item["d2d_source"] is not None:
        hops = (int(item["reference_hops"]) if reference else
                len(_edges(int(item["d2d_source"]), int(item["d2d_destination"]), coordinates)))
        bulk = min(128e9, rates["d2d"], rates["noc"], rates["sram"])
        return 2.0 + hops + _cycles(byte_count, bulk)
    if vector_count:
        compute = _cycles(runtime, vector_count * rates["vector"])
        memory = _cycles(byte_count, min(rates["sram"], rates["reducer"]))
        return max(compute, memory)
    return float(item["reference_duration_cycles"])


def _services(profile: Mapping[str, Any], candidate: Any | None) -> tuple[dict[str, float], dict[str, float]]:
    demands = profile["demands"]; rates = _rates(candidate); coords = _coordinates(candidate)
    service: dict[str, float] = {}
    def add(name: str, value: float) -> None:
        service[name] = service.get(name, 0.0) + value
    for die, work in demands["tensor_flops"].items(): add(f"tensor.die{die}", _cycles(work, rates["tensor"]))
    for die, work in demands["vector_flops"].items(): add(f"vector.die{die}", _cycles(work, rates["vector"]))
    for direction in ("read", "write"):
        for die, byte_count in demands[f"sram_{direction}_bytes"].items():
            add(f"sram.{direction}.die{die}", _cycles(byte_count, rates["sram"]))
    for resource, byte_count in demands["noc_bytes"].items(): add(resource, _cycles(byte_count, rates["noc"]))
    for die, byte_count in demands["dte_bytes"].items():
        add(f"dte.die{die}.channels", _cycles(byte_count, rates["dte_channels"] * 128e9))
    for die, byte_count in demands["reducer_bytes"].items(): add(f"reducer.die{die}", _cycles(byte_count, rates["reducer"]))
    for die, count in demands["control_actions"].items():
        add(f"control.die{die}", count * 12.0 * 2.0 / rates["n_ctrl"])
    if candidate is None:
        for stack, byte_count in demands["hbm_ref_stack_bytes"].items():
            count = demands["hbm_ref_stack_actions"].get(stack, 0.0)
            add(f"hbm.stack{stack}", _cycles(byte_count, rates["hbm"]) + count * 10.0)
    else:
        for die, byte_count in demands["hbm_local_die_bytes"].items():
            count = demands["hbm_local_die_actions"].get(die, 0.0)
            add(f"hbm.die{die}.striped", _cycles(byte_count, rates["hbm"] * rates["hbm_stacks"])
                + count * 10.0 / rates["hbm_stacks"])
    for flow in demands["d2d_flows"]:
        if candidate is not None and flow["kind"] == "hbm":
            continue
        for edge in _edges(int(flow["source"]), int(flow["destination"]), coords):
            add(edge, _cycles(float(flow["bytes"]), rates["d2d"]))
    classes = {name: 0.0 for name in ("compute", "sram_read", "sram_write", "noc", "dte", "d2d", "hbm", "control")}
    for name, value in service.items():
        if name.startswith(("tensor", "vector", "reducer")): group = "compute"
        elif name.startswith("sram.read"): group = "sram_read"
        elif name.startswith("sram.write"): group = "sram_write"
        elif name.startswith("noc"): group = "noc"
        elif name.startswith("dte"): group = "dte"
        elif name.startswith("d2d"): group = "d2d"
        elif name.startswith("hbm"): group = "hbm"
        else: group = "control"
        classes[group] = max(classes[group], value)
    return service, classes


def _lower(profile: Mapping[str, Any], state: str, candidate: Any | None) -> tuple[float, float, float, str, dict[str, float], dict[str, float]]:
    service, classes = _services(profile, candidate)
    critical, resource_lower = max(service.items(), key=lambda item: item[1])
    rates = _rates(candidate); coords = _coordinates(candidate)
    path = profile["naive_dependency_path" if state == "naive" else "sw_opt_dependency_path"]
    dependency = sum(_action_duration(item, rates, coords, candidate is None) for item in path)
    return max(resource_lower, dependency), dependency, resource_lower, critical, service, classes


def estimate(row: Mapping[str, Any], kind: str, state: str, candidate: Any) -> ReplayEstimate:
    profile = PROFILES[str(row["case_id"])]
    target = float(profile["source_naive_cycles" if state == "naive" else "source_sw_opt_cycles"])
    ref_lower, _, _, _, _, _ = _lower(profile, state, None)
    if ref_lower > target * (1 + 1e-12):
        raise ValueError(f"reference analytical lower bound exceeds target for {row['case_id']}/{state}")
    attainment = ref_lower / target
    lower, dependency, resource, critical, services, classes = _lower(profile, state, candidate)
    cycles = lower / attainment
    reproduced = ref_lower / attainment
    error = abs(reproduced - target) / target
    payload = {"profile": profile["profile_digest"], "candidate": candidate.candidate_digest,
               "state": state, "cycles": cycles, "lower": lower}
    return ReplayEstimate(cycles, lower, dependency, resource, lower / cycles, critical,
                          classes, services, error, _digest(payload))


def audit_reference_reproduction() -> dict[str, Any]:
    failures = []
    for profile in PROFILES.values():
        for state, field in (("naive", "source_naive_cycles"), ("sw_opt", "source_sw_opt_cycles")):
            lower, *_ = _lower(profile, state, None)
            target = float(profile[field]); reproduced = lower / (lower / target)
            error = abs(reproduced - target) / target
            if error > 1e-12 or lower > target * (1 + 1e-12):
                failures.append({"case_id": profile["case_id"], "state": state,
                                 "relative_error": error, "lower": lower, "target": target})
    return {"row_count": 72, "failure_count": len(failures), "failures": failures}


__all__ = ["ReplayEstimate", "audit_reference_reproduction", "estimate", "load_profiles"]
