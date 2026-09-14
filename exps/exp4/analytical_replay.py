#!/usr/bin/env python3
"""Compressed, resource-explicit replay for the exp4 hardware sweep.

The frozen exp2 per-resource ledgers are treated as work counters measured at
the exp2 reference rates.  We recover service demand per resource class and
re-serve it at each candidate rate.  This is deliberately not a whole-program
peak-rate scaling: compute, SRAM, NoC, DTE, D2D, HBM and control remain
separate bottlenecks and software overlap changes their composition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


CLOCK_HZ = 500_000_000
EXP2_TENSOR_TFLOPS_PER_DIE = 128.0
EXP2_VECTOR_TFLOPS_PER_DIE = 8.0
EXP2_BW_GBS = {"sram": 256.0, "noc": 256.0, "dte": 128.0,
               "d2d": 1000.0, "hbm": 256.0, "reducer": 256.0}


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def load_exp2_resource_rows(exp2_root: Path) -> dict[str, dict[str, Any]]:
    """Return raw frozen rows keyed by case_id without mutating exp2."""
    names = ("training_e2e.json", "inference_prefill_pd_breakdown.json",
             "inference_decode_e2e.json")
    rows: dict[str, dict[str, Any]] = {}
    for name in names:
        value = json.loads((exp2_root / "results" / name).read_text(encoding="utf-8"))
        for row in value:
            case_id = str(row["case_id"])
            if case_id in rows:
                raise ValueError(f"duplicate exp2 case_id: {case_id}")
            rows[case_id] = row
    # Exp2's published prefill breakdown intentionally omitted the resource
    # ledger. Its same-model/same-sequence training forward DAG is the closest
    # frozen counter source, so use that class mix and calibrate it separately
    # to the exact prefill base/overlap cycles. This is an explicit analytical
    # surrogate, not fabricated cycle evidence.
    training = {
        (r["model_id"], int(r["seq_len"])): r
        for r in rows.values()
        if str(r["case_id"]).startswith("train__")
    }
    for row in rows.values():
        if not str(row["case_id"]).startswith("prefill_pd__"):
            continue
        surrogate = training[(row["model_id"], int(row["seq_len"]))]
        ledger = surrogate["base_resource_service_cycles"]
        row["base_resource_service_cycles"] = ledger
        row["overlap_resource_service_cycles"] = ledger
        row["resource_ledger_provenance"] = surrogate["result_digest"]
    return rows


def _ledger(row: Mapping[str, Any], kind: str, state: str) -> dict[str, float]:
    if state == "naive":
        field = "base_resource_service_cycles"
    elif kind == "training":
        field = "full_train_resource_service_cycles"
    else:
        field = "overlap_resource_service_cycles"
    value = row.get(field)
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"missing {field} for {row.get('case_id')}")
    return {str(key): float(cycles) for key, cycles in value.items()}


def _die_id(resource: str) -> str:
    match = re.search(r"die(\d+)", resource)
    return match.group(1) if match else "global"


@dataclass(frozen=True)
class ReplayEstimate:
    estimate_cycles: float
    theory_lower_cycles: float
    attainment: float
    critical_resource: str
    class_service_cycles: dict[str, float]
    resource_service_cycles: dict[str, float]
    calibration_factor: float
    result_digest: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rates(hardware: Any | None) -> dict[str, float]:
    if hardware is None:
        return {
            "tensor": EXP2_TENSOR_TFLOPS_PER_DIE,
            "vector": EXP2_VECTOR_TFLOPS_PER_DIE,
            "sram": EXP2_BW_GBS["sram"], "noc": 256.0,
            "dte": 128.0, "dte_channels": 2.0, "d2d": 1000.0,
            "hbm": 256.0, "hbm_stacks": 4.0, "reducer": 256.0,
            "control_parallelism": 2.0,
        }
    tensor = float(hardware.N * hardware.N * hardware.P_core_TFLOPs)
    return {
        "tensor": tensor,
        "vector": tensor / 10.0,
        # Exp2 exposes per-die SRAM resources. Candidate B_s is per core, so
        # the compressed ledger shares each direction across N^2 core budgets.
        "sram": float(hardware.N * hardware.N * hardware.B_s_GBs),
        "noc": float(hardware.B_GBs),
        "dte": 128.0,
        "dte_channels": float(hardware.DTE_channel),
        "d2d": float(hardware.D2D_edge_one_dir_GBs),
        "hbm": float(hardware.HBM_stack_sustained_GBs),
        "hbm_stacks": float(36 * hardware.HBM_stack_count),
        "reducer": float(hardware.B_GBs),
        "control_parallelism": float(hardware.n_ctrl),
    }


def _serve(ledger: Mapping[str, float], rates: Mapping[str, float]) -> dict[str, float]:
    served: dict[str, float] = {}
    sram: dict[tuple[str, str], float] = {}
    dte: dict[str, float] = {}
    hbm_total = 0.0
    for resource, cycles in ledger.items():
        if resource.startswith("tensor."):
            served[resource] = cycles * EXP2_TENSOR_TFLOPS_PER_DIE / rates["tensor"]
        elif resource.startswith("vector."):
            served[resource] = cycles * EXP2_VECTOR_TFLOPS_PER_DIE / rates["vector"]
        elif resource.startswith("sram"):
            direction = "read" if "read" in resource else "write"
            key = (_die_id(resource), direction)
            sram[key] = sram.get(key, 0.0) + cycles
        elif resource.startswith("dte."):
            die = _die_id(resource)
            dte[die] = dte.get(die, 0.0) + cycles
        elif resource.startswith("d2d."):
            served[resource] = cycles * EXP2_BW_GBS["d2d"] / rates["d2d"]
        elif resource.startswith("hbm.stack") or resource.startswith("hbm.append"):
            hbm_total += cycles
        elif resource.startswith("hbm.ingress"):
            # Physical HBM stack service is accounted above; ingress remains a
            # topology/NoC stage and is therefore served at candidate NoC rate.
            served[resource] = cycles * EXP2_BW_GBS["noc"] / rates["noc"]
        elif resource.startswith("noc."):
            served[resource] = cycles * EXP2_BW_GBS["noc"] / rates["noc"]
        elif resource.startswith("reducer."):
            served[resource] = cycles * EXP2_BW_GBS["reducer"] / rates["reducer"]
        elif resource.startswith("control."):
            served[resource] = cycles * 2.0 / rates["control_parallelism"]
        else:
            served[resource] = cycles
    for (die, direction), cycles in sram.items():
        served[f"sram.{direction}.die{die}.shared"] = cycles * EXP2_BW_GBS["sram"] / rates["sram"]
    for die, cycles in dte.items():
        served[f"dte.die{die}.independent_channels"] = (
            cycles * 2.0 / rates["dte_channels"]
        )
    if hbm_total:
        # Recover aggregate bytes from the four-stack exp2 ledger, then stripe
        # over all physical stacks of the active 36-module replica.
        served["hbm.physical_stacks.striped"] = (
            hbm_total * EXP2_BW_GBS["hbm"] /
            (rates["hbm"] * rates["hbm_stacks"])
        )
    return served


def _classes(served: Mapping[str, float]) -> dict[str, float]:
    groups = {name: 0.0 for name in
              ("compute", "sram_read", "sram_write", "noc", "dte", "d2d", "hbm", "control")}
    for resource, cycles in served.items():
        if resource.startswith(("tensor.", "vector.", "reducer.")):
            group = "compute"
        elif resource.startswith("sram.read"):
            group = "sram_read"
        elif resource.startswith("sram.write"):
            group = "sram_write"
        elif resource.startswith("noc.") or resource.startswith("hbm.ingress"):
            group = "noc"
        elif resource.startswith("dte."):
            group = "dte"
        elif resource.startswith("d2d."):
            group = "d2d"
        elif resource.startswith("hbm."):
            group = "hbm"
        else:
            group = "control"
        groups[group] = max(groups[group], cycles)
    return groups


def _proxy(classes: Mapping[str, float], state: str) -> float:
    compute = classes["compute"]
    memory = max(classes["sram_read"], classes["sram_write"], classes["hbm"])
    fabric = classes["noc"] + classes["dte"] + classes["d2d"]
    control = classes["control"]
    if state == "naive":
        return compute + memory + fabric + control
    # Inter-/intra-die software scheduling overlaps the three major pipelines;
    # physical resources within the fabric remain serialized.
    return max(compute, memory, fabric) + control


def estimate(row: Mapping[str, Any], kind: str, state: str, hardware: Any) -> ReplayEstimate:
    ledger = _ledger(row, kind, state)
    target_cycles = float(row["T_base_cycles"] if state == "naive" else
                          (row["T_full_train_overlap_cycles"] if kind == "training" else row["T_overlap_cycles"]))
    ref_served = _serve(ledger, _rates(None))
    ref_classes = _classes(ref_served)
    ref_proxy = _proxy(ref_classes, state)
    calibration = target_cycles / ref_proxy if ref_proxy > 0 else 1.0
    served = _serve(ledger, _rates(hardware))
    classes = _classes(served)
    critical_resource, lower = max(served.items(), key=lambda item: item[1])
    estimate_cycles = max(lower, calibration * _proxy(classes, state))
    if state == "sw_opt":
        # Anchor software benefit exactly at exp2, then adjust it by how much
        # independent compute/memory/fabric work is available to overlap.
        naive_cycles = estimate(row, kind, "naive", hardware).estimate_cycles
        def opportunity(values: Mapping[str, float]) -> float:
            stages = (values["compute"],
                      max(values["sram_read"], values["sram_write"], values["hbm"]),
                      values["noc"] + values["dte"] + values["d2d"])
            peak = max(stages)
            return (sum(stages) - peak) / peak if peak else 0.0
        reference = max(opportunity(ref_classes), 1e-12)
        opportunity_scale = min(1.5, max(0.5, opportunity(classes) / reference))
        exp2_speedup = float(row["T_base_cycles"]) / target_cycles
        adjusted_speedup = max(1.0, 1.0 + (exp2_speedup - 1.0) * opportunity_scale)
        estimate_cycles = max(lower, naive_cycles / adjusted_speedup)
    attainment = lower / estimate_cycles if estimate_cycles else 1.0
    payload = {"case_id": row["case_id"], "candidate": hardware.candidate_digest,
               "state": state, "cycles": estimate_cycles, "services": served}
    return ReplayEstimate(estimate_cycles, lower, attainment, critical_resource,
                          classes, served, calibration, canonical_digest(payload))


__all__ = ["CLOCK_HZ", "ReplayEstimate", "estimate", "load_exp2_resource_rows"]
