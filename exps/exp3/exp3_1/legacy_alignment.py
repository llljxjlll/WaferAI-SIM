#!/usr/bin/env python3
"""Compatibility bridge from Exp3.1 to the audited Exp1 estimators.

The functions in this module intentionally do not reproduce either legacy
cost model.  The Exp1.1 and Exp1.2 Python entry points are loaded from their
source files and called directly.  This keeps the four native states used by
Exp3.1 tied to the historical implementations and makes drift detectable by
``compatibility_audit``.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterator, Mapping
from inter_port_model import core_to_d2d_port_metrics, stream_port_and_fabric



_HERE = Path(__file__).resolve().parent
_EXPS = _HERE.parents[1]
_EXP1_1 = _EXPS / "exp1" / "exp1_1"
_EXP1_2 = _EXPS / "exp1" / "exp1_2"
_MISSING = object()
_STATES = ("W00", "W10", "W01", "W11")


@contextmanager
def _temporary_modules(aliases: Mapping[str, ModuleType]) -> Iterator[None]:
    previous = {name: sys.modules.get(name, _MISSING) for name in aliases}
    sys.modules.update(aliases)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value  # type: ignore[assignment]


def _load_source(name: str, path: Path, aliases: Mapping[str, ModuleType] | None = None) -> ModuleType:
    """Load one legacy source file under a collision-free module name."""
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        with _temporary_modules(aliases or {}):
            spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@lru_cache(maxsize=1)
def _dense_modules() -> tuple[ModuleType, ModuleType]:
    runner = _load_source(
        "_exp3_1_legacy_exp1_1_run_experiment",
        _EXP1_1 / "run_experiment.py",
    )
    estimator = _load_source(
        "_exp3_1_legacy_exp1_1_estimate_trace_replay",
        _EXP1_1 / "estimate_trace_replay.py",
        {"run_experiment": runner},
    )
    return runner, estimator


@lru_cache(maxsize=1)
def _moe_module() -> ModuleType:
    physical = _load_source(
        "_exp3_1_legacy_exp1_2_physical_model",
        _EXP1_2 / "physical_model.py",
    )
    return _load_source(
        "_exp3_1_legacy_exp1_2_run_experiment",
        _EXP1_2 / "run_experiment.py",
        {"physical_model": physical},
    )


def _value(case: object, *names: str, default: object = _MISSING) -> object:
    for name in names:
        if isinstance(case, Mapping) and name in case:
            return case[name]
        if hasattr(case, name):
            return getattr(case, name)
    if default is not _MISSING:
        return default
    raise ValueError(f"case is missing one of: {', '.join(names)}")


def _text(value: object) -> str:
    if hasattr(value, "name"):
        value = getattr(value, "name")
    return str(value)


def _canonical_model(value: object, *, moe: bool) -> str:
    key = _text(value).lower().replace("×", "x").replace("_", "-")
    if "mixtral" in key:
        return "Mixtral-8x7B"
    if "deepseek" in key:
        return "DeepSeek-V3"
    if not moe and "llama" in key and ("2" in key or "7b" in key):
        return "LLaMA-2-7B"
    if not moe and "gpt" in key and ("175" in key or "3" in key):
        return "GPT-3-175B"
    family = "MoE" if moe else "dense"
    raise ValueError(f"unsupported {family} model: {value!r}")


def _dense_layer(case: object) -> str:
    raw = _value(case, "layer", "stage", "projection", default="")
    text = _text(raw).lower().replace("-", "_")
    if any(token in text for token in ("attention", "o_proj", "oproj")):
        return "attention"
    if any(token in text for token in ("mlp", "down_proj", "downproj")):
        return "mlp"
    case_id = _text(_value(case, "case_id", default="")).lower()
    if "o_proj" in case_id or "attention" in case_id:
        return "attention"
    if "down" in case_id or "mlp" in case_id:
        return "mlp"
    raise ValueError(f"cannot infer dense layer from {raw!r}")


def _moe_operator(case: object) -> str:
    raw = _value(case, "operator", "stage", "layer", default="")
    text = _text(raw).upper().replace("-", "_").replace("+", "_")
    if "DISPATCH" in text or "GATE" in text or text == "UP":
        return "DISPATCH_GEMM"
    if "COMBINE" in text or "DOWN" in text:
        return "GEMM_COMBINE"
    case_id = _text(_value(case, "case_id", default="")).upper()
    if "DISPATCH" in case_id or "GATE" in case_id:
        return "DISPATCH_GEMM"
    if "COMBINE" in case_id or "DOWN" in case_id:
        return "GEMM_COMBINE"
    raise ValueError(f"cannot infer MoE stage from {raw!r}")


def _sequence_length(case: object) -> int:
    value = int(_value(case, "seq_len", "S", "sequence_length"))
    if value <= 0:
        raise ValueError("sequence length must be positive")
    return value


def _mesh(case: object) -> tuple[int, int]:
    px = _value(case, "px", "Px", default=None)
    py = _value(case, "py", "Py", default=None)
    if px is not None and py is not None:
        result = int(px), int(py)
    else:
        mesh = _value(case, "mesh", default=None)
        if mesh is not None and hasattr(mesh, "rows") and hasattr(mesh, "columns"):
            result = int(getattr(mesh, "rows")), int(getattr(mesh, "columns"))
        else:
            dies = int(_value(case, "D", "dies", "die_count"))
            known = {4: (1, 4), 6: (2, 3), 9: (3, 3), 36: (6, 6)}
            if dies not in known:
                raise ValueError(f"mesh dimensions are required for D={dies}")
            result = known[dies]
    if min(result) <= 0:
        raise ValueError("mesh dimensions must be positive")
    dies = _value(case, "D", "dies", "die_count", default=result[0] * result[1])
    if int(dies) != result[0] * result[1]:
        raise ValueError("D does not match Px*Py")
    return result


def _find_named(items: object, name: str) -> object:
    for item in items:  # type: ignore[union-attr]
        if getattr(item, "name") == name:
            return item
    raise ValueError(f"legacy profile not found: {name}")


def _state_cycles(record: Mapping[str, object]) -> dict[str, int]:
    return {state: int(record[f"T{state[1:]}_cycles"]) for state in _STATES}


def _dense_stages(estimator: ModuleType, legacy_case: object) -> dict[str, object]:
    shape = estimator.normalize(legacy_case)
    stages = estimator._analytical_stages(legacy_case, shape)
    keep = (
        "compute_naive", "compute_ideal", "compute_opt", "communication",
        "hbm", "local_transport", "reduce", "intra_pm", "intra_pn",
        "intra_pk", "active_cores", "spatial_utilization",
        "compute_efficiency",
    )
    return {
        "runtime_shape": {
            "M": int(shape["runtime_M"]),
            "N": int(shape["runtime_N"]),
            "K": int(shape["runtime_K"]),
            "rank_N": int(shape["rank_N"]),
            "rank_K": int(shape["rank_K"]),
        },
        **{key: stages[key] for key in keep},
    }


def _dense_native_exp1_reference(case: object) -> dict[str, object]:
    """Return Exp1.1-native W00/W10/W01/W11 cycles and stage details.

    ``case`` may be an Exp3.1 dataclass or a mapping.  Expected semantic fields
    are model, layer/stage, sequence length and either D or Px/Py.  The legacy
    estimator is called with no optional trace calibration, matching the
    committed Exp1.1 historical JSON.
    """
    runner, estimator = _dense_modules()
    model_name = _canonical_model(_value(case, "model", "model_or_moe_config"), moe=False)
    layer = _dense_layer(case)
    seq_len = _sequence_length(case)
    px, py = _mesh(case)
    model = _find_named(runner.MODELS, model_name)
    mesh = runner.MeshSpec(f"{px}x{py}", px, py, (seq_len, seq_len))
    legacy_case = runner.make_case(mesh, "GEMM_RS", model, layer, seq_len)
    record = estimator.estimate_case(legacy_case, {})
    cycles = _state_cycles(record)
    return {
        **cycles,
        "state_cycles": cycles,
        "stages": _dense_stages(estimator, legacy_case),
        "case_id": record["case_id"],
        "model": model_name,
        "layer": layer,
        "operator": "GEMM_RS",
        "D": px * py,
        "Px": px,
        "Py": py,
        "S": seq_len,
        "cycle_time_ns": 1.0,
        "evidence": record["estimate_source"],
        "legacy_source": str(_EXP1_1 / "estimate_trace_replay.py"),
        "legacy_record": record,
    }

def _dense_intra_schedule(
    estimator: ModuleType, local_noc_model: ModuleType, shape: Mapping[str, int],
    flops_per_die: float, pm: int, pn: int, pk: int, *, optimize_mapping: bool,
) -> dict[str, object]:
    """Evaluate one full-16-core Dense intra-die mapping.

    The traffic equations are the Exp1.1 GEMM equations. The directed-link
    occupancy is replayed by Exp1.2's existing 4x4 core-mesh router so the
    ablation observes local congestion, rather than treating total byte-hops
    as a fictitious serial link.
    """
    cores = int(estimator.DIE_CORES)
    if pm * pn * pk != cores:
        raise ValueError("Dense intra mapping must occupy all 16 cores")
    tm, tn = int(shape["Tm"]), int(shape["Tn"])
    if pm > tm or pn > tn:
        raise ValueError("Dense intra mapping exceeds the tiled M or N extent")
    output_tiles = tm * tn
    spatial_slots = math.ceil(tm / pm) * pm * math.ceil(tn / pn) * pn
    spatial_utilization = output_tiles / spatial_slots
    a_broadcast = (
        output_tiles * (pn - 1) * estimator.TILE_M * estimator.TILE_K
        * estimator.DTYPE_BYTES
    )
    b_broadcast = (
        output_tiles * (pm - 1) * estimator.TILE_K * estimator.TILE_N
        * estimator.DTYPE_BYTES
    )
    reduction = output_tiles * (pk - 1) * estimator.TILE_M * estimator.TILE_N * 4
    noc = local_noc_model._local_noc_route_metrics(
        1, pm, pn, pk, a_broadcast, b_broadcast, reduction,
        optimize_mapping=optimize_mapping,
    )
    compute_efficiency = (
        0.92 - 0.015 * math.log2(pk)
    ) * spatial_utilization
    compute_cycles = (
        flops_per_die
        / (cores * estimator.CORE_FLOPS * compute_efficiency)
        * estimator.CLOCK_HZ
    )
    local_transport_cycles = (
        float(noc["local_noc_max_directed_link_bytes"])
        / estimator.NOC_BPS * estimator.CLOCK_HZ
    )
    return {
        "intra_pm": pm,
        "intra_pn": pn,
        "intra_pk": pk,
        "active_cores": cores,
        "spatial_utilization": spatial_utilization,
        "compute_efficiency": compute_efficiency,
        "compute_cycles": compute_cycles,
        "a_broadcast_bytes": a_broadcast,
        "b_broadcast_bytes": b_broadcast,
        "reduction_bytes": reduction,
        "local_transport_bytes": a_broadcast + b_broadcast + reduction,
        "local_transport_cycles": local_transport_cycles,
        **noc,
    }


def _dense_optimized_overlap_efficiency(
    shape: Mapping[str, int], schedule: Mapping[str, object],
) -> float:
    """Retain the Exp1.1 steady-state penalty for the optimized schedule."""
    qfull = int(shape["Tm"]) * int(shape["Tn"])
    wave_fill_efficiency = qfull / (qfull + 256.0)
    split_k_penalty = 0.006 * math.log2(int(schedule["intra_pk"]))
    fanout_penalty = 0.0015 * (
        int(schedule["intra_pm"]) + int(schedule["intra_pn"]) - 2
    )
    spatial_penalty = 0.04 * (1.0 - float(schedule["spatial_utilization"]))
    return max(
        0.70,
        min(
            0.83,
            0.765 + 0.065 * wave_fill_efficiency
            - split_k_penalty - fanout_penalty - spatial_penalty,
        ),
    )


def _fixed_16core_port_control(**kwargs: object) -> dict[str, object]:
    """Compose C00/C10 with fixed core mapping and explicit D2D gateways."""
    values = kwargs
    local_default = float(values.get("local_cycles", 0.0))
    local_off = float(values.get("local_off_cycles", local_default))
    local_on = float(values.get("local_on_cycles", local_default))
    waves = max(1, int(values["waves"]))
    port = core_to_d2d_port_metrics(
        float(values["payload_bytes_per_die"]),
        noc_link_bps=float(values["noc_link_bps"]),
        dte_channel_bps=float(values["dte_channel_bps"]),
        clock_hz=float(values["clock_hz"]),
    )
    communication_off, _ = stream_port_and_fabric(
        port, fabric_cycles=float(values["fabric_off_cycles"]), waves=waves,
    )
    _, communication_on = stream_port_and_fabric(
        port, fabric_cycles=float(values["fabric_on_cycles"]), waves=waves,
    )
    balance = min(local_on, communication_on) / max(local_on, communication_on, 1.0)
    tail = 0.012 + 0.028 * balance + 0.010 * float(values["mesh_span"])
    c00 = max(1, round(local_off + communication_off))
    c10 = max(1, round(
        max(local_on, communication_on)
        + min(local_on, communication_on) * (1.0 / waves + tail)
    ))
    return {
        "state_cycles": {"C00": c00, "C10": c10},
        "local_off_cycles": local_off,
        "local_on_cycles": local_on,
        "port": port,
        "communication_off_cycles": communication_off,
        "communication_on_cycles": communication_on,
        "overlap_tail": tail,
    }

def _port_control_stage(
    baseline_stage: Mapping[str, object], *, communication_cycles: float,
    fabric_cycles: float, gateway_cycles: float, composition: str,
    overlap_tail: float | None, port: Mapping[str, object],
    local_cycles: float | None = None, hbm_base_cycles: float | None = None,
) -> dict[str, object]:
    """Attach a shared 16-core core-to-port replay to a C-state stage."""
    local = float(baseline_stage["compute_cycles"]) if local_cycles is None else float(local_cycles)
    stage = dict(baseline_stage)
    stage.update({
        "compute_cycles": local,
        "communication_cycles": communication_cycles,
        "composition": composition,
        "overlap_tail": overlap_tail,
        "controlled_total_cycles": None,
        "inter_port_cycles": gateway_cycles,
        "inter_port": {
            **dict(port),
            "fabric_cycles": fabric_cycles,
            "port_pipeline_cycles": gateway_cycles,
            "combined_communication_cycles": communication_cycles,
        },
    })
    if hbm_base_cycles is not None:
        stage.update({
            "hbm_base_cycles": float(hbm_base_cycles),
            "hbm_cycles": float(hbm_base_cycles),
        })
    return stage


def _mesh_span_for_d(D: int) -> float:
    """Return the normalized span of the evaluated die mesh."""
    known = {4: (2, 2), 6: (2, 3), 9: (3, 3), 36: (6, 6)}
    px, py = known.get(int(D), (math.isqrt(int(D)), math.isqrt(int(D))))
    if px * py != int(D):
        px, py = math.ceil(math.sqrt(D)), math.ceil(math.sqrt(D))
    return min(1.0, (px + py - 2) / 10.0)


def _dense_controlled_inter(case: object) -> dict[str, object]:
    """Return all fixed-16-core Dense states used by the paired metrics.

    W00/W11 compare a canonical 16-core baseline against an optimized 16-core
    intra+inter implementation.  C00/C10 retain the canonical implementation
    and change only the inter-die communication schedule.
    """
    reference = _dense_native_exp1_reference(case)
    _, estimator = _dense_modules()
    local_noc_model = _moe_module()
    legacy_record = reference["legacy_record"]
    legacy_stages = reference["stages"]
    if not isinstance(legacy_record, Mapping) or not isinstance(legacy_stages, Mapping):
        raise ValueError("Dense Exp1 reference lacks legacy stage details")
    px, py = int(reference["Px"]), int(reference["Py"])
    model_name, layer, seq_len = (
        str(reference["model"]), str(reference["layer"]), int(reference["S"]),
    )
    runner, _ = _dense_modules()
    model = _find_named(runner.MODELS, model_name)
    mesh = runner.MeshSpec(f"{px}x{py}", px, py, (seq_len, seq_len))
    legacy_case = runner.make_case(mesh, "GEMM_RS", model, layer, seq_len)
    shape = estimator.normalize(legacy_case)
    flops_per_die = (
        2.0 * int(shape["runtime_M"]) * int(shape["runtime_N"])
        * int(shape["runtime_K"]) / (px * py)
    )
    baseline_schedule = _dense_intra_schedule(
        estimator, local_noc_model, shape, flops_per_die, 4, 4, 1,
        optimize_mapping=False,
    )
    optimized_schedule = _dense_intra_schedule(
        estimator, local_noc_model, shape, flops_per_die,
        int(legacy_stages["intra_pm"]), int(legacy_stages["intra_pn"]),
        int(legacy_stages["intra_pk"]), optimize_mapping=True,
    )
    hbm = float(legacy_stages["hbm"])
    baseline_local = (
        float(baseline_schedule["compute_cycles"])
        + hbm + float(baseline_schedule["local_transport_cycles"])
    )
    optimized_overlap_efficiency = _dense_optimized_overlap_efficiency(
        shape, optimized_schedule,
    )
    optimized_local = max(
        float(optimized_schedule["compute_cycles"]), hbm,
        float(optimized_schedule["local_transport_cycles"]),
    ) / optimized_overlap_efficiency
    communication = float(legacy_stages["communication"])
    fused_communication = communication / (
        float(legacy_record["segmented_comm_efficiency"])
        * float(legacy_record["topology_efficiency"])
    )
    qfull = int(shape["Tm"]) * int(shape["Tn"])
    mesh_span = min(1.0, (px + py - 2) / 10.0)

    payload_bytes = communication / estimator.CLOCK_HZ * estimator.NOC_BPS
    port_control = _fixed_16core_port_control(
        local_off_cycles=baseline_local, local_on_cycles=baseline_local, fabric_off_cycles=communication,
        fabric_on_cycles=fused_communication, payload_bytes_per_die=payload_bytes,
        noc_link_bps=estimator.NOC_BPS, dte_channel_bps=estimator.NOC_BPS,
        clock_hz=estimator.CLOCK_HZ, waves=qfull, mesh_span=mesh_span,
    )
    optimized_port_control = _fixed_16core_port_control(
        local_off_cycles=optimized_local, local_on_cycles=optimized_local,
        fabric_off_cycles=communication, fabric_on_cycles=fused_communication,
        payload_bytes_per_die=payload_bytes, noc_link_bps=estimator.NOC_BPS,
        dte_channel_bps=estimator.NOC_BPS, clock_hz=estimator.CLOCK_HZ,
        waves=qfull, mesh_span=mesh_span,
    )
    port_communication_off = float(port_control["communication_off_cycles"])
    port_communication_on = float(port_control["communication_on_cycles"])
    port_metrics = port_control["port"]
    if not isinstance(port_metrics, Mapping):
        raise ValueError("Dense controlled port replay lacks metrics")
    baseline_tail = float(port_control["overlap_tail"])
    optimized_tail = float(optimized_port_control["overlap_tail"])

    cycles = {
        "W00": int(port_control["state_cycles"]["C00"]),
        "W10": int(port_control["state_cycles"]["C10"]),
        "W01": int(optimized_port_control["state_cycles"]["C00"]),
        "W11": int(optimized_port_control["state_cycles"]["C10"]),
    }

    def state_stage(
        schedule: Mapping[str, object], local_cycles: float,
        comm_cycles: float, composition: str, tail: float | None,
    ) -> dict[str, object]:
        return {
            "compute_cycles": local_cycles,
            "communication_cycles": comm_cycles,
            "composition": composition,
            "overlap_tail": tail,
            "local_compute_cycles": schedule["compute_cycles"],
            "hbm_cycles": hbm,
            "local_transport_cycles": schedule["local_transport_cycles"],
            "intra_schedule": dict(schedule),
        }

    state_stages = {
        "W00": state_stage(
            baseline_schedule, baseline_local, communication, "serial", None,
        ),
        "W10": state_stage(
            baseline_schedule, baseline_local, fused_communication, "overlap",
            baseline_tail,
        ),
        "W01": state_stage(
            optimized_schedule, optimized_local, communication, "serial", None,
        ),
        "W11": state_stage(
            optimized_schedule, optimized_local, fused_communication, "overlap",
            optimized_tail,
        ),
    }
    def port_stage(
        schedule: Mapping[str, object], local_cycles: float, comm_cycles: float, fabric_cycles: float,
        gateway_cycles: float, composition: str, tail: float | None,
    ) -> dict[str, object]:
        return {
            "compute_cycles": local_cycles,
            "communication_cycles": comm_cycles,
            "composition": composition,
            "overlap_tail": tail,
            "local_compute_cycles": schedule["compute_cycles"],
            "hbm_base_cycles": hbm,
            "hbm_cycles": hbm,
            "local_transport_cycles": schedule["local_transport_cycles"],
            "intra_schedule": dict(schedule),
            "inter_port_cycles": gateway_cycles,
            "inter_port": {
                **dict(port_metrics),
                "fabric_cycles": fabric_cycles,
                "port_pipeline_cycles": gateway_cycles,
                "combined_communication_cycles": comm_cycles,
            },
        }

    state_stages["W00"] = port_stage(
        baseline_schedule, baseline_local, port_communication_off, communication,
        float(port_metrics["gateway_serial_cycles"]),
        "serial_core_to_port_fabric", None,
    )
    state_stages["W10"] = port_stage(
        baseline_schedule, baseline_local, port_communication_on, fused_communication,
        float(port_metrics["gateway_streaming_cycles"]),
        "overlap_core_to_port_fabric", float(port_control["overlap_tail"]),
    )
    state_stages["W01"] = port_stage(
        optimized_schedule, optimized_local,
        float(optimized_port_control["communication_off_cycles"]), communication,
        float(port_metrics["gateway_serial_cycles"]),
        "serial_core_to_port_fabric", None,
    )
    state_stages["W11"] = port_stage(
        optimized_schedule, optimized_local,
        float(optimized_port_control["communication_on_cycles"]), fused_communication,
        float(port_metrics["gateway_streaming_cycles"]),
        "overlap_core_to_port_fabric", float(optimized_port_control["overlap_tail"]),
    )
    return {
        **cycles,
        "state_cycles": cycles,
        "stages": legacy_stages,
        "state_stages": state_stages,
        "intra_ablation": {
            "baseline": dict(baseline_schedule),
            "optimized": dict(optimized_schedule),
            "optimized_overlap_efficiency": optimized_overlap_efficiency,
            "baseline_mapping": "fixed_4x4x1_canonical_order",
            "optimized_mapping": "adaptive_16core_mapping_with_min_max_local_noc_link",
            "inter_port_model": dict(port_metrics),
            "dte_port_mapping": "fixed_two_edge_ports_with_x_first_core_routes",
        },
        **{
            key: reference[key] for key in (
                "case_id", "model", "layer", "operator", "D", "Px", "Py", "S",
                "cycle_time_ns", "legacy_source", "legacy_record",
            )
        },
        "evidence": "analytical_fixed_16core_core_to_d2d_port_full_and_inter_replay_from_exp1",
    }

def dense_native(case: object) -> dict[str, object]:
    """Return fixed-16-core full and inter-only tracks for Dense GEMM."""
    reference = _dense_native_exp1_reference(case)
    controlled = _dense_controlled_inter(case)
    controlled_cycles = controlled["state_cycles"]
    controlled_stages = controlled["state_stages"]
    if not isinstance(controlled_cycles, Mapping) or not isinstance(controlled_stages, Mapping):
        raise ValueError("Dense controlled-inter replay lacks state details")
    return {
        **reference,
        **{state: int(controlled_cycles[state]) for state in _STATES},
        "state_cycles": {
            state: int(controlled_cycles[state]) for state in _STATES
        },
        "state_stages": {state: dict(controlled_stages[state]) for state in _STATES},
        "controlled_state_cycles": {
            "C00": int(controlled_cycles["W00"]),
            "C10": int(controlled_cycles["W10"]),
        },
        "controlled_state_stages": {
            "C00": dict(controlled_stages["W00"]),
            "C10": dict(controlled_stages["W10"]),
        },
        "intra_ablation": controlled["intra_ablation"],
        "controlled_evidence": "analytical_fixed_16core_core_to_d2d_port_inter_only_replay_from_exp1",
        "evidence": "analytical_fixed_16core_core_to_d2d_port_full_and_inter_replay_from_exp1",
        "full_reference_evidence": "exp1_1_direct_reference_retained_for_compatibility_audit",
    }

def _validate_moe_profile(case: object, model: object) -> None:
    expected = {
        "hidden_size": int(getattr(model, "hidden_size")),
        "expert_intermediate_size": int(getattr(model, "expert_intermediate_size")),
        "expert_count": int(getattr(model, "expert_count")),
        "top_k": int(getattr(model, "top_k")),
    }
    aliases = {
        "hidden_size": ("hidden_size", "H"),
        "expert_intermediate_size": (
            "expert_intermediate_size", "intermediate_size", "I",
        ),
        "expert_count": ("expert_count", "experts", "E"),
        "top_k": ("top_k", "topk"),
    }
    for field, names in aliases.items():
        actual = _value(case, *names, default=None)
        if actual is not None and int(actual) != expected[field]:
            raise ValueError(
                f"{getattr(model, 'name')} must use real-model {field}="
                f"{expected[field]}, got {actual}"
            )


def _moe_stage_breakdown(record: Mapping[str, object]) -> dict[str, dict[str, object]]:
    naive = float(record["intra_naive_cycles"])
    optimized = float(record["intra_optimized_cycles"])
    unfused_comm = float(record["comm_unfused_cycles"])
    fused_comm = float(record["comm_fused_stream_cycles"]) + float(record["fusion_setup_cycles"])
    baseline_schedule = {
        "active_cores": int(record["baseline_active_cores"]),
        "intra_pe": int(record["baseline_intra_pe"]),
        "intra_pm": int(record["baseline_intra_pm"]),
        "intra_pn": int(record["baseline_intra_pn"]),
        "intra_pk": int(record["baseline_intra_pk"]),
        "local_noc_max_directed_link_bytes": float(record["baseline_local_noc_max_directed_link_bytes"]),
        "local_noc_total_byte_hops": float(record["baseline_local_noc_total_byte_hops"]),
    }
    optimized_schedule = {
        "active_cores": int(record["active_cores"]),
        "intra_pe": int(record["intra_pe"]),
        "intra_pm": int(record["intra_pm"]),
        "intra_pn": int(record["intra_pn"]),
        "intra_pk": int(record["intra_pk"]),
        "local_noc_max_directed_link_bytes": float(record["local_noc_max_directed_link_bytes"]),
        "local_noc_total_byte_hops": float(record["local_noc_total_byte_hops"]),
    }
    values = {
        "W00": (naive, unfused_comm, "serial"),
        "W10": (naive, fused_comm, "overlap"),
        "W01": (optimized, unfused_comm, "serial"),
        "W11": (optimized, fused_comm, "overlap"),
    }
    result: dict[str, dict[str, object]] = {}
    for state, (compute, communication, composition) in values.items():
        total = int(record[f"T{state[1:]}_cycles"])
        reference = compute + communication if composition == "serial" else max(compute, communication)
        result[state] = {
            "compute_cycles": compute,
            "communication_cycles": communication,
            "composition": composition,
            "anchor_total_cycles": total,
            "composition_factor": total / max(reference, 1.0),
            "intra_schedule": baseline_schedule if state in ("W00", "W10") else optimized_schedule,
        }
    return result


def moe_d4_anchor(case: object) -> dict[str, object]:
    """Return the real-model Exp1.2 compact/H128/architecture D=4 anchor."""
    module = _moe_module()
    model_name = _canonical_model(_value(case, "model", "model_or_moe_config"), moe=True)
    operator = _moe_operator(case)
    seq_len = _sequence_length(case)
    model = _find_named(module.MODELS, model_name)
    _validate_moe_profile(case, model)
    placement = _find_named(module.PLACEMENTS, "compact")
    legacy_case = module.ExperimentCase(operator, placement, model, seq_len)
    record = module.estimate_case(
        legacy_case,
        include_hbm=True,
        hardware_profile="H128",
        network_scenario="isolated_group",
    )
    cycles = _state_cycles(record)
    stages = _moe_stage_breakdown(record)
    baseline_stage = stages["W00"]
    port_control = _fixed_16core_port_control(
        local_off_cycles=float(baseline_stage["compute_cycles"]),
        local_on_cycles=float(baseline_stage["compute_cycles"]),
        fabric_off_cycles=float(stages["W00"]["communication_cycles"]),
        fabric_on_cycles=float(stages["W10"]["communication_cycles"]),
        payload_bytes_per_die=float(record["runtime_comm_bytes_per_die"]),
        noc_link_bps=float(module.LOCAL_NOC_BPS),
        dte_channel_bps=float(module.NOC_LINK_BPS),
        clock_hz=float(module.CLOCK_HZ),
        waves=int(record["inter_pipeline_waves"]), mesh_span=_mesh_span_for_d(4),
    )
    port = port_control["port"]
    if not isinstance(port, Mapping):
        raise ValueError("MoE controlled port replay lacks metrics")
    controlled_cycles = dict(port_control["state_cycles"])
    controlled_stages = {
        "C00": _port_control_stage(
            baseline_stage, communication_cycles=float(port_control["communication_off_cycles"]),
            fabric_cycles=float(stages["W00"]["communication_cycles"]),
            gateway_cycles=float(port["gateway_serial_cycles"]),
            composition="serial_core_to_port_fabric", overlap_tail=None, port=port,
            local_cycles=float(baseline_stage["compute_cycles"]),
            hbm_base_cycles=float(record["modeled_hbm_cycles"]),
        ),
        "C10": _port_control_stage(
            baseline_stage, communication_cycles=float(port_control["communication_on_cycles"]),
            fabric_cycles=float(stages["W10"]["communication_cycles"]),
            gateway_cycles=float(port["gateway_streaming_cycles"]),
            composition="overlap_core_to_port_fabric",
            overlap_tail=float(port_control["overlap_tail"]), port=port,
            local_cycles=float(baseline_stage["compute_cycles"]),
            hbm_base_cycles=float(record["modeled_hbm_cycles"]),
        ),
    }
    for state, total in controlled_cycles.items():
        stage = controlled_stages[state]
        if not isinstance(stage, dict):
            raise ValueError("MoE controlled stage must be mutable")
        stage["controlled_total_cycles"] = int(total)
    return {
        "state_cycles": cycles,
        **cycles,
        "stages": stages,
        "controlled_state_cycles": controlled_cycles,
        "controlled_state_stages": controlled_stages,
        "controlled_evidence": "exp1_2_fixed_16core_core_to_d2d_port_inter_only_replay",
        "controlled_port_template": {
            "payload_bytes_per_die": float(record["runtime_comm_bytes_per_die"]),
            "noc_link_bps": float(module.LOCAL_NOC_BPS),
            "dte_channel_bps": float(module.NOC_LINK_BPS),
            "clock_hz": float(module.CLOCK_HZ),
            "waves": int(record["inter_pipeline_waves"]),
            "mesh_span": _mesh_span_for_d(4),
        },
        "controlled_base_hbm_cycles": float(record["modeled_hbm_cycles"]),
        "intra_ablation": {
            "baseline": dict(stages["W00"]["intra_schedule"]),
            "optimized": dict(stages["W11"]["intra_schedule"]),
            "mapping_source": "exp1_2_grouped_gemm_adaptive_schedule",
            "inter_port_model": dict(port),
            "dte_port_mapping": "fixed_two_edge_ports_with_x_first_core_routes",
        },
        "case_id": record["case_id"],
        "model": model_name,
        "operator": operator,
        "D": 4,
        "Px": 2,
        "Py": 2,
        "S": seq_len,
        "cycle_time_ns": 1.0,
        "profile": "H128",
        "memory_mode": "architecture",
        "placement": "compact",
        "evidence": "exp1_2_d4_direct",
        "analytical_extrapolation": False,
        "legacy_source": str(_EXP1_2 / "run_experiment.py"),
        "legacy_record": record,
    }


def scale_moe_anchor_to_d(anchor: Mapping[str, object], D: int) -> dict[str, object]:
    """Scale an Exp1.2 D=4 anchor to D using the Exp3.1 analytic rule.

    Compute stages scale by ``4/D`` and communication stages by
    ``sqrt(4/D)``.  At D=4 the historical integer cycles are returned exactly.
    For inter-off states the scaled stages are serialized.  For inter-on
    states, the D=4 overlap composition factor is retained around the scaled
    critical resource.  Every D>4 result is explicitly labelled analytical.
    """
    D = int(D)
    if D < 4:
        raise ValueError("MoE anchor scaling requires D >= 4")
    if int(anchor.get("D", 4)) != 4:
        raise ValueError("scale_moe_anchor_to_d requires a D=4 anchor")
    source_cycles = anchor.get("state_cycles")
    stages = anchor.get("stages")
    if not isinstance(source_cycles, Mapping) or not isinstance(stages, Mapping):
        raise ValueError("anchor must contain state_cycles and stages")

    result = deepcopy(dict(anchor))
    if D == 4:
        exact = {state: int(source_cycles[state]) for state in _STATES}
        result.update(exact)
        result.update({
            "state_cycles": exact,
            "D": 4,
            "compute_scale": 1.0,
            "communication_scale": 1.0,
            "analytical_extrapolation": False,
            "evidence": "exp1_2_d4_direct",
        })
        return result

    compute_scale = 4.0 / D
    communication_scale = math.sqrt(4.0 / D)
    scaled_stages: dict[str, dict[str, object]] = {}
    scaled_cycles: dict[str, int] = {}
    for state in _STATES:
        stage = stages[state]
        if not isinstance(stage, Mapping):
            raise ValueError(f"anchor stage {state} must be a mapping")
        compute = float(stage["compute_cycles"]) * compute_scale
        communication = float(stage["communication_cycles"]) * communication_scale
        composition = str(stage["composition"])
        factor = float(stage["composition_factor"])
        base = compute + communication if composition == "serial" else max(compute, communication)
        total = max(1, round(base * factor))
        scaled_cycles[state] = total
        scaled_stages[state] = {
            **dict(stage),
            "compute_cycles": compute,
            "communication_cycles": communication,
            "anchor_total_cycles": int(source_cycles[state]),
            "scaled_total_cycles": total,
        }
    result.update(scaled_cycles)
    template = anchor.get("controlled_port_template")
    if not isinstance(template, Mapping):
        raise ValueError("MoE anchor lacks controlled D2D-port template")
    scaled_port_template = {
        **dict(template),
        "payload_bytes_per_die": float(template["payload_bytes_per_die"]) * communication_scale,
        "mesh_span": _mesh_span_for_d(D),
    }
    base_hbm_cycles = float(anchor.get("controlled_base_hbm_cycles", 0.0)) * compute_scale
    port_control = _fixed_16core_port_control(
        local_off_cycles=float(scaled_stages["W00"]["compute_cycles"]),
        local_on_cycles=float(scaled_stages["W00"]["compute_cycles"]),
        fabric_off_cycles=float(scaled_stages["W00"]["communication_cycles"]),
        fabric_on_cycles=float(scaled_stages["W10"]["communication_cycles"]),
        payload_bytes_per_die=float(scaled_port_template["payload_bytes_per_die"]),
        noc_link_bps=float(scaled_port_template["noc_link_bps"]),
        dte_channel_bps=float(scaled_port_template["dte_channel_bps"]),
        clock_hz=float(scaled_port_template["clock_hz"]),
        waves=int(scaled_port_template["waves"]),
        mesh_span=float(scaled_port_template["mesh_span"]),
    )
    port = port_control["port"]
    if not isinstance(port, Mapping):
        raise ValueError("scaled MoE controlled port replay lacks metrics")
    controlled_cycles = dict(port_control["state_cycles"])
    controlled_stages = {
        "C00": _port_control_stage(
            scaled_stages["W00"], communication_cycles=float(port_control["communication_off_cycles"]),
            fabric_cycles=float(scaled_stages["W00"]["communication_cycles"]),
            gateway_cycles=float(port["gateway_serial_cycles"]),
            composition="serial_core_to_port_fabric", overlap_tail=None, port=port,
            local_cycles=float(scaled_stages["W00"]["compute_cycles"]),
            hbm_base_cycles=base_hbm_cycles,
        ),
        "C10": _port_control_stage(
            scaled_stages["W00"], communication_cycles=float(port_control["communication_on_cycles"]),
            fabric_cycles=float(scaled_stages["W10"]["communication_cycles"]),
            gateway_cycles=float(port["gateway_streaming_cycles"]),
            composition="overlap_core_to_port_fabric",
            overlap_tail=float(port_control["overlap_tail"]), port=port,
            local_cycles=float(scaled_stages["W00"]["compute_cycles"]),
            hbm_base_cycles=base_hbm_cycles,
        ),
    }
    for state, stage in controlled_stages.items():
        stage["controlled_total_cycles"] = int(controlled_cycles[state])
    result.update({
        "stages": scaled_stages,
        "state_cycles": scaled_cycles,
        "controlled_state_cycles": controlled_cycles,
        "controlled_state_stages": controlled_stages,
        "controlled_port_template": scaled_port_template,
        "controlled_base_hbm_cycles": base_hbm_cycles,
        "controlled_evidence": "analytical_fixed_16core_core_to_d2d_port_inter_only_extrapolation_from_exp1_2",
        "D": D,
        "compute_scale": compute_scale,
        "communication_scale": communication_scale,
        "analytical_extrapolation": True,
        "evidence": "analytical_extrapolation_from_exp1_2_d4_h128_architecture",
        "extrapolation_rule": "compute=anchor*(4/D); communication=anchor*sqrt(4/D)",
    })
    intra_ablation = result.get("intra_ablation")
    if isinstance(intra_ablation, Mapping):
        result["intra_ablation"] = {
            **dict(intra_ablation),
            "inter_port_model": dict(port),
            "dte_port_mapping": "fixed_two_edge_ports_with_x_first_core_routes",
            "port_payload_scaling": "anchor*sqrt(4/D)",
        }
    result.pop("legacy_record", None)
    return result


def _load_records(path: Path) -> dict[str, Mapping[str, object]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {str(record["case_id"]): record for record in records}


def compatibility_audit(*, raise_on_error: bool = True) -> dict[str, object]:
    """Recompute canonical shared cases and compare every W state to JSON."""
    mismatches: list[dict[str, object]] = []
    checked: list[dict[str, object]] = []

    dense_history = _load_records(_EXP1_1 / "results" / "results.json")
    for model in ("LLaMA-2-7B", "GPT-3-175B"):
        for layer in ("attention", "mlp"):
            actual = _dense_native_exp1_reference({
                "model": model, "layer": layer, "S": 2048,
                "D": 6, "Px": 2, "Py": 3,
            })
            expected = dense_history[str(actual["case_id"])]
            for state in _STATES:
                expected_cycles = int(expected[f"T{state[1:]}_cycles"])
                actual_cycles = int(actual[state])
                checked.append({
                    "family": "dense", "case_id": actual["case_id"],
                    "state": state, "expected": expected_cycles,
                    "actual": actual_cycles,
                })
                if actual_cycles != expected_cycles:
                    mismatches.append(checked[-1])

    # These four points are true overlaps with the current Exp3.1 main matrix.
    # Keeping them separate from the small D=6/S=2048 regression fixtures makes
    # the W11 historical-alignment claim directly auditable.
    main_overlap_checked: list[dict[str, object]] = []
    for model in ("LLaMA-2-7B", "GPT-3-175B"):
        for layer in ("attention", "mlp"):
            actual = _dense_native_exp1_reference({
                "model": model, "layer": layer, "S": 36864,
                "D": 36, "Px": 6, "Py": 6,
            })
            expected = dense_history[str(actual["case_id"])]
            for state in _STATES:
                item = {
                    "family": "dense_main_matrix_overlap",
                    "case_id": actual["case_id"], "state": state,
                    "expected": int(expected[f"T{state[1:]}_cycles"]),
                    "actual": int(actual[state]),
                }
                checked.append(item)
                main_overlap_checked.append(item)
                if item["actual"] != item["expected"]:
                    mismatches.append(item)

    moe_history = _load_records(
        _EXP1_2 / "results" / "h128" / "architecture" / "results.json"
    )
    for model in ("Mixtral-8x7B", "DeepSeek-V3"):
        for operator in ("DISPATCH_GEMM", "GEMM_COMBINE"):
            for seq_len in (2304, 36864):
                actual = moe_d4_anchor({
                    "model": model, "operator": operator, "S": seq_len,
                })
                expected = moe_history[str(actual["case_id"])]
                for state in _STATES:
                    expected_cycles = int(expected[f"T{state[1:]}_cycles"])
                    actual_cycles = int(actual[state])
                    checked.append({
                        "family": "moe", "case_id": actual["case_id"],
                        "state": state, "expected": expected_cycles,
                        "actual": actual_cycles,
                    })
                    if actual_cycles != expected_cycles:
                        mismatches.append(checked[-1])

    report: dict[str, object] = {
        "passed": not mismatches,
        "dense_cases": 8,
        "dense_main_matrix_overlap_cases": 4,
        "moe_cases": 8,
        "states_per_case": 4,
        "comparisons": len(checked),
        "main_matrix_overlap": main_overlap_checked,
        "mismatches": mismatches,
        "sources": {
            "dense": str(_EXP1_1 / "results" / "results.json"),
            "moe": str(
                _EXP1_2 / "results" / "h128" / "architecture" / "results.json"
            ),
        },
    }
    if mismatches and raise_on_error:
        first = mismatches[0]
        raise AssertionError(
            "legacy compatibility drift: "
            f"{first['case_id']} {first['state']} expected "
            f"{first['expected']}, got {first['actual']}"
        )
    return report


def calibration_evidence_audit() -> dict[str, object]:
    """Summarize exact small-motif evidence without promoting it to target calibration."""

    dense_path = _EXP1_1 / "calibration" / "existing_calibration.json"
    retained_path = _EXP1_1 / "calibration" / "retained_replay.json"
    moe_path = _EXP1_2 / "calibration" / "source_evidence.json"
    dense = json.loads(dense_path.read_text(encoding="utf-8"))
    retained = json.loads(retained_path.read_text(encoding="utf-8"))
    moe = json.loads(moe_path.read_text(encoding="utf-8"))
    anchors = dense.get("end_to_end_cycle_anchors", {})
    if not isinstance(anchors, Mapping):
        anchors = {}
    return {
        "strategy": "small_cycle_exact_motifs_plus_large_analytical_extrapolation",
        "cycle_accurate_motifs": {
            "dense": {
                "source": str(dense_path),
                "end_to_end_anchor_count": len(anchors),
                "anchor_names": sorted(str(name) for name in anchors),
                "retained_replay_source": str(retained_path),
                "retained_replay_status": retained.get("status"),
                "retained_replay_makespan_cycles": retained.get("expected_makespan_cycles"),
                "limitation": retained.get("limitation"),
            },
            "moe": {
                "source": str(moe_path),
                "source_status": moe.get("source_status"),
                "flexible_mesh_smokes": moe.get("flexible_mesh_smokes"),
                "isolated_group_gemm_smoke": moe.get("isolated_group_gemm_smoke"),
                "target_applicability": moe.get("target_applicability"),
            },
        },
        "main_matrix_evidence": {
            "dense": "Exp1.1 analytical resource replay; exact history regression",
            "moe": "Exp1.2 D4 analytical anchor; D6/9/36 explicit extrapolation",
            "gpu": "placeholder roofline until measured YAML is supplied",
        },
        "publication_boundary": (
            "motifs constrain structure/setup only; target absolute cycles remain "
            "analytical until target-bound cycle and GPU measurements close"
        ),
    }


__all__ = [
    "dense_native",
    "moe_d4_anchor",
    "scale_moe_anchor_to_d",
    "compatibility_audit",
    "calibration_evidence_audit",
]

