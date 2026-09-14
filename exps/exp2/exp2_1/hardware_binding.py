#!/usr/bin/env python3
"""Generate the exp2-1 hardware unit-closure audit.

The JSON binding intentionally uses the strongest units the current simulator
can express.  This script separately derives realized rates from implementation
units, so a target label can never turn an unsupported datapath into a closed
cycle-accurate target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
EXP_ROOT = Path(__file__).resolve().parent
DEFAULT_HARDWARE = EXP_ROOT / "configs" / "target_hardware.json"
DEFAULT_SIMULATION = EXP_ROOT / "configs" / "target_simulation.json"
DEFAULT_OUTPUT = EXP_ROOT / "calibration" / "hardware_unit_closure.json"

CYCLE_NS = 2
P2P_PAYLOAD_BYTES = 16
EXPECTED_D2D_BPS = 1.0e12
EXPECTED_NOC_INJECTION_BPS = 256.0e9
EXPECTED_DTE_INJECTION_BPS = 256.0e9
EXPECTED_SRAM_PORT_BPS = 256.0e9
EXPECTED_CORE_TFLOPS = 8.0
EXPECTED_DIE_TFLOPS = 128.0
EXPECTED_STACK_BPS = 256.0e9
EXPECTED_STACK_CAPACITY_BYTES = 16 * 1024**3
EXPECTED_STACK_DIES = (1, 4, 31, 34)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate_entry(
    *, expected: float, configured: float, derived: float, direct_validated: bool
) -> dict[str, Any]:
    arithmetic_match = abs(derived - expected) <= max(1.0, expected) * 1e-12
    configured_match = abs(configured - expected) <= max(1.0, expected) * 1e-12
    return {
        "expected_Bps": expected,
        "configured_Bps": configured,
        "derived_Bps": derived,
        "configured_match": configured_match,
        "arithmetic_match": arithmetic_match,
        "direct_saturation_sweep_validated": direct_validated,
        "gate_pass": configured_match and arithmetic_match and direct_validated,
    }


def _validate_regions(hardware: dict[str, Any]) -> tuple[int, bool]:
    sram = hardware["memory"]["sram"]
    capacity = int(sram["capacity_bytes"])
    intervals = sorted(
        (int(region["base_bytes"]), int(region["base_bytes"]) + int(region["size_bytes"]))
        for region in sram["regions"]
    )
    nonoverlap = all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    in_bounds = bool(intervals) and intervals[0][0] == 0 and intervals[-1][1] <= capacity
    return capacity, nonoverlap and in_bounds


def build_unit_closure(
    hardware_path: Path = DEFAULT_HARDWARE,
    simulation_path: Path = DEFAULT_SIMULATION,
) -> dict[str, Any]:
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    cycle_seconds = CYCLE_NS * 1e-9

    core = hardware["cores"][0]
    compute_flops_per_cycle = (
        int(core["exu_x"]) ** 2
        * 2
        * int(core["sa_cnt"])
        * float(hardware["operand"]["comp_util"])
    )
    compute_core_flops = compute_flops_per_cycle / cycle_seconds
    compute_die_flops = compute_core_flops * int(hardware["x"]) * int(hardware["y"])

    d2d = hardware["die_ports"]["c2c"]
    link_rate = d2d["link_rate"]
    link_packets_per_cycle = int(link_rate["num"]) / int(link_rate["den"])
    d2d_derived = link_packets_per_cycle * P2P_PAYLOAD_BYTES / cycle_seconds

    noc_scale = int(hardware["noc"]["noc_payload_per_cycle"])
    noc_grouped_derived = noc_scale * P2P_PAYLOAD_BYTES / cycle_seconds
    dte_width_bits = int(core["dte_bit_width"])
    dte_derived = dte_width_bits / 8.0 / cycle_seconds
    sram_width_bits = int(
        hardware["memory"]["sram"]["ports"]["compute"]["read"]["width_bits"]
    )
    sram_derived = sram_width_bits / 8.0 / cycle_seconds

    stacks = hardware["memory_system"]["hbm_stacks"]
    profile = hardware["memory_system"]["profiles"]["target_hbm2_16ch"]
    hbm_profile_derived = (
        float(profile["data_rate_gbps_per_pin"])
        * 1e9
        * int(profile["stack_bus_width_bits"])
        / 8.0
    )
    home_ranges = hardware["memory_system"]["address_policy"]["home_ranges"]
    intervals = sorted(
        (int(item["base"]), int(item["base"]) + int(item["size_bytes"]))
        for item in home_ranges
    )
    ranges_nonoverlap = all(
        left[1] <= right[0] for left, right in zip(intervals, intervals[1:])
    )
    ranges_contiguous = bool(intervals) and intervals[0][0] == 0 and all(
        left[1] == right[0] for left, right in zip(intervals, intervals[1:])
    )
    capacity_sum = sum(int(stack["capacity_bytes"]) for stack in stacks)
    sram_capacity, sram_regions_valid = _validate_regions(hardware)

    source_paths = {
        "cycle_constant": ROOT / "llm/include/macros/macros.h",
        "p2p_payload": ROOT / "llm/include/dte/p2p_payload.h",
        "d2d_rate_parser": ROOT / "llm/src/die/port_config.cpp",
        "geometry_loader": ROOT / "llm/src/utils/config_utils.cpp",
        "compute_cost": ROOT / "llm/src/isa/npu_cost_model.cpp",
        "hbm_address_map": ROOT / "llm/src/memory/hbm_address_map.cpp",
        "hbm_memspec": ROOT / "llm/src/memory/hbm_memspec.cpp",
        "hbm2_config": ROOT / "DRAMSys/configs/hbm2-example.json",
        "hbm2_memspec": ROOT / "DRAMSys/configs/memspec/HBM2.json",
    }

    d2d_gate = _rate_entry(
        expected=EXPECTED_D2D_BPS,
        configured=d2d_derived,
        derived=d2d_derived,
        direct_validated=False,
    )
    d2d_gate.update(
        {
            "packet_payload_bytes": P2P_PAYLOAD_BYTES,
            "packets_per_cycle": link_packets_per_cycle,
            "representability_limit": "rate.num <= rate.den; one sc_bv<256> per cycle",
            "target_to_realized_ratio": EXPECTED_D2D_BPS / d2d_derived,
            "failure": "single-lane target needs 125 business packets/cycle, but parser rejects rates above 1 packet/cycle",
        }
    )

    noc_gate = _rate_entry(
        expected=EXPECTED_NOC_INJECTION_BPS,
        configured=noc_grouped_derived,
        derived=noc_grouped_derived,
        direct_validated=False,
    )
    noc_gate.update(
        {
            "payload_group_scale": noc_scale,
            "interpretation": "collective logical-payload grouping arithmetic, not a measured per-core router byte sweep",
        }
    )
    dte_gate = _rate_entry(
        expected=EXPECTED_DTE_INJECTION_BPS,
        configured=dte_derived,
        derived=dte_derived,
        direct_validated=False,
    )
    dte_gate["shared_aggregate_width_bits"] = dte_width_bits
    sram_gate = _rate_entry(
        expected=EXPECTED_SRAM_PORT_BPS,
        configured=sram_derived,
        derived=sram_derived,
        direct_validated=False,
    )
    sram_gate.update(
        {
            "read_port_width_bits": sram_width_bits,
            "capacity_bytes": sram_capacity,
            "regions_nonoverlap_and_in_bounds": sram_regions_valid,
        }
    )

    hbm_arithmetic = all(
        abs(float(stack["bandwidth_cap_GBps"]) * 1e9 - EXPECTED_STACK_BPS) <= 1.0
        and int(stack["capacity_bytes"]) == EXPECTED_STACK_CAPACITY_BYTES
        for stack in stacks
    ) and abs(hbm_profile_derived - EXPECTED_STACK_BPS) <= 1.0
    hbm_gate = {
        "stack_count": len(stacks),
        "expected_stack_count": 4,
        "home_die_ids": [int(stack["compute_die_id"]) for stack in stacks],
        "expected_home_die_ids": list(EXPECTED_STACK_DIES),
        "per_stack_expected_Bps": EXPECTED_STACK_BPS,
        "per_stack_configured_Bps": [
            float(stack["bandwidth_cap_GBps"]) * 1e9 for stack in stacks
        ],
        "profile_derived_stack_Bps": hbm_profile_derived,
        "per_stack_capacity_bytes": [int(stack["capacity_bytes"]) for stack in stacks],
        "aggregate_capacity_bytes": capacity_sum,
        "address_ranges": [
            [int(item["base"]), int(item["base"]) + int(item["size_bytes"])]
            for item in home_ranges
        ],
        "address_ranges_nonoverlap": ranges_nonoverlap,
        "address_ranges_contiguous": ranges_contiguous,
        "configuration_arithmetic_match": hbm_arithmetic,
        "direct_saturation_sweep_validated": False,
        "gate_pass": False,
    }

    compute_arithmetic = (
        abs(compute_core_flops / 1e12 - EXPECTED_CORE_TFLOPS) <= 1e-12
        and abs(compute_die_flops / 1e12 - EXPECTED_DIE_TFLOPS) <= 1e-12
    )
    compute_gate = {
        "exu_x": int(core["exu_x"]),
        "sa_count": int(core["sa_cnt"]),
        "compute_utilization": float(hardware["operand"]["comp_util"]),
        "logical_flops_per_cycle": compute_flops_per_cycle,
        "derived_core_TFLOPs": compute_core_flops / 1e12,
        "expected_core_TFLOPs": EXPECTED_CORE_TFLOPS,
        "derived_die_TFLOPs": compute_die_flops / 1e12,
        "expected_die_TFLOPs": EXPECTED_DIE_TFLOPS,
        "configuration_arithmetic_match": compute_arithmetic,
        "target_bound_isolated_gemm_validated": False,
        "gate_pass": False,
    }

    structural = {
        "wafer": {
            "die_mesh": [int(hardware["die"]["x"]), int(hardware["die"]["y"])],
            "die_count": int(hardware["die"]["x"]) * int(hardware["die"]["y"]),
            "core_mesh_per_die": [int(hardware["x"]), int(hardware["y"])],
            "worker_cores_per_die": int(hardware["x"]) * int(hardware["y"]),
            "gate_pass": hardware["die"] == {"x": 6, "y": 6}
            and hardware["x"] == 4
            and hardware["y"] == 4,
        },
        "control": {
            "mode": hardware["control_cores"]["mode"],
            "dedicated_control_is_not_addressed_as_worker": True,
            "worker_count_remains_16": True,
            "gate_pass": hardware["control_cores"]["mode"] == "dual_dte_dedicated",
        },
        "hbm_edge_placement": {
            "home_die_coordinates_xy": [[1, 0], [4, 0], [1, 5], [4, 5]],
            "home_die_ids_row_major": [1, 4, 31, 34],
            "gate_pass": [int(stack["compute_die_id"]) for stack in stacks]
            == list(EXPECTED_STACK_DIES),
        },
    }

    reasons = [
        "D2D single-lane realized business bandwidth is 8 GB/s, 125x below the 1 TB/s target.",
        "No target-bound isolated GEMM has validated the configured 8 TFLOP/s/core arithmetic.",
        "No target-bound NoC/DTE/SRAM/HBM byte sweep has validated saturation rates.",
        "The retained Stage3/Stage4/Flexible cases use tiny, different hardware bindings and are structural priors only.",
        "The vector rate is not calibrated on this target binding.",
    ]

    return {
        "schema_version": "exp2_1.hardware_unit_closure/v1",
        "generated_from": {
            "hardware_path": str(hardware_path.relative_to(ROOT)),
            "hardware_sha256": sha256_file(hardware_path),
            "simulation_path": str(simulation_path.relative_to(ROOT)),
            "simulation_sha256": sha256_file(simulation_path),
            "cycle_time_ns": CYCLE_NS,
            "implementation_sources": {
                name: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                }
                for name, path in source_paths.items()
            },
        },
        "structural_closure": structural,
        "unit_gates": {
            "compute": compute_gate,
            "sram_read_port": sram_gate,
            "dte_injection": dte_gate,
            "noc_logical_payload": noc_gate,
            "d2d_single_lane": d2d_gate,
            "hbm": hbm_gate,
        },
        "simulator_unit_closure": False,
        "closure_status": "failed_target_unit_closure",
        "publish_target_absolute_cycles": False,
        "allowed_cycle_evidence_use": "structural_lifecycle_prior_only",
        "failure_reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, default=DEFAULT_HARDWARE)
    parser.add_argument("--simulation", type=Path, default=DEFAULT_SIMULATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    closure = build_unit_closure(args.hardware.resolve(), args.simulation.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output}: simulator_unit_closure="
        f"{str(closure['simulator_unit_closure']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
