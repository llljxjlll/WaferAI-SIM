#!/usr/bin/env python3
"""R1 canonical profiles, legacy gate, and production capability boundaries."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"


def simulation(selector: dict) -> dict:
    config = json.loads((HERE / "simulation" / "v1_cycle.json").read_text())
    collective = config["noc"]["collective"]
    collective.clear()
    collective.update(enabled=True, **selector)
    config["noc"]["transport"] = "conventional"
    return config


def legacy_simulation() -> dict:
    return simulation({
        "broadcast_backend": "multicast",
        "reduce_backend": "legacy_router_alu",
        "reduce_wire": "legacy_two_segment",
        "allow_legacy_backend": True,
    })


def collective_workload(op: str, group: list[int] | None = None) -> dict:
    members = group or [0, 1, 2]
    descriptor = {
        "op": op, "collective_id": 910 + len(op), "group": members,
        "root": members[0], "count": 16, "chunk_bits": 128,
        "stride_bits": 128, "terminal": True,
    }
    if op in ("reduce", "reducescatter", "allreduce"):
        descriptor.update(dtype="uint8", reduce_op="sum")
    return {
        "id_space": "global", "vars": {"B": 1, "T": 1},
        "pipeline": 1, "source": [],
        "chips": [{"chip_id": 0,
                   "cores": [{"id": core, "loop": 1, "worklist": []}
                             for core in members],
                   "collectives": [descriptor]}],
    }


def run(work: Path, sim: Path, hardware: Path | None = None,
        timeout: int = 30) -> subprocess.CompletedProcess[str]:
    hardware_arg = hardware or (HERE / "hardware" / "v1.json")
    return subprocess.run(
        [str(NPUSIM), "--workload-config", str(work),
         "--hardware-config", str(hardware_arg),
         "--simulation-config", str(sim),
         "--mapping-config",
         "../llm/test/noc_collective/mapping/identity.spec"],
        cwd=ROOT / "build", text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout,
    )


def drained(output: str) -> bool:
    return ("Catch test finished" in output and
            "[DRAIN] router_residual=0" in output and
            "data_balanced=1 ctrl_balanced=1" in output and
            "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
            "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0" in output)


def flow_count(output: str) -> int:
    line = next((line for line in output.splitlines()
                 if "[FLOW_DONE]" in line), "")
    return line.count("@")


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    oracle = subprocess.run(
        ["python3", str(HERE / "oracle.py")], cwd=ROOT / "build",
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=20,
    )
    tests.append(("independent profile/tree oracle",
                  oracle.returncode == 0 and "oracle self-test: PASS" in oracle.stdout,
                  oracle.stdout.strip()))
    unit = subprocess.run(
        [str(NPUSIM), "--coll-r1-selftest"], cwd=ROOT / "build",
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=20,
    )
    tests.append(("R1 canonical config contract",
                  unit.returncode == 0 and "PASS (42 checks)" in unit.stdout,
                  "42 checks"))

    with tempfile.TemporaryDirectory(prefix="coll_r1_") as td:
        tmp = Path(td)
        broadcast_work = tmp / "broadcast.json"
        broadcast_work.write_text(json.dumps(collective_workload("broadcast")))

        profile_expectations = {
            "baseline": (2, 0, 0),
            "broadcast_only": (0, 1, 2),
            "reduce_only": (2, 0, 0),
            "reduce_broadcast": (0, 1, 2),
        }
        for profile, (flows, tx, rx) in profile_expectations.items():
            sim = tmp / f"{profile}.json"
            sim.write_text(json.dumps(simulation({"profile": profile})))
            result = run(broadcast_work, sim)
            actual = (flow_count(result.stdout),
                      result.stdout.count("[COLL_V4_TX]"),
                      result.stdout.count("[COLL_V4_RX]"))
            expected = (flows, tx, rx)
            tests.append((f"{profile} Broadcast backend",
                          result.returncode == 0 and drained(result.stdout) and
                          actual == expected,
                          f"flows/v4tx/v4rx={actual}, expected={expected}"))

        dormant_dca = tmp / "dormant_dca.json"
        dormant_dca.write_text(json.dumps(simulation({
            "profile": "baseline",
            "dca": {
                "vector_bits": 0,
                "slice_bits": 0,
                "slices_per_tile": 0,
                "header_fifo_depth": 0,
                "latency": {"uint8": {"sum": 0}},
                "value_mode": "fp_exact",
            },
        })))
        result = run(broadcast_work, dormant_dca)
        tests.append(("inactive DCA settings ignored by endpoint profile",
                      result.returncode == 0 and drained(result.stdout) and
                      flow_count(result.stdout) == 2,
                      f"exit={result.returncode} flows={flow_count(result.stdout)}"))

        reduce_work = tmp / "reduce.json"
        reduce_work.write_text(json.dumps(collective_workload("reduce")))
        tier2 = tmp / "tier2.json"
        tier2.write_text(json.dumps(simulation({"tier": 2})))
        result = run(reduce_work, tier2)
        tests.append(("tier=2 aliases new DCA without legacy fallback",
                      result.returncode == 0 and
                      "[COLL_STREAM_RESULT]" in result.stdout and
                      "[COLL_V5_RESULT]" not in result.stdout,
                      f"exit={result.returncode} stream="
                      f"{'[COLL_STREAM_RESULT]' in result.stdout}"))

        int64_workload = collective_workload("reduce")
        int64_desc = int64_workload["chips"][0]["collectives"][0]
        int64_desc.update(dtype="int64", chunk_bits=1024)
        int64_work = tmp / "int64_reduce.json"
        int64_work.write_text(json.dumps(int64_workload))
        dtype_sim = tmp / "dtype_width.json"
        dtype_sim.write_text(json.dumps(simulation({
            "profile": "reduce_only",
            "dca": {
                "vector_bits": 96,
                "slice_bits": 32,
                "slices_per_tile": 3,
            },
        })))
        result = run(int64_work, dtype_sim)
        dtype_blocked = "whole lanes for workload dtype" in result.stdout
        tests.append(("DCA width validated against reduction dtype",
                      result.returncode != 0 and dtype_blocked,
                      f"exit={result.returncode} blocked={dtype_blocked}"))

        legacy = tmp / "legacy.json"
        legacy.write_text(json.dumps(legacy_simulation()))
        result = run(reduce_work, legacy)
        tests.append(("explicit legacy debug backend",
                      result.returncode == 0 and drained(result.stdout) and
                      result.stdout.count("[COLL_V5_RESULT]") == 1,
                      "one frozen V5 result"))

        negative_configs: list[tuple[str, dict, str]] = []
        missing_gate = legacy_simulation()
        missing_gate["noc"]["collective"].pop("allow_legacy_backend")
        negative_configs.append(("legacy gate required", missing_gate,
                                 "allow_legacy_backend=true"))
        conflict = simulation({"profile": "baseline", "tier": 1})
        negative_configs.append(("profile/tier conflict", conflict,
                                 "conflicts with profile"))
        smart = simulation({"profile": "baseline"})
        smart["noc"]["transport"] = "smart"
        negative_configs.append(("SMART transport gate", smart,
                                 "noc.transport=smart is not implemented"))
        bad_dca = simulation({"profile": "reduce_only",
                              "dca": {"vector_bits": 256,
                                      "slice_bits": 64,
                                      "slices_per_tile": 8}})
        negative_configs.append(("invalid DCA geometry", bad_dca,
                                 "vector_bits must equal"))
        for index, (name, config, marker) in enumerate(negative_configs):
            sim = tmp / f"negative_{index}.json"
            sim.write_text(json.dumps(config))
            result = run(broadcast_work, sim)
            tests.append((name, result.returncode != 0 and marker in result.stdout,
                          f"exit={result.returncode} marker={marker in result.stdout}"))

        multi_hw = json.loads((HERE / "hardware" / "v1.json").read_text())
        multi_hw["die"] = {"x": 2, "y": 1}
        hardware = tmp / "two_die.json"
        hardware.write_text(json.dumps(multi_hw))
        cross_broadcast = tmp / "cross_broadcast.json"
        cross_broadcast.write_text(json.dumps(
            collective_workload("broadcast", [0, 4])))
        cross_reduce = tmp / "cross_reduce.json"
        cross_reduce.write_text(json.dumps(
            collective_workload("reduce", [0, 4])))

        multicast_sim = tmp / "cross_multicast.json"
        multicast_sim.write_text(json.dumps(
            simulation({"profile": "broadcast_only"})))
        result = run(cross_broadcast, multicast_sim, hardware)
        tests.append(("cross-die multicast rejected",
                      result.returncode != 0 and
                      "multicast/DCA collective cannot cross dies" in result.stdout,
                      f"exit={result.returncode}"))

        dca_sim = tmp / "cross_dca.json"
        dca_sim.write_text(json.dumps(simulation({"profile": "reduce_only"})))
        result = run(cross_reduce, dca_sim, hardware)
        tests.append(("cross-die DCA rejected",
                      result.returncode != 0 and
                      "multicast/DCA collective cannot cross dies" in result.stdout,
                      f"exit={result.returncode}"))

    for name, passed, detail in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    passed_count = sum(passed for _, passed, _ in tests)
    print(f"NoC collective R1 runner: {passed_count}/{len(tests)} passed")
    return 0 if passed_count == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
