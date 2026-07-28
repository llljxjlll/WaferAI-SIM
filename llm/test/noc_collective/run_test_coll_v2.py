#!/usr/bin/env python3
"""NoC collective V2 finite Gather reorder acceptance matrix."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"
OPS = {"gather": (2, 1), "allgather": (6, 3), "alltoall": (6, 3)}


def workload(op: str, depth: int = 2) -> dict:
    return {
        "vars": {"B": 1, "T": 1}, "pipeline": 1, "source": [],
        "chips": [{"chip_id": 0,
            "cores": [{"id": i, "loop": 1, "worklist": []} for i in range(3)],
            "collectives": [{"op": op, "collective_id": 200,
                "group": [0, 1, 2], "root": 0, "count": 3,
                "chunk_bits": 1024, "stride_bits": 1024,
                "gather_reorder_depth": depth, "terminal": True}]}]
    }


def run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT / "build", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    unit = run([str(NPUSIM), "--coll-v2-selftest"])
    tests.append(("finite reorder self-test",
                  unit.returncode == 0 and "PASS (14 checks)" in unit.stdout,
                  "14 checks"))

    with tempfile.TemporaryDirectory(prefix="coll_v2_") as td:
        for op, (expected_flows, expected_drains) in OPS.items():
            path = Path(td) / f"{op}.json"
            path.write_text(json.dumps(workload(op)))
            for backend in ("cycle", "beha"):
                result = run([str(NPUSIM), "--workload-config", str(path),
                    "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
                    "--simulation-config", f"../llm/test/noc_collective/simulation/v1_{backend}.json",
                    "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"])
                output = result.stdout
                flow_line = next((line for line in output.splitlines()
                                  if "[FLOW_DONE]" in line), "")
                accepts = output.count("[COLL_REORDER] event=accept")
                drains = output.count("[COLL_REORDER] event=drain")
                ok = (result.returncode == 0 and flow_line.count("@") == expected_flows
                      and accepts == expected_flows and drains == expected_drains
                      and "router_residual=0" in output
                      and "data_balanced=1 ctrl_balanced=1" in output)
                tests.append((f"{op}/{backend}", ok,
                              f"flows={flow_line.count('@')}/{expected_flows}, "
                              f"accepts={accepts}, drains={drains}/{expected_drains}"))

        bad = workload("gather")
        bad["chips"][0]["collectives"][0]["op"] = "broadcast"
        bad_path = Path(td) / "bad_non_gather_depth.json"
        bad_path.write_text(json.dumps(bad))
        result = run([str(NPUSIM), "--workload-config", str(bad_path),
            "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
            "--simulation-config", "../llm/test/noc_collective/simulation/v1_cycle.json",
            "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"])
        tests.append(("depth on non-Gather RX rejected", result.returncode != 0,
                      f"exit={result.returncode}"))

    for name, ok, detail in tests:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = sum(ok for _, ok, _ in tests)
    print(f"NoC collective V2: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
