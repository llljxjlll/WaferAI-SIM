#!/usr/bin/env python3
"""NoC collective V3 Tier0 reduction acceptance matrix."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"
OPS = {"reduce": 2, "reducescatter": 4, "allreduce": 4}
DTYPE_BITS = {"uint8": 8, "int32": 32, "int64": 64, "fp32": 32}


def workload(op: str, dtype: str = "int32", reduce_op: str = "sum",
             count: int = 10) -> dict:
    bits = count * DTYPE_BITS[dtype]
    return {
        "vars": {"B": 1, "T": 1}, "pipeline": 1, "source": [],
        "chips": [{"chip_id": 0,
            "cores": [{"id": i, "loop": 1, "worklist": []} for i in range(3)],
            "collectives": [{"op": op, "collective_id": 300,
                "group": [0, 1, 2], "root": 0, "count": count,
                "dtype": dtype, "reduce_op": reduce_op,
                "chunk_bits": bits, "stride_bits": bits,
                "src_addr": 4096, "dst_addr": 8192, "terminal": True}]}]
    }


def run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT / "build", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)


def simulate(path: Path, backend: str) -> subprocess.CompletedProcess[str]:
    return run([str(NPUSIM), "--workload-config", str(path),
        "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
        "--simulation-config", f"../llm/test/noc_collective/simulation/v1_{backend}.json",
        "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"])


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    oracle = run([sys.executable, str(HERE / "oracle.py")])
    tests.append(("independent oracle", oracle.returncode == 0 and
                  "V0/V1/V3 oracle self-test: PASS" in oracle.stdout,
                  "flow/phase/compute formulas"))
    unit = run([str(NPUSIM), "--coll-v3-selftest"])
    tests.append(("planner/RX/compute self-test", unit.returncode == 0 and
                  "PASS (14 checks)" in unit.stdout, "14 checks"))

    with tempfile.TemporaryDirectory(prefix="coll_v3_") as td:
        temp = Path(td)
        for op, expected_flows in OPS.items():
            path = temp / f"{op}.json"
            path.write_text(json.dumps(workload(op, count=100)))
            for backend in ("cycle", "beha"):
                result = simulate(path, backend)
                output = result.stdout
                flow_line = next((line for line in output.splitlines()
                                  if "[FLOW_DONE]" in line), "")
                accepts = output.count("[COLL_REDUCE_RX] event=accept")
                completes = output.count("[COLL_REDUCE_RX] event=complete")
                computes = output.count("[COLL_REDUCE_COMPUTE]")
                cycle = re.search(r"Catch test finished.*?\|.*?(\d+) ns", output)
                ok = (result.returncode == 0 and flow_line.count("@") == expected_flows
                      and accepts == 2 and completes == 1 and computes == 1
                      and "operations=200 cycles=4 lanes=64" in output
                      and "router_residual=0" in output
                      and "data_balanced=1 ctrl_balanced=1" in output)
                tests.append((f"{op}/{backend}", ok,
                              f"flows={flow_line.count('@')}/{expected_flows}, "
                              f"rx={accepts}/2, complete={completes}, compute={computes}, "
                              f"cycle={cycle.group(1) if cycle else '?'}"))

        for dtype in ("uint8", "int32", "int64"):
            for reduce_op in ("sum", "max"):
                path = temp / f"supported_{dtype}_{reduce_op}.json"
                path.write_text(json.dumps(workload("reduce", dtype, reduce_op)))
                result = simulate(path, "cycle")
                tests.append((f"supported {dtype}/{reduce_op}",
                              result.returncode == 0 and
                              result.stdout.count("[COLL_REDUCE_COMPUTE]") == 1,
                              f"exit={result.returncode}"))

        negatives: list[tuple[str, dict]] = []
        missing_op = workload("reduce"); del missing_op["chips"][0]["collectives"][0]["reduce_op"]
        negatives.append(("missing reduce_op", missing_op))
        negatives.append(("fp32 unsupported", workload("reduce", "fp32")))
        bad_size = workload("reduce"); bad_size["chips"][0]["collectives"][0]["chunk_bits"] += 1
        negatives.append(("payload shape mismatch", bad_size))
        bad_align = workload("reduce"); bad_align["chips"][0]["collectives"][0]["src_addr"] += 1
        negatives.append(("misaligned source", bad_align))
        bad_depth = workload("reduce"); bad_depth["chips"][0]["collectives"][0]["gather_reorder_depth"] = 2
        negatives.append(("Gather depth on Reduce RX", bad_depth))
        for index, (name, work) in enumerate(negatives):
            path = temp / f"negative_{index}.json"; path.write_text(json.dumps(work))
            result = simulate(path, "cycle")
            tests.append((name, result.returncode != 0, f"exit={result.returncode}"))

    for name, ok, detail in tests:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = sum(ok for _, ok, _ in tests)
    print(f"NoC collective V3: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
