#!/usr/bin/env python3
"""Production Router multicast and in-network-reduce end-to-end matrix."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"

def workload(op: str, dtype: str = "uint8", reduce_op: str = "sum") -> dict:
    bits = {"uint8": 8, "int32": 32, "int64": 64}[dtype]
    count = 128 // bits * 8
    coll = {"op": op, "collective_id": 500 + len(op),
            "group": [0, 1, 2], "root": 0, "count": count,
            "chunk_bits": count * bits, "stride_bits": count * bits,
            "terminal": True}
    if op in ("reduce", "reducescatter", "allreduce"):
        coll.update(dtype=dtype, reduce_op=reduce_op)
    return {"vars": {"B": 1, "T": 1}, "pipeline": 1, "source": [],
            "chips": [{"chip_id": 0,
                "cores": [{"id": i, "loop": 1, "worklist": []}
                          for i in range(3)],
                "collectives": [coll]}]}

def run(work: Path, sim: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(NPUSIM), "--workload-config", str(work),
        "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
        "--simulation-config", str(sim),
        "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"],
        cwd=ROOT / "build", text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=40)

def drained(result: subprocess.CompletedProcess[str]) -> bool:
    out = result.stdout
    return (result.returncode == 0 and "Catch test finished" in out and
            "router_residual=0" in out and
            "data_balanced=1 ctrl_balanced=1" in out and
            "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
            "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0" in out)

def select_legacy_tier2(sim: dict) -> None:
    coll = sim["noc"]["collective"]
    coll.pop("tier", None)
    coll.update(broadcast_backend="multicast",
                reduce_backend="legacy_router_alu",
                reduce_wire="legacy_two_segment",
                allow_legacy_backend=True)

def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    for version, checks in ((4, 13), (5, 20)):
        unit = subprocess.run([str(NPUSIM), f"--coll-v{version}-selftest"],
            cwd=ROOT / "build", text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=20)
        tests.append((f"V{version} contract", unit.returncode == 0 and
            f"PASS ({checks} checks)" in unit.stdout, f"{checks} checks"))
    with tempfile.TemporaryDirectory(prefix="coll_v45_") as td:
        tmp = Path(td)
        base = json.loads((HERE / "simulation" / "v1_cycle.json").read_text())
        for tier in (1, 2):
            sim = json.loads(json.dumps(base))
            if tier == 2:
                select_legacy_tier2(sim)
            else:
                sim["noc"]["collective"]["tier"] = tier
            (tmp / f"tier{tier}.json").write_text(json.dumps(sim))
        for backend in ("cycle", "beha"):
            sim = json.loads((HERE / "simulation" / f"v1_{backend}.json").read_text())
            sim["noc"]["collective"]["tier"] = 1
            sim_path = tmp / f"tier1_{backend}.json"; sim_path.write_text(json.dumps(sim))
            work_path = tmp / f"broadcast_{backend}.json"
            work_path.write_text(json.dumps(workload("broadcast")))
            result = run(work_path, sim_path)
            ok = (drained(result) and result.stdout.count("[COLL_V4_TX]") == 1 and
                  result.stdout.count("[COLL_V4_RX]") == 2)
            tests.append((f"Tier1 broadcast/{backend}", ok,
                "single injection, 2/2 exact receivers"))
        for op in ("reduce", "reducescatter", "allreduce"):
            work_path = tmp / f"{op}.json"; work_path.write_text(json.dumps(workload(op)))
            result = run(work_path, tmp / "tier2.json")
            ok = (drained(result) and result.stdout.count("[COLL_V5_TX]") == 3 and
                  result.stdout.count("[COLL_V5_RESULT]") == 1)
            if op == "allreduce":
                ok = ok and result.stdout.count("[COLL_V4_TX]") == 1 and result.stdout.count("[COLL_V4_RX]") == 2
            tests.append((f"Tier2 {op}", ok, "3 operands, one verified root result"))
        for dtype, reduce_op in (("int32", "max"), ("int64", "sum")):
            work_path = tmp / f"{dtype}_{reduce_op}.json"
            work_path.write_text(json.dumps(workload("reduce", dtype, reduce_op)))
            result = run(work_path, tmp / "tier2.json")
            tests.append((f"Tier2 {dtype}/{reduce_op}", drained(result) and
                "[COLL_V5_RESULT]" in result.stdout, "value=verified"))
    for name, ok, detail in tests:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = sum(ok for _, ok, _ in tests)
    print(f"NoC collective V4/V5: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1

if __name__ == "__main__": raise SystemExit(main())
