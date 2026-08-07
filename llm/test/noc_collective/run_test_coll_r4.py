#!/usr/bin/env python3
"""R4 streaming reduce production-path end-to-end tests."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"


def workload(group_size: int, count: int, op: str = "reduce") -> dict:
    group = list(range(group_size))
    coll = {
        "op": op,
        "group_id": 64 + group_size,
        "collective_id": 6400 + group_size * 100 + count,
        "group": group,
        "root": 0,
        "count": count,
        "chunk_bits": count * 8,
        "stride_bits": count * 8,
        "dtype": "uint8",
        "reduce_op": "sum",
        "terminal": True,
    }
    return {
        "vars": {"B": 1, "T": 1},
        "pipeline": 1,
        "source": [],
        "chips": [{
            "chip_id": 0,
            "cores": [
                {"id": rank, "loop": 1, "worklist": []}
                for rank in group
            ],
            "collectives": [coll],
        }],
    }


def simulation() -> dict:
    sim = json.loads((HERE / "simulation" / "v1_cycle.json").read_text())
    sim["noc"]["collective"] = {
        "enabled": True,
        "profile": "reduce_only",
        "dca": {
            "vector_bits": 512,
            "slice_bits": 64,
            "slices_per_tile": 8,
            "header_fifo_depth": 8,
            "operand_fifo_depth": 8,
            "result_fifo_depth": 1,
            "arbitration": "round_robin",
            "value_mode": "integer_exact",
            "latency": {"uint8": {"sum": 4}},
            "initiation_interval": {"uint8": {"sum": 1}},
        },
    }
    return sim


def drained(result: subprocess.CompletedProcess[str]) -> bool:
    out = result.stdout
    return (
        result.returncode == 0
        and "Catch test finished" in out
        and "router_residual=0" in out
        and "data_balanced=1 ctrl_balanced=1" in out
        and "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
            "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0" in out
    )


def run(work: Path, sim: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(NPUSIM),
            "--workload-config", str(work),
            "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
            "--simulation-config", str(sim),
            "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec",
        ],
        cwd=ROOT / "build",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    unit = subprocess.run(
        [str(NPUSIM), "--coll-r4-selftest"],
        cwd=ROOT / "build", text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=20,
    )
    tests.append(("R4 contract", unit.returncode == 0 and
                  "R4 self-test: 12/12 checks passed" in unit.stdout,
                  "12 checks"))
    with tempfile.TemporaryDirectory(prefix="coll_r4_") as td:
        tmp = Path(td)
        sim = tmp / "reduce_only.json"
        sim.write_text(json.dumps(simulation()))
        cases = [
            ("two-input-tail", 2, 70),
            ("three-input", 3, 129),
            ("large-64-flit", 3, 1024),
        ]
        for name, ranks, count in cases:
            work = tmp / f"{name}.json"
            work.write_text(json.dumps(workload(ranks, count)))
            result = run(work, sim)
            ok = (
                drained(result)
                and result.stdout.count("[COLL_STREAM_TX]") == ranks
                and result.stdout.count("[COLL_STREAM_RESULT]") == 1
                and "value=verified" in result.stdout
                and result.stdout.count("[COLL_V6_RELEASE]") == 1
            )
            detail = f"{ranks} inputs, count={count}, exact+drained"
            if not ok:
                detail += f"; rc={result.returncode}; tail={result.stdout[-1200:]}"
            tests.append((name, ok, detail))
    for name, ok, detail in tests:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = sum(ok for _, ok, _ in tests)
    print(f"NoC collective R4: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
