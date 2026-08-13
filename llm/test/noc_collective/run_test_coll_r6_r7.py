#!/usr/bin/env python3
"""R6/R7 pure contracts, real multicast, and behavioral-DCA rejection."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"
DTYPE_BITS = {"uint8": 8, "int32": 32, "int64": 64,
              "fp32": 32, "fp16": 16, "fp8": 8}


def simulation(profile: str, value_mode: str = "integer_exact") -> dict:
    sim = json.loads((HERE / "simulation" / "v1_cycle.json").read_text())
    sim["noc"]["collective"] = {
        "enabled": True,
        "profile": profile,
        "dca": {
            "vector_bits": 512,
            "slice_bits": 64,
            "slices_per_tile": 8,
            "header_fifo_depth": 8,
            "operand_fifo_depth": 8,
            "result_fifo_depth": 2,
            "arbitration": "round_robin",
            "value_mode": value_mode,
            "latency": {
                "uint8": {"sum": 4, "max": 3},
                "int32": {"sum": 5, "max": 4},
                "int64": {"sum": 6, "max": 5},
                "fp32": {"sum": 7, "max": 6},
                "fp16": {"sum": 5, "max": 4},
                "fp8": {"sum": 4, "max": 3},
            },
            "initiation_interval": {
                "uint8": {"sum": 1, "max": 1},
                "int32": {"sum": 1, "max": 1},
                "int64": {"sum": 2, "max": 2},
                "fp32": {"sum": 1, "max": 1},
                "fp16": {"sum": 1, "max": 1},
                "fp8": {"sum": 1, "max": 1},
            },
        },
    }
    return sim


def workload(op: str, group: list[int], root: int, count: int,
             dtype: str = "uint8", reduce_op: str = "sum",
             cid: int = 700) -> dict:
    bits = DTYPE_BITS[dtype]
    coll: dict = {
        "op": op,
        "group_id": 70 + cid,
        "collective_id": cid,
        "group": group,
        "root": root,
        "count": count,
        "chunk_bits": count * bits,
        "stride_bits": count * bits,
        "terminal": True,
    }
    if op in ("reduce", "reducescatter", "allreduce"):
        coll.update(dtype=dtype, reduce_op=reduce_op)
    if op == "allgather":
        coll["gather_reorder_depth"] = max(1, len(group))
    return {
        "vars": {"B": 1, "T": 1}, "pipeline": 1, "source": [],
        "chips": [{"chip_id": 0,
                   "cores": [{"id": i, "loop": 1, "worklist": []}
                             for i in range(4)],
                   "collectives": [coll]}],
    }


def run(work: Path, sim: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(NPUSIM), "--workload-config", str(work),
         "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
         "--simulation-config", str(sim),
         "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"],
        cwd=ROOT / "build", text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=90)


def drained(out: str, rc: int) -> bool:
    return (rc == 0 and "Catch test finished" in out and
            "[DRAIN] router_residual=0" in out and
            "data_balanced=1 ctrl_balanced=1" in out and
            "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
            "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0" in out)


def behavioral_dca_rejected(
        result: subprocess.CompletedProcess[str], profile: str) -> bool:
    """Behavioral-SRAM fixtures must not impersonate production DCA bytes."""
    out = result.stdout
    return (
        result.returncode != 0 and
        f"profile={profile}" in out and
        "reduce_backend=dca_offload" in out and
        "DCA stream TX requires the real SRAM data path" in out and
        "[COLL_STREAM_RESULT]" not in out and
        "[COLL_DCA_RESULT]" not in out and
        "[COLL_V5_RESULT]" not in out and
        "value=" not in out
    )


def selftest(flag: str, marker: str) -> tuple[bool, str]:
    result = subprocess.run(
        [str(NPUSIM), flag], cwd=ROOT / "build", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    ok = result.returncode == 0 and marker in result.stdout
    detail = marker if ok else f"rc={result.returncode}; tail={result.stdout[-800:]}"
    return ok, detail


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    r6_selftest = selftest("--coll-r6-selftest", "R6 self-test: 7/7")
    tests.append(("R6 standalone contract", *r6_selftest))
    r7_selftest = selftest("--coll-r7-selftest", "R7 self-test: 6/6")
    tests.append(("R7 standalone contract", *r7_selftest))
    with tempfile.TemporaryDirectory(prefix="coll_r67_") as td:
        tmp = Path(td)
        sims: dict[tuple[str, str], Path] = {}
        for profile in ("baseline", "broadcast_only", "reduce_only",
                        "reduce_broadcast"):
            mode = "integer_exact"
            path = tmp / f"{profile}.json"
            path.write_text(json.dumps(simulation(profile, mode)))
            sims[(profile, mode)] = path
        for mode in ("fp_exact", "timing_only"):
            path = tmp / f"reduce_only_{mode}.json"
            path.write_text(json.dumps(simulation("reduce_only", mode)))
            sims[("reduce_only", mode)] = path

        # R6 behavioral SRAM remains a valid broadcast/allgather fixture, but
        # cannot stand in for the production AccessUnit required by DCA.
        r6_cases = [
            ("broadcast", [0, 1, 2, 3], 2, 65, "uint8", "sum"),
            ("allgather", [0, 2, 3], 2, 33, "uint8", "sum"),
            ("reduce", [0], 0, 17, "uint8", "sum"),
            ("reduce", [0, 3], 3, 19, "int32", "max"),
            ("reducescatter", [0, 1, 2, 3], 2, 67, "uint8", "sum"),
            ("allreduce", [0, 1, 2, 3], 1, 35, "int64", "sum"),
        ]
        for index, case in enumerate(r6_cases):
            op, group, root, count, dtype, reduce_op = case
            work = tmp / f"r6_{index}_{op}.json"
            work.write_text(json.dumps(workload(
                op, group, root, count, dtype, reduce_op, 800 + index)))
            result = run(work, sims[("reduce_only", "integer_exact")])
            reduction = op in ("reduce", "reducescatter", "allreduce")
            if reduction:
                ok = behavioral_dca_rejected(result, "reduce_only")
                detail = ("behavioral SRAM rejected before DCA source read; "
                          "no stream result/V5 fallback/value")
            else:
                ok = (drained(result.stdout, result.returncode) and
                      result.stdout.count("[COLL_V4_TX]") == 0)
                detail = "unicast distribution; no multicast"
            if not ok:
                detail += f"; rc={result.returncode}; tail={result.stdout[-1000:]}"
            tests.append((f"R6 {op}/N={len(group)}", ok, detail))

        # FP/timing behavioral fixtures select DCA but must fail at the same
        # real-SRAM gate before producing a value or fallback result.
        for mode in ("fp_exact", "timing_only"):
            work = tmp / f"r5_{mode}.json"
            work.write_text(json.dumps(workload(
                "reduce", [0, 1, 2], 0, 37, "fp32", "sum",
                880 if mode == "fp_exact" else 881)))
            result = run(work, sims[("reduce_only", mode)])
            fp_ok = behavioral_dca_rejected(result, "reduce_only")
            fp_detail = ("DCA selected; behavioral SRAM rejected without "
                         "stream result/V5 fallback/value")
            if not fp_ok:
                fp_detail += (f"; rc={result.returncode}; "
                              f"tail={result.stdout[-1200:]}")
            tests.append((f"R5 FP32/{mode}", fp_ok, fp_detail))

        for index, dtype in enumerate(("fp16", "fp8")):
            work = tmp / f"r5_{dtype}_timing.json"
            work.write_text(json.dumps(workload(
                "reduce", [0, 1, 2], 0, 37, dtype, "sum",
                884 + index)))
            result = run(work, sims[("reduce_only", "timing_only")])
            tests.append((f"R5 {dtype.upper()}/timing_only",
                          behavioral_dca_rejected(result, "reduce_only"),
                          "DCA selected; behavioral SRAM rejected without fallback"))

        # This legacy behavioral fixture used to impersonate shared-pool
        # production; it must now stop at the real-SRAM DCA source gate.
        shared = workload("reduce", [0, 1, 2, 3], 0, 129,
                          "uint8", "sum", 889)
        shared["chips"][0]["collectives"][0]["core_contention_beats"] = 12
        work = tmp / "r5_shared_production.json"
        work.write_text(json.dumps(shared))
        result = run(work, sims[("reduce_only", "integer_exact")])
        shared_ok = behavioral_dca_rejected(result, "reduce_only")
        shared_detail = ("DCA selected; behavioral SRAM rejected before "
                         "CORE/DCA result or fallback")
        if not shared_ok:
            shared_detail += (f"; rc={result.returncode}; "
                              f"tail={result.stdout[-1400:]}")
        tests.append(("R5 behavioral CORE+DCA real-SRAM gate",
                      shared_ok, shared_detail))

        # A normal-unicast side flow does not authorize the behavioral
        # collective source to bypass the production SRAM byte gate.
        mixed = workload("reduce", [0, 2], 0, 257,
                         "uint8", "sum", 890)
        mixed["vars"].update(B=1, T=4, C=16, OC=16,
                             BTP=64, matmul_data=0)
        mixed["source"] = [{"dest": 3, "size": "BTP"}]
        cores = mixed["chips"][0]["cores"]
        cores[0]["worklist"] = [
            {"recv_cnt": 1, "recv_tag": 0, "cast": []}]
        cores[3]["worklist"] = [{
            "recv_cnt": 1,
            "prims": [{
                "type": "Matmul_f", "B": "B", "T": "T",
                "C": "C", "OC": "OC",
                "sram_address": {
                    "indata": "_input_label", "outdata": "mixed_out"},
                "dram_address": {"data": "matmul_data"},
            }],
            "cast": [{"dest": 0, "tag": 0}],
        }]
        work = tmp / "r6_mixed_regular_unicast.json"
        work.write_text(json.dumps(mixed))
        result = run(work, sims[("reduce_only", "integer_exact")])
        mixed_ok = behavioral_dca_rejected(result, "reduce_only")
        mixed_detail = ("DCA selected; behavioral SRAM rejected before "
                        "mixed stream result or fallback")
        if not mixed_ok:
            mixed_detail += (f"; rc={result.returncode}; "
                             f"tail={result.stdout[-1600:]}")
        tests.append(("R6 behavioral DCA + regular-unicast gate",
                      mixed_ok, mixed_detail))

        # R7 multicast mapping: one injection per Broadcast/source/result.
        r7_cases = [
            ("broadcast", [0, 1, 2, 3], 1, 65, 1, 3),
            ("allgather", [0, 1, 2, 3], 0, 33, 4, 12),
            ("reduce", [0, 1, 2, 3], 2, 67, 0, 0),
            ("reducescatter", [0, 1, 2, 3], 2, 67, 0, 0),
            ("allreduce", [0, 1, 2, 3], 2, 67, 1, 3),
        ]
        for index, (op, group, root, count, txs, rxs) in enumerate(r7_cases):
            work = tmp / f"r7_{index}_{op}.json"
            work.write_text(json.dumps(workload(
                op, group, root, count, "uint8", "sum", 900 + index)))
            result = run(work, sims[("reduce_broadcast", "integer_exact")])
            reduction = op in ("reduce", "reducescatter", "allreduce")
            if reduction:
                ok = behavioral_dca_rejected(
                    result, "reduce_broadcast")
                detail = ("DCA selected; behavioral SRAM rejected before "
                          "stream result/V5 fallback/value")
            else:
                ok = (drained(result.stdout, result.returncode) and
                      result.stdout.count("[COLL_V4_TX]") == txs and
                      result.stdout.count("[COLL_V4_RX]") == rxs)
                detail = f"multicast tx/rx={txs}/{rxs}"
            if not ok:
                detail += f"; rc={result.returncode}; tail={result.stdout[-1000:]}"
            tests.append((f"R7 {op}", ok, detail))

    for name, ok, detail in tests:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = sum(ok for _, ok, _ in tests)
    print(f"NoC collective R6/R7: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
