#!/usr/bin/env python3
"""R8 behavioral compatibility experiment and production-DCA guard.

The common workload is Broadcast + UINT8/SUM AllReduce on a 2x2 mesh.  It
checks exact mesh-link flit-hops, backend traces and drain for the two
behavioral-SRAM-compatible profiles. DCA profiles must fail before reading a
source because production DCA requires the real SRAM data path; their positive
performance coverage belongs to the P7/P8 program gates.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"
POSITIVE_PROFILES = ("baseline", "broadcast_only")
DCA_GUARD_PROFILES = ("reduce_only", "reduce_broadcast")
PAYLOAD_BITS = (5120, 8192, 65536, 32 * 1024 * 8)
GROUP = [0, 1, 2, 3]
ROOT_CORE = 0
FLIT_BITS = 128
VECTOR_BITS = 512
DCA_LATENCY = 7
DCA_II = 1
UNICAST_EDGE_HOPS = 4  # 0->1, 0->2, 0->1->3
TREE_EDGES = 3
REAL_SRAM_ERROR = "DCA stream TX requires the real SRAM data path"


def ceil_div(value: int, divisor: int) -> int:
    return value // divisor + (value % divisor != 0)


@dataclass(frozen=True)
class Theory:
    normal_hops: int
    collective_hops: int


@dataclass(frozen=True)
class Result:
    profile: str
    bits: int
    time_ns: int
    normal_hops: int
    collective_hops: int


def theory(profile: str, bits: int) -> Theory:
    flits = ceil_div(bits, FLIT_BITS)
    if profile == "baseline":
        normal, collective = 3 * UNICAST_EDGE_HOPS * flits, 0
    elif profile == "broadcast_only":
        normal = 2 * UNICAST_EDGE_HOPS * flits
        collective = TREE_EDGES * flits
    else:
        raise ValueError(profile)
    return Theory(normal, collective)


def collective(op: str, cid: int, bits: int, terminal: bool) -> dict:
    value = {
        "op": op, "group_id": 108, "collective_id": cid,
        "group": GROUP, "root": ROOT_CORE, "count": bits // 8,
        "chunk_bits": bits, "stride_bits": bits, "terminal": terminal,
    }
    if op == "allreduce":
        value.update(dtype="uint8", reduce_op="sum")
    return value


def workload(bits: int) -> dict:
    return {
        "vars": {"B": 1, "T": 1}, "pipeline": 1, "source": [],
        "chips": [{"chip_id": 0,
                   "cores": [{"id": i, "loop": 1, "worklist": []}
                             for i in GROUP],
                   "collectives": [collective("broadcast", 1000, bits, False),
                                   collective("allreduce", 1001, bits, True)]}],
    }


def simulation(profile: str) -> dict:
    sim = json.loads((HERE / "simulation" / "v1_cycle.json").read_text())
    sim["noc"]["collective"] = {
        "enabled": True, "profile": profile,
        "dca": {
            "vector_bits": VECTOR_BITS, "slice_bits": 64,
            "slices_per_tile": 8, "header_fifo_depth": 8,
            "operand_fifo_depth": 8, "result_fifo_depth": 2,
            "arbitration": "round_robin", "value_mode": "integer_exact",
            "latency": {"uint8": {"sum": DCA_LATENCY}},
            "initiation_interval": {"uint8": {"sum": DCA_II}},
        },
    }
    return sim


def parse(profile: str, bits: int, output: str) -> Result:
    match = re.search(r"Catch test finished.*?\|.*?(\d+) ns", output)
    if not match:
        raise AssertionError(f"{profile}/{bits}: missing completion")
    shared = [tuple(map(int, row)) for row in re.findall(
        r"\[COLL_SHARED\] router=(\d+) output=(\d+) "
        r"normal_flits=(\d+) collective_flits=(\d+)", output)]
    return Result(profile, bits, int(match.group(1)),
                  sum(row[2] for row in shared),
                  sum(row[3] for row in shared))


def run_process(
        profile: str, bits: int,
        tmp: Path) -> subprocess.CompletedProcess[str]:
    work = tmp / f"work_{bits}.json"
    sim = tmp / f"{profile}.json"
    work.write_text(json.dumps(workload(bits)))
    sim.write_text(json.dumps(simulation(profile)))
    return subprocess.run(
        [str(NPUSIM), "--workload-config", str(work),
         "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
         "--simulation-config", str(sim),
         "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"],
        cwd=ROOT / "build", text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=180)


def run_positive(profile: str, bits: int, tmp: Path) -> Result:
    process = run_process(profile, bits, tmp)
    if process.returncode != 0:
        raise AssertionError(f"{profile}/{bits} rc={process.returncode}\n" +
                             process.stdout[-2000:])
    drained = ("[DRAIN] router_residual=0" in process.stdout and
               "data_balanced=1 ctrl_balanced=1" in process.stdout and
               "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
               "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0" in process.stdout)
    if not drained:
        raise AssertionError(f"{profile}/{bits}: residual state")
    expected_modes = {
        "baseline": (0, 0), "broadcast_only": (1, 0),
    }
    actual_modes = (process.stdout.count("[COLL_V4_TX]"),
                    process.stdout.count("[COLL_STREAM_TX]"))
    if actual_modes != expected_modes[profile]:
        raise AssertionError(f"{profile}/{bits}: backend trace "
                             f"{actual_modes}!={expected_modes[profile]}")
    return parse(profile, bits, process.stdout)


def run_dca_guard(profile: str, bits: int, tmp: Path) -> None:
    process = run_process(profile, bits, tmp)
    output = process.stdout
    forbidden = (
        "Catch test finished",
        "[COLL_STREAM_RESULT]",
        "[COLL_DCA_RESULT]",
        "[COLL_V5_RESULT]",
        "[COLL_REDUCE_COMPUTE]",
        "value=",
        "fallback",
    )
    valid = (
        process.returncode != 0
        and f"profile={profile}" in output
        and "reduce_backend=dca_offload" in output
        and output.count(REAL_SRAM_ERROR) == 1
        and all(marker not in output for marker in forbidden)
    )
    if not valid:
        raise AssertionError(
            f"{profile}/{bits}: invalid behavioral-SRAM DCA guard; "
            f"rc={process.returncode}\n{output[-2000:]}")


def validate(results: list[Result]) -> None:
    for result in results:
        expected = theory(result.profile, result.bits)
        actual_hops = (result.normal_hops, result.collective_hops)
        expected_hops = (expected.normal_hops, expected.collective_hops)
        if actual_hops != expected_hops:
            raise AssertionError(f"{result.profile}/{result.bits}: flit-hop "
                                 f"{actual_hops}!={expected_hops}")


def report(results: list[Result]) -> None:
    print("R8 behavioral compatibility + production-DCA guard")
    print("positive workload: 2x2, Broadcast + UINT8/SUM AllReduce")
    print("| payload | profile | time(ns) | normal hops | collective hops | "
          "vs baseline |")
    print("|---:|---|---:|---:|---:|---:|")
    lookup = {(r.bits, r.profile): r for r in results}
    for bits in PAYLOAD_BITS:
        baseline = lookup[(bits, "baseline")].time_ns
        for profile in POSITIVE_PROFILES:
            result = lookup[(bits, profile)]
            print(f"| {bits // 8} B | {profile} | {result.time_ns} | "
                  f"{result.normal_hops} | {result.collective_hops} | "
                  f"{baseline / result.time_ns:.3f}x |")


def main() -> int:
    selftest = subprocess.run(
        [str(NPUSIM), "--coll-r8-selftest"], cwd=ROOT / "build",
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=30)
    if selftest.returncode != 0 or "R8 self-test: 6/6" not in selftest.stdout:
        raise AssertionError("R8 standalone oracle self-test failed\n" +
                             selftest.stdout[-1200:])
    with tempfile.TemporaryDirectory(prefix="coll_r8_") as td:
        tmp = Path(td)
        results = [run_positive(profile, bits, tmp) for bits in PAYLOAD_BITS
                   for profile in POSITIVE_PROFILES]
        for bits in PAYLOAD_BITS:
            for profile in DCA_GUARD_PROFILES:
                run_dca_guard(profile, bits, tmp)
    validate(results)
    report(results)
    for bits in PAYLOAD_BITS:
        for profile in DCA_GUARD_PROFILES:
            print(f"[PASS] {profile}/{bits}: behavioral SRAM rejected before "
                  "DCA source read; no completion/result/value/fallback")
    print("R8 standalone oracle self-test: 6/6")
    print("Production DCA performance is covered by the P7/P8 program gates.")
    total = len(results) + len(PAYLOAD_BITS) * len(DCA_GUARD_PROFILES)
    print("NoC collective R8 behavioral compatibility + production-DCA "
          f"guard: PASS ({total} cases: 8 positive + 8 negative)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
