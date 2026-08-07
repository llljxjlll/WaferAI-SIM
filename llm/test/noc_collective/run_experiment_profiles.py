#!/usr/bin/env python3
"""R8 four-profile NoC collective experiment and independent oracle.

The common workload is Broadcast + UINT8/SUM AllReduce on a 2x2 mesh.  It
checks exact mesh-link flit-hops, vector issue counts, backend traces, drain,
and reports the L/II decomposition without charging fixed latency per chunk.
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
PROFILES = ("baseline", "broadcast_only", "reduce_only", "reduce_broadcast")
PAYLOAD_BITS = (5120, 8192, 65536, 32 * 1024 * 8)
GROUP = [0, 1, 2, 3]
ROOT_CORE = 0
FLIT_BITS = 128
VECTOR_BITS = 512
DCA_LATENCY = 7
DCA_II = 1
UNICAST_EDGE_HOPS = 4  # 0->1, 0->2, 0->1->3
TREE_EDGES = 3


def ceil_div(value: int, divisor: int) -> int:
    return value // divisor + (value % divisor != 0)


@dataclass(frozen=True)
class Theory:
    normal_hops: int
    collective_hops: int
    vector_beats: int
    dca_issues: int
    dca_fill: int
    dca_steady: int


@dataclass(frozen=True)
class Result:
    profile: str
    bits: int
    time_ns: int
    normal_hops: int
    collective_hops: int
    dca_issues: int
    dca_completions: int


def theory(profile: str, bits: int) -> Theory:
    flits = ceil_div(bits, FLIT_BITS)
    beats = ceil_div(bits, VECTOR_BITS)
    uses_dca = profile in ("reduce_only", "reduce_broadcast")
    # Root fan-in is 3 (2 pairwise issues/beat), router 2 fan-in is 2
    # (1 issue/beat): 3B total across the tree.
    issues = 3 * beats if uses_dca else 0
    if profile == "baseline":
        normal, collective = 3 * UNICAST_EDGE_HOPS * flits, 0
    elif profile == "broadcast_only":
        normal = 2 * UNICAST_EDGE_HOPS * flits
        collective = TREE_EDGES * flits
    elif profile == "reduce_only":
        normal = 2 * UNICAST_EDGE_HOPS * flits
        collective = TREE_EDGES * (1 + flits)
    elif profile == "reduce_broadcast":
        normal = 0
        collective = TREE_EDGES * (3 * flits + 1)
    else:
        raise ValueError(profile)
    return Theory(normal, collective, beats, issues,
                  DCA_LATENCY if uses_dca else 0,
                  (issues - 1) * DCA_II if issues else 0)


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
    dca = [tuple(map(int, row)) for row in re.findall(
        r"\[COLL_DCA\] router=\d+ core_issues=(\d+) "
        r"dca_issues=(\d+) completions=(\d+)", output)]
    return Result(profile, bits, int(match.group(1)),
                  sum(row[2] for row in shared),
                  sum(row[3] for row in shared),
                  sum(row[1] for row in dca),
                  sum(row[2] for row in dca))


def run(profile: str, bits: int, tmp: Path) -> Result:
    work = tmp / f"work_{bits}.json"
    sim = tmp / f"{profile}.json"
    work.write_text(json.dumps(workload(bits)))
    sim.write_text(json.dumps(simulation(profile)))
    process = subprocess.run(
        [str(NPUSIM), "--workload-config", str(work),
         "--hardware-config", "../llm/test/noc_collective/hardware/v1.json",
         "--simulation-config", str(sim),
         "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"],
        cwd=ROOT / "build", text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=180)
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
        "reduce_only": (0, 4), "reduce_broadcast": (2, 4),
    }
    actual_modes = (process.stdout.count("[COLL_V4_TX]"),
                    process.stdout.count("[COLL_STREAM_TX]"))
    if actual_modes != expected_modes[profile]:
        raise AssertionError(f"{profile}/{bits}: backend trace "
                             f"{actual_modes}!={expected_modes[profile]}")
    return parse(profile, bits, process.stdout)


def validate(results: list[Result]) -> None:
    for result in results:
        expected = theory(result.profile, result.bits)
        actual_hops = (result.normal_hops, result.collective_hops)
        expected_hops = (expected.normal_hops, expected.collective_hops)
        if actual_hops != expected_hops:
            raise AssertionError(f"{result.profile}/{result.bits}: flit-hop "
                                 f"{actual_hops}!={expected_hops}")
        if (result.dca_issues, result.dca_completions) != (
                expected.dca_issues, expected.dca_issues):
            raise AssertionError(f"{result.profile}/{result.bits}: DCA issue "
                                 f"{result.dca_issues}/completion "
                                 f"{result.dca_completions}, expected "
                                 f"{expected.dca_issues}")
    # Fixed latency is one pipeline-fill term. The oracle deliberately uses
    # L+(issues-1)*II, never issues*L; assert that distinction is material.
    largest = theory("reduce_only", PAYLOAD_BITS[-1])
    assert largest.dca_fill + largest.dca_steady < (
        largest.dca_issues * DCA_LATENCY)


def report(results: list[Result]) -> None:
    print("NoC collective R8 four-profile experiment")
    print("workload: 2x2, Broadcast + UINT8/SUM AllReduce, L=7, II=1")
    print("| payload | profile | time(ns) | normal hops | collective hops | "
          "B | DCA issues | fill | steady | vs baseline |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lookup = {(r.bits, r.profile): r for r in results}
    for bits in PAYLOAD_BITS:
        baseline = lookup[(bits, "baseline")].time_ns
        for profile in PROFILES:
            result = lookup[(bits, profile)]
            expected = theory(profile, bits)
            print(f"| {bits // 8} B | {profile} | {result.time_ns} | "
                  f"{result.normal_hops} | {result.collective_hops} | "
                  f"{expected.vector_beats} | {result.dca_issues} | "
                  f"{expected.dca_fill} | {expected.dca_steady} | "
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
        results = [run(profile, bits, tmp) for bits in PAYLOAD_BITS
                   for profile in PROFILES]
    validate(results)
    report(results)
    print("R8 standalone oracle self-test: 6/6")
    print(f"NoC collective R8: PASS ({len(results)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
