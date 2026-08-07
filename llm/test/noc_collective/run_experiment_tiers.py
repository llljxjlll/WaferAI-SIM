#!/usr/bin/env python3
"""Compare NoC collective Tier0/Tier1/Tier2 on one identical workload.

The workload is a four-rank Broadcast followed by UINT8/SUM AllReduce on a
2x2 mesh.  Three payload sizes expose both the one-chunk fixed-overhead case
and the multi-chunk DCA-throughput case.  The test independently derives XY
unicast/tree flit-hop counts, then checks the production traces and completion
cycles for every tier.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"
GROUP = [0, 1, 2, 3]
ROOT_CORE = 0
MESH_X = 2
PAYLOAD_BITS = (128, 512, 2048)
FLIT_PAYLOAD_BITS = 128


@dataclass(frozen=True)
class Theory:
    normal_flit_hops: int
    collective_flit_hops: int

    @property
    def total_flit_hops(self) -> int:
        return self.normal_flit_hops + self.collective_flit_hops


@dataclass(frozen=True)
class Result:
    bits: int
    tier: int
    time_ns: int
    normal_flit_hops: int
    collective_flit_hops: int


def ceil_div(value: int, divisor: int) -> int:
    return value // divisor + (value % divisor != 0)


def xy_path(src: int, dst: int) -> list[tuple[int, int]]:
    """Return directed router edges for the simulator's X-then-Y routing."""
    path: list[tuple[int, int]] = []
    current = src
    while current % MESH_X != dst % MESH_X:
        nxt = current + (1 if dst % MESH_X > current % MESH_X else -1)
        path.append((current, nxt))
        current = nxt
    while current != dst:
        nxt = current + (MESH_X if dst > current else -MESH_X)
        path.append((current, nxt))
        current = nxt
    return path


def theory(bits: int, tier: int) -> Theory:
    flits = ceil_div(bits, FLIT_PAYLOAD_BITS)
    unicast_broadcast_edges = sum(
        len(xy_path(ROOT_CORE, dst)) for dst in GROUP if dst != ROOT_CORE
    )
    tree_edges = len({
        edge
        for dst in GROUP
        if dst != ROOT_CORE
        for edge in xy_path(ROOT_CORE, dst)
    })
    tier0_allreduce_edges = 2 * unicast_broadcast_edges

    if tier == 0:
        return Theory(
            (unicast_broadcast_edges + tier0_allreduce_edges) * flits, 0
        )
    if tier == 1:
        return Theory(tier0_allreduce_edges * flits, tree_edges * flits)
    if tier == 2:
        # Initial Broadcast + two-segment reduce operand per tree edge +
        # AllReduce result Broadcast.
        return Theory(0, (tree_edges + 2 * tree_edges + tree_edges) * flits)
    raise ValueError(f"invalid collective tier: {tier}")


def collective(op: str, collective_id: int, bits: int,
               terminal: bool) -> dict:
    result = {
        "op": op,
        "collective_id": collective_id,
        "group": GROUP,
        "root": ROOT_CORE,
        "count": bits // 8,
        "chunk_bits": bits,
        "stride_bits": bits,
        "terminal": terminal,
    }
    if op == "allreduce":
        result.update(dtype="uint8", reduce_op="sum")
    return result


def workload(bits: int) -> dict:
    return {
        "vars": {"B": 1, "T": 1},
        "pipeline": 1,
        "source": [],
        "chips": [{
            "chip_id": 0,
            "cores": [
                {"id": core, "loop": 1, "worklist": []} for core in GROUP
            ],
            "collectives": [
                collective("broadcast", 800, bits, False),
                collective("allreduce", 801, bits, True),
            ],
        }],
    }


def hardware() -> dict:
    result = json.loads((HERE / "hardware" / "v1.json").read_text())
    result["x"] = MESH_X
    return result


def simulation(tier: int) -> dict:
    result = json.loads(
        (HERE / "simulation" / "v1_cycle.json").read_text()
    )
    coll = result["noc"]["collective"]
    if tier == 2:
        coll.pop("tier", None)
        coll.update(broadcast_backend="multicast",
                    reduce_backend="legacy_router_alu",
                    reduce_wire="legacy_two_segment",
                    allow_legacy_backend=True)
    else:
        coll["tier"] = tier
    return result


def parse_result(bits: int, tier: int, output: str) -> Result:
    cycle = re.search(r"Catch test finished.*?\|.*?(\d+) ns", output)
    if cycle is None:
        raise AssertionError(f"tier{tier}/{bits}b did not finish")

    shared = [
        tuple(map(int, row))
        for row in re.findall(
            r"\[COLL_SHARED\] router=(\d+) output=(\d+) "
            r"normal_flits=(\d+) collective_flits=(\d+)",
            output,
        )
    ]
    if not shared:
        raise AssertionError(f"tier{tier}/{bits}b has no COLL_SHARED trace")
    return Result(
        bits,
        tier,
        int(cycle.group(1)),
        sum(row[2] for row in shared),
        sum(row[3] for row in shared),
    )


def run_case(bits: int, tier: int, directory: Path) -> Result:
    work_path = directory / f"work_{bits}.json"
    hardware_path = directory / "hardware.json"
    simulation_path = directory / f"tier{tier}.json"
    work_path.write_text(json.dumps(workload(bits)))
    hardware_path.write_text(json.dumps(hardware()))
    simulation_path.write_text(json.dumps(simulation(tier)))

    process = subprocess.run(
        [
            str(NPUSIM),
            "--workload-config", str(work_path),
            "--hardware-config", str(hardware_path),
            "--simulation-config", str(simulation_path),
            "--mapping-config",
            "../llm/test/noc_collective/mapping/identity.spec",
        ],
        cwd=ROOT / "build",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    if process.returncode != 0:
        raise AssertionError(
            f"tier{tier}/{bits}b exited {process.returncode}\n"
            + "\n".join(process.stdout.splitlines()[-40:])
        )

    drain = (
        "[DRAIN] router_residual=0" in process.stdout
        and "data_balanced=1 ctrl_balanced=1" in process.stdout
        and "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
        "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0" in process.stdout
    )
    if not drain:
        raise AssertionError(f"tier{tier}/{bits}b did not fully drain")

    expected_modes = {
        0: (0, 0),
        1: (1, 0),
        2: (2, 1),
    }
    actual_modes = (
        process.stdout.count("[COLL_V4_TX]"),
        process.stdout.count("[COLL_V5_RESULT]"),
    )
    if actual_modes != expected_modes[tier]:
        raise AssertionError(
            f"tier{tier}/{bits}b mode mismatch: "
            f"expected={expected_modes[tier]} actual={actual_modes}"
        )
    return parse_result(bits, tier, process.stdout)


def validate(results: list[Result]) -> None:
    by_case = {(result.bits, result.tier): result for result in results}
    for result in results:
        expected = theory(result.bits, result.tier)
        actual = (result.normal_flit_hops, result.collective_flit_hops)
        wanted = (expected.normal_flit_hops,
                  expected.collective_flit_hops)
        if actual != wanted:
            raise AssertionError(
                f"tier{result.tier}/{result.bits}b flit-hop mismatch: "
                f"theory={wanted} actual={actual}"
            )

    one_chunk = [by_case[(128, tier)].time_ns for tier in range(3)]
    if not one_chunk[2] < one_chunk[1] < one_chunk[0]:
        raise AssertionError(
            f"one-chunk cycle order must be T2<T1<T0, got {one_chunk}"
        )
    for bits in (512, 2048):
        cycles = [by_case[(bits, tier)].time_ns for tier in range(3)]
        if not cycles[1] < cycles[0] < cycles[2]:
            raise AssertionError(
                f"multi-chunk cycle order must be T1<T0<T2 at {bits}b, "
                f"got {cycles}"
            )


def print_report(results: list[Result]) -> None:
    by_case = {(result.bits, result.tier): result for result in results}
    print("NoC collective tier comparison")
    print("workload: 2x2 mesh, ranks=[0,1,2,3], root=0, "
          "Broadcast + UINT8/SUM AllReduce")
    print()
    print("| bits | chunks | tier | time (ns) | normal hops | collective hops "
          "| total hops | vs Tier0 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bits in PAYLOAD_BITS:
        baseline = by_case[(bits, 0)].time_ns
        for tier in range(3):
            result = by_case[(bits, tier)]
            speedup = baseline / result.time_ns
            print(
                f"| {bits} | {ceil_div(bits, FLIT_PAYLOAD_BITS)} | "
                f"{tier} | {result.time_ns} | "
                f"{result.normal_flit_hops} | "
                f"{result.collective_flit_hops} | "
                f"{result.normal_flit_hops + result.collective_flit_hops} | "
                f"{speedup:.3f}x |"
            )


def main() -> int:
    if not NPUSIM.exists():
        print(f"missing simulator: {NPUSIM}", file=sys.stderr)
        return 2
    try:
        with tempfile.TemporaryDirectory(prefix="coll_tier_experiment_") as td:
            results = [
                run_case(bits, tier, Path(td))
                for bits in PAYLOAD_BITS
                for tier in range(3)
            ]
        validate(results)
        print_report(results)
        print("\nNoC collective tier experiment: PASS (9/9 runs)")
        return 0
    except (AssertionError, subprocess.TimeoutExpired) as error:
        print(f"NoC collective tier experiment: FAIL: {error}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
