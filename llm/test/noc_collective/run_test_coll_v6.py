#!/usr/bin/env python3
"""V6 mixed-traffic, multi-tree and lifecycle production matrix."""
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


def sim_config(tier: int) -> dict:
    cfg = json.loads((HERE / "simulation" / "v1_cycle.json").read_text())
    coll = cfg["noc"]["collective"]
    if tier == 2:
        coll.pop("tier", None)
        coll.update(broadcast_backend="multicast",
                    reduce_backend="legacy_router_alu",
                    reduce_wire="legacy_two_segment",
                    allow_legacy_backend=True)
    else:
        coll["tier"] = tier
    return cfg


def hardware() -> dict:
    cfg = json.loads((HERE / "hardware" / "v1.json").read_text())
    cfg["x"] = 4
    return cfg


def empty_cores() -> list[dict]:
    return [{"id": i, "loop": 1, "worklist": []} for i in range(4)]


def collective(op: str, cid: int, group: list[int], root: int,
               *, terminal: bool = True, epoch: int = 0,
               chunk_bits: int = 4096) -> dict:
    item = {"op": op, "collective_id": cid, "epoch": epoch,
            "group": group, "root": root, "count": chunk_bits // 8,
            "chunk_bits": chunk_bits, "stride_bits": chunk_bits,
            "terminal": terminal}
    if op in ("reduce", "reducescatter", "allreduce"):
        item.update(dtype="uint8", reduce_op="sum")
    return item


def workload(collectives: list[dict], cores: list[dict] | None = None,
             source: list[dict] | None = None, vars_: dict | None = None) -> dict:
    return {"vars": vars_ or {"B": 1, "T": 1}, "pipeline": 1,
            "source": source or [],
            "chips": [{"chip_id": 0, "cores": cores or empty_cores(),
                       "collectives": collectives}]}


def run(work: Path, hw: Path, sim: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(NPUSIM), "--workload-config", str(work),
         "--hardware-config", str(hw),
         "--simulation-config", str(sim),
         "--mapping-config", "../llm/test/noc_collective/mapping/identity.spec"],
        cwd=ROOT / "build", text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60)


def drained(out: str) -> bool:
    return ("Catch test finished" in out and
            "[DRAIN] router_residual=0" in out and
            "data_balanced=1 ctrl_balanced=1" in out and
            "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
            "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0" in out)


def shared_rows(out: str) -> dict[tuple[int, int], tuple[int, int]]:
    rows: dict[tuple[int, int], tuple[int, int]] = {}
    for r, d, normal, coll in re.findall(
            r"\[COLL_SHARED\] router=(\d+) output=(\d+) "
            r"normal_flits=(\d+) collective_flits=(\d+)", out):
        rows[(int(r), int(d))] = (int(normal), int(coll))
    return rows


def tree_rows(out: str) -> list[tuple[int, int, int, int, int]]:
    return [tuple(map(int, row)) for row in re.findall(
        r"\[COLL_LINK\] tree=(\d+) router=(\d+) output=(\d+) "
        r"flits=(\d+) stalls=(\d+)", out)]


def stable_group(group: list[int]) -> int:
    h = 2166136261
    for value in group:
        h = ((h ^ value) * 16777619) & 0xffffffff
    return h


def tree_id(group_id: int, cid: int, epoch: int = 0) -> int:
    h = 2166136261
    for value in (group_id, cid, epoch):
        h = ((h ^ value) * 16777619) & 0xffffffff
    return 1 + h % 0xffff


def collision_ids(group: list[int]) -> tuple[int, int]:
    gid = stable_group(group)
    seen: dict[int, int] = {}
    for cid in range(200000):
        tid = tree_id(gid, cid)
        if tid in seen:
            return seen[tid], cid
        seen[tid] = cid
    raise RuntimeError("failed to find deterministic 16-bit tree collision")


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    unit = subprocess.run([str(NPUSIM), "--coll-v6-selftest"], cwd=ROOT / "build",
                          text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=20)
    tests.append(("V6 lifecycle contract",
                  unit.returncode == 0 and "PASS (11 checks)" in unit.stdout,
                  "11 checks"))

    with tempfile.TemporaryDirectory(prefix="coll_v6_") as td:
        tmp = Path(td)
        hw = tmp / "hardware.json"
        hw.write_text(json.dumps(hardware()))
        sims: dict[int, Path] = {}
        for tier in (1, 2):
            sims[tier] = tmp / f"tier{tier}.json"
            sims[tier].write_text(json.dumps(sim_config(tier)))

        multi_b = workload([
            collective("broadcast", 601, [0, 3], 0, chunk_bits=8192),
            collective("broadcast", 602, [1, 2], 1, chunk_bits=8192),
        ])
        p = tmp / "multi_broadcast.json"
        p.write_text(json.dumps(multi_b))
        result = run(p, hw, sims[1])
        links = tree_rows(result.stdout)
        shared_tree_ids = {tree for tree, router, output, flits, _ in links
                           if router == 1 and output == 1 and flits == 64}
        contended = sum(stalls for _, router, output, _, stalls in links
                        if router == 1 and output == 1) > 0
        ok = (result.returncode == 0 and drained(result.stdout) and
              result.stdout.count("[COLL_V4_TX]") == 2 and
              result.stdout.count("[COLL_V4_RX]") == 2 and
              result.stdout.count("[COLL_V6_RELEASE]") == 2 and
              len(shared_tree_ids) == 2 and contended)
        tests.append(("Tier1 two-tree shared-link contention", ok,
                      f"trees={len(shared_tree_ids)} contended={contended}"))

        multi_r = workload([
            collective("reduce", 611, [0, 3], 0, chunk_bits=2048),
            collective("reduce", 612, [1, 2], 1, chunk_bits=2048),
        ])
        p = tmp / "multi_reduce.json"
        p.write_text(json.dumps(multi_r))
        result = run(p, hw, sims[2])
        ok = (result.returncode == 0 and drained(result.stdout) and
              result.stdout.count("[COLL_V5_TX]") == 4 and
              result.stdout.count("[COLL_V5_RESULT]") == 2 and
              result.stdout.count("[COLL_V6_RELEASE]") == 2)
        tests.append(("Tier2 two reduce trees", ok,
                      "4 operands streams, 2 verified roots"))

        cores = empty_cores()
        cores[1]["worklist"] = [{
            "recv_cnt": 1,
            "prims": [{"type": "Matmul_f", "B": "B", "T": "T",
                       "C": "C", "OC": "OC",
                       "sram_address": {"indata": "_input_label",
                                        "outdata": "mixed_out"},
                       "dram_address": {"data": "matmul_data"}}],
            "cast": [{"dest": 2, "tag": 2}]}]
        cores[2]["worklist"] = [{"recv_cnt": 1, "recv_tag": 2, "cast": []}]
        mixed = workload(
            [collective("broadcast", 621, [0, 3], 0,
                        chunk_bits=65536)], cores,
            source=[{"dest": 1, "size": "BTP"}],
            vars_={"B": 1, "T": 16, "C": 16, "OC": 16,
                   "BTP": 4096, "matmul_data": 0})
        p = tmp / "mixed.json"
        p.write_text(json.dumps(mixed))
        result = run(p, hw, sims[1])
        row = shared_rows(result.stdout).get((1, 1), (0, 0))
        ok = (result.returncode == 0 and drained(result.stdout) and
              result.stdout.count("[COLL_V4_TX]") == 1 and
              result.stdout.count("[COLL_V4_RX]") == 1 and
              row[0] > 0 and row[1] > 0 and "1:2:2@" in result.stdout)
        tests.append(("Tier1 collective + regular DATA shared output", ok,
                      f"router1/E normal={row[0]} collective={row[1]}"))

        epochs = workload([
            collective("broadcast", 631, [0, 3], 0, epoch=0,
                       chunk_bits=1024),
            collective("broadcast", 631, [0, 3], 0, epoch=1,
                       chunk_bits=1024),
        ])
        p = tmp / "epochs.json"
        p.write_text(json.dumps(epochs))
        result = run(p, hw, sims[1])
        ok = (result.returncode == 0 and drained(result.stdout) and
              result.stdout.count("[COLL_V4_TX]") == 2 and
              result.stdout.count("[COLL_V4_RX]") == 2 and
              result.stdout.count("[COLL_V6_RELEASE]") == 2)
        tests.append(("consecutive epoch tree lifecycle", ok,
                      "2 releases, final registry zero"))

        first, second = collision_ids([0, 3])
        collision = workload([
            collective("broadcast", first, [0, 3], 0),
            collective("broadcast", second, [0, 3], 0),
        ])
        p = tmp / "collision.json"
        p.write_text(json.dumps(collision))
        result = run(p, hw, sims[1])
        ok = (result.returncode != 0 and
              "collective tree_id hash collision" in result.stdout)
        tests.append(("tree-ID collision startup gate", ok,
                      f"ids={first}/{second}"))

    for name, ok, detail in tests:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = sum(ok for _, ok, _ in tests)
    print(f"NoC collective V6: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
