#!/usr/bin/env python3
"""Production WorkerCore SRAM pipeline regression and overlap oracle."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = ROOT / "build"
NPUSIM = BUILD / "npusim"
TRACE = BUILD / "events.json"
MAPPING = "../llm/test/noc_congestion/mapping/identity.spec"

MARKER = re.compile(
    r"\[SRAM_PIPELINE_DONE\] engine=(lsu|dte) "
    r"schedule=(blocking|double_buffer) tiles=(\d+) bytes=(\d+) "
    r"checksum=(\d+) measured_ns=([0-9.]+)"
)


def spans(events: list[dict], category: str, tid: int) -> list[tuple[float, float, str]]:
    opened: dict[str, list[float]] = {}
    result: list[tuple[float, float, str]] = []
    for event in events:
        if event.get("cat") != category or event.get("tid") != tid:
            continue
        name = event.get("name", "")
        if event.get("ph") == "B":
            opened.setdefault(name, []).append(float(event["ts"]))
        elif event.get("ph") == "E" and opened.get(name):
            result.append((opened[name].pop(0), float(event["ts"]), name))
    return result


def overlaps(lhs: tuple[float, float, str], rhs: tuple[float, float, str]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


def main() -> int:
    TRACE.unlink(missing_ok=True)
    command = [
        str(NPUSIM),
        "--trace-window", "1000000",
        "--workload-config", "../llm/test/sram/workload.json",
        "--hardware-config", "../llm/test/sram/hardware.json",
        "--simulation-config", "../llm/test/sram/simulation.json",
        "--mapping-config", MAPPING,
    ]
    try:
        proc = subprocess.run(
            command, cwd=BUILD, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("[SRAM PIPELINE] FAIL: workload timed out")
        return 1

    if proc.returncode != 0:
        print(proc.stdout)
        print(f"[SRAM PIPELINE] FAIL: npusim returned {proc.returncode}")
        return 1
    records = {
        (engine, schedule): {
            "tiles": int(tiles), "bytes": int(size),
            "checksum": int(checksum), "ns": float(elapsed),
        }
        for engine, schedule, tiles, size, checksum, elapsed
        in MARKER.findall(proc.stdout)
    }
    expected_keys = {
        ("lsu", "blocking"), ("lsu", "double_buffer"),
        ("dte", "blocking"), ("dte", "double_buffer"),
    }
    if records.keys() != expected_keys:
        print(proc.stdout)
        print(f"[SRAM PIPELINE] FAIL: marker set is {set(records)}")
        return 1
    if proc.stdout.count("[SRAM_REGION_BIND_DONE]") != 4:
        print("[SRAM PIPELINE] FAIL: production role-region probe did not run four times")
        return 1
    for key, record in records.items():
        if record["tiles"] != 4 or record["bytes"] != 256 or record["checksum"] <= 0:
            print(f"[SRAM PIPELINE] FAIL: invalid data oracle for {key}: {record}")
            return 1
    checksums = {record["checksum"] for record in records.values()}
    if len(checksums) != 1:
        print(f"[SRAM PIPELINE] FAIL: engine/schedule checksums differ: {checksums}")
        return 1
    for engine in ("lsu", "dte"):
        blocking = records[(engine, "blocking")]["ns"]
        pipelined = records[(engine, "double_buffer")]["ns"]
        if not pipelined < blocking:
            print(f"[SRAM PIPELINE] FAIL: {engine} pipeline {pipelined}ns >= blocking {blocking}ns")
            return 1

    if not TRACE.exists():
        print("[SRAM PIPELINE] FAIL: events.json was not generated")
        return 1
    events = json.loads(TRACE.read_text())["traceEvents"]
    required_stages = {
        ("SRAM_0", 0): "SRAM_queue",
        ("SRAM_0", 1): "SRAM_read",
        ("SRAM_0", 2): "SRAM_write",
        ("SRAM_0", 3): "SRAM_bank_wait",
        ("LSU_0", 0): "LSU_issue",
        ("LSU_0", 1): "LSU_wait",
        ("LSU_0", 2): "LSU_hbm",
        ("LSU_0", 3): "LSU_sram",
        ("DTE_mem_0", 0): "DTE_mem_commit",
        ("DTE_mem_0", 1): "DTE_mem_axi",
        ("DTE_mem_0", 2): "DTE_mem_spm",
        ("DTE_mem_0", 3): "DTE_mem_hbm",
        ("Compute_0", 0): "Compute_tile",
        ("SRAM_region_0", 0): "SRAM_region_alloc",
        ("SRAM_region_0", 1): "SRAM_region_free",
    }
    for key, label in required_stages.items():
        phases = [e.get("ph") for e in events
                  if (e.get("cat"), e.get("tid")) == key]
        if not phases or phases.count("B") != phases.count("E"):
            print(f"[SRAM PIPELINE] FAIL: unbalanced or absent {label}: {phases}")
            return 1
    region_names = {role for role in ("input", "intermediate", "comm")
                    if any(role in e.get("name", "") for e in events
                           if e.get("cat") == "SRAM_region_0")}
    if region_names != {"input", "intermediate", "comm"}:
        print(f"[SRAM PIPELINE] FAIL: role-region traces are {region_names}")
        return 1
    TRACE.unlink(missing_ok=True)
    compute = spans(events, "Compute_0", 0)
    lsu_hbm = spans(events, "LSU_0", 3)
    dte_hbm = spans(events, "DTE_mem_0", 3)
    if len(compute) != 16 or not lsu_hbm or not dte_hbm:
        print(f"[SRAM PIPELINE] FAIL: spans compute/lsu/dte={len(compute)}/{len(lsu_hbm)}/{len(dte_hbm)}")
        return 1

    groups = {
        "lsu_blocking": compute[0:4],
        "lsu_double": compute[4:8],
        "dte_blocking": compute[8:12],
        "dte_double": compute[12:16],
    }
    if any(overlaps(c, m) for c in groups["lsu_blocking"] for m in lsu_hbm):
        print("[SRAM PIPELINE] FAIL: LSU blocking compute overlaps HBM")
        return 1
    if not any(overlaps(c, m) for c in groups["lsu_double"] for m in lsu_hbm):
        print("[SRAM PIPELINE] FAIL: LSU double buffer has no compute/HBM overlap")
        return 1
    if any(overlaps(c, m) for c in groups["dte_blocking"] for m in dte_hbm):
        print("[SRAM PIPELINE] FAIL: DTE blocking compute overlaps HBM")
        return 1
    if not any(overlaps(c, m) for c in groups["dte_double"] for m in dte_hbm):
        print("[SRAM PIPELINE] FAIL: DTE double buffer has no compute/HBM overlap")
        return 1

    print("[SRAM PIPELINE] PASS: non-zero round-trip checksum", checksums.pop())
    print(
        "[SRAM PIPELINE] PASS: blocking/double ns",
        f"LSU={records[('lsu', 'blocking')]['ns']}/{records[('lsu', 'double_buffer')]['ns']}",
        f"DTE={records[('dte', 'blocking')]['ns']}/{records[('dte', 'double_buffer')]['ns']}",
    )
    print("[SRAM PIPELINE] PASS: trace proves LSU and DTE memory/compute overlap")
    print("[SRAM PIPELINE] PASS: full staged trace and input/intermediate/comm lifecycle")
    return 0


if __name__ == "__main__":
    sys.exit(main())
