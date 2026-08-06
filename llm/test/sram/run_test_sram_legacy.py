#!/usr/bin/env python3
"""Legacy-private real SRAM/HBM production transport regression."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = ROOT / "build"
MARKER = re.compile(
    r"\[SRAM_PIPELINE_DONE\] engine=(lsu|dte) "
    r"schedule=(blocking|double_buffer) tiles=(\d+) bytes=(\d+) "
    r"checksum=(\d+) measured_ns=([0-9.]+)"
)


def main() -> int:
    command = [
        str(BUILD / "npusim"),
        "--trace-window", "1000000",
        "--workload-config", "../llm/test/sram/workload.json",
        "--hardware-config", "../llm/test/sram/hardware_legacy.json",
        "--simulation-config", "../llm/test/sram/simulation.json",
        "--mapping-config", "../llm/test/noc_congestion/mapping/identity.spec",
    ]
    try:
        proc = subprocess.run(
            command, cwd=BUILD, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=60,
        )
    except subprocess.TimeoutExpired:
        print("[SRAM LEGACY] FAIL: workload timed out")
        return 1
    if proc.returncode != 0:
        print(proc.stdout)
        print(f"[SRAM LEGACY] FAIL: npusim returned {proc.returncode}")
        return 1
    records = {
        (engine, schedule): (int(tiles), int(size), int(checksum), float(ns))
        for engine, schedule, tiles, size, checksum, ns
        in MARKER.findall(proc.stdout)
    }
    expected = {
        ("lsu", "blocking"), ("lsu", "double_buffer"),
        ("dte", "blocking"), ("dte", "double_buffer"),
    }
    if records.keys() != expected:
        print(f"[SRAM LEGACY] FAIL: marker set is {set(records)}")
        return 1
    checksums = {record[2] for record in records.values()}
    if len(checksums) != 1 or next(iter(checksums)) <= 0 or any(
        record[0:2] != (4, 256) for record in records.values()
    ):
        print(f"[SRAM LEGACY] FAIL: data records are {records}")
        return 1
    for engine in ("lsu", "dte"):
        if records[(engine, "double_buffer")][3] >= records[(engine, "blocking")][3]:
            print(f"[SRAM LEGACY] FAIL: {engine} double buffer did not improve time")
            return 1
    print(
        "[SRAM LEGACY] PASS: LSU/DTE real payload round-trip checksum",
        checksums.pop(),
    )
    print("[SRAM LEGACY] PASS: legacy_private transport preserves overlap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
