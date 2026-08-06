#!/usr/bin/env python3
"""Two-die remote-NUMA WorkerCore SRAM regression."""
import re
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"
MARKER = re.compile(r"\[SRAM_PIPELINE_DONE\] engine=(lsu|dte) schedule=double_buffer tiles=4 bytes=256 checksum=(\d+) measured_ns=([0-9.]+)")
D2D = re.compile(r"\[D2D\] in_pkts=(\d+) out_pkts=(\d+)")
def main():
    cmd = [str(BUILD / "npusim"), "--trace-window", "1000000", "--workload-config", "../llm/test/sram/workload_numa.json", "--hardware-config", "../llm/test/sram/hardware_numa.json", "--simulation-config", "../llm/test/sram/simulation.json", "--mapping-config", "../llm/test/noc_congestion/mapping/identity.spec"]
    try:
        p = subprocess.run(cmd, cwd=BUILD, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    except subprocess.TimeoutExpired:
        print("[SRAM NUMA] FAIL: timeout"); return 1
    if p.returncode:
        print(p.stdout); print("[SRAM NUMA] FAIL: npusim", p.returncode); return 1
    rows = MARKER.findall(p.stdout)
    if {r[0] for r in rows} != {"lsu", "dte"} or len(rows) != 2 or any(int(r[1]) <= 0 for r in rows):
        print("[SRAM NUMA] FAIL: pipeline markers", rows); return 1
    d2d = D2D.findall(p.stdout)
    if not d2d or int(d2d[-1][0]) == 0 or int(d2d[-1][1]) == 0:
        print("[SRAM NUMA] FAIL: no D2D traffic", d2d); return 1
    print("[SRAM NUMA] PASS: core0 LSU/core16 DTE remote-home round trips", rows)
    print("[SRAM NUMA] PASS: D2D packets in/out", d2d[-1])
    return 0
if __name__ == "__main__": sys.exit(main())
