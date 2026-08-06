#!/usr/bin/env python3
"""Legacy SRAM/NpuBase compatibility workload regression."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "build"

def main() -> int:
    command = [
        str(BUILD / "npusim"),
        "--trace-window", "1000000",
        "--workload-config", "../llm/test/sram/workload_compat.json",
        "--hardware-config", "../llm/test/sram/hardware_compat.json",
        "--simulation-config", "../llm/test/sram/simulation.json",
        "--mapping-config", "../llm/test/noc_congestion/mapping/identity.spec",
    ]
    try:
        proc = subprocess.run(command, cwd=BUILD, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=60)
    except subprocess.TimeoutExpired:
        print("[SRAM COMPAT] FAIL: workload timed out")
        return 1
    if proc.returncode != 0 or "[PROTO_WAIT]" in proc.stdout:
        print(proc.stdout)
        print(f"[SRAM COMPAT] FAIL: npusim returned {proc.returncode}")
        return 1
    if "Core 0 end compute primitive Relu_f" not in proc.stdout:
        print("[SRAM COMPAT] FAIL: legacy Relu_f did not complete")
        return 1
    print("[SRAM COMPAT] PASS: real-data off legacy NpuBase helper path completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
