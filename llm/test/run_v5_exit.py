#!/usr/bin/env python3
"""V5 aggregate freeze gate. Any non-zero child exit blocks the baseline tag."""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D2D = os.path.join(HERE, "d2d_link")

GATES = [
    ("V0-V4 frozen aggregate gate",
     [sys.executable, os.path.join(HERE, "run_v4_exit.py")]),
    ("V5 multi-port/striping/dynamic production (23/23)",
     [sys.executable, os.path.join(D2D, "run_test_d2d_v5.py")]),
]


def main() -> int:
    results = []
    for name, command in GATES:
        print(f"\n########## V5 GATE: {name} ##########", flush=True)
        try:
            rc = subprocess.run(command, timeout=300).returncode
        except subprocess.TimeoutExpired:
            rc = 124
            print("gate timed out after 300 seconds", flush=True)
        results.append((name, rc))

    print("\n==================== D2D V5 FREEZE GATE ====================")
    for name, rc in results:
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] {name} (exit={rc})")
    ok = all(rc == 0 for _, rc in results)
    print(f"AGGREGATE EXIT={0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
