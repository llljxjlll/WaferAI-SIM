#!/usr/bin/env python3
"""V4 aggregate freeze gate. Any non-zero child exit blocks the baseline tag."""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D2D = os.path.join(HERE, "d2d_link")

GATES = [
    ("V0-V2 historical self/link/e2e (all pass)",
     [sys.executable, os.path.join(D2D, "run_test_d2d_v0.py")]),
    ("V3 production bounded SAF (16/16)",
     [sys.executable, os.path.join(D2D, "run_test_d2d_v3.py")]),
    ("V4 independent Python oracle (8/8)",
     [sys.executable, os.path.join(D2D, "behavioral", "oracle.py"), "--selftest"]),
    ("V4 Behavioral production/calibration (13/13)",
     [sys.executable, os.path.join(D2D, "run_test_d2d_v4.py")]),
    ("NoC frozen exact 14781/29109, 14833/45441",
     [sys.executable, os.path.join(HERE, "noc_congestion",
                                   "run_test_noc_congestion.py")]),
]


def main() -> int:
    results = []
    for name, command in GATES:
        print(f"\n########## V4 GATE: {name} ##########", flush=True)
        try:
            rc = subprocess.run(command, timeout=300).returncode
        except subprocess.TimeoutExpired:
            rc = 124
            print("gate timed out after 300 seconds", flush=True)
        results.append((name, rc))

    print("\n==================== D2D V4 FREEZE GATE ====================")
    for name, rc in results:
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] {name} (exit={rc})")
    ok = all(rc == 0 for _, rc in results)
    print(f"AGGREGATE EXIT={0 if ok else 1}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
