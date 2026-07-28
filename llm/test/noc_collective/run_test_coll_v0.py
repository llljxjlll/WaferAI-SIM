#!/usr/bin/env python3
"""Run the independent oracle and C++ NoC collective V0 self-test."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"


def run(command: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=timeout)


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    oracle = run([sys.executable, str(HERE / "oracle.py")])
    tests.append(("independent oracle", oracle.returncode == 0 and
                  "oracle self-test: PASS" in oracle.stdout, oracle.stdout.strip()))
    if not NPUSIM.exists():
        tests.append(("C++ contract self-test", False, f"missing {NPUSIM}"))
    else:
        result = run([str(NPUSIM), "--coll-v0-selftest"])
        summary = next((line for line in reversed(result.stdout.splitlines())
                        if "NoC collective V0 self-test:" in line), result.stdout[-500:])
        tests.append(("C++ contract self-test", result.returncode == 0 and
                      "NoC collective V0 self-test: PASS" in result.stdout, summary))
    for name, passed, detail in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return 0 if all(passed for _, passed, _ in tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
