#!/usr/bin/env python3
"""R0 refactor-contract oracle and C++ self-test runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"


def run(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT / "build", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)


def main() -> int:
    tests: list[tuple[str, bool, str]] = []

    oracle = run(["python3", str(HERE / "oracle.py")])
    tests.append(("independent vector/timing oracle",
                  oracle.returncode == 0 and
                  "NoC collective V0/V1/V3 oracle self-test: PASS" in oracle.stdout,
                  oracle.stdout.strip()))

    unit = run([str(NPUSIM), "--coll-r0-selftest"])
    tests.append(("C++ R0 frozen contracts",
                  unit.returncode == 0 and "PASS (19 checks)" in unit.stdout,
                  unit.stdout.strip()))

    for name, passed, detail in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    passed_count = sum(passed for _, passed, _ in tests)
    print(f"NoC collective R0 runner: {passed_count}/{len(tests)} passed")
    return 0 if passed_count == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
