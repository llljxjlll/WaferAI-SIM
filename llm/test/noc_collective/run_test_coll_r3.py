#!/usr/bin/env python3
"""R3 isolated stream-wire, binary-stage, and finite-state runner."""

from __future__ import annotations

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"


def run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=ROOT / "build", text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )


def main() -> int:
    tests: list[tuple[str, bool, str]] = []

    oracle = run(["python3", str(HERE / "r3_oracle.py")])
    tests.append((
        "independent stage/geometry/framing oracle",
        oracle.returncode == 0 and
        "NoC collective R3 oracle self-test: PASS" in oracle.stdout,
        oracle.stdout.strip(),
    ))

    unit = run([str(NPUSIM), "--coll-r3-selftest"])
    tests.append((
        "stream wire and finite backpressure state",
        unit.returncode == 0 and
        "R3 self-test: 39/39 checks passed" in unit.stdout,
        "39 checks",
    ))

    for name, passed, detail in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    passed_count = sum(passed for _, passed, _ in tests)
    print(f"NoC collective R3 runner: {passed_count}/{len(tests)} passed")
    return 0 if passed_count == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
