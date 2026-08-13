#!/usr/bin/env python3
"""Freeze the legacy Tier2 32/40/64-chunk progress boundary for R0."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import run_experiment_tiers as tiers


def run(bits: int, directory: Path) -> subprocess.CompletedProcess[str]:
    work_path = directory / f"work_{bits}.json"
    hardware_path = directory / "hardware.json"
    simulation_path = directory / "tier2.json"
    work_path.write_text(json.dumps(tiers.workload(bits)))
    hardware_path.write_text(json.dumps(tiers.hardware()))
    simulation_path.write_text(json.dumps(tiers.simulation(2)))
    return subprocess.run(
        [
            str(tiers.NPUSIM),
            "--workload-config", str(work_path),
            "--hardware-config", str(hardware_path),
            "--simulation-config", str(simulation_path),
            "--mapping-config",
            "../llm/test/noc_collective/mapping/identity.spec",
        ],
        cwd=tiers.ROOT / "build", text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60,
    )


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="coll_legacy_pressure_") as td:
        directory = Path(td)
        completed = run(4096, directory)
        try:
            result = tiers.parse_result(4096, 2, completed.stdout)
            time_ns = result.time_ns
        except AssertionError:
            time_ns = -1
        drained = (
            "[DRAIN] router_residual=0" in completed.stdout
            and "data_balanced=1 ctrl_balanced=1" in completed.stdout
            and "[COLL_DRAIN] tree_entries=0 reduce_nodes=0 barriers=0 "
                "gather=0 reduce_rx=0 endpoints=0 dte_tokens=0"
                in completed.stdout
        )
        tests.append(("legacy Tier2 32 chunks completes",
                      completed.returncode == 0 and time_ns == 6330 and drained,
                      f"exit={completed.returncode} time={time_ns}ns drain={drained}"))

        for bits, chunks in ((5120, 40), (8192, 64)):
            stalled = run(bits, directory)
            watchdog = "[PROTO_WAIT]" in stalled.stdout
            finished = "Catch test finished" in stalled.stdout
            tests.append((f"legacy Tier2 {chunks} chunks watchdog boundary",
                          stalled.returncode != 0 and watchdog and not finished,
                          f"exit={stalled.returncode} watchdog={watchdog} "
                          f"finished={finished}"))

    for name, passed, detail in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    passed_count = sum(passed for _, passed, _ in tests)
    print(f"NoC collective legacy pressure: {passed_count}/{len(tests)} passed")
    return 0 if passed_count == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
