#!/usr/bin/env python3
"""Run the independent oracle and the SystemC DTE V0 self-test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NPUSIM = ROOT / "build" / "npusim"


def validate_sample_configs() -> str:
    hardware = json.loads((HERE / "hardware" / "v0.json").read_text())
    simulation = json.loads((HERE / "simulation" / "v0.json").read_text())

    dte = hardware["dte"]
    core = hardware["cores"][0]
    assert dte["gamma_ns"] == 40000
    assert dte["tau_launch_avg_ns"] == 2000
    channels = core["dte_channel_count"]
    width_bits = core["dte_bit_width"]
    assert channels > 0
    assert width_bits > 0
    assert isinstance(simulation["dte"]["use_beha_dte"], bool)
    return ("loaded hardware/v0.json and simulation/v0.json; "
            f"channels={channels}, width={width_bits} bits")


def run(command: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT / "build", text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout)


def main() -> int:
    tests = []
    try:
        sample_detail = validate_sample_configs()
        tests.append(("sample DTE configs", True, sample_detail))
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
        tests.append(("sample DTE configs", False, str(error)))

    oracle = run([sys.executable, str(HERE / "oracle.py")])
    tests.append(("independent cycle oracle",
                  oracle.returncode == 0 and "oracle self-test: PASS" in oracle.stdout,
                  oracle.stdout.strip()))

    if not NPUSIM.exists():
        tests.append(("SystemC DTE V0 self-test", False,
                      f"npusim not found at {NPUSIM}; build it first"))
    else:
        sim = run([str(NPUSIM), "--dte-v0-selftest"])
        tests.append(("SystemC DTE V0 self-test",
                      sim.returncode == 0 and "DTE V0 self-test: PASS" in sim.stdout,
                      next((line for line in reversed(sim.stdout.splitlines())
                            if "DTE V0 self-test:" in line), sim.stdout[-500:])))

    for name, passed, detail in tests:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
    return 0 if all(item[1] for item in tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
