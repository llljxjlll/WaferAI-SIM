#!/usr/bin/env python3
"""Replay a retained finalized cycle-accurate artifact without compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CASE = ROOT / "notes/frontend/intra_die/reports/2x2_model_v3/A11/run"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", type=Path, default=ROOT / "build-release-final/npusim")
    parser.add_argument("--output", type=Path, default=HERE / "retained_replay.json")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    paths = {
        "program": CASE / "program/program.npup",
        "manifest": CASE / "compile/linked_manifest.json",
        "program_io": CASE / "program/program_io.json",
        "hardware": CASE / "inputs/hardware.json",
        "simulation": CASE / "inputs/simulation.json",
        "mapping": CASE / "inputs/mapping.spec",
    }
    missing = [str(path) for path in (args.npusim, *paths.values()) if not path.is_file()]
    if missing:
        raise SystemExit(f"missing replay inputs: {missing}")
    command = [
        str(args.npusim.resolve()),
        "--program", str(paths["program"].resolve()),
        "--linked-manifest", str(paths["manifest"].resolve()),
        "--program-io", str(paths["program_io"].resolve()),
        "--hardware-config", str(paths["hardware"].resolve()),
        "--simulation-config", str(paths["simulation"].resolve()),
        "--mapping-config", str(paths["mapping"].resolve()),
        "--trace-window", "200000",
    ]
    completed = subprocess.run(
        command,
        cwd=args.npusim.resolve().parent,
        text=True,
        capture_output=True,
        timeout=args.timeout,
    )
    output = ANSI.sub("", completed.stdout + "\n" + completed.stderr)
    done_times = [int(value) for value in re.findall(r"End DONE reception.*?([0-9]+) ns", output)]
    io_passes = re.findall(r"\[PROGRAM_IO\].*?phase=(resolved|applied|verify).*?pass=1", output)
    host = re.search(r"\[HOSTLANE\] done_total=(\d+) ack_total=(\d+) mismatch=(\d+)", output)
    drain = re.search(r"\[DRAIN\] router_residual=(\d+)", output)
    credit = re.search(r"\[CREDIT\] data_balanced=(\d+) ctrl_balanced=(\d+)", output)
    expected = json.loads((CASE / "run_report.json").read_text())["runtime"]
    passed = (
        completed.returncode == 0
        and io_passes == ["resolved", "applied", "verify"]
        and host is not None
        and tuple(map(int, host.groups())) == (expected["done_total"], expected["ack_total"], 0)
        and drain is not None and int(drain.group(1)) == 0
        and credit is not None and tuple(map(int, credit.groups())) == (1, 1)
        and done_times and done_times[-1] == 2 * expected["makespan_cycles"] + 1
    )
    result = {
        "schema_version": "exp1.retained_no_compile_replay/v1",
        "status": "pass" if passed else "fail",
        "method": "retained_finalized_artifact_single_replay_no_compile",
        "expected_makespan_cycles": expected["makespan_cycles"],
        "final_done_timestamp_ns": done_times[-1] if done_times else None,
        "cycle_ns": 2,
        "program_io_phases_passed": io_passes,
        "hostlane": {
            "done_total": int(host.group(1)) if host else None,
            "ack_total": int(host.group(2)) if host else None,
            "mismatch": int(host.group(3)) if host else None,
        },
        "router_residual": int(drain.group(1)) if drain else None,
        "credit_balanced": bool(credit and credit.groups() == ("1", "1")),
        "returncode": completed.returncode,
        "npusim_sha256": digest(args.npusim),
        "input_sha256": {name: digest(path) for name, path in paths.items()},
        "source_case": str(CASE.relative_to(ROOT)),
        "limitation": "replays the retained small A11 hardware/config; validates simulator and marker stability but does not calibrate current exp1 hardware",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
