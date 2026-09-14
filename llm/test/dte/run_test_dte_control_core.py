#!/usr/bin/env python3
"""Small end-to-end gate for selectable DTE controller modes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FINISH_RE = re.compile(r"All requests finished.*?(\d+)\s*ns")
CTRL_STATS_RE = re.compile(
    r"\[DTE_CTRL_STATS\] core=(\d+) mode=(\S+) "
    r"submitted=(\d+) dispatched=(\d+) completed=(\d+).*?"
    r"queued=(\d+) outstanding=(\d+)"
)


def run(
    npusim: Path,
    hardware: Path,
    *,
    workload: Path | None = None,
    simulation: Path | None = None,
) -> tuple[int, list[dict], str]:
    trace = npusim.parent / "events.json"
    trace.unlink(missing_ok=True)
    command = [
        str(npusim),
        "--trace-window", "1000000",
        "--workload-config",
        str(workload or
            ROOT / "llm/test/noc_congestion/workload/gemm_no_congestion.json"),
        "--hardware-config", str(hardware),
        "--simulation-config",
        str(simulation or ROOT / "llm/test/dte/simulation/v1_beha_on.json"),
        "--mapping-config",
        str(ROOT / "llm/test/noc_congestion/mapping/identity.spec"),
    ]
    proc = subprocess.run(
        command,
        cwd=npusim.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    matches = FINISH_RE.findall(proc.stdout)
    if proc.returncode != 0 or not matches:
        raise RuntimeError(
            f"simulation failed for {hardware.name}\n{proc.stdout}"
        )
    events: list[dict] = []
    if trace.exists():
        events = json.loads(trace.read_text()).get("traceEvents", [])
        trace.unlink()
    return int(matches[-1]), events, proc.stdout


def intervals(events: list[dict], prefix: str) -> list[tuple[float, float]]:
    points: dict[str, dict[str, float]] = {}
    for event in events:
        name = str(event.get("name", ""))
        phase = event.get("ph")
        if not name.startswith(prefix) or phase not in {"B", "E"}:
            continue
        points.setdefault(name, {})[phase] = float(event["ts"])
    return [
        (point["B"], point["E"])
        for point in points.values()
        if set(point) == {"B", "E"}
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npusim",
        type=Path,
        default=ROOT / "build" / "npusim",
        help="path to the npusim binary",
    )
    args = parser.parse_args()
    npusim = args.npusim.resolve()
    hardware_dir = ROOT / "llm/test/dte/hardware"

    legacy_ns, legacy_events, _ = run(
        npusim, hardware_dir / "control_core_legacy.json"
    )
    dedicated_ns, dedicated_events, _ = run(
        npusim, hardware_dir / "control_core_dedicated.json"
    )
    legacy_names = {str(event.get("name", "")) for event in legacy_events}
    dedicated_names = {str(event.get("name", "")) for event in dedicated_events}
    if any("DTE_CTRL_" in name for name in legacy_names):
        raise RuntimeError("legacy mode unexpectedly emitted controller trace")
    expected = {"DTE_CTRL_dispatch", "DTE_CTRL_notify"}
    present = {
        stage
        for stage in expected
        if any(stage in name for name in dedicated_names)
    }
    if present != expected:
        raise RuntimeError(
            f"dedicated mode is missing controller traces: {expected - present}"
        )
    if dedicated_ns < legacy_ns:
        raise RuntimeError(
            "positive controller latencies unexpectedly shortened makespan: "
            f"legacy={legacy_ns} ns dedicated={dedicated_ns} ns"
        )
    _, async_events, async_stdout = run(
        npusim,
        hardware_dir / "control_core_dedicated.json",
        workload=ROOT / "llm/test/dte/workload/v3_overlap.json",
        simulation=ROOT / "llm/test/dte/simulation/v3_async_on.json",
    )
    async_names = {str(event.get("name", "")) for event in async_events}
    expected_opcodes = {"issue_token", "poll_token", "wait_token"}
    present_opcodes = {
        opcode
        for opcode in expected_opcodes
        if any(f"opcode={opcode}" in name for name in async_names)
    }
    if present_opcodes != expected_opcodes:
        raise RuntimeError(
            "dedicated async trace is missing controller opcodes: "
            f"{expected_opcodes - present_opcodes}"
        )
    compute = intervals(async_events, "Matmul_f")
    transmit = intervals(async_events, "DTE_transmit")
    if not any(max(comp[0], tx[0]) < min(comp[1], tx[1])
               for comp in compute for tx in transmit):
        raise RuntimeError(
            "dedicated async workload has no Matmul/DTE_transmit overlap"
        )
    stats = [
        tuple(int(value) if index != 1 else value
              for index, value in enumerate(match.groups()))
        for match in CTRL_STATS_RE.finditer(async_stdout)
    ]
    if (not stats or not any(row[2] > 0 for row in stats)
            or any(row[1] != "dual_dte_dedicated"
                   or row[2] != row[4] or row[5] != 0 or row[6] != 0
                   for row in stats)):
        raise RuntimeError(
            "dedicated async controller stats do not drain cleanly: "
            f"{stats}"
        )
    print(
        "DTE control-core integration: PASS "
        f"(legacy={legacy_ns} ns, dedicated={dedicated_ns} ns, "
        "async overlap/opcodes/residual=PASS)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
