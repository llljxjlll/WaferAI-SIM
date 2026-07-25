#!/usr/bin/env python3
"""Run DTE V1 workload integration and configuration-gate tests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = ROOT / "build"
NPUSIM = BUILD / "npusim"
TRACE = BUILD / "events.json"

WORKLOAD = "../llm/test/noc_congestion/workload/gemm_no_congestion.json"
HARDWARE = "../llm/test/dte/hardware/v1.json"
MAPPING = "../llm/test/noc_congestion/mapping/identity.spec"

FINISH_RE = re.compile(r"All requests finished.*?(\d+)\s*ns")
DTE_RE = re.compile(
    r"^(DTE_(?:pending|launch|bus_wait|transmit)) "
    r"xfer=(\d+) core=(\d+) channel=(-?\d+) "
    r"dir=(SPM_TO_REMOTE|REMOTE_TO_SPM) bits=(\d+)$"
)


@dataclass
class RunResult:
    returncode: int
    finish_ns: int | None
    stdout: str
    trace_events: list[dict]


def run_sim(
    simulation: str,
    workload: str = WORKLOAD,
    hardware: str = HARDWARE,
    timeout: int = 60,
) -> RunResult:
    TRACE.unlink(missing_ok=True)
    command = [
        str(NPUSIM),
        "--trace-window", "1000000",
        "--workload-config", workload,
        "--hardware-config", hardware,
        "--simulation-config", simulation,
        "--mapping-config", MAPPING,
    ]
    proc = subprocess.run(
        command,
        cwd=BUILD,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    matches = FINISH_RE.findall(proc.stdout)
    finish_ns = int(matches[-1]) if matches else None
    trace_events: list[dict] = []
    if TRACE.exists():
        trace_events = json.loads(TRACE.read_text())["traceEvents"]
        TRACE.unlink()
    return RunResult(proc.returncode, finish_ns, proc.stdout, trace_events)


def dte_spans(events: list[dict]) -> list[dict]:
    parsed = []
    for event in events:
        match = DTE_RE.match(event.get("name", ""))
        if not match or event.get("ph") not in {"B", "E"}:
            continue
        stage, xfer, core, channel, direction, bits = match.groups()
        parsed.append({
            "module": event["cat"],
            "stage": stage,
            "xfer": int(xfer),
            "core": int(core),
            "channel": int(channel),
            "direction": direction,
            "bits": int(bits),
            "phase": event["ph"],
            "ns": round(float(event["ts"]) * 1000),
        })
    return parsed


def validate_on_trace(events: list[dict]) -> tuple[bool, str, set[int]]:
    spans = dte_spans(events)
    if len(spans) != 128:
        return False, f"expected 128 DTE B/E events, got {len(spans)}", set()

    grouped: dict[tuple[str, int], list[dict]] = {}
    for event in spans:
        grouped.setdefault((event["module"], event["xfer"]), []).append(event)
    if len(grouped) != 16:
        return False, f"expected 16 per-core transfers, got {len(grouped)}", set()

    expected_stages = {
        ("DTE_pending", "B"), ("DTE_pending", "E"),
        ("DTE_launch", "B"), ("DTE_launch", "E"),
        ("DTE_bus_wait", "B"), ("DTE_bus_wait", "E"),
        ("DTE_transmit", "B"), ("DTE_transmit", "E"),
    }
    direction_counts = {"SPM_TO_REMOTE": 0, "REMOTE_TO_SPM": 0}
    payloads: set[int] = set()
    source_transmits = []
    source_done = []
    destination_issue = []

    for key, flow in grouped.items():
        fields = {(event["stage"], event["phase"]) for event in flow}
        if fields != expected_stages:
            return False, f"{key} has incomplete DTE stages: {fields}", set()
        directions = {event["direction"] for event in flow}
        bits = {event["bits"] for event in flow}
        cores = {event["core"] for event in flow}
        if len(directions) != 1 or len(bits) != 1 or len(cores) != 1:
            return False, f"{key} changes direction/payload/core", set()

        direction = next(iter(directions))
        payload = next(iter(bits))
        direction_counts[direction] += 1
        payloads.add(payload)
        times = {
            (event["stage"], event["phase"]): event["ns"] for event in flow
        }
        order = [
            times[("DTE_pending", "B")],
            times[("DTE_pending", "E")],
            times[("DTE_launch", "B")],
            times[("DTE_launch", "E")],
            times[("DTE_bus_wait", "B")],
            times[("DTE_bus_wait", "E")],
            times[("DTE_transmit", "B")],
            times[("DTE_transmit", "E")],
        ]
        if order != sorted(order):
            return False, f"{key} has non-monotonic stage order: {order}", set()
        if order[-1] - order[0] != 2054:
            return False, f"{key} total DTE latency is {order[-1] - order[0]} ns", set()
        if (times[("DTE_launch", "E")] -
                times[("DTE_launch", "B")]) != 6:
            return False, f"{key} launch latency is not 6 ns", set()
        if (times[("DTE_transmit", "E")] -
                times[("DTE_transmit", "B")]) != 2048:
            return False, f"{key} transmit latency is not 2048 ns", set()

        if direction == "SPM_TO_REMOTE":
            source_transmits.append((
                times[("DTE_transmit", "B")],
                times[("DTE_transmit", "E")],
            ))
            source_done.append(times[("DTE_transmit", "E")])
        else:
            destination_issue.append(times[("DTE_pending", "B")])

    if direction_counts != {"SPM_TO_REMOTE": 8, "REMOTE_TO_SPM": 8}:
        return False, f"unexpected direction counts: {direction_counts}", set()
    if payloads != {2097152}:
        return False, f"unexpected logical payloads: {payloads}", set()
    if max(source_done) > min(destination_issue):
        return False, "destination DTE begins before all source DTEs complete", set()

    # Separate per-core buses must overlap; a global DTE would serialize these spans.
    first = source_transmits[0]
    if not any(start < first[1] and first[0] < end
               for start, end in source_transmits[1:]):
        return False, "per-core source DTE transmit spans did not overlap", set()

    return True, "16 transfers, exact 2,097,152-bit payload and 2054 ns/DTE", payloads


def validate_cross_die_trace(events: list[dict]) -> tuple[bool, str]:
    spans = dte_spans(events)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for event in spans:
        grouped.setdefault((event["module"], event["xfer"]), []).append(event)
    if len(spans) != 32 or len(grouped) != 4:
        return False, f"expected 4 transfers/32 events, got {len(grouped)}/{len(spans)}"

    cores = {event["core"] for event in spans}
    directions = {event["direction"] for event in spans}
    payloads = {event["bits"] for event in spans}
    if cores != {5, 7, 8, 24}:
        return False, f"unexpected DTE cores: {cores}"
    if directions != {"SPM_TO_REMOTE", "REMOTE_TO_SPM"}:
        return False, f"missing DTE direction: {directions}"
    if payloads != {16384}:
        return False, f"unexpected cross-die payload: {payloads}"

    for key, flow in grouped.items():
        times = {
            (event["stage"], event["phase"]): event["ns"] for event in flow
        }
        if (times[("DTE_transmit", "E")] -
                times[("DTE_pending", "B")]) != 22:
            return False, f"{key} does not have expected 22 ns DTE latency"
    return True, "stripe=4, cores 8→24 cross die and 5→7 local, 16,384 bits"


def validate_multi_source_trace(events: list[dict]) -> tuple[bool, str]:
    starts = [
        event for event in dte_spans(events)
        if event["stage"] == "DTE_pending" and event["phase"] == "B"
    ]
    actual = {
        (event["core"], event["direction"], event["bits"])
        for event in starts
    }
    expected = {
        (0, "SPM_TO_REMOTE", 16384),
        (1, "SPM_TO_REMOTE", 16384),
        (2, "REMOTE_TO_SPM", 32768),
    }
    if actual != expected:
        return False, f"unexpected multi-source DTE descriptors: {actual}"
    return True, "two 16,384-bit sources aggregate into one 32,768-bit destination"


def validate_heterogeneous_trace(events: list[dict]) -> tuple[bool, str]:
    spans = dte_spans(events)
    grouped: dict[tuple[str, int], list[dict]] = {}
    for event in spans:
        grouped.setdefault((event["module"], event["xfer"]), []).append(event)
    if len(grouped) != 3:
        return False, f"expected 3 heterogeneous transfers, got {len(grouped)}"

    actual: dict[tuple[int, str, int], int] = {}
    for flow in grouped.values():
        descriptor = (flow[0]["core"], flow[0]["direction"], flow[0]["bits"])
        times = {(event["stage"], event["phase"]): event["ns"] for event in flow}
        actual[descriptor] = (
            times[("DTE_transmit", "E")] - times[("DTE_pending", "B")]
        )

    expected = {
        (0, "SPM_TO_REMOTE", 16384): 22,
        (1, "SPM_TO_REMOTE", 16384): 38,
        (2, "REMOTE_TO_SPM", 32768): 22,
    }
    if actual != expected:
        return False, f"unexpected heterogeneous DTE latencies: {actual}"
    return True, "core widths 2048/1024/4096 bits produce 22/38/22 ns"


def validate_repeated_flow_trace(events: list[dict]) -> tuple[bool, str]:
    starts = [
        event for event in dte_spans(events)
        if event["stage"] == "DTE_pending" and event["phase"] == "B"
    ]
    expected = {
        (0, "SPM_TO_REMOTE", 16384): 2,
        (1, "SPM_TO_REMOTE", 16384): 2,
        (2, "REMOTE_TO_SPM", 32768): 2,
    }
    actual: dict[tuple[int, str, int], int] = {}
    for event in starts:
        descriptor = (event["core"], event["direction"], event["bits"])
        actual[descriptor] = actual.get(descriptor, 0) + 1
    if actual != expected:
        return False, f"unexpected repeated-flow descriptors: {actual}"

    destination_starts = sorted(
        event["ns"] for event in starts
        if event["direction"] == "REMOTE_TO_SPM"
    )
    if len(destination_starts) != 2 or destination_starts[0] >= destination_starts[1]:
        return False, f"destination iterations are not ordered: {destination_starts}"
    return True, "same (source,tag) reused for two completed 2-source iterations"


def main() -> int:
    if not NPUSIM.exists():
        print(f"[FAIL] npusim not found at {NPUSIM}; build it first")
        return 1

    tests: list[tuple[str, bool, str]] = []

    cycle_off = run_sim("../llm/test/dte/simulation/v1_cycle_off.json")
    tests.append((
        "DTE off + physical NoC",
        cycle_off.returncode == 0
        and cycle_off.finish_ns == 29109
        and not dte_spans(cycle_off.trace_events),
        f"exit={cycle_off.returncode}, finish={cycle_off.finish_ns} ns",
    ))

    beha_off = run_sim("../llm/test/dte/simulation/v1_beha_off.json")
    tests.append((
        "DTE off + behavioral NoC",
        beha_off.returncode == 0
        and beha_off.finish_ns == 14781
        and not dte_spans(beha_off.trace_events),
        f"exit={beha_off.returncode}, finish={beha_off.finish_ns} ns",
    ))

    cycle_on = run_sim("../llm/test/dte/simulation/v1_cycle_on.json")
    cycle_trace_ok, cycle_detail, cycle_payloads = validate_on_trace(
        cycle_on.trace_events
    )
    tests.append((
        "DTE on + physical NoC",
        cycle_on.returncode == 0
        and cycle_on.finish_ns == 33217
        and cycle_trace_ok,
        f"finish={cycle_on.finish_ns} ns; {cycle_detail}",
    ))

    beha_on = run_sim("../llm/test/dte/simulation/v1_beha_on.json")
    beha_trace_ok, beha_detail, beha_payloads = validate_on_trace(
        beha_on.trace_events
    )
    tests.append((
        "DTE on + behavioral NoC",
        beha_on.returncode == 0
        and beha_on.finish_ns == 18889
        and beha_trace_ok
        and beha_payloads == cycle_payloads,
        f"finish={beha_on.finish_ns} ns; {beha_detail}",
    ))

    tests.append((
        "exact store-and-forward delta",
        cycle_on.finish_ns is not None
        and beha_on.finish_ns is not None
        and cycle_on.finish_ns - 29109 == 4108
        and beha_on.finish_ns - 14781 == 4108,
        "source 2054 ns + destination 2054 ns = 4108 ns",
    ))

    cross_workload = "../llm/test/dte/workload/v1_cross_die_stripe4.json"
    cross_hardware = "../llm/test/dte/hardware/v1_cross_die.json"
    cross_off = run_sim(
        "../llm/test/dte/simulation/v1_cycle_off.json",
        cross_workload,
        cross_hardware,
    )
    tests.append((
        "cross-die stripe=4 DTE off",
        cross_off.returncode == 0
        and cross_off.finish_ns == 652
        and not dte_spans(cross_off.trace_events),
        f"exit={cross_off.returncode}, finish={cross_off.finish_ns} ns",
    ))

    cross_on = run_sim(
        "../llm/test/dte/simulation/v1_cycle_on.json",
        cross_workload,
        cross_hardware,
    )
    cross_trace_ok, cross_detail = validate_cross_die_trace(
        cross_on.trace_events
    )
    d2d_used = (
        "[D2D_TYPE] request_in=4 request_out=4 ack_in=4 ack_out=4 "
        "data_in=32 data_out=32"
    ) in cross_on.stdout
    tests.append((
        "cross-die stripe=4 DTE on",
        cross_on.returncode == 0
        and cross_on.finish_ns == 696
        and cross_trace_ok
        and d2d_used,
        f"finish={cross_on.finish_ns} ns; {cross_detail}; D2D used={d2d_used}",
    ))
    tests.append((
        "cross-die exact DTE delta",
        cross_on.finish_ns is not None
        and cross_off.finish_ns is not None
        and cross_on.finish_ns - cross_off.finish_ns == 44,
        "source 22 ns + destination 22 ns = 44 ns",
    ))

    stripe2_workload = "../llm/test/dte/workload/v1_cross_die_stripe2.json"
    stripe2_off = run_sim(
        "../llm/test/dte/simulation/v1_cycle_off.json",
        stripe2_workload,
        cross_hardware,
    )
    stripe2_on = run_sim(
        "../llm/test/dte/simulation/v1_cycle_on.json",
        stripe2_workload,
        cross_hardware,
    )
    stripe2_trace_ok, _ = validate_cross_die_trace(stripe2_on.trace_events)
    stripe2_d2d = (
        "[D2D_TYPE] request_in=2 request_out=2 ack_in=2 ack_out=2 "
        "data_in=32 data_out=32"
    ) in stripe2_on.stdout
    tests.append((
        "cross-die stripe=2 exact payload",
        stripe2_off.returncode == 0
        and stripe2_on.returncode == 0
        and stripe2_off.finish_ns == 656
        and stripe2_on.finish_ns == 700
        and stripe2_on.finish_ns - stripe2_off.finish_ns == 44
        and stripe2_trace_ok
        and stripe2_d2d,
        f"off/on={stripe2_off.finish_ns}/{stripe2_on.finish_ns} ns; "
        "same 16,384-bit payload and exact 44 ns DTE delta",
    ))

    multi_workload = "../llm/test/dte/workload/v1_multi_source.json"
    multi_off = run_sim(
        "../llm/test/dte/simulation/v1_cycle_off.json",
        multi_workload,
    )
    multi_on = run_sim(
        "../llm/test/dte/simulation/v1_cycle_on.json",
        multi_workload,
    )
    multi_trace_ok, multi_detail = validate_multi_source_trace(
        multi_on.trace_events
    )
    tests.append((
        "multi-source destination aggregation",
        multi_off.returncode == 0
        and multi_on.returncode == 0
        and multi_off.finish_ns == 623
        and multi_on.finish_ns == 683
        and multi_on.finish_ns - multi_off.finish_ns == 60
        and multi_trace_ok,
        f"off/on={multi_off.finish_ns}/{multi_on.finish_ns} ns; {multi_detail}",
    ))

    repeated_workload = "../llm/test/dte/workload/v1_repeated_flow.json"
    repeated_off = run_sim(
        "../llm/test/dte/simulation/v1_cycle_off.json",
        repeated_workload,
    )
    repeated_on = run_sim(
        "../llm/test/dte/simulation/v1_cycle_on.json",
        repeated_workload,
    )
    repeated_ok, repeated_detail = validate_repeated_flow_trace(
        repeated_on.trace_events
    )
    tests.append((
        "repeated source/tag lifecycle across pipeline iterations",
        repeated_off.returncode == 0
        and repeated_on.returncode == 0
        and repeated_off.finish_ns == 1033
        and repeated_on.finish_ns == 1115
        and not dte_spans(repeated_off.trace_events)
        and repeated_ok,
        f"off/on={repeated_off.finish_ns}/{repeated_on.finish_ns} ns; "
        f"{repeated_detail}",
    ))

    heterogeneous_on = run_sim(
        "../llm/test/dte/simulation/v1_cycle_on.json",
        multi_workload,
        "../llm/test/dte/hardware/v1_heterogeneous.json",
    )
    heterogeneous_ok, heterogeneous_detail = validate_heterogeneous_trace(
        heterogeneous_on.trace_events
    )
    tests.append((
        "per-core heterogeneous DTE configuration",
        heterogeneous_on.returncode == 0 and heterogeneous_ok,
        heterogeneous_detail,
    ))

    parallel = run_sim(
        "../llm/test/dte/simulation/v2_parallel_default.json"
    )
    parallel_trace_ok, parallel_detail, _ = validate_on_trace(
        parallel.trace_events
    )
    tests.append((
        "DTE parallel dataflow compatibility after V2a",
        parallel.returncode == 0
        and parallel.finish_ns == 33189
        and parallel_trace_ok,
        f"exit={parallel.returncode}, finish={parallel.finish_ns} ns; "
        f"{parallel_detail}",
    ))

    for name, passed, detail in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return 0 if all(passed for _, passed, _ in tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
