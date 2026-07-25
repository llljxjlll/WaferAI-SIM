#!/usr/bin/env python3
"""Run DTE V3a async issue/dependency WorkerCore integration tests."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict

from run_test_dte_v1 import BUILD, NPUSIM, dte_spans, run_sim

SIM_ON = "../llm/test/dte/simulation/v3_async_on.json"
SIM_OFF = "../llm/test/dte/simulation/v3_async_off.json"

ASYNC_RE = re.compile(
    r"^(DTE_async_(?:issue|wait|poll|fence|cancel|hazard)) "
    r"core=(\d+) token=(\d+) xfer=(\d+) outstanding=(\d+)"
    r"(?: (.*))?$"
)


def workload(name: str) -> str:
    return f"../llm/test/dte/workload/{name}.json"


def hardware(channels: int) -> str:
    return f"../llm/test/dte/hardware/v2_channel{channels}.json"


def async_events(events: list[dict]) -> list[dict]:
    parsed = []
    for event in events:
        match = ASYNC_RE.match(str(event.get("name", "")))
        if not match or event.get("ph") not in {"B", "E"}:
            continue
        stage, core, token, xfer, outstanding, extra = match.groups()
        parsed.append({
            "stage": stage,
            "core": int(core),
            "token": int(token),
            "xfer": int(xfer),
            "outstanding": int(outstanding),
            "extra": extra or "",
            "phase": event["ph"],
            "ns": round(float(event["ts"]) * 1000),
        })
    return parsed


def span(events: list[dict], name: str) -> tuple[int, int] | None:
    points = [
        (round(float(event["ts"]) * 1000), event["ph"])
        for event in events
        if event.get("name") == name and event.get("ph") in {"B", "E"}
    ]
    begins = [ns for ns, phase in points if phase == "B"]
    ends = [ns for ns, phase in points if phase == "E"]
    if len(begins) != 1 or len(ends) != 1:
        return None
    return begins[0], ends[0]


def transmit_spans(events: list[dict]) -> dict[int, tuple[int, int, int]]:
    grouped: dict[int, dict[str, int]] = defaultdict(dict)
    bits: dict[int, int] = {}
    for event in dte_spans(events):
        if event["stage"] != "DTE_transmit":
            continue
        grouped[event["xfer"]][event["phase"]] = event["ns"]
        bits[event["xfer"]] = event["bits"]
    return {
        xfer: (times["B"], times["E"], bits[xfer])
        for xfer, times in grouped.items()
        if set(times) == {"B", "E"}
    }


def max_active(events: list[dict]) -> int:
    flows: dict[int, dict[str, int]] = defaultdict(dict)
    for event in dte_spans(events):
        if event["stage"] == "DTE_launch" and event["phase"] == "B":
            flows[event["xfer"]]["B"] = event["ns"]
        if event["stage"] == "DTE_transmit" and event["phase"] == "E":
            flows[event["xfer"]]["E"] = event["ns"]
    points = []
    for times in flows.values():
        if set(times) != {"B", "E"}:
            continue
        points.append((times["B"], 1))
        points.append((times["E"], -1))
    active = maximum = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def issue_xfers(events: list[dict], token: int) -> list[int]:
    return [
        event["xfer"] for event in async_events(events)
        if event["stage"] == "DTE_async_issue"
        and event["phase"] == "B" and event["token"] == token
    ]


def main() -> int:
    tests: list[tuple[str, bool, str]] = []

    selftest = subprocess.run(
        [str(NPUSIM), "--dte-v3-selftest"], cwd=BUILD, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
    )
    tests.append((
        "V3a token/dependency SystemC selftest",
        selftest.returncode == 0 and "PASS (31/31 checks)" in selftest.stdout,
        f"exit={selftest.returncode}",
    ))

    blocking = run_sim(SIM_ON, workload("v3_blocking"), hardware(2))
    overlap = run_sim(SIM_ON, workload("v3_overlap"), hardware(2))
    blocking_comp = span(blocking.trace_events, "Matmul_f")
    overlap_comp = span(overlap.trace_events, "Matmul_f")
    blocking_tx = transmit_spans(blocking.trace_events).get(0)
    overlap_tx = transmit_spans(overlap.trace_events).get(0)
    tests.append((
        "dependency wait blocks compute until DMA completion",
        blocking.returncode == 0 and blocking.finish_ns == 421
        and blocking_comp is not None and blocking_tx is not None
        and blocking_comp[0] > blocking_tx[1],
        f"finish={blocking.finish_ns} ns, tx={blocking_tx}, comp={blocking_comp}",
    ))
    tests.append((
        "independent compute overlaps DMA",
        overlap.returncode == 0 and overlap.finish_ns == 345
        and overlap_comp is not None and overlap_tx is not None
        and overlap_comp[0] < overlap_tx[0] < overlap_tx[1] < overlap_comp[1],
        f"finish={overlap.finish_ns} ns, tx={overlap_tx}, comp={overlap_comp}",
    ))
    overlap_async = async_events(overlap.trace_events)
    tests.append((
        "blocking and async schedules execute the same operations",
        blocking.returncode == overlap.returncode == 0
        and blocking_tx is not None and overlap_tx is not None
        and blocking_tx[2] == overlap_tx[2] == 65536
        and blocking_comp is not None and overlap_comp is not None
        and blocking_comp[1] - blocking_comp[0]
            == overlap_comp[1] - overlap_comp[0] == 209
        and issue_xfers(blocking.trace_events, 1) == [0]
        and issue_xfers(overlap.trace_events, 1) == [0],
        "one 65,536-bit DMA plus one 209-ns Matmul in both schedules",
    ))
    tests.append((
        "poll is non-blocking and overlap shortens total time",
        blocking.finish_ns is not None and overlap.finish_ns is not None
        and blocking.finish_ns - overlap.finish_ns == 76
        and any(event["stage"] == "DTE_async_poll"
                and event["phase"] == "E"
                and event["extra"] == "complete=0"
                for event in overlap_async),
        f"blocking/overlap={blocking.finish_ns}/{overlap.finish_ns} ns",
    ))

    selective = run_sim(SIM_ON, workload("v3_selective"), hardware(2))
    selective_async = async_events(selective.trace_events)
    selective_tx = transmit_spans(selective.trace_events)
    selective_comp = span(selective.trace_events, "Matmul_f")
    wait1_end = [
        event["ns"] for event in selective_async
        if event["stage"] == "DTE_async_wait"
        and event["token"] == 1 and event["phase"] == "E"
    ]
    fence_end = [
        event["ns"] for event in selective_async
        if event["stage"] == "DTE_async_fence"
        and event["phase"] == "E"
    ]
    tests.append((
        "selective wait leaves another DMA outstanding",
        selective.returncode == 0 and selective.finish_ns == 375
        and wait1_end == [144] and selective_comp == (152, 361)
        and selective_tx.get(1) == (144, 272, 131072)
        and selective_comp[0] < selective_tx[1][1] < selective_comp[1]
        and any(event["stage"] == "DTE_async_poll"
                and event["token"] == 2 and event["extra"] == "complete=1"
                for event in selective_async)
        and fence_end == [365],
        f"finish={selective.finish_ns} ns, wait1={wait1_end}, fence={fence_end}",
    ))

    scans = {
        channels: run_sim(SIM_ON, workload("v3_four"), hardware(channels))
        for channels in (1, 2, 4)
    }
    scan_max = {channels: max_active(result.trace_events)
                for channels, result in scans.items()}
    scan_finish = {channels: result.finish_ns
                   for channels, result in scans.items()}
    scan_bus = {
        channels: sum(end - begin for begin, end, _
                      in transmit_spans(result.trace_events).values())
        for channels, result in scans.items()
    }
    tests.append((
        "channel=1/2/4 bounds active descriptors",
        all(result.returncode == 0 for result in scans.values())
        and scan_max == {1: 1, 2: 2, 4: 4}
        and scan_finish == {1: 324, 2: 264, 4: 264},
        f"max_active={scan_max}, finish={scan_finish}",
    ))
    tests.append((
        "channel count does not multiply shared bus bandwidth",
        scan_bus == {1: 128, 2: 128, 4: 128},
        f"summed transmit ns={scan_bus}",
    ))

    hazard = run_sim(SIM_ON, workload("v3_hazard"), hardware(2))
    hazard_async = async_events(hazard.trace_events)
    hazard_tx = transmit_spans(hazard.trace_events)
    hazard_begin = [
        event["ns"] for event in hazard_async
        if event["stage"] == "DTE_async_hazard"
        and event["phase"] == "B"
    ]
    hazard_end = [
        event["ns"] for event in hazard_async
        if event["stage"] == "DTE_async_hazard"
        and event["phase"] == "E"
    ]
    tests.append((
        "overlapping RAW range is serialized before second issue",
        hazard.returncode == 0 and hazard.finish_ns == 276
        and hazard_begin and hazard_end and hazard_begin[0] < hazard_end[0]
        and hazard_tx.get(0) is not None and hazard_tx.get(1) is not None
        and hazard_tx[0][1] <= hazard_tx[1][0],
        f"finish={hazard.finish_ns} ns, hazard={hazard_begin}→{hazard_end}",
    ))

    cancel = run_sim(SIM_ON, workload("v3_cancel"), hardware(1))
    cancel_async = async_events(cancel.trace_events)
    tests.append((
        "pending cancel removes one token without leaking the active DMA",
        cancel.returncode == 0 and cancel.finish_ns == 260
        and issue_xfers(cancel.trace_events, 1) == [0]
        and issue_xfers(cancel.trace_events, 2) == [1]
        and any(event["stage"] == "DTE_async_cancel"
                and event["token"] == 2 and event["phase"] == "E"
                and event["outstanding"] == 1
                for event in cancel_async),
        f"finish={cancel.finish_ns} ns",
    ))

    reuse = run_sim(SIM_ON, workload("v3_reuse"), hardware(1))
    refill = run_sim(SIM_ON, workload("v3_refill"), hardware(1))
    tests.append((
        "logical token reuse gets fresh xfer ids in one queue",
        reuse.returncode == 0 and reuse.finish_ns == 160
        and issue_xfers(reuse.trace_events, 5) == [0, 1],
        f"finish={reuse.finish_ns} ns, xfers={issue_xfers(reuse.trace_events, 5)}",
    ))
    tests.append((
        "pipeline refill reuses token only after prior wait",
        refill.returncode == 0 and refill.finish_ns == 218
        and issue_xfers(refill.trace_events, 5) == [0, 1],
        f"finish={refill.finish_ns} ns, xfers={issue_xfers(refill.trace_events, 5)}",
    ))

    unfenced = run_sim(SIM_ON, workload("v3_unfenced"), hardware(1))
    tests.append((
        "SEND_DONE rejects leaked outstanding tokens",
        unfenced.returncode != 0
        and "SEND_DONE reached with outstanding tokens" in unfenced.stdout,
        f"exit={unfenced.returncode}",
    ))
    disabled = run_sim(SIM_OFF, workload("v3_overlap"), hardware(2))
    tests.append((
        "Dte_async primitive requires explicit async mode",
        disabled.returncode != 0
        and "requires dte.async=true" in disabled.stdout,
        f"exit={disabled.returncode}",
    ))

    failures = 0
    for name, passed, detail in tests:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        failures += not passed
    print(f"DTE V3a integration: {len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
