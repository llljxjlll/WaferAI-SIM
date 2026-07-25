#!/usr/bin/env python3
"""Run DTE V3b aggregation/coalescing integration tests."""

from __future__ import annotations

import math
import re
import subprocess
import sys

from oracle import aggregation_oracle
from run_test_dte_v1 import BUILD, NPUSIM, dte_spans, run_sim
from run_test_dte_v3 import async_events, span, transmit_spans

HARDWARE = "../llm/test/dte/hardware/v3b.json"

FLUSH_RE = re.compile(
    r"^DTE_coalesce_flush core=(\d+) token=(\d+) xfer=(\d+) "
    r"outstanding=(\d+) group=(\d+|standalone) members=(\d+) "
    r"bits=(\d+) reason=(\w+) saved=(\d+) cumulative_utilization_ppm=(\d+)$"
)
BIND_RE = re.compile(
    r"^DTE_coalesce_bind core=(\d+) token=(\d+) xfer=(\d+) "
    r"outstanding=(\d+) group=(\d+) head=(\d+) members=(\d+)$"
)


def simulation(name: str) -> str:
    return f"../llm/test/dte/simulation/{name}.json"


def workload(name: str) -> str:
    return f"../llm/test/dte/workload/{name}.json"


def flushes(events: list[dict]) -> list[dict]:
    result = []
    for event in events:
        match = FLUSH_RE.match(str(event.get("name", "")))
        if not match or event.get("ph") != "E":
            continue
        (core, token, xfer, outstanding, group, members, bits,
         reason, saved, cumulative_utilization) = match.groups()
        result.append({
            "core": int(core), "token": int(token), "xfer": int(xfer),
            "outstanding": int(outstanding), "group": group,
            "members": int(members), "bits": int(bits), "reason": reason,
            "saved": int(saved),
            "cumulative_utilization_ppm": int(cumulative_utilization),
            "ns": round(float(event["ts"]) * 1000),
        })
    return result


def bindings(events: list[dict]) -> list[tuple[int, int, int]]:
    result = []
    for event in events:
        match = BIND_RE.match(str(event.get("name", "")))
        if match and event.get("ph") == "E":
            _, token, xfer, _, _, _, members = match.groups()
            result.append((int(token), int(xfer), int(members)))
    return result


def physical_starts(events: list[dict]) -> list[dict]:
    return [event for event in dte_spans(events)
            if event["stage"] == "DTE_pending" and event["phase"] == "B"]


def logical_issue_begin(events: list[dict]) -> int | None:
    times = [event["ns"] for event in async_events(events)
             if event["stage"] == "DTE_async_issue"
             and event["phase"] == "B"]
    return min(times) if times else None


def last_transmit_end(events: list[dict]) -> int | None:
    times = [event["ns"] for event in dte_spans(events)
             if event["stage"] == "DTE_transmit"
             and event["phase"] == "E"]
    return max(times) if times else None


def utilization_ppm(events: list[dict], width_bits: int = 128) -> int:
    starts = physical_starts(events)
    useful = sum(event["bits"] for event in starts)
    capacity = sum(math.ceil(event["bits"] / width_bits) * width_bits
                   for event in starts)
    return round(useful * 1_000_000 / capacity) if capacity else 0


def main() -> int:
    tests: list[tuple[str, bool, str]] = []

    selftest = subprocess.run(
        [str(NPUSIM), "--dte-v3b-selftest"], cwd=BUILD, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
    )
    tests.append((
        "V3b aggregation SystemC selftest",
        selftest.returncode == 0 and "PASS (21/21 checks)" in selftest.stdout,
        f"exit={selftest.returncode}",
    ))

    labels = ["off", "group2", "group4", "group8", "group16"]
    group_sizes = [1, 2, 4, 8, 16]
    simulation_names = {
        "off": "v3b_agg_off", "group2": "v3b_group2",
        "group4": "v3b_group4", "group8": "v3b_group8",
        "group16": "v3b_group16",
    }
    scan = {
        label: run_sim(simulation(simulation_names[label]),
                       workload("v3b_small16"), HARDWARE)
        for label in labels
    }
    oracle = {
        label: aggregation_oracle(
            logical_descriptors=16, payload_bits=64,
            max_descriptors=group, max_payload_bytes=4096,
            bit_width_bits=128, launch_cycles=10,
        )
        for label, group in zip(labels, group_sizes)
    }
    physical = {label: len(physical_starts(result.trace_events))
                for label, result in scan.items()}
    elapsed = {
        label: last_transmit_end(result.trace_events)
               - logical_issue_begin(result.trace_events)
        for label, result in scan.items()
    }
    finish = {label: result.finish_ns for label, result in scan.items()}
    tests.append((
        "COMET 1/2/4/8/16 grouping reduces physical launches",
        all(result.returncode == 0 for result in scan.values())
        and physical == {"off": 16, "group2": 8, "group4": 4,
                         "group8": 2, "group16": 1},
        f"physical={physical}",
    ))
    tests.append((
        "compound completion matches the independent one-channel oracle",
        elapsed == {
            label: item.completion_cycle_from_first_issue * 2
            for label, item in oracle.items()
        } and finish == {"off": 548, "group2": 374, "group4": 298,
                         "group8": 266, "group16": 262},
        f"elapsed={elapsed}, finish={finish}",
    ))
    saved = {label: sum(event["saved"]
                        for event in flushes(result.trace_events))
             for label, result in scan.items()}
    tests.append((
        "launch savings equal logical descriptors minus compound transfers",
        saved == {"off": 0, "group2": 8, "group4": 12,
                  "group8": 14, "group16": 15},
        f"saved={saved}",
    ))
    utilization = {label: utilization_ppm(result.trace_events)
                   for label, result in scan.items()}
    trace_cumulative_utilization = {
        label: flushes(scan[label].trace_events)[-1][
            "cumulative_utilization_ppm"]
        for label in labels if label != "off"
    }
    group16_bindings = bindings(scan["group16"].trace_events)
    tests.append((
        "trace binds every logical token to its compound physical xfer",
        group16_bindings == [(token, 0, 16) for token in range(1, 17)],
        f"bindings={group16_bindings}",
    ))
    tests.append((
        "useful-bit bus utilization matches the independent oracle",
        utilization == {
            label: item.bandwidth_utilization_ppm
            for label, item in oracle.items()
        } == {"off": 500000, "group2": 1000000,
              "group4": 1000000, "group8": 1000000,
              "group16": 1000000}
        and trace_cumulative_utilization == {
            "group2": 1000000, "group4": 1000000,
            "group8": 1000000, "group16": 1000000},
        f"utilization_ppm={utilization}, "
        f"trace_cumulative={trace_cumulative_utilization}",
    ))
    tests.append((
        "fixed-data microbenchmark reproduces COMET launch-amortization trend",
        list(finish.values()) == sorted(finish.values(), reverse=True)
        and finish["off"] - finish["group16"] == 286,
        f"finish={finish}",
    ))

    fanout = run_sim(simulation("v3b_group2"),
                     workload("v3b_fanout"), HARDWARE)
    fanout_async = async_events(fanout.trace_events)
    wait_xfers = [event["xfer"] for event in fanout_async
                  if event["stage"] == "DTE_async_wait"
                  and event["phase"] == "E"]
    tests.append((
        "compound completion fans out once to every logical token",
        fanout.returncode == 0 and fanout.finish_ns == 148
        and len(physical_starts(fanout.trace_events)) == 1
        and wait_xfers == [0, 0]
        and any(event["stage"] == "DTE_async_poll"
                and event["token"] == 2 and event["extra"] == "complete=1"
                for event in fanout_async),
        f"finish={fanout.finish_ns}, wait_xfers={wait_xfers}",
    ))

    noncontig = run_sim(simulation("v3b_group16"),
                        workload("v3b_noncontiguous"), HARDWARE)
    non_flush = flushes(noncontig.trace_events)
    tests.append((
        "non-contiguous local/remote ranges split compound descriptors",
        noncontig.returncode == 0 and noncontig.finish_ns == 172
        and [(event["members"], event["reason"]) for event in non_flush]
            == [(2, "incompatible"), (2, "fence")],
        f"flush={[(e['members'], e['reason']) for e in non_flush]}",
    ))

    incompatible = run_sim(simulation("v3b_group16"),
                           workload("v3b_incompatible"), HARDWARE)
    inc_flush = flushes(incompatible.trace_events)
    tests.append((
        "direction, peer and address block never cross-coalesce",
        incompatible.returncode == 0 and incompatible.finish_ns == 220
        and len(physical_starts(incompatible.trace_events)) == 4
        and all(event["members"] == 1 for event in inc_flush),
        f"physical={len(physical_starts(incompatible.trace_events))}",
    ))

    timeout = run_sim(simulation("v3b_timeout"),
                      workload("v3b_timeout"), HARDWARE)
    timeout_flush = flushes(timeout.trace_events)
    timeout_issue = logical_issue_begin(timeout.trace_events)
    timeout_comp = span(timeout.trace_events, "Matmul_f")
    timeout_tx = transmit_spans(timeout.trace_events).get(0)
    tests.append((
        "partial group flushes at timeout while unrelated compute continues",
        timeout.returncode == 0 and timeout.finish_ns == 341
        and len(timeout_flush) == 1 and timeout_flush[0]["reason"] == "timeout"
        and timeout_issue is not None
        and timeout_flush[0]["ns"] - timeout_issue == 20
        and timeout_comp is not None and timeout_tx is not None
        and timeout_comp[0] < timeout_tx[0] < timeout_tx[1] < timeout_comp[1],
        f"flush={timeout_flush}, comp={timeout_comp}, tx={timeout_tx}",
    ))

    maxbytes = run_sim(simulation("v3b_max16bytes"),
                       workload("v3b_maxbytes"), HARDWARE)
    max_flush = flushes(maxbytes.trace_events)
    tests.append((
        "max byte limit issues a full compound descriptor immediately",
        maxbytes.returncode == 0 and maxbytes.finish_ns == 136
        and len(max_flush) == 1 and max_flush[0]["members"] == 2
        and max_flush[0]["bits"] == 128
        and max_flush[0]["reason"] == "limit",
        f"flush={max_flush}",
    ))

    large_on = run_sim(simulation("v3b_group16"),
                       workload("v3b_large"), HARDWARE)
    large_off = run_sim(simulation("v3b_agg_off"),
                        workload("v3b_large"), HARDWARE)
    tests.append((
        "single large request receives no artificial aggregation benefit",
        large_on.returncode == large_off.returncode == 0
        and large_on.finish_ns == large_off.finish_ns == 638
        and transmit_spans(large_on.trace_events).get(0)
            == transmit_spans(large_off.trace_events).get(0),
        f"on/off={large_on.finish_ns}/{large_off.finish_ns}",
    ))

    v3a = run_sim(simulation("v3b_agg_off"),
                  workload("v3_overlap"),
                  "../llm/test/dte/hardware/v2_channel2.json")
    tests.append((
        "aggregation disabled preserves the frozen V3a workload timing",
        v3a.returncode == 0 and v3a.finish_ns == 345
        and len(physical_starts(v3a.trace_events)) == 1,
        f"finish={v3a.finish_ns}",
    ))

    invalid_async = run_sim(simulation("v3b_invalid_without_async"),
                            workload("v3b_small16"), HARDWARE)
    invalid_size = run_sim(simulation("v3b_invalid_group1"),
                           workload("v3b_small16"), HARDWARE)
    invalid_block = run_sim(simulation("v3b_group16"),
                            workload("v3b_invalid_block"), HARDWARE)
    tests.append((
        "aggregation requires async mode and a valid group limit",
        invalid_async.returncode != 0
        and "aggregation requires dte.async=true" in invalid_async.stdout
        and invalid_size.returncode != 0
        and "aggregation_max_descriptors must be >= 2" in invalid_size.stdout,
        f"exit={invalid_async.returncode}/{invalid_size.returncode}",
    ))
    tests.append((
        "declared address block must contain the full remote range",
        invalid_block.returncode != 0
        and "must fit in its declared address_block" in invalid_block.stdout,
        f"exit={invalid_block.returncode}",
    ))

    failures = 0
    for name, passed, detail in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        failures += not passed
    print(f"DTE V3b integration: {len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
