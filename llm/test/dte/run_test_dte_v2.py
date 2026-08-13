#!/usr/bin/env python3
"""Run DTE V2a parallel-SEND integration and lifecycle tests."""

from __future__ import annotations

from collections import Counter, defaultdict

from run_test_dte_v1 import dte_spans, run_sim


SIM_ON = "../llm/test/dte/simulation/v2_parallel_on.json"
SIM_OFF = "../llm/test/dte/simulation/v2_parallel_off.json"

WORKLOADS = {
    1: "../llm/test/dte/workload/v2_parallel_one.json",
    2: "../llm/test/dte/workload/v2_parallel_two.json",
    4: "../llm/test/dte/workload/v2_parallel_four.json",
}

EXPECTED_FINISH_NS = {1: 641, 2: 903, 4: 1371}
EXPECTED_SOURCE_COMPLETION_NS = {
    (1, 1): [36],
    (1, 2): [36],
    (1, 4): [36],
    (2, 1): [36, 72],
    (2, 2): [36, 52],
    (2, 4): [36, 52],
    (4, 1): [36, 72, 108, 144],
    (4, 2): [36, 52, 72, 88],
    (4, 4): [36, 52, 68, 84],
}


def hardware(channel_count: int) -> str:
    return f"../llm/test/dte/hardware/v2_channel{channel_count}.json"


def grouped_spans(events: list[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for event in dte_spans(events):
        grouped[(event["module"], event["xfer"])].append(event)
    return grouped


def times(flow: list[dict]) -> dict[tuple[str, str], int]:
    return {
        (event["stage"], event["phase"]): event["ns"]
        for event in flow
    }


def max_overlap(intervals: list[tuple[int, int]]) -> int:
    # Half-open intervals: a completion frees its channel before an admission
    # at the same timestamp consumes it.
    points = []
    for start, end in intervals:
        points.append((start, 1))
        points.append((end, -1))
    active = 0
    maximum = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def validate_equal_batch(
    events: list[dict], send_count: int, channel_count: int
) -> tuple[bool, str]:
    spans = dte_spans(events)
    grouped = grouped_spans(events)
    if len(spans) != send_count * 16 or len(grouped) != send_count * 2:
        return False, (
            f"expected {send_count * 2} transfers/{send_count * 16} events, "
            f"got {len(grouped)}/{len(spans)}"
        )

    sources = []
    destinations = []
    for flow in grouped.values():
        descriptor = (flow[0]["core"], flow[0]["direction"], flow[0]["bits"])
        if descriptor[1] == "SPM_TO_REMOTE":
            sources.append(flow)
        else:
            destinations.append(flow)

    if len(sources) != send_count or len(destinations) != send_count:
        return False, "source/destination transfer count mismatch"
    if {flow[0]["core"] for flow in sources} != {0}:
        return False, "parallel source transfers are not all on core 0"
    if {flow[0]["core"] for flow in destinations} != set(
        range(1, send_count + 1)
    ):
        return False, "destination core set is incorrect"
    if {flow[0]["bits"] for flow in sources + destinations} != {16384}:
        return False, "equal-length batch payload is not exactly 16,384 bits"

    sources.sort(key=lambda flow: flow[0]["xfer"])
    issue_times = [times(flow)[("DTE_pending", "B")] for flow in sources]
    if len(set(issue_times)) != 1:
        return False, f"source transfers were not issued together: {issue_times}"
    issue = issue_times[0]
    completions = [
        times(flow)[("DTE_transmit", "E")] - issue for flow in sources
    ]
    expected_completions = EXPECTED_SOURCE_COMPLETION_NS[
        (send_count, channel_count)
    ]
    if completions != expected_completions:
        return False, (
            f"source completions {completions}, expected {expected_completions}"
        )

    active_intervals = [
        (
            times(flow)[("DTE_pending", "E")],
            times(flow)[("DTE_transmit", "E")],
        )
        for flow in sources
    ]
    launch_intervals = [
        (
            times(flow)[("DTE_launch", "B")],
            times(flow)[("DTE_launch", "E")],
        )
        for flow in sources
    ]
    expected_active = min(send_count, channel_count)
    if max_overlap(active_intervals) != expected_active:
        return False, "active channel maximum does not match channel_count"
    if max_overlap(launch_intervals) != expected_active:
        return False, "launch overlap does not match channel_count"

    transmit_intervals = sorted(
        (
            times(flow)[("DTE_transmit", "B")],
            times(flow)[("DTE_transmit", "E")],
        )
        for flow in sources
    )
    if any(end - start != 16 for start, end in transmit_intervals):
        return False, "a 16,384-bit source did not use exactly 16 ns of bus"
    if any(
        transmit_intervals[index - 1][1] > transmit_intervals[index][0]
        for index in range(1, len(transmit_intervals))
    ):
        return False, "shared-bus transmit intervals overlap"

    return True, (
        f"source completion offsets={completions} ns, "
        f"max_active={expected_active}, shared bus={send_count * 16} ns"
    )


def validate_mixed_batch(
    events: list[dict], stdout: str, channel_count: int
) -> tuple[bool, str]:
    grouped = grouped_spans(events)
    sources = sorted(
        (
            flow for flow in grouped.values()
            if flow[0]["core"] == 0
            and flow[0]["direction"] == "SPM_TO_REMOTE"
        ),
        key=lambda flow: flow[0]["xfer"],
    )
    destinations = [
        flow for flow in grouped.values()
        if flow[0]["core"] == 1
        and flow[0]["direction"] == "REMOTE_TO_SPM"
    ]
    expected_payloads = [16384, 8192, 4096, 2048]
    if [flow[0]["bits"] for flow in sources] != expected_payloads:
        return False, "mixed source payloads are incorrect or out of issue order"
    if sorted(flow[0]["bits"] for flow in destinations) != sorted(
        expected_payloads
    ):
        return False, "same-destination payloads are incorrect"

    issue = times(sources[0])[("DTE_pending", "B")]
    completion_offsets = [
        times(flow)[("DTE_transmit", "E")] - issue for flow in sources
    ]
    expected = {
        1: [36, 64, 88, 110],
        2: [36, 44, 60, 66],
        4: [36, 44, 48, 50],
    }[channel_count]
    if completion_offsets != expected:
        return False, f"mixed completion offsets {completion_offsets}, expected {expected}"

    flow_line = next(
        (line for line in stdout.splitlines() if "[FLOW_DONE]" in line), ""
    )
    positions = [flow_line.find(f"0:{tag}:1") for tag in range(10, 14)]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        return False, f"network DATA completion order is not tag 10→13: {flow_line}"

    send_data_begins = [
        event for event in events
        if event.get("cat") == "Core 000"
        and event.get("name") == "Send_primSEND_DATA"
        and event.get("ph") == "B"
    ]
    send_req_begins = [
        event for event in events
        if event.get("cat") == "Core 000"
        and event.get("name") == "Send_primSEND_REQ"
        and event.get("ph") == "B"
    ]
    recv_ack_begins = [
        event for event in events
        if event.get("cat") == "Core 000"
        and event.get("name") == "Recv_primRECV_ACK"
        and event.get("ph") == "B"
    ]
    if not (
        len(send_data_begins) == len(send_req_begins) == len(recv_ack_begins) == 4
    ):
        return False, "control/data primitive count changed in parallel batch"

    return True, (
        f"same destination, payloads={expected_payloads}, "
        f"completion offsets={completion_offsets} ns, ordered tags 10→13"
    )


def validate_refill(events: list[dict]) -> tuple[bool, str]:
    starts = [
        event for event in dte_spans(events)
        if event["stage"] == "DTE_pending" and event["phase"] == "B"
    ]
    actual = Counter(
        (event["core"], event["direction"], event["bits"])
        for event in starts
    )
    expected = Counter({
        (0, "SPM_TO_REMOTE", 16384): 2,
        (1, "SPM_TO_REMOTE", 16384): 2,
        (2, "REMOTE_TO_SPM", 32768): 2,
    })
    if actual != expected:
        return False, f"unexpected refill descriptors: {actual}"
    xfer_ids = defaultdict(list)
    for event in starts:
        xfer_ids[event["core"]].append(event["xfer"])
    if any(sorted(ids) != [0, 1] for ids in xfer_ids.values()):
        return False, f"refill reused or skipped a context id: {dict(xfer_ids)}"
    return True, "two rounds consume/reinsert metadata and use fresh xfer ids"


def main() -> int:
    tests: list[tuple[str, bool, str]] = []

    off = run_sim(SIM_OFF, WORKLOADS[4], hardware(2))
    tests.append((
        "parallel DTE-off compatibility",
        off.returncode == 0
        and off.finish_ns == 1323
        and not dte_spans(off.trace_events),
        f"exit={off.returncode}, finish={off.finish_ns} ns, no DTE spans",
    ))

    for send_count in (1, 2, 4):
        for channel_count in (1, 2, 4):
            result = run_sim(
                SIM_ON, WORKLOADS[send_count], hardware(channel_count)
            )
            trace_ok, detail = validate_equal_batch(
                result.trace_events, send_count, channel_count
            )
            tests.append((
                f"{send_count} SEND × channel={channel_count}",
                result.returncode == 0
                and result.finish_ns == EXPECTED_FINISH_NS[send_count]
                and trace_ok,
                f"finish={result.finish_ns} ns; {detail}",
            ))

    mixed_workload = (
        "../llm/test/dte/workload/v2_parallel_mixed_same_dest.json"
    )
    for channel_count in (1, 2, 4):
        result = run_sim(SIM_ON, mixed_workload, hardware(channel_count))
        trace_ok, detail = validate_mixed_batch(
            result.trace_events, result.stdout, channel_count
        )
        tests.append((
            f"mixed lengths, same destination, channel={channel_count}",
            result.returncode == 0
            and result.finish_ns == 985
            and trace_ok,
            f"finish={result.finish_ns} ns; {detail}",
        ))

    refill_workload = "../llm/test/dte/workload/v1_repeated_flow.json"
    refill_off = run_sim(SIM_OFF, refill_workload, hardware(2))
    refill_on = run_sim(SIM_ON, refill_workload, hardware(2))
    refill_ok, refill_detail = validate_refill(refill_on.trace_events)
    tests.append((
        "parallel pipeline/refill lifecycle",
        refill_off.returncode == 0
        and refill_on.returncode == 0
        and refill_off.finish_ns == 965
        and refill_on.finish_ns == 1051
        and not dte_spans(refill_off.trace_events)
        and refill_ok,
        f"off/on={refill_off.finish_ns}/{refill_on.finish_ns} ns; "
        f"{refill_detail}",
    ))

    unsupported = run_sim(
        SIM_ON,
        "../llm/test/dte/workload/v1_cross_die_stripe2.json",
        "../llm/test/dte/hardware/v1_cross_die.json",
    )
    unsupported_error = (
        "V5 striping is supported by the sequential dataflow path, "
        "not parallel-send pipeline mode"
    )
    tests.append((
        "unsupported parallel stripe rejected clearly",
        unsupported.returncode != 0 and unsupported_error in unsupported.stdout,
        f"exit={unsupported.returncode}, clear error="
        f"{unsupported_error in unsupported.stdout}",
    ))

    for name, passed, detail in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return 0 if all(passed for _, passed, _ in tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
