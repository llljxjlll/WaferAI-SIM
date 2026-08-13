#!/usr/bin/env python3
"""Run DTE V2b streaming, bottleneck-migration, and gate tests."""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict

from oracle import streaming_completion_ns
from run_test_dte_v1 import dte_spans, run_sim

WORKLOAD = "../llm/test/dte/workload/v2_parallel_one.json"
CYCLE_STREAM = "../llm/test/dte/simulation/v2b_cycle_streaming.json"
CYCLE_STORE = "../llm/test/dte/simulation/v2b_cycle_store.json"
BEHA_STREAM = "../llm/test/dte/simulation/v2b_beha_streaming.json"
BEHA_STORE = "../llm/test/dte/simulation/v2b_beha_store.json"
CYCLE_NS = 2

STREAM_RE = re.compile(
    r"^(DTE_stream_(?:source_fill|network|destination)) "
    r"source=(\d+) dest=(\d+) tag=(\d+) bits=(\d+)$"
)


def hardware(name: str) -> str:
    return f"../llm/test/dte/hardware/{name}.json"


def stream_flows(events: list[dict]) -> dict[tuple[int, int, int], list[dict]]:
    flows: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for event in events:
        match = STREAM_RE.match(str(event.get("name", "")))
        if not match or event.get("ph") not in {"B", "E"}:
            continue
        stage, source, dest, tag, bits = match.groups()
        flows[(int(source), int(dest), int(tag))].append({
            "stage": stage,
            "phase": event["ph"],
            "bits": int(bits),
            "ns": round(float(event["ts"]) * 1000),
        })
    return dict(flows)


def validate_flow(
    events: list[dict], key: tuple[int, int, int], expected_bits: int,
    require_bulk_overlap: bool = True,
) -> tuple[bool, str, tuple[int, int, int]]:
    flows = stream_flows(events)
    if key not in flows:
        return False, f"missing stream flow {key}; got {sorted(flows)}", (0, 0, 0)
    flow = flows[key]
    expected_fields = {
        ("DTE_stream_source_fill", "B"),
        ("DTE_stream_source_fill", "E"),
        ("DTE_stream_network", "B"),
        ("DTE_stream_network", "E"),
        ("DTE_stream_destination", "B"),
        ("DTE_stream_destination", "E"),
    }
    fields = {(event["stage"], event["phase"]) for event in flow}
    if len(flow) != 6 or fields != expected_fields:
        return False, f"incomplete stream stages: {fields}", (0, 0, 0)
    if {event["bits"] for event in flow} != {expected_bits}:
        return False, "stream trace payload mismatch", (0, 0, 0)

    times = {
        (event["stage"], event["phase"]): event["ns"] for event in flow
    }
    source_fill_b = times[("DTE_stream_source_fill", "B")]
    source_first = times[("DTE_stream_source_fill", "E")]
    network_b = times[("DTE_stream_network", "B")]
    network_e = times[("DTE_stream_network", "E")]
    destination_b = times[("DTE_stream_destination", "B")]
    destination_e = times[("DTE_stream_destination", "E")]
    if not (
        source_fill_b <= source_first == network_b < destination_b
        < network_e < destination_e
    ):
        return False, f"non-monotonic stream boundaries: {times}", (0, 0, 0)

    source, dest, _ = key
    spans = dte_spans(events)
    source_done = [
        event["ns"] for event in spans
        if event["core"] == source
        and event["direction"] == "SPM_TO_REMOTE"
        and event["stage"] == "DTE_transmit"
        and event["phase"] == "E"
    ]
    destination_done = [
        event["ns"] for event in spans
        if event["core"] == dest
        and event["direction"] == "REMOTE_TO_SPM"
        and event["stage"] == "DTE_transmit"
        and event["phase"] == "E"
    ]
    if len(source_done) != 1 or len(destination_done) != 1:
        return False, "flow does not map to exactly two endpoint DTE transfers", (0, 0, 0)

    source_candidate = (
        max(source_first, source_done[0]) + (destination_b - source_first)
        + CYCLE_NS
    )
    network_candidate = network_e + CYCLE_NS
    destination_candidate = destination_done[0]
    oracle = streaming_completion_ns(
        source_first_ns=source_first,
        source_done_ns=source_done[0],
        destination_first_ns=destination_b,
        network_last_ns=network_e,
        destination_dte_done_ns=destination_done[0],
        drain_ns=CYCLE_NS,
    )
    if destination_e != oracle:
        return False, f"destination end {destination_e}, oracle {oracle}", (0, 0, 0)
    if require_bulk_overlap and not (
        network_b < source_done[0] and destination_b < network_e
    ):
        return False, "source/network/destination spans do not overlap", (0, 0, 0)
    return True, (
        f"fill={source_fill_b}→{source_first}, network={network_b}→{network_e}, "
        f"destination={destination_b}→{destination_e} ns"
    ), (source_candidate, network_candidate, destination_candidate)


def main() -> int:
    tests: list[tuple[str, bool, str]] = []
    expected_cycle = {
        "v2b_source_slow": (1125, 927),
        "v2b_network_slow": (621, 603),
        "v2b_destination_slow": (1125, 931),
        "v2b_equal": (861, 605),
    }
    expected_beha = {
        "v2b_source_slow": (1001, 923),
        "v2b_network_slow": (497, 479),
        "v2b_destination_slow": (1001, 931),
        "v2b_equal": (737, 547),
    }

    cycle_payloads: set[int] = set()
    for name, (store_ns, stream_ns) in expected_cycle.items():
        store = run_sim(CYCLE_STORE, WORKLOAD, hardware(name))
        stream = run_sim(CYCLE_STREAM, WORKLOAD, hardware(name))
        flow_ok, detail, _ = validate_flow(stream.trace_events, (0, 1, 1), 16384)
        payloads = {
            event["bits"] for event in dte_spans(stream.trace_events)
            if event["stage"] == "DTE_pending" and event["phase"] == "B"
        }
        cycle_payloads |= payloads
        tests.append((
            f"physical {name.removeprefix('v2b_')}",
            store.returncode == stream.returncode == 0
            and store.finish_ns == store_ns
            and stream.finish_ns == stream_ns
            and stream_ns <= store_ns
            and not stream_flows(store.trace_events)
            and flow_ok,
            f"store/stream={store.finish_ns}/{stream.finish_ns} ns; {detail}",
        ))

    expected_owner = {
        "v2b_source_slow": 0,
        "v2b_network_slow": 1,
        "v2b_destination_slow": 2,
    }
    beha_payloads: set[int] = set()
    for name, (store_ns, stream_ns) in expected_beha.items():
        store = run_sim(BEHA_STORE, WORKLOAD, hardware(name))
        stream = run_sim(BEHA_STREAM, WORKLOAD, hardware(name))
        flow_ok, detail, candidates = validate_flow(
            stream.trace_events, (0, 1, 1), 16384
        )
        payloads = {
            event["bits"] for event in dte_spans(stream.trace_events)
            if event["stage"] == "DTE_pending" and event["phase"] == "B"
        }
        beha_payloads |= payloads
        owner_ok = True
        if name in expected_owner:
            owner_ok = candidates.index(max(candidates)) == expected_owner[name]
        tests.append((
            f"behavioral {name.removeprefix('v2b_')}",
            store.returncode == stream.returncode == 0
            and store.finish_ns == store_ns
            and stream.finish_ns == stream_ns
            and stream_ns <= store_ns
            and flow_ok and owner_ok,
            f"store/stream={store.finish_ns}/{stream.finish_ns} ns; "
            f"tail candidates={candidates}; {detail}",
        ))

    tests.append((
        "physical/behavioral payload contract",
        cycle_payloads == beha_payloads == {16384},
        f"physical={cycle_payloads}, behavioral={beha_payloads}",
    ))

    width_expected = {64: 931, 128: 675, 256: 547, 512: 483,
                      1024: 479, 2048: 479, 4096: 479}
    width_results = []
    width_ok = True
    for width, expected in width_expected.items():
        result = run_sim(BEHA_STREAM, WORKLOAD, hardware(f"v2b_width_{width}"))
        width_results.append(result.finish_ns)
        width_ok &= result.returncode == 0 and result.finish_ns == expected
    width_ok &= all(a >= b for a, b in zip(width_results, width_results[1:]))
    width_ok &= width_results[-3:] == [479, 479, 479]
    tests.append((
        "width scan crosses into network bottleneck",
        width_ok,
        f"64→4096 bit finish times={width_results} ns",
    ))

    small = run_sim(
        CYCLE_STREAM,
        "../llm/test/dte/workload/v2b_small.json",
        hardware("v2b_equal"),
    )
    small_ok, small_detail, _ = validate_flow(
        small.trace_events, (0, 1, 1), 8, require_bulk_overlap=False
    )
    tests.append((
        "small payload fill/launch",
        small.returncode == 0 and small.finish_ns == 252 and small_ok,
        f"finish={small.finish_ns} ns; {small_detail}",
    ))

    nondiv = run_sim(
        CYCLE_STREAM,
        "../llm/test/dte/workload/v2b_nondiv.json",
        hardware("v2b_equal"),
    )
    nondiv_ok, nondiv_detail, _ = validate_flow(
        nondiv.trace_events, (0, 1, 1), 16416
    )
    source_transmit = [
        event for event in dte_spans(nondiv.trace_events)
        if event["core"] == 0 and event["stage"] == "DTE_transmit"
    ]
    tx_times = {(event["phase"]): event["ns"] for event in source_transmit}
    tests.append((
        "non-divisible 16,416-bit payload",
        nondiv.returncode == 0 and nondiv.finish_ns == 626
        and nondiv_ok and tx_times.get("E", 0) - tx_times.get("B", 0) == 130,
        f"finish={nondiv.finish_ns} ns, source transmit=130 ns; {nondiv_detail}",
    ))

    cross_workload = "../llm/test/dte/workload/v1_cross_die_stripe4.json"
    cross_hw = hardware("v2b_cross_die_behavioral")
    cross_store = run_sim(BEHA_STORE, cross_workload, cross_hw)
    cross_stream = run_sim(BEHA_STREAM, cross_workload, cross_hw)
    local_ok, local_detail, _ = validate_flow(
        cross_stream.trace_events, (5, 7, 7), 16384
    )
    remote_ok, remote_detail, _ = validate_flow(
        cross_stream.trace_events, (8, 24, 24), 16384
    )
    tests.append((
        "behavioral D2D stripe=4 local/cross-die",
        cross_store.returncode == cross_stream.returncode == 0
        and cross_store.finish_ns == 632
        and cross_stream.finish_ns == 604
        and local_ok and remote_ok,
        f"store/stream={cross_store.finish_ns}/{cross_stream.finish_ns} ns; "
        f"local {local_detail}; cross {remote_detail}",
    ))

    multi = run_sim(
        CYCLE_STREAM,
        "../llm/test/dte/workload/v1_multi_source.json",
        hardware("v2b_equal"),
    )
    multi_starts = Counter(
        (event["core"], event["direction"], event["bits"])
        for event in dte_spans(multi.trace_events)
        if event["stage"] == "DTE_pending" and event["phase"] == "B"
    )
    multi_flows = stream_flows(multi.trace_events)
    multi_flow_ok = (
        set(multi_flows) == {(0, 2, 2), (1, 2, 2)}
        and all(len(flow) == 6 for flow in multi_flows.values())
        and all({event["bits"] for event in flow} == {16384}
                for flow in multi_flows.values())
    )
    tests.append((
        "multi-source destination contexts",
        multi.returncode == 0 and multi.finish_ns == 747
        and multi_starts == Counter({
            (0, "SPM_TO_REMOTE", 16384): 1,
            (1, "SPM_TO_REMOTE", 16384): 1,
            (2, "REMOTE_TO_SPM", 16384): 2,
        }) and multi_flow_ok,
        f"finish={multi.finish_ns} ns, descriptors={dict(multi_starts)}",
    ))

    repeated = run_sim(
        CYCLE_STREAM,
        "../llm/test/dte/workload/v1_repeated_flow.json",
        hardware("v2b_equal"),
    )
    repeated_starts = [
        event for event in dte_spans(repeated.trace_events)
        if event["stage"] == "DTE_pending" and event["phase"] == "B"
    ]
    repeated_ids: dict[int, list[int]] = defaultdict(list)
    for event in repeated_starts:
        repeated_ids[event["core"]].append(event["xfer"])
    repeated_flows = stream_flows(repeated.trace_events)
    repeated_flow_ok = (
        set(repeated_flows) == {(0, 2, 2), (1, 2, 2)}
        and all(len(flow) == 12 for flow in repeated_flows.values())
        and all(
            sum(event["stage"] == stage and event["phase"] == phase
                for event in flow) == 2
            for flow in repeated_flows.values()
            for stage in (
                "DTE_stream_source_fill", "DTE_stream_network",
                "DTE_stream_destination",
            )
            for phase in ("B", "E")
        )
    )
    tests.append((
        "streaming repeated source/tag lifecycle",
        repeated.returncode == 0 and repeated.finish_ns == 1235
        and {core: sorted(ids) for core, ids in repeated_ids.items()} == {
            0: [0, 1], 1: [0, 1], 2: [0, 1, 2, 3]
        } and repeated_flow_ok,
        f"finish={repeated.finish_ns} ns, xfer_ids={dict(repeated_ids)}",
    ))

    invalid_off = run_sim(
        "../llm/test/dte/simulation/v2b_invalid_without_dte.json",
        WORKLOAD, hardware("v2b_equal"),
    )
    tests.append((
        "streaming requires DTE",
        invalid_off.returncode != 0
        and "dte.streaming requires dte.use_beha_dte=true" in invalid_off.stdout,
        f"exit={invalid_off.returncode}",
    ))
    invalid_non_dataflow = run_sim(
        CYCLE_STREAM,
        "../llm/test/dte/workload/v2b_invalid_non_dataflow.json",
        hardware("v2b_equal"),
    )
    tests.append((
        "streaming non-dataflow gate",
        invalid_non_dataflow.returncode != 0
        and "supported only for dataflow workloads"
        in invalid_non_dataflow.stdout,
        f"exit={invalid_non_dataflow.returncode}",
    ))
    invalid_parallel = run_sim(
        "../llm/test/dte/simulation/v2b_invalid_parallel.json",
        WORKLOAD, hardware("v2b_equal"),
    )
    tests.append((
        "streaming parallel dispatcher gate",
        invalid_parallel.returncode != 0
        and "does not yet support the parallel dispatcher" in invalid_parallel.stdout,
        f"exit={invalid_parallel.returncode}",
    ))

    failures = 0
    for name, passed, detail in tests:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        failures += not passed
    print(f"DTE V2b integration: {len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
