#!/usr/bin/env python3
"""Independent cycle-level oracle for the DTE V0 resource model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil_div requires numerator >= 0 and denominator > 0")
    return numerator // denominator + (numerator % denominator != 0)


def streaming_completion_ns(
    *,
    source_first_ns: int,
    source_done_ns: int,
    destination_first_ns: int,
    network_last_ns: int,
    destination_dte_done_ns: int,
    drain_ns: int,
) -> int:
    """V2b flow-level tail oracle.

    The source tail is projected through the observed first-unit network
    latency.  The final write is the maximum of source, network, and
    destination-DTE tails, with one destination drain interval after either
    upstream tail.
    """
    values = (
        source_first_ns,
        source_done_ns,
        destination_first_ns,
        network_last_ns,
        destination_dte_done_ns,
        drain_ns,
    )
    if any(value < 0 for value in values) or drain_ns == 0:
        raise ValueError("streaming times must be non-negative and drain > 0")
    if destination_first_ns < source_first_ns:
        raise ValueError("destination first unit precedes source first unit")
    first_latency_ns = destination_first_ns - source_first_ns
    source_tail_at_destination = (
        max(source_first_ns, source_done_ns) + first_latency_ns
    )
    return max(
        destination_dte_done_ns,
        source_tail_at_destination + drain_ns,
        network_last_ns + drain_ns,
    )


@dataclass(frozen=True)
class Request:
    issue_cycle: int
    payload_bits: int


def simulate(requests: Iterable[Request], channel_count: int, bit_width_bits: int,
             gamma_cycles: int, tau_launch_cycles: int) -> list[dict]:
    requests = list(requests)
    if channel_count <= 0 or bit_width_bits <= 0:
        raise ValueError("channel_count and bit_width_bits must be > 0")
    if gamma_cycles < 0 or tau_launch_cycles < 0:
        raise ValueError("launch components must be >= 0")
    if any(r.issue_cycle < 0 or r.payload_bits <= 0 for r in requests):
        raise ValueError("requests require issue_cycle >= 0 and payload_bits > 0")

    ordered = sorted(enumerate(requests), key=lambda item: (item[1].issue_cycle, item[0]))
    pending: list[int] = []
    active: list[dict | None] = [None] * channel_count
    result = [{"id": i, "issue": r.issue_cycle} for i, r in enumerate(requests)]
    cursor = 0
    now = 0
    bus: dict | None = None
    last_served = -1
    completed = 0

    while completed < len(requests):
        deadlines = []
        if cursor < len(ordered):
            deadlines.append(ordered[cursor][1].issue_cycle)
        if bus is not None:
            deadlines.append(bus["done"])
        deadlines.extend(a["launch_done"] for a in active
                         if a is not None and a["state"] == "launch")
        if not deadlines:
            raise RuntimeError("oracle deadlock")
        now = min(deadlines)

        if bus is not None and bus["done"] == now:
            channel = bus["channel"]
            req_id = bus["id"]
            result[req_id]["complete"] = now
            active[channel] = None
            bus = None
            completed += 1

        for item in active:
            if item is not None and item["state"] == "launch" and item["launch_done"] <= now:
                item["state"] = "ready"

        while cursor < len(ordered) and ordered[cursor][1].issue_cycle <= now:
            pending.append(ordered[cursor][0])
            cursor += 1

        while pending and any(item is None for item in active):
            channel = active.index(None)
            req_id = pending.pop(0)
            launch_done = now + gamma_cycles + tau_launch_cycles
            active[channel] = {"id": req_id, "channel": channel,
                               "state": "launch", "launch_done": launch_done}
            result[req_id].update(channel=channel, admitted=now,
                                  launch_done=launch_done)

        # Zero-cycle launch must become ready without advancing time.
        for item in active:
            if item is not None and item["state"] == "launch" and item["launch_done"] <= now:
                item["state"] = "ready"

        if bus is None:
            for offset in range(1, channel_count + 1):
                channel = (last_served + offset + channel_count) % channel_count
                item = active[channel]
                if item is None or item["state"] != "ready":
                    continue
                req_id = item["id"]
                done = now + ceil_div(requests[req_id].payload_bits, bit_width_bits)
                item["state"] = "transmit"
                bus = {"id": req_id, "channel": channel, "done": done}
                last_served = channel
                result[req_id]["transmit_start"] = now
                break

    return result



@dataclass(frozen=True)
class AggregationOracleResult:
    group_sizes: tuple[int, ...]
    physical_transfers: int
    launch_savings: int
    bus_cycles: int
    completion_cycle_from_first_issue: int
    bandwidth_utilization_ppm: int


def aggregation_oracle(
    *,
    logical_descriptors: int,
    payload_bits: int,
    max_descriptors: int,
    max_payload_bytes: int,
    bit_width_bits: int,
    launch_cycles: int,
    issue_spacing_cycles: int = 1,
) -> AggregationOracleResult:
    """V3b contiguous-request, one-channel compound-descriptor oracle.

    Groups flush when either the descriptor or byte limit is reached. Physical
    groups queue on one DTE channel; collection time before each group launch is
    retained instead of being hidden by the launch-savings calculation.
    """
    if logical_descriptors <= 0 or payload_bits <= 0:
        raise ValueError("aggregation oracle needs positive work")
    if payload_bits % 8:
        raise ValueError("V3b coalescing requires byte-aligned payloads")
    if max_descriptors < 1 or max_payload_bytes <= 0:
        raise ValueError("aggregation limits must be positive")
    if bit_width_bits <= 0 or launch_cycles < 0 or issue_spacing_cycles < 0:
        raise ValueError("invalid DTE timing parameters")

    payload_bytes = payload_bits // 8
    per_group = min(max_descriptors, max_payload_bytes // payload_bytes)
    if per_group <= 0:
        per_group = 1
    groups = []
    remaining = logical_descriptors
    while remaining:
        members = min(per_group, remaining)
        groups.append(members)
        remaining -= members

    completion = 0
    issued = 0
    bus_cycles = 0
    bus_capacity_bits = 0
    for members in groups:
        issued += members
        physical_issue = (issued - 1) * issue_spacing_cycles
        transfer_cycles = ceil_div(members * payload_bits, bit_width_bits)
        completion = max(completion, physical_issue) + launch_cycles + transfer_cycles
        bus_cycles += transfer_cycles
        bus_capacity_bits += transfer_cycles * bit_width_bits

    useful_bits = logical_descriptors * payload_bits
    utilization = round(useful_bits * 1_000_000 / bus_capacity_bits)
    return AggregationOracleResult(
        group_sizes=tuple(groups),
        physical_transfers=len(groups),
        launch_savings=logical_descriptors - len(groups),
        bus_cycles=bus_cycles,
        completion_cycle_from_first_issue=completion,
        bandwidth_utilization_ppm=utilization,
    )


V4_DIRECTION_PORTS = {
    "SPM_TO_REMOTE": ("SPM_READ",),
    "REMOTE_TO_SPM": ("SPM_WRITE",),
    "SPM_TO_SPM": ("SPM_READ", "SPM_WRITE"),
    "SPM_TO_DRAM": ("SPM_READ", "AXI_WRITE"),
    "DRAM_TO_SPM": ("AXI_READ", "SPM_WRITE"),
    "DRAM_TO_REMOTE": ("AXI_READ",),
}


def v4_port_service_cycles(
    direction: str, payload_bits: int, widths: dict[str, int]
) -> dict[str, int]:
    # Endpoint-port service only; network and DRAM media are excluded.
    if direction not in V4_DIRECTION_PORTS or payload_bits <= 0:
        raise ValueError("invalid V4 direction or payload")
    result = {}
    for port in V4_DIRECTION_PORTS[direction]:
        width = widths.get(port, 0)
        if width <= 0:
            raise ValueError("V4 port width must be positive")
        result[port] = ceil_div(payload_bits, width)
    return result


def v4_dynamic_energy_pj(
    direction: str, payload_bits: int, launch_pj: float,
    spm_pj_per_bit: float, axi_pj_per_bit: float,
) -> float:
    ports = V4_DIRECTION_PORTS[direction]
    spm_accesses = sum(port.startswith("SPM_") for port in ports)
    axi_accesses = sum(port.startswith("AXI_") for port in ports)
    return (launch_pj + payload_bits *
            (spm_accesses * spm_pj_per_bit +
             axi_accesses * axi_pj_per_bit))


def v4_area_um2(
    *, channels: int, command_slots: int, widths: dict[str, int],
    base_area: float, channel_area: float, slot_area: float,
    port_bit_area: float,
) -> float:
    if channels <= 0 or command_slots <= 0:
        raise ValueError("invalid V4 area dimensions")
    return (base_area + channels * channel_area +
            channels * command_slots * slot_area +
            sum(widths.values()) * port_bit_area)

def _self_test() -> None:
    burst = [Request(0, 129) for _ in range(4)]
    expected = {
        1: [5, 10, 15, 20],
        2: [5, 7, 10, 12],
        4: [5, 7, 9, 11],
    }
    for channels, completions in expected.items():
        got = simulate(burst, channels, 128, 2, 1)
        assert [item["complete"] for item in got] == completions
        intervals = sorted((item["transmit_start"], item["complete"])
                           for item in got)
        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    assert ceil_div(1, 128) == 1
    assert ceil_div(128, 128) == 1
    assert ceil_div(129, 128) == 2
    assert streaming_completion_ns(
        source_first_ns=405,
        source_done_ns=899,
        destination_first_ns=415,
        network_last_ns=913,
        destination_dte_done_ns=429,
        drain_ns=2,
    ) == 915
    assert streaming_completion_ns(
        source_first_ns=391,
        source_done_ns=399,
        destination_first_ns=401,
        network_last_ns=527,
        destination_dte_done_ns=919,
        drain_ns=2,
    ) == 919
    trend = [
        aggregation_oracle(
            logical_descriptors=16,
            payload_bits=64,
            max_descriptors=group,
            max_payload_bytes=4096,
            bit_width_bits=128,
            launch_cycles=10,
        )
        for group in (1, 2, 4, 8, 16)
    ]
    assert [item.physical_transfers for item in trend] == [16, 8, 4, 2, 1]
    assert [item.completion_cycle_from_first_issue for item in trend] == [176, 89, 51, 35, 33]
    assert [item.bandwidth_utilization_ppm for item in trend] == [500000, 1000000, 1000000, 1000000, 1000000]
    widths = {"SPM_READ": 64, "SPM_WRITE": 32,
              "AXI_READ": 16, "AXI_WRITE": 128}
    assert v4_port_service_cycles("SPM_TO_DRAM", 4096, widths) == {
        "SPM_READ": 64, "AXI_WRITE": 32}
    assert v4_port_service_cycles("DRAM_TO_SPM", 4096, widths) == {
        "AXI_READ": 256, "SPM_WRITE": 128}
    energy = sum(v4_dynamic_energy_pj(
        direction, 4096, 10.0, 0.01, 0.02)
        for direction in V4_DIRECTION_PORTS)
    assert abs(energy - 551.52) < 1e-9
    assert v4_area_um2(
        channels=1, command_slots=2, widths=widths,
        base_area=1000.0, channel_area=100.0, slot_area=10.0,
        port_bit_area=0.5) == 1240.0
    print("DTE V0/V2b/V3b/V4 oracle self-test: PASS")


if __name__ == "__main__":
    _self_test()
