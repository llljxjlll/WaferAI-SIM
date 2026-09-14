#!/usr/bin/env python3
"""Explicit 4x4-core to D2D-port congestion model for Exp3.1 controls.

This model is intentionally used only by the C00/C10 controlled inter-die
ablation.  The W00/W11 Exp1 compatibility track must remain untouched.
"""

from __future__ import annotations

import math
from typing import Iterable


CORE_MESH_SIDE = 4
DTE_CHANNEL_COUNT = 2
# Two opposing edge attachments implement the frozen ``DTE_channel=2`` plan
# assumption without granting a hidden all-core injection crossbar.
DTE_PORT_ATTACHMENTS = ((0, 1), (3, 2))

Coordinate = tuple[int, int]
DirectedLink = tuple[Coordinate, Coordinate]


def _xy_path(source: Coordinate, destination: Coordinate) -> Iterable[DirectedLink]:
    """Yield deterministic X-first links on the 4x4 core mesh."""
    x, y = source
    dx, dy = destination
    while x != dx:
        step = 1 if dx > x else -1
        nxt = (x + step, y)
        yield (x, y), nxt
        x += step
    while y != dy:
        step = 1 if dy > y else -1
        nxt = (x, y + step)
        yield (x, y), nxt
        y += step


def _port_for_core(core: Coordinate) -> int:
    """Use the nearest frozen DTE port, with a deterministic tie-break."""
    return min(
        range(DTE_CHANNEL_COUNT),
        key=lambda index: (
            abs(core[0] - DTE_PORT_ATTACHMENTS[index][0])
            + abs(core[1] - DTE_PORT_ATTACHMENTS[index][1]),
            index,
        ),
    )


def _route_loads(
    messages: Iterable[tuple[Coordinate, Coordinate, float]],
) -> dict[DirectedLink, float]:
    loads: dict[DirectedLink, float] = {}
    for source, destination, payload in messages:
        for link in _xy_path(source, destination):
            loads[link] = loads.get(link, 0.0) + payload
    return loads


def _max_bytes(loads: dict[DirectedLink, float]) -> float:
    return max(loads.values(), default=0.0)


def _serialized_loads(loads: dict[DirectedLink, float]) -> list[dict[str, object]]:
    return [
        {
            "source": list(source), "destination": list(destination),
            "bytes": loads[(source, destination)],
        }
        for source, destination in sorted(loads)
    ]


def core_to_d2d_port_metrics(
    payload_bytes_per_die: float,
    *,
    noc_link_bps: float,
    dte_channel_bps: float,
    clock_hz: float,
) -> dict[str, object]:
    """Replay canonical core->port injection and port->core ejection traffic.

    Every active core contributes an equal share of the die's remote payload.
    The route is explicit and directional.  Source and destination traffic are
    retained separately for C00 serialization and combined for C10 streaming
    contention on the same local NoC links.
    """
    payload = float(payload_bytes_per_die)
    if not math.isfinite(payload) or payload < 0:
        raise ValueError("payload_bytes_per_die must be finite and non-negative")
    if min(noc_link_bps, dte_channel_bps, clock_hz) <= 0:
        raise ValueError("link bandwidths and clock must be positive")
    cores = [
        (row, column)
        for row in range(CORE_MESH_SIDE)
        for column in range(CORE_MESH_SIDE)
    ]
    payload_per_core = payload / len(cores)
    assignments = {core: _port_for_core(core) for core in cores}
    source_messages = [
        (core, DTE_PORT_ATTACHMENTS[assignments[core]], payload_per_core)
        for core in cores
    ]
    destination_messages = [
        (DTE_PORT_ATTACHMENTS[assignments[core]], core, payload_per_core)
        for core in cores
    ]
    source_loads = _route_loads(source_messages)
    destination_loads = _route_loads(destination_messages)
    combined_loads = dict(source_loads)
    for link, value in destination_loads.items():
        combined_loads[link] = combined_loads.get(link, 0.0) + value
    per_port_bytes = [0.0] * DTE_CHANNEL_COUNT
    per_port_core_counts = [0] * DTE_CHANNEL_COUNT
    for port in assignments.values():
        per_port_bytes[port] += payload_per_core
        per_port_core_counts[port] += 1
    source_noc = _max_bytes(source_loads) / noc_link_bps * clock_hz
    destination_noc = _max_bytes(destination_loads) / noc_link_bps * clock_hz
    shared_noc = _max_bytes(combined_loads) / noc_link_bps * clock_hz
    port_service = max(per_port_bytes, default=0.0) / dte_channel_bps * clock_hz
    return {
        "model": "explicit_4x4_core_to_two_dte_ports_x_first",
        "dte_channel_count": DTE_CHANNEL_COUNT,
        "dte_port_attachments": [list(port) for port in DTE_PORT_ATTACHMENTS],
        "noc_link_bps": noc_link_bps,
        "dte_channel_bps": dte_channel_bps,
        "payload_bytes_per_die": payload,
        "payload_bytes_per_core": payload_per_core,
        "per_port_payload_bytes": per_port_bytes,
        "per_port_core_counts": per_port_core_counts,
        "source_port_noc_cycles": source_noc,
        "destination_port_noc_cycles": destination_noc,
        "shared_port_noc_cycles": shared_noc,
        "source_port_service_cycles": port_service,
        "destination_port_service_cycles": port_service,
        "gateway_serial_cycles": source_noc + port_service + port_service + destination_noc,
        "gateway_streaming_cycles": max(shared_noc, port_service),
        "source_port_max_directed_link_bytes": _max_bytes(source_loads),
        "destination_port_max_directed_link_bytes": _max_bytes(destination_loads),
        "shared_port_max_directed_link_bytes": _max_bytes(combined_loads),
        "shared_port_total_byte_hops": sum(combined_loads.values()),
        "shared_port_directed_link_loads": _serialized_loads(combined_loads),
    }


def stream_port_and_fabric(
    metrics: dict[str, object], *, fabric_cycles: float, waves: int,
) -> tuple[float, float]:
    """Return C00 serial and C10 streaming communication durations.

    C10 does not change the canonical core/port mapping.  It merely streams
    equal chunks through that same congestion-limited path, then pays a small
    fill/drain penalty.  C00 instead fully serializes source mesh routing,
    source port service, fabric, destination port service, and destination
    mesh routing.
    """
    if not math.isfinite(fabric_cycles) or fabric_cycles < 0:
        raise ValueError("fabric_cycles must be finite and non-negative")
    wave_count = max(1, int(waves))
    serial_gateway = float(metrics["gateway_serial_cycles"])
    streaming_gateway = float(metrics["gateway_streaming_cycles"])
    serial = serial_gateway + fabric_cycles
    # The steady-state bottleneck is shared by the source and destination
    # core-mesh traffic, the DTE-port service, and the existing Exp1 fabric
    # model.  One initial chunk plus a small drain completes the pipeline.
    streaming = max(streaming_gateway, fabric_cycles)
    streaming += min(streaming_gateway, fabric_cycles) / wave_count
    streaming += streaming_gateway * 0.015
    return serial, streaming


__all__ = [
    "CORE_MESH_SIDE",
    "DTE_CHANNEL_COUNT",
    "DTE_PORT_ATTACHMENTS",
    "core_to_d2d_port_metrics",
    "stream_port_and_fabric",
]
