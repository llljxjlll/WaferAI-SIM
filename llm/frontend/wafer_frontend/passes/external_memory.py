"""Deterministic Python external-memory transfer and queue service."""

from __future__ import annotations

import hashlib

from ..errors import SchemaError
from ..schema.external_memory import (
    ExternalMemoryConnection,
    ExternalMemoryFabric,
    ExternalTransferCompletion,
    ExternalTransferDirection,
    ExternalTransferReport,
    ExternalTransferRequest,
    ExternalTransferStats,
    build_external_transfer_stats,
)
from ..schema.memory_plan import MemoryTierCapacity


class SparseMemoryImage:
    """Capacity-bounded sparse byte storage used by the transfer service."""

    __slots__ = ("capacity", "_nonzero")

    def __init__(self, capacity: MemoryTierCapacity) -> None:
        if type(capacity) is not MemoryTierCapacity:
            raise SchemaError(
                "must be a MemoryTierCapacity",
                path="capacity",
            )
        capacity.validate("capacity")
        self.capacity = capacity
        self._nonzero: dict[int, int] = {}

    def _validate_range(self, address: int, size_bytes: int, path: str) -> None:
        if type(address) is not int or type(size_bytes) is not int:
            raise SchemaError(
                "address and size must be integers",
                path=path,
            )
        if size_bytes <= 0:
            raise SchemaError(
                "size must be greater than zero",
                path=f"{path}.size_bytes",
            )
        end = address + size_bytes
        capacity_end = (
            self.capacity.base_address + self.capacity.capacity_bytes
        )
        if (
            address < self.capacity.base_address
            or end < address
            or end > capacity_end
        ):
            raise SchemaError(
                "memory access exceeds configured capacity",
                path=path,
                code="external_memory_address_out_of_range",
            )

    def read(self, address: int, size_bytes: int) -> bytes:
        self._validate_range(address, size_bytes, "memory_image.read")
        return bytes(
            self._nonzero.get(address + offset, 0)
            for offset in range(size_bytes)
        )

    def write(self, address: int, payload: bytes) -> None:
        if type(payload) is not bytes or not payload:
            raise SchemaError(
                "payload must be non-empty bytes",
                path="memory_image.write.payload",
            )
        self._validate_range(address, len(payload), "memory_image.write")
        for offset, value in enumerate(payload):
            byte_address = address + offset
            if value:
                self._nonzero[byte_address] = value
            else:
                self._nonzero.pop(byte_address, None)

    @property
    def nonzero_byte_count(self) -> int:
        return len(self._nonzero)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def recompute_external_transfer_stats(
    requests: tuple[ExternalTransferRequest, ...],
    completions: tuple[ExternalTransferCompletion, ...],
) -> ExternalTransferStats:
    """Recompute statistics from completed request timing."""

    return build_external_transfer_stats(requests, completions)


def _validate_images(
    *,
    capacities: tuple[MemoryTierCapacity, ...],
    images: dict[str, SparseMemoryImage],
    path: str,
) -> None:
    expected_refs = {item.id for item in capacities}
    if set(images) != expected_refs:
        raise SchemaError(
            "images must exactly cover configured capacities",
            path=path,
        )
    for capacity in capacities:
        image = images[capacity.id]
        if type(image) is not SparseMemoryImage:
            raise SchemaError(
                "must be a SparseMemoryImage",
                path=f"{path}[{capacity.id!r}]",
            )
        if image.capacity != capacity:
            raise SchemaError(
                "image capacity does not match fabric",
                path=f"{path}[{capacity.id!r}]",
            )


def _connection_and_capacities(
    fabric: ExternalMemoryFabric,
    request: ExternalTransferRequest,
) -> tuple[
    ExternalMemoryConnection,
    MemoryTierCapacity,
    MemoryTierCapacity,
]:
    connections = {item.id: item for item in fabric.connections}
    connection = connections.get(request.connection_ref)
    if connection is None:
        raise SchemaError(
            "request has no declared external connection",
            path="request.connection_ref",
            code="external_connection_missing",
        )
    link = next(item for item in fabric.links if item.id == connection.link_ref)
    external = next(
        item
        for item in fabric.external_capacities
        if item.id == link.external_capacity_ref
    )
    hbm = next(
        item
        for item in fabric.hbm_capacities
        if item.id == connection.hbm_capacity_ref
    )
    return connection, external, hbm


def execute_external_transfers(
    *,
    fabric: ExternalMemoryFabric,
    requests: tuple[ExternalTransferRequest, ...],
    external_images: dict[str, SparseMemoryImage],
    hbm_images: dict[str, SparseMemoryImage],
) -> ExternalTransferReport:
    """Execute deterministic payload copies through shared half-duplex links."""

    if type(fabric) is not ExternalMemoryFabric:
        raise SchemaError(
            "must be an ExternalMemoryFabric",
            path="fabric",
        )
    fabric.validate("fabric")
    _validate_images(
        capacities=fabric.external_capacities,
        images=external_images,
        path="external_images",
    )
    _validate_images(
        capacities=fabric.hbm_capacities,
        images=hbm_images,
        path="hbm_images",
    )

    ordered_requests = tuple(
        sorted(requests, key=lambda item: (item.issue_cycle, item.id))
    )
    if len({item.id for item in ordered_requests}) != len(ordered_requests):
        raise SchemaError("duplicate request", path="requests")
    link_by_id = {item.id: item for item in fabric.links}
    completions_by_link: dict[str, list[ExternalTransferCompletion]] = {
        item.id: [] for item in fabric.links
    }
    completions: list[ExternalTransferCompletion] = []

    for index, request in enumerate(ordered_requests):
        request.validate(f"requests[{index}]")
        connection, external_capacity, hbm_capacity = (
            _connection_and_capacities(fabric, request)
        )
        link = link_by_id[connection.link_ref]
        external_image = external_images[external_capacity.id]
        hbm_image = hbm_images[hbm_capacity.id]
        external_image._validate_range(
            request.external_address,
            request.size_bytes,
            f"requests[{index}].external_address",
        )
        hbm_image._validate_range(
            request.hbm_address,
            request.size_bytes,
            f"requests[{index}].hbm_address",
        )

        prior = completions_by_link[link.id]
        outstanding = sum(
            item.completion_cycle > request.issue_cycle for item in prior
        )
        if outstanding >= link.max_outstanding:
            raise SchemaError(
                "external link max_outstanding exhausted",
                path=f"requests[{index}]",
                code="external_queue_exhausted",
            )
        free_cycle = (
            prior[-1].completion_cycle if prior else request.issue_cycle
        )
        start_cycle = max(request.issue_cycle, free_cycle)
        queued_before = sum(
            item.start_cycle > request.issue_cycle for item in prior
        )
        if (
            start_cycle > request.issue_cycle
            and queued_before >= link.queue_depth
        ):
            raise SchemaError(
                "external link queue depth exhausted",
                path=f"requests[{index}]",
                code="external_queue_exhausted",
            )

        external_cycles = (
            link.latency_cycles
            + _ceil_div(request.size_bytes, link.bytes_per_cycle)
        )
        route_cycles = 0
        if len(connection.route_die_ids) > 1:
            assert connection.route_bytes_per_cycle is not None
            route_cycles = (
                connection.route_latency_cycles
                + _ceil_div(
                    request.size_bytes,
                    connection.route_bytes_per_cycle,
                )
            )
        completion_cycle = (
            start_cycle + external_cycles + route_cycles
        )

        if request.direction is ExternalTransferDirection.EXTERNAL_TO_HBM:
            payload = external_image.read(
                request.external_address,
                request.size_bytes,
            )
            hbm_image.write(request.hbm_address, payload)
        else:
            payload = hbm_image.read(
                request.hbm_address,
                request.size_bytes,
            )
            external_image.write(request.external_address, payload)
        completion = ExternalTransferCompletion.create(
            request_ref=request.id,
            link_ref=link.id,
            direction=request.direction,
            start_cycle=start_cycle,
            completion_cycle=completion_cycle,
            external_service_cycles=external_cycles,
            route_service_cycles=route_cycles,
            queue_stall_cycles=start_cycle - request.issue_cycle,
            payload_bytes=request.size_bytes,
            payload_digest=hashlib.sha256(payload).hexdigest(),
        )
        prior.append(completion)
        completions.append(completion)

    stats = recompute_external_transfer_stats(
        ordered_requests,
        tuple(completions),
    )
    return ExternalTransferReport.create(
        fabric=fabric,
        requests=ordered_requests,
        completions=tuple(completions),
        stats=stats,
    )


__all__ = [
    "SparseMemoryImage",
    "execute_external_transfers",
    "recompute_external_transfer_stats",
]
