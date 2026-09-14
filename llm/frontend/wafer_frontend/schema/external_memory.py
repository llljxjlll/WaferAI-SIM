"""Strict contracts for the first external-memory transfer service.

The service is a deterministic Python timing and payload model.  These
contracts do not imply that C++ NpuSim can execute external-memory requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import (
    UINT64_MAX,
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
)
from .memory_plan import MemoryTier, MemoryTierCapacity


EXTERNAL_LINK_SCHEMA_VERSION = "wafer_frontend.external_memory_link/v1alpha1"
EXTERNAL_CONNECTION_SCHEMA_VERSION = (
    "wafer_frontend.external_memory_connection/v1alpha1"
)
EXTERNAL_FABRIC_SCHEMA_VERSION = "wafer_frontend.external_memory_fabric/v1alpha2"
EXTERNAL_TRANSFER_REQUEST_SCHEMA_VERSION = (
    "wafer_frontend.external_transfer_request/v1alpha1"
)
EXTERNAL_TRANSFER_COMPLETION_SCHEMA_VERSION = (
    "wafer_frontend.external_transfer_completion/v1alpha1"
)
EXTERNAL_TRANSFER_STATS_SCHEMA_VERSION = (
    "wafer_frontend.external_transfer_stats/v1alpha1"
)
EXTERNAL_TRANSFER_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.external_transfer_report/v1alpha1"
)


class ExternalDuplexMode(str, Enum):
    HALF_DUPLEX_SHARED = "half_duplex_shared"


class ExternalTransferDirection(str, Enum):
    EXTERNAL_TO_HBM = "external_to_hbm"
    HBM_TO_EXTERNAL = "hbm_to_external"


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


def _stable_id(kind: str, key: object, version: str) -> str:
    return stable_artifact_id(kind, key, schema_version=version)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _validate_stable_id(
    *,
    actual: str,
    kind: str,
    key: object,
    version: str,
    path: str,
) -> None:
    expected = _stable_id(kind, key, version)
    if actual != expected:
        raise SchemaError(
            f"unstable artifact id; expected {expected!r}",
            path=f"{path}.id",
        )


@dataclass(frozen=True, slots=True)
class ExternalMemoryLink:
    id: str
    external_capacity_ref: str
    ingress_die_id: int
    bytes_per_cycle: int
    latency_cycles: int
    queue_depth: int
    max_outstanding: int
    duplex: ExternalDuplexMode

    @classmethod
    def create(
        cls,
        *,
        external_capacity_ref: str,
        ingress_die_id: int,
        bytes_per_cycle: int,
        latency_cycles: int,
        queue_depth: int,
        max_outstanding: int,
        duplex: ExternalDuplexMode = ExternalDuplexMode.HALF_DUPLEX_SHARED,
    ) -> "ExternalMemoryLink":
        key = {
            "external_capacity_ref": external_capacity_ref,
            "ingress_die_id": ingress_die_id,
            "bytes_per_cycle": bytes_per_cycle,
            "latency_cycles": latency_cycles,
            "queue_depth": queue_depth,
            "max_outstanding": max_outstanding,
            "duplex": duplex,
        }
        result = cls(
            id=_stable_id(
                "external_memory_link",
                key,
                EXTERNAL_LINK_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "external_capacity_ref": self.external_capacity_ref,
            "ingress_die_id": self.ingress_die_id,
            "bytes_per_cycle": self.bytes_per_cycle,
            "latency_cycles": self.latency_cycles,
            "queue_depth": self.queue_depth,
            "max_outstanding": self.max_outstanding,
            "duplex": self.duplex,
        }

    def validate(self, path: str = "external_memory_link") -> None:
        validate_nonempty(
            self.external_capacity_ref,
            f"{path}.external_capacity_ref",
        )
        validate_uint64(self.ingress_die_id, f"{path}.ingress_die_id")
        _positive(self.bytes_per_cycle, f"{path}.bytes_per_cycle")
        validate_uint64(self.latency_cycles, f"{path}.latency_cycles")
        _positive(self.queue_depth, f"{path}.queue_depth")
        _positive(self.max_outstanding, f"{path}.max_outstanding")
        if self.max_outstanding > self.queue_depth + 1:
            raise SchemaError(
                "must not exceed queue_depth + one active request",
                path=f"{path}.max_outstanding",
            )
        if self.duplex is not ExternalDuplexMode.HALF_DUPLEX_SHARED:
            raise SchemaError(
                "only half-duplex shared service is supported",
                path=f"{path}.duplex",
            )
        _validate_stable_id(
            actual=self.id,
            kind="external_memory_link",
            key=self._key(),
            version=EXTERNAL_LINK_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class ExternalMemoryConnection:
    id: str
    link_ref: str
    hbm_capacity_ref: str
    target_die_id: int
    route_die_ids: tuple[int, ...]
    route_latency_cycles: int
    route_bytes_per_cycle: int | None

    @classmethod
    def create(
        cls,
        *,
        link_ref: str,
        hbm_capacity_ref: str,
        target_die_id: int,
        route_die_ids: tuple[int, ...],
        route_latency_cycles: int,
        route_bytes_per_cycle: int | None,
    ) -> "ExternalMemoryConnection":
        key = {
            "link_ref": link_ref,
            "hbm_capacity_ref": hbm_capacity_ref,
            "target_die_id": target_die_id,
            "route_die_ids": route_die_ids,
            "route_latency_cycles": route_latency_cycles,
            "route_bytes_per_cycle": route_bytes_per_cycle,
        }
        result = cls(
            id=_stable_id(
                "external_memory_connection",
                key,
                EXTERNAL_CONNECTION_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "link_ref": self.link_ref,
            "hbm_capacity_ref": self.hbm_capacity_ref,
            "target_die_id": self.target_die_id,
            "route_die_ids": self.route_die_ids,
            "route_latency_cycles": self.route_latency_cycles,
            "route_bytes_per_cycle": self.route_bytes_per_cycle,
        }

    def validate(self, path: str = "external_memory_connection") -> None:
        validate_nonempty(self.link_ref, f"{path}.link_ref")
        validate_nonempty(self.hbm_capacity_ref, f"{path}.hbm_capacity_ref")
        validate_uint64(self.target_die_id, f"{path}.target_die_id")
        if type(self.route_die_ids) is not tuple or not self.route_die_ids:
            raise SchemaError(
                "must be a non-empty immutable tuple",
                path=f"{path}.route_die_ids",
            )
        for index, die_id in enumerate(self.route_die_ids):
            validate_uint64(die_id, f"{path}.route_die_ids[{index}]")
        if len(set(self.route_die_ids)) != len(self.route_die_ids):
            raise SchemaError(
                "route must be simple and cannot repeat a die",
                path=f"{path}.route_die_ids",
            )
        if self.route_die_ids[-1] != self.target_die_id:
            raise SchemaError(
                "route must end at target_die_id",
                path=f"{path}.route_die_ids",
            )
        validate_uint64(
            self.route_latency_cycles,
            f"{path}.route_latency_cycles",
        )
        if self.route_bytes_per_cycle is not None:
            _positive(
                self.route_bytes_per_cycle,
                f"{path}.route_bytes_per_cycle",
            )
        _validate_stable_id(
            actual=self.id,
            kind="external_memory_connection",
            key=self._key(),
            version=EXTERNAL_CONNECTION_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class ExternalMemoryFabric:
    schema_version: str
    producer_pass: str
    id: str
    external_capacities: tuple[MemoryTierCapacity, ...]
    hbm_capacities: tuple[MemoryTierCapacity, ...]
    links: tuple[ExternalMemoryLink, ...]
    connections: tuple[ExternalMemoryConnection, ...]

    @classmethod
    def create(
        cls,
        *,
        external_capacities: tuple[MemoryTierCapacity, ...],
        hbm_capacities: tuple[MemoryTierCapacity, ...],
        links: tuple[ExternalMemoryLink, ...],
        connections: tuple[ExternalMemoryConnection, ...],
    ) -> "ExternalMemoryFabric":
        external_capacities = tuple(
            sorted(external_capacities, key=lambda item: item.id)
        )
        hbm_capacities = tuple(
            sorted(hbm_capacities, key=lambda item: item.id)
        )
        links = tuple(sorted(links, key=lambda item: item.id))
        connections = tuple(
            sorted(connections, key=lambda item: item.id)
        )
        key = {
            "external_capacities": external_capacities,
            "hbm_capacities": hbm_capacities,
            "links": links,
            "connections": connections,
        }
        result = cls(
            schema_version=EXTERNAL_FABRIC_SCHEMA_VERSION,
            producer_pass="external_memory_fabric_builder",
            id=_stable_id(
                "external_memory_fabric",
                key,
                EXTERNAL_FABRIC_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "external_capacities": self.external_capacities,
            "hbm_capacities": self.hbm_capacities,
            "links": self.links,
            "connections": self.connections,
        }

    @staticmethod
    def _index(
        items: tuple[object, ...],
        *,
        path: str,
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for index, item in enumerate(items):
            validator = getattr(item, "validate", None)
            if validator is None:
                raise SchemaError("unsupported item type", path=f"{path}[{index}]")
            validator(f"{path}[{index}]")
            item_id = getattr(item, "id", None)
            if item_id in result:
                raise SchemaError("duplicate id", path=f"{path}[{index}].id")
            result[item_id] = item
        return result

    @staticmethod
    def _validate_nonoverlap(
        capacities: tuple[MemoryTierCapacity, ...],
        *,
        path: str,
    ) -> None:
        ranges_by_owner: dict[
            tuple[MemoryTier, str],
            list[tuple[int, int]],
        ] = {}
        for index, capacity in enumerate(capacities):
            start = capacity.base_address
            end = start + capacity.capacity_bytes
            owner = (capacity.tier, capacity.location_ref)
            ranges = ranges_by_owner.setdefault(owner, [])
            if any(
                start < old_end and old_start < end
                for old_start, old_end in ranges
            ):
                raise SchemaError(
                    "address spaces for one typed owner must not overlap",
                    path=f"{path}[{index}]",
                )
            ranges.append((start, end))

    @staticmethod
    def _hbm_owner_die_id(
        capacity: MemoryTierCapacity,
        *,
        path: str,
    ) -> int:
        prefix = "die:"
        if not capacity.location_ref.startswith(prefix):
            raise SchemaError(
                "HBM owner must use canonical die:<id> form",
                path=f"{path}.location_ref",
            )
        raw_die_id = capacity.location_ref[len(prefix) :]
        if not raw_die_id.isascii() or not raw_die_id.isdecimal():
            raise SchemaError(
                "HBM owner must use canonical die:<id> form",
                path=f"{path}.location_ref",
            )
        die_id = int(raw_die_id)
        validate_uint64(die_id, f"{path}.location_ref")
        if raw_die_id != str(die_id):
            raise SchemaError(
                "HBM owner die id must use canonical decimal form",
                path=f"{path}.location_ref",
            )
        return die_id

    def validate(self, path: str = "external_memory_fabric") -> None:
        if self.schema_version != EXTERNAL_FABRIC_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "external_memory_fabric_builder":
            raise SchemaError(
                "must be 'external_memory_fabric_builder'",
                path=f"{path}.producer_pass",
            )
        for name in (
            "external_capacities",
            "hbm_capacities",
            "links",
            "connections",
        ):
            items = getattr(self, name)
            if not items:
                raise SchemaError(
                    "must not be empty",
                    path=f"{path}.{name}",
                )
            if items != tuple(sorted(items, key=lambda item: item.id)):
                raise SchemaError(
                    "must use canonical id order",
                    path=f"{path}.{name}",
                )
        external = self._index(
            self.external_capacities,
            path=f"{path}.external_capacities",
        )
        hbm = self._index(
            self.hbm_capacities,
            path=f"{path}.hbm_capacities",
        )
        links = self._index(self.links, path=f"{path}.links")
        connections = self._index(
            self.connections,
            path=f"{path}.connections",
        )
        if any(
            capacity.tier is not MemoryTier.EXTERNAL
            for capacity in self.external_capacities
        ):
            raise SchemaError(
                "all capacities must use EXTERNAL tier",
                path=f"{path}.external_capacities",
            )
        if any(
            capacity.tier is not MemoryTier.HBM
            for capacity in self.hbm_capacities
        ):
            raise SchemaError(
                "all capacities must use HBM tier",
                path=f"{path}.hbm_capacities",
            )
        self._validate_nonoverlap(
            self.external_capacities,
            path=f"{path}.external_capacities",
        )
        self._validate_nonoverlap(
            self.hbm_capacities,
            path=f"{path}.hbm_capacities",
        )

        link_by_backing: dict[str, ExternalMemoryLink] = {}
        for link in self.links:
            if link.external_capacity_ref not in external:
                raise SchemaError(
                    "link references an unknown external capacity",
                    path=f"{path}.links",
                )
            if link.external_capacity_ref in link_by_backing:
                raise SchemaError(
                    "external capacity has multiple service links",
                    path=f"{path}.links",
                )
            link_by_backing[link.external_capacity_ref] = link
        if set(link_by_backing) != set(external):
            raise SchemaError(
                "every external capacity requires one service link",
                path=f"{path}.links",
            )

        seen_targets: set[tuple[str, int]] = set()
        directly_connected_links: set[str] = set()
        connected_hbm: set[str] = set()
        for connection in self.connections:
            link = links.get(connection.link_ref)
            if not isinstance(link, ExternalMemoryLink):
                raise SchemaError(
                    "connection references an unknown link",
                    path=f"{path}.connections",
                )
            hbm_capacity = hbm.get(connection.hbm_capacity_ref)
            if not isinstance(hbm_capacity, MemoryTierCapacity):
                raise SchemaError(
                    "connection references an unknown HBM capacity",
                    path=f"{path}.connections",
                )
            owner_die_id = self._hbm_owner_die_id(
                hbm_capacity,
                path=(
                    f"{path}.hbm_capacities"
                    f"[{self.hbm_capacities.index(hbm_capacity)}]"
                ),
            )
            if connection.target_die_id != owner_die_id:
                raise SchemaError(
                    "connection target die does not own the HBM capacity",
                    path=f"{path}.connections",
                )
            target_key = (connection.link_ref, connection.target_die_id)
            if target_key in seen_targets:
                raise SchemaError(
                    "duplicate link/target connection",
                    path=f"{path}.connections",
                )
            seen_targets.add(target_key)
            connected_hbm.add(connection.hbm_capacity_ref)
            if connection.target_die_id == link.ingress_die_id:
                if (
                    connection.route_die_ids != (link.ingress_die_id,)
                    or connection.route_latency_cycles != 0
                    or connection.route_bytes_per_cycle is not None
                ):
                    raise SchemaError(
                        "direct connection must use a one-die zero-cost route",
                        path=f"{path}.connections",
                    )
                directly_connected_links.add(link.id)
            else:
                if (
                    len(connection.route_die_ids) < 2
                    or connection.route_die_ids[0] != link.ingress_die_id
                    or connection.route_latency_cycles == 0
                    or connection.route_bytes_per_cycle is None
                ):
                    raise SchemaError(
                        "non-direct connection requires explicit nonzero route metadata",
                        path=f"{path}.connections",
                    )
        if directly_connected_links != set(links):
            raise SchemaError(
                "every link requires its ingress-die connection",
                path=f"{path}.connections",
            )
        if connected_hbm != set(hbm):
            raise SchemaError(
                "every HBM capacity requires an external connection",
                path=f"{path}.connections",
            )
        _validate_stable_id(
            actual=self.id,
            kind="external_memory_fabric",
            key=self._key(),
            version=EXTERNAL_FABRIC_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class ExternalTransferRequest:
    id: str
    connection_ref: str
    direction: ExternalTransferDirection
    external_address: int
    hbm_address: int
    size_bytes: int
    issue_cycle: int

    @classmethod
    def create(
        cls,
        *,
        connection_ref: str,
        direction: ExternalTransferDirection,
        external_address: int,
        hbm_address: int,
        size_bytes: int,
        issue_cycle: int,
    ) -> "ExternalTransferRequest":
        key = {
            "connection_ref": connection_ref,
            "direction": direction,
            "external_address": external_address,
            "hbm_address": hbm_address,
            "size_bytes": size_bytes,
            "issue_cycle": issue_cycle,
        }
        result = cls(
            id=_stable_id(
                "external_transfer_request",
                key,
                EXTERNAL_TRANSFER_REQUEST_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "connection_ref": self.connection_ref,
            "direction": self.direction,
            "external_address": self.external_address,
            "hbm_address": self.hbm_address,
            "size_bytes": self.size_bytes,
            "issue_cycle": self.issue_cycle,
        }

    def validate(self, path: str = "external_transfer_request") -> None:
        validate_nonempty(self.connection_ref, f"{path}.connection_ref")
        if type(self.direction) is not ExternalTransferDirection:
            raise SchemaError(
                "must be an ExternalTransferDirection",
                path=f"{path}.direction",
            )
        validate_uint64(self.external_address, f"{path}.external_address")
        validate_uint64(self.hbm_address, f"{path}.hbm_address")
        _positive(self.size_bytes, f"{path}.size_bytes")
        validate_uint64(self.issue_cycle, f"{path}.issue_cycle")
        for name, start in (
            ("external_address", self.external_address),
            ("hbm_address", self.hbm_address),
        ):
            if start > UINT64_MAX - self.size_bytes:
                raise SchemaError(
                    "transfer range overflows uint64",
                    path=f"{path}.{name}",
                )
        _validate_stable_id(
            actual=self.id,
            kind="external_transfer_request",
            key=self._key(),
            version=EXTERNAL_TRANSFER_REQUEST_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class ExternalTransferCompletion:
    id: str
    request_ref: str
    link_ref: str
    direction: ExternalTransferDirection
    start_cycle: int
    completion_cycle: int
    external_service_cycles: int
    route_service_cycles: int
    queue_stall_cycles: int
    payload_bytes: int
    payload_digest: str

    @classmethod
    def create(
        cls,
        *,
        request_ref: str,
        link_ref: str,
        direction: ExternalTransferDirection,
        start_cycle: int,
        completion_cycle: int,
        external_service_cycles: int,
        route_service_cycles: int,
        queue_stall_cycles: int,
        payload_bytes: int,
        payload_digest: str,
    ) -> "ExternalTransferCompletion":
        key = {
            "request_ref": request_ref,
            "link_ref": link_ref,
            "direction": direction,
            "start_cycle": start_cycle,
            "completion_cycle": completion_cycle,
            "external_service_cycles": external_service_cycles,
            "route_service_cycles": route_service_cycles,
            "queue_stall_cycles": queue_stall_cycles,
            "payload_bytes": payload_bytes,
            "payload_digest": payload_digest,
        }
        result = cls(
            id=_stable_id(
                "external_transfer_completion",
                key,
                EXTERNAL_TRANSFER_COMPLETION_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "request_ref": self.request_ref,
            "link_ref": self.link_ref,
            "direction": self.direction,
            "start_cycle": self.start_cycle,
            "completion_cycle": self.completion_cycle,
            "external_service_cycles": self.external_service_cycles,
            "route_service_cycles": self.route_service_cycles,
            "queue_stall_cycles": self.queue_stall_cycles,
            "payload_bytes": self.payload_bytes,
            "payload_digest": self.payload_digest,
        }

    def validate(self, path: str = "external_transfer_completion") -> None:
        validate_nonempty(self.request_ref, f"{path}.request_ref")
        validate_nonempty(self.link_ref, f"{path}.link_ref")
        if type(self.direction) is not ExternalTransferDirection:
            raise SchemaError(
                "must be an ExternalTransferDirection",
                path=f"{path}.direction",
            )
        for name in (
            "start_cycle",
            "completion_cycle",
            "external_service_cycles",
            "route_service_cycles",
            "queue_stall_cycles",
            "payload_bytes",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.completion_cycle <= self.start_cycle:
            raise SchemaError(
                "must be greater than start_cycle",
                path=f"{path}.completion_cycle",
            )
        _positive(self.external_service_cycles, f"{path}.external_service_cycles")
        _positive(self.payload_bytes, f"{path}.payload_bytes")
        if (
            len(self.payload_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.payload_digest)
        ):
            raise SchemaError(
                "must be a lowercase SHA-256 hex digest",
                path=f"{path}.payload_digest",
            )
        _validate_stable_id(
            actual=self.id,
            kind="external_transfer_completion",
            key=self._key(),
            version=EXTERNAL_TRANSFER_COMPLETION_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class ExternalTransferStats:
    id: str
    submitted_requests: int
    completed_requests: int
    external_read_bytes: int
    external_write_bytes: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    shared_link_busy_cycles: int
    queue_stall_cycles: int
    max_queue_occupancy: int
    max_outstanding_observed: int
    makespan_cycles: int
    pending_requests: int

    @classmethod
    def create(cls, **values: int) -> "ExternalTransferStats":
        key = dict(values)
        result = cls(
            id=_stable_id(
                "external_transfer_stats",
                key,
                EXTERNAL_TRANSFER_STATS_SCHEMA_VERSION,
            ),
            **values,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "external_transfer_stats") -> None:
        for name in self.__dataclass_fields__:
            if name != "id":
                validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.completed_requests > self.submitted_requests:
            raise SchemaError(
                "cannot exceed submitted_requests",
                path=f"{path}.completed_requests",
            )
        if (
            self.pending_requests
            != self.submitted_requests - self.completed_requests
        ):
            raise SchemaError(
                "must equal submitted minus completed",
                path=f"{path}.pending_requests",
            )
        _validate_stable_id(
            actual=self.id,
            kind="external_transfer_stats",
            key=self._key(),
            version=EXTERNAL_TRANSFER_STATS_SCHEMA_VERSION,
            path=path,
        )


def build_external_transfer_stats(
    requests: tuple[ExternalTransferRequest, ...],
    completions: tuple[ExternalTransferCompletion, ...],
) -> ExternalTransferStats:
    completion_by_request = {
        completion.request_ref: completion for completion in completions
    }
    if set(completion_by_request) != {request.id for request in requests}:
        raise SchemaError(
            "completions must exactly cover requests",
            path="completions",
        )
    external_read_bytes = 0
    external_write_bytes = 0
    hbm_read_bytes = 0
    hbm_write_bytes = 0
    max_outstanding = 0
    max_queue = 0
    prior: list[ExternalTransferCompletion] = []
    for request in requests:
        completion = completion_by_request[request.id]
        if request.direction is ExternalTransferDirection.EXTERNAL_TO_HBM:
            external_read_bytes += request.size_bytes
            hbm_write_bytes += request.size_bytes
        else:
            hbm_read_bytes += request.size_bytes
            external_write_bytes += request.size_bytes
        same_link = [
            item for item in prior if item.link_ref == completion.link_ref
        ]
        outstanding = (
            sum(
                item.completion_cycle > request.issue_cycle
                for item in same_link
            )
            + 1
        )
        queued = (
            sum(
                item.start_cycle > request.issue_cycle
                for item in same_link
            )
            + int(completion.start_cycle > request.issue_cycle)
        )
        max_outstanding = max(max_outstanding, outstanding)
        max_queue = max(max_queue, queued)
        prior.append(completion)
    makespan = 0
    if requests:
        makespan = max(
            completion.completion_cycle for completion in completions
        ) - min(request.issue_cycle for request in requests)
    return ExternalTransferStats.create(
        submitted_requests=len(requests),
        completed_requests=len(completions),
        external_read_bytes=external_read_bytes,
        external_write_bytes=external_write_bytes,
        hbm_read_bytes=hbm_read_bytes,
        hbm_write_bytes=hbm_write_bytes,
        shared_link_busy_cycles=sum(
            item.external_service_cycles + item.route_service_cycles
            for item in completions
        ),
        queue_stall_cycles=sum(
            item.queue_stall_cycles for item in completions
        ),
        max_queue_occupancy=max_queue,
        max_outstanding_observed=max_outstanding,
        makespan_cycles=makespan,
        pending_requests=len(requests) - len(completions),
    )


@dataclass(frozen=True, slots=True)
class ExternalTransferReport:
    schema_version: str
    producer_pass: str
    id: str
    fabric: ExternalMemoryFabric
    requests: tuple[ExternalTransferRequest, ...]
    completions: tuple[ExternalTransferCompletion, ...]
    stats: ExternalTransferStats
    simulator_runtime_integrated: bool

    @classmethod
    def create(
        cls,
        *,
        fabric: ExternalMemoryFabric,
        requests: tuple[ExternalTransferRequest, ...],
        completions: tuple[ExternalTransferCompletion, ...],
        stats: ExternalTransferStats,
    ) -> "ExternalTransferReport":
        requests = tuple(
            sorted(requests, key=lambda item: (item.issue_cycle, item.id))
        )
        completions = tuple(
            sorted(completions, key=lambda item: (item.start_cycle, item.id))
        )
        key = {
            "fabric": fabric,
            "requests": requests,
            "completions": completions,
            "stats": stats,
            "simulator_runtime_integrated": False,
        }
        result = cls(
            schema_version=EXTERNAL_TRANSFER_REPORT_SCHEMA_VERSION,
            producer_pass="external_transfer_service",
            id=_stable_id(
                "external_transfer_report",
                key,
                EXTERNAL_TRANSFER_REPORT_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "fabric": self.fabric,
            "requests": self.requests,
            "completions": self.completions,
            "stats": self.stats,
            "simulator_runtime_integrated": self.simulator_runtime_integrated,
        }

    def validate(self, path: str = "external_transfer_report") -> None:
        if self.schema_version != EXTERNAL_TRANSFER_REPORT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "external_transfer_service":
            raise SchemaError(
                "must be 'external_transfer_service'",
                path=f"{path}.producer_pass",
            )
        if self.simulator_runtime_integrated is not False:
            raise SchemaError(
                "Python service is not integrated with simulator runtime",
                path=f"{path}.simulator_runtime_integrated",
            )
        self.fabric.validate(f"{path}.fabric")
        self.stats.validate(f"{path}.stats")
        if self.requests != tuple(
            sorted(self.requests, key=lambda item: (item.issue_cycle, item.id))
        ):
            raise SchemaError(
                "must use issue-cycle/id order",
                path=f"{path}.requests",
            )
        if self.completions != tuple(
            sorted(self.completions, key=lambda item: (item.start_cycle, item.id))
        ):
            raise SchemaError(
                "must use start-cycle/id order",
                path=f"{path}.completions",
            )
        request_by_id: dict[str, ExternalTransferRequest] = {}
        link_by_id = {item.id: item for item in self.fabric.links}
        connection_by_id = {
            item.id: item for item in self.fabric.connections
        }
        external_by_id = {
            item.id: item for item in self.fabric.external_capacities
        }
        hbm_by_id = {
            item.id: item for item in self.fabric.hbm_capacities
        }
        for index, request in enumerate(self.requests):
            request.validate(f"{path}.requests[{index}]")
            if request.id in request_by_id:
                raise SchemaError(
                    "duplicate request",
                    path=f"{path}.requests[{index}].id",
                )
            connection = connection_by_id.get(request.connection_ref)
            if connection is None:
                raise SchemaError(
                    "request references an unknown connection",
                    path=f"{path}.requests[{index}].connection_ref",
                )
            link = link_by_id[connection.link_ref]
            external = external_by_id[link.external_capacity_ref]
            hbm = hbm_by_id[connection.hbm_capacity_ref]
            for name, address, capacity in (
                ("external_address", request.external_address, external),
                ("hbm_address", request.hbm_address, hbm),
            ):
                if (
                    address < capacity.base_address
                    or address + request.size_bytes
                    > capacity.base_address + capacity.capacity_bytes
                ):
                    raise SchemaError(
                        "transfer exceeds configured capacity",
                        path=f"{path}.requests[{index}].{name}",
                    )
            request_by_id[request.id] = request
        completion_by_request: dict[str, ExternalTransferCompletion] = {}
        free_cycle_by_link: dict[str, int] = {}
        for index, completion in enumerate(self.completions):
            completion.validate(f"{path}.completions[{index}]")
            request = request_by_id.get(completion.request_ref)
            if request is None:
                raise SchemaError(
                    "completion references an unknown request",
                    path=f"{path}.completions[{index}].request_ref",
                )
            if request.id in completion_by_request:
                raise SchemaError(
                    "request has duplicate completion",
                    path=f"{path}.completions[{index}]",
                )
            connection = connection_by_id.get(request.connection_ref)
            if connection is None:
                raise SchemaError(
                    "request references an unknown connection",
                    path=f"{path}.requests",
                )
            link = link_by_id[connection.link_ref]
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
            start = max(
                request.issue_cycle,
                free_cycle_by_link.get(link.id, request.issue_cycle),
            )
            completion_cycle = start + external_cycles + route_cycles
            expected = (
                link.id,
                request.direction,
                start,
                completion_cycle,
                external_cycles,
                route_cycles,
                start - request.issue_cycle,
                request.size_bytes,
            )
            actual = (
                completion.link_ref,
                completion.direction,
                completion.start_cycle,
                completion.completion_cycle,
                completion.external_service_cycles,
                completion.route_service_cycles,
                completion.queue_stall_cycles,
                completion.payload_bytes,
            )
            if actual != expected:
                raise SchemaError(
                    "completion does not match deterministic service timing",
                    path=f"{path}.completions[{index}]",
                )
            free_cycle_by_link[link.id] = completion_cycle
            completion_by_request[request.id] = completion
        if set(completion_by_request) != set(request_by_id):
            raise SchemaError(
                "completions must exactly cover requests",
                path=f"{path}.completions",
            )
        prior_by_link: dict[str, list[ExternalTransferCompletion]] = {
            item.id: [] for item in self.fabric.links
        }
        for index, request in enumerate(self.requests):
            completion = completion_by_request[request.id]
            link = link_by_id[completion.link_ref]
            prior = prior_by_link[link.id]
            outstanding = (
                sum(
                    item.completion_cycle > request.issue_cycle
                    for item in prior
                )
                + 1
            )
            queued = (
                sum(
                    item.start_cycle > request.issue_cycle
                    for item in prior
                )
                + int(completion.start_cycle > request.issue_cycle)
            )
            if outstanding > link.max_outstanding:
                raise SchemaError(
                    "completion exceeds link max_outstanding",
                    path=f"{path}.completions[{index}]",
                )
            if queued > link.queue_depth:
                raise SchemaError(
                    "completion exceeds link queue_depth",
                    path=f"{path}.completions[{index}]",
                )
            prior.append(completion)
        expected_stats = build_external_transfer_stats(
            self.requests,
            self.completions,
        )
        if self.stats != expected_stats:
            raise SchemaError(
                "does not match recomputed transfer statistics",
                path=f"{path}.stats",
            )
        _validate_stable_id(
            actual=self.id,
            kind="external_transfer_report",
            key=self._key(),
            version=EXTERNAL_TRANSFER_REPORT_SCHEMA_VERSION,
            path=path,
        )


__all__ = [
    "EXTERNAL_CONNECTION_SCHEMA_VERSION",
    "EXTERNAL_FABRIC_SCHEMA_VERSION",
    "EXTERNAL_LINK_SCHEMA_VERSION",
    "EXTERNAL_TRANSFER_COMPLETION_SCHEMA_VERSION",
    "EXTERNAL_TRANSFER_REPORT_SCHEMA_VERSION",
    "EXTERNAL_TRANSFER_REQUEST_SCHEMA_VERSION",
    "EXTERNAL_TRANSFER_STATS_SCHEMA_VERSION",
    "ExternalDuplexMode",
    "ExternalMemoryConnection",
    "ExternalMemoryFabric",
    "ExternalMemoryLink",
    "ExternalTransferCompletion",
    "ExternalTransferDirection",
    "ExternalTransferReport",
    "ExternalTransferRequest",
    "ExternalTransferStats",
    "build_external_transfer_stats",
]
