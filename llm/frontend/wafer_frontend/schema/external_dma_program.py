"""Versioned sidecar contract for executing a blocking offload transfer plan."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import (
    UINT64_MAX,
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
)
from .external_memory import (
    ExternalMemoryFabric,
    ExternalTransferDirection,
)
from .serde import canonical_digest


EXTERNAL_DMA_REQUEST_SCHEMA_VERSION = (
    "npusim.external_dma_request/v1alpha1"
)
EXTERNAL_DMA_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.external_dma_backend_binding/v1alpha1"
)
EXTERNAL_DMA_DESCRIPTOR_SCHEMA_VERSION = (
    "wafer_frontend.external_dma_descriptor/v1alpha1"
)
EXTERNAL_DMA_SEED_SCHEMA_VERSION = (
    "wafer_frontend.external_dma_seed/v1alpha1"
)
EXTERNAL_DMA_PROBE_SCHEMA_VERSION = (
    "wafer_frontend.external_dma_probe/v1alpha1"
)
EXTERNAL_DMA_PROGRAM_SCHEMA_VERSION = (
    "wafer_frontend.external_dma_program/v1alpha1"
)


def _stable_id(kind: str, key: object, version: str) -> str:
    return stable_artifact_id(kind, key, schema_version=version)


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _hex_payload(value: str, path: str) -> int:
    if type(value) is not str or not value or len(value) % 2:
        raise SchemaError(
            "must be a non-empty even-length hex payload",
            path=path,
        )
    if any(character not in "0123456789abcdef" for character in value):
        raise SchemaError("must use lowercase hexadecimal", path=path)
    return len(value) // 2


def _range(address: int, size_bytes: int, path: str) -> None:
    validate_uint64(address, f"{path}.address")
    validate_uint64(size_bytes, f"{path}.size_bytes")
    if size_bytes == 0:
        raise SchemaError("must be greater than zero", path=path)
    if address > UINT64_MAX - size_bytes:
        raise SchemaError("byte range overflows uint64", path=path)


def _require_id(
    actual: str,
    *,
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
class ExternalDmaBackendBinding:
    id: str
    hbm_capacity_ref: str
    owner_die_id: int
    stack_id: int
    channel_id: int

    @classmethod
    def create(
        cls,
        *,
        hbm_capacity_ref: str,
        owner_die_id: int,
        stack_id: int,
        channel_id: int,
    ) -> "ExternalDmaBackendBinding":
        key = {
            "hbm_capacity_ref": hbm_capacity_ref,
            "owner_die_id": owner_die_id,
            "stack_id": stack_id,
            "channel_id": channel_id,
        }
        result = cls(
            id=_stable_id(
                "external_dma_backend_binding",
                key,
                EXTERNAL_DMA_BINDING_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "external_dma_backend_binding") -> None:
        validate_nonempty(
            self.hbm_capacity_ref,
            f"{path}.hbm_capacity_ref",
        )
        for name in ("owner_die_id", "stack_id", "channel_id"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        for name in ("stack_id", "channel_id"):
            if getattr(self, name) > (1 << 31) - 1:
                raise SchemaError(
                    "must fit the HBMRuntime endpoint integer range",
                    path=f"{path}.{name}",
                )
        _require_id(
            self.id,
            kind="external_dma_backend_binding",
            key=self._key(),
            version=EXTERNAL_DMA_BINDING_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class ExternalDmaDescriptor:
    id: str
    sequence: int
    operation_ref: str
    source_transfer_request_ref: str
    source_operation_deps: tuple[str, ...]
    depends_on: tuple[str, ...]
    request_schema_version: str
    connection_ref: str
    direction: ExternalTransferDirection
    external_address: int
    hbm_address: int
    size_bytes: int
    planned_issue_cycle: int
    planned_ready_cycle: int

    @classmethod
    def create(
        cls,
        *,
        sequence: int,
        operation_ref: str,
        source_transfer_request_ref: str,
        source_operation_deps: tuple[str, ...],
        depends_on: tuple[str, ...],
        connection_ref: str,
        direction: ExternalTransferDirection,
        external_address: int,
        hbm_address: int,
        size_bytes: int,
        planned_issue_cycle: int,
        planned_ready_cycle: int,
    ) -> "ExternalDmaDescriptor":
        key = {
            "sequence": sequence,
            "operation_ref": operation_ref,
            "source_transfer_request_ref": source_transfer_request_ref,
            "source_operation_deps": source_operation_deps,
            "depends_on": depends_on,
            "request_schema_version": (
                EXTERNAL_DMA_REQUEST_SCHEMA_VERSION
            ),
            "connection_ref": connection_ref,
            "direction": direction,
            "external_address": external_address,
            "hbm_address": hbm_address,
            "size_bytes": size_bytes,
            "planned_issue_cycle": planned_issue_cycle,
            "planned_ready_cycle": planned_ready_cycle,
        }
        result = cls(
            id=_stable_id(
                "external_dma_descriptor",
                key,
                EXTERNAL_DMA_DESCRIPTOR_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "external_dma_descriptor") -> None:
        validate_uint64(self.sequence, f"{path}.sequence")
        validate_nonempty(self.operation_ref, f"{path}.operation_ref")
        validate_nonempty(
            self.source_transfer_request_ref,
            f"{path}.source_transfer_request_ref",
        )
        for name in ("source_operation_deps", "depends_on"):
            refs = getattr(self, name)
            if type(refs) is not tuple or len(set(refs)) != len(refs):
                raise SchemaError(
                    "must be an immutable tuple without duplicates",
                    path=f"{path}.{name}",
                )
            for index, value in enumerate(refs):
                validate_nonempty(value, f"{path}.{name}[{index}]")
        if (
            self.request_schema_version
            != EXTERNAL_DMA_REQUEST_SCHEMA_VERSION
        ):
            raise SchemaError(
                "unsupported request schema version",
                path=f"{path}.request_schema_version",
            )
        validate_nonempty(self.connection_ref, f"{path}.connection_ref")
        if type(self.direction) is not ExternalTransferDirection:
            raise SchemaError(
                "must be an ExternalTransferDirection",
                path=f"{path}.direction",
            )
        _range(self.external_address, self.size_bytes, path)
        _range(self.hbm_address, self.size_bytes, path)
        validate_uint64(
            self.planned_issue_cycle,
            f"{path}.planned_issue_cycle",
        )
        validate_uint64(
            self.planned_ready_cycle,
            f"{path}.planned_ready_cycle",
        )
        if self.planned_ready_cycle <= self.planned_issue_cycle:
            raise SchemaError(
                "must follow planned_issue_cycle",
                path=f"{path}.planned_ready_cycle",
            )
        _require_id(
            self.id,
            kind="external_dma_descriptor",
            key=self._key(),
            version=EXTERNAL_DMA_DESCRIPTOR_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class ExternalDmaSeed:
    id: str
    external_capacity_ref: str
    address: int
    payload_hex: str

    @classmethod
    def create(
        cls,
        *,
        external_capacity_ref: str,
        address: int,
        payload: bytes,
    ) -> "ExternalDmaSeed":
        if type(payload) is not bytes or not payload:
            raise SchemaError("must be non-empty bytes", path="payload")
        key = {
            "external_capacity_ref": external_capacity_ref,
            "address": address,
            "payload_hex": payload.hex(),
        }
        result = cls(
            id=_stable_id(
                "external_dma_seed",
                key,
                EXTERNAL_DMA_SEED_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "external_dma_seed") -> None:
        validate_nonempty(
            self.external_capacity_ref,
            f"{path}.external_capacity_ref",
        )
        size = _hex_payload(self.payload_hex, f"{path}.payload_hex")
        _range(self.address, size, path)
        _require_id(
            self.id,
            kind="external_dma_seed",
            key=self._key(),
            version=EXTERNAL_DMA_SEED_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class ExternalDmaProbe:
    id: str
    external_capacity_ref: str
    address: int
    expected_payload_hex: str

    @classmethod
    def create(
        cls,
        *,
        external_capacity_ref: str,
        address: int,
        expected_payload: bytes,
    ) -> "ExternalDmaProbe":
        if type(expected_payload) is not bytes or not expected_payload:
            raise SchemaError(
                "must be non-empty bytes",
                path="expected_payload",
            )
        key = {
            "external_capacity_ref": external_capacity_ref,
            "address": address,
            "expected_payload_hex": expected_payload.hex(),
        }
        result = cls(
            id=_stable_id(
                "external_dma_probe",
                key,
                EXTERNAL_DMA_PROBE_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "external_dma_probe") -> None:
        validate_nonempty(
            self.external_capacity_ref,
            f"{path}.external_capacity_ref",
        )
        size = _hex_payload(
            self.expected_payload_hex,
            f"{path}.expected_payload_hex",
        )
        _range(self.address, size, path)
        _require_id(
            self.id,
            kind="external_dma_probe",
            key=self._key(),
            version=EXTERNAL_DMA_PROBE_SCHEMA_VERSION,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class ExternalDmaProgram:
    schema_version: str
    producer_pass: str
    id: str
    case_digest: str
    request_digest: str
    logical_graph_digest: str
    source_memory_plan_digest: str
    blocking_offload_plan_id: str
    blocking_offload_plan_digest: str
    fabric: ExternalMemoryFabric
    backend_bindings: tuple[ExternalDmaBackendBinding, ...]
    descriptors: tuple[ExternalDmaDescriptor, ...]
    external_seeds: tuple[ExternalDmaSeed, ...]
    external_probes: tuple[ExternalDmaProbe, ...]

    @classmethod
    def create(
        cls,
        *,
        case_digest: str,
        request_digest: str,
        logical_graph_digest: str,
        source_memory_plan_digest: str,
        blocking_offload_plan_id: str,
        blocking_offload_plan_digest: str,
        fabric: ExternalMemoryFabric,
        backend_bindings: tuple[ExternalDmaBackendBinding, ...],
        descriptors: tuple[ExternalDmaDescriptor, ...],
        external_seeds: tuple[ExternalDmaSeed, ...],
        external_probes: tuple[ExternalDmaProbe, ...],
    ) -> "ExternalDmaProgram":
        backend_bindings = tuple(
            sorted(
                backend_bindings,
                key=lambda item: item.hbm_capacity_ref,
            )
        )
        descriptors = tuple(
            sorted(descriptors, key=lambda item: item.sequence)
        )
        external_seeds = tuple(
            sorted(
                external_seeds,
                key=lambda item: (
                    item.external_capacity_ref,
                    item.address,
                    item.id,
                ),
            )
        )
        external_probes = tuple(
            sorted(
                external_probes,
                key=lambda item: (
                    item.external_capacity_ref,
                    item.address,
                    item.id,
                ),
            )
        )
        key = {
            "case_digest": case_digest,
            "request_digest": request_digest,
            "logical_graph_digest": logical_graph_digest,
            "source_memory_plan_digest": source_memory_plan_digest,
            "blocking_offload_plan_id": blocking_offload_plan_id,
            "blocking_offload_plan_digest": blocking_offload_plan_digest,
            "fabric": fabric,
            "backend_bindings": backend_bindings,
            "descriptors": descriptors,
            "external_seeds": external_seeds,
            "external_probes": external_probes,
        }
        result = cls(
            schema_version=EXTERNAL_DMA_PROGRAM_SCHEMA_VERSION,
            producer_pass="external_dma_program_finalizer",
            id=_stable_id(
                "external_dma_program",
                key,
                EXTERNAL_DMA_PROGRAM_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "external_dma_program") -> None:
        if self.schema_version != EXTERNAL_DMA_PROGRAM_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "external_dma_program_finalizer":
            raise SchemaError(
                "unexpected producer pass",
                path=f"{path}.producer_pass",
            )
        for name in (
            "case_digest",
            "request_digest",
            "logical_graph_digest",
            "source_memory_plan_digest",
            "blocking_offload_plan_digest",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        validate_nonempty(
            self.blocking_offload_plan_id,
            f"{path}.blocking_offload_plan_id",
        )
        self.fabric.validate(f"{path}.fabric")
        self._validate_bindings(path)
        self._validate_descriptors(path)
        self._validate_io(path)
        _require_id(
            self.id,
            kind="external_dma_program",
            key=self._key(),
            version=EXTERNAL_DMA_PROGRAM_SCHEMA_VERSION,
            path=path,
        )

    def _validate_bindings(self, path: str) -> None:
        expected = {
            item.id: int(item.location_ref.removeprefix("die:"))
            for item in self.fabric.hbm_capacities
        }
        actual: dict[str, ExternalDmaBackendBinding] = {}
        endpoints: set[tuple[int, int]] = set()
        for index, binding in enumerate(self.backend_bindings):
            binding.validate(f"{path}.backend_bindings[{index}]")
            if binding.hbm_capacity_ref in actual:
                raise SchemaError(
                    "duplicate HBM capacity binding",
                    path=f"{path}.backend_bindings[{index}]",
                )
            if binding.hbm_capacity_ref not in expected:
                raise SchemaError(
                    "binding references unknown HBM capacity",
                    path=f"{path}.backend_bindings[{index}]",
                )
            if binding.owner_die_id != expected[binding.hbm_capacity_ref]:
                raise SchemaError(
                    "binding owner differs from HBM capacity owner",
                    path=f"{path}.backend_bindings[{index}]",
                )
            endpoint = (binding.stack_id, binding.channel_id)
            if endpoint in endpoints:
                raise SchemaError(
                    "backend endpoint is bound more than once",
                    path=f"{path}.backend_bindings[{index}]",
                )
            endpoints.add(endpoint)
            actual[binding.hbm_capacity_ref] = binding
        if set(actual) != set(expected):
            raise SchemaError(
                "bindings must exactly cover HBM capacities",
                path=f"{path}.backend_bindings",
            )
        canonical = tuple(
            sorted(
                self.backend_bindings,
                key=lambda item: item.hbm_capacity_ref,
            )
        )
        if self.backend_bindings != canonical:
            raise SchemaError(
                "must use canonical capacity order",
                path=f"{path}.backend_bindings",
            )

    def _validate_descriptors(self, path: str) -> None:
        connections = {
            item.id: item for item in self.fabric.connections
        }
        hbm_capacities = {
            item.id: item for item in self.fabric.hbm_capacities
        }
        external_capacities = {
            item.id: item for item in self.fabric.external_capacities
        }
        links = {item.id: item for item in self.fabric.links}
        ids: set[str] = set()
        operation_refs: set[str] = set()
        transfer_refs: set[str] = set()
        for index, descriptor in enumerate(self.descriptors):
            descriptor.validate(f"{path}.descriptors[{index}]")
            if descriptor.sequence != index:
                raise SchemaError(
                    "sequence must be contiguous from zero",
                    path=f"{path}.descriptors[{index}].sequence",
                )
            expected_dep = () if index == 0 else (
                self.descriptors[index - 1].id,
            )
            if descriptor.depends_on != expected_dep:
                raise SchemaError(
                    "blocking DMA requires the prior descriptor dependency",
                    path=f"{path}.descriptors[{index}].depends_on",
                )
            if (
                index > 0
                and descriptor.planned_issue_cycle
                < self.descriptors[index - 1].planned_ready_cycle
            ):
                raise SchemaError(
                    "planned transfer starts before its dependency",
                    path=(
                        f"{path}.descriptors[{index}]"
                        ".planned_issue_cycle"
                    ),
                )
            if (
                descriptor.id in ids
                or descriptor.operation_ref in operation_refs
                or descriptor.source_transfer_request_ref in transfer_refs
            ):
                raise SchemaError(
                    "descriptor provenance must be unique",
                    path=f"{path}.descriptors[{index}]",
                )
            connection = connections.get(descriptor.connection_ref)
            if connection is None:
                raise SchemaError(
                    "descriptor references unknown connection",
                    path=f"{path}.descriptors[{index}].connection_ref",
                )
            hbm = hbm_capacities[connection.hbm_capacity_ref]
            external = external_capacities[
                links[connection.link_ref].external_capacity_ref
            ]
            if not (
                hbm.base_address
                <= descriptor.hbm_address
                <= hbm.base_address
                + hbm.capacity_bytes
                - descriptor.size_bytes
            ):
                raise SchemaError(
                    "descriptor exceeds HBM capacity",
                    path=f"{path}.descriptors[{index}]",
                )
            if not (
                external.base_address
                <= descriptor.external_address
                <= external.base_address
                + external.capacity_bytes
                - descriptor.size_bytes
            ):
                raise SchemaError(
                    "descriptor exceeds external capacity",
                    path=f"{path}.descriptors[{index}]",
                )
            ids.add(descriptor.id)
            operation_refs.add(descriptor.operation_ref)
            transfer_refs.add(descriptor.source_transfer_request_ref)
        if not self.descriptors:
            raise SchemaError(
                "must contain at least one transfer",
                path=f"{path}.descriptors",
            )

    def _validate_io(self, path: str) -> None:
        capacities = {
            item.id: item for item in self.fabric.external_capacities
        }
        for collection_name in ("external_seeds", "external_probes"):
            collection = getattr(self, collection_name)
            if not collection:
                raise SchemaError(
                    "must not be empty",
                    path=f"{path}.{collection_name}",
                )
            seen_ranges: dict[str, list[tuple[int, int]]] = {}
            for index, record in enumerate(collection):
                record.validate(
                    f"{path}.{collection_name}[{index}]"
                )
                capacity = capacities.get(record.external_capacity_ref)
                if capacity is None:
                    raise SchemaError(
                        "references unknown external capacity",
                        path=(
                            f"{path}.{collection_name}[{index}]"
                            ".external_capacity_ref"
                        ),
                    )
                payload_hex = getattr(
                    record,
                    "payload_hex",
                    getattr(record, "expected_payload_hex", ""),
                )
                size = len(payload_hex) // 2
                start = record.address
                end = start + size
                if (
                    start < capacity.base_address
                    or end
                    > capacity.base_address + capacity.capacity_bytes
                ):
                    raise SchemaError(
                        "record exceeds external capacity",
                        path=f"{path}.{collection_name}[{index}]",
                    )
                ranges = seen_ranges.setdefault(
                    record.external_capacity_ref,
                    [],
                )
                if any(
                    start < old_end and old_start < end
                    for old_start, old_end in ranges
                ):
                    raise SchemaError(
                        "records overlap",
                        path=f"{path}.{collection_name}[{index}]",
                    )
                ranges.append((start, end))
            canonical = tuple(
                sorted(
                    collection,
                    key=lambda item: (
                        item.external_capacity_ref,
                        item.address,
                        item.id,
                    ),
                )
            )
            if collection != canonical:
                raise SchemaError(
                    "must use canonical address order",
                    path=f"{path}.{collection_name}",
                )


def external_dma_program_digest(program: ExternalDmaProgram) -> str:
    program.validate()
    return canonical_digest(program)


__all__ = [
    "EXTERNAL_DMA_PROGRAM_SCHEMA_VERSION",
    "EXTERNAL_DMA_REQUEST_SCHEMA_VERSION",
    "ExternalDmaBackendBinding",
    "ExternalDmaDescriptor",
    "ExternalDmaProbe",
    "ExternalDmaProgram",
    "ExternalDmaSeed",
    "external_dma_program_digest",
]
