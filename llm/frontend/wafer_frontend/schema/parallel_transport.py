"""Deterministic transport plan for an explicit parallel placement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .artifact_manifest import CoreRuntimeBinding
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .global_action import LogicalCoreRef
from .ir1 import PairRoute, PhysicalFabric
from .parallel_placement import ParallelPlacement
from .serde import canonical_digest


PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION = (
    "wafer_frontend.parallel_transport_plan/v1alpha1"
)


class ParallelCommunicationKind(str, Enum):
    P2P = "p2p"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_REDUCE = "all_reduce"
    ALL_TO_ALL = "all_to_all"


class ParallelTransferEndpointRole(str, Enum):
    SEND = "send"
    RECV = "recv"


@dataclass(frozen=True, slots=True)
class ParallelCommunicationRequest:
    """One direct P2P transfer or one collective transport request.

    transfer_bytes is the payload of one generated point-to-point transfer.
    For ring collectives it is one ring chunk. Compute-side reduction and
    concatenation remain outside this P1 transport-only contract.
    """

    id: str
    workload_case_id: str
    workload_request_digest: str
    logical_operation_ref: str
    workload_phase: str
    workload_step: int
    workload_layer: int | None
    payload_value_refs: tuple[str, ...]
    is_noop: bool
    kind: ParallelCommunicationKind
    group_id: str | None
    source_rank: int | None
    destination_rank: int | None
    transfer_bytes: int

    def validate(self, path: str = "parallel_communication_request") -> None:
        for name in (
            "id",
            "workload_case_id",
            "workload_request_digest",
            "logical_operation_ref",
            "workload_phase",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if (
            len(self.workload_request_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.workload_request_digest)
        ):
            raise SchemaError("must be a lowercase SHA-256 digest", path=f"{path}.workload_request_digest")
        validate_uint64(self.workload_step, f"{path}.workload_step")
        if self.workload_layer is not None:
            validate_uint64(self.workload_layer, f"{path}.workload_layer")
        if type(self.payload_value_refs) is not tuple or not self.payload_value_refs:
            raise SchemaError("must contain tensor value refs", path=f"{path}.payload_value_refs")
        if len(set(self.payload_value_refs)) != len(self.payload_value_refs):
            raise SchemaError("contains duplicate tensor value refs", path=f"{path}.payload_value_refs")
        for index, value_ref in enumerate(self.payload_value_refs):
            validate_nonempty(value_ref, f"{path}.payload_value_refs[{index}]")
        if type(self.is_noop) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.is_noop")
        if type(self.kind) is not ParallelCommunicationKind:
            raise SchemaError(
                "must be a ParallelCommunicationKind", path=f"{path}.kind"
            )
        validate_uint64(self.transfer_bytes, f"{path}.transfer_bytes")
        if (self.transfer_bytes == 0) != self.is_noop:
            raise SchemaError(
                "zero bytes must be an explicit no-op", path=f"{path}.transfer_bytes"
            )
        if self.kind is ParallelCommunicationKind.P2P:
            if self.group_id is not None:
                raise SchemaError(
                    "P2P must not reference a group", path=f"{path}.group_id"
                )
            for name in ("source_rank", "destination_rank"):
                value = getattr(self, name)
                if value is None:
                    raise SchemaError(
                        "P2P requires both endpoint ranks", path=f"{path}.{name}"
                    )
                validate_uint64(value, f"{path}.{name}")
            if self.source_rank == self.destination_rank:
                raise SchemaError("P2P endpoints must differ", path=path)
        else:
            if self.group_id is None:
                raise SchemaError(
                    "collective requires a group", path=f"{path}.group_id"
                )
            validate_nonempty(self.group_id, f"{path}.group_id")
            if self.source_rank is not None or self.destination_rank is not None:
                raise SchemaError(
                    "collective endpoints come from its group", path=path
                )


@dataclass(frozen=True, slots=True)
class ParallelTransferEndpoint:
    id: str
    transfer_id: str
    role: ParallelTransferEndpointRole
    rank: int
    peer_rank: int
    logical_core: LogicalCoreRef
    bytes: int

    def validate(self, path: str = "parallel_transfer_endpoint") -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.transfer_id, f"{path}.transfer_id")
        if type(self.role) is not ParallelTransferEndpointRole:
            raise SchemaError(
                "must be a ParallelTransferEndpointRole", path=f"{path}.role"
            )
        for name in ("rank", "peer_rank", "bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.rank == self.peer_rank:
            raise SchemaError("endpoint and peer ranks must differ", path=path)
        if self.bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes")
        if type(self.logical_core) is not LogicalCoreRef:
            raise SchemaError(
                "must be a LogicalCoreRef", path=f"{path}.logical_core"
            )
        self.logical_core.validate(f"{path}.logical_core")


@dataclass(frozen=True, slots=True)
class ParallelTransfer:
    id: str
    request_id: str
    workload_phase: str
    workload_step: int
    workload_layer: int | None
    logical_operation_ref: str
    payload_value_refs: tuple[str, ...]
    algorithm_phase: str
    algorithm_step: int
    source_rank: int
    destination_rank: int
    bytes: int
    route_id: str
    send_endpoint_id: str
    recv_endpoint_id: str

    def validate(self, path: str = "parallel_transfer") -> None:
        for name in (
            "id",
            "request_id",
            "workload_phase",
            "logical_operation_ref",
            "algorithm_phase",
            "route_id",
            "send_endpoint_id",
            "recv_endpoint_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in (
            "workload_step",
            "algorithm_step",
            "source_rank",
            "destination_rank",
            "bytes",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.workload_layer is not None:
            validate_uint64(self.workload_layer, f"{path}.workload_layer")
        if type(self.payload_value_refs) is not tuple or not self.payload_value_refs:
            raise SchemaError("must contain tensor value refs", path=f"{path}.payload_value_refs")
        if self.source_rank == self.destination_rank:
            raise SchemaError("transfer ranks must differ", path=path)
        if self.bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes")
        if self.send_endpoint_id == self.recv_endpoint_id:
            raise SchemaError("send and receive endpoints must differ", path=path)


def expected_transfer_specs(
    requests: tuple[ParallelCommunicationRequest, ...],
    placement: ParallelPlacement,
) -> tuple[tuple[str, str, int, int, int, int], ...]:
    """Return canonical request, phase, step, src, dst, bytes records."""

    groups = {group.id: group for group in placement.groups}
    specs: list[tuple[str, str, int, int, int, int]] = []
    for request in requests:
        request.validate()
        if request.is_noop:
            continue
        if request.kind is ParallelCommunicationKind.P2P:
            assert request.source_rank is not None
            assert request.destination_rank is not None
            specs.append(
                (
                    request.id,
                    ParallelCommunicationKind.P2P.value,
                    0,
                    request.source_rank,
                    request.destination_rank,
                    request.transfer_bytes,
                )
            )
            continue
        assert request.group_id is not None
        group = groups.get(request.group_id)
        if group is None:
            raise SchemaError(
                "request references an unknown placement group",
                path=f"requests[{request.id}].group_id",
            )
        ranks = group.ranks
        if len(ranks) == 1:
            continue
        if request.kind is ParallelCommunicationKind.ALL_TO_ALL:
            specs.extend(
                (
                    request.id,
                    request.kind.value,
                    0,
                    source,
                    destination,
                    request.transfer_bytes,
                )
                for source in ranks
                for destination in ranks
                if source != destination
            )
            continue
        phases = (
            (
                ParallelCommunicationKind.REDUCE_SCATTER.value,
                ParallelCommunicationKind.ALL_GATHER.value,
            )
            if request.kind is ParallelCommunicationKind.ALL_REDUCE
            else (request.kind.value,)
        )
        for phase in phases:
            for step in range(len(ranks) - 1):
                specs.extend(
                    (
                        request.id,
                        phase,
                        step,
                        rank,
                        ranks[(rank_index + 1) % len(ranks)],
                        request.transfer_bytes,
                    )
                    for rank_index, rank in enumerate(ranks)
                )
    return tuple(specs)


@dataclass(frozen=True, slots=True)
class ParallelTransportPlan:
    schema_version: str
    producer_pass: str
    id: str
    placement_digest: str
    fabric_digest: str
    requests: tuple[ParallelCommunicationRequest, ...]
    core_bindings: tuple[CoreRuntimeBinding, ...]
    routes: tuple[PairRoute, ...]
    transfers: tuple[ParallelTransfer, ...]
    endpoints: tuple[ParallelTransferEndpoint, ...]

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        placement_digest: str,
        fabric_digest: str,
        requests: tuple[ParallelCommunicationRequest, ...],
        core_bindings: tuple[CoreRuntimeBinding, ...],
        routes: tuple[PairRoute, ...],
        transfers: tuple[ParallelTransfer, ...],
        endpoints: tuple[ParallelTransferEndpoint, ...],
    ) -> "ParallelTransportPlan":
        semantic = {
            "placement_digest": placement_digest,
            "fabric_digest": fabric_digest,
            "requests": requests,
            "core_bindings": core_bindings,
            "routes": routes,
            "transfers": transfers,
            "endpoints": endpoints,
        }
        result = cls(
            schema_version=PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "parallel_transport_plan",
                semantic,
                schema_version=PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            "placement_digest": self.placement_digest,
            "fabric_digest": self.fabric_digest,
            "requests": self.requests,
            "core_bindings": self.core_bindings,
            "routes": self.routes,
            "transfers": self.transfers,
            "endpoints": self.endpoints,
        }

    def validate(self, path: str = "parallel_transport_plan") -> None:
        if self.schema_version != PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        for name in ("placement_digest", "fabric_digest"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SchemaError(
                    "must be a lowercase SHA-256 digest", path=f"{path}.{name}"
                )
        request_by_id: dict[str, ParallelCommunicationRequest] = {}
        for index, request in enumerate(self.requests):
            if type(request) is not ParallelCommunicationRequest:
                raise SchemaError(
                    "must be a ParallelCommunicationRequest",
                    path=f"{path}.requests[{index}]",
                )
            request.validate(f"{path}.requests[{index}]")
            if request.id in request_by_id:
                raise SchemaError(
                    "duplicate request id", path=f"{path}.requests[{index}].id"
                )
            request_by_id[request.id] = request
        if tuple(sorted(self.requests, key=lambda item: item.id)) != self.requests:
            raise SchemaError(
                "requests must use canonical id order", path=f"{path}.requests"
            )

        logical_cores: set[LogicalCoreRef] = set()
        runtime_cores: set[int] = set()
        for index, binding in enumerate(self.core_bindings):
            if type(binding) is not CoreRuntimeBinding:
                raise SchemaError(
                    "must be a CoreRuntimeBinding",
                    path=f"{path}.core_bindings[{index}]",
                )
            binding.validate(f"{path}.core_bindings[{index}]")
            if (
                binding.logical_core in logical_cores
                or binding.runtime_core_id in runtime_cores
            ):
                raise SchemaError(
                    "core bindings must be one-to-one",
                    path=f"{path}.core_bindings[{index}]",
                )
            logical_cores.add(binding.logical_core)
            runtime_cores.add(binding.runtime_core_id)

        routes = self._validate_unique_ids(self.routes, "routes", PairRoute, path)
        transfers = self._validate_unique_ids(
            self.transfers, "transfers", ParallelTransfer, path
        )
        endpoints = self._validate_unique_ids(
            self.endpoints, "endpoints", ParallelTransferEndpoint, path
        )
        used_routes: set[str] = set()
        used_endpoints: set[str] = set()
        for index, transfer in enumerate(self.transfers):
            transfer_path = f"{path}.transfers[{index}]"
            if transfer.request_id not in request_by_id:
                raise SchemaError(
                    "transfer references an unknown request", path=transfer_path
                )
            request = request_by_id[transfer.request_id]
            if (
                transfer.workload_phase,
                transfer.workload_step,
                transfer.workload_layer,
                transfer.logical_operation_ref,
                transfer.payload_value_refs,
                transfer.bytes,
            ) != (
                request.workload_phase,
                request.workload_step,
                request.workload_layer,
                request.logical_operation_ref,
                request.payload_value_refs,
                request.transfer_bytes,
            ):
                raise SchemaError(
                    "transfer workload/payload binding disagrees with request",
                    path=transfer_path,
                )
            route = routes.get(transfer.route_id)
            if route is None or (
                route.source_rank,
                route.destination_rank,
            ) != (transfer.source_rank, transfer.destination_rank):
                raise SchemaError(
                    "transfer route does not match its ranks",
                    path=f"{transfer_path}.route_id",
                )
            send = endpoints.get(transfer.send_endpoint_id)
            recv = endpoints.get(transfer.recv_endpoint_id)
            if send is None or (
                send.transfer_id,
                send.role,
                send.rank,
                send.peer_rank,
                send.bytes,
            ) != (
                transfer.id,
                ParallelTransferEndpointRole.SEND,
                transfer.source_rank,
                transfer.destination_rank,
                transfer.bytes,
            ):
                raise SchemaError(
                    "send endpoint does not match transfer",
                    path=f"{transfer_path}.send_endpoint_id",
                )
            if recv is None or (
                recv.transfer_id,
                recv.role,
                recv.rank,
                recv.peer_rank,
                recv.bytes,
            ) != (
                transfer.id,
                ParallelTransferEndpointRole.RECV,
                transfer.destination_rank,
                transfer.source_rank,
                transfer.bytes,
            ):
                raise SchemaError(
                    "receive endpoint does not match transfer",
                    path=f"{transfer_path}.recv_endpoint_id",
                )
            used_routes.add(route.id)
            used_endpoints.update((send.id, recv.id))
        if used_routes != set(routes):
            raise SchemaError("routes must be used by transfers", path=f"{path}.routes")
        if used_endpoints != set(endpoints):
            raise SchemaError(
                "endpoints must be the exact send/receive closure",
                path=f"{path}.endpoints",
            )
        expected_id = stable_artifact_id(
            "parallel_transport_plan",
            self._semantic(),
            schema_version=PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable transport plan id", path=f"{path}.id")

    @staticmethod
    def _validate_unique_ids(
        items: tuple[object, ...],
        name: str,
        expected_type: type,
        path: str,
    ) -> dict[str, object]:
        if type(items) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.{name}")
        result: dict[str, object] = {}
        for index, item in enumerate(items):
            item_path = f"{path}.{name}[{index}]"
            if type(item) is not expected_type:
                raise SchemaError(
                    f"must be a {expected_type.__name__}", path=item_path
                )
            item.validate(item_path)
            item_id = item.id
            if item_id in result:
                raise SchemaError("duplicate id", path=f"{item_path}.id")
            result[item_id] = item
        if tuple(sorted(items, key=lambda item: item.id)) != items:
            raise SchemaError(
                f"{name} must use canonical id order", path=f"{path}.{name}"
            )
        return result

    def validate_against(
        self,
        placement: ParallelPlacement,
        fabric: PhysicalFabric,
        path: str = "parallel_transport_plan",
    ) -> None:
        self.validate(path)
        if type(placement) is not ParallelPlacement:
            raise SchemaError("must be a ParallelPlacement", path="placement")
        if type(fabric) is not PhysicalFabric:
            raise SchemaError("must be a PhysicalFabric", path="fabric")
        placement.validate("placement")
        fabric.validate("fabric")
        if self.placement_digest != placement.digest:
            raise SchemaError(
                "plan was built for a different placement",
                path=f"{path}.placement_digest",
            )
        if self.fabric_digest != canonical_digest(fabric):
            raise SchemaError(
                "plan was built for a different physical fabric",
                path=f"{path}.fabric_digest",
            )
        _validate_mesh_fabric(placement, fabric)
        expected_specs = expected_transfer_specs(self.requests, placement)
        actual_specs = tuple(
            (
                transfer.request_id,
                transfer.algorithm_phase,
                transfer.algorithm_step,
                transfer.source_rank,
                transfer.destination_rank,
                transfer.bytes,
            )
            for transfer in self.transfers
        )
        if tuple(sorted(actual_specs)) != tuple(sorted(expected_specs)):
            raise SchemaError(
                "transfers do not implement the requested deterministic algorithm",
                path=f"{path}.transfers",
            )
        expected_cores = {
            LogicalCoreRef(item.die_id, item.local_core)
            for item in placement.rank_placements
        }
        if {item.logical_core for item in self.core_bindings} != expected_cores:
            raise SchemaError(
                "core bindings must exactly cover placed workload ranks",
                path=f"{path}.core_bindings",
            )
        die_by_id = {die.id: die for die in fabric.dies}
        for binding in self.core_bindings:
            core = next(
                (
                    item
                    for item in die_by_id[binding.logical_core.die_id].cores
                    if item.local_core_id == binding.logical_core.local_core_id
                ),
                None,
            )
            if core is None or (
                binding.core_spec_ref,
                binding.runtime_core_id,
                binding.sram_profile_ref,
            ) != (core.id, core.runtime_core_id, core.sram_profile_ref):
                raise SchemaError(
                    "core binding disagrees with the physical fabric",
                    path=f"{path}.core_bindings",
                )
        rank_to_die = {
            item.logical_rank: item.die_id for item in placement.rank_placements
        }
        for index, route in enumerate(self.routes):
            route.validate_against(
                fabric,
                rank_to_die,
                f"{path}.routes[{index}]",
            )
        binding_by_rank = {
            item.logical_rank: LogicalCoreRef(item.die_id, item.local_core)
            for item in placement.rank_placements
        }
        for index, endpoint in enumerate(self.endpoints):
            if endpoint.logical_core != binding_by_rank.get(endpoint.rank):
                raise SchemaError(
                    "endpoint core disagrees with rank placement",
                    path=f"{path}.endpoints[{index}].logical_core",
                )


def _validate_mesh_fabric(
    placement: ParallelPlacement, fabric: PhysicalFabric
) -> None:
    mesh = placement.mesh
    if fabric.die_grid != mesh.physical_shape:
        raise SchemaError(
            "fabric shape differs from placement Mesh", path="fabric.die_grid"
        )
    dies = {die.id: die for die in fabric.dies}
    if set(dies) != set(range(mesh.rank_count)):
        raise SchemaError(
            "fabric must contain every physical Mesh Die", path="fabric.dies"
        )
    for die_id, die in dies.items():
        if die.coord != mesh.coordinate(die_id):
            raise SchemaError(
                "fabric Die coordinates differ from placement Mesh",
                path=f"fabric.dies[{die_id}].coord",
            )


__all__ = [
    "PARALLEL_TRANSPORT_PLAN_SCHEMA_VERSION",
    "ParallelCommunicationKind",
    "ParallelCommunicationRequest",
    "ParallelTransfer",
    "ParallelTransferEndpoint",
    "ParallelTransferEndpointRole",
    "ParallelTransportPlan",
    "expected_transfer_specs",
]
