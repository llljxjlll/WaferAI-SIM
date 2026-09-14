"""Typed preflight artifact joining workload, placement, transport, and memory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .e2e_workload_graph import E2EWorkloadGraph
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .memory_plan import MemoryObjectKind, MemoryPlan
from .parallel_placement import ParallelPlacement, ParallelWorkloadKind
from .parallel_transport import (
    ParallelCommunicationRequest,
    ParallelTransportPlan,
)
from .serde import canonical_digest
from .workload_run import (
    WorkloadCapabilityLevel,
    WorkloadFamily,
    WorkloadRunCapability,
    WorkloadRunRequest,
)


WORKLOAD_MATERIALIZATION_SCHEMA_VERSION = (
    "wafer_frontend.workload_materialization/v1alpha1"
)
WORKLOAD_STATE_INVENTORY_SCHEMA_VERSION = (
    "wafer_frontend.workload_state_inventory/v1alpha1"
)


class WorkloadMaterializationStatus(str, Enum):
    UNSUPPORTED = "unsupported"
    PARTIAL = "partial"


class WorkloadArtifactStatus(str, Enum):
    NOT_MATERIALIZED = "not_materialized"


def _stable(kind: str, key: object, version: str) -> str:
    return stable_artifact_id(kind, key, schema_version=version)


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class WorkloadStateInventoryItem:
    id: str
    logical_name: str
    object_kind: MemoryObjectKind
    logical_rank: int
    owner_domain_ref: str | None
    size_bytes: int
    writable: bool

    @classmethod
    def create(
        cls,
        *,
        logical_name: str,
        object_kind: MemoryObjectKind,
        logical_rank: int,
        owner_domain_ref: str | None,
        size_bytes: int,
        writable: bool,
    ) -> "WorkloadStateInventoryItem":
        key = {
            "logical_name": logical_name,
            "object_kind": object_kind,
            "logical_rank": logical_rank,
            "owner_domain_ref": owner_domain_ref,
            "size_bytes": size_bytes,
            "writable": writable,
        }
        result = cls(
            id=_stable(
                "workload_state",
                key,
                WORKLOAD_STATE_INVENTORY_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            "logical_name": self.logical_name,
            "object_kind": self.object_kind,
            "logical_rank": self.logical_rank,
            "owner_domain_ref": self.owner_domain_ref,
            "size_bytes": self.size_bytes,
            "writable": self.writable,
        }

    def validate(self, path: str = "state_inventory") -> None:
        validate_nonempty(self.logical_name, f"{path}.logical_name")
        if type(self.object_kind) is not MemoryObjectKind:
            raise SchemaError("must be a MemoryObjectKind", path=f"{path}.object_kind")
        validate_uint64(self.logical_rank, f"{path}.logical_rank")
        if self.owner_domain_ref is not None:
            validate_nonempty(self.owner_domain_ref, f"{path}.owner_domain_ref")
        validate_uint64(self.size_bytes, f"{path}.size_bytes")
        if self.size_bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.size_bytes")
        if type(self.writable) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.writable")
        expected = _stable(
            "workload_state",
            self._key(),
            WORKLOAD_STATE_INVENTORY_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable state id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class WorkloadMaterializationManifest:
    schema_version: str
    producer_pass: str
    id: str
    request: WorkloadRunRequest
    request_digest: str
    capability: WorkloadRunCapability
    capability_digest: str
    status: WorkloadMaterializationStatus
    unsupported_requirements: tuple[str, ...]
    logical_graph: E2EWorkloadGraph
    logical_graph_digest: str
    state_inventory: tuple[WorkloadStateInventoryItem, ...]
    placement: ParallelPlacement
    transport_requests: tuple[ParallelCommunicationRequest, ...]
    transport_plan: ParallelTransportPlan | None
    memory_plan: MemoryPlan
    lowering_status: WorkloadArtifactStatus
    runtime_status: WorkloadArtifactStatus

    @classmethod
    def create(
        cls,
        *,
        request: WorkloadRunRequest,
        capability: WorkloadRunCapability,
        logical_graph: E2EWorkloadGraph,
        state_inventory: tuple[WorkloadStateInventoryItem, ...],
        placement: ParallelPlacement,
        transport_requests: tuple[ParallelCommunicationRequest, ...],
        transport_plan: ParallelTransportPlan | None,
        memory_plan: MemoryPlan,
    ) -> "WorkloadMaterializationManifest":
        unsupported = capability.unsupported_requirements(request)
        key = {
            "request": request,
            "request_digest": request.digest,
            "capability": capability,
            "capability_digest": capability.digest,
            "status": (
                WorkloadMaterializationStatus.UNSUPPORTED
                if unsupported
                else WorkloadMaterializationStatus.PARTIAL
            ),
            "unsupported_requirements": unsupported,
            "logical_graph": logical_graph,
            "logical_graph_digest": canonical_digest(logical_graph),
            "state_inventory": state_inventory,
            "placement": placement,
            "transport_requests": transport_requests,
            "transport_plan": transport_plan,
            "memory_plan": memory_plan,
            "lowering_status": WorkloadArtifactStatus.NOT_MATERIALIZED,
            "runtime_status": WorkloadArtifactStatus.NOT_MATERIALIZED,
        }
        result = cls(
            schema_version=WORKLOAD_MATERIALIZATION_SCHEMA_VERSION,
            producer_pass="workload_preflight_materializer",
            id=_stable(
                "workload_materialization",
                key,
                WORKLOAD_MATERIALIZATION_SCHEMA_VERSION,
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

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "workload_materialization") -> None:
        if self.schema_version != WORKLOAD_MATERIALIZATION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "workload_preflight_materializer":
            raise SchemaError("unexpected producer", path=f"{path}.producer_pass")
        if type(self.request) is not WorkloadRunRequest:
            raise SchemaError("must be a WorkloadRunRequest", path=f"{path}.request")
        self.request.validate(f"{path}.request")
        if type(self.capability) is not WorkloadRunCapability:
            raise SchemaError(
                "must be a WorkloadRunCapability", path=f"{path}.capability"
            )
        self.capability.validate(f"{path}.capability")
        _validate_digest(self.request_digest, f"{path}.request_digest")
        _validate_digest(self.capability_digest, f"{path}.capability_digest")
        _validate_digest(
            self.logical_graph_digest,
            f"{path}.logical_graph_digest",
        )
        if self.request_digest != self.request.digest:
            raise SchemaError("does not match request", path=f"{path}.request_digest")
        if self.capability_digest != self.capability.digest:
            raise SchemaError(
                "does not match capability", path=f"{path}.capability_digest"
            )
        if type(self.logical_graph) is not E2EWorkloadGraph:
            raise SchemaError(
                "must be an E2EWorkloadGraph",
                path=f"{path}.logical_graph",
            )
        self.logical_graph.validate(f"{path}.logical_graph")
        if self.logical_graph.request != self.request:
            raise SchemaError(
                "does not match request",
                path=f"{path}.logical_graph.request",
            )
        if self.logical_graph.placement != self.placement:
            raise SchemaError(
                "does not match manifest placement",
                path=f"{path}.logical_graph.placement",
            )
        if self.logical_graph_digest != canonical_digest(self.logical_graph):
            raise SchemaError(
                "does not match logical graph",
                path=f"{path}.logical_graph_digest",
            )
        expected_unsupported = self.capability.unsupported_requirements(self.request)
        if self.unsupported_requirements != expected_unsupported:
            raise SchemaError(
                "does not match capability evaluation",
                path=f"{path}.unsupported_requirements",
            )
        expected_status = (
            WorkloadMaterializationStatus.UNSUPPORTED
            if expected_unsupported
            else WorkloadMaterializationStatus.PARTIAL
        )
        if self.status is not expected_status:
            raise SchemaError("does not match capability", path=f"{path}.status")
        if self.lowering_status is not WorkloadArtifactStatus.NOT_MATERIALIZED:
            raise SchemaError("lowering is not produced here", path=f"{path}.lowering_status")
        if self.runtime_status is not WorkloadArtifactStatus.NOT_MATERIALIZED:
            raise SchemaError("runtime is not produced here", path=f"{path}.runtime_status")
        self._validate_inventory(path)
        self._validate_materialized_artifacts(path)
        expected_id = _stable(
            "workload_materialization",
            self._key(),
            WORKLOAD_MATERIALIZATION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable materialization id", path=f"{path}.id")

    def _validate_inventory(self, path: str) -> None:
        if type(self.state_inventory) is not tuple or not self.state_inventory:
            raise SchemaError(
                "must be a non-empty tuple",
                path=f"{path}.state_inventory",
            )
        ids: set[str] = set()
        for index, item in enumerate(self.state_inventory):
            item.validate(f"{path}.state_inventory[{index}]")
            if item.id in ids:
                raise SchemaError(
                    "contains duplicate id",
                    path=f"{path}.state_inventory",
                )
            ids.add(item.id)

    def _validate_materialized_artifacts(self, path: str) -> None:
        if type(self.placement) is not ParallelPlacement:
            raise SchemaError("must be a ParallelPlacement", path=f"{path}.placement")
        self.placement.validate(f"{path}.placement")
        expected_kind = (
            ParallelWorkloadKind.MOE
            if self.request.family.is_moe
            else ParallelWorkloadKind.DENSE
        )
        if self.placement.workload_kind is not expected_kind:
            raise SchemaError("family disagrees with placement", path=f"{path}.placement")
        expected_degrees = (
            self.request.parallel.tp,
            self.request.parallel.dp,
            self.request.parallel.ep,
            self.request.parallel.pp,
        )
        actual_degrees = (
            self.placement.tp_degree,
            self.placement.dp_degree,
            self.placement.ep_degree,
            self.placement.pp_degree,
        )
        if actual_degrees != expected_degrees:
            raise SchemaError("parallel degrees disagree", path=f"{path}.placement")
        expected_mesh = (self.request.mesh.rows, self.request.mesh.columns)
        actual_mesh = (self.placement.mesh.rows, self.placement.mesh.columns)
        if actual_mesh != expected_mesh:
            raise SchemaError(
                "mesh dimensions disagree", path=f"{path}.placement.mesh"
            )
        expected_active = self.request.parallel.active_die_ids or tuple(
            range(self.request.parallel.logical_rank_count)
        )
        if self.placement.active_die_ids != expected_active:
            raise SchemaError(
                "active Die mapping disagrees",
                path=f"{path}.placement.rank_placements",
            )
        owner_ids = {item.id for item in self.placement.ownership_domains}
        for index, state in enumerate(self.state_inventory):
            if state.logical_rank >= self.placement.logical_rank_count:
                raise SchemaError(
                    "state rank is not placed",
                    path=f"{path}.state_inventory[{index}].logical_rank",
                )
            if (
                state.owner_domain_ref is not None
                and state.owner_domain_ref not in owner_ids
            ):
                raise SchemaError(
                    "state references unknown ownership domain",
                    path=f"{path}.state_inventory[{index}].owner_domain_ref",
                )
        group_ids = {item.id for item in self.placement.groups}
        transport_ids: set[str] = set()
        for index, request in enumerate(self.transport_requests):
            request.validate(f"{path}.transport_requests[{index}]")
            if request.id in transport_ids:
                raise SchemaError(
                    "contains duplicate id", path=f"{path}.transport_requests"
                )
            transport_ids.add(request.id)
            if request.group_id is not None and request.group_id not in group_ids:
                raise SchemaError(
                    "references unknown placement group",
                    path=f"{path}.transport_requests[{index}].group_id",
                )
        if self.transport_plan is not None:
            self.transport_plan.validate(f"{path}.transport_plan")
            if self.transport_plan.requests != self.transport_requests:
                raise SchemaError(
                    "requests disagree with transport plan",
                    path=f"{path}.transport_plan.requests",
                )
            if self.transport_plan.placement_digest != self.placement.digest:
                raise SchemaError(
                    "placement digest disagrees",
                    path=f"{path}.transport_plan.placement_digest",
                )
        if type(self.memory_plan) is not MemoryPlan:
            raise SchemaError("must be a MemoryPlan", path=f"{path}.memory_plan")
        self.memory_plan.validate(f"{path}.memory_plan")
        inventory_ids = {item.id for item in self.state_inventory}
        planned_state_ids = {item.state_ref for item in self.memory_plan.state_versions}
        if inventory_ids != planned_state_ids:
            raise SchemaError(
                "memory plan must cover the exact state inventory",
                path=f"{path}.memory_plan.state_versions",
            )


__all__ = [
    "WORKLOAD_MATERIALIZATION_SCHEMA_VERSION",
    "WORKLOAD_STATE_INVENTORY_SCHEMA_VERSION",
    "WorkloadArtifactStatus",
    "WorkloadMaterializationManifest",
    "WorkloadMaterializationStatus",
    "WorkloadStateInventoryItem",
]
