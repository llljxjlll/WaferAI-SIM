"""Signed physical AdamW StateABI value provenance and true low-HBM resident reject.

This observer derives real external bytes from five P3 DMA seeds and 83 linked
StateABIs; only NpuSim's native external probes establish runtime readback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from ..errors import SchemaError
from ..schema.memory_plan import (
    MemoryAllocationRequest, MemoryObjectKind, MemoryStateVersion, MemoryTier,
)
from ..schema.persistent_state import StateKind
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import WorkloadStateInventoryItem
from .memory_plan import plan_hierarchical_memory


_ROLES = {
    "trainable_parameter": (StateKind.TRAINABLE_PARAMETER, 15, 4576),
    "optimizer_master": (StateKind.OPTIMIZER_MASTER, 17, 9152),
    "optimizer_moment1": (StateKind.OPTIMIZER_MOMENT1, 17, 9152),
    "optimizer_moment2": (StateKind.OPTIMIZER_MOMENT2, 17, 9152),
    "optimizer_step": (StateKind.OPTIMIZER_STEP, 17, 68),
}


@dataclass(frozen=True, slots=True)
class PhysicalAdamwStateValue:
    state_ref: str
    linked_state_abi_id: str
    source_allocation_ref: str
    kind: str
    external_address: int
    original_hbm_address: int
    paged_hbm_address: int
    size_bytes: int
    seed_digest: str
    inventory_ref: str
    source_version_ref: str
    pinned_request_ref: str


@dataclass(frozen=True, slots=True)
class PhysicalAdamwRoleValue:
    kind: str
    state_count: int
    size_bytes: int
    seed_digest: str


@dataclass(frozen=True, slots=True)
class PhysicalAdamwAuthority:
    offload_request_digest: str
    resident_request_digest: str
    source_memory_plan_digest: str
    original_linked_manifest_digest: str
    paged_linked_manifest_digests: tuple[str, str]
    external_dma_program_digest: str
    paged_contract_digest: str
    states: tuple[PhysicalAdamwStateValue, ...]
    roles: tuple[PhysicalAdamwRoleValue, ...]
    pinned_inventory_digest: str
    pinned_state_versions_digest: str
    pinned_hbm_requests_digest: str
    resident_rejection_code: str
    resident_rejection: str
    hbm_capacity_bytes: int
    workspace_bytes: int

    @property
    def digest(self) -> str:
        return canonical_digest(self)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prove_physical_adamw_authority_and_capacity(
    resident, window, original, paged, program,
) -> PhysicalAdamwAuthority:
    """Reject 83 ABI states resident in real P3 homes; sign external seed bytes."""
    resident.validate("resident")
    window.materialization.validate("offload")
    original.validate("original")
    paged.validate("paged")
    program.validate("program")
    source = window.materialization
    if (
        resident.request.model != source.request.model
        or resident.request.steps != source.request.steps
        or resident.request.mesh != source.request.mesh
        or resident.request.parallel != source.request.parallel
        or resident.request.optimizer != source.request.optimizer
        or resident.request.family != source.request.family
        or tuple((op.kind, op.step, op.layer, op.expert, op.parameter_ref)
                 for op in resident.logical_graph.operations)
        != tuple((op.kind, op.step, op.layer, op.expert, op.parameter_ref)
                 for op in source.logical_graph.operations)
        or original.materialization.id != source.id
        or program.request_digest != source.request_digest
        or program.source_memory_plan_digest != canonical_digest(source.memory_plan)
        or paged.request_digest != source.request_digest
        or paged.source_memory_plan_digest != canonical_digest(source.memory_plan)
        or paged.source_dma_program_digest != canonical_digest(program)
        or window.resident_rejection_code != "memory_capacity_exceeded"
        or window.resident_hbm_capacity_bytes != paged.hbm_capacity_bytes
    ):
        raise SchemaError("AdamW physical witness no longer binds same source model/low HBM",
                          path="physical_adamw.source")
    original_abis = {item.state_ref: item
                     for item in original.manifest.fragments[0].state_abi}
    paged_spans = {item.state_ref: item for item in paged.state_spans}
    if len(original_abis) != len(paged_spans) or len(original_abis) != 83:
        raise SchemaError("83 original/paged linked StateABIs are incomplete",
                          path="physical_adamw.abis")
    seeds = tuple(program.external_seeds)
    if len(seeds) != 5 or len(program.external_probes) != 5:
        raise SchemaError("five signed external roles missing",
                          path="physical_adamw.program")
    seed_by_range = {
        (seed.address, len(seed.payload_hex) // 2): bytes.fromhex(seed.payload_hex)
        for seed in seeds
    }
    if len(seed_by_range) != 5:
        raise SchemaError("external seeds alias or duplicated",
                          path="physical_adamw.program")
    source_requests = {item.id: item for item in source.memory_plan.requests}
    source_versions = {item.id: item for item in source.memory_plan.state_versions}
    source_inventory = {item.id: item for item in source.state_inventory}
    source_allocations = {item.id: item for item in source.memory_plan.allocations}
    inventory = []
    versions = []
    physical_requests = []
    state_values = []
    role_values = []
    for role, (kind, count, size) in _ROLES.items():
        spans = tuple(sorted((item for item in paged.state_spans
                              if item.kind == kind.value),
                             key=lambda item: item.external_address))
        if len(spans) != count or sum(item.size_bytes for item in spans) != size:
            raise SchemaError("optimizer role's 83 linked ABIs differ from source",
                              path=f"physical_adamw.{role}")
        allocation_ref = spans[0].source_allocation_ref
        allocation = source_allocations.get(allocation_ref)
        if allocation is None:
            raise SchemaError("role lost P3 source allocation",
                              path=f"physical_adamw.{role}")
        request = source_requests[allocation.request_ref]
        grouped_state = source_inventory[
            source_versions[request.state_version_ref].state_ref]
        expected_object_kind = (MemoryObjectKind.PARAMETER if role == "trainable_parameter"
                                else MemoryObjectKind.OPTIMIZER)
        if (request.tier is not MemoryTier.EXTERNAL
                or grouped_state.object_kind is not expected_object_kind):
            raise SchemaError("external P3 role kind mismatch",
                              path=f"physical_adamw.{role}")
        key = (allocation.address, request.size_bytes)
        payload = seed_by_range.get(key)
        if payload is None or len(payload) != size:
            raise SchemaError("role exact P3 external seed bytes missing",
                              path=f"physical_adamw.{role}")
        cursor = allocation.address
        for span in spans:
            original_abi = original_abis.get(span.state_ref)
            if (original_abi is None or original_abi.kind is not kind
                    or original_abi.size_bytes != span.size_bytes
                    or span.source_allocation_ref != allocation_ref
                    or cursor != span.external_address
                    or original_abi.die_id != 0
                    or original_abi.address + original_abi.size_bytes
                       > window.resident_hbm_capacity_bytes):
                raise SchemaError("linked ABI/role/P3 external source mapping changed",
                                  path=f"physical_adamw.{role}.{span.state_ref}")
            piece = payload[cursor - allocation.address:
                            cursor - allocation.address + span.size_bytes]
            if len(piece) != span.size_bytes:
                raise SchemaError("seed physical ABI slice truncated",
                                  path=f"physical_adamw.{span.state_ref}")
            state = WorkloadStateInventoryItem.create(
                logical_name="physical.adamw." + span.state_ref,
                object_kind=expected_object_kind,
                logical_rank=len(original_abi.shape),
                owner_domain_ref="die:0",
                size_bytes=span.size_bytes, writable=True,
            )
            version = MemoryStateVersion.create(
                state_ref=state.id, generation=0,
                predecessor_ref=None, writable=True,
            )
            pinned = MemoryAllocationRequest.create(
                state_version_ref=version.id, object_kind=expected_object_kind,
                tier=MemoryTier.HBM, location_ref="die:0",
                size_bytes=span.size_bytes, alignment_bytes=16,
                lifetime_start=0, lifetime_end_exclusive=2,
                pinned_address=original_abi.address,
            )
            inventory.append(state)
            versions.append(version)
            physical_requests.append(pinned)
            state_values.append(PhysicalAdamwStateValue(
                span.state_ref, original_abi.id, allocation_ref, role,
                span.external_address, original_abi.address,
                span.hbm_address, span.size_bytes, _digest(piece),
                state.id, version.id, pinned.id,
            ))
            cursor += span.size_bytes
        if cursor != allocation.address + request.size_bytes:
            raise SchemaError("five-role source allocation bytes contain a gap",
                              path=f"physical_adamw.{role}")
        role_values.append(PhysicalAdamwRoleValue(
            role, count, size,
            _digest(b"".join(payload[span.external_address - allocation.address:
                                      span.external_address - allocation.address
                                      + span.size_bytes]
                             for span in sorted(spans, key=lambda item: item.state_ref))),
        ))
    if (len(inventory) != 83 or sum(item.size_bytes for item in inventory) != 32100
            or len({item.id for item in inventory}) != 83
            or len({item.pinned_address for item in physical_requests}) != 83):
        raise SchemaError("83 role state inventory/address alias/incomplete",
                          path="physical_adamw.inventory")
    workspace_requests = tuple(item for item in source.memory_plan.requests
                               if item.tier is MemoryTier.HBM)
    workspace_versions = tuple(item for item in source.memory_plan.state_versions
                               if item.id in {r.state_version_ref for r in workspace_requests})
    capacity = tuple(item for item in source.memory_plan.capacities
                     if item.tier is MemoryTier.HBM)
    if (len(capacity) != 1 or capacity[0].location_ref != "die:0"
            or capacity[0].capacity_bytes != window.resident_hbm_capacity_bytes
            or not workspace_requests or len(workspace_versions)
               != len({item.state_version_ref for item in workspace_requests})):
        raise SchemaError("P3 signed workspace/physical capacity incomplete",
                          path="physical_adamw.hbm")
    all_requests = (*workspace_requests, *physical_requests)
    try:
        plan_hierarchical_memory(
            capacities=capacity,
            state_versions=(*workspace_versions, *versions),
            requests=all_requests,
        )
    except SchemaError as error:
        if error.code != "memory_capacity_exceeded":
            raise
        rejection = str(error)
    else:
        raise SchemaError("83 ABI resident-only physical states unexpectedly fit low HBM",
                          path="physical_adamw.capacity")
    return PhysicalAdamwAuthority(
        source.request_digest, resident.request_digest,
        canonical_digest(source.memory_plan),
        canonical_digest(original.manifest),
        paged.linked_manifest_digests,
        canonical_digest(program), paged.digest,
        tuple(sorted(state_values, key=lambda item: item.state_ref)),
        tuple(sorted(role_values, key=lambda item: item.kind)),
        canonical_digest(tuple(sorted(inventory, key=lambda item: item.logical_name))),
        canonical_digest((*workspace_versions, *versions)),
        canonical_digest(all_requests),
        "memory_capacity_exceeded", rejection,
        capacity[0].capacity_bytes, window.hbm_workspace_peak_bytes,
    )


__all__ = [
    "PhysicalAdamwStateValue", "PhysicalAdamwRoleValue", "PhysicalAdamwAuthority",
    "prove_physical_adamw_authority_and_capacity",
]
