"""Check resident-only HBM against the real linked Dense training StateABIs.

The P3 workload graph remains the same. Its aggregated parameter inventory is
replaced by one pinned parameter allocation for each production linked StateABI
before the native MemoryPlan allocator checks the actual Die HBM homes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import RegionManifest
from ..schema.memory_plan import (
    MemoryAllocationRequest, MemoryObjectKind, MemoryStateVersion, MemoryTier,
)
from ..schema.serde import canonical_digest
from .dense_training_physical_offload_source import PhysicalTrainingOffloadSource
from .memory_plan import plan_hierarchical_memory


@dataclass(frozen=True, slots=True)
class PhysicalResidentCapacityRejection:
    source_digest: str
    workload_request_digest: str
    linked_manifest_digest: str
    state_inventory_digest: str
    hbm_requests_digest: str
    linked_state_abi_count: int
    linked_state_logical_bytes: int
    linked_state_padding_bytes: int
    hbm_capacity_bytes_per_die: int
    rejection_code: str
    rejection: str

    @property
    def digest(self) -> str:
        return canonical_digest(self)


def reject_physically_resident_dense_training(
    sequence, source: PhysicalTrainingOffloadSource,
) -> PhysicalResidentCapacityRejection:
    """Prove the same source-bound training request cannot keep 60 ABIs resident.

    Only the physical parameter declarations change. The resident request,
    graph, nonparameter versions/requests, Die placement and HBM homes come
    from the exact accepted P3 compile, not from an external/offload sidecar.
    """

    sequence.validate()
    resident = sequence.materialization
    linked = sequence.segments[0].linked_program.manifest
    if (source.resident_materialization_digest != resident.digest or
            source.resident_linked_manifest_digest != canonical_digest(linked) or
            source.external_manifest.request.model != resident.request.model or
            source.external_manifest.request.steps != resident.request.steps or
            source.external_manifest.placement != resident.placement):
        raise SchemaError("physical source differs from resident request/link/placement",
                          path="physical_resident.source")
    source.external_manifest.validate()
    linked_abis = {
        abi.id: abi
        for fragment in linked.fragments
        for abi in (fragment.fragment.state_abi
                    if isinstance(fragment, RegionManifest) else fragment.state_abi)
    }
    declaration_by_abi = {item.linked_state_abi_id: item for item in source.declarations}
    if len(source.declarations) != len(linked_abis) or set(declaration_by_abi) != set(linked_abis):
        raise SchemaError("physical declarations do not match all linked StateABIs",
                          path="physical_resident.declarations")

    inventory_by_id = {item.id: item for item in source.external_manifest.state_inventory}
    retained_inventory = tuple(item for item in resident.state_inventory
                               if item.object_kind is not MemoryObjectKind.PARAMETER)
    retained_state_refs = {item.id for item in retained_inventory}
    retained_versions = tuple(item for item in resident.memory_plan.state_versions
                              if item.state_ref in retained_state_refs)
    retained_version_ids = {item.id for item in retained_versions}
    retained_requests = tuple(item for item in resident.memory_plan.requests
                              if item.state_version_ref in retained_version_ids)
    physical_inventory = []
    physical_versions = []
    physical_requests = []
    for declaration in sorted(source.declarations,
                              key=lambda item: (item.die_id, item.hbm_address, item.id)):
        abi = linked_abis[declaration.linked_state_abi_id]
        state = inventory_by_id.get(declaration.inventory_ref)
        if (state is None or state.object_kind is not MemoryObjectKind.PARAMETER or
                abi.die_id != declaration.die_id or
                abi.address != declaration.hbm_address or
                abi.size_bytes != declaration.logical_bytes or
                state.size_bytes != abi.size_bytes or
                state.logical_rank != declaration.logical_rank):
            raise SchemaError("declared parameter does not match linked StateABI",
                              path="physical_resident.declarations")
        version = MemoryStateVersion.create(
            state_ref=state.id, generation=0, predecessor_ref=None, writable=True,
        )
        if version.id != declaration.source_version_ref:
            raise SchemaError("physical source version differs from linked inventory",
                              path="physical_resident.declarations")
        physical_inventory.append(state)
        physical_versions.append(version)
        physical_requests.append(MemoryAllocationRequest.create(
            state_version_ref=version.id, object_kind=MemoryObjectKind.PARAMETER,
            tier=MemoryTier.HBM, location_ref=f"die:{abi.die_id}",
            size_bytes=abi.size_bytes, alignment_bytes=16,
            lifetime_start=0,
            lifetime_end_exclusive=max(request.lifetime_end_exclusive
                                       for request in resident.memory_plan.requests),
            pinned_address=abi.address,
        ))
    if (len({item.id for item in physical_inventory}) != len(linked_abis) or
            len({item.id for item in (*retained_versions, *physical_versions)}) !=
                len(retained_versions) + len(physical_versions) or
            len(source.hbm_capacities) != len(resident.placement.active_die_ids)):
        raise SchemaError("physical inventory/Die homes are incomplete",
                          path="physical_resident.inventory")
    all_inventory = tuple(sorted((*retained_inventory, *physical_inventory),
                                 key=lambda item: item.logical_name))
    all_versions = (*retained_versions, *physical_versions)
    all_requests = (*retained_requests, *physical_requests)
    if ({item.state_ref for item in all_versions} != {item.id for item in all_inventory} or
            any(item.tier is not MemoryTier.HBM for item in all_requests)):
        raise SchemaError("resident-only source must cover exact physical inventory in HBM",
                          path="physical_resident.requests")

    try:
        plan_hierarchical_memory(
            capacities=source.hbm_capacities,
            state_versions=all_versions,
            requests=all_requests,
        )
    except SchemaError as error:
        if error.code != "memory_capacity_exceeded":
            raise
        return PhysicalResidentCapacityRejection(
            source_digest=source.digest,
            workload_request_digest=resident.request.digest,
            linked_manifest_digest=canonical_digest(linked),
            state_inventory_digest=canonical_digest(all_inventory),
            hbm_requests_digest=canonical_digest(all_requests),
            linked_state_abi_count=len(linked_abis),
            linked_state_logical_bytes=sum(item.size_bytes for item in linked_abis.values()),
            linked_state_padding_bytes=sum(
                max(item.address + item.size_bytes for item in linked_abis.values()
                    if item.die_id == die) -
                min(item.address for item in linked_abis.values()
                    if item.die_id == die) -
                sum(item.size_bytes for item in linked_abis.values()
                    if item.die_id == die)
                for die in resident.placement.active_die_ids
            ),
            hbm_capacity_bytes_per_die=source.hbm_capacities[0].capacity_bytes,
            rejection_code=error.code, rejection=str(error),
        )
    raise SchemaError("60 linked parameter states unexpectedly fit in resident-only HBM",
                      path="physical_resident.capacity")


__all__ = ["PhysicalResidentCapacityRejection",
           "reject_physically_resident_dense_training"]
