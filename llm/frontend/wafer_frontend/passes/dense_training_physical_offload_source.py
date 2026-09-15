"""Materialize signed per-StateABI external sources for linked Dense SGD.

The generic P3 aggregate parameter inventory is kept for the workload graph.
For offload, the actual two-step production link supplies an additional,
physical parameter inventory with exactly one source allocation per StateABI.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import RegionManifest
from ..schema.common import stable_artifact_id
from ..schema.memory_plan import (
    MemoryAllocationRequest, MemoryObjectKind, MemoryStateVersion, MemoryTier,
    MemoryTierCapacity, ResidencyStatus,
)
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import (
    WorkloadMaterializationManifest, WorkloadStateInventoryItem,
)
from ..schema.workload_run import (
    WorkloadMemoryMode, WorkloadMemoryPolicy, WorkloadRunRequest,
)
from .memory_plan import plan_hierarchical_memory
from .workload_materialization import materialize_workload_preflight


_DECL_VERSION = "wafer_frontend.dense_training_physical_state_decl/v1alpha1"


@dataclass(frozen=True, slots=True)
class PhysicalTrainingStateDecl:
    id: str
    linked_state_abi_id: str
    linked_state_ref: str
    p3_parameter_refs: tuple[str, ...]
    die_id: int
    logical_rank: int
    hbm_address: int
    logical_bytes: int
    inventory_ref: str
    source_version_ref: str
    external_allocation_ref: str

    @classmethod
    def create(cls, **fields: object) -> "PhysicalTrainingStateDecl":
        result = cls(
            id=stable_artifact_id("dense_training_physical_state_decl", fields,
                                  schema_version=_DECL_VERSION),
            **fields,
        )
        return result


@dataclass(frozen=True, slots=True)
class PhysicalTrainingOffloadSource:
    resident_materialization_digest: str
    resident_linked_manifest_digest: str
    external_materialization_digest: str
    declarations: tuple[PhysicalTrainingStateDecl, ...]
    external_manifest: WorkloadMaterializationManifest
    hbm_capacities: tuple[MemoryTierCapacity, ...]
    external_capacity: MemoryTierCapacity
    resident_rejection: str

    @property
    def digest(self) -> str:
        return canonical_digest((
            self.resident_materialization_digest,
            self.resident_linked_manifest_digest,
            self.external_materialization_digest,
            self.declarations,
            self.hbm_capacities,
            self.external_capacity,
            self.resident_rejection,
        ))


def build_dense_training_physical_offload_source(
    sequence,
    *,
    hbm_capacity_bytes: int,
    external_capacity_bytes: int,
) -> PhysicalTrainingOffloadSource:
    """Make a validated P3 workload manifest with all 60 physical sources.

    Every source is traced through a validated P3 parameter binding and a real
    legacy shard to one exact linked StateABI, including shared fused states.
    The resulting manifest retains the original graph/request placement while
    supplying a separate typed physical parameter inventory and MemoryPlan.
    """

    sequence.validate()
    resident = sequence.materialization
    linked = sequence.segments[0].linked_program.manifest
    active = resident.placement.active_die_ids
    homes = {
        int(item.location_ref.removeprefix("die:")): item
        for item in resident.memory_plan.capacities
        if item.tier is MemoryTier.HBM and item.location_ref.startswith("die:")
    }
    if set(homes) != set(active) or hbm_capacity_bytes <= 0 or external_capacity_bytes <= 0:
        raise SchemaError("every active Die needs a finite real HBM home",
                          path="physical_offload.capacities")
    hbm = tuple(MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref=f"die:{die}",
        base_address=homes[die].base_address,
        capacity_bytes=hbm_capacity_bytes,
        alignment_bytes=homes[die].alignment_bytes,
    ) for die in sorted(active))
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL, location_ref="host:0", base_address=0,
        capacity_bytes=external_capacity_bytes, alignment_bytes=64,
    )
    try:
        materialize_workload_preflight(
            resident.request, resident.capability, capacities=hbm,
        )
    except SchemaError as error:
        if error.code != "memory_capacity_exceeded":
            raise
        resident_rejection = str(error)
    else:
        raise SchemaError("same-model resident-only source fits bounded HBM",
                          path="physical_offload.hbm_capacity_bytes")
    request = resident.request
    offload_request = WorkloadRunRequest.create(
        family=request.family, model=request.model, steps=request.steps,
        mesh=request.mesh, parallel=request.parallel,
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
            external_tier_ref=external.location_ref,
        ),
        optimizer=request.optimizer, execution=request.execution,
    )
    generic = materialize_workload_preflight(
        offload_request, resident.capability,
        capacities=(external, *hbm),
    )
    abi_by_id = {
        abi.id: abi
        for fragment in linked.fragments
        for abi in (fragment.fragment.state_abi
                    if isinstance(fragment, RegionManifest) else fragment.state_abi)
    }
    if not abi_by_id:
        raise SchemaError("linked training program has no physical states",
                          path="linked.fragments.state_abi")
    rank_by_die = {item.die_id: item.logical_rank
                   for item in resident.placement.rank_placements}
    source_param_by_rank = {
        item.logical_rank: item
        for item in resident.state_inventory
        if item.object_kind is MemoryObjectKind.PARAMETER
    }
    referenced: dict[str, set[str]] = {}
    for binding in sequence.segments[0].parameter_bindings:
        for shard in binding.legacy_shards:
            for abi_ref, rank, address in zip(
                shard.state_abi_refs, shard.owner_ranks, shard.hbm_addresses
            ):
                abi = abi_by_id.get(abi_ref)
                if (abi is None or abi.die_id != rank or abi.address != address or
                        rank_by_die.get(abi.die_id) != rank):
                    raise SchemaError("P3/legacy shard differs from physical StateABI",
                                      path="physical_offload.legacy_shards")
                referenced.setdefault(abi_ref, set()).add(binding.parameter_ref)
    if set(referenced) != set(abi_by_id):
        raise SchemaError("P3 bindings do not cover every physical StateABI",
                          path="physical_offload.legacy_shards")

    physical_inventory = []
    physical_versions = []
    physical_requests = []
    by_inventory: dict[str, tuple[object, tuple[str, ...]]] = {}
    lifetime_end = max(
        item.lifetime_end_exclusive for item in generic.memory_plan.requests
    )
    for abi in sorted(abi_by_id.values(), key=lambda x: (x.die_id, x.address, x.id)):
        owner = source_param_by_rank.get(rank_by_die[abi.die_id])
        if owner is None:
            raise SchemaError("physical StateABI has no P3 rank-local parameter owner",
                              path="physical_offload.legacy_shards")
        inventory = WorkloadStateInventoryItem.create(
            logical_name=f"parameter.linked.{abi.state_ref}.rank{owner.logical_rank}",
            object_kind=MemoryObjectKind.PARAMETER,
            logical_rank=owner.logical_rank,
            owner_domain_ref=owner.owner_domain_ref,
            size_bytes=abi.size_bytes, writable=True,
        )
        version = MemoryStateVersion.create(
            state_ref=inventory.id, generation=0,
            predecessor_ref=None, writable=True,
        )
        physical_inventory.append(inventory)
        physical_versions.append(version)
        physical_requests.append(MemoryAllocationRequest.create(
            state_version_ref=version.id,
            object_kind=MemoryObjectKind.PARAMETER,
            tier=MemoryTier.EXTERNAL, location_ref=external.location_ref,
            size_bytes=abi.size_bytes, alignment_bytes=16,
            lifetime_start=0, lifetime_end_exclusive=lifetime_end,
        ))
        by_inventory[inventory.id] = (abi, tuple(sorted(referenced[abi.id])))

    retained_inventory = tuple(item for item in generic.state_inventory
                               if item.object_kind is not MemoryObjectKind.PARAMETER)
    retained_refs = {item.id for item in retained_inventory}
    retained_versions = tuple(item for item in generic.memory_plan.state_versions
                              if item.state_ref in retained_refs)
    retained_version_ids = {item.id for item in retained_versions}
    retained_requests = tuple(item for item in generic.memory_plan.requests
                              if item.state_version_ref in retained_version_ids)
    dirty_old = tuple(sorted({item.state_version_ref
                              for item in generic.memory_plan.residencies
                              if item.status is ResidencyStatus.DIRTY and
                              item.state_version_ref in retained_version_ids}))
    plan = plan_hierarchical_memory(
        capacities=(external, *hbm),
        state_versions=(*retained_versions, *physical_versions),
        requests=(*retained_requests, *physical_requests),
        dirty_state_version_refs=(*dirty_old, *(item.id for item in physical_versions)),
    )
    physical_manifest = WorkloadMaterializationManifest.create(
        request=offload_request, capability=resident.capability,
        logical_graph=generic.logical_graph,
        state_inventory=tuple(sorted((*retained_inventory, *physical_inventory),
                                     key=lambda item: item.logical_name)),
        placement=generic.placement,
        transport_requests=generic.transport_requests,
        transport_plan=generic.transport_plan,
        memory_plan=plan,
    )
    by_version = {item.id: item for item in physical_versions}
    by_request = {item.state_version_ref: item
                  for item in physical_requests}
    allocations = {item.request_ref: item for item in plan.allocations}
    declarations = []
    for item in physical_inventory:
        abi, parameter_refs = by_inventory[item.id]
        version = next(version for version in by_version.values()
                       if version.state_ref == item.id)
        request_item = by_request[version.id]
        allocations_item = allocations[request_item.id]
        declarations.append(PhysicalTrainingStateDecl.create(
            linked_state_abi_id=abi.id,
            linked_state_ref=abi.state_ref,
            p3_parameter_refs=parameter_refs,
            die_id=abi.die_id, logical_rank=item.logical_rank,
            hbm_address=abi.address, logical_bytes=abi.size_bytes,
            inventory_ref=item.id, source_version_ref=version.id,
            external_allocation_ref=allocations_item.id,
        ))
    source = PhysicalTrainingOffloadSource(
        resident_materialization_digest=resident.digest,
        resident_linked_manifest_digest=canonical_digest(linked),
        external_materialization_digest=physical_manifest.digest,
        declarations=tuple(declarations), external_manifest=physical_manifest,
        hbm_capacities=hbm, external_capacity=external,
        resident_rejection=resident_rejection,
    )
    if (len(source.declarations) != len(abi_by_id) or
            {item.linked_state_abi_id for item in source.declarations} != set(abi_by_id)):
        raise SchemaError("physical offload declarations do not cover linked ABIs",
                          path="physical_offload.declarations")
    return source


__all__ = [
    "PhysicalTrainingOffloadSource", "PhysicalTrainingStateDecl",
    "build_dense_training_physical_offload_source",
]
