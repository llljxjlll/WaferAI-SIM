"""Source-bound physical external StateABI authority for TP4 Dense inference.

P3's rank-local parameter aggregate has logical bytes. This pass declares one
external source per *linked* physical parameter/KV ABI before paging. It does
not change the useful Dense operation graph or claim a native multi-Die run.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import RegionManifest, StateKind
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


_DECL_VERSION = "wafer_frontend.dense_inference_rect_physical_state_decl/v1alpha1"
_KV = (StateKind.KV_KEY, StateKind.KV_VALUE)


@dataclass(frozen=True, slots=True)
class PhysicalInferenceStateDecl:
    id: str
    role: str
    die_id: int
    logical_rank: int
    linked_state_refs: tuple[str, str, str]
    linked_hbm_binding_refs: tuple[str, str, str]
    linked_state_abi_ids: tuple[str, str, str]
    source_hbm_address: int
    physical_bytes: int
    p3_rank_inventory_ref: str
    inventory_ref: str
    source_version_ref: str
    external_allocation_ref: str

    @classmethod
    def create(cls, **fields: object) -> "PhysicalInferenceStateDecl":
        return cls(id=stable_artifact_id(
            "dense_inference_rect_physical_state_decl", fields,
            schema_version=_DECL_VERSION,
        ), **fields)


@dataclass(frozen=True, slots=True)
class PhysicalInferenceOffloadSource:
    resident_materialization_digest: str
    source_linked_manifest_digests: tuple[str, str, str]
    external_materialization_digest: str
    declarations: tuple[PhysicalInferenceStateDecl, ...]
    external_manifest: WorkloadMaterializationManifest
    hbm_capacities: tuple[MemoryTierCapacity, ...]
    external_capacity: MemoryTierCapacity
    resident_rejection: str
    p3_parameter_bytes_per_die: int
    physical_parameter_bytes_per_die: int
    physical_kv_bytes_per_die: int

    @property
    def digest(self) -> str:
        return canonical_digest((
            self.resident_materialization_digest,
            self.source_linked_manifest_digests,
            self.external_materialization_digest, self.declarations,
            self.hbm_capacities, self.external_capacity, self.resident_rejection,
        ))


def _abis(manifest):
    manifest.validate("dense_inference_rect.source_linked")
    result = {}
    for fragment in manifest.fragments:
        for abi in (fragment.fragment.state_abi
                    if isinstance(fragment, RegionManifest) else fragment.state_abi):
            old = result.setdefault(abi.id, abi)
            if old != abi:
                raise SchemaError("shared physical StateABI definitions conflict",
                                  path="source_linked.fragments.state_abi")
    bound = {item.state_abi_id for item in manifest.state_operand_bindings}
    if bound != set(result):
        raise SchemaError("every source StateABI requires a linked LSU binding",
                          path="source_linked.state_operand_bindings")
    return result


def _useful_graph_digest(manifest):
    """Compare useful operations by typed names, not memory-policy IDs."""
    graph = manifest.logical_graph
    states = {item.id: item for item in graph.state_versions}
    values = {item.id: item for item in graph.tensor_values}
    order = {item.id: item.sequence_index for item in graph.operations}

    def state(ref):
        item = states[ref]
        return (item.logical_name, item.kind.value, item.version,
                item.layer, item.expert)

    def value(ref):
        item = values[ref]
        return (item.logical_name, item.shape, item.dtype.value,
                item.size_bytes, state(item.state_ref), item.logical_rank,
                item.tp_shard)

    return canonical_digest(tuple(
        (item.sequence_index, item.kind.value, item.phase, item.step,
         item.layer, item.expert, item.parameter_ref,
         tuple(state(ref) for ref in item.reads),
         tuple(state(ref) for ref in item.writes),
         tuple(value(ref) for ref in item.input_value_refs),
         tuple(value(ref) for ref in item.output_value_refs),
         tuple(order[ref] for ref in item.deps))
        for item in graph.operations
    ))


def build_dense_inference_rect_physical_source(
    sequence, *, hbm_capacity_bytes: int = 12288,
    external_capacity_bytes: int = 131072,
) -> PhysicalInferenceOffloadSource:
    """Declare all 76 real TP4 StateABIs as signed physical external sources."""

    sequence.validate()
    resident = sequence.materialization
    active = tuple(resident.placement.active_die_ids)
    if (len(active) != 4 or set(active) != {0, 1, 2, 3}
            or (resident.request.mesh.rows, resident.request.mesh.columns)
            not in ((1, 4), (2, 2), (4, 1))
            or hbm_capacity_bytes <= 0 or external_capacity_bytes <= 0):
        raise SchemaError("requires a physical four-Die rectangle and bounded homes",
                          path="physical_source.mesh")
    home = {
        int(item.location_ref.removeprefix("die:")): item
        for item in resident.memory_plan.capacities
        if item.tier is MemoryTier.HBM and item.location_ref.startswith("die:")
    }
    if set(home) != set(active):
        raise SchemaError("resident source lacks exact four HBM owners",
                          path="physical_source.capacities")
    hbm = tuple(MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref=f"die:{die}",
        base_address=home[die].base_address,
        capacity_bytes=hbm_capacity_bytes,
        alignment_bytes=home[die].alignment_bytes,
    ) for die in active)
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
        raise SchemaError("same-model resident source fits bounded HBM",
                          path="physical_source.hbm_capacity_bytes")
    request = resident.request
    offload_request = WorkloadRunRequest.create(
        family=request.family, model=request.model, steps=request.steps,
        mesh=request.mesh, parallel=request.parallel,
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
            external_tier_ref=external.location_ref,
        ), optimizer=request.optimizer, execution=request.execution,
    )
    generic = materialize_workload_preflight(
        offload_request, resident.capability, capacities=(external, *hbm),
    )
    if (_useful_graph_digest(generic) != _useful_graph_digest(resident)
            or generic.placement != resident.placement):
        raise SchemaError("offload changed useful graph or four-Die placement",
                          path="physical_source.offload")
    linked = tuple(segment.linked_manifest for segment in sequence.segments)
    if len(linked) != 3:
        raise SchemaError("requires Prefill+2Decode source manifests",
                          path="physical_source.sequence")
    by_step = tuple(_abis(item) for item in linked)
    by_key = tuple({
        (abi.die_id, abi.kind, abi.address): abi
        for abi in table.values()
    } for table in by_step)
    if any(len(table) != 76 for table in by_step) or any(len(table) != 76 for table in by_key):
        raise SchemaError("physical TP4 requires 76 unique linked StateABIs per segment",
                          path="physical_source.state_abi")
    if set(by_key[0]) != set(by_key[1]) or set(by_key[0]) != set(by_key[2]):
        raise SchemaError("three linked segments changed physical state inventory",
                          path="physical_source.state_abi")
    rank_by_die = {item.die_id: item.logical_rank
                   for item in resident.placement.rank_placements}
    owner = {
        (item.object_kind, item.logical_rank): item
        for item in resident.state_inventory
        if item.object_kind in (MemoryObjectKind.PARAMETER, MemoryObjectKind.KV)
    }
    if len(owner) != 8 or set(rank_by_die) != set(active):
        raise SchemaError("P3 rank-local parameter/KV owners are not one-to-one",
                          path="physical_source.rank_owners")
    logical_parameter_bytes = {owner[(MemoryObjectKind.PARAMETER, rank_by_die[d])].size_bytes
                               for d in active}
    if logical_parameter_bytes != {14416}:
        raise SchemaError("P3 logical rank-local parameter aggregate changed",
                          path="physical_source.p3_parameters")
    for die in active:
        weights = sorted(
            (abi for (d, kind, _), abi in by_key[0].items()
             if d == die and kind is StateKind.PARAMETER),
            key=lambda item: item.address,
        )
        kvs = sorted(
            (abi for (d, kind, _), abi in by_key[2].items()
             if d == die and kind in _KV), key=lambda item: item.address,
        )
        if len(weights) != 15 or len(kvs) != 4 or (
            sum(item.size_bytes for item in weights) != 26944
            or sum(item.size_bytes for item in kvs) != 1536
            or owner[(MemoryObjectKind.KV, rank_by_die[die])].size_bytes != 1536
        ):
            raise SchemaError("physical parameter/KV bytes differ from true TP4 ABIs",
                              path=f"physical_source.die[{die}]")
        cursor = weights[0].address
        for abi in weights:
            if abi.address != cursor:
                raise SchemaError("physical weights are not a contiguous linked source",
                                  path=f"physical_source.die[{die}]")
            cursor += abi.size_bytes
        if tuple(item.address for item in kvs) != tuple(cursor + 384 * i for i in range(4)):
            raise SchemaError("KV source homes lack four stable versioned pages",
                              path=f"physical_source.die[{die}]")
        for key, latest in by_key[2].items():
            if key[0] != die:
                continue
            chain = tuple(step[key] for step in by_key)
            if any(item.address != latest.address for item in chain):
                raise SchemaError("linked state home moved between segments",
                                  path=f"physical_source.die[{die}]")
            if latest.kind is StateKind.PARAMETER:
                if any(item.size_bytes != latest.size_bytes
                       or item.state_ref != latest.state_ref
                       or item.hbm_binding_ref != latest.hbm_binding_ref
                       for item in chain):
                    raise SchemaError("parameter source physical bytes changed",
                                      path=f"physical_source.die[{die}]")
            elif latest.kind in _KV:
                if tuple(item.size_bytes for item in chain) != (256, 320, 384):
                    raise SchemaError("KV physical page version growth changed",
                                      path=f"physical_source.die[{die}]")
            else:
                raise SchemaError("unexpected linked persistent state role",
                                  path=f"physical_source.die[{die}]")

    lifetime_end = max(item.lifetime_end_exclusive
                       for item in generic.memory_plan.requests)
    physical_inventory = []
    physical_versions = []
    physical_requests = []
    source_by_inventory = {}
    for key, latest in sorted(by_key[2].items(), key=lambda pair: (
        pair[0][0], pair[1].address, pair[1].id,
    )):
        die, kind, _ = key
        object_kind = (MemoryObjectKind.PARAMETER if kind is StateKind.PARAMETER
                       else MemoryObjectKind.KV)
        p3 = owner[(object_kind, rank_by_die[die])]
        item = WorkloadStateInventoryItem.create(
            logical_name=f"{object_kind.value}.linked.{latest.state_ref}.rank{p3.logical_rank}",
            object_kind=object_kind, logical_rank=p3.logical_rank,
            owner_domain_ref=p3.owner_domain_ref,
            size_bytes=latest.size_bytes, writable=object_kind is MemoryObjectKind.KV,
        )
        version = MemoryStateVersion.create(
            state_ref=item.id, generation=0, predecessor_ref=None,
            writable=object_kind is MemoryObjectKind.KV,
        )
        request_item = MemoryAllocationRequest.create(
            state_version_ref=version.id, object_kind=object_kind,
            tier=MemoryTier.EXTERNAL, location_ref=external.location_ref,
            size_bytes=latest.size_bytes, alignment_bytes=16,
            lifetime_start=0, lifetime_end_exclusive=lifetime_end,
        )
        physical_inventory.append(item)
        physical_versions.append(version)
        physical_requests.append(request_item)
        source_by_inventory[item.id] = (key, p3.id, version, request_item)
    retained = tuple(item for item in generic.state_inventory
                     if item.object_kind not in (MemoryObjectKind.PARAMETER, MemoryObjectKind.KV))
    retained_refs = {item.id for item in retained}
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
        dirty_state_version_refs=(*dirty_old, *(item.id for item in physical_versions
                                                if item.writable)),
    )
    manifest = WorkloadMaterializationManifest.create(
        request=offload_request, capability=resident.capability,
        logical_graph=generic.logical_graph,
        state_inventory=tuple(sorted((*retained, *physical_inventory),
                                     key=lambda item: item.logical_name)),
        placement=generic.placement,
        transport_requests=generic.transport_requests,
        transport_plan=generic.transport_plan,
        memory_plan=plan,
    )
    allocations = {item.request_ref: item for item in plan.allocations}
    declarations = []
    for item in physical_inventory:
        key, p3_ref, version, request_item = source_by_inventory[item.id]
        latest = by_key[2][key]
        declarations.append(PhysicalInferenceStateDecl.create(
            role="parameter" if latest.kind is StateKind.PARAMETER else latest.kind.value,
            die_id=latest.die_id, logical_rank=rank_by_die[latest.die_id],
            linked_state_refs=tuple(step[key].state_ref for step in by_key),
            linked_hbm_binding_refs=tuple(step[key].hbm_binding_ref for step in by_key),
            linked_state_abi_ids=tuple(step[key].id for step in by_key),
            source_hbm_address=latest.address,
            physical_bytes=latest.size_bytes,
            p3_rank_inventory_ref=p3_ref, inventory_ref=item.id,
            source_version_ref=version.id,
            external_allocation_ref=allocations[request_item.id].id,
        ))
    if len(declarations) != 76 or sum(item.physical_bytes for item in declarations) != 113920:
        raise SchemaError("physical external 60 weight+16 KV declarations incomplete",
                          path="physical_source.declarations")
    return PhysicalInferenceOffloadSource(
        resident_materialization_digest=resident.digest,
        source_linked_manifest_digests=tuple(canonical_digest(item) for item in linked),
        external_materialization_digest=manifest.digest,
        declarations=tuple(declarations), external_manifest=manifest,
        hbm_capacities=hbm, external_capacity=external,
        resident_rejection=resident_rejection,
        p3_parameter_bytes_per_die=14416,
        physical_parameter_bytes_per_die=26944,
        physical_kv_bytes_per_die=1536,
    )


__all__ = ["PhysicalInferenceStateDecl", "PhysicalInferenceOffloadSource",
           "build_dense_inference_rect_physical_source"]
