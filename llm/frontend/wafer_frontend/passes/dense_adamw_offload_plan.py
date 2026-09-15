"""Source-bound five-aggregate Dense AdamW blocking external DMA plan."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.external_memory import (
    ExternalMemoryConnection, ExternalMemoryFabric, ExternalMemoryLink,
)
from ..schema.memory_plan import MemoryObjectKind, MemoryTier
from ..schema.offload import (
    BlockingOffloadPlan, OffloadChunk, OffloadEventKind,
    OffloadStateMapping, OffloadTraceEvent,
)
from ..schema.serde import canonical_digest
from ..schema.workload_materialization import WorkloadMaterializationManifest
from ..schema.workload_run import WorkloadMemoryMode
from .offload import plan_offload_blocking


def plan_dense_adamw_source_offload(
    source: WorkloadMaterializationManifest,
) -> BlockingOffloadPlan:
    """Plan all five exact source groups; runtime residency needs an extra proof."""

    source.validate("source")
    if (
        source.request.optimizer is None
        or source.request.optimizer.kind.value != "adamw"
        or source.request.memory.mode is not WorkloadMemoryMode.EXTERNAL_OFFLOAD
        or source.request.parallel.tp != 1
        or source.request.mesh.rows != 1
        or source.request.mesh.columns != 1
    ):
        raise SchemaError("requires a 1x1 AdamW EXTERNAL_OFFLOAD source", path="source")
    capacities = source.memory_plan.capacities
    hbm = tuple(item for item in capacities if item.tier is MemoryTier.HBM)
    external = tuple(item for item in capacities if item.tier is MemoryTier.EXTERNAL)
    if len(hbm) != 1 or len(external) != 1:
        raise SchemaError("requires one real HBM and external capacity", path="source")
    hbm_capacity, external_capacity = hbm[0], external[0]
    link = ExternalMemoryLink.create(
        external_capacity_ref=external_capacity.id,
        ingress_die_id=0, bytes_per_cycle=256, latency_cycles=2,
        queue_depth=2, max_outstanding=2,
    )
    connection = ExternalMemoryConnection.create(
        link_ref=link.id, hbm_capacity_ref=hbm_capacity.id,
        target_die_id=0, route_die_ids=(0,), route_latency_cycles=0,
        route_bytes_per_cycle=None,
    )
    fabric = ExternalMemoryFabric.create(
        external_capacities=(external_capacity,),
        hbm_capacities=(hbm_capacity,), links=(link,),
        connections=(connection,),
    )
    requests = {item.id: item for item in source.memory_plan.requests}
    versions = {item.id: item for item in source.memory_plan.state_versions}
    source_allocations = tuple(
        item for item in source.memory_plan.allocations
        if requests[item.request_ref].tier is MemoryTier.EXTERNAL
        and requests[item.request_ref].object_kind in (
            MemoryObjectKind.PARAMETER, MemoryObjectKind.OPTIMIZER,
        )
    )
    groups = []
    mappings = []
    for allocation in source_allocations:
        request = requests[allocation.request_ref]
        version = versions[request.state_version_ref]
        chunk = OffloadChunk.create(
            initial_version=version, object_kind=request.object_kind,
            size_bytes=request.size_bytes,
            alignment_bytes=request.alignment_bytes,
            external_capacity_ref=external_capacity.id,
            external_address=allocation.address,
            connection_ref=connection.id,
        )
        groups.append(chunk)
        mappings.append(OffloadStateMapping.create(
            chunk_ref=chunk.id, source_state_version_ref=version.id,
            source_allocation_ref=allocation.id,
        ))
    if len(groups) != 5 or sum(group.size_bytes for group in groups) != 32100:
        raise SchemaError("five full exact 83-StateABI source groups required", path="source")
    events = tuple(
        OffloadTraceEvent.create(
            ordinal=index, kind=OffloadEventKind.READ, chunk_ref=chunk.id,
        )
        for index, chunk in enumerate(groups)
    ) + tuple(
        OffloadTraceEvent.create(
            ordinal=index + len(groups), kind=OffloadEventKind.WRITE,
            chunk_ref=chunk.id,
        )
        for index, chunk in enumerate(groups)
    )
    plan = plan_offload_blocking(
        request_digest=source.request.digest,
        logical_graph_digest=source.logical_graph_digest,
        source_memory_plan_digest=canonical_digest(source.memory_plan),
        source_memory_plan=source.memory_plan,
        state_mappings=tuple(mappings), fabric=fabric,
        chunks=tuple(groups), events=events,
    )
    if (
        plan.stats.bring_in_count != 5
        or plan.stats.dirty_writeback_count != 5
        or plan.stats.final_dirty_chunks != 0
        or plan.stats.final_resident_chunks != 0
        or plan.stats.final_pin_count != 0
        or plan.stats.transfer_bytes != 2 * 32100
    ):
        raise SchemaError("five full bring-in/dirty-writeback dependencies missing", path="plan")
    return plan


__all__ = ["plan_dense_adamw_source_offload"]
