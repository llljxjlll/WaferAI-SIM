"""Lower source-bound per-StateABI Dense SGD restores and dirty writebacks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.external_dma_action_graph import (
    ExternalDmaRuntimeBinding, ExternalDmaRuntimePhaseMode,
)
from ..schema.external_dma_program import (
    ExternalDmaBackendBinding, ExternalDmaProbe, ExternalDmaSeed,
)
from ..schema.external_memory import (
    ExternalMemoryConnection, ExternalMemoryFabric, ExternalMemoryLink,
    ExternalTransferDirection,
)
from ..schema.memory_plan import MemoryObjectKind, MemoryTier
from ..schema.offload import (
    OffloadChunk, OffloadEventKind, OffloadStateMapping, OffloadTraceEvent,
)
from ..schema.serde import canonical_digest
from .dense_training_physical_offload_source import PhysicalTrainingOffloadSource
from .external_dma_action_graph import build_external_dma_action_graph
from .external_dma_program import finalize_external_dma_program
from .offload import plan_offload_blocking


@dataclass(frozen=True, slots=True)
class PhysicalTrainingOffloadCase:
    source: PhysicalTrainingOffloadSource
    plan: object
    program: object
    action_graph: object
    runtime_binding: ExternalDmaRuntimeBinding


def build_dense_training_physical_offload_program(
    source: PhysicalTrainingOffloadSource,
    state_seeds: Mapping[str, bytes],
) -> PhysicalTrainingOffloadCase:
    """Obtain every physical DMA descriptor from the validated blocking planner.

    External addresses come solely from the source MemoryPlan. HBM targets
    come solely from source-declared, actually linked StateABI addresses;
    neither is constructed as an unsigned, handwritten runtime descriptor.
    """

    manifest = source.external_manifest
    manifest.validate()
    if manifest.digest != source.external_materialization_digest:
        raise SchemaError("physical source manifest changed after binding",
                          path="physical_source.external_materialization_digest")
    declarations = source.declarations
    if not declarations or len({item.linked_state_ref for item in declarations}) != len(declarations):
        raise SchemaError("physical linked state declarations are not one-to-one",
                          path="physical_source.declarations")
    if (set(state_seeds) != {item.linked_state_ref for item in declarations} or
            any(type(state_seeds[item.linked_state_ref]) is not bytes or
                len(state_seeds[item.linked_state_ref]) != item.logical_bytes
                for item in declarations) or
            not any(any(seed) for seed in state_seeds.values())):
        raise SchemaError("nonzero external payload must exactly cover linked ABI bytes",
                          path="state_seeds")
    allocations = {item.id: item for item in manifest.memory_plan.allocations}
    requests = {item.id: item for item in manifest.memory_plan.requests}
    versions = {item.id: item for item in manifest.memory_plan.state_versions}
    external = source.external_capacity
    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id,
        ingress_die_id=min(item.die_id for item in declarations),
        bytes_per_cycle=256, latency_cycles=2,
        queue_depth=2, max_outstanding=2,
    )
    ingress = link.ingress_die_id
    connections = tuple(ExternalMemoryConnection.create(
        link_ref=link.id,
        hbm_capacity_ref=next(item.id for item in source.hbm_capacities
                              if item.location_ref == f"die:{die}"),
        target_die_id=die,
        route_die_ids=tuple(range(ingress, die + 1)),
        route_latency_cycles=die - ingress,
        route_bytes_per_cycle=None if die == ingress else 256,
    ) for die in sorted({item.die_id for item in declarations}))
    if any(item.target_die_id < ingress for item in connections):
        raise SchemaError("physical offload requires X-forward mesh ingress",
                          path="physical_source.declarations")
    fabric = ExternalMemoryFabric.create(
        external_capacities=(external,),
        hbm_capacities=source.hbm_capacities,
        links=(link,), connections=connections,
    )
    conn_by_die = {item.target_die_id: item for item in connections}
    chunks = []
    mappings = []
    decl_by_chunk = {}
    for declaration in declarations:
        allocation = allocations.get(declaration.external_allocation_ref)
        version = versions.get(declaration.source_version_ref)
        if allocation is None or version is None:
            raise SchemaError("physical declaration is outside signed source plan",
                              path="physical_source.declarations")
        request = requests[allocation.request_ref]
        if (version.state_ref != declaration.inventory_ref or
                request.state_version_ref != version.id or
                request.object_kind is not MemoryObjectKind.PARAMETER or
                request.tier is not MemoryTier.EXTERNAL or
                request.location_ref != external.location_ref or
                request.size_bytes != declaration.logical_bytes):
            raise SchemaError("physical StateABI does not match its source allocation",
                              path="physical_source.declarations")
        chunk = OffloadChunk.create(
            initial_version=version,
            object_kind=MemoryObjectKind.PARAMETER,
            size_bytes=request.size_bytes,
            alignment_bytes=request.alignment_bytes,
            external_capacity_ref=external.id,
            external_address=allocation.address,
            connection_ref=conn_by_die[declaration.die_id].id,
        )
        chunks.append(chunk)
        mappings.append(OffloadStateMapping.create(
            chunk_ref=chunk.id,
            source_state_version_ref=version.id,
            source_allocation_ref=allocation.id,
        ))
        decl_by_chunk[chunk.id] = declaration
    ordered = tuple(sorted(chunks, key=lambda item: (
        decl_by_chunk[item.id].die_id,
        decl_by_chunk[item.id].hbm_address,
    )))
    events = tuple(OffloadTraceEvent.create(
        ordinal=ordinal, kind=OffloadEventKind.WRITE, chunk_ref=chunk.id,
    ) for ordinal, chunk in enumerate(ordered))
    plan = plan_offload_blocking(
        request_digest=manifest.request_digest,
        logical_graph_digest=manifest.logical_graph_digest,
        source_memory_plan_digest=canonical_digest(manifest.memory_plan),
        source_memory_plan=manifest.memory_plan,
        state_mappings=tuple(mappings),
        fabric=fabric, chunks=tuple(chunks), events=events,
        pinned_hbm_addresses={chunk.id: decl_by_chunk[chunk.id].hbm_address
                              for chunk in chunks},
    )
    backend = tuple(ExternalDmaBackendBinding.create(
        hbm_capacity_ref=capacity.id,
        owner_die_id=int(capacity.location_ref.removeprefix("die:")),
        stack_id=int(capacity.location_ref.removeprefix("die:")),
        channel_id=0,
    ) for capacity in source.hbm_capacities)
    ordered_decls = tuple(sorted(declarations, key=lambda item: (
        item.die_id, item.hbm_address, item.linked_state_abi_id,
    )))
    seeds = tuple(ExternalDmaSeed.create(
        external_capacity_ref=external.id,
        address=allocations[item.external_allocation_ref].address,
        payload=state_seeds[item.linked_state_ref],
    ) for item in ordered_decls)
    probes = tuple(ExternalDmaProbe.create(
        external_capacity_ref=external.id,
        address=allocations[item.external_allocation_ref].address,
        expected_payload=state_seeds[item.linked_state_ref],
    ) for item in ordered_decls)
    program = finalize_external_dma_program(
        plan=plan,
        case_digest=canonical_digest(manifest.request.case_id),
        backend_bindings=backend,
        external_seeds=seeds,
        external_probes=probes,
    )
    graph = build_external_dma_action_graph(
        manifest=manifest, plan=plan, program=program,
    )
    runtime_binding = ExternalDmaRuntimeBinding.create(
        action_graph_digest=graph.digest,
        program_relative_path="artifacts/external_dma_program.json",
        phase_mode=ExternalDmaRuntimePhaseMode.BRING_IN_THEN_FINAL_WRITEBACK,
        case_digest=program.case_digest,
        request_digest=program.request_digest,
        logical_graph_digest=program.logical_graph_digest,
        source_memory_plan_digest=program.source_memory_plan_digest,
        blocking_offload_plan_digest=program.blocking_offload_plan_digest,
    )
    by_id = {item.id: item for item in chunks}
    transfers = {item.id: item for item in plan.transfer_requests}
    by_direction = {(item.linked_state_abi_id, direction): [] for item in declarations for direction in (
        ExternalTransferDirection.EXTERNAL_TO_HBM,
        ExternalTransferDirection.HBM_TO_EXTERNAL,
    )}
    for operation in plan.operations:
        if operation.transfer_request_ref is None:
            continue
        chunk = by_id[operation.chunk_ref]
        declaration = decl_by_chunk[chunk.id]
        transfer = transfers[operation.transfer_request_ref]
        if (transfer.hbm_address != declaration.hbm_address or
                transfer.size_bytes != declaration.logical_bytes or
                transfer.external_address !=
                    allocations[declaration.external_allocation_ref].address or
                conn_by_die[declaration.die_id].id != transfer.connection_ref):
            raise SchemaError("planned DMA omitted linked StateABI or physical HBM address",
                              path="blocking_offload_plan.transfer_requests")
        by_direction[declaration.linked_state_abi_id, transfer.direction].append(transfer.id)
    if (len(program.descriptors) != 2 * len(declarations) or
            any(len(items) != 1 for items in by_direction.values())):
        raise SchemaError("every physical StateABI needs one restore and one final writeback",
                          path="external_dma_program.descriptors")
    return PhysicalTrainingOffloadCase(
        source=source, plan=plan, program=program,
        action_graph=graph, runtime_binding=runtime_binding,
    )


__all__ = ["PhysicalTrainingOffloadCase",
           "build_dense_training_physical_offload_program"]
