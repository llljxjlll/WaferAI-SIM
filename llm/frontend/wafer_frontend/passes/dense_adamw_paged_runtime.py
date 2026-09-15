"""Bind production AdamW paged ABI and two real LSU orders to source DMA fabric."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.dense_adamw_linked import DenseAdamwLinkedProgram
from ..schema.dense_adamw_paged_runtime import (
    DenseAdamwPagedEvent, DenseAdamwPagedRuntime, DenseAdamwPagedState,
)
from ..schema.external_dma_program import ExternalDmaProgram
from ..schema.serde import canonical_digest
from .dense_adamw_mid_program_residency import (
    DenseAdamwBoundedSlots, DenseAdamwMidProgramResidency,
)
from .dense_adamw_offload_preflight import DenseAdamwOffloadWindow


def build_dense_adamw_paged_runtime(
    window: DenseAdamwOffloadWindow,
    schedule: DenseAdamwMidProgramResidency,
    slots: DenseAdamwBoundedSlots,
    linked: tuple[DenseAdamwLinkedProgram, DenseAdamwLinkedProgram],
    program: ExternalDmaProgram,
) -> DenseAdamwPagedRuntime:
    """Create a typed ABI/external-source binding; NpuSim must still execute it."""

    program.validate("source_dma_program")
    if (
        len(linked) != 2
        or any(item.materialization.id != window.materialization.id
               for item in linked)
        or tuple(item.step_index for item in linked) != (0, 1)
        or schedule.source_memory_plan_digest
        != canonical_digest(window.materialization.memory_plan)
        or program.request_digest != window.materialization.request_digest
        or program.logical_graph_digest != window.materialization.logical_graph_digest
        or program.source_memory_plan_digest != schedule.source_memory_plan_digest
        or slots.workspace_end_bytes != window.hbm_workspace_peak_bytes
        or slots.highest_state_end_bytes > window.resident_hbm_capacity_bytes
    ):
        raise SchemaError("pager binding must sign the actual P3 external/linked source", path="source")
    abis = {
        item.state_ref: item for item in linked[0].manifest.fragments[0].state_abi
    }
    slots_by_ref = dict(slots.state_addresses)
    bindings = {}
    for index, source_linked in enumerate(linked):
        source_linked.validate(f"linked[{index}]")
        fragment = source_linked.manifest.fragments[0]
        state_by_abi = {item.id: item for item in fragment.state_abi}
        if {item.state_ref: item.id for item in fragment.state_abi} != {
            ref: abi.id for ref, abi in abis.items()
        }:
            raise SchemaError("paged two-step physical StateABI continuity drifted", path="linked")
        events = []
        for binding in sorted(source_linked.manifest.state_operand_bindings,
                              key=lambda item: item.fragment_record_index):
            record = fragment.core_streams[0].records[binding.fragment_record_index]
            abi = state_by_abi[binding.state_abi_id]
            events.append((binding.fragment_record_index, record.opcode.value, abi.state_ref))
        bindings[index] = tuple(events)
    from ..schema.artifact_manifest import RecordOpcode
    planned = tuple(
        (item.linked_record_index,
         RecordOpcode.LSU_LOAD.value if item.kind == "restore_before_lsu_load"
         else RecordOpcode.LSU_STORE.value,
         item.state_ref)
        for item in schedule.events if item.step_index == 0
    )
    if bindings[0] != planned or bindings[1] != planned:
        raise SchemaError("paged production records changed signed per-state LSU gate", path="linked")
    seeds = tuple(program.external_seeds)
    if len(seeds) != 5 or len(program.external_probes) != 5:
        raise SchemaError("83 external states need five true source seed/probe groups", path="program")
    spans = tuple(sorted((
        DenseAdamwPagedState(
            item.state_ref, abis[item.state_ref].id,
            item.source_allocation_ref, item.external_address,
            slots_by_ref[item.state_ref], item.size_bytes, item.kind.value,
        )
        for item in schedule.spans
    ), key=lambda item: item.state_ref))
    for span in spans:
        if not any(
            seed.address <= span.external_address
            and span.external_address + span.size_bytes
            <= seed.address + len(seed.payload_hex) // 2
            for seed in seeds
        ):
            raise SchemaError("paged ABI external bytes outside genuine source seed", path=span.state_ref)
    events = tuple(
        DenseAdamwPagedEvent(
            item.step_index, item.linked_record_index, item.kind,
            item.state_ref, item.external_address,
            slots_by_ref[item.state_ref], item.size_bytes,
        )
        for item in schedule.events
    )
    return DenseAdamwPagedRuntime.create(
        source_dma_program_relative_path="artifacts/external_dma_program.json",
        source_dma_program_id=program.id,
        source_dma_program_digest=canonical_digest(program),
        request_digest=program.request_digest,
        logical_graph_digest=program.logical_graph_digest,
        source_memory_plan_digest=program.source_memory_plan_digest,
        blocking_offload_plan_digest=program.blocking_offload_plan_digest,
        linked_manifest_ids=(linked[0].manifest.id, linked[1].manifest.id),
        linked_manifest_digests=(canonical_digest(linked[0].manifest),
                                 canonical_digest(linked[1].manifest)),
        hbm_capacity_bytes=window.resident_hbm_capacity_bytes,
        workspace_end_bytes=window.hbm_workspace_peak_bytes,
        state_bytes=32100, state_spans=spans, events=events,
    )


__all__ = ["build_dense_adamw_paged_runtime"]
