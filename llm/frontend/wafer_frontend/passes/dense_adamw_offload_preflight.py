"""Bounded HBM AdamW resident reject and truthful external-offload preflight."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.memory_plan import (
    MemoryObjectKind, MemoryTier, MemoryTierCapacity,
)
from ..schema.workload_materialization import WorkloadMaterializationManifest
from ..schema.workload_run import (
    WorkloadMemoryMode, WorkloadMemoryPolicy, WorkloadRunRequest,
)
from .workload_materialization import materialize_workload_preflight


@dataclass(frozen=True, slots=True)
class DenseAdamwOffloadWindow:
    materialization: WorkloadMaterializationManifest
    resident_rejection_code: str
    resident_hbm_capacity_bytes: int
    external_capacity_bytes: int
    external_state_bytes: int
    external_peak_bytes: int
    hbm_workspace_peak_bytes: int
    startup_full_state_hbm_bytes: int
    startup_combined_peak_bytes: int
    startup_window_sufficient: bool


def preflight_dense_adamw_offload_window(
    resident: WorkloadMaterializationManifest,
    *,
    hbm_capacity_bytes: int = 36864,
    external_capacity_bytes: int = 32896,
) -> DenseAdamwOffloadWindow:
    """Report the exact same model; a preflight pass is not a DMA runtime pass."""

    resident.validate("resident")
    if (
        resident.request.optimizer is None
        or resident.request.optimizer.kind.value != "adamw"
        or resident.request.memory.mode is not WorkloadMemoryMode.RESIDENT_HBM
        or resident.request.model.num_layers != 2
        or resident.request.model.hidden_size != 16
    ):
        raise SchemaError("requires the two-layer H16 AdamW resident source", path="resident")
    hbm = MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref="die:0", base_address=0,
        capacity_bytes=hbm_capacity_bytes, alignment_bytes=64,
    )
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL, location_ref="host:0", base_address=0,
        capacity_bytes=external_capacity_bytes, alignment_bytes=16,
    )
    try:
        materialize_workload_preflight(
            resident.request, resident.capability, capacities=(hbm,),
        )
    except SchemaError as error:
        if error.code != "memory_capacity_exceeded":
            raise
        rejection = error.code
    else:
        raise SchemaError("same-model resident must reject bounded HBM", path="hbm")
    source = resident.request
    offload_request = WorkloadRunRequest.create(
        family=source.family, model=source.model, steps=source.steps,
        mesh=source.mesh, parallel=source.parallel,
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
            external_tier_ref="host:0",
        ),
        optimizer=source.optimizer, execution=source.execution,
    )
    offload = materialize_workload_preflight(
        offload_request, resident.capability, capacities=(external, hbm),
    )
    source_ops = tuple(
        (item.kind, item.step, item.layer, item.expert, item.parameter_ref)
        for item in resident.logical_graph.operations
    )
    offload_ops = tuple(
        (item.kind, item.step, item.layer, item.expert, item.parameter_ref)
        for item in offload.logical_graph.operations
    )
    if source_ops != offload_ops:
        raise SchemaError("memory policy changed the full E2E operation source", path="offload")
    external_items = tuple(
        item for item in offload.state_inventory
        if item.object_kind in (MemoryObjectKind.PARAMETER, MemoryObjectKind.OPTIMIZER)
    )
    external_state_bytes = sum(item.size_bytes for item in external_items)
    if (
        len(external_items) != 5
        or external_state_bytes != 32100
        or sum(item.size_bytes for item in external_items
               if item.logical_name.startswith("optimizer.adamw.step.")) != 68
    ):
        raise SchemaError("offload source misses any of the 83 exact ABI state bytes", path="offload")
    peaks = {item.capacity_ref: item.peak_bytes for item in offload.memory_plan.peaks}
    hbm_peak = peaks[hbm.id]
    external_peak = peaks[external.id]
    combined = external_state_bytes + hbm_peak
    return DenseAdamwOffloadWindow(
        materialization=offload,
        resident_rejection_code=rejection,
        resident_hbm_capacity_bytes=hbm_capacity_bytes,
        external_capacity_bytes=external_capacity_bytes,
        external_state_bytes=external_state_bytes,
        external_peak_bytes=external_peak,
        hbm_workspace_peak_bytes=hbm_peak,
        startup_full_state_hbm_bytes=external_state_bytes,
        startup_combined_peak_bytes=combined,
        startup_window_sufficient=combined <= hbm_capacity_bytes,
    )


__all__ = ["DenseAdamwOffloadWindow", "preflight_dense_adamw_offload_window"]
