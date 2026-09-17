"""Source-backed capacity window for the full two-step Dense AdamW program.

This is a necessary blocking-offload window, not an executable pager or an
external-memory success receipt.  Native execution still requires a signed
external DMA schedule and rebased HBM StateABI addresses.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.ir0 import OpKind
from ..schema.n6 import _leaf_fragments
from ..schema.train_n6 import TrainLinkedProgram


@dataclass(frozen=True, slots=True)
class FullDenseAdamwOffloadWindow:
    source_ir1_id: str
    linked_manifest_id: str
    state_count: int
    state_payload_bytes: int
    resident_hbm_highwater_bytes: int
    atomic_update_hbm_slot_bytes: int
    external_state_capacity_min_bytes: int
    saved_activation_values: int
    saved_activation_payload_bytes: int
    sram_allocated_highwater_bytes: int
    sram_live_storage_peak_bytes: int
    hbm_capacity_bytes: int
    external_capacity_bytes: int
    sram_capacity_bytes: int
    resident_only_rejected: bool
    blocking_offload_window_necessary: bool


def _aligned(size: int, alignment: int) -> int:
    return (size + alignment - 1) // alignment * alignment


def derive_full_dense_adamw_offload_window(
    program: TrainLinkedProgram,
    plan: FlexibleDenseTrainPlan,
    *,
    hbm_capacity_bytes: int,
    external_capacity_bytes: int,
    sram_capacity_bytes: int,
) -> FullDenseAdamwOffloadWindow:
    """Measure true StateABI/BufferABI extents and reject impossible windows."""
    program.validate("full_dense_adamw_offload_preflight")
    plan.validate("full_dense_adamw_offload_plan")
    if (plan.spec.tp_degree != 1 or plan.spec.dp_degree != 1
            or len(program.source.replicas) != 1
            or any(type(x) is not int or x <= 0 for x in (
                hbm_capacity_bytes, external_capacity_bytes, sram_capacity_bytes))):
        raise SchemaError("full AdamW capacity window requires positive TP1/DP1 physical capacities",
                          path="capacities")
    graph = program.source.replicas[0].lowering_context.ir1
    if len(graph.nodes) != 170 or len(graph.persistent_state_manifest.declarations) != 75:
        raise SchemaError("offload window needs complete two-step AdamW source",
                          path="source.ir1")
    leaves = _leaf_fragments(program.manifest.fragments)
    states = {}
    buffers = {}
    for leaf in leaves:
        for abi in leaf.state_abi:
            previous = states.setdefault(abi.state_ref, abi)
            if previous != abi:
                raise SchemaError("StateABI changes across source fragments", path=abi.state_ref)
        for abi in leaf.buffer_abi:
            previous = buffers.setdefault(abi.id, abi)
            if previous != abi:
                raise SchemaError("BufferABI changes across source fragments", path=abi.id)
    if (set(states) != {item.id for item in graph.persistent_state_manifest.declarations}
            or len(states) != 75 or not buffers):
        raise SchemaError("full AdamW physical state/activation inventory is incomplete",
                          path="manifest.fragments")
    state_payload = sum(item.size_bytes for item in states.values())
    resident_end = max(item.address + item.size_bytes for item in states.values())
    by_node = {}
    for access in graph.state_accesses:
        node = next((item for item in graph.nodes if item.id == access.node_ref), None)
        if node is not None and node.kind is OpKind.OPTIMIZER_UPDATE:
            by_node.setdefault(node.id, []).append(access)
    if len(by_node) != 30 or any(len(accesses) != 5 for accesses in by_node.values()):
        raise SchemaError("each full AdamW step needs five real state accesses",
                          path="source.state_accesses")
    atomic = max(sum(_aligned(states[access.state_ref].size_bytes,
                              max(64, states[access.state_ref].alignment_bytes))
                     for access in accesses)
                 for accesses in by_node.values())
    nodes = {node.id: node for node in graph.nodes}
    saved = tuple(value for value in graph.values
                  if value.producer is not None
                  and nodes[value.producer].phase.value == "fwd"
                  and any(nodes[consumer].phase.value in ("dgrad", "wgrad")
                          for consumer in value.consumers))
    if not saved or any(value.dtype not in (DType.FP16, DType.FP32) for value in saved):
        raise SchemaError("full AdamW source lacks typed saved forward activations",
                          path="source.values")
    saved_bytes = sum(prod(value.shape) * (2 if value.dtype is DType.FP16 else 4)
                      for value in saved)
    sram_high = max(item.region_offset_bytes + item.size_bytes for item in buffers.values())
    times = sorted({item.lifetime_start for item in buffers.values()}
                   | {item.lifetime_end_exclusive for item in buffers.values()})
    live_peak = max(sum({item.storage_id: item.size_bytes for item in buffers.values()
                         if item.lifetime_start <= t < item.lifetime_end_exclusive}.values())
                    for t in times)
    if sram_capacity_bytes < sram_high:
        raise SchemaError("full AdamW fixed SRAM activation/scratch allocations exceed capacity",
                          path="sram_capacity_bytes", code="adamw_sram_window_exceeded")
    if external_capacity_bytes < resident_end:
        raise SchemaError("external backing cannot hold all 75 source StateABI ranges",
                          path="external_capacity_bytes", code="adamw_external_state_exceeded")
    if hbm_capacity_bytes < atomic:
        raise SchemaError("one authentic five-state AdamW update cannot fit HBM",
                          path="hbm_capacity_bytes", code="adamw_atomic_update_window_exceeded")
    return FullDenseAdamwOffloadWindow(
        source_ir1_id=graph.id, linked_manifest_id=program.manifest.id,
        state_count=len(states), state_payload_bytes=state_payload,
        resident_hbm_highwater_bytes=resident_end,
        atomic_update_hbm_slot_bytes=atomic,
        external_state_capacity_min_bytes=resident_end,
        saved_activation_values=len(saved),
        saved_activation_payload_bytes=saved_bytes,
        sram_allocated_highwater_bytes=sram_high,
        sram_live_storage_peak_bytes=live_peak,
        hbm_capacity_bytes=hbm_capacity_bytes,
        external_capacity_bytes=external_capacity_bytes,
        sram_capacity_bytes=sram_capacity_bytes,
        resident_only_rejected=hbm_capacity_bytes < resident_end,
        blocking_offload_window_necessary=(atomic <= hbm_capacity_bytes < resident_end),
    )


def require_full_dense_adamw_resident_capacity(window: FullDenseAdamwOffloadWindow) -> None:
    if window.resident_only_rejected:
        raise SchemaError("full AdamW resident StateABI address range exceeds HBM",
                          path="hbm_capacity_bytes", code="memory_capacity_exceeded")


__all__ = [
    "FullDenseAdamwOffloadWindow", "derive_full_dense_adamw_offload_window",
    "require_full_dense_adamw_resident_capacity",
]
