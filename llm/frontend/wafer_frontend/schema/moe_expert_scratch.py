"""Exact physical scratch roots for an expert macro without fake graph operands."""

from __future__ import annotations

from ..errors import SchemaError
from .common import DType, stable_artifact_id
from .ir0 import OpKind
from .ir1 import IR1
from .ir2 import (
    BufferBinding, BufferOwnership, BufferUseRole, CoreOrder, IntraDieDAG,
    MoeExpertScratchBinding, MoeExpertScratchRole, TaskBufferUse, TaskPlacement,
    TensorSlice,
)
from .moe_full_training_block_workload import (
    MoeForwardBlockKind, MoeFullTrainingBlockWorkload,
)


def _banks(start: int, size: int, count: int, interleave: int) -> tuple[int, ...]:
    first = start // interleave
    last = (start + size - 1) // interleave
    if last - first + 1 >= count:
        return tuple(range(count))
    return tuple(sorted((first + index) % count
                        for index in range(last - first + 1)))


def derive_moe_expert_scratch_bindings(
    dag: IntraDieDAG, ir1: IR1,
    placements: tuple[TaskPlacement, ...],
    buffer_bindings: tuple[BufferBinding, ...],
    task_buffer_uses: tuple[TaskBufferUse, ...],
    core_orders: tuple[CoreOrder, ...],
) -> tuple[MoeExpertScratchBinding, ...]:
    """Derive two private roots per physical expert, after all public roots."""
    bindings = {binding.id: binding for binding in buffer_bindings}
    placement = {item.task_id: item.core_id for item in placements}
    positions = {task_id: index for order in core_orders
                 for index, task_id in enumerate(order.task_ids)}
    die = next((item for item in ir1.fabric.dies if item.id == dag.die_id), None)
    if die is None:
        raise SchemaError("expert scratch needs a real physical Die", path="ir1.fabric")
    result: list[MoeExpertScratchBinding] = []
    for task in dag.tasks:
        if task.op_kind is not OpKind.MOE_EXPERT_FORWARD:
            continue
        if (task.compute is None
                or type(task.compute.workload) is not MoeFullTrainingBlockWorkload
                or task.compute.workload.kind is not MoeForwardBlockKind.EXPERT):
            raise SchemaError("MoE expert scratch requires exact COMP workload",
                              path=f"dag.tasks.{task.id}")
        core_id = placement.get(task.id)
        uses = tuple(use for use in task_buffer_uses if use.task_id == task.id)
        if core_id is None or len(uses) != 5 or task.id not in positions:
            raise SchemaError("expert scratch needs five scheduled public operands",
                              path=f"dag.tasks.{task.id}")
        public = tuple(bindings[use.binding_id] for use in uses)
        if (len({binding.core_id for binding in public}) != 1
                or public[0].core_id != core_id
                or len({binding.region_ref for binding in public}) != 1
                or public[0].dtype is not DType.FP16):
            raise SchemaError("expert operands need one FP16 physical core/region",
                              path=f"dag.tasks.{task.id}")
        core = next((item for item in die.cores
                     if item.runtime_core_id == core_id), None)
        profile = (next((item for item in ir1.fabric.sram_profiles
                         if item.id == core.sram_profile_ref), None)
                   if core is not None else None)
        region = (next((item for item in profile.regions
                        if item.id == public[0].region_ref), None)
                  if profile is not None else None)
        if region is None:
            raise SchemaError("expert core lacks public SRAM region",
                              path=f"dag.tasks.{task.id}")
        high = max(binding.region_offset_bytes + binding.size_bytes
                   for binding in buffer_bindings
                   if binding.core_id == core_id
                   and binding.region_ref == region.id)
        align = profile.allocation_alignment_bytes
        concat = (high + align - 1) // align * align
        m = task.compute.workload.owned_token_count
        i = task.compute.workload.intermediate_size
        projection_bytes = 2 * m * i
        activated = (concat + 2 * projection_bytes + align - 1) // align * align
        if (activated + projection_bytes > region.size_bytes
                or region.base_bytes + activated + projection_bytes
                   > profile.capacity_bytes
                or region.base_bytes + activated + projection_bytes > (1 << 16)):
            raise SchemaError("two expert scratch roots exceed physical SRAM",
                              path=f"dag.tasks.{task.id}")
        specs = (
            (MoeExpertScratchRole.GATE_UP_CONCAT, (m, 2*i),
             concat, 2*projection_bytes),
            (MoeExpertScratchRole.SWIGLU_ACTIVATED, (m, i),
             activated, projection_bytes),
        )
        for role, shape, offset, size in specs:
            value_id = f"{task.id}:{role.value}"
            identity = {"dag": dag.id, "task": task.id, "role": role.value,
                        "core": core_id, "region": region.id,
                        "offset": offset, "bytes": size}
            binding = BufferBinding(
                id=stable_artifact_id("moe_expert_scratch_binding", identity,
                                      schema_version="moe_expert_scratch/v1"),
                value_id=value_id,
                tensor_slice=TensorSlice(value_id, (0, 0), shape),
                core_id=core_id, region_ref=region.id,
                region_offset_bytes=offset, size_bytes=size,
                alignment_bytes=align,
                banks=_banks(region.base_bytes + offset, size,
                             profile.bank_count, profile.bank_interleave_bytes),
                storage_id=stable_artifact_id("moe_expert_scratch_storage", identity,
                                              schema_version="moe_expert_scratch/v1"),
                alias_of=None, ownership=BufferOwnership.OWNED,
                lifetime_start=positions[task.id],
                lifetime_end_exclusive=positions[task.id] + 1,
                dtype=DType.FP16, layout=public[0].layout,
            )
            binding.validate(f"moe_expert_scratch.{task.id}.{role.value}")
            result.append(MoeExpertScratchBinding(task.id, role, binding))
    return tuple(sorted(result, key=lambda item: (item.task_id, item.role.value)))


__all__ = ["derive_moe_expert_scratch_bindings"]
