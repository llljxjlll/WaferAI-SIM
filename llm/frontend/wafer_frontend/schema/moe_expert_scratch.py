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
from .moe_expert_backward_workload import MoeExpertBackwardWorkload
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
    """Derive exact private roots per physical expert after all public roots."""
    bindings = {binding.id: binding for binding in buffer_bindings}
    placement = {item.task_id: item.core_id for item in placements}
    positions = {task_id: index for order in core_orders
                 for index, task_id in enumerate(order.task_ids)}
    die = next((item for item in ir1.fabric.dies if item.id == dag.die_id), None)
    if die is None:
        raise SchemaError("expert scratch needs a real physical Die", path="ir1.fabric")
    result: list[MoeExpertScratchBinding] = []
    for task in dag.tasks:
        if task.op_kind not in (OpKind.MOE_EXPERT_FORWARD,
                                OpKind.MOE_EXPERT_BACKWARD):
            continue
        backward = task.op_kind is OpKind.MOE_EXPERT_BACKWARD
        if (task.compute is None
                or (backward and type(task.compute.workload)
                    is not MoeExpertBackwardWorkload)
                or (not backward and
                    (type(task.compute.workload) is not MoeFullTrainingBlockWorkload
                     or task.compute.workload.kind is not MoeForwardBlockKind.EXPERT))):
            raise SchemaError("MoE expert scratch requires exact COMP workload",
                              path=f"dag.tasks.{task.id}")
        core_id = placement.get(task.id)
        uses = tuple(use for use in task_buffer_uses if use.task_id == task.id)
        if (core_id is None or len(uses) != (9 if backward else 5)
                or task.id not in positions):
            raise SchemaError("expert scratch needs all scheduled public operands",
                              path=f"dag.tasks.{task.id}")
        public = tuple(bindings[use.binding_id] for use in uses)
        if (len({binding.core_id for binding in public}) != 1
                or public[0].core_id != core_id
                or len({binding.region_ref for binding in public}) != 1
                or public[0].dtype is not DType.FP16
                or (backward and tuple(binding.dtype for binding in public) !=
                    (DType.FP16,) * 6 + (DType.FP32,) * 3)):
            raise SchemaError("expert operands need exact FP16/FP32 physical core/region",
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
        m = task.compute.workload.token_count if backward else task.compute.workload.owned_token_count
        i = task.compute.workload.intermediate_size
        h = task.compute.workload.hidden_size
        projection_bytes = 2 * m * i
        if backward:
            dx_bytes = 2 * m * h
            dx_stride = (dx_bytes + 63) // 64 * 64
            dx_storage = dx_stride + dx_bytes
            specs_raw = (
                (MoeExpertScratchRole.BACKWARD_GATE_UP_CONCAT,
                 (m, 2*i), 2*projection_bytes),
                (MoeExpertScratchRole.BACKWARD_SWIGLU_ACTIVATED,
                 (m, i), projection_bytes),
                (MoeExpertScratchRole.BACKWARD_ACTIVATED_GRADIENT,
                 (m, i), projection_bytes),
                (MoeExpertScratchRole.BACKWARD_GATE_UP_GRADIENT,
                 (m, 2*i), 2*projection_bytes),
                (MoeExpertScratchRole.BACKWARD_DX_PARTS,
                 (dx_storage // 2,), dx_storage),
            )
        else:
            specs_raw = (
                (MoeExpertScratchRole.GATE_UP_CONCAT,
                 (m, 2*i), 2*projection_bytes),
                (MoeExpertScratchRole.SWIGLU_ACTIVATED,
                 (m, i), projection_bytes),
            )
        specs = []
        cursor = concat
        for role, shape, size in specs_raw:
            cursor = (cursor + align - 1) // align * align
            specs.append((role, shape, cursor, size))
            cursor += size
        if (cursor > region.size_bytes
                or region.base_bytes + cursor > profile.capacity_bytes
                or region.base_bytes + cursor > (1 << 16)):
            raise SchemaError("expert scratch roots exceed physical SRAM",
                              path=f"dag.tasks.{task.id}")
        for role, shape, offset, size in specs:
            value_id = f"{task.id}:{role.value}"
            identity = {"dag": dag.id, "task": task.id, "role": role.value,
                        "core": core_id, "region": region.id,
                        "offset": offset, "bytes": size}
            binding = BufferBinding(
                id=stable_artifact_id("moe_expert_scratch_binding", identity,
                                      schema_version="moe_expert_scratch/v1"),
                value_id=value_id,
                tensor_slice=TensorSlice(value_id, tuple(0 for _ in shape), shape),
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
