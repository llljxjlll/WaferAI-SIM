"""Source-bound native instruction geometry for one MoE expert forward action.

This is the operation plan consumed by the dedicated N6 path.  It does not
claim that a four-instruction expert block is a one-opcode Dense COMP leaf.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import RecordOpcode
from ..schema.common import DType, stable_artifact_id
from ..schema.global_action import GlobalAction
from ..schema.ir0 import OpKind
from ..schema.ir2 import BufferBinding, BufferOwnership, BufferUseRole, IntraDieSchedule, SemanticTaskKind
from ..schema.ir1 import IR1
from ..schema.moe_full_training_block_workload import MoeForwardBlockKind, MoeFullTrainingBlockWorkload


@dataclass(frozen=True, slots=True)
class MoeExpertNativeOp:
    role: str
    opcode: RecordOpcode
    parameters: tuple[int, ...]
    input_ref: str
    weight_ref: str | None
    output_ref: str
    input_bytes: int
    weight_bytes: int
    output_bytes: int
    output_offset_bytes: int


@dataclass(frozen=True, slots=True)
class MoeExpertNativePlan:
    source_action_id: str
    source_schedule_id: str
    source_ir1_ref: str
    source_operation_ref: str
    source_route_trace_digest: str
    logical_core_die: int
    logical_core_id: int
    runtime_core_id: int
    concat_scratch_bytes: int
    activated_scratch_bytes: int
    scratch_region_ref: str
    concat_region_offset_bytes: int
    activated_region_offset_bytes: int
    scratch_ownership: BufferOwnership
    core_order_index: int
    operations: tuple[MoeExpertNativeOp, ...]
    id: str


def plan_moe_expert_native_forward(
    action: GlobalAction, schedule: IntraDieSchedule, ir1: IR1,
) -> MoeExpertNativePlan:
    """Derive four real operations from one scheduled physical expert.

    The two scratch roots are explicit requirements for N6 allocation; no
    source weight or public output is silently reused as scratch storage.
    """
    if type(ir1) is not IR1:
        raise SchemaError("expert scratch needs physical IR1 hardware", path="ir1")
    if (action.task_kind is not SemanticTaskKind.COMP
            or action.op_kind is not OpKind.MOE_EXPERT_FORWARD
            or action.compute is None
            or action.compute.op_kind is not OpKind.MOE_EXPERT_FORWARD
            or type(action.compute.workload) is not MoeFullTrainingBlockWorkload
            or action.compute.workload.kind is not MoeForwardBlockKind.EXPERT
            or action.logical_core is None
            or action.source.schedule_id != schedule.id
            or action.logical_core.die_id != schedule.die_id
            or action.core_order_index is None):
        raise SchemaError("requires one scheduled physical MoE expert COMP", path="action")
    workload = action.compute.workload
    workload.validate("action.compute.workload")
    if workload.owned_token_count == 0:
        raise SchemaError("zero-token expert needs explicit zero-work lowering", path="action.compute.workload")
    if (tuple(x.role for x in action.compute.inputs) !=
            ("expert_activation", "gate_weight", "up_weight", "down_weight")
            or tuple(x.role for x in action.compute.outputs) != ("expert_output",)):
        raise SchemaError("expert ordered source operands drifted", path="action.compute")
    bindings = {binding.id: binding for binding in schedule.buffer_bindings}
    if len(bindings) != len(schedule.buffer_bindings):
        raise SchemaError("schedule has duplicate SRAM bindings", path="schedule.buffer_bindings")
    placements = [placement for placement in schedule.placements
                  if placement.task_id == action.source.task_id]
    if len(placements) != 1:
        raise SchemaError("expert task lacks one physical core placement", path="schedule.placements")
    runtime_core_id = placements[0].core_id
    selected: list[BufferBinding] = []
    for role, index in ((*((BufferUseRole.COMP_INPUT, i) for i in range(4)),
                         (BufferUseRole.COMP_OUTPUT, 0))):
        uses = [use for use in action.buffer_uses
                if use.role is role and use.operand_index == index]
        if len(uses) != 1 or uses[0].binding_id not in bindings:
            raise SchemaError("expert operand lacks exact scheduled BufferABI", path="action.buffer_uses")
        binding = bindings[uses[0].binding_id]
        expected = (action.compute.inputs[index].value_id if role is BufferUseRole.COMP_INPUT
                    else action.compute.outputs[0].value_id)
        if (binding.value_id != expected or binding.core_id != runtime_core_id
                or binding.dtype is not DType.FP16
                or uses[0].tensor_slice != binding.tensor_slice):
            raise SchemaError("expert operand differs from signed full tensor binding", path="action.buffer_uses")
        selected.append(binding)
    if len(action.buffer_uses) != 5:
        raise SchemaError("expert has extra or missing BufferABI uses", path="action.buffer_uses")
    activation, gate, up, down, output = selected
    m, h, i = workload.owned_token_count, workload.hidden_size, workload.intermediate_size
    expected_bytes = (2*m*h, 2*h*i, 2*h*i, 2*i*h, 2*m*h)
    if tuple(binding.size_bytes for binding in selected) != expected_bytes:
        raise SchemaError("expert source operands need exact FP16 projection extents", path="action.buffer_uses")
    if len({binding.storage_id for binding in selected}) != 5:
        raise SchemaError("expert activation/weights/output cannot alias", path="action.buffer_uses")
    concat = f"{action.id}:gate_up_concat"
    activated = f"{action.id}:swiglu_activated"
    projection_bytes = 2*m*i
    region_refs = {binding.region_ref for binding in selected}
    if len(region_refs) != 1:
        raise SchemaError("expert operands need one physical SRAM region",
                          path="action.buffer_uses")
    region_ref = next(iter(region_refs))
    die = next((die for die in ir1.fabric.dies
                if die.id == action.logical_core.die_id), None)
    core = (next((core for core in die.cores
                  if core.local_core_id == action.logical_core.local_core_id
                  and core.runtime_core_id == runtime_core_id), None)
            if die is not None else None)
    profile = (next((profile for profile in ir1.fabric.sram_profiles
                     if profile.id == core.sram_profile_ref), None)
               if core is not None else None)
    region = (next((region for region in profile.regions
                    if region.id == region_ref), None)
              if profile is not None else None)
    if region is None:
        raise SchemaError("expert scratch region is absent from actual physical core",
                          path="ir1.fabric")
    active = tuple(binding for binding in schedule.buffer_bindings
                   if binding.core_id == runtime_core_id
                   and binding.region_ref == region_ref)
    high = max(binding.region_offset_bytes + binding.size_bytes
               for binding in active)
    alignment = profile.allocation_alignment_bytes
    concat_offset = (high + alignment - 1) // alignment * alignment
    activated_offset = (concat_offset + 2*projection_bytes + alignment - 1) // alignment * alignment
    scratch_end = activated_offset + projection_bytes
    if (scratch_end > region.size_bytes
            or region.base_bytes + scratch_end > profile.capacity_bytes
            or region.base_bytes + scratch_end > (1 << 16)):
        raise SchemaError("two distinct expert scratch roots exceed physical SRAM",
                          path="ir1.fabric.sram_profiles")
    operations = (
        MoeExpertNativeOp("gate", RecordOpcode.MATMUL, (1,m,h,i), activation.id,
                          gate.id, concat, 2*m*h, 2*h*i, projection_bytes, 0),
        MoeExpertNativeOp("up", RecordOpcode.MATMUL, (1,m,h,i), activation.id,
                          up.id, concat, 2*m*h, 2*h*i, projection_bytes, projection_bytes),
        MoeExpertNativeOp("swiglu", RecordOpcode.SWIGLU, (m*i,), concat,
                          None, activated, 2*projection_bytes, 0, projection_bytes, 0),
        MoeExpertNativeOp("down", RecordOpcode.MATMUL, (1,m,i,h), activated,
                          down.id, output.id, projection_bytes, 2*i*h, 2*m*h, 0),
    )
    # Recheck public artifact identities after exact geometry checks: a caller
    # cannot change a binding while retaining the old signed schedule id.
    action.validate("action")
    schedule.validate("schedule")
    ir1.validate("ir1")
    semantic = dict(source_action_id=action.id, source_schedule_id=schedule.id,
                    source_ir1_ref=ir1.id,
                    source_operation_ref=workload.source_operation_ref,
                    source_route_trace_digest=workload.source_route_trace_digest,
                    logical_core_die=action.logical_core.die_id,
                    logical_core_id=action.logical_core.local_core_id,
                    runtime_core_id=runtime_core_id,
                    concat_scratch_bytes=2*projection_bytes,
                    activated_scratch_bytes=projection_bytes,
                    scratch_region_ref=region_ref,
                    concat_region_offset_bytes=concat_offset,
                    activated_region_offset_bytes=activated_offset,
                    scratch_ownership=BufferOwnership.OWNED,
                    core_order_index=action.core_order_index,
                    operations=operations)
    return MoeExpertNativePlan(**semantic, id=stable_artifact_id(
        "moe_expert_native_plan", semantic, schema_version="moe_expert_native_plan/v1"))


__all__ = ["MoeExpertNativeOp", "MoeExpertNativePlan", "plan_moe_expert_native_forward"]
