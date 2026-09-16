"""Native EP1 identity dispatch copy proven by signed token-slot routing."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressRelocation, CommandFragment, CoreFragmentStream, FragmentKind,
    ProgramSymbolKind, RecordOpcode, RecordOperand, RelocatableRecord,
    RuntimeOperandField, RuntimeRelocation, RuntimeSymbolKind,
    SemanticOperandId,
)
from ..schema.common import DType
from ..schema.global_action import GlobalAction
from ..schema.ir0 import OpKind
from ..schema.ir2 import BufferUseRole, SemanticTaskKind
from ..schema.moe_full_training_block_workload import (
    MoeForwardBlockKind, MoeFullTrainingBlockWorkload,
)
from .coarse import _binding_for_use, _buffer_abi, _program_symbol, _view_addend_for_use
from .context import LoweringContext
from .isa_region import _runtime_symbol


_PRODUCER = "moe_full_train_dispatch_lowering"


def lower_moe_dispatch(action: GlobalAction,
                           context: LoweringContext) -> CommandFragment:
    """Copy activation into expert slots only for signed EP1 identity routing."""
    context.validate()
    if action not in context.global_dag.actions or (
        action.task_kind is not SemanticTaskKind.COMP
        or action.op_kind is not OpKind.MOE_DISPATCH
        or action.compute is None
        or action.compute.impl_ref != "moe_dispatch"
        or type(action.compute.workload) is not MoeFullTrainingBlockWorkload
        or action.compute.workload.kind is not MoeForwardBlockKind.DISPATCH
        or action.logical_core is None
        or action.runtime_binding is None
        or action.runtime_binding.token_symbol is None
    ):
        raise SchemaError("requires exact scheduled physical EP1 dispatch COMP",
                          path="action")
    schedule = next((schedule for schedule in context.schedule_set.schedules
                     if schedule.id == action.source.schedule_id), None)
    if schedule is None:
        raise SchemaError("dispatch schedule is missing", path="action.source.schedule_id")
    bindings = {binding.id: binding for binding in schedule.buffer_bindings}
    source = _binding_for_use(action, bindings, BufferUseRole.COMP_INPUT, 0,
                              path="action.buffer_uses")
    route = _binding_for_use(action, bindings, BufferUseRole.COMP_INPUT, 1,
                             path="action.buffer_uses")
    output = _binding_for_use(action, bindings, BufferUseRole.COMP_OUTPUT, 0,
                              path="action.buffer_uses")
    workload = action.compute.workload
    count = workload.token_count * workload.hidden_size * 2
    if (workload.expert_count != 1
            or workload.frozen_expert_by_token != (0,) * workload.token_count
            or workload.frozen_slot_by_token != tuple(range(workload.token_count))
            or source.dtype is not DType.FP16
            or route.dtype is not DType.INT32
            or route.tensor_slice.shape != (workload.token_count, 5)
            or route.size_bytes != workload.token_count * 20
            or output.dtype is not DType.FP16
            or source.tensor_slice.shape != (workload.token_count, workload.hidden_size)
            or output.tensor_slice.shape != source.tensor_slice.shape
            or source.size_bytes != count
            or output.size_bytes != count
            or source.core_id != output.core_id
            or source.region_ref != output.region_ref
            or len(action.buffer_uses) != 3):
        raise SchemaError("dispatch requires signed EP1 identity slots and FP16 local copy ABI",
                          path="action.buffer_uses")
    source_addend = _view_addend_for_use(
        action, source, BufferUseRole.COMP_INPUT, 0, path="action.buffer_uses")
    output_addend = _view_addend_for_use(
        action, output, BufferUseRole.COMP_OUTPUT, 0, path="action.buffer_uses")
    source_symbol = _program_symbol(schedule_id=schedule.id, binding=source,
                                    kind=ProgramSymbolKind.ABSOLUTE_ADDRESS)
    output_symbol = _program_symbol(schedule_id=schedule.id, binding=output,
                                    kind=ProgramSymbolKind.ABSOLUTE_ADDRESS)
    token = _runtime_symbol(RuntimeSymbolKind.DTE_TOKEN,
                            action.runtime_binding.token_symbol,
                            ("moe_full_train_dispatch", action.id))
    issue = RelocatableRecord(action.id, RecordOpcode.DTE_ISSUE, (
        RecordOperand.literal("direction", 0),
        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
        RecordOperand.literal("payload_bits", count * 8),
        RecordOperand.literal("size_bytes", count),
        RecordOperand.literal("hbm_address", 0),
        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS,
                              source_symbol.id),
        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS,
                              output_symbol.id),
    ))
    wait = RelocatableRecord(action.id, RecordOpcode.DTE_WAIT, (
        RecordOperand.runtime("token", RuntimeOperandField.DTE_TOKEN, token.id),
    ))
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER,
        source_global_dag_id=context.global_dag.id,
        kind=FragmentKind.COARSE,
        claimed_action_ids=(action.id,),
        core_streams=(CoreFragmentStream(
            action.logical_core, (issue, wait),
            (RuntimeRelocation(0, RuntimeOperandField.DTE_TOKEN, token.id),
             RuntimeRelocation(1, RuntimeOperandField.DTE_TOKEN, token.id)),
            tuple(sorted((
                AddressRelocation(0, SemanticOperandId.SOURCE_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS,
                                  source_symbol.id, source_addend),
                AddressRelocation(0, SemanticOperandId.DESTINATION_ADDRESS,
                                  ProgramSymbolKind.ABSOLUTE_ADDRESS,
                                  output_symbol.id, output_addend),
            ), key=lambda item: int(item.operand_id))),
        ),),
        runtime_symbols=(token,),
        program_symbols=tuple(sorted((source_symbol, output_symbol),
                                     key=lambda item: item.id)),
        buffer_abi=tuple(sorted((_buffer_abi(schedule.id, binding, action.logical_core)
                                 for binding in (source, route, output)),
                                key=lambda abi: abi.id)),
        state_abi=(),
    )
    fragment.validate_against(context.global_dag)
    return fragment


__all__ = ["lower_moe_dispatch"]
