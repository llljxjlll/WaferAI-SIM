"""Source-bound EP1 weighted MoE combine using public native opcode 0x27."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressRelocation, CommandFragment, CoreFragmentStream, FragmentKind,
    ProgramSymbolKind, RecordOpcode, RecordOperand, RelocatableRecord,
    SemanticOperandId,
)
from ..schema.common import DType
from ..schema.global_action import GlobalAction, SemanticTaskKind
from ..schema.ir0 import OpKind
from ..schema.ir2 import BufferUseRole
from ..schema.moe_full_training_block_workload import (
    MoeForwardBlockKind, MoeFullTrainingBlockWorkload,
)
from .coarse import _binding_for_use, _buffer_abi, _program_symbol, _view_addend_for_use
from .context import LoweringContext


_PRODUCER = "moe_full_train_combine_lowering"


def lower_moe_weighted_combine(action: GlobalAction,
                               context: LoweringContext) -> CommandFragment:
    """Read route+dynamic score+expert result and produce weighted FP16 output."""
    context.validate()
    if action not in context.global_dag.actions or (
        action.task_kind is not SemanticTaskKind.COMP
        or action.op_kind is not OpKind.MOE_COMBINE
        or action.compute is None
        or action.compute.impl_ref != "moe_combine"
        or type(action.compute.workload) is not MoeFullTrainingBlockWorkload
        or action.compute.workload.kind is not MoeForwardBlockKind.COMBINE
        or action.compute.workload.expert_count != 1
        or action.logical_core is None
    ):
        raise SchemaError("requires exact scheduled physical EP1 weighted combine",
                          path="action")
    schedule = next((item for item in context.schedule_set.schedules
                     if item.id == action.source.schedule_id), None)
    if schedule is None:
        raise SchemaError("combine schedule is missing", path="action.source.schedule_id")
    bindings = {item.id: item for item in schedule.buffer_bindings}
    returned = _binding_for_use(action, bindings, BufferUseRole.COMP_INPUT, 0,
                                path="action.buffer_uses")
    route = _binding_for_use(action, bindings, BufferUseRole.COMP_INPUT, 1,
                             path="action.buffer_uses")
    score = _binding_for_use(action, bindings, BufferUseRole.COMP_INPUT, 2,
                             path="action.buffer_uses")
    output = _binding_for_use(action, bindings, BufferUseRole.COMP_OUTPUT, 0,
                              path="action.buffer_uses")
    workload = action.compute.workload
    m, h = workload.token_count, workload.hidden_size
    if (len(action.buffer_uses) != 4
            or workload.frozen_expert_by_token != (0,) * m
            or workload.frozen_slot_by_token != tuple(range(m))
            or route.dtype is not DType.INT32
            or route.tensor_slice.shape != (m, 5)
            or route.size_bytes != 20*m
            or score.dtype is not DType.FP16
            or score.tensor_slice.shape != (m, 1)
            or score.size_bytes != 2*m
            or any(item.dtype is not DType.FP16 for item in (returned, output))
            or returned.tensor_slice.shape != (m, h)
            or output.tensor_slice.shape != (m, h)
            or returned.size_bytes != 2*m*h
            or output.size_bytes != 2*m*h
            or len({item.storage_id for item in (route, score, returned, output)}) != 4):
        raise SchemaError("weighted combine requires distinct full route, score, return and output ABI",
                          path="action.buffer_uses")
    sources = (route, score, returned)
    source_uses = (1, 2, 0)
    input_labels = tuple(_program_symbol(schedule_id=schedule.id, binding=item,
                                         kind=ProgramSymbolKind.SRAM_LABEL)
                         for item in sources)
    output_label = _program_symbol(schedule_id=schedule.id, binding=output,
                                   kind=ProgramSymbolKind.SRAM_LABEL)
    addresses = tuple(_program_symbol(schedule_id=schedule.id, binding=item,
                                      kind=ProgramSymbolKind.ABSOLUTE_ADDRESS)
                      for item in (*sources, output))
    bind = RelocatableRecord(action.id, RecordOpcode.SRAM_BIND, (
        RecordOperand.literal("input_count", 3),
        *(RecordOperand.address(f"input_label_{index}",
                                SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0)+index),
                                symbol.id)
          for index, symbol in enumerate(input_labels)),
        *(RecordOperand.literal(f"input_label_{index}", 0)
          for index in range(3, 16)),
        RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
                              output_label.id),
    ))
    operand_ids = (SemanticOperandId.COMPUTE_ROUTE_TABLE_ADDRESS,
                   SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                   SemanticOperandId.COMPUTE_DATA_ADDRESS,
                   SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
    names = ("route_address", "score_address", "return_address", "combined_address")
    compute = RelocatableRecord(action.id, RecordOpcode.MOE_SCORE_WEIGHTED_FORWARD, (
        *(RecordOperand.literal(name, value) for name, value in (
            ("route_datatype", 2), ("score_datatype", 1),
            ("expert_datatype", 1), ("combined_datatype", 1))),
        *(RecordOperand.address(name, operand_id, symbol.id)
          for name, operand_id, symbol in zip(names, operand_ids, addresses)),
        *(RecordOperand.literal(name, value) for name, value in (
            ("rank_rows", m), ("hidden_size", h),
            ("expert_count", 1), ("route_bytes", 20*m))),
    ))
    relocations = tuple(sorted((
        *(AddressRelocation(0,
                            SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0)+index),
                            symbol.kind, symbol.id, 0)
          for index, symbol in enumerate(input_labels)),
        AddressRelocation(0, SemanticOperandId.SRAM_BIND_OUTPUT,
                          output_label.kind, output_label.id, 0),
        *(AddressRelocation(1, operand_id, symbol.kind, symbol.id,
                            _view_addend_for_use(action, binding, BufferUseRole.COMP_INPUT,
                                                 source_uses[index], path="action.buffer_uses")
                            if index < 3 else
                            _view_addend_for_use(action, output, BufferUseRole.COMP_OUTPUT,
                                                 0, path="action.buffer_uses"))
          for index, (operand_id, symbol, binding) in enumerate(
              zip(operand_ids, addresses, (*sources, output)))),
    ), key=lambda item: (item.record_index, int(item.operand_id))))
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER, source_global_dag_id=context.global_dag.id,
        kind=FragmentKind.COARSE, claimed_action_ids=(action.id,),
        core_streams=(CoreFragmentStream(action.logical_core,
                                         (bind, compute), (), relocations),),
        runtime_symbols=(),
        program_symbols=tuple(sorted((*input_labels, output_label, *addresses),
                                     key=lambda symbol: symbol.id)),
        buffer_abi=tuple(sorted((_buffer_abi(schedule.id, item, action.logical_core)
                                 for item in (*sources, output)), key=lambda abi: abi.id)),
        state_abi=(),
    )
    fragment.validate()
    return fragment


__all__ = ["lower_moe_weighted_combine"]
