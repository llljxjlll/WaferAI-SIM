"""Lower exact EP1 source-bound MoE 0x28 from physical dCombined."""

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
from ..schema.moe_combine_backward_workload import MoeCombineBackwardWorkload
from .coarse import (
    _binding_for_use, _buffer_abi, _program_symbol, _view_addend_for_use,
)
from .context import LoweringContext


def lower_moe_combine_backward(
    action: GlobalAction, context: LoweringContext,
) -> CommandFragment:
    context.validate()
    if (action not in context.global_dag.actions
            or action.task_kind is not SemanticTaskKind.COMP
            or action.op_kind is not OpKind.MOE_COMBINE_BACKWARD
            or action.compute is None
            or action.compute.impl_ref != "moe_combine_backward"
            or type(action.compute.workload) is not MoeCombineBackwardWorkload
            or action.compute.workload.expert_count != 1
            or action.logical_core is None):
        raise SchemaError("requires scheduled source-bound EP1 0x28 action",
                          path="action")
    schedule = next((item for item in context.schedule_set.schedules
                     if item.id == action.source.schedule_id), None)
    if schedule is None:
        raise SchemaError("0x28 schedule is missing",
                          path="action.source.schedule_id")
    bindings = {item.id: item for item in schedule.buffer_bindings}
    inputs = tuple(_binding_for_use(
        action, bindings, BufferUseRole.COMP_INPUT, index,
        path="action.buffer_uses") for index in range(4))
    outputs = tuple(_binding_for_use(
        action, bindings, BufferUseRole.COMP_OUTPUT, index,
        path="action.buffer_uses") for index in range(2))
    workload = action.compute.workload
    m, h, e = (workload.token_count, workload.hidden_size,
               workload.expert_count)
    expected = (
        ((m, 5), DType.INT32, 20*m),
        ((m, e), DType.FP16, 2*m*e),
        ((m, h), DType.FP16, 2*m*h),
        ((m, h), DType.FP16, 2*m*h),
        ((m, e), DType.FP16, 2*m*e),
        ((m, h), DType.FP16, 2*m*h),
    )
    all_bindings = (*inputs, *outputs)
    if (len(action.buffer_uses) != 6
            or len({item.storage_id for item in all_bindings}) != 6
            or any((item.tensor_slice.shape, item.dtype, item.size_bytes)
                   != contract for item, contract in zip(
                       all_bindings, expected, strict=True))):
        raise SchemaError("0x28 requires six disjoint exact physical SRAM "
                          "tapes for route/score/expert/dCombined/dScore/dExpert",
                          path="action.buffer_uses")
    labels = tuple(_program_symbol(
        schedule_id=schedule.id, binding=item,
        kind=ProgramSymbolKind.SRAM_LABEL,
    ) for item in inputs)
    output_label = _program_symbol(
        schedule_id=schedule.id, binding=outputs[0],
        kind=ProgramSymbolKind.SRAM_LABEL,
    )
    addresses = tuple(_program_symbol(
        schedule_id=schedule.id, binding=item,
        kind=ProgramSymbolKind.ABSOLUTE_ADDRESS,
    ) for item in all_bindings)
    bind = RelocatableRecord(action.id, RecordOpcode.SRAM_BIND, (
        RecordOperand.literal("input_count", 4),
        *(RecordOperand.address(
            f"input_label_{index}",
            SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0)+index),
            label.id,
        ) for index, label in enumerate(labels)),
        *(RecordOperand.literal(f"input_label_{index}", 0)
          for index in range(4, 16)),
        RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
                              output_label.id),
    ))
    address_fields = (
        ("route_address", SemanticOperandId.COMPUTE_ROUTE_TABLE_ADDRESS),
        ("score_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS),
        ("return_address", SemanticOperandId.COMPUTE_DATA_ADDRESS),
        ("dcombined_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
        ("dscore_address", SemanticOperandId.COMPUTE_AUX_ADDRESS),
        ("dexpert_address", SemanticOperandId.COMPUTE_ROUTER_DEXPERT_ADDRESS),
    )
    compute = RelocatableRecord(action.id, RecordOpcode.MOE_SCORE_WEIGHT_BACKWARD, (
        *(RecordOperand.literal(name, value) for name, value in (
            ("route_datatype", 2), ("score_datatype", 1),
            ("expert_datatype", 1), ("upstream_datatype", 1),
            ("dscore_datatype", 1), ("dexpert_datatype", 1))),
        *(RecordOperand.address(name, operand_id, symbol.id)
          for (name, operand_id), symbol in zip(
              address_fields, addresses, strict=True)),
        *(RecordOperand.literal(name, value) for name, value in (
            ("rank_rows", m), ("hidden_size", h),
            ("expert_count", e), ("route_bytes", workload.route_bytes))),
    ))
    relocations = tuple(sorted((
        *(AddressRelocation(
            0, SemanticOperandId(int(SemanticOperandId.SRAM_BIND_INPUT_0)+index),
            label.kind, label.id, 0,
        ) for index, label in enumerate(labels)),
        AddressRelocation(0, SemanticOperandId.SRAM_BIND_OUTPUT,
                          output_label.kind, output_label.id, 0),
        *(AddressRelocation(
            1, operand_id, symbol.kind, symbol.id,
            _view_addend_for_use(
                action, binding,
                BufferUseRole.COMP_INPUT if index < 4
                else BufferUseRole.COMP_OUTPUT,
                index if index < 4 else index - 4,
                path="action.buffer_uses",
            ),
        ) for index, ((_, operand_id), symbol, binding) in enumerate(
            zip(address_fields, addresses, all_bindings, strict=True))),
    ), key=lambda item: (item.record_index, int(item.operand_id))))
    fragment = CommandFragment.create(
        producer_pass="moe_full_train_combine_backward_lowering",
        source_global_dag_id=context.global_dag.id,
        kind=FragmentKind.COARSE, claimed_action_ids=(action.id,),
        core_streams=(CoreFragmentStream(
            action.logical_core, (bind, compute), (), relocations,
        ),),
        runtime_symbols=(),
        program_symbols=tuple(sorted((*labels, output_label, *addresses),
                                     key=lambda symbol: symbol.id)),
        buffer_abi=tuple(sorted((
            _buffer_abi(schedule.id, binding, action.logical_core)
            for binding in all_bindings
        ), key=lambda abi: abi.id)),
        state_abi=(),
    )
    fragment.validate()
    return fragment
