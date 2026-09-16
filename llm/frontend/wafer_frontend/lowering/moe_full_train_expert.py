"""Source-bound EP1 expert native records with signed schedule scratch."""

from __future__ import annotations

from ..errors import SchemaError
from ..passes.moe_full_train_expert_microplan import plan_moe_expert_native_forward
from ..schema.artifact_manifest import (
    AddressRelocation, CommandFragment, CoreFragmentStream, FragmentKind,
    ProgramSymbol, ProgramSymbolKind, RecordOpcode, RecordOperand,
    RelocatableRecord, SemanticOperandId,
)
from ..schema.global_action import GlobalAction
from ..schema.ir1 import IR1
from ..schema.ir2 import BufferBinding, IntraDieSchedule, MoeExpertScratchRole
from .coarse import _buffer_abi, _program_symbol


_PRODUCER = "moe_full_train_expert_lowering"


def lower_moe_expert_record_fragment(
    action: GlobalAction, schedule: IntraDieSchedule, ir1: IR1,
    *, source_global_dag_id: str,
) -> CommandFragment:
    """Emit four native compute opcodes using distinct signed scratch roots."""
    plan = plan_moe_expert_native_forward(action, schedule, ir1)
    if not source_global_dag_id:
        raise SchemaError("expert fragment needs a source GlobalActionDAG", path="source_global_dag_id")
    scratch_by_role = {item.role: item.binding
                       for item in schedule.moe_expert_scratch_bindings
                       if item.task_id == action.source.task_id}
    scratch = (
        scratch_by_role[MoeExpertScratchRole.GATE_UP_CONCAT],
        scratch_by_role[MoeExpertScratchRole.SWIGLU_ACTIVATED],
    )
    if (scratch[0].id != plan.concat_scratch_binding_ref
            or scratch[1].id != plan.activated_scratch_binding_ref):
        raise SchemaError("fragment scratch differs from signed schedule sidecar",
                          path="schedule.moe_expert_scratch_bindings")
    value_to_binding = {binding.id: binding for binding in schedule.buffer_bindings}
    value_to_binding[plan.operations[0].output_ref] = scratch[0]
    value_to_binding[plan.operations[2].output_ref] = scratch[1]

    symbols: dict[str, ProgramSymbol] = {}
    records: list[RelocatableRecord] = []
    relocations: list[AddressRelocation] = []

    def symbol(binding: BufferBinding, kind: ProgramSymbolKind) -> ProgramSymbol:
        result = _program_symbol(schedule_id=schedule.id, binding=binding, kind=kind)
        symbols[result.id] = result
        return result

    def emit(opcode: RecordOpcode, operands: tuple[RecordOperand, ...],
             addresses: tuple[tuple[SemanticOperandId, ProgramSymbol, int], ...]) -> None:
        index = len(records)
        records.append(RelocatableRecord(action.id, opcode, operands))
        relocations.extend(AddressRelocation(index, operand_id, sym.kind, sym.id, addend)
                           for operand_id, sym, addend in addresses)

    for op in plan.operations:
        source = value_to_binding[op.input_ref]
        target = value_to_binding[op.output_ref]
        input_label = symbol(source, ProgramSymbolKind.SRAM_LABEL)
        output_label = symbol(target, ProgramSymbolKind.SRAM_LABEL)
        emit(RecordOpcode.SRAM_BIND, (
            RecordOperand.literal("input_count", 1),
            RecordOperand.address("input_label_0",
                                  SemanticOperandId.SRAM_BIND_INPUT_0,
                                  input_label.id),
            *(RecordOperand.literal(f"input_label_{index}", 0)
              for index in range(1, 16)),
            RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
                                  output_label.id),
        ), ((SemanticOperandId.SRAM_BIND_INPUT_0, input_label, 0),
            (SemanticOperandId.SRAM_BIND_OUTPUT, output_label, 0)))
        input_address = symbol(source, ProgramSymbolKind.ABSOLUTE_ADDRESS)
        output_address = symbol(target, ProgramSymbolKind.ABSOLUTE_ADDRESS)
        data_binding = value_to_binding[op.weight_ref] if op.weight_ref else None
        data_address = symbol(data_binding, ProgramSymbolKind.ABSOLUTE_ADDRESS) if data_binding else None
        emit(op.opcode, (
            RecordOperand.literal("datatype", 1),
            RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                                  input_address.id),
            (RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS,
                                   data_address.id)
             if data_address is not None else RecordOperand.literal("data_address", 0)),
            RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                                  output_address.id),
            RecordOperand.literal("parameters", op.parameters),
        ), ((SemanticOperandId.COMPUTE_INPUT_ADDRESS, input_address, 0),
            *((((SemanticOperandId.COMPUTE_DATA_ADDRESS, data_address, 0),)
               if data_address is not None else ())),
            (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, output_address,
             op.output_offset_bytes)))
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER, source_global_dag_id=source_global_dag_id,
        kind=FragmentKind.COARSE, claimed_action_ids=(action.id,),
        core_streams=(CoreFragmentStream(
            action.logical_core, tuple(records), (),
            tuple(sorted(relocations, key=lambda relocation:
                         (relocation.record_index, int(relocation.operand_id))))),),
        runtime_symbols=(), program_symbols=tuple(sorted(symbols.values(),
                                                           key=lambda symbol: symbol.id)),
        buffer_abi=tuple(sorted(
            (_buffer_abi(schedule.id, binding, action.logical_core)
             for binding in (*(
                 value_to_binding[ref] for ref in (
                     plan.operations[0].input_ref,
                     plan.operations[0].weight_ref,
                     plan.operations[1].weight_ref,
                     plan.operations[3].weight_ref,
                     plan.operations[3].output_ref,
                 )), *scratch)),
            key=lambda abi: abi.id)),
        state_abi=(),
    )
    fragment.validate()
    return fragment


__all__ = ["lower_moe_expert_record_fragment"]
