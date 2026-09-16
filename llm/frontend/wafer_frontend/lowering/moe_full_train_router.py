"""Physical EP1 MoE router score matmul from signed action and ScheduleSet."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressRelocation, CommandFragment, CoreFragmentStream, FragmentKind,
    ProgramSymbolKind, RecordOpcode, RecordOperand, RelocatableRecord,
    SemanticOperandId,
)
from ..schema.global_action import GlobalAction, SemanticTaskKind
from ..schema.ir0 import OpKind
from ..schema.moe_full_training_block_workload import (
    MoeForwardBlockKind, MoeFullTrainingBlockWorkload,
)
from ..schema.ir1 import IR1
from ..schema.ir2 import BufferUseRole, IntraDieSchedule
from .coarse import _buffer_abi, _program_symbol


_PRODUCER = "moe_full_train_router_lowering"


def lower_moe_router_score_fragment(
    action: GlobalAction, schedule: IntraDieSchedule, ir1: IR1,
    *, source_global_dag_id: str,
) -> CommandFragment:
    """Emit one true score MATMUL; all three addresses remain source-bound."""
    compute = action.compute
    if (action.task_kind is not SemanticTaskKind.COMP
            or action.op_kind is not OpKind.MOE_ROUTER
            or compute is None or compute.impl_ref != "moe_router"
            or type(compute.workload) is not MoeFullTrainingBlockWorkload
            or compute.workload.kind is not MoeForwardBlockKind.ROUTER
            or compute.workload.expert_count != 1
            or action.logical_core is None
            or action.source.schedule_id != schedule.id
            or schedule.die_id != action.logical_core.die_id
            or not source_global_dag_id):
        raise SchemaError("router leaf requires scheduled EP1 physical source",
                          path="action.compute")
    uses = {(use.role, use.operand_index): use for use in action.buffer_uses}
    if set(uses) != {(BufferUseRole.COMP_INPUT, 0),
                     (BufferUseRole.COMP_INPUT, 1),
                     (BufferUseRole.COMP_OUTPUT, 0)}:
        raise SchemaError("router score needs activation, gate weight and score",
                          path="action.buffer_uses")
    bindings = {binding.id: binding for binding in schedule.buffer_bindings}
    inputs = tuple(bindings[uses[(BufferUseRole.COMP_INPUT, i)].binding_id]
                   for i in range(2))
    output = bindings[uses[(BufferUseRole.COMP_OUTPUT, 0)].binding_id]
    workload = compute.workload
    m, h = workload.token_count, workload.hidden_size
    if (inputs[0].tensor_slice.shape != (m, h)
            or inputs[1].tensor_slice.shape != (h, 1)
            or output.tensor_slice.shape != (m, 1)
            or tuple(binding.size_bytes for binding in (*inputs, output))
               != (2*m*h, 2*h, 2*m)
            or len({binding.storage_id for binding in (*inputs, output)}) != 3):
        raise SchemaError("router score physical FP16 extents or storage drifted",
                          path="schedule.buffer_bindings")
    source_label = _program_symbol(schedule_id=schedule.id, binding=inputs[0],
                                   kind=ProgramSymbolKind.SRAM_LABEL)
    output_label = _program_symbol(schedule_id=schedule.id, binding=output,
                                   kind=ProgramSymbolKind.SRAM_LABEL)
    addresses = tuple(_program_symbol(schedule_id=schedule.id, binding=binding,
                                      kind=ProgramSymbolKind.ABSOLUTE_ADDRESS)
                      for binding in (*inputs, output))
    records = (
        RelocatableRecord(action.id, RecordOpcode.SRAM_BIND, (
            RecordOperand.literal("input_count", 1),
            RecordOperand.address("input_label_0", SemanticOperandId.SRAM_BIND_INPUT_0,
                                  source_label.id),
            *(RecordOperand.literal(f"input_label_{index}", 0)
              for index in range(1, 16)),
            RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
                                  output_label.id),
        )),
        RelocatableRecord(action.id, RecordOpcode.MATMUL, (
            RecordOperand.literal("datatype", 1),
            RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                                  addresses[0].id),
            RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS,
                                  addresses[1].id),
            RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                                  addresses[2].id),
            RecordOperand.literal("parameters", (1, m, h, 1)),
        )),
    )
    relocations = (
        AddressRelocation(0, SemanticOperandId.SRAM_BIND_INPUT_0,
                          source_label.kind, source_label.id, 0),
        AddressRelocation(0, SemanticOperandId.SRAM_BIND_OUTPUT,
                          output_label.kind, output_label.id, 0),
        AddressRelocation(1, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                          addresses[0].kind, addresses[0].id, 0),
        AddressRelocation(1, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                          addresses[1].kind, addresses[1].id, 0),
        AddressRelocation(1, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                          addresses[2].kind, addresses[2].id, 0),
    )
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER, source_global_dag_id=source_global_dag_id,
        kind=FragmentKind.COARSE, claimed_action_ids=(action.id,),
        core_streams=(CoreFragmentStream(action.logical_core, records, (), relocations),),
        runtime_symbols=(),
        program_symbols=tuple(sorted((source_label, output_label, *addresses),
                                     key=lambda symbol: symbol.id)),
        buffer_abi=tuple(sorted((_buffer_abi(schedule.id, binding, action.logical_core)
                                 for binding in (*inputs, output)), key=lambda abi: abi.id)),
        state_abi=(),
    )
    fragment.validate()
    return fragment


__all__ = ["lower_moe_router_score_fragment"]
