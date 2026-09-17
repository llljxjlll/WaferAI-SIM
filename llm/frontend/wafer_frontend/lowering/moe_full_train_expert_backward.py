"""Native EP1 expert reverse from real dExpert and signed recomputation SRAM."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    AddressRelocation, CommandFragment, CoreFragmentStream, FragmentKind,
    ProgramSymbol, ProgramSymbolKind, RecordOpcode, RecordOperand,
    RelocatableRecord, SemanticOperandId,
)
from ..schema.common import DType
from ..schema.global_action import GlobalAction
from ..schema.ir0 import OpKind
from ..schema.ir1 import IR1
from ..schema.ir2 import (
    BufferBinding, BufferOwnership, BufferUseRole, IntraDieSchedule,
    MoeExpertScratchRole, SemanticTaskKind,
)
from ..schema.moe_expert_backward_workload import MoeExpertBackwardWorkload
from .coarse import _buffer_abi, _program_symbol

_PRODUCER = "moe_full_train_expert_backward_lowering"


def lower_moe_expert_backward_fragment(
    action: GlobalAction, schedule: IntraDieSchedule, ir1: IR1,
    *, source_global_dag_id: str,
) -> CommandFragment:
    """Emit recompute, down/gate/up dX, SwiGLU reverse and three FP32 WGRADs."""
    if (action.task_kind is not SemanticTaskKind.COMP
            or action.op_kind is not OpKind.MOE_EXPERT_BACKWARD
            or action.compute is None
            or type(action.compute.workload) is not MoeExpertBackwardWorkload
            or action.logical_core is None
            or action.source.schedule_id != schedule.id
            or not source_global_dag_id):
        raise SchemaError("requires one physical source-bound expert reverse",
                          path="action")
    action.validate("action")
    workload = action.compute.workload
    workload.validate("action.compute.workload")
    m, h, i = (workload.token_count, workload.hidden_size,
               workload.intermediate_size)
    if tuple(item.role for item in action.compute.inputs) != (
            "expert_activation", "gate_weight", "up_weight",
            "down_weight", "expert_output_gradient") or tuple(
            item.role for item in action.compute.outputs) != (
            "expert_activation_gradient", "gate_weight_gradient",
            "up_weight_gradient", "down_weight_gradient"):
        raise SchemaError("expert reverse ordered operands drifted",
                          path="action.compute")
    by_id = {item.id: item for item in schedule.buffer_bindings}
    public = []
    for role, count in ((BufferUseRole.COMP_INPUT, 5),
                        (BufferUseRole.COMP_OUTPUT, 4)):
        for index in range(count):
            uses = [item for item in action.buffer_uses
                    if item.role is role and item.operand_index == index]
            if len(uses) != 1 or uses[0].binding_id not in by_id:
                raise SchemaError("expert reverse lacks exact public BufferABI",
                                  path="action.buffer_uses")
            binding = by_id[uses[0].binding_id]
            expected = (action.compute.inputs[index].value_id
                        if role is BufferUseRole.COMP_INPUT else
                        action.compute.outputs[index].value_id)
            if binding.value_id != expected or uses[0].tensor_slice != binding.tensor_slice:
                raise SchemaError("expert reverse public binding source drifted",
                                  path="action.buffer_uses")
            public.append(binding)
    if len(action.buffer_uses) != 9:
        raise SchemaError("expert reverse has extra public bindings",
                          path="action.buffer_uses")
    x, gate, up, down, dy, dx, dgate, dup, ddown = public
    scratch_by_role = {item.role: item.binding for item in
                       schedule.moe_expert_scratch_bindings
                       if item.task_id == action.source.task_id}
    expected_roles = {
        MoeExpertScratchRole.BACKWARD_GATE_UP_CONCAT,
        MoeExpertScratchRole.BACKWARD_SWIGLU_ACTIVATED,
        MoeExpertScratchRole.BACKWARD_ACTIVATED_GRADIENT,
        MoeExpertScratchRole.BACKWARD_GATE_UP_GRADIENT,
        MoeExpertScratchRole.BACKWARD_DX_PARTS,
    }
    if set(scratch_by_role) != expected_roles:
        raise SchemaError("expert reverse needs five signed scratch roots",
                          path="schedule.moe_expert_scratch_bindings")
    concat, activated, dactivated, gate_up_dy, dx_parts = (
        scratch_by_role[role] for role in (
            MoeExpertScratchRole.BACKWARD_GATE_UP_CONCAT,
            MoeExpertScratchRole.BACKWARD_SWIGLU_ACTIVATED,
            MoeExpertScratchRole.BACKWARD_ACTIVATED_GRADIENT,
            MoeExpertScratchRole.BACKWARD_GATE_UP_GRADIENT,
            MoeExpertScratchRole.BACKWARD_DX_PARTS,
        )
    )
    scratch = (concat, activated, dactivated, gate_up_dy, dx_parts)
    expected_bytes = (2*m*h, 2*h*i, 2*h*i, 2*i*h, 2*m*h,
                      2*m*h, 4*h*i, 4*h*i, 4*i*h)
    if (tuple(item.size_bytes for item in public) != expected_bytes
            or tuple(item.dtype for item in public) !=
                (DType.FP16,) * 6 + (DType.FP32,) * 3
            or any(item.ownership is not BufferOwnership.OWNED
                   for item in scratch)
            or len({item.storage_id for item in (*public, *scratch)}) != 14
            or len({item.core_id for item in (*public, *scratch)}) != 1
            or len({item.region_ref for item in (*public, *scratch)}) != 1):
        raise SchemaError("expert reverse public/scratch geometry or ownership drifted",
                          path="schedule")
    row_bytes = 2 * m * h
    projected_bytes = 2 * m * i
    stride = (row_bytes + 63) // 64 * 64
    if (tuple(item.size_bytes for item in scratch) !=
            (2*projected_bytes, projected_bytes, projected_bytes,
             2*projected_bytes, stride + row_bytes)):
        raise SchemaError("expert reverse scratch extents differ from physical stride",
                          path="schedule.moe_expert_scratch_bindings")
    ir1.validate("ir1")
    schedule.validate("schedule")

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
        relocations.extend(AddressRelocation(index, operand_id, sym.kind, sym.id, offset)
                           for operand_id, sym, offset in addresses)

    def bind(input_binding: BufferBinding, output_binding: BufferBinding,
             second_input: BufferBinding | None = None) -> None:
        inp = symbol(input_binding, ProgramSymbolKind.SRAM_LABEL)
        second = (symbol(second_input, ProgramSymbolKind.SRAM_LABEL)
                  if second_input is not None else None)
        out = symbol(output_binding, ProgramSymbolKind.SRAM_LABEL)
        emit(RecordOpcode.SRAM_BIND, (
            RecordOperand.literal("input_count", 2 if second else 1),
            RecordOperand.address("input_label_0", SemanticOperandId.SRAM_BIND_INPUT_0,
                                  inp.id),
            (RecordOperand.address("input_label_1", SemanticOperandId.SRAM_BIND_INPUT_1,
                                   second.id) if second else
             RecordOperand.literal("input_label_1", 0)),
            *(RecordOperand.literal(f"input_label_{index}", 0)
              for index in range(2, 16)),
            RecordOperand.address("output_label", SemanticOperandId.SRAM_BIND_OUTPUT,
                                  out.id),
        ), ((SemanticOperandId.SRAM_BIND_INPUT_0, inp, 0),
            *((((SemanticOperandId.SRAM_BIND_INPUT_1, second, 0),)
               if second else ())),
            (SemanticOperandId.SRAM_BIND_OUTPUT, out, 0)))

    def compute(opcode: RecordOpcode,
                input_binding: BufferBinding, input_offset: int,
                data_binding: BufferBinding | None, data_offset: int,
                output_binding: BufferBinding, output_offset: int,
                literals: tuple[tuple[str, int | tuple[int, ...]], ...]) -> None:
        bind(input_binding, output_binding,
             data_binding if opcode in (
                 RecordOpcode.GEMM_DX_TIMING,
                 RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
                 RecordOpcode.SWIGLU_BACKWARD_TIMING) else None)
        inp = symbol(input_binding, ProgramSymbolKind.ABSOLUTE_ADDRESS)
        data = (symbol(data_binding, ProgramSymbolKind.ABSOLUTE_ADDRESS)
                if data_binding is not None else None)
        out = symbol(output_binding, ProgramSymbolKind.ABSOLUTE_ADDRESS)
        if opcode in (RecordOpcode.MATMUL, RecordOpcode.SWIGLU,
                      RecordOpcode.SWIGLU_BACKWARD_TIMING):
            operands = (
                RecordOperand.literal("datatype", 1),
                RecordOperand.address("input_address", SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                                      inp.id),
                (RecordOperand.address("data_address", SemanticOperandId.COMPUTE_DATA_ADDRESS,
                                       data.id) if data else
                 RecordOperand.literal("data_address", 0)),
                RecordOperand.address("output_address", SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                                      out.id),
                *(RecordOperand.literal(name, value) for name, value in literals),
            )
        else:
            names = {
                RecordOpcode.GEMM_DX_TIMING:
                    ("weight_datatype", "upstream_datatype", "dx_datatype",
                     "weight_address", "upstream_address", "dx_address"),
                RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING:
                    ("activation_datatype", "upstream_datatype", "gradient_datatype",
                     "activation_address", "upstream_address", "gradient_address"),
            }[opcode]
            operands = (
                *(RecordOperand.literal(name, 3 if opcode is RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING
                                        and position == 2 else 1)
                  for position, name in enumerate(names[:3])),
                RecordOperand.address(names[3], SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                                      inp.id),
                RecordOperand.address(names[4], SemanticOperandId.COMPUTE_DATA_ADDRESS,
                                      data.id),
                RecordOperand.address(names[5], SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                                      out.id),
                *(RecordOperand.literal(name, value) for name, value in literals),
            )
        emit(opcode, operands, (
            (SemanticOperandId.COMPUTE_INPUT_ADDRESS, inp, input_offset),
            *((((SemanticOperandId.COMPUTE_DATA_ADDRESS, data, data_offset),)
               if data else ())),
            (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, out, output_offset),
        ))

    # Recompute gate/up from true X and weight states; they share one signed
    # concat scratch with disjoint half-tensor relocation views.
    compute(RecordOpcode.MATMUL, x, 0, gate, 0, concat, 0,
            (("parameters", (1, m, h, i)),))
    compute(RecordOpcode.MATMUL, x, 0, up, 0, concat, projected_bytes,
            (("parameters", (1, m, h, i)),))
    compute(RecordOpcode.SWIGLU, concat, 0, None, 0, activated, 0,
            (("parameters", (m*i,)),))
    # dY→dActivated→SwiGLU gate/up derivative from recomputed real tape.
    compute(RecordOpcode.GEMM_DX_TIMING, down, 0, dy, 0, dactivated, 0,
            (("m", i), ("n", h), ("k", m)))
    compute(RecordOpcode.SWIGLU_BACKWARD_TIMING, concat, 0, dactivated, 0,
            gate_up_dy, 0, (("parameters", (m*i,)),))
    compute(RecordOpcode.GEMM_DX_TIMING, gate, 0, gate_up_dy, 0,
            dx_parts, 0, (("m", h), ("n", i), ("k", m)))
    compute(RecordOpcode.GEMM_DX_TIMING, up, 0, gate_up_dy, projected_bytes,
            dx_parts, stride, (("m", h), ("n", i), ("k", m)))
    # WGRAD has a true FP16 activation, true derivative, FP32 separate output.
    compute(RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
            activated, 0, dy, 0, ddown, 0,
            (("m", i), ("n", h), ("k", m)))
    compute(RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
            x, 0, gate_up_dy, 0, dgate, 0,
            (("m", h), ("n", i), ("k", m)))
    compute(RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
            x, 0, gate_up_dy, projected_bytes, dup, 0,
            (("m", h), ("n", i), ("k", m)))
    # LOCAL_REDUCE is address-driven, not a COMPUTE SRAM_BIND consumer.
    # Its two inputs occupy the signed scratch span at a 64B physical stride.
    src = symbol(dx_parts, ProgramSymbolKind.ABSOLUTE_ADDRESS)
    dst = symbol(dx, ProgramSymbolKind.ABSOLUTE_ADDRESS)
    emit(RecordOpcode.LOCAL_REDUCE, (
        RecordOperand.literal("input_dtype", 0),
        RecordOperand.literal("accumulator_dtype", 1),
        RecordOperand.literal("output_dtype", 0),
        RecordOperand.literal("reduce_op", 1),
        RecordOperand.literal("rounding", 0),
        RecordOperand.literal("order", 0),
        RecordOperand.literal("input_count", 2),
        RecordOperand.literal("element_count", m*h),
        RecordOperand.literal("input_stride_bytes", stride),
        RecordOperand.address("source_address", SemanticOperandId.SOURCE_ADDRESS,
                              src.id),
        RecordOperand.address("destination_address", SemanticOperandId.DESTINATION_ADDRESS,
                              dst.id),
    ), ((SemanticOperandId.SOURCE_ADDRESS, src, 0),
        (SemanticOperandId.DESTINATION_ADDRESS, dst, 0)))
    fragment = CommandFragment.create(
        producer_pass=_PRODUCER, source_global_dag_id=source_global_dag_id,
        kind=FragmentKind.COARSE, claimed_action_ids=(action.id,),
        core_streams=(CoreFragmentStream(
            action.logical_core, tuple(records), (),
            tuple(sorted(relocations, key=lambda item:
                         (item.record_index, int(item.operand_id))))),),
        runtime_symbols=(),
        program_symbols=tuple(sorted(symbols.values(), key=lambda item: item.id)),
        buffer_abi=tuple(sorted(
            (_buffer_abi(schedule.id, binding, action.logical_core)
             for binding in (*public, *scratch)),
            key=lambda item: item.id)),
        state_abi=(),
    )
    fragment.validate()
    return fragment


__all__ = ["lower_moe_expert_backward_fragment"]
