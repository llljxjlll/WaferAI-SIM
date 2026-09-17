"""Exact public ISA and relocation closure for one expert reverse macro."""

from __future__ import annotations

from ..errors import SchemaError
from .artifact_manifest import (
    BufferABI, ProgramSymbol, ProgramSymbolKind, RecordOpcode,
    RelocatableRecord, SemanticOperandId, AddressRelocation,
)
from .common import DType
from .global_action import GlobalAction
from .ir0 import OpKind
from .ir2 import BufferOwnership, BufferUseRole
from .moe_expert_backward_workload import MoeExpertBackwardWorkload


def validate_moe_expert_backward_records(
    action: GlobalAction, producer_pass: str,
    records: tuple[RelocatableRecord, ...], indices: list[int],
    buffer_abi: tuple[BufferABI, ...],
    relocations: tuple[AddressRelocation, ...],
    symbols: dict[str, ProgramSymbol], *, path: str,
) -> None:
    compute = action.compute
    if (producer_pass != "moe_full_train_expert_backward_lowering"
            or action.op_kind is not OpKind.MOE_EXPERT_BACKWARD
            or compute is None
            or compute.impl_ref != "moe_expert_backward_recompute"
            or type(compute.workload) is not MoeExpertBackwardWorkload
            or len(indices) != 21):
        raise SchemaError("expert reverse requires ten compute pairs and one local reduction",
                          path=path)
    uses = {(use.role, use.operand_index): use for use in action.buffer_uses}
    if (len(uses) != 9 or set(uses) != {
            *((BufferUseRole.COMP_INPUT, index) for index in range(5)),
            *((BufferUseRole.COMP_OUTPUT, index) for index in range(4))}):
        raise SchemaError("expert reverse public operands changed", path=path)
    abis = {item.binding_id: item for item in buffer_abi}
    if len(abis) != 14:
        raise SchemaError("expert reverse requires nine public and five scratch ABIs",
                          path=path)
    x, gate, up, down, dy = tuple(
        abis[uses[(BufferUseRole.COMP_INPUT, index)].binding_id]
        for index in range(5))
    dx, dgate, dup, ddown = tuple(
        abis[uses[(BufferUseRole.COMP_OUTPUT, index)].binding_id]
        for index in range(4))
    task = action.source.task_id
    roles = ("backward_gate_up_concat", "backward_swiglu_activated",
             "backward_activated_gradient", "backward_gate_up_gradient",
             "backward_dx_parts")
    scratch = []
    for role in roles:
        matches = [item for item in buffer_abi
                   if item.value_id == f"{task}:{role}"]
        if len(matches) != 1:
            raise SchemaError("expert reverse scratch role missing or duplicated",
                              path=path)
        scratch.append(matches[0])
    concat, activated, dactivated, gate_up_dy, dx_parts = scratch
    workload = compute.workload
    m, h, i = workload.token_count, workload.hidden_size, workload.intermediate_size
    projected = 2*m*i
    row_bytes = 2*m*h
    stride = (row_bytes + 63) // 64 * 64
    all_abis = (x, gate, up, down, dy, dx, dgate, dup, ddown,
                *scratch)
    if (set(abis) != {item.binding_id for item in all_abis}
            or len({item.storage_id for item in all_abis}) != 14
            or tuple(item.size_bytes for item in all_abis) != (
                2*m*h, 2*h*i, 2*h*i, 2*i*h, 2*m*h,
                2*m*h, 4*h*i, 4*h*i, 4*i*h,
                2*projected, projected, projected,
                2*projected, stride + row_bytes)
            or tuple(item.dtype for item in all_abis) !=
                (DType.FP16,) * 6 + (DType.FP32,) * 3 + (DType.FP16,) * 5
            or any(item.ownership is not BufferOwnership.OWNED
                   for item in scratch)):
        raise SchemaError("expert reverse ABI geometry or ownership changed",
                          path=path)
    reloc = {(item.record_index, item.operand_id): item
             for item in relocations}

    def expect(index: int, operand: SemanticOperandId,
               kind: ProgramSymbolKind, source: str, offset: int) -> None:
        item = reloc.get((index, operand))
        symbol = symbols.get(item.symbol_ref) if item else None
        if (item is None or symbol is None or symbol.kind is not kind
                or symbol.source_ref != source or item.addend != offset):
            raise SchemaError("expert reverse record relocated to wrong source/tape",
                              path=path)

    operations = (
        (RecordOpcode.MATMUL, x, 0, gate, 0, concat, 0,
         {"datatype": 1, "parameters": (1,m,h,i)}),
        (RecordOpcode.MATMUL, x, 0, up, 0, concat, projected,
         {"datatype": 1, "parameters": (1,m,h,i)}),
        (RecordOpcode.SWIGLU, concat, 0, None, 0, activated, 0,
         {"datatype": 1, "parameters": (m*i,)}),
        (RecordOpcode.GEMM_DX_TIMING, down, 0, dy, 0, dactivated, 0,
         {"weight_datatype": 1, "upstream_datatype": 1, "dx_datatype": 1,
          "m": i, "n": h, "k": m}),
        (RecordOpcode.SWIGLU_BACKWARD_TIMING, concat, 0, dactivated, 0,
         gate_up_dy, 0, {"datatype": 1, "parameters": (m*i,)}),
        (RecordOpcode.GEMM_DX_TIMING, gate, 0, gate_up_dy, 0, dx_parts, 0,
         {"weight_datatype": 1, "upstream_datatype": 1, "dx_datatype": 1,
          "m": h, "n": i, "k": m}),
        (RecordOpcode.GEMM_DX_TIMING, up, 0, gate_up_dy, projected,
         dx_parts, stride,
         {"weight_datatype": 1, "upstream_datatype": 1, "dx_datatype": 1,
          "m": h, "n": i, "k": m}),
        (RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING, activated, 0, dy, 0,
         ddown, 0,
         {"activation_datatype": 1, "upstream_datatype": 1,
          "gradient_datatype": 3, "m": i, "n": h, "k": m}),
        (RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING, x, 0, gate_up_dy, 0,
         dgate, 0,
         {"activation_datatype": 1, "upstream_datatype": 1,
          "gradient_datatype": 3, "m": h, "n": i, "k": m}),
        (RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING, x, 0, gate_up_dy, projected,
         dup, 0,
         {"activation_datatype": 1, "upstream_datatype": 1,
          "gradient_datatype": 3, "m": h, "n": i, "k": m}),
    )
    for op_index, (opcode, inp, input_offset, data, data_offset,
                   out, output_offset, literal_values) in enumerate(operations):
        bind_index, compute_index = indices[2*op_index:2*op_index+2]
        bind, operation = records[bind_index], records[compute_index]
        if (compute_index != bind_index + 1
                or bind.opcode is not RecordOpcode.SRAM_BIND
                or opcode is not operation.opcode
                or bind.operands[0].literal_value != (
                    2 if opcode in (RecordOpcode.GEMM_DX_TIMING,
                                    RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
                                    RecordOpcode.SWIGLU_BACKWARD_TIMING) else 1)
                or {operand.name: operand.literal_value
                    for operand in operation.operands
                    if operand.literal_value is not None} !=
                    ({**literal_values, "data_address": 0}
                     if data is None else literal_values)):
            raise SchemaError("expert reverse opcode or frozen literal drifted",
                              path=path)
        expect(bind_index, SemanticOperandId.SRAM_BIND_INPUT_0,
               ProgramSymbolKind.SRAM_LABEL, inp.storage_id, 0)
        expect(bind_index, SemanticOperandId.SRAM_BIND_OUTPUT,
               ProgramSymbolKind.SRAM_LABEL, out.storage_id, 0)
        if opcode in (RecordOpcode.GEMM_DX_TIMING,
                      RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
                      RecordOpcode.SWIGLU_BACKWARD_TIMING):
            expect(bind_index, SemanticOperandId.SRAM_BIND_INPUT_1,
                   ProgramSymbolKind.SRAM_LABEL, data.storage_id, 0)
        expect(compute_index, SemanticOperandId.COMPUTE_INPUT_ADDRESS,
               ProgramSymbolKind.ABSOLUTE_ADDRESS, inp.binding_id, input_offset)
        if data is not None:
            expect(compute_index, SemanticOperandId.COMPUTE_DATA_ADDRESS,
                   ProgramSymbolKind.ABSOLUTE_ADDRESS, data.binding_id, data_offset)
        expect(compute_index, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
               ProgramSymbolKind.ABSOLUTE_ADDRESS, out.binding_id, output_offset)
    reduce_index = indices[-1]
    reduce = records[reduce_index]
    expected_reduce = {
        "input_dtype": 0, "accumulator_dtype": 1, "output_dtype": 0,
        "reduce_op": 1, "rounding": 0, "order": 0, "input_count": 2,
        "element_count": m*h, "input_stride_bytes": stride,
    }
    if (reduce.opcode is not RecordOpcode.LOCAL_REDUCE
            or {operand.name: operand.literal_value
                for operand in reduce.operands
                if operand.literal_value is not None} != expected_reduce):
        raise SchemaError("expert reverse dX sum literal/stride changed", path=path)
    expect(reduce_index, SemanticOperandId.SOURCE_ADDRESS,
           ProgramSymbolKind.ABSOLUTE_ADDRESS, dx_parts.binding_id, 0)
    expect(reduce_index, SemanticOperandId.DESTINATION_ADDRESS,
           ProgramSymbolKind.ABSOLUTE_ADDRESS, dx.binding_id, 0)


__all__ = ["validate_moe_expert_backward_records"]
