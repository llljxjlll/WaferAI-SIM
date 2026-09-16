"""Public 0x26 maps source W/dY to independent FP16 dX fixed record."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.coarse import _fixed_compute_operands
from llm.frontend.wafer_frontend.schema.action import (
    ComputeContract, ComputeOperand, canonical_compute_operand_roles,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    OperandKind, ProgramSymbol, ProgramSymbolKind, RecordOpcode,
    RecordOperand, RelocatableRecord, SemanticOperandId, _compute_record_abi,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.gemm_input_dx_workload import GemmInputDxWorkload
from llm.frontend.wafer_frontend.schema.ir0 import (
    EffectKind, NodeEffects, NodeMath, NumericalPolicy, OpKind,
)


class GemmInputDxPublicBridgeTest(unittest.TestCase):
    def test_named_record_preserves_real_weight_upstream_and_fp16_output(self) -> None:
        workload = GemmInputDxWorkload(4, 8, 16, "T0.layer0.gate_up", "weight.state0")
        reads, writes = canonical_compute_operand_roles(
            OpKind.GEMM_INPUT_DX, workload, tiled=False,
        )
        self.assertEqual((reads, writes),
                         (("forward_weight", "upstream_gradient"),
                          ("input_gradient",)))
        compute = ComputeContract(
            OpKind.GEMM_INPUT_DX, workload,
            NodeMath(DType.FP32, NumericalPolicy.TOLERANCE),
            NodeEffects(EffectKind.PURE, None, None), "gemm_input_dx_timing",
            tuple(ComputeOperand(f"read.{i}", role) for i, role in enumerate(reads)),
            tuple(ComputeOperand(f"write.{i}", role) for i, role in enumerate(writes)),
        )
        compute.validate("compute")
        abi = _compute_record_abi(compute, path="compute")
        self.assertEqual((abi.opcode, abi.bind_input_count,
                          abi.data_input_index, abi.aux_input_index),
                         (RecordOpcode.GEMM_DX_TIMING, 2, 1, None))
        weight, dy, dx = (
            ProgramSymbol(f"p.{i}", ProgramSymbolKind.ABSOLUTE_ADDRESS,
                          f"storage.{i}") for i in range(3)
        )
        record = RelocatableRecord("source", abi.opcode,
            _fixed_compute_operands(compute, abi, weight, dy, None, dx))
        record.validate("record")
        self.assertEqual(tuple(item.symbol_ref for item in record.operands
                               if item.kind is OperandKind.ADDRESS_SYMBOL),
                         (weight.id, dy.id, dx.id))
        self.assertEqual(tuple(item.operand_id for item in record.operands
                               if item.kind is OperandKind.ADDRESS_SYMBOL),
                         (SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                          SemanticOperandId.COMPUTE_DATA_ADDRESS,
                          SemanticOperandId.COMPUTE_OUTPUT_ADDRESS))
        self.assertEqual((workload.weight_bytes, workload.upstream_bytes,
                          workload.output_bytes), (256, 128, 64))
        for name, value in (("dx_datatype", 3), ("k", 0), ("n", 10000)):
            operands = list(record.operands)
            index = next(i for i, item in enumerate(operands) if item.name == name)
            operands[index] = RecordOperand.literal(name, value)
            with self.subTest(tamper=name), self.assertRaises(SchemaError):
                replace(record, operands=tuple(operands)).validate("tampered")


if __name__ == "__main__":
    unittest.main()
