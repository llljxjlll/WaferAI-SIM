"""Native 0x23/0x24 preserve genuine typed TRAIN operands into fixed records."""

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
    RecordOperand, RelocatableRecord, SemanticOperandId,
    _compute_record_abi,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import (
    EffectKind, NodeEffects, NodeMath, NumericalPolicy, OpKind,
)
from llm.frontend.wafer_frontend.schema.moe_training_ir0_workloads import (
    EmbeddingTableWgradWorkload, NormGammaWgradWorkload,
)


def _contract(kind: OpKind, workload: object, impl_ref: str) -> ComputeContract:
    reads, writes = canonical_compute_operand_roles(kind, workload, tiled=False)
    return ComputeContract(
        kind, workload, NodeMath(DType.FP32, NumericalPolicy.TOLERANCE),
        NodeEffects(EffectKind.PURE, None, None), impl_ref,
        tuple(ComputeOperand(f"read.{index}", role) for index, role in enumerate(reads)),
        tuple(ComputeOperand(f"write.{index}", role) for index, role in enumerate(writes)),
    )


def _symbols(count: int) -> tuple[ProgramSymbol, ...]:
    return tuple(ProgramSymbol(f"p.{index}", ProgramSymbolKind.ABSOLUTE_ADDRESS,
                               f"source.{index}") for index in range(count))


class WeightGradientPublicBridgeTest(unittest.TestCase):
    def test_nonzero_indexed_embedding_scatter_has_three_real_sources(self) -> None:
        trace = (3, 3, 5, 7) + (0,) * 12
        workload = EmbeddingTableWgradWorkload(4, 4, 1, 16, 0, 16, 8, trace)
        compute = _contract(OpKind.EMBEDDING_TABLE_WGRAD, workload,
                            "embedding_table_wgrad_timing")
        compute.validate("compute")
        abi = _compute_record_abi(compute, path="compute")
        self.assertEqual((abi.opcode, abi.bind_input_count,
                          abi.data_input_index, abi.aux_input_index),
                         (RecordOpcode.EMBEDDING_TABLE_WGRAD_TIMING, 3, 1, 2))
        sources = _symbols(4)
        record = RelocatableRecord("source", abi.opcode,
            _fixed_compute_operands(compute, abi, *sources))
        record.validate("record")
        self.assertEqual(len(record.operands), 31)
        self.assertEqual(tuple(item.symbol_ref for item in record.operands
                               if item.kind is OperandKind.ADDRESS_SYMBOL),
                         tuple(symbol.id for symbol in sources))
        self.assertEqual(tuple(item.literal_value for item in record.operands[-16:]), trace)
        self.assertEqual(record.operands[6].operand_id, SemanticOperandId.COMPUTE_AUX_ADDRESS)
        self.assertEqual((workload.input_index_bytes, workload.table_tile_bytes,
                          workload.upstream_bytes, workload.physical_gradient_bytes),
                         (16, 256, 64, 512))
        for name, value in (("index01", 16), ("index04", 2),
                            ("gradient_datatype", 1), ("vocab_start", 15)):
            operands = list(record.operands)
            index = next(i for i, item in enumerate(operands) if item.name == name)
            operands[index] = RecordOperand.literal(name, value)
            with self.subTest(tamper=name), self.assertRaises(SchemaError):
                replace(record, operands=tuple(operands)).validate("tampered")

    def test_norm_gamma_wgrad_has_independent_fp32_output(self) -> None:
        workload = NormGammaWgradWorkload(4, 2, 2, 8, 0)
        compute = _contract(OpKind.NORM_GAMMA_WGRAD, workload,
                            "norm_gamma_wgrad_timing")
        compute.validate("compute")
        abi = _compute_record_abi(compute, path="compute")
        self.assertEqual((abi.opcode, abi.bind_input_count,
                          abi.data_input_index, abi.aux_input_index),
                         (RecordOpcode.NORM_GAMMA_WGRAD_TIMING, 2, 1, None))
        source, upstream, gradient = _symbols(3)
        record = RelocatableRecord("source", abi.opcode,
            _fixed_compute_operands(compute, abi, source, upstream, None, gradient))
        record.validate("record")
        self.assertEqual((workload.forward_bytes, workload.upstream_bytes,
                          workload.physical_gradient_bytes), (32, 32, 32))
        for name, value in (("gradient_datatype", 1), ("mode", 2),
                            ("logical_rows", 3)):
            operands = list(record.operands)
            index = next(i for i, item in enumerate(operands) if item.name == name)
            operands[index] = RecordOperand.literal(name, value)
            with self.subTest(tamper=name), self.assertRaises(SchemaError):
                replace(record, operands=tuple(operands)).validate("tampered")


if __name__ == "__main__":
    unittest.main()
