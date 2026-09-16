"""Public 0x27/28 ABI tests; only a scoped router primitive, never E2E training."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode, RecordOperand, RelocatableRecord, SemanticOperandId,
    _compute_record_abi,
)


class MoeSignedRouterPublicRecordTest(unittest.TestCase):
    @staticmethod
    def forward() -> RelocatableRecord:
        s = SemanticOperandId
        return RelocatableRecord("source_scoped_router_forward", RecordOpcode.MOE_SCORE_WEIGHTED_FORWARD, (
            RecordOperand.literal("route_datatype", 2),
            RecordOperand.literal("score_datatype", 1),
            RecordOperand.literal("expert_datatype", 1),
            RecordOperand.literal("combined_datatype", 1),
            RecordOperand.address("route_address", s.COMPUTE_ROUTE_TABLE_ADDRESS, "abs.route"),
            RecordOperand.address("score_address", s.COMPUTE_INPUT_ADDRESS, "abs.score"),
            RecordOperand.address("return_address", s.COMPUTE_DATA_ADDRESS, "abs.return"),
            RecordOperand.address("combined_address", s.COMPUTE_OUTPUT_ADDRESS, "abs.combined"),
            RecordOperand.literal("rank_rows", 4),
            RecordOperand.literal("hidden_size", 4),
            RecordOperand.literal("expert_count", 2),
            RecordOperand.literal("route_bytes", 80),
        ))

    @staticmethod
    def backward() -> RelocatableRecord:
        s = SemanticOperandId
        return RelocatableRecord("source_scoped_router_backward", RecordOpcode.MOE_SCORE_WEIGHT_BACKWARD, (
            RecordOperand.literal("route_datatype", 2),
            RecordOperand.literal("score_datatype", 1),
            RecordOperand.literal("expert_datatype", 1),
            RecordOperand.literal("upstream_datatype", 1),
            RecordOperand.literal("dscore_datatype", 1),
            RecordOperand.literal("dexpert_datatype", 1),
            RecordOperand.address("route_address", s.COMPUTE_ROUTE_TABLE_ADDRESS, "abs.route"),
            RecordOperand.address("score_address", s.COMPUTE_INPUT_ADDRESS, "abs.score"),
            RecordOperand.address("return_address", s.COMPUTE_DATA_ADDRESS, "abs.return"),
            RecordOperand.address("dcombined_address", s.COMPUTE_OUTPUT_ADDRESS, "abs.upstream"),
            RecordOperand.address("dscore_address", s.COMPUTE_AUX_ADDRESS, "abs.dscore"),
            RecordOperand.address("dexpert_address", s.COMPUTE_ROUTER_DEXPERT_ADDRESS, "abs.dexpert"),
            RecordOperand.literal("rank_rows", 4),
            RecordOperand.literal("hidden_size", 4),
            RecordOperand.literal("expert_count", 2),
            RecordOperand.literal("route_bytes", 80),
        ))

    @staticmethod
    def change(record: RelocatableRecord, name: str, replacement: RecordOperand) -> RelocatableRecord:
        return replace(record, operands=tuple(
            replacement if op.name == name else op for op in record.operands
        ))

    def test_complete_route_and_distinct_backward_outputs(self):
        self.forward().validate("public_router_forward")
        self.backward().validate("public_router_backward")
        outputs = self.backward().operands[10:12]
        self.assertEqual(tuple(op.operand_id for op in outputs), (
            SemanticOperandId.COMPUTE_AUX_ADDRESS,
            SemanticOperandId.COMPUTE_ROUTER_DEXPERT_ADDRESS,
        ))
        self.assertNotEqual(outputs[0].symbol_ref, outputs[1].symbol_ref)

    def test_reject_truncated_route(self):
        for original in (self.forward(), self.backward()):
            bad = self.change(original, "route_bytes", RecordOperand.literal("route_bytes", 64))
            with self.assertRaises(SchemaError):
                bad.validate("truncated_route")

    def test_reject_wrong_dtypes(self):
        for name, value in (("route_datatype", 1), ("score_datatype", 3),
                            ("dscore_datatype", 3), ("dexpert_datatype", 3)):
            original = self.backward()
            bad = self.change(original, name, RecordOperand.literal(name, value))
            with self.subTest(name=name), self.assertRaises(SchemaError):
                bad.validate("wrong_dtype")

    def test_reject_borrowed_single_output_address(self):
        original = self.backward()
        bad = self.change(original, "dexpert_address", RecordOperand.address(
            "dexpert_address", SemanticOperandId.COMPUTE_AUX_ADDRESS, "abs.dscore"))
        with self.assertRaises(SchemaError):
            bad.validate("dexpert_aliased_operand_id")

    def test_not_mapped_to_unproven_full_training_compute(self):
        self.assertNotIn("moe_score_weight_backward", _compute_record_abi.__globals__["_COMPUTE_OPCODE_BY_IMPL_REF"])
        self.assertNotIn("moe_score_weighted_forward", _compute_record_abi.__globals__["_COMPUTE_OPCODE_BY_IMPL_REF"])


if __name__ == "__main__":
    unittest.main()
