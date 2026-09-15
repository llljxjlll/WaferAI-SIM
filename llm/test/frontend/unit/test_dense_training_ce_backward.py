"""Check CE gradient derives its physical tape from the forward CE record."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_training_ce_backward import (
    build_dense_training_ce_backward_record,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    ProgramSymbol,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
)
from llm.test.frontend.unit.test_lite_train_backend_abi import _record


def _forward():
    return _record(
        RecordOpcode.CROSS_ENTROPY_FORWARD,
        {
            "logits_datatype": 1,
            "label_datatype": 2,
            "loss_datatype": 3,
            "reduction": 0,
            "logical_rows": 16,
            "rank_rows": 4,
            "tp_degree": 4,
            "vocab_size": 32,
        },
    )


def _symbols():
    return {
        key: ProgramSymbol(
            identifier, ProgramSymbolKind.ABSOLUTE_ADDRESS, f"sram:{key}"
        )
        for key, identifier in (
            ("logits", "symbol_logits_address"),
            ("labels", "symbol_labels_address"),
            ("upstream", "symbol_loss_address"),
            ("logits_gradient", "symbol_ce_logits_gradient"),
        )
    }


class DenseNativeCeBackwardTest(unittest.TestCase):
    def test_native_gradient_binds_original_logits_and_labels(self) -> None:
        result = build_dense_training_ce_backward_record(
            _forward(), **_symbols()
        )
        self.assertIs(result.opcode, RecordOpcode.CROSS_ENTROPY_BACKWARD)
        self.assertEqual(
            tuple(item.symbol_ref for item in result.operands[6:10]),
            tuple(item.id for item in _symbols().values()),
        )
        self.assertEqual(
            tuple(item.literal_value for item in result.operands[10:]),
            (16, 4, 4, 32, 4),
        )
        result.validate("native_ce_backward")

    def test_different_forward_label_symbol_fails_closed(self) -> None:
        symbols = _symbols()
        symbols["labels"] = replace(symbols["labels"], id="different_label")
        with self.assertRaisesRegex(SchemaError, "differ from physical forward"):
            build_dense_training_ce_backward_record(_forward(), **symbols)

    def test_different_forward_loss_symbol_fails_closed(self) -> None:
        symbols = _symbols()
        symbols["upstream"] = replace(symbols["upstream"], id="stale_loss")
        with self.assertRaisesRegex(SchemaError, "differ from physical forward"):
            build_dense_training_ce_backward_record(_forward(), **symbols)

    def test_native_ce_backward_cannot_label_output_as_fp32(self) -> None:
        result = build_dense_training_ce_backward_record(
            _forward(), **_symbols()
        )
        operands = list(result.operands)
        operands[3] = RecordOperand.literal("output_datatype", 3)
        with self.assertRaises(SchemaError):
            replace(result, operands=tuple(operands)).validate("illegal_fp32_output")


if __name__ == "__main__":
    unittest.main()
