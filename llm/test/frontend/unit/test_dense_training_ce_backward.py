"""Check CE gradient derives its physical tape from the forward CE record."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_training_ce_backward import (
    build_dense_training_ce_backward_record,
    build_dense_training_ce_backward_from_loss_gradient,
    dense_training_per_row_loss_gradient_seed,
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
        with self.assertRaisesRegex(SchemaError, "legacy timing surrogate"):
            build_dense_training_ce_backward_record(_forward(), **symbols)

    def test_independent_loss_gradient_and_nonzero_per_row_seed(self) -> None:
        symbols = _symbols()
        independent = ProgramSymbol("symbol_independent_dloss", ProgramSymbolKind.ABSOLUTE_ADDRESS,
                                    "sram:dloss")
        result = build_dense_training_ce_backward_from_loss_gradient(
            _forward(), logits=symbols["logits"], labels=symbols["labels"],
            loss_gradient=independent, logits_gradient=symbols["logits_gradient"],
        )
        self.assertIs(result.opcode, RecordOpcode.CROSS_ENTROPY_BACKWARD)
        self.assertEqual(result.operands[8].symbol_ref, independent.id)
        self.assertNotEqual(result.operands[8].symbol_ref, _forward().operands[6].symbol_ref)
        self.assertEqual(dense_training_per_row_loss_gradient_seed(4), b"\x00\x00\x80\x3f" * 4)
        result.validate("independent_dloss_native")

    def test_forward_loss_cannot_be_its_own_gradient(self) -> None:
        symbols = _symbols()
        with self.assertRaisesRegex(SchemaError, "independent of forward loss"):
            build_dense_training_ce_backward_from_loss_gradient(
                _forward(), logits=symbols["logits"], labels=symbols["labels"],
                loss_gradient=symbols["upstream"],
                logits_gradient=symbols["logits_gradient"],
            )

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
