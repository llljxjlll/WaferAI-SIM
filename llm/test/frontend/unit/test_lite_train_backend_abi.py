from __future__ import annotations

from dataclasses import replace
import struct
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    REGION_MANIFEST_SCHEMA_VERSION,
    OperandKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    SemanticOperandId,
    _OPERAND_SCHEMAS,
    _address_operand_role,
)
from llm.frontend.wafer_frontend.schema.ir2 import BufferUseRole
from llm.frontend.wafer_frontend.schema.n6 import (
    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    STAGE4_LINKED_PROGRAM_SCHEMA_VERSION,
    STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION,
)
from llm.frontend.wafer_frontend.schema.train_n6 import (
    TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
    TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
)


def _bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _record(
    opcode: RecordOpcode,
    literals: dict[str, int],
) -> RelocatableRecord:
    operands: list[RecordOperand] = []
    for spec in _OPERAND_SCHEMAS[opcode]:
        if OperandKind.ADDRESS_SYMBOL in spec.allowed_kinds:
            symbol = f"symbol_{spec.name}"
            if opcode is RecordOpcode.SGD_UPDATE and spec.name in (
                "weight_address",
                "updated_weight_address",
            ):
                symbol = "symbol_weight_in_place"
            if opcode is RecordOpcode.ADAMW_UPDATE:
                aliases = {
                    "updated_weight_address": "weight_address",
                    "updated_master_weight_address": "master_weight_address",
                    "updated_first_moment_address": "first_moment_address",
                    "updated_second_moment_address": "second_moment_address",
                    "updated_step_counter_address": "step_counter_address",
                }
                symbol = f"symbol_{aliases.get(spec.name, spec.name)}"
            operands.append(RecordOperand.address(spec.name, spec.operand_id, symbol))
        else:
            operands.append(RecordOperand.literal(spec.name, literals[spec.name]))
    return RelocatableRecord("action", opcode, tuple(operands))


def _tamper(
    record: RelocatableRecord,
    name: str,
    value: int,
) -> RelocatableRecord:
    operands = list(record.operands)
    index = next(i for i, operand in enumerate(operands) if operand.name == name)
    operands[index] = RecordOperand.literal(name, value)
    return replace(record, operands=tuple(operands))


class LiteTrainBackendAbiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ce = _record(
            RecordOpcode.CROSS_ENTROPY_BACKWARD,
            {
                "logits_datatype": 1,
                "label_datatype": 2,
                "upstream_datatype": 3,
                "output_datatype": 1,
                "reduction": 0,
                "upstream_mode": 1,
                "logical_rows": 8,
                "rank_rows": 4,
                "tp_degree": 2,
                "vocab_size": 32,
                "upstream_elements": 4,
            },
        )
        self.sgd = _record(
            RecordOpcode.SGD_UPDATE,
            {
                "weight_datatype": 1,
                "gradient_datatype": 3,
                "output_datatype": 1,
                "rounding": 0,
                "element_count": 128,
                "learning_rate_f64_bits": _bits(0.125),
                "momentum_f64_bits": 0,
            },
        )
        self.adamw = _record(
            RecordOpcode.ADAMW_UPDATE,
            {
                "weight_datatype": 1,
                "gradient_datatype": 3,
                "state_datatype": 3,
                "output_datatype": 1,
                "rounding": 0,
                "element_count": 128,
                "step": 2,
                "learning_rate_f64_bits": _bits(0.001),
                "beta1_f64_bits": _bits(0.9),
                "beta2_f64_bits": _bits(0.999),
                "epsilon_f64_bits": _bits(1.0e-8),
                "weight_decay_f64_bits": _bits(0.01),
            },
        )

    def test_versions_opcodes_operands_and_roles_are_exact(self) -> None:
        self.assertEqual(
            (
                COMMAND_FRAGMENT_SCHEMA_VERSION,
                REGION_MANIFEST_SCHEMA_VERSION,
                LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
                LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
                LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
                STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION,
                STAGE4_LINKED_PROGRAM_SCHEMA_VERSION,
                TRAIN_LOWERED_PROGRAM_SCHEMA_VERSION,
                TRAIN_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            (
                "wafer_frontend.command_fragment/v1alpha13",
                "wafer_frontend.region_manifest/v1alpha12",
                "wafer_frontend.linked_program_manifest/v1alpha14",
                "wafer_frontend.lowered_program_bundle/v1alpha9",
                "wafer_frontend.linked_program_bundle/v1alpha10",
                "wafer_frontend.stage4_lowered_program/v1alpha5",
                "wafer_frontend.stage4_linked_program/v1alpha5",
                "wafer_frontend.train_lowered_program/v1alpha3",
                "wafer_frontend.train_linked_program/v1alpha3",
            ),
        )
        self.assertEqual(
            (int(self.ce.opcode), int(self.sgd.opcode)),
            (0x1F, 0x20),
        )
        self.assertEqual(
            tuple(operand.name for operand in self.ce.operands),
            (
                "logits_datatype",
                "label_datatype",
                "upstream_datatype",
                "output_datatype",
                "reduction",
                "upstream_mode",
                "logits_address",
                "labels_address",
                "upstream_address",
                "logits_grad_address",
                "logical_rows",
                "rank_rows",
                "tp_degree",
                "vocab_size",
                "upstream_elements",
            ),
        )
        self.assertEqual(
            tuple(operand.name for operand in self.sgd.operands),
            (
                "weight_datatype",
                "gradient_datatype",
                "output_datatype",
                "rounding",
                "weight_address",
                "gradient_address",
                "updated_weight_address",
                "element_count",
                "learning_rate_f64_bits",
                "momentum_f64_bits",
            ),
        )
        for record in (self.ce, self.sgd):
            record.validate("record")
        self.assertEqual(
            _address_operand_role(
                self.ce.opcode,
                SemanticOperandId.COMPUTE_AUX_ADDRESS,
                "record",
            ),
            (BufferUseRole.COMP_INPUT, 2),
        )
        self.assertEqual(
            _address_operand_role(
                self.sgd.opcode,
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
                "record",
            ),
            (BufferUseRole.COMP_OUTPUT, 0),
        )

    def test_ce_backward_scalar_and_per_row_are_exact(self) -> None:
        scalar = _tamper(_tamper(self.ce, "upstream_mode", 0), "upstream_elements", 1)
        scalar.validate("scalar")
        for name, value in (
            ("logits_datatype", 3),
            ("label_datatype", 1),
            ("upstream_datatype", 1),
            ("output_datatype", 3),
            ("reduction", 1),
            ("upstream_mode", 2),
            ("upstream_elements", 1),
            ("logical_rows", 9),
            ("vocab_size", 1),
        ):
            with self.subTest(name=name):
                with self.assertRaises(SchemaError):
                    _tamper(self.ce, name, value).validate("ce")

    def test_sgd_dtype_rounding_lr_momentum_and_alias_are_exact(self) -> None:
        for name, value in (
            ("weight_datatype", 3),
            ("gradient_datatype", 1),
            ("output_datatype", 3),
            ("rounding", 1),
            ("element_count", 0),
            ("learning_rate_f64_bits", _bits(0.0)),
            ("learning_rate_f64_bits", _bits(float("inf"))),
            ("momentum_f64_bits", _bits(0.5)),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(SchemaError):
                    _tamper(self.sgd, name, value).validate("sgd")

        operands = list(self.sgd.operands)
        operands[6] = RecordOperand.address(
            "updated_weight_address",
            SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
            "symbol_not_in_place",
        )
        with self.assertRaisesRegex(SchemaError, "in-place"):
            replace(self.sgd, operands=tuple(operands)).validate("sgd")

    def test_adamw_state_abi_and_hyperparameters_are_exact(self) -> None:
        self.adamw.validate("adamw")
        self.assertEqual(int(self.adamw.opcode), 0x21)
        self.assertEqual(len(self.adamw.operands), 23)
        self.assertEqual(
            _address_operand_role(
                self.adamw.opcode,
                SemanticOperandId.COMPUTE_SECOND_MOMENT_ADDRESS,
                "adamw",
            ),
            (BufferUseRole.COMP_INPUT, 4),
        )
        self.assertEqual(
            _address_operand_role(
                self.adamw.opcode,
                SemanticOperandId.COMPUTE_UPDATED_STEP_ADDRESS,
                "adamw",
            ),
            (BufferUseRole.COMP_OUTPUT, 4),
        )
        for name, value in (
            ("state_datatype", 1),
            ("step", 0),
            ("learning_rate_f64_bits", _bits(0.0)),
            ("beta1_f64_bits", _bits(1.0)),
            ("beta2_f64_bits", _bits(0.0)),
            ("epsilon_f64_bits", _bits(float("inf"))),
            ("weight_decay_f64_bits", _bits(-0.01)),
        ):
            with self.subTest(name=name):
                with self.assertRaises(SchemaError):
                    _tamper(self.adamw, name, value).validate("adamw")

        operands = list(self.adamw.operands)
        index = next(
            i
            for i, operand in enumerate(operands)
            if operand.name == "updated_first_moment_address"
        )
        operands[index] = RecordOperand.address(
            "updated_first_moment_address",
            SemanticOperandId.COMPUTE_UPDATED_FIRST_MOMENT_ADDRESS,
            "symbol_not_in_place",
        )
        with self.assertRaisesRegex(SchemaError, "in place"):
            replace(self.adamw, operands=tuple(operands)).validate("adamw")


if __name__ == "__main__":
    unittest.main()
