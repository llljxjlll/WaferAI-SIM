from __future__ import annotations

import struct
import unittest
from dataclasses import replace

from wafer_frontend.schema.artifact_manifest import (
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
from wafer_frontend.schema.common import SchemaError
from wafer_frontend.schema.global_action import BufferUseRole


def _fixed_record(
    opcode: RecordOpcode,
    literals: dict[str, object],
) -> RelocatableRecord:
    operands = []
    for spec in _OPERAND_SCHEMAS[opcode]:
        if OperandKind.ADDRESS_SYMBOL in spec.allowed_kinds:
            operands.append(
                RecordOperand.address(
                    spec.name,
                    spec.operand_id,
                    f"symbol_{spec.name}",
                )
            )
        else:
            operands.append(RecordOperand.literal(spec.name, literals[spec.name]))
    return RelocatableRecord("action", opcode, tuple(operands))


def _tamper(
    record: RelocatableRecord,
    name: str,
    value: object,
) -> RelocatableRecord:
    operands = list(record.operands)
    index = next(
        index for index, operand in enumerate(operands) if operand.name == name
    )
    operands[index] = RecordOperand.literal(name, value)
    return replace(record, operands=tuple(operands))


class Stage2ArtifactSchemaTest(unittest.TestCase):
    def test_versions_and_public_opcodes(self) -> None:
        self.assertEqual(COMMAND_FRAGMENT_SCHEMA_VERSION.rsplit("/", 1)[-1], "v1alpha10")
        self.assertEqual(REGION_MANIFEST_SCHEMA_VERSION.rsplit("/", 1)[-1], "v1alpha9")
        self.assertEqual(
            LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION.rsplit("/", 1)[-1],
            "v1alpha11",
        )
        self.assertEqual(
            (
                int(RecordOpcode.ROPE_QK_EXACT),
                int(RecordOpcode.ATTENTION_EXACT),
                int(RecordOpcode.EMBEDDING_LOOKUP),
                int(RecordOpcode.GREEDY_SAMPLE),
            ),
            (0x1A, 0x1B, 0x1C, 0x1D),
        )

    def test_fixed_records_validate_and_tamper_fail_closed(self) -> None:
        records = {
            RecordOpcode.ROPE_QK_EXACT: _fixed_record(
                RecordOpcode.ROPE_QK_EXACT,
                {
                    "datatype": 1,
                    "packed_layout": 0,
                    "logical_tokens": 4,
                    "tp_degree": 2,
                    "num_heads": 4,
                    "num_kv_heads": 2,
                    "rank_num_heads": 2,
                    "rank_num_kv_heads": 1,
                    "head_dim": 8,
                    "rotary_dim": 8,
                    "max_position_embeddings": 32,
                    "context_max": 4,
                    "rope_theta_f64_bits": struct.unpack(
                        "<Q", struct.pack("<d", 10000.0)
                    )[0],
                },
            ),
            RecordOpcode.ATTENTION_EXACT: _fixed_record(
                RecordOpcode.ATTENTION_EXACT,
                {
                    "datatype": 1,
                    "mode": 0,
                    "packed_layout": 0,
                    "causal": True,
                    "query_tokens": 4,
                    "tp_degree": 2,
                    "num_heads": 4,
                    "num_kv_heads": 2,
                    "rank_num_heads": 2,
                    "rank_num_kv_heads": 1,
                    "head_dim": 8,
                    "context_sum": 4,
                    "context_max": 4,
                    "query_key_pairs": 10,
                    "rank_kv_read_bytes": 0,
                    "rank_kv_write_bytes": 128,
                },
            ),
            RecordOpcode.EMBEDDING_LOOKUP: _fixed_record(
                RecordOpcode.EMBEDDING_LOOKUP,
                {
                    "index_datatype": 2,
                    "table_datatype": 1,
                    "output_datatype": 1,
                    "placement": 0,
                    "logical_rows": 4,
                    "rank_rows": 2,
                    "tp_degree": 2,
                    "vocab_size": 32,
                    "hidden_size": 16,
                },
            ),
            RecordOpcode.GREEDY_SAMPLE: _fixed_record(
                RecordOpcode.GREEDY_SAMPLE,
                {
                    "logits_datatype": 1,
                    "output_datatype": 2,
                    "mode": 0,
                    "row_selection": 0,
                    "tp_degree": 1,
                    "token_rows": 4,
                    "vocab_size": 32,
                    "sample_count": 1,
                    "comparisons": 31,
                },
            ),
        }
        for opcode, record in records.items():
            with self.subTest(opcode=opcode):
                record.validate("record")

        _tamper(
            records[RecordOpcode.ROPE_QK_EXACT],
            "logical_tokens",
            8,
        ).validate("rope_batch_record")

        tampered = (
            _tamper(records[RecordOpcode.ROPE_QK_EXACT], "rotary_dim", 7),
            _tamper(records[RecordOpcode.ATTENTION_EXACT], "query_key_pairs", 9),
            _tamper(records[RecordOpcode.EMBEDDING_LOOKUP], "logical_rows", 5),
            _tamper(records[RecordOpcode.GREEDY_SAMPLE], "comparisons", 30),
        )
        for record in tampered:
            with self.subTest(opcode=record.opcode):
                with self.assertRaises(SchemaError):
                    record.validate("record")

    def test_rmsnorm_requires_real_data_operand_role(self) -> None:
        record = _fixed_record(
            RecordOpcode.RMSNORM,
            {
                "datatype": 1,
                "parameters": (1, 4, 16),
            },
        )
        record.validate("record")
        data = next(
            operand
            for operand in record.operands
            if operand.name == "data_address"
        )
        self.assertEqual(data.operand_id, SemanticOperandId.COMPUTE_DATA_ADDRESS)
        self.assertEqual(
            _address_operand_role(
                RecordOpcode.RMSNORM,
                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                "record",
            ),
            (BufferUseRole.COMP_INPUT, 1),
        )


if __name__ == "__main__":
    unittest.main()
