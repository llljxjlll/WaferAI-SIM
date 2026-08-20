from __future__ import annotations

from dataclasses import replace
import unittest

from test_train_forward_global_action import _global_action

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.coarse import NaiveCoarseLowering
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    REGION_MANIFEST_SCHEMA_VERSION,
    OperandKind,
    RecordOpcode,
    RecordOperand,
    SemanticOperandId,
    _compute_record_abi,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import AttentionMode, OpKind
from llm.frontend.wafer_frontend.schema.ir2 import BufferUseRole
from llm.frontend.wafer_frontend.schema.n6 import (
    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    STAGE4_LINKED_PROGRAM_SCHEMA_VERSION,
    STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION,
)
from llm.frontend.wafer_frontend.schema.train_n6 import (
    train_replica_lowering_context,
)


class CrossEntropyArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _scheduled, global_action = _global_action()
        cls.replica = global_action.replicas[0]
        cls.context = train_replica_lowering_context(cls.replica)
        cls.action = next(
            action
            for action in cls.replica.global_dag.actions
            if action.op_kind is OpKind.CE_FORWARD
        )
        cls.fragment = NaiveCoarseLowering().lower(cls.action, cls.context)
        cls.attention_action = next(
            action
            for action in cls.replica.global_dag.actions
            if action.op_kind is OpKind.ATTENTION
            and action.compute.workload.mode is AttentionMode.TRAIN_FORWARD
        )
        cls.attention_fragment = NaiveCoarseLowering().lower(
            cls.attention_action,
            cls.context,
        )

    def test_fixed_wire_lowering_roles_bytes_and_versions(self) -> None:
        fragment = self.fragment
        fragment.validate_against(self.replica.global_dag)
        self.assertEqual(
            (
                COMMAND_FRAGMENT_SCHEMA_VERSION,
                REGION_MANIFEST_SCHEMA_VERSION,
                LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
                LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
                LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
                STAGE4_LOWERED_PROGRAM_SCHEMA_VERSION,
                STAGE4_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            (
                "wafer_frontend.command_fragment/v1alpha12",
                "wafer_frontend.region_manifest/v1alpha11",
                "wafer_frontend.linked_program_manifest/v1alpha13",
                "wafer_frontend.lowered_program_bundle/v1alpha8",
                "wafer_frontend.linked_program_bundle/v1alpha9",
                "wafer_frontend.stage4_lowered_program/v1alpha4",
                "wafer_frontend.stage4_linked_program/v1alpha4",
            ),
        )
        self.assertEqual(len(fragment.core_streams), 1)
        bind, record = fragment.core_streams[0].records
        self.assertEqual(bind.opcode, RecordOpcode.SRAM_BIND)
        self.assertEqual(bind.operands[0].literal_value, 2)
        self.assertEqual(record.opcode, RecordOpcode.CROSS_ENTROPY_FORWARD)
        self.assertEqual(int(record.opcode), 0x1E)
        self.assertEqual(
            tuple(operand.name for operand in record.operands),
            (
                "logits_datatype",
                "label_datatype",
                "loss_datatype",
                "reduction",
                "logits_address",
                "labels_address",
                "loss_address",
                "logical_rows",
                "rank_rows",
                "tp_degree",
                "vocab_size",
            ),
        )
        self.assertEqual(
            tuple(
                operand.literal_value
                for operand in record.operands
                if operand.kind is OperandKind.LITERAL
            ),
            (1, 2, 3, 0, 8, 4, 2, 32),
        )
        abi = _compute_record_abi(self.action.compute, path="action.compute")
        self.assertEqual(
            (abi.opcode, abi.bind_input_count, abi.data_input_index),
            (RecordOpcode.CROSS_ENTROPY_FORWARD, 2, 1),
        )
        self.assertEqual(
            tuple(
                (relocation.record_index, relocation.operand_id)
                for relocation in fragment.core_streams[0].address_relocations
            ),
            (
                (0, SemanticOperandId.SRAM_BIND_INPUT_0),
                (0, SemanticOperandId.SRAM_BIND_INPUT_1),
                (0, SemanticOperandId.SRAM_BIND_OUTPUT),
                (1, SemanticOperandId.COMPUTE_INPUT_ADDRESS),
                (1, SemanticOperandId.COMPUTE_DATA_ADDRESS),
                (1, SemanticOperandId.COMPUTE_OUTPUT_ADDRESS),
            ),
        )
        bindings = {
            binding.id: binding
            for schedule in self.replica.scheduled.schedule_set.schedules
            for binding in schedule.buffer_bindings
        }
        inputs = tuple(
            use
            for use in self.action.buffer_uses
            if use.role is BufferUseRole.COMP_INPUT
        )
        output = next(
            use
            for use in self.action.buffer_uses
            if use.role is BufferUseRole.COMP_OUTPUT
        )
        self.assertEqual(
            tuple(bindings[use.binding_id].dtype for use in inputs),
            (DType.FP16, DType.INT32),
        )
        self.assertEqual(bindings[output.binding_id].dtype, DType.FP32)
        workload = self.action.compute.workload
        rank_rows = workload.rank_logits_shape[0]
        vocabulary = workload.rank_logits_shape[1]
        self.assertEqual(
            sum(bindings[use.binding_id].size_bytes for use in inputs),
            rank_rows * vocabulary * 2 + rank_rows * 4,
        )
        self.assertEqual(
            bindings[output.binding_id].size_bytes,
            rank_rows * 4,
        )

    def test_dtype_reduction_and_address_tamper_fail_closed(self) -> None:
        record = self.fragment.core_streams[0].records[1]
        for value in (0, 1, 2):
            operands = list(record.operands)
            operands[2] = RecordOperand.literal("loss_datatype", value)
            with self.subTest(loss_datatype=value):
                with self.assertRaisesRegex(SchemaError, "FP16/INT32/FP32"):
                    replace(record, operands=tuple(operands)).validate("record")

        operands = list(record.operands)
        operands[3] = RecordOperand.literal("reduction", 1)
        with self.assertRaisesRegex(SchemaError, "reduction NONE"):
            replace(record, operands=tuple(operands)).validate("record")

        operands = list(record.operands)
        operands[5] = replace(
            operands[5],
            operand_id=SemanticOperandId.COMPUTE_INPUT_ADDRESS,
        )
        with self.assertRaisesRegex(SchemaError, "semantic operand id"):
            replace(record, operands=tuple(operands)).validate("record")

    def test_train_attention_mode3_and_kv_zero_are_exact(self) -> None:
        record = next(
            record
            for record in self.attention_fragment.core_streams[0].records
            if record.opcode is RecordOpcode.ATTENTION_EXACT
        )
        literals = {
            operand.name: operand.literal_value
            for operand in record.operands
            if operand.kind is OperandKind.LITERAL
        }
        self.assertEqual(
            (
                literals["mode"],
                literals["query_tokens"],
                literals["query_key_pairs"],
                literals["rank_kv_read_bytes"],
                literals["rank_kv_write_bytes"],
            ),
            (3, 8, 36, 0, 0),
        )
        record.validate("record")

        for name, value in (
            ("mode", 1),
            ("rank_kv_read_bytes", 1),
            ("rank_kv_write_bytes", 1),
        ):
            operands = tuple(
                RecordOperand.literal(name, value)
                if operand.name == name
                else operand
                for operand in record.operands
            )
            with self.subTest(name=name):
                with self.assertRaisesRegex(SchemaError, "derived counts"):
                    replace(record, operands=operands).validate("record")


if __name__ == "__main__":
    unittest.main()
