from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import link_lite_moe_n6, lower_lite_moe_n6
from llm.frontend.wafer_frontend.schema import (
    LITE_MOE_LINKED_PROGRAM_SCHEMA_VERSION,
    LITE_MOE_LOWERED_PROGRAM_SCHEMA_VERSION,
    LiteMoeLinkedProgram,
    LiteMoeLoweredProgram,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    ManifestInputKind,
    RecordOpcode,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass

from lite_moe_cases import build_lite_moe_execution_case


class LiteMoeLinkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_lite_moe_execution_case()
        cls.lowered = lower_lite_moe_n6(
            cls.case.n6_intent,
            cls.case.global_dag,
            cls.case.schedule,
            cls.case.projection,
            cls.case.n4,
        )
        cls.linked = link_lite_moe_n6(cls.lowered)

    def test_exact_lower_and_single_manifest(self) -> None:
        records = tuple(
            record
            for fragment in self.lowered.fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(len(self.lowered.fragments), 72)
        self.assertEqual(
            Counter(record.opcode for record in records),
            Counter({
                RecordOpcode.SRAM_ALLOC_AT: 64,
                RecordOpcode.SRAM_FREE: 64,
                RecordOpcode.SRAM_BIND: 32,
                RecordOpcode.MATMUL: 24,
                RecordOpcode.SWIGLU: 8,
                RecordOpcode.LSU_LOAD: 24,
                RecordOpcode.DTE_SEND: 8,
                RecordOpcode.DTE_RECV: 8,
                RecordOpcode.DTE_WAIT: 8,
            }),
        )
        manifest = self.linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.core_streams),
                sum(len(stream.records) for stream in manifest.core_streams),
                len(manifest.input_digests),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
            ),
            (72, 2, 240, 77, 34, 141, 384, 24),
        )
        self.assertEqual(
            Counter(item.kind for item in manifest.input_digests),
            Counter({
                ManifestInputKind.S3_LITE_MOE: 1,
                ManifestInputKind.IR1: 1,
                ManifestInputKind.IR2_PROJECTION: 1,
                ManifestInputKind.SCHEDULE_SET: 1,
                ManifestInputKind.GLOBAL_ACTION_DAG: 1,
                ManifestInputKind.COMMAND_FRAGMENT: 72,
            }),
        )

    def test_packed_views_round_trip_and_tamper_fail_closed(self) -> None:
        compute = tuple(
            binding
            for binding in self.linked.manifest.address_operand_bindings
            if binding.operand_id in (
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
            )
        )
        self.assertIn((0, 64), tuple((item.tensor_slices[0].offset[-1], item.tensor_slices[0].shape[-1]) for item in compute))
        self.assertIn((0, 32), tuple((item.tensor_slices[0].offset[-1], item.tensor_slices[0].shape[-1]) for item in compute))
        self.assertIn((32, 32), tuple((item.tensor_slices[0].offset[-1], item.tensor_slices[0].shape[-1]) for item in compute))
        self.assertEqual(
            loads_dataclass(LiteMoeLoweredProgram, canonical_json(self.lowered), path="lowered"),
            self.lowered,
        )
        self.assertEqual(
            loads_dataclass(LiteMoeLinkedProgram, canonical_json(self.linked), path="linked"),
            self.linked,
        )
        with self.assertRaises(SchemaError):
            replace(
                self.linked,
                manifest=replace(
                    self.linked.manifest,
                    address_operand_bindings=self.linked.manifest.address_operand_bindings[:-1],
                ),
            ).validate()

    def test_versions_and_immediate_old_wire_rejected(self) -> None:
        self.assertEqual(
            (
                LITE_MOE_LOWERED_PROGRAM_SCHEMA_VERSION,
                LITE_MOE_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            (
                "wafer_frontend.s3_lite_moe_lowered_program/v1alpha1",
                "wafer_frontend.s3_lite_moe_linked_program/v1alpha1",
            ),
        )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                self.lowered.fragments[0],
                schema_version="wafer_frontend.command_fragment/v1alpha12",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unsupported lowered schema"):
            replace(
                self.lowered,
                schema_version="wafer_frontend.s3_lite_moe_lowered_program/v1alpha0",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unsupported linked schema"):
            replace(
                self.linked,
                schema_version="wafer_frontend.s3_lite_moe_linked_program/v1alpha0",
            ).validate()


if __name__ == "__main__":
    unittest.main()
