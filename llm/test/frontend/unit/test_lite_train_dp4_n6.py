from __future__ import annotations

from dataclasses import replace
import unittest

from llm.test.frontend.unit.test_lite_train_dp4 import _chain

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.lite_train_dp4_link_program import (
    link_s2_lite_dp4_tree_ar,
)
from llm.frontend.wafer_frontend.passes.lite_train_dp4_lower_program import (
    lower_s2_lite_dp4_tree_ar,
)
from llm.frontend.wafer_frontend.passes.lite_train_dp4_n6 import (
    build_s2_lite_dp4_tree_ar_n6_intent,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
    ManifestInputDigest,
    ManifestInputKind,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.lite_train_dp4_n6 import (
    Dp4TreeExecutableKind,
    S2_LITE_DP4_TREE_AR_LINKED_PROGRAM_SCHEMA_VERSION,
    S2_LITE_DP4_TREE_AR_LOWERED_PROGRAM_SCHEMA_VERSION,
    S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION,
    S2LiteDp4TreeArLinkedProgram,
    S2LiteDp4TreeArLoweredProgram,
    S2LiteDp4TreeArN6Intent,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass


class S2LiteDp4TreeArN6Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.global_action = _chain()[-1]
        cls.intent = build_s2_lite_dp4_tree_ar_n6_intent(cls.global_action)
        cls.lowered = lower_s2_lite_dp4_tree_ar(cls.intent)
        cls.linked = link_s2_lite_dp4_tree_ar(cls.lowered)

    def test_exact_intent_lower_and_single_manifest_counts(self) -> None:
        self.assertEqual(
            (
                len(self.intent.lowering_contexts),
                len(self.intent.gradient_buffer_abis),
                len(self.intent.scratch_buffer_abis),
                len(self.intent.units),
            ),
            (4, 4, 4, 23),
        )
        self.assertEqual(
            {
                kind: tuple(unit.kind for unit in self.intent.units).count(kind)
                for kind in Dp4TreeExecutableKind
            },
            {
                Dp4TreeExecutableKind.LOCAL_COPY: 2,
                Dp4TreeExecutableKind.FLOW_SEND: 6,
                Dp4TreeExecutableKind.FLOW_RECV: 6,
                Dp4TreeExecutableKind.FLOW_WAIT: 6,
                Dp4TreeExecutableKind.LOCAL_REDUCE: 3,
            },
        )
        self.assertEqual(
            tuple(
                (abi.logical_core.die_id, abi.region_offset_bytes, abi.size_bytes)
                for abi in self.intent.scratch_buffer_abis
            ),
            (
                (0, 32768, 2048),
                (0, 34816, 2048),
                (2, 32768, 2048),
                (2, 34816, 2048),
            ),
        )
        records = tuple(
            record
            for fragment in self.lowered.overlay_fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(
            (
                len(self.lowered.local_fragments),
                len(self.lowered.overlay_fragments),
                len(records),
                sum(len(fragment.claimed_action_ids) for fragment in self.lowered.overlay_fragments),
            ),
            (184, 17, 33, 23),
        )
        self.assertEqual(
            {
                opcode: tuple(record.opcode for record in records).count(opcode)
                for opcode in (
                    RecordOpcode.SRAM_ALLOC_AT,
                    RecordOpcode.SRAM_FREE,
                    RecordOpcode.DTE_ISSUE,
                    RecordOpcode.DTE_SEND,
                    RecordOpcode.DTE_RECV,
                    RecordOpcode.DTE_WAIT,
                    RecordOpcode.LOCAL_REDUCE,
                )
            },
            {
                RecordOpcode.SRAM_ALLOC_AT: 4,
                RecordOpcode.SRAM_FREE: 4,
                RecordOpcode.DTE_ISSUE: 2,
                RecordOpcode.DTE_SEND: 6,
                RecordOpcode.DTE_RECV: 6,
                RecordOpcode.DTE_WAIT: 8,
                RecordOpcode.LOCAL_REDUCE: 3,
            },
        )
        manifest = self.linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.core_streams),
                sum(len(stream.records) for stream in manifest.core_streams),
                len(manifest.input_digests),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
            ),
            (201, 4, 709, 218, 1262, 68, 30, 448),
        )
        self.assertEqual(
            tuple(len(stream.records) for stream in manifest.core_streams),
            (183, 172, 182, 172),
        )

    def test_versions_and_strict_serde(self) -> None:
        self.assertEqual(
            (
                S2_LITE_DP4_TREE_AR_N6_INTENT_SCHEMA_VERSION,
                S2_LITE_DP4_TREE_AR_LOWERED_PROGRAM_SCHEMA_VERSION,
                S2_LITE_DP4_TREE_AR_LINKED_PROGRAM_SCHEMA_VERSION,
            ),
            (
                "wafer_frontend.s2_lite_dp4_tree_ar_n6_intent/v1alpha1",
                "wafer_frontend.s2_lite_dp4_tree_ar_lowered_program/v1alpha1",
                "wafer_frontend.s2_lite_dp4_tree_ar_linked_program/v1alpha1",
            ),
        )
        self.assertEqual(
            loads_dataclass(S2LiteDp4TreeArN6Intent, canonical_json(self.intent)),
            self.intent,
        )
        self.assertEqual(
            loads_dataclass(S2LiteDp4TreeArLoweredProgram, canonical_json(self.lowered)),
            self.lowered,
        )
        self.assertEqual(
            loads_dataclass(S2LiteDp4TreeArLinkedProgram, canonical_json(self.linked)),
            self.linked,
        )

    def test_scratch_unit_and_leaf_tamper_fail_closed(self) -> None:
        first, second, *rest = self.intent.scratch_buffer_abis
        with self.assertRaisesRegex(SchemaError, "contiguous"):
            replace(
                self.intent,
                scratch_buffer_abis=(
                    first,
                    replace(second, region_offset_bytes=second.region_offset_bytes + 64),
                    *rest,
                ),
            ).validate()
        units = list(self.intent.units)
        units[-1] = replace(units[-1], deps=())
        with self.assertRaisesRegex(SchemaError, "23-unit"):
            replace(self.intent, units=tuple(units)).validate()
        with self.assertRaisesRegex(SchemaError, "17 leaves"):
            replace(
                self.lowered,
                overlay_fragments=self.lowered.overlay_fragments[:-1],
            ).validate()

    def test_top_schema_controls_exact_lineage_cardinality(self) -> None:
        manifest = self.linked.manifest
        top_index = next(
            index
            for index, digest in enumerate(manifest.input_digests)
            if digest.kind is ManifestInputKind.S2_LITE_ROOTED_AR
        )
        top = manifest.input_digests[top_index]
        digests = list(manifest.input_digests)
        digests[top_index] = ManifestInputDigest(
            top.kind,
            top.artifact_id,
            "wafer_frontend.s2_lite_unknown_rooted/v1alpha1",
            top.digest,
        )
        semantic = manifest._semantic_key()
        semantic["input_digests"] = tuple(digests)
        with self.assertRaisesRegex(SchemaError, "unsupported top schema"):
            forged = LinkedProgramManifest.create(
                producer_pass=manifest.producer_pass,
                **semantic,
            )
            forged.validate("forged")


if __name__ == "__main__":
    unittest.main()
