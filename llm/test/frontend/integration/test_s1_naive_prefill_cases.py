from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
    REGION_MANIFEST_SCHEMA_VERSION,
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
)
from llm.frontend.wafer_frontend.schema.s1_naive_evidence import (
    S1_N_BASELINE_EPOCH,
    S1NaivePrefillCase,
)

from s1_naive_prefill_cases import build_s1_naive_prefill_case


class S1NaivePrefillCurrentCaseTest(unittest.TestCase):
    def test_f_p1_f_p2_use_current_epoch_and_exact_policy(self) -> None:
        expected = {
            S1NaivePrefillCase.F_P1: (1, 44, 44, 159, 0),
            S1NaivePrefillCase.F_P2: (2, 160, 92, 510, 2048),
        }
        for case_id, golden in expected.items():
            with self.subTest(case=case_id.value):
                case = build_s1_naive_prefill_case(case_id)
                source = case.source
                leaves = source.profile.leaf_fragments
                opcodes = Counter(
                    record.opcode.name
                    for fragment in leaves
                    for stream in fragment.core_streams
                    for record in stream.records
                )
                collective_bytes = (
                    source.oracle.collectives.all_gather.group_payload_bytes_total
                    + source.oracle.collectives.reduce_scatter.group_payload_bytes_total
                )
                self.assertEqual(case.baseline_epoch, S1_N_BASELINE_EPOCH)
                self.assertEqual(
                    (
                        case.tp_degree,
                        len(source.global_dag.actions),
                        len(leaves),
                        sum(opcodes.values()),
                        collective_bytes,
                    ),
                    golden,
                )
                self.assertEqual(
                    source.lowered_bundle.schema_version,
                    LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
                )
                self.assertEqual(
                    source.linked_bundle.schema_version,
                    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
                )
                self.assertEqual(
                    source.manifest.schema_version,
                    LINKED_PROGRAM_MANIFEST_SCHEMA_VERSION,
                )
                self.assertTrue(
                    all(
                        fragment.schema_version == COMMAND_FRAGMENT_SCHEMA_VERSION
                        for fragment in leaves
                    )
                )
                self.assertTrue(
                    all(
                        item.schema_version == REGION_MANIFEST_SCHEMA_VERSION
                        for item in source.lowered.fragments
                        if type(item) is RegionManifest
                    )
                )
                case.policy.validate_against(
                    source.planning_context, source.scheduling_context
                )

    def test_policy_context_tamper_is_rejected(self) -> None:
        case = build_s1_naive_prefill_case(S1NaivePrefillCase.F_P1)
        forged = replace(
            case.policy,
            scheduling_context_id="intra_die_scheduling_context_forged",
        )
        with self.assertRaisesRegex(
            SchemaError, "does not match the supplied production policy contexts"
        ):
            forged.validate_against(
                case.source.planning_context,
                case.source.scheduling_context,
            )


if __name__ == "__main__":
    unittest.main()
