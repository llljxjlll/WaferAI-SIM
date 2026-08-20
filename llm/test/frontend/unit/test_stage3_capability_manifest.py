from __future__ import annotations

from fractions import Fraction
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_stage2_capability_manifest,
    build_stage3_capability_manifest,
    build_stage3_case_matrix,
)
from llm.frontend.wafer_frontend.schema.capability import (
    CapabilityStage,
    CapabilityStatus,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.stage3_profile import Stage3ProfileMode
from llm.frontend.wafer_frontend.schema.stage3_static_profile_evidence import (
    STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
)

from test_capability_manifest import _stage1a_truth
from test_stage2_dense_forward_evidence import _report as _stage2_report
from test_stage3_static_profile_evidence import _report as _stage3_report


def _stage2_truth():
    reports = {tp: _stage2_report(tp) for tp in (1, 2, 4)}
    return build_stage2_capability_manifest(
        *_stage1a_truth(),
        tp1_report=reports[1][0],
        tp1_oracle=reports[1][1],
        tp2_report=reports[2][0],
        tp2_oracle=reports[2][1],
        tp4_report=reports[4][0],
        tp4_oracle=reports[4][1],
        dense_forward_fail_closed_evidence_digest="6" * 64,
    )


class Stage3CapabilityManifestTest(unittest.TestCase):
    def test_matrix_extension_is_zero_score_and_immutable(self) -> None:
        stage2_matrix, stage2_manifest = _stage2_truth()
        before = (
            canonical_digest(stage2_matrix),
            canonical_digest(stage2_manifest),
        )
        matrix = build_stage3_case_matrix(stage2_matrix)
        self.assertEqual(build_stage3_case_matrix(stage2_matrix), matrix)
        self.assertEqual(
            (
                canonical_digest(stage2_matrix),
                canonical_digest(stage2_manifest),
            ),
            before,
        )
        definition = next(
            item
            for item in matrix.capabilities
            if item.key == "s1.foundation.static_profile_fail_closed"
        )
        case = next(
            item
            for item in matrix.cases
            if item.id == "case.s1.foundation.static_profile_fail_closed"
        )
        self.assertEqual(
            (definition.score_numerator, definition.score_denominator),
            (0, 1),
        )
        self.assertIs(case.maximum_status, CapabilityStatus.UNIT_ONLY)
        with self.assertRaisesRegex(SchemaError, "collide"):
            build_stage3_case_matrix(matrix)

    def test_exact_profiles_upgrade_only_their_three_existing_cases(self) -> None:
        stage2_matrix, stage2_manifest = _stage2_truth()
        reports = {
            mode: _stage3_report(mode) for mode in Stage3ProfileMode
        }
        kwargs = {
            "prefill_report": reports[Stage3ProfileMode.PREFILL][0],
            "prefill_oracle": reports[Stage3ProfileMode.PREFILL][1],
            "decode_report": reports[Stage3ProfileMode.DECODE][0],
            "decode_oracle": reports[Stage3ProfileMode.DECODE][1],
            "mixed_report": reports[Stage3ProfileMode.MIXED][0],
            "mixed_oracle": reports[Stage3ProfileMode.MIXED][1],
            "static_profile_fail_closed_evidence_digest": "7" * 64,
        }
        matrix, manifest = build_stage3_capability_manifest(
            stage2_matrix, stage2_manifest, **kwargs
        )
        self.assertEqual(
            build_stage3_capability_manifest(
                stage2_matrix, stage2_manifest, **kwargs
            ),
            (matrix, manifest),
        )
        self.assertEqual(
            manifest.baseline_epoch, STAGE3_STATIC_PROFILE_BASELINE_EPOCH
        )
        self.assertEqual(
            manifest.coverage_score(CapabilityStage.S1), Fraction(7, 2)
        )
        self.assertEqual(
            manifest.acceptance_score(CapabilityStage.S1), (1, 3)
        )
        claims = {item.key: item for item in manifest.claims}
        for key in (
            "s1.naive_case.f_p1",
            "s1.naive_case.f_d1",
            "s1.naive_case.f_m1",
        ):
            self.assertIs(claims[key].status, CapabilityStatus.E2E_TIMING)
            self.assertEqual(claims[key].score_numerator, 0)
        self.assertIs(
            claims["s1.foundation.static_profile_fail_closed"].status,
            CapabilityStatus.UNIT_ONLY,
        )
        for key in (
            "s1.naive_case.f_p2",
            "s1.naive_case.f_pdf",
            "s1.naive_case.f_pdr",
            "s1.naive_case.f_pds",
            "s1.naive_case.f_tf",
        ):
            self.assertIs(claims[key].status, CapabilityStatus.UNSUPPORTED)
        self.assertTrue(
            all(
                item.status is CapabilityStatus.UNSUPPORTED
                for item in manifest.claims
                if item.stage in (CapabilityStage.S2, CapabilityStage.S3)
            )
        )

    def test_partial_pair_and_wrong_mode_fail_closed(self) -> None:
        stage2_matrix, stage2_manifest = _stage2_truth()
        prefill_report, prefill_oracle = _stage3_report(
            Stage3ProfileMode.PREFILL
        )
        decode_report, decode_oracle = _stage3_report(
            Stage3ProfileMode.DECODE
        )
        with self.assertRaisesRegex(SchemaError, "supplied together"):
            build_stage3_capability_manifest(
                stage2_matrix,
                stage2_manifest,
                prefill_oracle=prefill_oracle,
            )
        with self.assertRaisesRegex(SchemaError, "prefill evidence"):
            build_stage3_capability_manifest(
                stage2_matrix,
                stage2_manifest,
                prefill_oracle=decode_oracle,
                prefill_report=decode_report,
            )
        matrix, manifest = build_stage3_capability_manifest(
            stage2_matrix,
            stage2_manifest,
            prefill_oracle=prefill_oracle,
            prefill_report=prefill_report,
        )
        claims = {item.key: item for item in manifest.claims}
        self.assertIs(
            claims["s1.naive_case.f_p1"].status,
            CapabilityStatus.E2E_TIMING,
        )
        self.assertIs(
            claims["s1.naive_case.f_d1"].status,
            CapabilityStatus.UNSUPPORTED,
        )
        manifest.validate_against(matrix)


if __name__ == "__main__":
    unittest.main()
