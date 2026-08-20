from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_stage2_capability_manifest,
    build_stage3_capability_manifest,
    build_stage4_capability_manifest,
    build_stage4_pd_case_matrix,
)
from llm.frontend.wafer_frontend.schema.capability import (
    CapabilityStage,
    CapabilityStatus,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.stage4_pd_evidence import (
    STAGE4_PD_BASELINE_EPOCH,
)

from test_capability_manifest import _stage1a_truth
from test_stage4_pd_evidence import _recreate, _report


def _stage3_truth():
    stage2 = build_stage2_capability_manifest(*_stage1a_truth())
    return build_stage3_capability_manifest(*stage2)


def _ready_matrix():
    fused = _report(1, 1, fused=True)[0]
    pds = _recreate(
        _report(1, 1)[0],
        case_id="case.stage4.pds.tp1",
    )
    pdr = _report(2, 1)[0]
    pdr_manifest_digest = "8" * 64
    pdr_artifact = replace(
        pdr.artifact,
        linked_manifest_id="stage4_pdr_manifest",
        linked_manifest_digest=pdr_manifest_digest,
        program_artifact_sha256="9" * 64,
    )
    pdr_inputs = tuple(
        replace(item, digest=pdr_manifest_digest)
        if item.name == "manifest"
        else item
        for item in pdr.input_digests
    )
    pdr = _recreate(
        pdr,
        case_id="case.stage4.pdr.tp2_to_tp1",
        artifact=pdr_artifact,
        input_digests=pdr_inputs,
    )
    return build_stage4_pd_case_matrix((fused, pds, pdr))


class Stage4CapabilityManifestTest(unittest.TestCase):
    def test_complete_matrix_promotes_only_pd_and_kv_handoff(self) -> None:
        stage3_matrix, stage3_manifest = _stage3_truth()
        before = (
            canonical_digest(stage3_matrix),
            canonical_digest(stage3_manifest),
        )
        stage4_matrix = _ready_matrix()
        matrix, manifest = build_stage4_capability_manifest(
            stage3_matrix,
            stage3_manifest,
            stage4_matrix,
        )
        self.assertIs(matrix, stage3_matrix)
        self.assertEqual(
            build_stage4_capability_manifest(
                stage3_matrix,
                stage3_manifest,
                stage4_matrix,
            ),
            (matrix, manifest),
        )
        self.assertEqual(
            (
                canonical_digest(stage3_matrix),
                canonical_digest(stage3_manifest),
            ),
            before,
        )
        self.assertEqual(manifest.baseline_epoch, STAGE4_PD_BASELINE_EPOCH)
        self.assertEqual(
            manifest.coverage_score(CapabilityStage.S1), Fraction(11, 2)
        )
        self.assertEqual(
            manifest.acceptance_score(CapabilityStage.S1), (1, 3)
        )
        before_claims = {item.key: item for item in stage3_manifest.claims}
        claims = {item.key: item for item in manifest.claims}
        changed = {
            key
            for key, claim in claims.items()
            if claim.status is not before_claims[key].status
        }
        self.assertEqual(changed, {"s1.kv_handoff", "s1.pd"})
        for key in changed:
            self.assertIs(claims[key].status, CapabilityStatus.E2E_TIMING)
            self.assertEqual(claims[key].score_numerator, 1)
        for key in (
            "s1.gemm_collective.optimized",
            "s1.naive_case.f_pdf",
            "s1.naive_case.f_pdr",
            "s1.naive_case.f_pds",
            "s1.validation_ladder",
        ):
            self.assertIs(claims[key].status, CapabilityStatus.UNSUPPORTED)
        evidence = {item.case_id: item for item in manifest.evidence}
        for case_id in ("case.s1.kv_handoff", "case.s1.pd"):
            self.assertIs(
                evidence[case_id].status, CapabilityStatus.E2E_TIMING
            )
            self.assertEqual(
                evidence[case_id].report_digest,
                canonical_digest(stage4_matrix),
            )
        manifest.validate_against(matrix)

    def test_empty_partial_and_tampered_matrices_do_not_promote(self) -> None:
        stage3_matrix, stage3_manifest = _stage3_truth()
        ready = _ready_matrix()
        for reports in ((), ready.reports[:1], ready.reports[:2]):
            with self.subTest(report_count=len(reports)):
                partial = build_stage4_pd_case_matrix(reports)
                matrix, manifest = build_stage4_capability_manifest(
                    stage3_matrix,
                    stage3_manifest,
                    partial,
                )
                claims = {item.key: item for item in manifest.claims}
                self.assertIs(
                    claims["s1.kv_handoff"].status,
                    CapabilityStatus.UNSUPPORTED,
                )
                self.assertIs(
                    claims["s1.pd"].status,
                    CapabilityStatus.UNSUPPORTED,
                )
                self.assertEqual(
                    manifest.coverage_score(CapabilityStage.S1),
                    Fraction(7, 2),
                )
                self.assertEqual(
                    manifest.acceptance_score(CapabilityStage.S1), (1, 3)
                )
                manifest.validate_against(matrix)
        with self.assertRaisesRegex(SchemaError, "Stage 4 ready"):
            build_stage4_capability_manifest(
                stage3_matrix,
                stage3_manifest,
                replace(ready, stage4_ready=False),
            )
        with self.assertRaisesRegex(SchemaError, "timing/state-transport"):
            build_stage4_capability_manifest(
                stage3_matrix,
                stage3_manifest,
                replace(
                    ready,
                    reports=(
                        replace(ready.reports[0], model_functional=True),
                        *ready.reports[1:],
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
