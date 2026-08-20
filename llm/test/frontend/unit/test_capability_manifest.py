from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_capability_manifest,
    build_s1_s3_case_matrix,
    build_stage1a_capability_manifest,
    build_stage1a_case_matrix,
    build_stage2_capability_manifest,
    build_stage2_case_matrix,
    build_stage0_capability_manifest,
)
from llm.frontend.wafer_frontend.schema import (
    CAPABILITY_MANIFEST_SCHEMA_VERSION,
    AcceptanceDefinition,
    CapabilityDefinition,
    CapabilityManifest,
    CapabilityStage,
    CapabilityStatus,
    CaseDefinition,
    CaseEvidence,
    CaseMatrix,
)
from llm.frontend.wafer_frontend.runner import NaiveRunReport
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.stage1a_evidence import (
    STAGE1A_BASELINE_EPOCH,
    Stage1aCase,
)
from llm.frontend.wafer_frontend.schema.stage2_dense_forward_evidence import (
    STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
    Stage2DenseForwardRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    load_json_dataclass,
    load_json_value,
    loads_dataclass,
)
from test_stage1a_evidence import _oracle, _replace_report, _report
from test_stage2_dense_forward_evidence import _report as _stage2_report


def _capability(
    key: str,
    case_id: str,
    *,
    numerator: int = 1,
    denominator: int = 1,
) -> CapabilityDefinition:
    return CapabilityDefinition(
        key=key,
        stage=CapabilityStage.S1,
        required_case_ids=(case_id,),
        score_numerator=numerator,
        score_denominator=denominator,
        notes=(f"definition for {key}",),
    )


def _case(case_id: str, capability_id: str) -> CaseDefinition:
    return CaseDefinition(
        id=case_id,
        stage=CapabilityStage.S1,
        maximum_status=CapabilityStatus.E2E_TIMING,
        capability_ids=(capability_id,),
        input_refs=(f"notes/frontend/examples/{case_id}.json",),
        description=f"evidence case for {capability_id}",
    )


def _s1_review_matrix() -> CaseMatrix:
    capabilities = (
        _capability(
            "s1.gemm_collective.naive",
            "case.s1.naive_e2e",
            numerator=1,
            denominator=2,
        ),
        _capability(
            "s1.gemm_collective.optimized",
            "case.s1.optimized_ablation",
            numerator=1,
            denominator=2,
        ),
        _capability("s1.kv", "case.s1.kv_decode"),
        _capability("s1.mesh", "case.s1.mesh_2x2"),
        _capability("s1.pd", "case.s1.pd"),
        _capability("s1.single_die", "case.s1.single_die"),
        _capability("s1.tp_sp", "case.s1.tp_sp"),
    )
    acceptance = (
        AcceptanceDefinition(
            key="s1.acceptance.ablation",
            stage=CapabilityStage.S1,
            required_case_ids=(),
            blocking_capability_ids=("s1.gemm_collective.optimized",),
            notes=("requires a real optimized implementation",),
        ),
        AcceptanceDefinition(
            key="s1.acceptance.e2e",
            stage=CapabilityStage.S1,
            required_case_ids=(),
            blocking_capability_ids=("s1.gemm_collective.naive",),
            notes=("requires a naive end-to-end timing run",),
        ),
        AcceptanceDefinition(
            key="s1.acceptance.validation_ladder",
            stage=CapabilityStage.S1,
            required_case_ids=(),
            blocking_capability_ids=("s1.kv",),
            notes=("requires the missing dynamic-state validation rung",),
        ),
    )
    cases = tuple(
        sorted(
            (
                _case("case.s1.kv_decode", "s1.kv"),
                _case("case.s1.mesh_2x2", "s1.mesh"),
                _case("case.s1.naive_e2e", "s1.gemm_collective.naive"),
                _case(
                    "case.s1.optimized_ablation",
                    "s1.gemm_collective.optimized",
                ),
                _case("case.s1.pd", "s1.pd"),
                _case("case.s1.single_die", "s1.single_die"),
                _case("case.s1.tp_sp", "s1.tp_sp"),
            ),
            key=lambda item: item.id,
        )
    )
    return CaseMatrix.create(
        capabilities=capabilities,
        acceptance=acceptance,
        cases=cases,
    )


def _evidence(case_id: str) -> CaseEvidence:
    return CaseEvidence(
        case_id=case_id,
        status=CapabilityStatus.E2E_TIMING,
        evidence_digest=canonical_digest({"evidence": case_id}),
        artifact_digest=canonical_digest({"artifact": case_id}),
        report_digest=canonical_digest({"report": case_id}),
    )


def _current_review_evidence() -> tuple[CaseEvidence, ...]:
    return tuple(
        _evidence(case_id)
        for case_id in (
            "case.s1.mesh_2x2",
            "case.s1.naive_e2e",
            "case.s1.single_die",
            "case.s1.tp_sp",
        )
    )


def _restable_manifest(manifest: CapabilityManifest) -> CapabilityManifest:
    return replace(
        manifest,
        id=stable_artifact_id(
            "capability_manifest",
            manifest._semantic_key(),
            schema_version=CAPABILITY_MANIFEST_SCHEMA_VERSION,
        ),
    )


def _stage0_truth() -> tuple[CaseMatrix, CapabilityManifest]:
    return build_stage0_capability_manifest(
        e1_artifact_digest="1" * 64,
        e1_report_digest="2" * 64,
        e2_artifact_digest="3" * 64,
        e2_report_digest="4" * 64,
        single_die_evidence_digest="5" * 64,
    )


def _stage1a_truth() -> tuple[CaseMatrix, CapabilityManifest]:
    return build_stage1a_capability_manifest(*_stage0_truth())


class CapabilityManifestTest(unittest.TestCase):
    def test_checked_stage0_files_rebuild_from_persistent_evidence(self) -> None:
        root = Path(
            "notes/frontend/baselines/stage0-policy-provenance-v1"
        )
        checked_matrix = load_json_dataclass(
            CaseMatrix,
            root / "case_matrix.json",
            path="case_matrix",
        )
        checked_manifest = load_json_dataclass(
            CapabilityManifest,
            root / "capability_manifest.json",
            path="capability_manifest",
        )
        reports = {
            case: load_json_dataclass(
                NaiveRunReport,
                root / case / "run_report.json",
                path=f"{case}.report",
            )
            for case in ("e1", "e2")
        }
        matrix, manifest = build_stage0_capability_manifest(
            e1_artifact_digest=reports["e1"].artifact["artifact_sha256"],
            e1_report_digest=canonical_digest(reports["e1"]),
            e2_artifact_digest=reports["e2"].artifact["artifact_sha256"],
            e2_report_digest=canonical_digest(reports["e2"]),
            single_die_evidence_digest=canonical_digest(
                {
                    "tests": (
                        "test_dense_ir0_validator",
                        "test_global_action_schema",
                        "test_ir2_schema",
                    )
                }
            ),
        )
        self.assertEqual(checked_matrix, matrix)
        self.assertEqual(checked_manifest, manifest)
        checked_manifest.validate_against(checked_matrix)
        for case in checked_matrix.cases:
            for input_ref in case.input_refs:
                self.assertTrue(Path(input_ref).is_file(), input_ref)

        review = load_json_value(
            root / "baseline_review.json",
            path="baseline_review",
        )
        self.assertIsInstance(review, dict)
        assert isinstance(review, dict)
        self.assertEqual(
            set(review),
            {
                "baseline_epoch",
                "cases",
                "caveats",
                "commands",
                "id",
                "producer_pass",
                "reason",
                "review_decision",
                "review_kind",
                "reviewer",
                "schema_version",
                "version_changes",
            },
        )
        review_key = {
            key: value
            for key, value in review.items()
            if key not in ("schema_version", "producer_pass", "id")
        }
        self.assertEqual(
            review["id"],
            stable_artifact_id(
                "baseline_review",
                review_key,
                schema_version=review["schema_version"],
            ),
        )
        self.assertEqual(
            tuple(
                (
                    row["case"],
                    row["prior_artifact_sha256"]
                    == row["artifact_sha256"],
                    row["semantic_differences"],
                    row["makespan_cycles"],
                )
                for row in review["cases"]
            ),
            (("E1", True, [], 1728), ("E2", True, [], 5333)),
        )

    def test_checked_development_matrix_covers_s1_s2_s3_without_overclaim(self) -> None:
        matrix = build_s1_s3_case_matrix()
        expected_plan_cases = {
            *(f"case.s1.{name}" for name in (
                "f_d1", "f_m1", "f_p1", "f_p2",
                "f_pdf", "f_pdr", "f_pds", "f_tf",
            )),
            *(f"case.s2.t{index}" for index in range(6)),
            *(f"case.s3.m{index}" for index in range(5)),
        }
        self.assertTrue(
            expected_plan_cases.issubset({case.id for case in matrix.cases})
        )
        evidence = (
            _evidence("case.s1.baseline.e1_tp2"),
            _evidence("case.s1.baseline.e2_tp4"),
            CaseEvidence(
                case_id="case.s1.baseline.single_die_unit",
                status=CapabilityStatus.UNIT_ONLY,
                evidence_digest=canonical_digest(
                    {"tests": ("test_dense_ir0_validator",)}
                ),
                artifact_digest=None,
                report_digest=None,
            ),
        )
        manifest = build_capability_manifest(
            matrix,
            evidence,
            baseline_epoch="stage0-policy-provenance-v1",
        )
        manifest.validate_against(matrix)
        self.assertEqual(
            manifest.coverage_score(CapabilityStage.S1),
            Fraction(7, 2),
        )
        self.assertEqual(
            manifest.acceptance_score(CapabilityStage.S1),
            (1, 3),
        )
        claims = {claim.key: claim for claim in manifest.claims}
        self.assertIs(
            claims["s1.gemm_collective.naive"].status,
            CapabilityStatus.E2E_TIMING,
        )
        self.assertIs(
            claims["s1.single_die"].status,
            CapabilityStatus.UNIT_ONLY,
        )
        self.assertIs(
            claims["s1.gemm_collective.optimized"].status,
            CapabilityStatus.UNSUPPORTED,
        )
        self.assertTrue(
            all(
                claim.status is CapabilityStatus.UNSUPPORTED
                for claim in manifest.claims
                if claim.stage in (CapabilityStage.S2, CapabilityStage.S3)
            )
        )

    def test_review_score_and_acceptance_are_exact_and_explicit(self) -> None:
        matrix = _s1_review_matrix()
        manifest = build_capability_manifest(
            matrix,
            _current_review_evidence(),
            baseline_epoch="pre-stage0-review",
        )
        manifest.validate_against(matrix)
        self.assertEqual(manifest.coverage_score(CapabilityStage.S1), Fraction(7, 2))
        self.assertEqual(
            sum(
                (
                    Fraction(
                        definition.score_numerator,
                        definition.score_denominator,
                    )
                    for definition in matrix.capabilities
                    if definition.stage is CapabilityStage.S1
                ),
                start=Fraction(0, 1),
            ),
            Fraction(6, 1),
        )
        self.assertEqual(manifest.acceptance_score(CapabilityStage.S1), (1, 3))
        claims = {claim.key: claim for claim in manifest.claims}
        self.assertEqual(
            claims["s1.gemm_collective.naive"].score_numerator, 1
        )
        self.assertEqual(
            claims["s1.gemm_collective.naive"].score_denominator, 2
        )
        self.assertIs(
            claims["s1.gemm_collective.optimized"].status,
            CapabilityStatus.UNSUPPORTED,
        )
        self.assertEqual(
            claims["s1.gemm_collective.optimized"].evidence_case_ids, ()
        )

    def test_strict_round_trip_stable_id_and_wire_spelling(self) -> None:
        matrix = _s1_review_matrix()
        manifest = build_capability_manifest(
            matrix,
            _current_review_evidence(),
            baseline_epoch="epoch-a",
        )
        encoded = canonical_json(manifest)
        self.assertIn('"e2e-timing"', encoded)
        self.assertNotIn("e2e_timing", encoded)
        decoded = loads_dataclass(CapabilityManifest, encoded)
        self.assertEqual(decoded, manifest)
        self.assertEqual(canonical_digest(decoded), canonical_digest(manifest))
        decoded.validate_against(matrix)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            manifest.baseline_epoch = "forged"  # type: ignore[misc]

    def test_builder_rejects_dangling_reordered_and_overclaimed_evidence(self) -> None:
        matrix = _s1_review_matrix()
        evidence = _current_review_evidence()
        with self.assertRaisesRegex(SchemaError, "strictly increasing"):
            build_capability_manifest(
                matrix,
                tuple(reversed(evidence)),
                baseline_epoch="epoch-a",
            )
        with self.assertRaisesRegex(SchemaError, "unknown case"):
            build_capability_manifest(
                matrix,
                (_evidence("case.s1.unknown"),),
                baseline_epoch="epoch-a",
            )
        functional = replace(
            evidence[0], status=CapabilityStatus.E2E_FUNCTIONAL
        )
        with self.assertRaisesRegex(SchemaError, "maximum proof level"):
            build_capability_manifest(
                matrix,
                (functional, *evidence[1:]),
                baseline_epoch="epoch-a",
            )

    def test_e2e_evidence_requires_artifact_and_report_digests(self) -> None:
        evidence = replace(
            _evidence("case.s1.naive_e2e"),
            artifact_digest=None,
            report_digest=None,
        )
        with self.assertRaisesRegex(SchemaError, "artifact and report"):
            evidence.validate()

    def test_matrix_rejects_reordering_and_nonreciprocal_references(self) -> None:
        matrix = _s1_review_matrix()
        with self.assertRaisesRegex(SchemaError, "strictly increasing"):
            CaseMatrix.create(
                capabilities=tuple(reversed(matrix.capabilities)),
                acceptance=matrix.acceptance,
                cases=matrix.cases,
            )
        changed_case = replace(
            matrix.cases[0], capability_ids=("s1.mesh",)
        )
        with self.assertRaisesRegex(SchemaError, "reciprocally"):
            CaseMatrix.create(
                capabilities=matrix.capabilities,
                acceptance=matrix.acceptance,
                cases=(changed_case, *matrix.cases[1:]),
            )

    def test_restable_claim_tamper_is_rejected_against_matrix(self) -> None:
        matrix = _s1_review_matrix()
        manifest = build_capability_manifest(
            matrix,
            _current_review_evidence(),
            baseline_epoch="epoch-a",
        )
        claim_index = next(
            index
            for index, claim in enumerate(manifest.claims)
            if claim.key == "s1.mesh"
        )
        claims = list(manifest.claims)
        claims[claim_index] = replace(claims[claim_index], score_numerator=0)
        forged = _restable_manifest(replace(manifest, claims=tuple(claims)))
        forged.validate()
        with self.assertRaisesRegex(SchemaError, "exact result"):
            forged.validate_against(matrix)

    def test_case_matrix_identity_is_exact_provenance(self) -> None:
        matrix = _s1_review_matrix()
        manifest = build_capability_manifest(
            matrix,
            _current_review_evidence(),
            baseline_epoch="epoch-a",
        )
        forged = _restable_manifest(
            replace(manifest, case_matrix_id="case_matrix_forged")
        )
        forged.validate()
        with self.assertRaisesRegex(SchemaError, "supplied case matrix"):
            forged.validate_against(matrix)

    def test_stage1a_matrix_is_zero_score_immutable_extension(self) -> None:
        stage0_matrix, stage0_manifest = _stage0_truth()
        matrix_digest = canonical_digest(stage0_matrix)
        manifest_digest = canonical_digest(stage0_manifest)
        extended = build_stage1a_case_matrix(stage0_matrix)
        self.assertEqual(build_stage1a_case_matrix(stage0_matrix), extended)
        self.assertEqual(canonical_digest(stage0_matrix), matrix_digest)
        self.assertEqual(canonical_digest(stage0_manifest), manifest_digest)
        self.assertEqual(extended.acceptance, stage0_matrix.acceptance)

        expected_capabilities = {
            "s1.foundation.parameter_dma_compute",
            "s1.foundation.kv_cross_action",
            "s1.foundation.synthetic_pd_state",
            "s1.foundation.state_fail_closed",
        }
        definitions = {
            item.key: item
            for item in extended.capabilities
            if item.key.startswith("s1.foundation.")
        }
        self.assertEqual(set(definitions), expected_capabilities)
        self.assertTrue(
            all(
                (item.score_numerator, item.score_denominator) == (0, 1)
                for item in definitions.values()
            )
        )
        cases = {
            item.id: item
            for item in extended.cases
            if item.id.startswith("case.s1.foundation.")
        }
        self.assertEqual(len(cases), 4)
        self.assertIs(
            cases["case.s1.foundation.state_fail_closed"].maximum_status,
            CapabilityStatus.UNIT_ONLY,
        )
        self.assertEqual(
            cases["case.s1.foundation.state_fail_closed"].input_refs,
            (
                "llm/test/frontend/integration/run_pd1_finalizer.py",
                "llm/test/frontend/integration/run_program_io_e1_t.py",
                "llm/test/frontend/integration/run_stage1a_negative_evidence.py",
                "llm/test/frontend/unit/test_persistent_state_schema.py",
                "llm/test/frontend/unit/test_program_io_hbm_validate.py",
                "llm/test/frontend/unit/test_stage1a_evidence.py",
                "llm/test/frontend/unit/test_state_dma_lowering.py",
                "llm/test/frontend/unit/test_state_transfer_lowering.py",
                "notes/frontend/S1-S3后续开发方案.md",
            ),
        )
        self.assertTrue(
            all(
                item.maximum_status is CapabilityStatus.E2E_TIMING
                for case_id, item in cases.items()
                if case_id != "case.s1.foundation.state_fail_closed"
            )
        )
        with self.assertRaisesRegex(SchemaError, "collide"):
            build_stage1a_case_matrix(extended)

    def test_stage1a_typed_evidence_is_deterministic_without_overclaim(self) -> None:
        stage0_matrix, stage0_manifest = _stage0_truth()
        before = (
            canonical_digest(stage0_matrix),
            canonical_digest(stage0_manifest),
        )
        empty_matrix, empty_manifest = build_stage1a_capability_manifest(
            stage0_matrix,
            stage0_manifest,
        )
        empty_claims = {claim.key: claim for claim in empty_manifest.claims}
        self.assertTrue(
            all(
                empty_claims[key].status is CapabilityStatus.UNSUPPORTED
                for key in (
                    "s1.foundation.parameter_dma_compute",
                    "s1.foundation.kv_cross_action",
                    "s1.foundation.synthetic_pd_state",
                    "s1.foundation.state_fail_closed",
                )
            )
        )
        empty_manifest.validate_against(empty_matrix)
        oracles = {case: _oracle(case) for case in Stage1aCase}
        kwargs = {
            "p1_oracle": oracles[Stage1aCase.P1],
            "p1_report": _report(oracles[Stage1aCase.P1]),
            "k1_oracle": oracles[Stage1aCase.K1],
            "k1_report": _report(oracles[Stage1aCase.K1]),
            "pd1_oracle": oracles[Stage1aCase.PD1],
            "pd1_report": _report(oracles[Stage1aCase.PD1]),
            "state_fail_closed_evidence_digest": "6" * 64,
        }
        matrix, manifest = build_stage1a_capability_manifest(
            stage0_matrix,
            stage0_manifest,
            **kwargs,
        )
        self.assertEqual(
            build_stage1a_capability_manifest(
                stage0_matrix,
                stage0_manifest,
                **kwargs,
            ),
            (matrix, manifest),
        )
        self.assertEqual(
            (
                canonical_digest(stage0_matrix),
                canonical_digest(stage0_manifest),
            ),
            before,
        )
        self.assertEqual(manifest.baseline_epoch, STAGE1A_BASELINE_EPOCH)
        self.assertEqual(
            manifest.coverage_score(CapabilityStage.S1), Fraction(7, 2)
        )
        self.assertEqual(
            manifest.acceptance_score(CapabilityStage.S1), (1, 3)
        )
        claims = {claim.key: claim for claim in manifest.claims}
        for key in (
            "s1.foundation.parameter_dma_compute",
            "s1.foundation.kv_cross_action",
            "s1.foundation.synthetic_pd_state",
        ):
            self.assertIs(claims[key].status, CapabilityStatus.E2E_TIMING)
            self.assertEqual(claims[key].score_numerator, 0)
        self.assertIs(
            claims["s1.foundation.state_fail_closed"].status,
            CapabilityStatus.UNIT_ONLY,
        )
        for key in (
            "s1.kv_handoff",
            "s1.pd",
            "s1.naive_case.f_d1",
            "s1.naive_case.f_pds",
        ):
            self.assertIs(claims[key].status, CapabilityStatus.UNSUPPORTED)
            self.assertEqual(claims[key].evidence_case_ids, ())
        evidence = {item.case_id: item for item in manifest.evidence}
        for case_id, argument in (
            ("case.s1.foundation.parameter_dma_compute", "p1_report"),
            ("case.s1.foundation.kv_cross_action", "k1_report"),
            ("case.s1.foundation.synthetic_pd_state", "pd1_report"),
        ):
            item = evidence[case_id]
            report = kwargs[argument]
            self.assertEqual(
                item.artifact_digest,
                report.artifact.program_artifact_sha256,
            )
            self.assertEqual(item.report_digest, canonical_digest(report))
        manifest.validate_against(matrix)

    def test_stage1a_rejects_orphan_overclaim_and_tamper(self) -> None:
        stage0_matrix, stage0_manifest = _stage0_truth()
        p1_oracle = _oracle(Stage1aCase.P1)
        p1_report = _report(p1_oracle)
        with self.assertRaisesRegex(SchemaError, "supplied together"):
            build_stage1a_capability_manifest(
                stage0_matrix,
                stage0_manifest,
                p1_report=p1_report,
            )
        with self.assertRaisesRegex(SchemaError, "P1 evidence"):
            k1_oracle = _oracle(Stage1aCase.K1)
            build_stage1a_capability_manifest(
                stage0_matrix,
                stage0_manifest,
                p1_oracle=p1_oracle,
                p1_report=_report(k1_oracle),
            )
        with self.assertRaisesRegex(SchemaError, "timing evidence only"):
            build_stage1a_capability_manifest(
                stage0_matrix,
                stage0_manifest,
                p1_oracle=p1_oracle,
                p1_report=replace(
                    p1_report,
                    capability_status=CapabilityStatus.E2E_FUNCTIONAL,
                ),
            )
        with self.assertRaisesRegex(SchemaError, "supplied oracle"):
            build_stage1a_capability_manifest(
                stage0_matrix,
                stage0_manifest,
                p1_oracle=p1_oracle,
                p1_report=_replace_report(
                    p1_report,
                    oracle_digest="f" * 64,
                ),
            )
        with self.assertRaisesRegex(SchemaError, "lowercase SHA-256"):
            build_stage1a_capability_manifest(
                stage0_matrix,
                stage0_manifest,
                state_fail_closed_evidence_digest="BAD",
            )

    def test_stage2_matrix_is_zero_score_immutable_exact_extension(self) -> None:
        stage1a_matrix, stage1a_manifest = _stage1a_truth()
        before = (
            canonical_digest(stage1a_matrix),
            canonical_digest(stage1a_manifest),
        )
        matrix = build_stage2_case_matrix(stage1a_matrix)
        self.assertEqual(build_stage2_case_matrix(stage1a_matrix), matrix)
        self.assertEqual(
            (
                canonical_digest(stage1a_matrix),
                canonical_digest(stage1a_manifest),
            ),
            before,
        )
        self.assertEqual(matrix.acceptance, stage1a_matrix.acceptance)

        definitions = {
            item.key: item
            for item in matrix.capabilities
            if item.key.startswith("s1.foundation.dense_forward")
        }
        self.assertEqual(
            set(definitions),
            {
                "s1.foundation.dense_forward_closure",
                "s1.foundation.dense_forward_fail_closed",
            },
        )
        self.assertTrue(
            all(
                (item.score_numerator, item.score_denominator) == (0, 1)
                for item in definitions.values()
            )
        )
        self.assertEqual(
            definitions[
                "s1.foundation.dense_forward_closure"
            ].required_case_ids,
            tuple(
                sorted(
                    f"case.s1.foundation.dense_forward_tp{tp}"
                    for tp in (1, 2, 4)
                )
            ),
        )
        cases = {
            item.id: item
            for item in matrix.cases
            if item.id.startswith("case.s1.foundation.dense_forward")
        }
        self.assertEqual(len(cases), 4)
        runtime_refs = tuple(
            sorted(
                (
                    "llm/test/frontend/integration/run_stage2_dense_forward.py",
                    "llm/test/frontend/integration/stage2_dense_forward_cases.py",
                    "notes/frontend/S1-S3后续开发方案.md",
                    "notes/frontend/S2开发报告.md",
                )
            )
        )
        for tp in (1, 2, 4):
            case = cases[f"case.s1.foundation.dense_forward_tp{tp}"]
            self.assertIs(case.maximum_status, CapabilityStatus.E2E_TIMING)
            self.assertEqual(case.input_refs, runtime_refs)
        negative = cases[
            "case.s1.foundation.dense_forward_fail_closed"
        ]
        self.assertIs(negative.maximum_status, CapabilityStatus.UNIT_ONLY)
        self.assertEqual(
            negative.input_refs,
            tuple(
                sorted(
                    (
                        "llm/test/frontend/unit/test_stage2_artifact_schema.py",
                        "llm/test/frontend/unit/test_stage2_coarse_lowering.py",
                        "llm/test/frontend/unit/test_stage2_dense_forward_evidence.py",
                        "llm/test/frontend/unit/test_stage2_dense_forward_graph.py",
                        "notes/frontend/S1-S3后续开发方案.md",
                        "notes/frontend/S2开发报告.md",
                    )
                )
            ),
        )
        for case in cases.values():
            for input_ref in case.input_refs:
                self.assertTrue(Path(input_ref).is_file(), input_ref)
        with self.assertRaisesRegex(SchemaError, "collide"):
            build_stage2_case_matrix(matrix)

    def test_stage2_typed_closure_is_all_or_nothing_and_deterministic(self) -> None:
        stage1a_matrix, stage1a_manifest = _stage1a_truth()
        before = (
            canonical_digest(stage1a_matrix),
            canonical_digest(stage1a_manifest),
        )
        empty_matrix, empty_manifest = build_stage2_capability_manifest(
            stage1a_matrix, stage1a_manifest
        )
        empty_claims = {claim.key: claim for claim in empty_manifest.claims}
        self.assertIs(
            empty_claims["s1.foundation.dense_forward_closure"].status,
            CapabilityStatus.UNSUPPORTED,
        )
        self.assertIs(
            empty_claims[
                "s1.foundation.dense_forward_fail_closed"
            ].status,
            CapabilityStatus.UNSUPPORTED,
        )

        reports_and_oracles = {
            tp: _stage2_report(tp) for tp in (1, 2, 4)
        }
        tp1_report, tp1_oracle = reports_and_oracles[1]
        partial_matrix, partial_manifest = build_stage2_capability_manifest(
            stage1a_matrix,
            stage1a_manifest,
            tp1_oracle=tp1_oracle,
            tp1_report=tp1_report,
        )
        partial_claims = {
            claim.key: claim for claim in partial_manifest.claims
        }
        self.assertIs(
            partial_claims["s1.foundation.dense_forward_closure"].status,
            CapabilityStatus.UNSUPPORTED,
        )
        self.assertEqual(
            partial_claims[
                "s1.foundation.dense_forward_closure"
            ].evidence_case_ids,
            (),
        )
        partial_manifest.validate_against(partial_matrix)

        kwargs = {
            f"tp{tp}_{kind}": value
            for tp, (report, oracle) in reports_and_oracles.items()
            for kind, value in (("oracle", oracle), ("report", report))
        }
        kwargs["dense_forward_fail_closed_evidence_digest"] = "a" * 64
        matrix, manifest = build_stage2_capability_manifest(
            stage1a_matrix,
            stage1a_manifest,
            **kwargs,
        )
        self.assertEqual(
            build_stage2_capability_manifest(
                stage1a_matrix,
                stage1a_manifest,
                **kwargs,
            ),
            (matrix, manifest),
        )
        self.assertEqual(
            (
                canonical_digest(stage1a_matrix),
                canonical_digest(stage1a_manifest),
            ),
            before,
        )
        self.assertEqual(
            manifest.baseline_epoch, STAGE2_DENSE_FORWARD_BASELINE_EPOCH
        )
        claims = {claim.key: claim for claim in manifest.claims}
        closure = claims["s1.foundation.dense_forward_closure"]
        self.assertIs(closure.status, CapabilityStatus.E2E_TIMING)
        self.assertEqual(closure.score_numerator, 0)
        self.assertEqual(
            closure.evidence_case_ids,
            tuple(
                sorted(
                    f"case.s1.foundation.dense_forward_tp{tp}"
                    for tp in (1, 2, 4)
                )
            ),
        )
        self.assertIs(
            claims["s1.foundation.dense_forward_fail_closed"].status,
            CapabilityStatus.UNIT_ONLY,
        )
        self.assertEqual(manifest.coverage_score(CapabilityStage.S1), Fraction(7, 2))
        self.assertEqual(manifest.acceptance_score(CapabilityStage.S1), (1, 3))
        self.assertEqual(manifest.coverage_score(CapabilityStage.S2), Fraction(0, 1))
        self.assertTrue(
            all(
                claim.status is CapabilityStatus.UNSUPPORTED
                for claim in manifest.claims
                if claim.key.startswith("s2.naive_case.")
            )
        )
        evidence = {item.case_id: item for item in manifest.evidence}
        for tp, (report, _oracle_value) in reports_and_oracles.items():
            item = evidence[f"case.s1.foundation.dense_forward_tp{tp}"]
            self.assertEqual(
                item.artifact_digest,
                report.artifact.program_artifact_sha256,
            )
            self.assertEqual(item.report_digest, canonical_digest(report))
        manifest.validate_against(matrix)

    def test_stage2_rejects_orphan_wrong_tp_tamper_and_bad_digest(self) -> None:
        stage1a_matrix, stage1a_manifest = _stage1a_truth()
        tp1_report, tp1_oracle = _stage2_report(1)
        tp2_report, tp2_oracle = _stage2_report(2)
        with self.assertRaisesRegex(SchemaError, "supplied together"):
            build_stage2_capability_manifest(
                stage1a_matrix,
                stage1a_manifest,
                tp1_report=tp1_report,
            )
        with self.assertRaisesRegex(SchemaError, "TP1 evidence"):
            build_stage2_capability_manifest(
                stage1a_matrix,
                stage1a_manifest,
                tp1_oracle=tp2_oracle,
                tp1_report=tp2_report,
            )
        tampered_report = Stage2DenseForwardRuntimeReport.create(
            **(tp1_report._semantic_key() | {"oracle_digest": "f" * 64})
        )
        with self.assertRaisesRegex(SchemaError, "supplied oracle"):
            build_stage2_capability_manifest(
                stage1a_matrix,
                stage1a_manifest,
                tp1_oracle=tp1_oracle,
                tp1_report=tampered_report,
            )
        with self.assertRaisesRegex(SchemaError, "lowercase SHA-256"):
            build_stage2_capability_manifest(
                stage1a_matrix,
                stage1a_manifest,
                dense_forward_fail_closed_evidence_digest="BAD",
            )
        _stage0_matrix, orphan_manifest = _stage0_truth()
        with self.assertRaisesRegex(SchemaError, "supplied case matrix"):
            build_stage2_capability_manifest(
                stage1a_matrix,
                orphan_manifest,
            )


if __name__ == "__main__":
    unittest.main()
