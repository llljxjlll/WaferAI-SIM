from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema import (
    SwizzleBenefitBranch as PublicSwizzleBenefitBranch,
    SwizzleScalePoint as PublicSwizzleScalePoint,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleAlgorithm
from llm.frontend.wafer_frontend.schema.swizzle_performance_evidence import (
    SwizzleBenefitBranch,
    SwizzleBranchTimingObservation,
)

from run_swizzle_scale_comparison import (
    build_scale_benefit_evidence,
    build_scale_official_matrix,
    run_python_scale_preflight,
)
from swizzle_scale_cases import build_first_green_swizzle_scale_cases
from swizzle_scale_comparison import build_swizzle_scale_comparison_suite
from swizzle_scale_runtime_provider import (
    ProductionSwizzleScaleProvider,
    build_actual_scale_program_io,
)


_MARKER = "a" * 64


def _observation(
    branch_plan,
    *,
    makespan: int,
    send_inflight: int,
    recv_inflight: int,
) -> SwizzleBranchTimingObservation:
    naive = branch_plan.branch is SwizzleBenefitBranch.NAIVE
    return SwizzleBranchTimingObservation(
        branch=branch_plan.branch,
        algorithm=branch_plan.algorithm,
        economic_auto_selected=branch_plan.economic_auto_selected,
        forced_deployment=branch_plan.forced_deployment,
        chunk_count=branch_plan.chunk_count,
        unroll_degree=branch_plan.unroll_degree,
        tile_shape=branch_plan.tile_shape,
        logical_bytes=256,
        byte_hops=512,
        gemm_flops=4096,
        record_count=8 if naive else 16,
        alloc_record_count=2,
        free_record_count=2,
        barrier_count=2 if naive else 4,
        event_record_count=2 if naive else 4,
        control_action_count=2 if naive else 4,
        sram_high_water_bytes=0 if naive else 1024,
        observed_max_inflight_send=send_inflight,
        observed_max_inflight_recv=recv_inflight,
        repeat_makespans=(makespan, makespan),
        repeat_marker_digests=(_MARKER, _MARKER),
    )


class SwizzleScaleComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_first_green_swizzle_scale_cases()
        cls.suite = build_swizzle_scale_comparison_suite(cls.cases)

    def test_plan_is_canonical_same_work_and_keeps_forced_diagnostic(self) -> None:
        self.assertEqual(len(self.suite.cases), 8)
        expected_auto_chunks = {
            ("S1", FusionPattern.AG_GEMM): 8,
            ("S2", FusionPattern.AG_GEMM): 16,
            ("S2", FusionPattern.GEMM_RS): 8,
            ("S3", FusionPattern.AG_GEMM): 32,
            ("S3", FusionPattern.GEMM_RS): 32,
        }
        for case_plan in self.suite.cases:
            with self.subTest(
                scale=case_plan.scale_name, pattern=case_plan.pattern.value
            ):
                self.assertIs(case_plan.branches[0].branch, SwizzleBenefitBranch.NAIVE)
                self.assertEqual(
                    {item.same_work_digest for item in case_plan.branches},
                    {case_plan.same_work_digest},
                )
                has_economic_auto = not (
                    case_plan.scale_name == "S0"
                    or (
                        case_plan.scale_name == "S1"
                        and case_plan.pattern is FusionPattern.GEMM_RS
                    )
                )
                expected_branches = (
                    (
                        SwizzleBenefitBranch.NAIVE,
                        SwizzleBenefitBranch.SWIZZLE_AUTO,
                        SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC,
                    )
                    if has_economic_auto
                    else
                    (
                        SwizzleBenefitBranch.NAIVE,
                        SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC,
                    )
                )
                self.assertEqual(
                    tuple(item.branch for item in case_plan.branches),
                    expected_branches,
                )
                self.assertEqual(
                    case_plan.forced_diagnostic.forced_deployment,
                    True,
                )
                if has_economic_auto:
                    self.assertTrue(case_plan.swizzle_branch.economic_auto_selected)
                    self.assertEqual(
                        case_plan.swizzle_branch.chunk_count,
                        expected_auto_chunks[
                            (case_plan.scale_name, case_plan.pattern)
                        ],
                    )
                    self.assertTrue(
                        all(
                            actual >= required
                            for actual, required in zip(
                                case_plan.swizzle_branch.tile_shape,
                                (4, 16, 16),
                            )
                        )
                    )
                    self.assertNotEqual(
                        case_plan.swizzle_branch.candidate_ref,
                        case_plan.forced_diagnostic.candidate_ref,
                    )

    def test_all_python_branches_link_and_build_zero_sha_program_io(self) -> None:
        result = run_python_scale_preflight()
        self.assertEqual(result.suite, self.suite)
        self.assertEqual(len(result.prepared), 21)
        for item in result.prepared:
            with self.subTest(
                scale=item.case_plan.scale_name,
                pattern=item.case_plan.pattern.value,
                branch=item.branch_plan.branch.value,
            ):
                expected_streams = 2 if item.case_plan.scale_name == "S0" else 4
                self.assertEqual(item.stream_count, expected_streams)
                self.assertEqual(item.program_io.program_artifact_sha256, "0" * 64)
                self.assertTrue(item.program_io.initializations)
                self.assertTrue(item.program_io.output_probes)
                self.assertTrue(item.program_io.blobs)

    def test_forced_cannot_masquerade_and_actual_inflight_is_lossless(self) -> None:
        s0 = self.suite.cases[0]
        with self.assertRaisesRegex(SchemaError, "forced diagnostic"):
            build_scale_benefit_evidence(
                s0,
                naive=_observation(
                    s0.branches[0], makespan=1200, send_inflight=1, recv_inflight=2
                ),
                swizzle_auto=_observation(
                    s0.forced_diagnostic,
                    makespan=900,
                    send_inflight=7,
                    recv_inflight=8,
                ),
            )

        s1 = self.suite.cases[2]
        naive = _observation(
            s1.branches[0], makespan=1200, send_inflight=3, recv_inflight=4
        )
        auto = _observation(
            s1.swizzle_branch, makespan=1000, send_inflight=7, recv_inflight=8
        )
        evidence = build_scale_benefit_evidence(
            s1, naive=naive, swizzle_auto=auto
        )
        self.assertEqual(
            (
                evidence.naive.observed_max_inflight_send,
                evidence.naive.observed_max_inflight_recv,
                evidence.swizzle_auto.observed_max_inflight_send,
                evidence.swizzle_auto.observed_max_inflight_recv,
            ),
            (3, 4, 7, 8),
        )
        forged = replace(
            s1.swizzle_branch,
            branch=SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC,
        )
        with self.assertRaisesRegex(SchemaError, "diagnostic"):
            forged.validate()

    def test_actual_sha_program_io_rebuilds_from_exact_prepared_source(self) -> None:
        case_plan = next(
            item
            for item in self.suite.cases
            if item.scale_name == "S0" and item.pattern is FusionPattern.AG_GEMM
        )
        prepared = ProductionSwizzleScaleProvider(
            cases=self.cases, suite=self.suite
        ).prepare(case_plan, SwizzleBenefitBranch.NAIVE)
        actual = build_actual_scale_program_io(prepared, _MARKER)
        self.assertEqual(actual.program_artifact_sha256, _MARKER)
        self.assertEqual(
            actual.source_linked_manifest_id, prepared.source.manifest.id
        )
        with self.assertRaisesRegex(SchemaError, "non-zero lowercase SHA-256"):
            build_actual_scale_program_io(prepared, "0" * 64)
        with self.assertRaisesRegex(SchemaError, "non-zero lowercase SHA-256"):
            build_actual_scale_program_io(prepared, "A" * 64)
        other_case = next(
            item
            for item in self.suite.cases
            if item.scale_name == "S0" and item.pattern is FusionPattern.GEMM_RS
        )
        forged = replace(
            prepared,
            case_plan=other_case,
            branch_plan=other_case.branches[0],
        )
        with self.assertRaisesRegex(SchemaError, "source provenance"):
            forged.validate()

    def test_official_matrix_uses_only_genuine_same_work_auto_pairs(self) -> None:
        matrix = build_scale_official_matrix(self.suite)
        self.assertEqual(
            tuple(
                (
                    target.case_plan.scale_name,
                    target.case_plan.pattern,
                    target.branch_plan.branch,
                )
                for target in matrix.official
            ),
            (
                ("S1", FusionPattern.AG_GEMM, SwizzleBenefitBranch.NAIVE),
                ("S1", FusionPattern.AG_GEMM, SwizzleBenefitBranch.SWIZZLE_AUTO),
                ("S2", FusionPattern.AG_GEMM, SwizzleBenefitBranch.NAIVE),
                ("S2", FusionPattern.AG_GEMM, SwizzleBenefitBranch.SWIZZLE_AUTO),
                ("S2", FusionPattern.GEMM_RS, SwizzleBenefitBranch.NAIVE),
                ("S2", FusionPattern.GEMM_RS, SwizzleBenefitBranch.SWIZZLE_AUTO),
            ),
        )
        self.assertEqual(
            tuple(
                (target.case_plan.scale_name, target.case_plan.pattern)
                for target in matrix.forced_preflight
            ),
            (
                ("S1", FusionPattern.AG_GEMM),
                ("S1", FusionPattern.GEMM_RS),
                ("S2", FusionPattern.AG_GEMM),
                ("S2", FusionPattern.GEMM_RS),
            ),
        )
        forged = replace(
            matrix,
            official=(matrix.forced_preflight[0],) + matrix.official[1:],
        )
        with self.assertRaisesRegex(SchemaError, "forced diagnostic"):
            forged.validate()

    def test_schema_package_exports_are_identity_exact(self) -> None:
        from llm.frontend.wafer_frontend.schema.swizzle_performance_evidence import (
            SwizzleBenefitBranch as DirectSwizzleBenefitBranch,
        )
        from llm.frontend.wafer_frontend.schema.swizzle_scale import (
            SwizzleScalePoint as DirectSwizzleScalePoint,
        )

        self.assertIs(PublicSwizzleBenefitBranch, DirectSwizzleBenefitBranch)
        self.assertIs(PublicSwizzleScalePoint, DirectSwizzleScalePoint)
        self.assertIsNot(SwizzleAlgorithm.UNFUSED, SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL)


if __name__ == "__main__":
    unittest.main()
