from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleAlgorithm
from llm.frontend.wafer_frontend.schema.swizzle_performance_evidence import (
    SwizzleBenefitBranch,
    SwizzleBenefitReport,
    SwizzleBranchTimingObservation,
    SwizzleCalibratedEfficiencyPoint,
    SwizzleCalibrationProfile,
    SwizzleScaleBenefitEvidence,
)


_DIGEST = "1" * 64


def _observation(
    branch: SwizzleBenefitBranch,
    makespan: int,
) -> SwizzleBranchTimingObservation:
    naive = branch is SwizzleBenefitBranch.NAIVE
    return SwizzleBranchTimingObservation(
        branch=branch,
        algorithm=(SwizzleAlgorithm.UNFUSED if naive else SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL),
        economic_auto_selected=not naive,
        forced_deployment=branch is SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC,
        chunk_count=0 if naive else 8,
        unroll_degree=0 if naive else 2,
        tile_shape=None if naive else (32, 64, 64),
        logical_bytes=65536,
        byte_hops=131072,
        gemm_flops=67108864,
        record_count=32 if naive else 48,
        alloc_record_count=4,
        free_record_count=4,
        barrier_count=2 if naive else 8,
        event_record_count=2 if naive else 8,
        control_action_count=2 if naive else 4,
        sram_high_water_bytes=0 if naive else 131072,
        observed_max_inflight_send=1 if naive else 2,
        observed_max_inflight_recv=1 if naive else 2,
        repeat_makespans=(makespan, makespan),
        repeat_marker_digests=(_DIGEST, _DIGEST),
    )


class SwizzlePerformanceEvidenceTests(unittest.TestCase):
    def test_calibration_and_adjacent_speedup_are_typed(self) -> None:
        points = (
            SwizzleCalibratedEfficiencyPoint(32, 64, 64, DType.FP16, DType.FP32, 0.75, 0.05, 3.0),
            SwizzleCalibratedEfficiencyPoint(64, 64, 64, DType.FP16, DType.FP32, 0.90, 0.03, 3.0),
        )
        profile = SwizzleCalibrationProfile.create(
            hardware_profile_ref="hardware.profile.2x2",
            matching_tool_sha256="a" * 64,
            source_kind="npusim_microbenchmark",
            points=points,
        )
        evidence = tuple(
            SwizzleScaleBenefitEvidence.create(
                scale_ref=f"scale.S{ordinal}",
                scale_ordinal=ordinal,
                pattern=FusionPattern.AG_GEMM,
                same_work_digest="b" * 64,
                naive=_observation(SwizzleBenefitBranch.NAIVE, 1200 + ordinal * 100),
                swizzle_auto=_observation(SwizzleBenefitBranch.SWIZZLE_AUTO, 1000 + ordinal * 80),
            )
            for ordinal in (2, 3)
        )
        report = SwizzleBenefitReport.create(
            calibration_profile_ref=profile.id,
            evidence=evidence,
        )
        self.assertTrue(report.reproducible_speedup)
        self.assertTrue(all(item.qualifies for item in report.evidence))

    def test_forced_branch_cannot_masquerade_as_auto(self) -> None:
        forced = _observation(SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC, 900)
        forged = replace(
            forced,
            branch=SwizzleBenefitBranch.SWIZZLE_AUTO,
            economic_auto_selected=True,
        )
        with self.assertRaisesRegex(SchemaError, "AUTO"):
            forged.validate("forged")

    def test_repeat_marker_digest_pair_is_exact(self) -> None:
        observation = _observation(SwizzleBenefitBranch.NAIVE, 1200)
        for digests in ((), (_DIGEST,), (_DIGEST, _DIGEST, _DIGEST)):
            with self.subTest(length=len(digests)):
                with self.assertRaisesRegex(SchemaError, "exactly two"):
                    replace(observation, repeat_marker_digests=digests).validate()

    def test_work_drift_is_rejected(self) -> None:
        naive = _observation(SwizzleBenefitBranch.NAIVE, 1200)
        swizzle = replace(
            _observation(SwizzleBenefitBranch.SWIZZLE_AUTO, 1000),
            logical_bytes=32768,
        )
        with self.assertRaisesRegex(SchemaError, "same work"):
            SwizzleScaleBenefitEvidence.create(
                scale_ref="scale.S2",
                scale_ordinal=2,
                pattern=FusionPattern.GEMM_RS,
                same_work_digest="c" * 64,
                naive=naive,
                swizzle_auto=swizzle,
            )

    def test_physical_byte_hops_are_branch_specific(self) -> None:
        evidence = SwizzleScaleBenefitEvidence.create(
            scale_ref="scale.S2",
            scale_ordinal=2,
            pattern=FusionPattern.GEMM_RS,
            same_work_digest="e" * 64,
            naive=replace(
                _observation(SwizzleBenefitBranch.NAIVE, 1200),
                byte_hops=196608,
            ),
            swizzle_auto=replace(
                _observation(SwizzleBenefitBranch.SWIZZLE_AUTO, 1000),
                byte_hops=131072,
            ),
        )
        self.assertEqual(evidence.naive.byte_hops, 196608)
        self.assertEqual(evidence.swizzle_auto.byte_hops, 131072)

    def test_tile_and_control_counts_are_fail_closed(self) -> None:
        fused = _observation(SwizzleBenefitBranch.SWIZZLE_AUTO, 1000)
        for tile in (None, (), (1, 2), (1, 2, 0), [1, 2, 3]):
            with self.subTest(tile=tile):
                with self.assertRaisesRegex(SchemaError, "tile"):
                    replace(fused, tile_shape=tile).validate()
        with self.assertRaisesRegex(SchemaError, "ALLOC and FREE"):
            replace(fused, free_record_count=3).validate()
        with self.assertRaises(SchemaError):
            replace(fused, barrier_count=-1).validate()

    def test_nonadjacent_wins_do_not_raise_capability(self) -> None:
        evidence = tuple(
            SwizzleScaleBenefitEvidence.create(
                scale_ref=f"scale.S{ordinal}",
                scale_ordinal=ordinal,
                pattern=FusionPattern.GEMM_RS,
                same_work_digest="d" * 64,
                naive=_observation(SwizzleBenefitBranch.NAIVE, 1200),
                swizzle_auto=_observation(SwizzleBenefitBranch.SWIZZLE_AUTO, 1000),
            )
            for ordinal in (1, 3)
        )
        report = SwizzleBenefitReport.create(
            calibration_profile_ref="profile.ref",
            evidence=evidence,
        )
        self.assertFalse(report.reproducible_speedup)


if __name__ == "__main__":
    unittest.main()
