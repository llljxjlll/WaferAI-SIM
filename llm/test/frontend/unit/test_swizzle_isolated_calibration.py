from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.swizzle.calibration import (
    export_measured_isolated_calibration,
    fit_isolated_calibration,
    materialize_calibrated_hardware_profile,
    parse_isolated_calibration_markers,
)
from llm.frontend.wafer_frontend.policies.swizzle.cost import interpolate_efficiency
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleEfficiencyPoint,
    SwizzleHardwareProfile,
)
from llm.frontend.wafer_frontend.schema.swizzle_calibration import (
    SwizzleCalibrationDisposition,
    SwizzleFixedMarkerKind,
    SwizzleFixedMarkerObservation,
    SwizzleMatmulMarkerObservation,
)


_TOOL_DIGEST = "a" * 64


def _base_profile() -> SwizzleHardwareProfile:
    return SwizzleHardwareProfile.create(
        peak_flops_per_cycle=1024.0,
        confidence_fraction=0.10,
        efficiency_points=(SwizzleEfficiencyPoint(16, 16, 16, 0.25),),
        dte_launch_cycles=99,
        dte_sync_cycles=98,
        hop_latency_cycles=97,
        lane_bytes_per_cycle=32.0,
        max_inflight_dte=2,
        min_transfer_bytes=64,
        efficient_tile_floor=(4, 16, 16),
        sram_budget_bytes=1 << 20,
        double_buffer_supported=True,
    )


def _matmul_observations() -> tuple[SwizzleMatmulMarkerObservation, ...]:
    return (
        SwizzleMatmulMarkerObservation(
            "isa.npu_cost_model_selftest/MATMUL.16x16x16",
            16,
            16,
            16,
            DType.FP16,
            DType.FP32,
            8192,
            1024.0,
            4,
            (20, 20),
        ),
        SwizzleMatmulMarkerObservation(
            "isa.npu_cost_model_selftest/MATMUL.32x16x16",
            32,
            16,
            16,
            DType.FP16,
            DType.FP32,
            16384,
            1024.0,
            4,
            (24, 24),
        ),
        SwizzleMatmulMarkerObservation(
            "isa.npu_cost_model_selftest/MATMUL.32x32x16",
            32,
            32,
            16,
            DType.FP16,
            DType.FP32,
            32768,
            1024.0,
            4,
            (36, 36),
        ),
    )


_FIXED_VALUES = {
    SwizzleFixedMarkerKind.DTE_LAUNCH: (5, 6, 7),
    SwizzleFixedMarkerKind.DTE_SYNC: (2, 3, 4),
    SwizzleFixedMarkerKind.DTE_HOP: (7, 8, 9),
    SwizzleFixedMarkerKind.SRAM_LIFECYCLE: (0, 0, 1),
    SwizzleFixedMarkerKind.CONTROL: (0, 1, 1),
}


def _fixed_observations() -> tuple[SwizzleFixedMarkerObservation, ...]:
    observations = tuple(
        SwizzleFixedMarkerObservation(
            kind,
            f"npusim/{kind.value}",
            ordinal,
            (cycles, cycles),
        )
        for kind, values in _FIXED_VALUES.items()
        for ordinal, cycles in enumerate(values)
    )
    return tuple(sorted(observations, key=lambda item: item.key))


def _fit(*, disposition: SwizzleCalibrationDisposition = SwizzleCalibrationDisposition.PROVISIONAL):
    return fit_isolated_calibration(
        hardware_profile=_base_profile(),
        matching_tool_sha256=_TOOL_DIGEST,
        disposition=disposition,
        uncertainty_fraction=(0.25 if disposition is SwizzleCalibrationDisposition.PROVISIONAL else 0.05),
        matmul_observations=_matmul_observations(),
        fixed_observations=_fixed_observations(),
    )


def _dedicated_output() -> str:
    lines = ["ISA v1 manifest/factory self-test: PASS (1136 checks)"]
    for item in _matmul_observations():
        lines.append(
            "[SWIZZLE_CALIBRATION] "
            f"kind=matmul marker_ref={item.marker_ref} "
            f"m={item.m} n={item.n} k={item.k} "
            f"dtype={item.dtype.value} "
            f"accumulation_dtype={item.accumulation_dtype.value} "
            f"flop_count={item.flop_count} "
            f"peak_flops_per_cycle={item.peak_flops_per_cycle} "
            f"setup_cycles={item.setup_cycles} "
            f"repeat0_cycles={item.repeat_cycles[0]} "
            f"repeat1_cycles={item.repeat_cycles[1]}"
        )
    for item in _fixed_observations():
        lines.append(
            "[SWIZZLE_CALIBRATION] "
            f"kind={item.kind.value} marker_ref={item.marker_ref} "
            f"sample_ordinal={item.sample_ordinal} "
            f"repeat0_cycles={item.repeat_cycles[0]} "
            f"repeat1_cycles={item.repeat_cycles[1]}"
        )
    return "\n".join(lines)


class SwizzleIsolatedCalibrationTests(unittest.TestCase):
    def test_multi_point_efficiency_and_fixed_medians_are_exact(self) -> None:
        profile = _fit()
        self.assertEqual(
            tuple(point.efficiency for point in profile.efficiency_points),
            (0.5, 0.8, 1.0),
        )
        self.assertEqual(profile.fixed_overheads.dte_launch_cycles, 6)
        self.assertEqual(profile.fixed_overheads.dte_sync_cycles, 3)
        self.assertEqual(profile.fixed_overheads.dte_hop_cycles, 8)
        self.assertEqual(profile.fixed_overheads.sram_lifecycle_cycles, 0)
        self.assertEqual(profile.fixed_overheads.control_cycles, 1)
        self.assertEqual(profile.id, _fit().id)

    def test_provisional_profile_is_fail_closed_without_explicit_opt_in(self) -> None:
        calibration = _fit()
        with self.assertRaisesRegex(SchemaError, "explicit opt-in"):
            materialize_calibrated_hardware_profile(calibration, _base_profile())
        profile = materialize_calibrated_hardware_profile(
            calibration,
            _base_profile(),
            allow_provisional=True,
        )
        self.assertEqual(profile.confidence_fraction, 0.25)
        self.assertEqual(profile.dte_launch_cycles, 6)
        self.assertEqual(profile.dte_sync_cycles, 3)
        self.assertEqual(profile.hop_latency_cycles, 8)
        self.assertEqual(interpolate_efficiency(profile, (32, 16, 16)), 0.8)

    def test_measured_profile_does_not_require_provisional_override(self) -> None:
        calibration = _fit(disposition=SwizzleCalibrationDisposition.MEASURED)
        profile = materialize_calibrated_hardware_profile(calibration, _base_profile())
        self.assertEqual(profile.confidence_fraction, 0.10)

    def test_repeat_drift_and_nonisolated_flop_count_are_rejected(self) -> None:
        observations = _matmul_observations()
        with self.assertRaisesRegex(SchemaError, "repeat cycle counts"):
            replace(observations[0], repeat_cycles=(20, 21)).validate()
        forged = (replace(observations[0], flop_count=4096), *observations[1:])
        with self.assertRaisesRegex(SchemaError, "2\*M\*N\*K"):
            fit_isolated_calibration(
                hardware_profile=_base_profile(),
                matching_tool_sha256=_TOOL_DIGEST,
                disposition=SwizzleCalibrationDisposition.MEASURED,
                uncertainty_fraction=0.05,
                matmul_observations=forged,
                fixed_observations=_fixed_observations(),
            )

    def test_missing_fixed_marker_and_noncanonical_samples_fail_closed(self) -> None:
        fixed = _fixed_observations()
        missing_control = tuple(
            item for item in fixed if item.kind is not SwizzleFixedMarkerKind.CONTROL
        )
        with self.assertRaisesRegex(SchemaError, "coverage is incomplete"):
            fit_isolated_calibration(
                hardware_profile=_base_profile(),
                matching_tool_sha256=_TOOL_DIGEST,
                disposition=SwizzleCalibrationDisposition.PROVISIONAL,
                uncertainty_fraction=0.25,
                matmul_observations=_matmul_observations(),
                fixed_observations=missing_control,
            )
        with self.assertRaisesRegex(SchemaError, "canonical"):
            fit_isolated_calibration(
                hardware_profile=_base_profile(),
                matching_tool_sha256=_TOOL_DIGEST,
                disposition=SwizzleCalibrationDisposition.PROVISIONAL,
                uncertainty_fraction=0.25,
                matmul_observations=_matmul_observations(),
                fixed_observations=tuple(reversed(fixed)),
            )

    def test_provisional_uncertainty_floor_and_no_speedup_input(self) -> None:
        with self.assertRaisesRegex(SchemaError, "25% uncertainty"):
            replace(_fit(), uncertainty_fraction=0.05).validate()
        self.assertNotIn("speedup", inspect.signature(fit_isolated_calibration).parameters)

    def test_generic_selftest_output_is_not_calibration_evidence(self) -> None:
        output = "\n".join(
            (
                "performance_cycle 139264 exu_flops 3119512",
                "performance_cycle 28 exu_flops 626",
                "ISA v1 manifest/factory self-test: PASS (1136 checks)",
            )
        )
        audit, matmul, fixed = parse_isolated_calibration_markers(
            output,
            matching_tool_sha256=_TOOL_DIGEST,
        )
        self.assertTrue(audit.source_selftest_passed)
        self.assertFalse(audit.complete)
        self.assertEqual(matmul, ())
        self.assertEqual(fixed, ())
        self.assertEqual(
            audit.missing_requirements,
            (
                "matmul>=3",
                "dte_launch>=3_odd",
                "dte_sync>=3_odd",
                "dte_hop>=3_odd",
                "sram_lifecycle>=3_odd",
                "control>=3_odd",
            ),
        )
        with self.assertRaisesRegex(SchemaError, "isolated marker export incomplete"):
            export_measured_isolated_calibration(
                output=output,
                hardware_profile=_base_profile(),
                matching_tool_sha256=_TOOL_DIGEST,
            )

    def test_complete_dedicated_markers_export_measured_profile(self) -> None:
        profile = export_measured_isolated_calibration(
            output=_dedicated_output(),
            hardware_profile=_base_profile(),
            matching_tool_sha256=_TOOL_DIGEST,
        )
        self.assertIs(profile.disposition, SwizzleCalibrationDisposition.MEASURED)
        self.assertEqual(profile.fixed_overheads.dte_launch_cycles, 6)
        self.assertEqual(len(profile.efficiency_points), 3)

    def test_malformed_and_duplicate_dedicated_markers_are_rejected(self) -> None:
        malformed = (
            "ISA v1 manifest/factory self-test: PASS (1136 checks)\n"
            "[SWIZZLE_CALIBRATION] kind=matmul surprise=1"
        )
        with self.assertRaisesRegex(SchemaError, "missing or unknown fields"):
            parse_isolated_calibration_markers(
                malformed,
                matching_tool_sha256=_TOOL_DIGEST,
            )
        lines = _dedicated_output().splitlines()
        duplicate = "\n".join((*lines, lines[1]))
        with self.assertRaisesRegex(SchemaError, "duplicate MATMUL marker"):
            parse_isolated_calibration_markers(
                duplicate,
                matching_tool_sha256=_TOOL_DIGEST,
            )

    def test_measured_exporter_has_no_branch_timing_inputs(self) -> None:
        parameters = inspect.signature(export_measured_isolated_calibration).parameters
        self.assertNotIn("speedup", parameters)
        self.assertNotIn("makespan", parameters)


if __name__ == "__main__":
    unittest.main()
