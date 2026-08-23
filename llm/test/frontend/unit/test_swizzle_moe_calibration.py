from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES,
    MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES,
    MoeCalibrationKind,
    MoeCalibrationStatus,
    MoeSwizzleCalibrationProfile,
    MoeSwizzleRuntimeMarkers,
)
from llm.test.frontend.integration.moe_swizzle_runtime_markers import (
    parse_moe_swizzle_calibration,
    parse_moe_swizzle_runtime_markers,
)


_DIGESTS = {
    "tool_sha256": "a" * 64,
    "hardware_sha256": "b" * 64,
    "simulation_sha256": "c" * 64,
    "mapping_sha256": "d" * 64,
}


def _calibration_line(
    kind: MoeCalibrationKind,
    sample: int,
    repeat: int,
    *,
    shape: tuple[int, int, int] | None = None,
) -> str:
    shape_text = "none" if shape is None else "x".join(str(item) for item in shape)
    dtype = "fp16" if shape is not None else "none"
    suffix = " ".join(f"{name}={value}" for name, value in _DIGESTS.items())
    return (
        "[MOE_SWIZZLE_CALIBRATION] "
        f"kind={kind.value} sample={sample} repeat={repeat} cycles={11 + sample + repeat} "
        f"shape={shape_text} dtype={dtype} {suffix}"
    )


def _complete_calibration() -> str:
    shapes = MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES
    return "\n".join(
        (
            *(
                _calibration_line(MoeCalibrationKind.GROUP_GEMM, sample, repeat, shape=shape)
                for shape in shapes
                for sample in range(3)
                for repeat in range(2)
            ),
            *(
                _calibration_line(MoeCalibrationKind.SWIGLU_GROUP, sample, repeat, shape=shape)
                for shape in MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES
                for sample in range(3)
                for repeat in range(2)
            ),
            *(
                _calibration_line(kind, sample, repeat)
                for kind in MoeCalibrationKind
                if kind not in (MoeCalibrationKind.GROUP_GEMM, MoeCalibrationKind.SWIGLU_GROUP)
                for sample in range(3)
                for repeat in range(2)
            ),
        )
    )


_EDGES = (
    (0, 1, "x+"),
    (1, 0, "x-"),
    (2, 3, "x+"),
    (3, 2, "x-"),
    (0, 2, "y+"),
    (2, 0, "y-"),
    (1, 3, "y+"),
    (3, 1, "y-"),
)


def _complete_runtime() -> str:
    return "\n".join(
        (
            "ordinary simulator output is not evidence",
            *(
                "[MOE_SWIZZLE_SESSION] "
                f"die={die} capacity_per_core=3 active_core_count=2 "
                f"aggregate_capacity=6 send_peak={die + 1} "
                f"recv_peak={4 - die} opens=9 retires=9"
                for die in range(4)
            ),
            *(
                "[MOE_SWIZZLE_OVERLAP] scope=die "
                f"die={die} compute_cycles=50 dte_cycles=40 "
                "compute_dte_cycles=10 window_cycles=100"
                for die in range(4)
            ),
            "[MOE_SWIZZLE_OVERLAP] scope=global die=all "
            "compute_cycles=70 dte_cycles=60 compute_dte_cycles=37 "
            "window_cycles=100",
            *(
                "[MOE_SWIZZLE_PORT_TIME] "
                f"source_die={source} destination_die={destination} direction={direction} "
                "busy_cycles=11 window_cycles=100"
                for source, destination, direction in _EDGES
            ),
            "[MOE_SWIZZLE_SETUP] group_gemm_primitives=8 "
            "group_gemm_setup_cycles=6 matmul_total_cycles=31 "
            "dte_launch_count=24 "
            "physical_root_count=20 event_record_count=16 "
            "sram_lifecycle_cycles=7 bind_cycles=5 event_control_cycles=9",
        )
    )


class MoeSwizzleCalibrationTest(unittest.TestCase):
    def test_absent_markers_remain_provisional_and_missing(self) -> None:
        profile = parse_moe_swizzle_calibration("performance_cycle 99", **_DIGESTS)
        self.assertIs(profile.status, MoeCalibrationStatus.PROVISIONAL)
        self.assertEqual(profile.samples, ())
        self.assertTrue(any(item.startswith("group_gemm_shape:") for item in profile.missing_measurements))
        runtime = parse_moe_swizzle_runtime_markers("performance_cycle 99")
        self.assertFalse(runtime.measurement_complete)
        self.assertIsNone(runtime.observed_max_inflight_send)
        self.assertIsNone(runtime.compute_dte_overlap_cycles)
        self.assertEqual(runtime.directional_port_times, ())
        self.assertGreaterEqual(len(runtime.missing_measurements), 9)

    def test_complete_calibration_and_runtime_are_exact_and_strict_serde(self) -> None:
        profile = parse_moe_swizzle_calibration(_complete_calibration(), **_DIGESTS)
        self.assertIs(profile.status, MoeCalibrationStatus.MEASURED)
        self.assertEqual(len(profile.samples), 168)
        self.assertEqual(profile.missing_measurements, ())
        self.assertEqual(
            loads_dataclass(
                MoeSwizzleCalibrationProfile,
                canonical_json(profile),
                path="profile",
            ),
            profile,
        )
        runtime = parse_moe_swizzle_runtime_markers(_complete_runtime())
        self.assertTrue(runtime.measurement_complete)
        self.assertEqual(runtime.missing_measurements, ())
        self.assertEqual(runtime.observed_max_inflight_send, 4)
        self.assertEqual(runtime.observed_max_inflight_recv, 4)
        self.assertEqual(len(runtime.die_session_capacities), 4)
        self.assertTrue(
            all(item.aggregate_capacity == 6 for item in runtime.die_session_capacities)
        )
        self.assertEqual(runtime.compute_dte_overlap_cycles, 37)
        self.assertEqual(len(runtime.directional_port_times), 8)
        self.assertEqual(runtime.group_gemm_setup_cycles, 6)
        self.assertEqual(runtime.matmul_total_cycles, 31)
        self.assertEqual(len(runtime.die_compute_dte_overlaps), 4)
        self.assertEqual(
            loads_dataclass(
                MoeSwizzleRuntimeMarkers,
                canonical_json(runtime),
                path="runtime",
            ),
            runtime,
        )

    def test_calibration_duplicate_unknown_field_and_sha_drift_fail_closed(self) -> None:
        lines = _complete_calibration().splitlines()
        with self.assertRaisesRegex(SchemaError, "duplicate calibration sample"):
            parse_moe_swizzle_calibration("\n".join((*lines, lines[0])), **_DIGESTS)
        with self.assertRaisesRegex(SchemaError, "missing or unknown fields"):
            parse_moe_swizzle_calibration(lines[0] + " surprise=1", **_DIGESTS)
        with self.assertRaisesRegex(SchemaError, "exact production coverage"):
            parse_moe_swizzle_calibration(
                _complete_calibration().replace("shape=1x8x32", "shape=3x7x11", 1),
                **_DIGESTS,
            )
        with self.assertRaisesRegex(SchemaError, "SHA drifted"):
            parse_moe_swizzle_calibration(
                lines[0].replace("tool_sha256=" + "a" * 64, "tool_sha256=" + "e" * 64),
                **_DIGESTS,
            )
        incomplete = parse_moe_swizzle_calibration("\n".join(lines[:-1]), **_DIGESTS)
        self.assertIs(incomplete.status, MoeCalibrationStatus.PROVISIONAL)
        self.assertIn(MoeCalibrationKind.TERMINAL_DONE.value, incomplete.missing_measurements)
        without_full_down = parse_moe_swizzle_calibration(
            "\n".join(item for item in lines if "shape=1x16x32" not in item),
            **_DIGESTS,
        )
        self.assertIs(without_full_down.status, MoeCalibrationStatus.PROVISIONAL)
        self.assertIn("group_gemm_shape:(1, 16, 32)", without_full_down.missing_measurements)
        without_swiglu_group = parse_moe_swizzle_calibration(
            "\n".join(item for item in lines if "kind=swiglu_group" not in item),
            **_DIGESTS,
        )
        self.assertIs(without_swiglu_group.status, MoeCalibrationStatus.PROVISIONAL)
        self.assertIn(
            "swiglu_group_shape:(1, 32, 32)",
            without_swiglu_group.missing_measurements,
        )

    def test_runtime_duplicates_unretired_sessions_and_wrong_edges_fail_closed(self) -> None:
        lines = _complete_runtime().splitlines()
        with self.assertRaisesRegex(SchemaError, "all sessions must retire"):
            parse_moe_swizzle_runtime_markers(
                _complete_runtime().replace("opens=9 retires=9", "opens=9 retires=8", 1)
            )
        with self.assertRaisesRegex(SchemaError, "session capacity"):
            parse_moe_swizzle_runtime_markers(
                _complete_runtime().replace(
                    "capacity_per_core=3 active_core_count=2 aggregate_capacity=6",
                    "capacity_per_core=3 active_core_count=2 aggregate_capacity=5",
                    1,
                )
            )
        with self.assertRaisesRegex(SchemaError, "duplicate .*overlap"):
            parse_moe_swizzle_runtime_markers("\n".join((*lines, lines[5])))
        with self.assertRaisesRegex(SchemaError, "directed 2x2 mesh"):
            parse_moe_swizzle_runtime_markers(
                _complete_runtime().replace(
                    "source_die=0 destination_die=1 direction=x+",
                    "source_die=0 destination_die=3 direction=x+",
                )
            )
        one_port = next(
            line for line in lines if line.startswith("[MOE_SWIZZLE_PORT_TIME]")
        )
        missing_edge = parse_moe_swizzle_runtime_markers(
            "\n".join(line for line in lines if line != one_port)
        )
        self.assertFalse(missing_edge.measurement_complete)
        self.assertIn(
            "directional_port_utilization_over_time",
            missing_edge.missing_measurements,
        )
        missing_die = parse_moe_swizzle_runtime_markers(
            "\n".join(
                line for line in lines
                if not (
                    line.startswith("[MOE_SWIZZLE_OVERLAP]")
                    and "scope=die die=3 " in line
                )
            )
        )
        self.assertFalse(missing_die.measurement_complete)
        self.assertIn(
            "die_compute_dte_overlap_over_time",
            missing_die.missing_measurements,
        )
        zero_overlap = parse_moe_swizzle_runtime_markers(
            _complete_runtime().replace(
                "scope=die die=0 compute_cycles=50 dte_cycles=40 "
                "compute_dte_cycles=10",
                "scope=die die=0 compute_cycles=50 dte_cycles=40 "
                "compute_dte_cycles=0",
            )
        )
        self.assertEqual(
            zero_overlap.die_compute_dte_overlaps[0].compute_dte_cycles, 0
        )
        with self.assertRaisesRegex(SchemaError, "missing or unknown fields"):
            parse_moe_swizzle_runtime_markers(
                one_port + " first_cycle=1 last_cycle=99"
            )
        with self.assertRaisesRegex(SchemaError, "unknown dedicated runtime marker"):
            parse_moe_swizzle_runtime_markers("[MOE_SWIZZLE_SURPRISE] value=1")


if __name__ == "__main__":
    unittest.main()
