"""Deterministic fitting of isolated npusim markers into Swizzle costs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import shlex
from statistics import median

from ...errors import SchemaError
from ...schema.common import DType
from ...schema.swizzle import SwizzleEfficiencyPoint, SwizzleHardwareProfile
from ...schema.swizzle_calibration import (
    SwizzleCalibrationDisposition,
    SwizzleFixedMarkerKind,
    SwizzleFixedMarkerObservation,
    SwizzleFixedOverheads,
    SwizzleIsolatedCalibrationProfile,
    SwizzleMatmulMarkerObservation,
)
from ...schema.swizzle_performance_evidence import SwizzleCalibratedEfficiencyPoint


_FIXED_KINDS = tuple(SwizzleFixedMarkerKind)
_MARKER_PREFIX = "[SWIZZLE_CALIBRATION] "


@dataclass(frozen=True, slots=True)
class SwizzleCalibrationExportAudit:
    """Coverage result for one matching npu cost-model self-test output."""

    matching_tool_sha256: str
    source_selftest_passed: bool
    matmul_observation_count: int
    fixed_observation_counts: tuple[tuple[str, int], ...]
    missing_requirements: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_requirements


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _parse_fields(line: str, line_number: int) -> dict[str, str]:
    try:
        words = shlex.split(line[len(_MARKER_PREFIX) :], posix=True)
    except ValueError as error:
        raise SchemaError(
            f"malformed dedicated calibration marker: {error}",
            path=f"selftest_output.lines[{line_number}]",
        ) from error
    fields: dict[str, str] = {}
    for word in words:
        if word.count("=") != 1:
            raise SchemaError(
                "dedicated calibration marker fields must be key=value",
                path=f"selftest_output.lines[{line_number}]",
            )
        key, value = word.split("=", 1)
        if not key or not value or key in fields:
            raise SchemaError(
                "dedicated calibration marker fields must be nonempty and unique",
                path=f"selftest_output.lines[{line_number}]",
            )
        fields[key] = value
    return fields


def _exact_fields(
    fields: dict[str, str], expected: set[str], line_number: int
) -> None:
    if set(fields) != expected:
        raise SchemaError(
            "dedicated calibration marker has missing or unknown fields; "
            f"expected={sorted(expected)}, actual={sorted(fields)}",
            path=f"selftest_output.lines[{line_number}]",
        )


def _uint(value: str, field: str, line_number: int) -> int:
    if not value.isascii() or not value.isdecimal():
        raise SchemaError(
            "must be an unsigned decimal integer",
            path=f"selftest_output.lines[{line_number}].{field}",
        )
    result = int(value)
    if result > (1 << 64) - 1:
        raise SchemaError(
            "must fit uint64",
            path=f"selftest_output.lines[{line_number}].{field}",
        )
    return result


def _positive_float(value: str, field: str, line_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise SchemaError(
            "must be a positive finite float",
            path=f"selftest_output.lines[{line_number}].{field}",
        ) from error
    if not (result > 0.0) or result == float("inf"):
        raise SchemaError(
            "must be a positive finite float",
            path=f"selftest_output.lines[{line_number}].{field}",
        )
    return result


def parse_isolated_calibration_markers(
    output: str,
    *,
    matching_tool_sha256: str,
) -> tuple[
    SwizzleCalibrationExportAudit,
    tuple[SwizzleMatmulMarkerObservation, ...],
    tuple[SwizzleFixedMarkerObservation, ...],
]:
    """Parse only dedicated isolated markers; generic simulator logs are evidence-free."""

    if type(output) is not str:
        raise SchemaError("must be text", path="selftest_output")
    _validate_digest(matching_tool_sha256, "matching_tool_sha256")
    matmul: list[SwizzleMatmulMarkerObservation] = []
    fixed: list[SwizzleFixedMarkerObservation] = []
    matmul_fields = {
        "kind",
        "marker_ref",
        "m",
        "n",
        "k",
        "dtype",
        "accumulation_dtype",
        "flop_count",
        "peak_flops_per_cycle",
        "setup_cycles",
        "repeat0_cycles",
        "repeat1_cycles",
    }
    fixed_fields = {
        "kind",
        "marker_ref",
        "sample_ordinal",
        "repeat0_cycles",
        "repeat1_cycles",
    }
    fixed_kinds = {kind.value: kind for kind in _FIXED_KINDS}
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.startswith(_MARKER_PREFIX):
            continue
        fields = _parse_fields(line, line_number)
        kind_name = fields.get("kind")
        if kind_name == "matmul":
            _exact_fields(fields, matmul_fields, line_number)
            try:
                dtype = DType(fields["dtype"])
                accumulation_dtype = DType(fields["accumulation_dtype"])
            except ValueError as error:
                raise SchemaError(
                    "must use a known dtype",
                    path=f"selftest_output.lines[{line_number}]",
                ) from error
            observation = SwizzleMatmulMarkerObservation(
                fields["marker_ref"],
                _uint(fields["m"], "m", line_number),
                _uint(fields["n"], "n", line_number),
                _uint(fields["k"], "k", line_number),
                dtype,
                accumulation_dtype,
                _uint(fields["flop_count"], "flop_count", line_number),
                _positive_float(
                    fields["peak_flops_per_cycle"],
                    "peak_flops_per_cycle",
                    line_number,
                ),
                _uint(fields["setup_cycles"], "setup_cycles", line_number),
                (
                    _uint(fields["repeat0_cycles"], "repeat0_cycles", line_number),
                    _uint(fields["repeat1_cycles"], "repeat1_cycles", line_number),
                ),
            )
            observation.validate(f"selftest_output.lines[{line_number}]")
            matmul.append(observation)
        elif kind_name in fixed_kinds:
            _exact_fields(fields, fixed_fields, line_number)
            observation = SwizzleFixedMarkerObservation(
                fixed_kinds[kind_name],
                fields["marker_ref"],
                _uint(fields["sample_ordinal"], "sample_ordinal", line_number),
                (
                    _uint(fields["repeat0_cycles"], "repeat0_cycles", line_number),
                    _uint(fields["repeat1_cycles"], "repeat1_cycles", line_number),
                ),
            )
            observation.validate(f"selftest_output.lines[{line_number}]")
            fixed.append(observation)
        else:
            raise SchemaError(
                "unknown dedicated calibration marker kind",
                path=f"selftest_output.lines[{line_number}].kind",
            )

    matmul_observations = tuple(sorted(matmul, key=lambda item: item.key))
    fixed_observations = tuple(sorted(fixed, key=lambda item: item.key))
    if len({item.key for item in matmul_observations}) != len(matmul_observations):
        raise SchemaError("duplicate MATMUL marker", path="selftest_output")
    if len({item.key for item in fixed_observations}) != len(fixed_observations):
        raise SchemaError("duplicate fixed marker", path="selftest_output")

    source_passed = "ISA v1 manifest/factory self-test: PASS (" in output
    fixed_counts = tuple(
        (
            kind.value,
            sum(item.kind is kind for item in fixed_observations),
        )
        for kind in _FIXED_KINDS
    )
    missing: list[str] = []
    if not source_passed:
        missing.append("npu_cost_model_selftest.pass")
    if len(matmul_observations) < 3:
        missing.append("matmul>=3")
    for kind_name, count in fixed_counts:
        if count < 3 or count % 2 == 0:
            missing.append(f"{kind_name}>=3_odd")
    audit = SwizzleCalibrationExportAudit(
        matching_tool_sha256,
        source_passed,
        len(matmul_observations),
        fixed_counts,
        tuple(missing),
    )
    return audit, matmul_observations, fixed_observations


def export_measured_isolated_calibration(
    *,
    output: str,
    hardware_profile: SwizzleHardwareProfile,
    matching_tool_sha256: str,
    uncertainty_fraction: float = 0.05,
) -> SwizzleIsolatedCalibrationProfile:
    """Export a measured profile only from complete dedicated marker coverage."""

    audit, matmul, fixed = parse_isolated_calibration_markers(
        output,
        matching_tool_sha256=matching_tool_sha256,
    )
    if not audit.complete:
        raise SchemaError(
            "isolated marker export incomplete: "
            + ", ".join(audit.missing_requirements),
            path="npu_cost_model_selftest",
        )
    return fit_isolated_calibration(
        hardware_profile=hardware_profile,
        matching_tool_sha256=matching_tool_sha256,
        disposition=SwizzleCalibrationDisposition.MEASURED,
        uncertainty_fraction=uncertainty_fraction,
        matmul_observations=matmul,
        fixed_observations=fixed,
    )


def _fit_efficiency(
    observation: SwizzleMatmulMarkerObservation,
    uncertainty_fraction: float,
) -> SwizzleCalibratedEfficiencyPoint:
    observation.validate("matmul_observation")
    expected_flops = 2 * observation.m * observation.n * observation.k
    if observation.flop_count != expected_flops:
        raise SchemaError(
            "MATMUL flop count must be exactly 2*M*N*K",
            path="matmul_observation.flop_count",
        )
    active_cycles = observation.repeat_cycles[0] - observation.setup_cycles
    efficiency = observation.flop_count / (
        active_cycles * observation.peak_flops_per_cycle
    )
    if efficiency <= 0.0 or efficiency > 1.0:
        raise SchemaError(
            "derived MATMUL efficiency must be in (0, 1]",
            path="matmul_observation.repeat_cycles",
        )
    return SwizzleCalibratedEfficiencyPoint(
        observation.m,
        observation.n,
        observation.k,
        observation.dtype,
        observation.accumulation_dtype,
        float(efficiency),
        uncertainty_fraction,
        float(observation.setup_cycles),
    )


def _fit_fixed_overheads(
    observations: tuple[SwizzleFixedMarkerObservation, ...],
) -> SwizzleFixedOverheads:
    by_kind: dict[SwizzleFixedMarkerKind, list[SwizzleFixedMarkerObservation]] = defaultdict(list)
    for index, observation in enumerate(observations):
        observation.validate(f"fixed_observations[{index}]")
        by_kind[observation.kind].append(observation)
    if set(by_kind) != set(_FIXED_KINDS):
        missing = sorted(kind.value for kind in set(_FIXED_KINDS) - set(by_kind))
        extra = sorted(kind.value for kind in set(by_kind) - set(_FIXED_KINDS))
        raise SchemaError(
            f"fixed marker coverage is incomplete; missing={missing}, extra={extra}",
            path="fixed_observations",
        )
    fitted: dict[SwizzleFixedMarkerKind, int] = {}
    for kind in _FIXED_KINDS:
        samples = by_kind[kind]
        ordinals = tuple(item.sample_ordinal for item in samples)
        if len(samples) < 3 or len(samples) % 2 == 0:
            raise SchemaError(
                "each fixed marker kind requires an odd sample count of at least three",
                path=f"fixed_observations.{kind.value}",
            )
        if ordinals != tuple(range(len(samples))):
            raise SchemaError(
                "sample ordinals must be consecutive and canonical",
                path=f"fixed_observations.{kind.value}",
            )
        fitted[kind] = int(median(item.repeat_cycles[0] for item in samples))
    return SwizzleFixedOverheads(
        dte_launch_cycles=fitted[SwizzleFixedMarkerKind.DTE_LAUNCH],
        dte_sync_cycles=fitted[SwizzleFixedMarkerKind.DTE_SYNC],
        dte_hop_cycles=fitted[SwizzleFixedMarkerKind.DTE_HOP],
        sram_lifecycle_cycles=fitted[SwizzleFixedMarkerKind.SRAM_LIFECYCLE],
        control_cycles=fitted[SwizzleFixedMarkerKind.CONTROL],
    )


def fit_isolated_calibration(
    *,
    hardware_profile: SwizzleHardwareProfile,
    matching_tool_sha256: str,
    disposition: SwizzleCalibrationDisposition,
    uncertainty_fraction: float,
    matmul_observations: tuple[SwizzleMatmulMarkerObservation, ...],
    fixed_observations: tuple[SwizzleFixedMarkerObservation, ...],
) -> SwizzleIsolatedCalibrationProfile:
    """Fit isolated observations without consuming branch or speedup data."""

    hardware_profile.validate("hardware_profile")
    if type(disposition) is not SwizzleCalibrationDisposition:
        raise SchemaError("must use a typed disposition", path="disposition")
    if type(matmul_observations) is not tuple or len(matmul_observations) < 3:
        raise SchemaError(
            "requires at least three MATMUL observations",
            path="matmul_observations",
        )
    observation_keys = tuple(item.key for item in matmul_observations)
    if observation_keys != tuple(sorted(set(observation_keys))):
        raise SchemaError(
            "MATMUL observations must be unique and canonical",
            path="matmul_observations",
        )
    if type(fixed_observations) is not tuple:
        raise SchemaError("must be an immutable tuple", path="fixed_observations")
    fixed_keys = tuple(item.key for item in fixed_observations)
    if fixed_keys != tuple(sorted(set(fixed_keys))):
        raise SchemaError(
            "fixed observations must be unique and canonical",
            path="fixed_observations",
        )
    points = tuple(
        sorted(
            (
                _fit_efficiency(observation, uncertainty_fraction)
                for observation in matmul_observations
            ),
            key=lambda point: point.key,
        )
    )
    profile = SwizzleIsolatedCalibrationProfile.create(
        hardware_profile_ref=hardware_profile.id,
        matching_tool_sha256=matching_tool_sha256,
        disposition=disposition,
        uncertainty_fraction=uncertainty_fraction,
        matmul_observations=matmul_observations,
        fixed_observations=fixed_observations,
        matmul_marker_refs=tuple(sorted({item.marker_ref for item in matmul_observations})),
        fixed_marker_refs=tuple(sorted({item.marker_ref for item in fixed_observations})),
        efficiency_points=points,
        fixed_overheads=_fit_fixed_overheads(fixed_observations),
    )
    return profile


def materialize_calibrated_hardware_profile(
    calibration: SwizzleIsolatedCalibrationProfile,
    base_profile: SwizzleHardwareProfile,
    *,
    allow_provisional: bool = False,
) -> SwizzleHardwareProfile:
    """Apply a calibration profile, fail-closed for provisional evidence."""

    calibration.validate("calibration")
    base_profile.validate("base_profile")
    if type(allow_provisional) is not bool:
        raise SchemaError("must be a bool", path="allow_provisional")
    if calibration.hardware_profile_ref != base_profile.id:
        raise SchemaError(
            "calibration targets a different hardware profile",
            path="calibration.hardware_profile_ref",
        )
    if (
        calibration.disposition is SwizzleCalibrationDisposition.PROVISIONAL
        and not allow_provisional
    ):
        raise SchemaError(
            "provisional calibration requires explicit opt-in",
            path="allow_provisional",
        )
    overheads = calibration.fixed_overheads
    return SwizzleHardwareProfile.create(
        peak_flops_per_cycle=base_profile.peak_flops_per_cycle,
        confidence_fraction=max(
            base_profile.confidence_fraction,
            calibration.uncertainty_fraction,
        ),
        efficiency_points=tuple(
            SwizzleEfficiencyPoint(point.m, point.n, point.k, point.efficiency)
            for point in calibration.efficiency_points
        ),
        dte_launch_cycles=overheads.dte_launch_cycles,
        dte_sync_cycles=overheads.dte_sync_cycles,
        hop_latency_cycles=overheads.dte_hop_cycles,
        lane_bytes_per_cycle=base_profile.lane_bytes_per_cycle,
        max_inflight_dte=base_profile.max_inflight_dte,
        min_transfer_bytes=base_profile.min_transfer_bytes,
        efficient_tile_floor=base_profile.efficient_tile_floor,
        sram_budget_bytes=base_profile.sram_budget_bytes,
        double_buffer_supported=base_profile.double_buffer_supported,
    )


__all__ = [
    "SwizzleCalibrationExportAudit",
    "export_measured_isolated_calibration",
    "fit_isolated_calibration",
    "materialize_calibrated_hardware_profile",
    "parse_isolated_calibration_markers",
]
