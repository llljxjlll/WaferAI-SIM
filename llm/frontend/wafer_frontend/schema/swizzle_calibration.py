"""Typed isolated calibration evidence for the Swizzle cost model.

The carrier intentionally has no branch timing or speedup field.  It accepts
only isolated npusim markers, so an end-to-end result cannot be fed back into
the analytical selector as a calibration input.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from statistics import median

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .swizzle_performance_evidence import SwizzleCalibratedEfficiencyPoint


SWIZZLE_ISOLATED_CALIBRATION_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_isolated_calibration/v1alpha1"
)


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _fraction(value: float, path: str) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise SchemaError("must be a finite fraction in (0, 1]", path=path)


def _exact_repeats(values: tuple[int, int], path: str, *, positive: bool) -> None:
    if type(values) is not tuple or len(values) != 2:
        raise SchemaError("requires exactly two repeat cycle counts", path=path)
    for index, value in enumerate(values):
        validate_uint64(value, f"{path}[{index}]")
        if positive and value == 0:
            raise SchemaError("must be positive", path=f"{path}[{index}]")
    if values[0] != values[1]:
        raise SchemaError("repeat cycle counts must be exact", path=path)


class SwizzleCalibrationDisposition(str, Enum):
    MEASURED = "measured"
    PROVISIONAL = "provisional"


class SwizzleFixedMarkerKind(str, Enum):
    DTE_LAUNCH = "dte_launch"
    DTE_SYNC = "dte_sync"
    DTE_HOP = "dte_hop"
    SRAM_LIFECYCLE = "sram_lifecycle"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class SwizzleMatmulMarkerObservation:
    marker_ref: str
    m: int
    n: int
    k: int
    dtype: DType
    accumulation_dtype: DType
    flop_count: int
    peak_flops_per_cycle: float
    setup_cycles: int
    repeat_cycles: tuple[int, int]

    @property
    def key(self) -> tuple[int, int, int, str, str, str]:
        return (
            self.m,
            self.n,
            self.k,
            self.dtype.value,
            self.accumulation_dtype.value,
            self.marker_ref,
        )

    def validate(self, path: str = "swizzle_matmul_marker") -> None:
        validate_nonempty(self.marker_ref, f"{path}.marker_ref")
        for name in ("m", "n", "k", "flop_count"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")
        if type(self.dtype) is not DType or type(self.accumulation_dtype) is not DType:
            raise SchemaError("must use typed dtypes", path=path)
        if (
            type(self.peak_flops_per_cycle) is not float
            or not math.isfinite(self.peak_flops_per_cycle)
            or self.peak_flops_per_cycle <= 0.0
        ):
            raise SchemaError(
                "must be a finite positive float",
                path=f"{path}.peak_flops_per_cycle",
            )
        validate_uint64(self.setup_cycles, f"{path}.setup_cycles")
        _exact_repeats(self.repeat_cycles, f"{path}.repeat_cycles", positive=True)
        if self.repeat_cycles[0] <= self.setup_cycles:
            raise SchemaError(
                "observed cycles must exceed setup cycles",
                path=f"{path}.repeat_cycles",
            )


@dataclass(frozen=True, slots=True)
class SwizzleFixedMarkerObservation:
    kind: SwizzleFixedMarkerKind
    marker_ref: str
    sample_ordinal: int
    repeat_cycles: tuple[int, int]

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.kind.value, self.sample_ordinal, self.marker_ref)

    def validate(self, path: str = "swizzle_fixed_marker") -> None:
        if type(self.kind) is not SwizzleFixedMarkerKind:
            raise SchemaError("must use a typed fixed marker kind", path=f"{path}.kind")
        validate_nonempty(self.marker_ref, f"{path}.marker_ref")
        validate_uint64(self.sample_ordinal, f"{path}.sample_ordinal")
        _exact_repeats(self.repeat_cycles, f"{path}.repeat_cycles", positive=False)


@dataclass(frozen=True, slots=True)
class SwizzleFixedOverheads:
    dte_launch_cycles: int
    dte_sync_cycles: int
    dte_hop_cycles: int
    sram_lifecycle_cycles: int
    control_cycles: int

    def validate(self, path: str = "swizzle_fixed_overheads") -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class SwizzleIsolatedCalibrationProfile:
    schema_version: str
    id: str
    hardware_profile_ref: str
    matching_tool_sha256: str
    disposition: SwizzleCalibrationDisposition
    uncertainty_fraction: float
    matmul_observations: tuple[SwizzleMatmulMarkerObservation, ...]
    fixed_observations: tuple[SwizzleFixedMarkerObservation, ...]
    matmul_marker_refs: tuple[str, ...]
    fixed_marker_refs: tuple[str, ...]
    efficiency_points: tuple[SwizzleCalibratedEfficiencyPoint, ...]
    fixed_overheads: SwizzleFixedOverheads

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleIsolatedCalibrationProfile":
        result = cls(
            schema_version=SWIZZLE_ISOLATED_CALIBRATION_SCHEMA_VERSION,
            id=stable_artifact_id(
                "swizzle_isolated_calibration",
                semantic,
                schema_version=SWIZZLE_ISOLATED_CALIBRATION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "swizzle_isolated_calibration") -> None:
        if self.schema_version != SWIZZLE_ISOLATED_CALIBRATION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.hardware_profile_ref, f"{path}.hardware_profile_ref")
        _digest(self.matching_tool_sha256, f"{path}.matching_tool_sha256")
        if type(self.disposition) is not SwizzleCalibrationDisposition:
            raise SchemaError("must use a typed disposition", path=f"{path}.disposition")
        _fraction(self.uncertainty_fraction, f"{path}.uncertainty_fraction")
        if (
            self.disposition is SwizzleCalibrationDisposition.PROVISIONAL
            and self.uncertainty_fraction < 0.25
        ):
            raise SchemaError(
                "provisional calibration requires at least 25% uncertainty",
                path=f"{path}.uncertainty_fraction",
            )
        if type(self.matmul_observations) is not tuple or len(self.matmul_observations) < 3:
            raise SchemaError(
                "requires at least three MATMUL observations",
                path=f"{path}.matmul_observations",
            )
        for index, observation in enumerate(self.matmul_observations):
            observation.validate(f"{path}.matmul_observations[{index}]")
        observation_keys = tuple(item.key for item in self.matmul_observations)
        if observation_keys != tuple(sorted(set(observation_keys))):
            raise SchemaError(
                "MATMUL observations must be unique and canonical",
                path=f"{path}.matmul_observations",
            )
        if type(self.fixed_observations) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.fixed_observations")
        for index, observation in enumerate(self.fixed_observations):
            observation.validate(f"{path}.fixed_observations[{index}]")
        fixed_keys = tuple(item.key for item in self.fixed_observations)
        if fixed_keys != tuple(sorted(set(fixed_keys))):
            raise SchemaError(
                "fixed observations must be unique and canonical",
                path=f"{path}.fixed_observations",
            )
        for name in ("matmul_marker_refs", "fixed_marker_refs"):
            refs = getattr(self, name)
            if type(refs) is not tuple or not refs:
                raise SchemaError("must contain marker refs", path=f"{path}.{name}")
            if refs != tuple(sorted(set(refs))):
                raise SchemaError("must be unique and canonical", path=f"{path}.{name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{name}[{index}]")
        expected_matmul_refs = tuple(sorted({item.marker_ref for item in self.matmul_observations}))
        expected_fixed_refs = tuple(sorted({item.marker_ref for item in self.fixed_observations}))
        if self.matmul_marker_refs != expected_matmul_refs:
            raise SchemaError(
                "must exactly cover MATMUL observations",
                path=f"{path}.matmul_marker_refs",
            )
        if self.fixed_marker_refs != expected_fixed_refs:
            raise SchemaError(
                "must exactly cover fixed observations",
                path=f"{path}.fixed_marker_refs",
            )
        if type(self.efficiency_points) is not tuple or len(self.efficiency_points) < 3:
            raise SchemaError(
                "requires at least three MATMUL efficiency points",
                path=f"{path}.efficiency_points",
            )
        for index, point in enumerate(self.efficiency_points):
            point.validate(f"{path}.efficiency_points[{index}]")
        point_keys = tuple(point.key for point in self.efficiency_points)
        if point_keys != tuple(sorted(set(point_keys))):
            raise SchemaError(
                "efficiency points must be unique and canonical",
                path=f"{path}.efficiency_points",
            )
        expected_point_keys = tuple(
            (
                item.m,
                item.n,
                item.k,
                item.dtype.value,
                item.accumulation_dtype.value,
            )
            for item in self.matmul_observations
        )
        if point_keys != expected_point_keys:
            raise SchemaError(
                "efficiency points must exactly cover MATMUL observations",
                path=f"{path}.efficiency_points",
            )
        for index, (point, observation) in enumerate(
            zip(self.efficiency_points, self.matmul_observations)
        ):
            active_cycles = observation.repeat_cycles[0] - observation.setup_cycles
            expected_efficiency = observation.flop_count / (
                active_cycles * observation.peak_flops_per_cycle
            )
            if not math.isclose(point.efficiency, expected_efficiency, rel_tol=0.0, abs_tol=1e-15):
                raise SchemaError(
                    "efficiency disagrees with isolated MATMUL observation",
                    path=f"{path}.efficiency_points[{index}].efficiency",
                )
            if point.confidence_fraction != self.uncertainty_fraction:
                raise SchemaError(
                    "point confidence must equal profile uncertainty",
                    path=f"{path}.efficiency_points[{index}].confidence_fraction",
                )
            if point.setup_cycles != float(observation.setup_cycles):
                raise SchemaError(
                    "setup cycles disagree with isolated MATMUL observation",
                    path=f"{path}.efficiency_points[{index}].setup_cycles",
                )
        if type(self.fixed_overheads) is not SwizzleFixedOverheads:
            raise SchemaError("must carry typed fixed overheads", path=f"{path}.fixed_overheads")
        self.fixed_overheads.validate(f"{path}.fixed_overheads")
        samples_by_kind = {
            kind: tuple(item for item in self.fixed_observations if item.kind is kind)
            for kind in SwizzleFixedMarkerKind
        }
        for kind, samples in samples_by_kind.items():
            ordinals = tuple(item.sample_ordinal for item in samples)
            if len(samples) < 3 or len(samples) % 2 == 0:
                raise SchemaError(
                    "each fixed marker kind requires an odd sample count of at least three",
                    path=f"{path}.fixed_observations.{kind.value}",
                )
            if ordinals != tuple(range(len(samples))):
                raise SchemaError(
                    "sample ordinals must be consecutive and canonical",
                    path=f"{path}.fixed_observations.{kind.value}",
                )
        expected_overheads = SwizzleFixedOverheads(
            dte_launch_cycles=int(median(
                item.repeat_cycles[0] for item in samples_by_kind[SwizzleFixedMarkerKind.DTE_LAUNCH]
            )),
            dte_sync_cycles=int(median(
                item.repeat_cycles[0] for item in samples_by_kind[SwizzleFixedMarkerKind.DTE_SYNC]
            )),
            dte_hop_cycles=int(median(
                item.repeat_cycles[0] for item in samples_by_kind[SwizzleFixedMarkerKind.DTE_HOP]
            )),
            sram_lifecycle_cycles=int(median(
                item.repeat_cycles[0] for item in samples_by_kind[SwizzleFixedMarkerKind.SRAM_LIFECYCLE]
            )),
            control_cycles=int(median(
                item.repeat_cycles[0] for item in samples_by_kind[SwizzleFixedMarkerKind.CONTROL]
            )),
        )
        if self.fixed_overheads != expected_overheads:
            raise SchemaError(
                "fixed overheads disagree with isolated marker medians",
                path=f"{path}.fixed_overheads",
            )
        expected = stable_artifact_id(
            "swizzle_isolated_calibration",
            self._semantic_key(),
            schema_version=SWIZZLE_ISOLATED_CALIBRATION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


__all__ = [
    "SWIZZLE_ISOLATED_CALIBRATION_SCHEMA_VERSION",
    "SwizzleCalibrationDisposition",
    "SwizzleFixedMarkerKind",
    "SwizzleFixedMarkerObservation",
    "SwizzleFixedOverheads",
    "SwizzleIsolatedCalibrationProfile",
    "SwizzleMatmulMarkerObservation",
]
