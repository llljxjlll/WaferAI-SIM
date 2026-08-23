"""Typed calibration and runtime-benefit evidence for Swizzle.

This module deliberately separates economic AUTO evidence from forced
deployment coverage.  A forced run can prove executability, but it can never
raise the reproducible-performance capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import FusionPattern
from .swizzle import SwizzleAlgorithm


SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_calibration_profile/v1alpha1"
)
SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_benefit_evidence/v1alpha1"
)
SWIZZLE_MINIMUM_QUALIFYING_SPEEDUP = 1.10


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _positive_float(value: float, path: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise SchemaError("must be a finite positive float", path=path)


def _fraction(value: float, path: str) -> None:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise SchemaError("must be a finite fraction in (0, 1]", path=path)


class SwizzleBenefitBranch(str, Enum):
    NAIVE = "naive"
    SWIZZLE_AUTO = "swizzle_auto"
    SWIZZLE_FORCED_DIAGNOSTIC = "swizzle_forced_diagnostic"


@dataclass(frozen=True, slots=True)
class SwizzleCalibratedEfficiencyPoint:
    m: int
    n: int
    k: int
    dtype: DType
    accumulation_dtype: DType
    efficiency: float
    confidence_fraction: float
    setup_cycles: float

    def validate(self, path: str = "swizzle_calibrated_efficiency_point") -> None:
        for name in ("m", "n", "k"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")
        if type(self.dtype) is not DType or type(self.accumulation_dtype) is not DType:
            raise SchemaError("must use typed dtypes", path=path)
        _fraction(self.efficiency, f"{path}.efficiency")
        _fraction(self.confidence_fraction, f"{path}.confidence_fraction")
        if (
            type(self.setup_cycles) is not float
            or not math.isfinite(self.setup_cycles)
            or self.setup_cycles < 0.0
        ):
            raise SchemaError("must be a finite non-negative float", path=f"{path}.setup_cycles")

    @property
    def key(self) -> tuple[int, int, int, str, str]:
        return (
            self.m,
            self.n,
            self.k,
            self.dtype.value,
            self.accumulation_dtype.value,
        )


@dataclass(frozen=True, slots=True)
class SwizzleCalibrationProfile:
    schema_version: str
    id: str
    hardware_profile_ref: str
    matching_tool_sha256: str
    source_kind: str
    points: tuple[SwizzleCalibratedEfficiencyPoint, ...]

    @classmethod
    def create(
        cls,
        *,
        hardware_profile_ref: str,
        matching_tool_sha256: str,
        source_kind: str,
        points: tuple[SwizzleCalibratedEfficiencyPoint, ...],
    ) -> "SwizzleCalibrationProfile":
        semantic = {
            "hardware_profile_ref": hardware_profile_ref,
            "matching_tool_sha256": matching_tool_sha256,
            "source_kind": source_kind,
            "points": points,
        }
        result = cls(
            SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION,
            stable_artifact_id(
                "swizzle_calibration_profile",
                semantic,
                schema_version=SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "hardware_profile_ref": self.hardware_profile_ref,
            "matching_tool_sha256": self.matching_tool_sha256,
            "source_kind": self.source_kind,
            "points": self.points,
        }

    def validate(self, path: str = "swizzle_calibration_profile") -> None:
        if self.schema_version != SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.hardware_profile_ref, f"{path}.hardware_profile_ref")
        _digest(self.matching_tool_sha256, f"{path}.matching_tool_sha256")
        if self.source_kind not in ("npusim_microbenchmark", "hardware_profile", "analytic_provisional"):
            raise SchemaError("unsupported calibration source", path=f"{path}.source_kind")
        if type(self.points) is not tuple or len(self.points) < 2:
            raise SchemaError("requires at least two calibrated points", path=f"{path}.points")
        for index, point in enumerate(self.points):
            point.validate(f"{path}.points[{index}]")
        keys = tuple(point.key for point in self.points)
        if keys != tuple(sorted(set(keys))):
            raise SchemaError("points must be unique and canonically ordered", path=f"{path}.points")
        expected = stable_artifact_id(
            "swizzle_calibration_profile",
            self._semantic_key(),
            schema_version=SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleBranchTimingObservation:
    branch: SwizzleBenefitBranch
    algorithm: SwizzleAlgorithm
    economic_auto_selected: bool
    forced_deployment: bool
    chunk_count: int
    unroll_degree: int
    tile_shape: tuple[int, int, int] | None
    logical_bytes: int
    byte_hops: int
    gemm_flops: int
    record_count: int
    alloc_record_count: int
    free_record_count: int
    barrier_count: int
    event_record_count: int
    control_action_count: int
    sram_high_water_bytes: int
    observed_max_inflight_send: int
    observed_max_inflight_recv: int
    repeat_makespans: tuple[int, int]
    repeat_marker_digests: tuple[str, str]

    def validate(self, path: str = "swizzle_branch_timing") -> None:
        if type(self.branch) is not SwizzleBenefitBranch or type(self.algorithm) is not SwizzleAlgorithm:
            raise SchemaError("must use typed branch/algorithm", path=path)
        if type(self.economic_auto_selected) is not bool or type(self.forced_deployment) is not bool:
            raise SchemaError("selection flags must be bools", path=path)
        for name in (
            "chunk_count",
            "unroll_degree",
            "logical_bytes",
            "byte_hops",
            "gemm_flops",
            "record_count",
            "alloc_record_count",
            "free_record_count",
            "barrier_count",
            "event_record_count",
            "control_action_count",
            "sram_high_water_bytes",
            "observed_max_inflight_send",
            "observed_max_inflight_recv",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.logical_bytes == 0 or self.gemm_flops == 0 or self.record_count == 0:
            raise SchemaError("work and record counts must be positive", path=path)
        if self.alloc_record_count != self.free_record_count:
            raise SchemaError(
                "ALLOC and FREE record counts must be exact",
                path=path,
            )
        if self.branch is SwizzleBenefitBranch.NAIVE:
            if (
                self.algorithm is not SwizzleAlgorithm.UNFUSED
                or self.forced_deployment
                or self.chunk_count != 0
                or self.unroll_degree != 0
                or self.tile_shape is not None
            ):
                raise SchemaError("NAIVE must retain the executable UNFUSED baseline", path=path)
        else:
            if self.algorithm is SwizzleAlgorithm.UNFUSED:
                raise SchemaError("Swizzle branch must carry a fused algorithm", path=path)
            if self.chunk_count == 0 or self.unroll_degree not in (1, 2):
                raise SchemaError("Swizzle branch requires decomposition parameters", path=path)
            if (
                type(self.tile_shape) is not tuple
                or len(self.tile_shape) != 3
                or any(type(item) is not int or item <= 0 for item in self.tile_shape)
            ):
                raise SchemaError(
                    "Swizzle branch requires a positive M/N/K tile triple",
                    path=f"{path}.tile_shape",
                )
        if self.branch is SwizzleBenefitBranch.SWIZZLE_AUTO:
            if not self.economic_auto_selected or self.forced_deployment:
                raise SchemaError("AUTO must be economically selected and not forced", path=path)
        if self.branch is SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC:
            if not self.forced_deployment:
                raise SchemaError("forced diagnostic must retain the force fact", path=path)
        if (
            type(self.repeat_makespans) is not tuple
            or len(self.repeat_makespans) != 2
            or self.repeat_makespans[0] == 0
            or self.repeat_makespans[0] != self.repeat_makespans[1]
        ):
            raise SchemaError("two repeat makespans must be positive and exact", path=f"{path}.repeat_makespans")
        if (
            type(self.repeat_marker_digests) is not tuple
            or len(self.repeat_marker_digests) != 2
        ):
            raise SchemaError(
                "requires exactly two repeat marker digests",
                path=f"{path}.repeat_marker_digests",
            )
        for index, digest in enumerate(self.repeat_marker_digests):
            _digest(digest, f"{path}.repeat_marker_digests[{index}]")
        if self.repeat_marker_digests[0] != self.repeat_marker_digests[1]:
            raise SchemaError("repeat marker digests must be exact", path=f"{path}.repeat_marker_digests")


@dataclass(frozen=True, slots=True)
class SwizzleScaleBenefitEvidence:
    schema_version: str
    id: str
    scale_ref: str
    scale_ordinal: int
    pattern: FusionPattern
    same_work_digest: str
    naive: SwizzleBranchTimingObservation
    swizzle_auto: SwizzleBranchTimingObservation
    speedup: float
    qualifies: bool

    @classmethod
    def create(
        cls,
        *,
        scale_ref: str,
        scale_ordinal: int,
        pattern: FusionPattern,
        same_work_digest: str,
        naive: SwizzleBranchTimingObservation,
        swizzle_auto: SwizzleBranchTimingObservation,
    ) -> "SwizzleScaleBenefitEvidence":
        speedup = float(naive.repeat_makespans[0] / swizzle_auto.repeat_makespans[0])
        semantic = {
            "scale_ref": scale_ref,
            "scale_ordinal": scale_ordinal,
            "pattern": pattern,
            "same_work_digest": same_work_digest,
            "naive": naive,
            "swizzle_auto": swizzle_auto,
            "speedup": speedup,
            "qualifies": speedup >= SWIZZLE_MINIMUM_QUALIFYING_SPEEDUP,
        }
        result = cls(
            SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION,
            stable_artifact_id(
                "swizzle_scale_benefit",
                semantic,
                schema_version=SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION,
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

    def validate(self, path: str = "swizzle_scale_benefit") -> None:
        if self.schema_version != SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.scale_ref, f"{path}.scale_ref")
        validate_uint64(self.scale_ordinal, f"{path}.scale_ordinal")
        if type(self.pattern) is not FusionPattern:
            raise SchemaError("must use a FusionPattern", path=f"{path}.pattern")
        _digest(self.same_work_digest, f"{path}.same_work_digest")
        self.naive.validate(f"{path}.naive")
        self.swizzle_auto.validate(f"{path}.swizzle_auto")
        if self.naive.branch is not SwizzleBenefitBranch.NAIVE:
            raise SchemaError("naive observation has the wrong branch", path=f"{path}.naive")
        if self.swizzle_auto.branch is not SwizzleBenefitBranch.SWIZZLE_AUTO:
            raise SchemaError("auto observation has the wrong branch", path=f"{path}.swizzle_auto")
        if (
            self.naive.logical_bytes,
            self.naive.gemm_flops,
        ) != (
            self.swizzle_auto.logical_bytes,
            self.swizzle_auto.gemm_flops,
        ):
            raise SchemaError("NAIVE and AUTO must execute exactly the same work", path=path)
        expected_speedup = self.naive.repeat_makespans[0] / self.swizzle_auto.repeat_makespans[0]
        if type(self.speedup) is not float or not math.isclose(self.speedup, expected_speedup):
            raise SchemaError("speedup does not match actual makespans", path=f"{path}.speedup")
        if self.qualifies != (self.speedup >= SWIZZLE_MINIMUM_QUALIFYING_SPEEDUP):
            raise SchemaError("qualifying flag does not match the threshold", path=f"{path}.qualifies")
        expected = stable_artifact_id(
            "swizzle_scale_benefit",
            self._semantic_key(),
            schema_version=SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleBenefitReport:
    schema_version: str
    id: str
    calibration_profile_ref: str
    evidence: tuple[SwizzleScaleBenefitEvidence, ...]
    reproducible_speedup: bool

    @classmethod
    def create(
        cls,
        *,
        calibration_profile_ref: str,
        evidence: tuple[SwizzleScaleBenefitEvidence, ...],
    ) -> "SwizzleBenefitReport":
        qualified = {
            (item.pattern, item.scale_ordinal)
            for item in evidence
            if item.qualifies
        }
        reproducible = any(
            (pattern, ordinal + 1) in qualified
            for pattern, ordinal in qualified
        )
        semantic = {
            "calibration_profile_ref": calibration_profile_ref,
            "evidence": evidence,
            "reproducible_speedup": reproducible,
        }
        result = cls(
            SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION,
            stable_artifact_id(
                "swizzle_benefit_report",
                semantic,
                schema_version=SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "calibration_profile_ref": self.calibration_profile_ref,
            "evidence": self.evidence,
            "reproducible_speedup": self.reproducible_speedup,
        }

    def validate(self, path: str = "swizzle_benefit_report") -> None:
        if self.schema_version != SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.calibration_profile_ref, f"{path}.calibration_profile_ref")
        if type(self.evidence) is not tuple or not self.evidence:
            raise SchemaError("requires scale evidence", path=f"{path}.evidence")
        for index, item in enumerate(self.evidence):
            item.validate(f"{path}.evidence[{index}]")
        keys = tuple((item.pattern.value, item.scale_ordinal) for item in self.evidence)
        if keys != tuple(sorted(set(keys))):
            raise SchemaError("evidence must be unique and canonically ordered", path=f"{path}.evidence")
        qualified = {
            (item.pattern, item.scale_ordinal)
            for item in self.evidence
            if item.qualifies
        }
        expected_reproducible = any(
            (pattern, ordinal + 1) in qualified
            for pattern, ordinal in qualified
        )
        if type(self.reproducible_speedup) is not bool or self.reproducible_speedup != expected_reproducible:
            raise SchemaError("reproducible flag requires adjacent qualifying scales", path=f"{path}.reproducible_speedup")
        expected = stable_artifact_id(
            "swizzle_benefit_report",
            self._semantic_key(),
            schema_version=SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


__all__ = [
    "SWIZZLE_BENEFIT_EVIDENCE_SCHEMA_VERSION",
    "SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION",
    "SWIZZLE_MINIMUM_QUALIFYING_SPEEDUP",
    "SwizzleBenefitBranch",
    "SwizzleBenefitReport",
    "SwizzleBranchTimingObservation",
    "SwizzleCalibratedEfficiencyPoint",
    "SwizzleCalibrationProfile",
    "SwizzleScaleBenefitEvidence",
]
