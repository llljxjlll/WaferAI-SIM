"""Strict W11/W12 runtime evidence and benefit report for MoE Swizzle V2."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import (
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MoeCalibrationStatus,
    MoeSwizzleRuntimeMarkers,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_scale import (
    MoeSwizzleScaleRole,
)

from moe_swizzle_runtime_suite import (
    MoeSwizzleRuntimeBranch,
    MoeSwizzleRuntimeCasePlan,
    MoeSwizzleRuntimeScope,
    MoeSwizzleRuntimeSuitePlan,
)


MOE_SWIZZLE_RUNTIME_OBSERVATION_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_runtime_observation/v1alpha1"
)
MOE_SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_runtime_report/v1alpha1"
)
MOE_SWIZZLE_MINIMUM_SPEEDUP = 1.10

_DRAIN_NAMES = (
    "lsu", "dte", "p2p_endpoint", "p2p_timing", "collective", "router", "d2d_link",
)


def _digest(value: str, path: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleNamedCount:
    name: str
    count: int

    def validate(self, path: str = "moe_swizzle_named_count") -> None:
        validate_nonempty(self.name, f"{path}.name")
        validate_uint64(self.count, f"{path}.count")


@dataclass(frozen=True, slots=True)
class MoeSwizzleRuntimeControlClosure:
    expected_probe_count: int
    observed_probe_count: int
    expected_probe_bytes: int
    observed_probe_bytes: int
    expected_ack_count: int
    observed_ack_count: int
    expected_done_count: int
    observed_done_count: int
    expected_physical_root_count: int
    residuals: tuple[MoeSwizzleNamedCount, ...]
    proto_wait_count: int
    program_io_digest: str

    def validate(self, path: str = "moe_swizzle_runtime_control") -> None:
        for name in (
            "expected_probe_count", "observed_probe_count",
            "expected_probe_bytes", "observed_probe_bytes",
            "expected_ack_count", "observed_ack_count",
            "expected_done_count", "observed_done_count",
            "expected_physical_root_count", "proto_wait_count",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.expected_probe_count == 0
            or self.expected_probe_bytes == 0
            or self.expected_ack_count == 0
            or self.expected_done_count == 0
            or self.expected_physical_root_count == 0
        ):
            raise SchemaError("expected control counts must be positive", path=path)
        if tuple(item.name for item in self.residuals) != _DRAIN_NAMES:
            raise SchemaError("requires canonical residual classes", path=f"{path}.residuals")
        for index, item in enumerate(self.residuals):
            item.validate(f"{path}.residuals[{index}]")
        _digest(self.program_io_digest, f"{path}.program_io_digest")

    @property
    def complete(self) -> bool:
        return (
            self.observed_probe_count == self.expected_probe_count
            and self.observed_probe_bytes == self.expected_probe_bytes
            and self.observed_ack_count == self.expected_ack_count
            and self.observed_done_count == self.expected_done_count
            and all(item.count == 0 for item in self.residuals)
            and self.proto_wait_count == 0
        )


@dataclass(frozen=True, slots=True)
class MoeSwizzleRuntimeObservation:
    schema_version: str
    producer_pass: str
    id: str
    case_ref: str
    branch: MoeSwizzleRuntimeBranch
    same_work_digest: str
    deployment_digest: str
    workload_selection_ref: str | None
    source_workload_selection_id: str | None
    repeat_makespans: tuple[int, int]
    repeat_marker_digests: tuple[str, str]
    repeat_markers: tuple[MoeSwizzleRuntimeMarkers, MoeSwizzleRuntimeMarkers]
    finalizer_artifact_sha256: tuple[str, str]
    raw_stdout_sha256: tuple[str, str]
    tool_sha256: str
    hardware_sha256: str
    simulation_sha256: str
    mapping_sha256: str
    control: MoeSwizzleRuntimeControlClosure
    timing_execution: bool
    functional_execution: bool
    correctness_complete: bool
    measurement_complete: bool

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleRuntimeObservation":
        semantic = dict(semantic)
        repeats = semantic["repeat_makespans"]
        markers = semantic["repeat_markers"]
        artifacts = semantic["finalizer_artifact_sha256"]
        control = semantic["control"]
        timing = semantic["timing_execution"]
        functional = semantic["functional_execution"]
        correctness = (
            type(repeats) is tuple
            and len(repeats) == 2
            and repeats[0] > 0
            and repeats[0] == repeats[1]
            and type(markers) is tuple
            and len(markers) == 2
            and markers[0] == markers[1]
            and type(artifacts) is tuple
            and len(artifacts) == 2
            and artifacts[0] == artifacts[1]
            and type(control) is MoeSwizzleRuntimeControlClosure
            and control.complete
            and timing is True
            and functional is False
        )
        measurement = correctness and all(
            item.measurement_complete and not item.missing_measurements
            for item in markers
        )
        semantic["correctness_complete"] = correctness
        semantic["measurement_complete"] = measurement
        result = cls(
            MOE_SWIZZLE_RUNTIME_OBSERVATION_SCHEMA_VERSION,
            "moe_swizzle_runtime_runner",
            stable_artifact_id(
                "moe_swizzle_runtime_observation",
                semantic,
                schema_version=MOE_SWIZZLE_RUNTIME_OBSERVATION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "moe_swizzle_runtime_observation") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_RUNTIME_OBSERVATION_SCHEMA_VERSION
            or self.producer_pass != "moe_swizzle_runtime_runner"
        ):
            raise SchemaError("unsupported runtime observation schema/producer", path=path)
        validate_nonempty(self.case_ref, f"{path}.case_ref")
        if type(self.branch) is not MoeSwizzleRuntimeBranch:
            raise SchemaError("requires a typed runtime branch", path=f"{path}.branch")
        for name in (
            "same_work_digest", "deployment_digest", "tool_sha256",
            "hardware_sha256", "simulation_sha256", "mapping_sha256",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        for name in ("workload_selection_ref", "source_workload_selection_id"):
            value = getattr(self, name)
            if value is not None:
                validate_nonempty(value, f"{path}.{name}")
        if type(self.repeat_makespans) is not tuple or len(self.repeat_makespans) != 2:
            raise SchemaError("requires exactly two makespans", path=f"{path}.repeat_makespans")
        for index, value in enumerate(self.repeat_makespans):
            validate_uint64(value, f"{path}.repeat_makespans[{index}]")
        if self.repeat_makespans[0] == 0 or self.repeat_makespans[0] != self.repeat_makespans[1]:
            raise SchemaError("repeat makespans must be positive and exact", path=f"{path}.repeat_makespans")
        for name in (
            "repeat_marker_digests", "finalizer_artifact_sha256", "raw_stdout_sha256",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 2:
                raise SchemaError("requires exactly two repeat digests", path=f"{path}.{name}")
            for index, value in enumerate(values):
                _digest(value, f"{path}.{name}[{index}]")
        if self.repeat_marker_digests[0] != self.repeat_marker_digests[1]:
            raise SchemaError("repeat marker digests must be exact", path=f"{path}.repeat_marker_digests")
        if self.finalizer_artifact_sha256[0] != self.finalizer_artifact_sha256[1]:
            raise SchemaError("finalizer repeats must be byte exact", path=f"{path}.finalizer_artifact_sha256")
        if type(self.repeat_markers) is not tuple or len(self.repeat_markers) != 2:
            raise SchemaError("requires exactly two runtime marker carriers", path=f"{path}.repeat_markers")
        for index, marker in enumerate(self.repeat_markers):
            marker.validate(f"{path}.repeat_markers[{index}]")
            if marker.marker_digest != self.repeat_marker_digests[index]:
                raise SchemaError("marker carrier/digest drifted", path=f"{path}.repeat_markers[{index}]")
        if self.repeat_markers[0] != self.repeat_markers[1]:
            raise SchemaError("runtime marker repeats must be exact", path=f"{path}.repeat_markers")
        self.control.validate(f"{path}.control")
        for name in (
            "timing_execution", "functional_execution", "correctness_complete",
            "measurement_complete",
        ):
            if type(getattr(self, name)) is not bool:
                raise SchemaError("must be bool", path=f"{path}.{name}")
        expected_correctness = (
            self.repeat_makespans[0] == self.repeat_makespans[1]
            and self.finalizer_artifact_sha256[0] == self.finalizer_artifact_sha256[1]
            and self.repeat_markers[0] == self.repeat_markers[1]
            and self.control.complete
            and self.timing_execution
            and not self.functional_execution
        )
        expected_measurement = expected_correctness and all(
            item.measurement_complete and not item.missing_measurements
            for item in self.repeat_markers
        )
        if (
            self.correctness_complete != expected_correctness
            or self.measurement_complete != expected_measurement
        ):
            raise SchemaError("runtime completion flags drifted", path=path)
        if self.control.expected_physical_root_count != self.repeat_markers[0].physical_root_count:
            raise SchemaError("manifest/runtime physical root closure drifted", path=f"{path}.control")
        expected = stable_artifact_id(
            "moe_swizzle_runtime_observation",
            self._semantic(),
            schema_version=MOE_SWIZZLE_RUNTIME_OBSERVATION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable runtime observation id", path=f"{path}.id")

    def validate_against(
        self,
        case: MoeSwizzleRuntimeCasePlan,
        path: str = "moe_swizzle_runtime_observation",
    ) -> None:
        self.validate(path)
        branch = next((item for item in case.branches if item.branch is self.branch), None)
        if (
            self.case_ref != case.id
            or branch is None
            or self.same_work_digest != canonical_digest(case.same_work)
            or self.deployment_digest != canonical_digest(branch)
            or self.workload_selection_ref != branch.workload_selection_ref
            or self.source_workload_selection_id != branch.workload_selection_ref
        ):
            raise SchemaError("observation does not belong to case/branch", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleRuntimePairEvidence:
    case_ref: str
    scale_name: str
    scale_role: MoeSwizzleScaleRole
    execution_mode: MoeScaleExecutionMode
    scope: MoeSwizzleRuntimeScope
    target_pattern: FusionPattern | None
    naive_observation_ref: str
    auto_observation_ref: str
    speedup: float
    threshold_met: bool
    runtime_overlap_met: bool
    benefit: bool

    @classmethod
    def create(
        cls,
        case: MoeSwizzleRuntimeCasePlan,
        naive: MoeSwizzleRuntimeObservation,
        auto: MoeSwizzleRuntimeObservation,
    ) -> "MoeSwizzleRuntimePairEvidence":
        naive.validate_against(case, "moe_runtime_pair.naive")
        auto.validate_against(case, "moe_runtime_pair.auto")
        if (
            naive.branch is not MoeSwizzleRuntimeBranch.NAIVE
            or auto.branch is not MoeSwizzleRuntimeBranch.SWIZZLE_AUTO
            or naive.same_work_digest != auto.same_work_digest
            or (naive.hardware_sha256, naive.simulation_sha256, naive.mapping_sha256)
            != (auto.hardware_sha256, auto.simulation_sha256, auto.mapping_sha256)
        ):
            raise SchemaError("NAIVE/AUTO pair is not exact same-work/config", path="moe_runtime_pair")
        speedup = naive.repeat_makespans[0] / auto.repeat_makespans[0]
        marker = auto.repeat_markers[0]
        runtime_overlap = (
            marker.observed_max_inflight_send is not None
            and marker.observed_max_inflight_recv is not None
            and marker.observed_max_inflight_send >= 2
            and marker.observed_max_inflight_recv >= 2
            and marker.compute_dte_overlap_cycles is not None
            and marker.compute_dte_overlap_cycles > 0
        )
        threshold = speedup >= MOE_SWIZZLE_MINIMUM_SPEEDUP
        benefit = (
            naive.measurement_complete
            and auto.measurement_complete
            and threshold
            and runtime_overlap
        )
        return cls(
            case.id,
            case.spec.name,
            case.spec.role,
            case.execution.mode,
            case.scope,
            case.target_pattern,
            naive.id,
            auto.id,
            speedup,
            threshold,
            runtime_overlap,
            benefit,
        )

    def validate(self, path: str = "moe_swizzle_runtime_pair") -> None:
        for name in ("case_ref", "scale_name", "naive_observation_ref", "auto_observation_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.scale_role) is not MoeSwizzleScaleRole:
            raise SchemaError("requires typed scale role", path=f"{path}.scale_role")
        if type(self.execution_mode) is not MoeScaleExecutionMode:
            raise SchemaError("requires typed mode", path=f"{path}.execution_mode")
        if type(self.scope) is not MoeSwizzleRuntimeScope:
            raise SchemaError("requires typed scope", path=f"{path}.scope")
        if self.scope is MoeSwizzleRuntimeScope.REGION_PREFLIGHT:
            if self.target_pattern not in (
                FusionPattern.MOE_DISPATCH_GEMM,
                FusionPattern.MOE_GEMM_COMBINE,
            ):
                raise SchemaError("preflight pair requires a target pattern", path=path)
        elif self.target_pattern is not None:
            raise SchemaError("workload pair forbids target pattern", path=path)
        if type(self.speedup) is not float or not math.isfinite(self.speedup) or self.speedup <= 0.0:
            raise SchemaError("speedup must be finite positive", path=f"{path}.speedup")
        for name in ("threshold_met", "runtime_overlap_met", "benefit"):
            if type(getattr(self, name)) is not bool:
                raise SchemaError("must be bool", path=f"{path}.{name}")
        if self.threshold_met != (self.speedup >= MOE_SWIZZLE_MINIMUM_SPEEDUP):
            raise SchemaError("speedup threshold flag drifted", path=f"{path}.threshold_met")
        if self.benefit and (not self.threshold_met or not self.runtime_overlap_met):
            raise SchemaError("benefit cannot bypass threshold/runtime gates", path=f"{path}.benefit")


@dataclass(frozen=True, slots=True)
class MoeSwizzleRuntimeBenefitReport:
    schema_version: str
    producer_pass: str
    id: str
    suite: MoeSwizzleRuntimeSuitePlan
    observations: tuple[MoeSwizzleRuntimeObservation, ...]
    pairs: tuple[MoeSwizzleRuntimePairEvidence, ...]
    correctness_complete: bool
    measurement_complete: bool
    performance_benefit: bool
    performance_complete: bool
    v2_complete: bool

    @classmethod
    def create(
        cls,
        suite: MoeSwizzleRuntimeSuitePlan,
        observations: tuple[MoeSwizzleRuntimeObservation, ...],
    ) -> "MoeSwizzleRuntimeBenefitReport":
        suite.validate("moe_runtime_report.suite")
        by_key = {(item.case_ref, item.branch): item for item in observations}
        if len(by_key) != len(observations):
            raise SchemaError("runtime report duplicates observations", path="moe_runtime_report.observations")
        expected = {
            (case.id, branch.branch)
            for case in suite.cases for branch in case.branches
        }
        if set(by_key) != expected:
            raise SchemaError("runtime report observation matrix is incomplete", path="moe_runtime_report.observations")
        ordered = tuple(
            by_key[(case.id, branch.branch)]
            for case in suite.cases for branch in case.branches
        )
        for case in suite.cases:
            for branch in case.branches:
                by_key[(case.id, branch.branch)].validate_against(
                    case, "moe_runtime_report.observation"
                )
        pairs = tuple(
            MoeSwizzleRuntimePairEvidence.create(
                case,
                by_key[(case.id, MoeSwizzleRuntimeBranch.NAIVE)],
                by_key[(case.id, MoeSwizzleRuntimeBranch.SWIZZLE_AUTO)],
            )
            for case in suite.cases
        )
        correctness = all(item.correctness_complete for item in ordered)
        measurement = (
            correctness
            and suite.calibration_profile.status is MoeCalibrationStatus.MEASURED
            and all(item.measurement_complete for item in ordered)
        )
        region_benefit = {
            item.target_pattern
            for item in pairs
            if item.scope is MoeSwizzleRuntimeScope.REGION_PREFLIGHT and item.benefit
        } == {
            FusionPattern.MOE_DISPATCH_GEMM,
            FusionPattern.MOE_GEMM_COMBINE,
        }
        workload_benefit = all(
            {
                item.scale_name
                for item in pairs
                if item.scope is MoeSwizzleRuntimeScope.WORKLOAD
                and item.execution_mode is mode
                and item.scale_role is MoeSwizzleScaleRole.VALIDATION
                and item.benefit
            }
            >= {"C2", "C3"}
            for mode in (
                MoeScaleExecutionMode.INFER_FORWARD,
                MoeScaleExecutionMode.TRAIN_FORWARD,
            )
        )
        performance = measurement and region_benefit and workload_benefit
        semantic = {
            "suite": suite,
            "observations": ordered,
            "pairs": pairs,
            "correctness_complete": correctness,
            "measurement_complete": measurement,
            "performance_benefit": performance,
            "performance_complete": performance,
            "v2_complete": correctness and measurement and performance,
        }
        result = cls(
            MOE_SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION,
            "build_moe_swizzle_runtime_benefit_report",
            stable_artifact_id(
                "moe_swizzle_runtime_report",
                semantic,
                schema_version=MOE_SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "moe_swizzle_runtime_report") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION
            or self.producer_pass != "build_moe_swizzle_runtime_benefit_report"
        ):
            raise SchemaError("unsupported runtime report schema/producer", path=path)
        self.suite.validate(f"{path}.suite")
        for index, item in enumerate(self.observations):
            item.validate(f"{path}.observations[{index}]")
        for index, item in enumerate(self.pairs):
            item.validate(f"{path}.pairs[{index}]")
        if len(self.observations) != 3 * len(self.suite.cases) or len(self.pairs) != len(self.suite.cases):
            raise SchemaError("report matrix cardinality drifted", path=path)
        expected_pairs = tuple(
            MoeSwizzleRuntimePairEvidence.create(
                case,
                next(item for item in self.observations if item.case_ref == case.id and item.branch is MoeSwizzleRuntimeBranch.NAIVE),
                next(item for item in self.observations if item.case_ref == case.id and item.branch is MoeSwizzleRuntimeBranch.SWIZZLE_AUTO),
            )
            for case in self.suite.cases
        )
        if self.pairs != expected_pairs:
            raise SchemaError("report pairs do not rebuild observations", path=f"{path}.pairs")
        correctness = all(item.correctness_complete for item in self.observations)
        measurement = correctness and all(item.measurement_complete for item in self.observations)
        region_benefit = {
            item.target_pattern for item in self.pairs
            if item.scope is MoeSwizzleRuntimeScope.REGION_PREFLIGHT and item.benefit
        } == {FusionPattern.MOE_DISPATCH_GEMM, FusionPattern.MOE_GEMM_COMBINE}
        workload_benefit = all(
            {item.scale_name for item in self.pairs if (
                item.scope is MoeSwizzleRuntimeScope.WORKLOAD
                and item.execution_mode is mode
                and item.scale_role is MoeSwizzleScaleRole.VALIDATION
                and item.benefit
            )} >= {"C2", "C3"}
            for mode in (MoeScaleExecutionMode.INFER_FORWARD, MoeScaleExecutionMode.TRAIN_FORWARD)
        )
        performance = measurement and region_benefit and workload_benefit
        expected_flags = (
            correctness,
            measurement,
            performance,
            performance,
            correctness and measurement and performance,
        )
        actual_flags = (
            self.correctness_complete,
            self.measurement_complete,
            self.performance_benefit,
            self.performance_complete,
            self.v2_complete,
        )
        if actual_flags != expected_flags:
            raise SchemaError("report completion flags drifted", path=path)
        expected = stable_artifact_id(
            "moe_swizzle_runtime_report",
            self._semantic(),
            schema_version=MOE_SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable runtime report id", path=f"{path}.id")


__all__ = [
    "MOE_SWIZZLE_MINIMUM_SPEEDUP",
    "MoeSwizzleNamedCount",
    "MoeSwizzleRuntimeBenefitReport",
    "MoeSwizzleRuntimeControlClosure",
    "MoeSwizzleRuntimeObservation",
    "MoeSwizzleRuntimePairEvidence",
]
