"""Typed comparison and timing-evidence carriers for Swizzle W10/W11.

The planning carrier is buildable before production Swizzle lowering exists.
The runtime report is deliberately stricter: it can only be created from two
finalizer outputs and two simulator observations for both NAIVE and SWIZZLE.
It never carries a functional-execution claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import FusionPattern
from .program_io import ProgramIoMode
from .swizzle import SwizzleAlgorithm, SwizzleCost, SwizzleTensorAxis


SWIZZLE_COMPARISON_PLAN_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_comparison_plan/v1alpha1"
)
SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_runtime_comparison_report/v1alpha1"
)
SWIZZLE_RUNTIME_MARKER_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_runtime_markers/v1"
)

_DRAIN_NAMES = ("collective", "global", "p2p", "timing")


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _bool(value: bool, path: str) -> None:
    if type(value) is not bool:
        raise SchemaError("must be a bool", path=path)


def _refs(values: tuple[str, ...], path: str, *, nonempty: bool = False) -> None:
    if type(values) is not tuple or len(set(values)) != len(values):
        raise SchemaError("must be an immutable tuple of unique refs", path=path)
    if nonempty and not values:
        raise SchemaError("must be non-empty", path=path)
    for index, value in enumerate(values):
        validate_nonempty(value, f"{path}[{index}]")


class SwizzleComparisonBranch(str, Enum):
    NAIVE = "naive"
    SWIZZLE = "swizzle"


@dataclass(frozen=True, slots=True)
class SwizzleComparisonCasePlan:
    schema_version: str
    id: str
    case_id: str
    pattern: FusionPattern
    synthetic_placement: bool
    source_ir0_id: str
    source_ir0_digest: str
    placed_ir1_id: str
    placed_ir1_digest: str
    partitioned_ir1_id: str
    partitioned_ir1_digest: str
    fusion_candidate_refs: tuple[str, ...]
    naive_fusion_refs: tuple[str, ...]
    swizzle_fusion_refs: tuple[str, ...]
    skeleton_ref: str
    naive_policy_ref: str
    swizzle_policy_ref: str
    policy_configuration_digest: str
    decision_ref: str
    decision_digest: str
    deployment_selection_ref: str
    candidate_ref: str
    swizzle_plan_ref: str
    swizzle_plan_digest: str
    projection_ref: str
    projection_digest: str
    algorithm: SwizzleAlgorithm
    split_axis: SwizzleTensorAxis
    chunk_count: int
    unroll_degree: int
    baseline_cost: SwizzleCost
    selected_cost: SwizzleCost

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleComparisonCasePlan":
        result = cls(
            schema_version=SWIZZLE_COMPARISON_PLAN_SCHEMA_VERSION,
            id=stable_artifact_id(
                "swizzle_comparison_case",
                semantic,
                schema_version=SWIZZLE_COMPARISON_PLAN_SCHEMA_VERSION,
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

    def validate(self, path: str = "swizzle_comparison_case") -> None:
        if self.schema_version != SWIZZLE_COMPARISON_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in (
            "case_id", "source_ir0_id", "placed_ir1_id", "partitioned_ir1_id",
            "skeleton_ref", "naive_policy_ref", "swizzle_policy_ref",
            "decision_ref", "deployment_selection_ref", "candidate_ref",
            "swizzle_plan_ref", "projection_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in (
            "source_ir0_digest", "placed_ir1_digest", "partitioned_ir1_digest",
            "policy_configuration_digest", "decision_digest",
            "swizzle_plan_digest", "projection_digest",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        if type(self.pattern) is not FusionPattern:
            raise SchemaError("must be a FusionPattern", path=f"{path}.pattern")
        _bool(self.synthetic_placement, f"{path}.synthetic_placement")
        _refs(self.fusion_candidate_refs, f"{path}.fusion_candidate_refs", nonempty=True)
        _refs(self.naive_fusion_refs, f"{path}.naive_fusion_refs")
        _refs(self.swizzle_fusion_refs, f"{path}.swizzle_fusion_refs", nonempty=True)
        if type(self.algorithm) is not SwizzleAlgorithm or self.algorithm is SwizzleAlgorithm.UNFUSED:
            raise SchemaError("must select a fused Swizzle algorithm", path=f"{path}.algorithm")
        if type(self.split_axis) is not SwizzleTensorAxis:
            raise SchemaError("must carry a typed split axis", path=f"{path}.split_axis")
        self.split_axis.validate(f"{path}.split_axis")
        validate_uint64(self.chunk_count, f"{path}.chunk_count")
        validate_uint64(self.unroll_degree, f"{path}.unroll_degree")
        if self.chunk_count == 0 or self.unroll_degree not in (1, 2):
            raise SchemaError("invalid decomposition parameters", path=path)
        self.baseline_cost.validate(f"{path}.baseline_cost")
        self.selected_cost.validate(f"{path}.selected_cost")
        expected = stable_artifact_id(
            "swizzle_comparison_case",
            self._semantic_key(),
            schema_version=SWIZZLE_COMPARISON_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleComparisonSuitePlan:
    schema_version: str
    id: str
    cases: tuple[SwizzleComparisonCasePlan, ...]

    @classmethod
    def create(cls, cases: tuple[SwizzleComparisonCasePlan, ...]) -> "SwizzleComparisonSuitePlan":
        semantic = {"cases": cases}
        result = cls(
            SWIZZLE_COMPARISON_PLAN_SCHEMA_VERSION,
            stable_artifact_id(
                "swizzle_comparison_suite",
                semantic,
                schema_version=SWIZZLE_COMPARISON_PLAN_SCHEMA_VERSION,
            ),
            cases,
        )
        result.validate()
        return result

    def validate(self, path: str = "swizzle_comparison_suite") -> None:
        if self.schema_version != SWIZZLE_COMPARISON_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if tuple(item.pattern for item in self.cases) != (
            FusionPattern.AG_GEMM,
            FusionPattern.GEMM_RS,
            FusionPattern.GEMM_AR,
        ):
            raise SchemaError("must contain canonical AG/RS/AR cases", path=f"{path}.cases")
        if len({item.case_id for item in self.cases}) != 3:
            raise SchemaError("case ids must be unique", path=f"{path}.cases")
        for index, item in enumerate(self.cases):
            item.validate(f"{path}.cases[{index}]")
        expected = stable_artifact_id(
            "swizzle_comparison_suite",
            {"cases": self.cases},
            schema_version=SWIZZLE_COMPARISON_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleNamedCount:
    name: str
    count: int

    def validate(self, path: str) -> None:
        validate_nonempty(self.name, f"{path}.name")
        validate_uint64(self.count, f"{path}.count")


@dataclass(frozen=True, slots=True)
class SwizzleCoreCount:
    runtime_core_id: int
    count: int

    def validate(self, path: str) -> None:
        validate_uint64(self.runtime_core_id, f"{path}.runtime_core_id")
        validate_uint64(self.count, f"{path}.count")
        if self.runtime_core_id > 0xFFFF or self.count == 0:
            raise SchemaError("requires uint16 core and positive count", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleRuntimeControlEvidence:
    ack_counts: tuple[SwizzleCoreCount, ...]
    done_counts: tuple[SwizzleCoreCount, ...]
    drain_residuals: tuple[SwizzleNamedCount, ...]
    proto_wait_count: int
    all_done_boundary_reached: bool

    def validate(self, path: str = "swizzle_runtime_control") -> None:
        for name in ("ack_counts", "done_counts"):
            values = getattr(self, name)
            if not values or tuple(item.runtime_core_id for item in values) != tuple(
                sorted({item.runtime_core_id for item in values})
            ):
                raise SchemaError("must be non-empty, unique and canonical", path=f"{path}.{name}")
            for index, item in enumerate(values):
                item.validate(f"{path}.{name}[{index}]")
        if tuple(item.runtime_core_id for item in self.ack_counts) != tuple(
            item.runtime_core_id for item in self.done_counts
        ):
            raise SchemaError("ACK/DONE core coverage must be exact", path=path)
        if tuple(item.name for item in self.drain_residuals) != _DRAIN_NAMES:
            raise SchemaError("requires canonical drain classes", path=f"{path}.drain_residuals")
        for index, item in enumerate(self.drain_residuals):
            item.validate(f"{path}.drain_residuals[{index}]")
            if item.count != 0:
                raise SchemaError("all residuals must drain", path=f"{path}.drain_residuals[{index}]")
        validate_uint64(self.proto_wait_count, f"{path}.proto_wait_count")
        if self.proto_wait_count != 0:
            raise SchemaError("PROTO_WAIT must be absent", path=f"{path}.proto_wait_count")
        _bool(self.all_done_boundary_reached, f"{path}.all_done_boundary_reached")
        if not self.all_done_boundary_reached:
            raise SchemaError("DONE boundary must be reached", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleRuntimeToolEvidence:
    finalizer_sha256: str
    resolver_sha256: str
    npusim_sha256: str

    def validate(self, path: str = "swizzle_runtime_tools") -> None:
        for name in self.__dataclass_fields__:
            _digest(getattr(self, name), f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class SwizzleRuntimeArtifactEvidence:
    linked_manifest_id: str
    linked_manifest_digest: str
    program_artifact_sha256: str
    artifact_size_bytes: int
    finalizer_artifact_sha256s: tuple[str, str]
    finalizer_report_digests: tuple[str, str]

    def validate(self, path: str = "swizzle_runtime_artifact") -> None:
        validate_nonempty(self.linked_manifest_id, f"{path}.linked_manifest_id")
        for name in ("linked_manifest_digest", "program_artifact_sha256"):
            _digest(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.artifact_size_bytes, f"{path}.artifact_size_bytes")
        if self.artifact_size_bytes == 0:
            raise SchemaError("artifact must be non-empty", path=f"{path}.artifact_size_bytes")
        for name in ("finalizer_artifact_sha256s", "finalizer_report_digests"):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) != 2:
                raise SchemaError("requires exactly two finalizer observations", path=f"{path}.{name}")
            for index, value in enumerate(values):
                _digest(value, f"{path}.{name}[{index}]")
            if values[0] != values[1]:
                raise SchemaError("finalizer repeats must be byte exact", path=f"{path}.{name}")
        if self.finalizer_artifact_sha256s[0] != self.program_artifact_sha256:
            raise SchemaError("artifact SHA must equal both finalizer outputs", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleRuntimeProgramIoEvidence:
    contract_id: str
    contract_digest: str
    program_artifact_sha256: str
    mode: ProgramIoMode
    initialization_count: int
    probe_count: int
    all_probes_passed: bool

    def validate(self, path: str = "swizzle_runtime_program_io") -> None:
        validate_nonempty(self.contract_id, f"{path}.contract_id")
        _digest(self.contract_digest, f"{path}.contract_digest")
        _digest(self.program_artifact_sha256, f"{path}.program_artifact_sha256")
        if self.mode is not ProgramIoMode.TIMING:
            raise SchemaError("requires timing ProgramIo", path=f"{path}.mode")
        for name in ("initialization_count", "probe_count"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
            if getattr(self, name) == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")
        _bool(self.all_probes_passed, f"{path}.all_probes_passed")
        if not self.all_probes_passed:
            raise SchemaError("all probes must pass", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleRuntimeMetrics:
    logical_bytes: int
    byte_hops: int
    packet_count: int
    direction_port_utilization: float
    sram_high_water_bytes: int
    control_action_count: int

    def validate(self, path: str = "swizzle_runtime_metrics") -> None:
        for name in ("logical_bytes", "byte_hops", "packet_count", "sram_high_water_bytes", "control_action_count"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if type(self.direction_port_utilization) is not float or not 0.0 <= self.direction_port_utilization <= 1.0:
            raise SchemaError("must be a finite utilization fraction", path=f"{path}.direction_port_utilization")


@dataclass(frozen=True, slots=True)
class SwizzleRuntimeRepeatEvidence:
    run_index: int
    makespan_cycles: int
    marker_digest: str
    control_digest: str
    metrics_digest: str

    def validate(self, path: str = "swizzle_runtime_repeat") -> None:
        validate_uint64(self.run_index, f"{path}.run_index")
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if self.makespan_cycles == 0:
            raise SchemaError("makespan must be positive", path=f"{path}.makespan_cycles")
        for name in ("marker_digest", "control_digest", "metrics_digest"):
            _digest(getattr(self, name), f"{path}.{name}")

    def repeat_key(self) -> tuple[object, ...]:
        return (
            self.makespan_cycles,
            self.marker_digest,
            self.control_digest,
            self.metrics_digest,
        )


@dataclass(frozen=True, slots=True)
class SwizzleBranchRuntimeEvidence:
    branch: SwizzleComparisonBranch
    tools: SwizzleRuntimeToolEvidence
    artifact: SwizzleRuntimeArtifactEvidence
    program_io: SwizzleRuntimeProgramIoEvidence
    control: SwizzleRuntimeControlEvidence
    metrics: SwizzleRuntimeMetrics
    marker_schema_version: str
    repeats: tuple[SwizzleRuntimeRepeatEvidence, SwizzleRuntimeRepeatEvidence]

    def validate(self, path: str = "swizzle_branch_runtime") -> None:
        if type(self.branch) is not SwizzleComparisonBranch:
            raise SchemaError("must be a comparison branch", path=f"{path}.branch")
        self.tools.validate(f"{path}.tools")
        self.artifact.validate(f"{path}.artifact")
        self.program_io.validate(f"{path}.program_io")
        if self.program_io.program_artifact_sha256 != self.artifact.program_artifact_sha256:
            raise SchemaError("ProgramIo must bind the actual artifact SHA", path=f"{path}.program_io")
        self.control.validate(f"{path}.control")
        self.metrics.validate(f"{path}.metrics")
        if self.marker_schema_version != SWIZZLE_RUNTIME_MARKER_SCHEMA_VERSION:
            raise SchemaError("unsupported marker schema", path=f"{path}.marker_schema_version")
        if type(self.repeats) is not tuple or len(self.repeats) != 2:
            raise SchemaError("requires exactly two npusim repeats", path=f"{path}.repeats")
        for index, repeat in enumerate(self.repeats):
            repeat.validate(f"{path}.repeats[{index}]")
            if repeat.run_index != index:
                raise SchemaError("repeat indices must be canonical", path=f"{path}.repeats[{index}]")
        if self.repeats[0].repeat_key() != self.repeats[1].repeat_key():
            raise SchemaError("npusim repeats must be marker/makespan exact", path=f"{path}.repeats")


@dataclass(frozen=True, slots=True)
class SwizzleCaseRuntimeEvidence:
    case_plan_ref: str
    naive: SwizzleBranchRuntimeEvidence
    swizzle: SwizzleBranchRuntimeEvidence

    def validate(self, path: str = "swizzle_case_runtime") -> None:
        validate_nonempty(self.case_plan_ref, f"{path}.case_plan_ref")
        self.naive.validate(f"{path}.naive")
        self.swizzle.validate(f"{path}.swizzle")
        if self.naive.branch is not SwizzleComparisonBranch.NAIVE or self.swizzle.branch is not SwizzleComparisonBranch.SWIZZLE:
            raise SchemaError("requires ordered naive/swizzle branches", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleRuntimeComparisonReport:
    schema_version: str
    id: str
    suite: SwizzleComparisonSuitePlan
    cases: tuple[SwizzleCaseRuntimeEvidence, ...]
    timing_execution: bool
    functional_execution: bool

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleRuntimeComparisonReport":
        result = cls(
            schema_version=SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION,
            id=stable_artifact_id(
                "swizzle_runtime_comparison_report",
                semantic,
                schema_version=SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "suite": self.suite,
            "cases": self.cases,
            "timing_execution": self.timing_execution,
            "functional_execution": self.functional_execution,
        }

    def validate(self, path: str = "swizzle_runtime_comparison_report") -> None:
        if self.schema_version != SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.suite.validate(f"{path}.suite")
        if tuple(item.case_plan_ref for item in self.cases) != tuple(item.id for item in self.suite.cases):
            raise SchemaError("runtime evidence must exactly cover suite cases", path=f"{path}.cases")
        for index, item in enumerate(self.cases):
            item.validate(f"{path}.cases[{index}]")
        _bool(self.timing_execution, f"{path}.timing_execution")
        _bool(self.functional_execution, f"{path}.functional_execution")
        if self.timing_execution is not True or self.functional_execution is not False:
            raise SchemaError("report is timing=true, functional=false only", path=path)
        expected = stable_artifact_id(
            "swizzle_runtime_comparison_report",
            self._semantic_key(),
            schema_version=SWIZZLE_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


__all__ = [name for name in globals() if name.startswith("Swizzle") or name.startswith("SWIZZLE_")]
