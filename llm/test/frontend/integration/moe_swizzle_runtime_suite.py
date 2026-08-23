"""Typed W11/W12 MoE Swizzle runtime suite planning.

This module only consumes production C0-C4 truth and production planner
objects.  It never derives bytes or FLOPs from runner-local constants.  The
economic decisions remain immutable; forced coverage is represented by an
explicit deployment selection and is never rewritten into an AUTO decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.discover_moe_swizzle import (
    discover_moe_swizzle_regions,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_comet_mesh import (
    build_comet_mesh_moe_candidate_grid,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import (
    decide_moe_swizzle,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_direct_xy import (
    build_direct_xy_moe_candidates,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_unfused import (
    build_executable_moe_unfused_baseline,
)
from llm.frontend.wafer_frontend.schema.common import (
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe import (
    MoeSwizzleDecision,
    MoeSwizzleWorkloadSelection,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MoeCalibrationKind,
    MoeCalibrationStatus,
    MoeSwizzleCalibrationProfile,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecution,
    MoeScaleExecutionMode,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_plan import (
    MoeSwizzleDeploymentMode,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_scale import (
    MoeSwizzleScaleOracle,
    MoeSwizzleScaleRole,
    MoeSwizzleScaleSpec,
)

from moe_swizzle_scale_cases import (
    MoeSwizzleScaleCase,
    build_moe_swizzle_scale_cases,
)


MOE_SWIZZLE_RUNTIME_SUITE_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_runtime_suite/v1alpha1"
)
MOE_SWIZZLE_RUNTIME_CASE_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_runtime_case/v1alpha1"
)
MOE_SWIZZLE_SAME_WORK_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_same_work/v1alpha1"
)

_PATTERNS = (
    FusionPattern.MOE_DISPATCH_GEMM,
    FusionPattern.MOE_GEMM_COMBINE,
)


def _candidate_comp_shapes(
    decision: MoeSwizzleDecision, candidate_ref: str
) -> tuple[tuple[int, int, int], ...]:
    candidate = next(
        (item for item in decision.ranked_candidates if item.id == candidate_ref),
        None,
    )
    if candidate is None:
        raise SchemaError(
            "deployed candidate is absent from decision",
            path="moe_swizzle_runtime_case.deployed_comp_shapes",
        )
    gemms = decision.problem.region.semantic_witness.traffic.expert_gemms
    shapes = {
        (
            len(action.assignment_refs),
            gemms[action.expert_index].n,
            gemms[action.expert_index].k,
        )
        for program in candidate.rank_programs
        for action in program.actions
        if action.kind.value == "comp"
    }
    if not shapes:
        raise SchemaError(
            "deployed candidate has no GroupGEMM COMP shape",
            path="moe_swizzle_runtime_case.deployed_comp_shapes",
        )
    return tuple(sorted(shapes))


def _candidate_swiglu_group_shapes(
    decision: MoeSwizzleDecision, candidate_ref: str
) -> tuple[tuple[int, int, int], ...]:
    candidate = next(
        (item for item in decision.ranked_candidates if item.id == candidate_ref),
        None,
    )
    if candidate is None:
        raise SchemaError(
            "deployed candidate is absent from decision",
            path="moe_swizzle_runtime_case.deployed_swiglu_group_shapes",
        )
    gemms = decision.problem.region.semantic_witness.traffic.expert_gemms
    shapes = {
        (
            len(action.assignment_refs),
            gemms[action.expert_index].n,
            len(action.assignment_refs) * gemms[action.expert_index].n,
        )
        for program in candidate.rank_programs
        for action in program.actions
        if action.kind is SwizzleActionKind.SWIGLU
    }
    return tuple(sorted(shapes))


class MoeSwizzleRuntimeScope(str, Enum):
    REGION_PREFLIGHT = "region_preflight"
    WORKLOAD = "workload"


class MoeSwizzleRuntimeBranch(str, Enum):
    NAIVE = "naive"
    SWIZZLE_AUTO = "swizzle_auto"
    SWIZZLE_FORCED_COVERAGE = "swizzle_forced_coverage"


@dataclass(frozen=True, slots=True)
class MoeSwizzleSameWorkOracle:
    schema_version: str
    producer_pass: str
    id: str
    source_execution_id: str
    scale_name: str
    scale_role: MoeSwizzleScaleRole
    execution_mode: MoeScaleExecutionMode
    trace_digest: str
    assignment_set_digest: str
    contributor_set_digest: str
    assignment_count: int
    contributor_count: int
    dispatch_logical_bytes: int
    combine_logical_bytes: int
    expert_group_gemm_flops: int
    combined_terminal_count: int
    combined_terminal_bytes: int
    tape_terminal_count: int
    tape_terminal_bytes: int
    terminal_set_digest: str

    @classmethod
    def create(
        cls,
        spec: MoeSwizzleScaleSpec,
        oracle: MoeSwizzleScaleOracle,
        execution: MoeScaleExecution,
    ) -> "MoeSwizzleSameWorkOracle":
        execution.validate_against(spec, oracle, "moe_same_work.execution")
        tape = execution.mode is MoeScaleExecutionMode.TRAIN_FORWARD
        semantic = {
            "source_execution_id": execution.id,
            "scale_name": spec.name,
            "scale_role": spec.role,
            "execution_mode": execution.mode,
            "trace_digest": oracle.trace_digest,
            "assignment_set_digest": canonical_digest(spec.trace.assignments),
            "contributor_set_digest": canonical_digest(
                (spec.trace.assignments, spec.top_k)
            ),
            "assignment_count": oracle.assignment_count,
            "contributor_count": oracle.contributor_count,
            "dispatch_logical_bytes": oracle.dispatch_logical_bytes,
            "combine_logical_bytes": oracle.combine_logical_bytes,
            "expert_group_gemm_flops": oracle.total_expert_gemm_flops,
            "combined_terminal_count": oracle.combined_terminal_count,
            "combined_terminal_bytes": oracle.combined_terminal_bytes,
            "tape_terminal_count": oracle.train_tape_terminal_count if tape else 0,
            "tape_terminal_bytes": oracle.train_tape_terminal_bytes if tape else 0,
            "terminal_set_digest": canonical_digest(execution.terminals),
        }
        result = cls(
            MOE_SWIZZLE_SAME_WORK_SCHEMA_VERSION,
            "build_moe_swizzle_same_work_oracle",
            stable_artifact_id(
                "moe_swizzle_same_work",
                semantic,
                schema_version=MOE_SWIZZLE_SAME_WORK_SCHEMA_VERSION,
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

    def validate(self, path: str = "moe_swizzle_same_work") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_SAME_WORK_SCHEMA_VERSION
            or self.producer_pass != "build_moe_swizzle_same_work_oracle"
        ):
            raise SchemaError("unsupported same-work schema/producer", path=path)
        for name in (
            "source_execution_id", "scale_name", "trace_digest",
            "assignment_set_digest", "contributor_set_digest",
            "terminal_set_digest",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.scale_role) is not MoeSwizzleScaleRole:
            raise SchemaError("requires a typed scale role", path=f"{path}.scale_role")
        if type(self.execution_mode) is not MoeScaleExecutionMode:
            raise SchemaError("requires a typed execution mode", path=f"{path}.execution_mode")
        for name in (
            "assignment_count", "contributor_count", "dispatch_logical_bytes",
            "combine_logical_bytes", "expert_group_gemm_flops",
            "combined_terminal_count", "combined_terminal_bytes",
            "tape_terminal_count", "tape_terminal_bytes",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.assignment_count == 0
            or self.contributor_count == 0
            or self.expert_group_gemm_flops == 0
            or self.combined_terminal_count == 0
            or self.combined_terminal_bytes == 0
        ):
            raise SchemaError("same-work production totals must be positive", path=path)
        tape = self.execution_mode is MoeScaleExecutionMode.TRAIN_FORWARD
        if tape != (self.tape_terminal_count > 0 and self.tape_terminal_bytes > 0):
            raise SchemaError("tape work must match execution mode exactly", path=path)
        expected = stable_artifact_id(
            "moe_swizzle_same_work",
            self._semantic(),
            schema_version=MOE_SWIZZLE_SAME_WORK_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable same-work id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeSwizzleRegionDeployment:
    pattern: FusionPattern
    decision_ref: str
    candidate_ref: str
    algorithm: SwizzleAlgorithm
    mode: MoeSwizzleDeploymentMode
    economic_selected: bool

    def validate(self, path: str = "moe_swizzle_region_deployment") -> None:
        if self.pattern not in _PATTERNS:
            raise SchemaError("unsupported MoE region pattern", path=f"{path}.pattern")
        validate_nonempty(self.decision_ref, f"{path}.decision_ref")
        validate_nonempty(self.candidate_ref, f"{path}.candidate_ref")
        if type(self.algorithm) is not SwizzleAlgorithm:
            raise SchemaError("requires a typed algorithm", path=f"{path}.algorithm")
        if type(self.mode) is not MoeSwizzleDeploymentMode:
            raise SchemaError("requires a typed deployment mode", path=f"{path}.mode")
        if type(self.economic_selected) is not bool:
            raise SchemaError("economic_selected must be bool", path=f"{path}.economic_selected")
        if (self.algorithm is SwizzleAlgorithm.UNFUSED) != (
            self.mode is MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE
        ):
            raise SchemaError("algorithm/deployment mode mismatch", path=path)
        if self.mode is MoeSwizzleDeploymentMode.FUSED_FORCED and self.economic_selected:
            raise SchemaError("forced coverage cannot be economic", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleRuntimeBranchPlan:
    branch: MoeSwizzleRuntimeBranch
    same_work_digest: str
    workload_selection_ref: str | None
    deployments: tuple[MoeSwizzleRegionDeployment, MoeSwizzleRegionDeployment]

    def validate(self, path: str = "moe_swizzle_runtime_branch") -> None:
        if type(self.branch) is not MoeSwizzleRuntimeBranch:
            raise SchemaError("requires a typed branch", path=f"{path}.branch")
        validate_nonempty(self.same_work_digest, f"{path}.same_work_digest")
        if self.workload_selection_ref is not None:
            validate_nonempty(
                self.workload_selection_ref, f"{path}.workload_selection_ref"
            )
        if tuple(item.pattern for item in self.deployments) != _PATTERNS:
            raise SchemaError(
                "deployments must use canonical pattern order",
                path=f"{path}.deployments",
            )
        for index, deployment in enumerate(self.deployments):
            deployment.validate(f"{path}.deployments[{index}]")
        if self.branch is MoeSwizzleRuntimeBranch.NAIVE and any(
            item.algorithm is not SwizzleAlgorithm.UNFUSED
            or item.mode is not MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE
            or item.economic_selected
            for item in self.deployments
        ):
            raise SchemaError("NAIVE requires two non-economic executable baselines", path=path)

    def validate_against(
        self,
        case: "MoeSwizzleRuntimeCasePlan",
        path: str = "moe_swizzle_runtime_branch",
    ) -> None:
        self.validate(path)
        if self.same_work_digest != canonical_digest(case.same_work):
            raise SchemaError(
                "branch same-work digest drifted", path=f"{path}.same_work_digest"
            )
        decisions = {item.problem.region.pattern: item for item in case.economic_decisions}
        for index, deployment in enumerate(self.deployments):
            decision = decisions[deployment.pattern]
            candidates = {item.id: item for item in decision.ranked_candidates}
            candidate = candidates.get(deployment.candidate_ref)
            if (
                deployment.decision_ref != decision.id
                or candidate is None
                or candidate.algorithm is not deployment.algorithm
            ):
                raise SchemaError(
                    "deployment is not closed over economic decision",
                    path=f"{path}.deployments[{index}]",
                )
        target = case.target_pattern
        if self.branch is MoeSwizzleRuntimeBranch.NAIVE:
            if self.workload_selection_ref is not None or any(
                item.algorithm is not SwizzleAlgorithm.UNFUSED
                or item.mode is not MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE
                or item.economic_selected
                for item in self.deployments
            ):
                raise SchemaError("NAIVE requires two executable baselines", path=path)
        elif self.branch is MoeSwizzleRuntimeBranch.SWIZZLE_AUTO:
            selection = case.workload_selection
            if case.scope is MoeSwizzleRuntimeScope.WORKLOAD:
                if (
                    selection is None
                    or self.workload_selection_ref != selection.id
                ):
                    raise SchemaError(
                        "WORKLOAD AUTO requires exact typed joint selection lineage",
                        path=f"{path}.workload_selection_ref",
                    )
                selected_refs = {
                    FusionPattern.MOE_DISPATCH_GEMM:
                        selection.selected_dispatch_candidate_ref,
                    FusionPattern.MOE_GEMM_COMBINE:
                        selection.selected_combine_candidate_ref,
                }
            else:
                if self.workload_selection_ref is not None:
                    raise SchemaError(
                        "REGION_PREFLIGHT AUTO forbids workload selection lineage",
                        path=f"{path}.workload_selection_ref",
                    )
                selected_refs = {}
            for item in self.deployments:
                decision = decisions[item.pattern]
                should_economic = (
                    case.scope is MoeSwizzleRuntimeScope.WORKLOAD
                    or item.pattern is target
                )
                expected_ref = (
                    selected_refs[item.pattern]
                    if case.scope is MoeSwizzleRuntimeScope.WORKLOAD
                    else decision.selected_candidate_ref
                    if should_economic
                    else decision.baseline.id
                )
                if (
                    item.candidate_ref != expected_ref
                    or item.economic_selected != should_economic
                ):
                    raise SchemaError("AUTO/economic region isolation drifted", path=path)
                if not should_economic and item.algorithm is not SwizzleAlgorithm.UNFUSED:
                    raise SchemaError(
                        "preflight non-target region must remain baseline", path=path
                    )
                expected_mode = (
                    MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE
                    if item.algorithm is SwizzleAlgorithm.UNFUSED
                    else MoeSwizzleDeploymentMode.FUSED_AUTO
                )
                if item.mode is not expected_mode:
                    raise SchemaError("AUTO deployment mode drifted", path=path)
        else:
            if self.workload_selection_ref is not None:
                raise SchemaError(
                    "forced coverage forbids workload selection lineage",
                    path=f"{path}.workload_selection_ref",
                )
            selection = case.workload_selection
            joint_refs = (
                {
                    FusionPattern.MOE_DISPATCH_GEMM:
                        selection.selected_dispatch_candidate_ref,
                    FusionPattern.MOE_GEMM_COMBINE:
                        selection.selected_combine_candidate_ref,
                }
                if selection is not None else {}
            )
            for item in self.deployments:
                decision = decisions[item.pattern]
                should_force = (
                    case.scope is MoeSwizzleRuntimeScope.WORKLOAD
                    or item.pattern is target
                )
                excluded_ref = (
                    joint_refs[item.pattern]
                    if case.scope is MoeSwizzleRuntimeScope.WORKLOAD
                    else decision.selected_candidate_ref
                )
                if should_force:
                    if (
                        item.algorithm is SwizzleAlgorithm.UNFUSED
                        or item.mode is not MoeSwizzleDeploymentMode.FUSED_FORCED
                        or item.economic_selected
                        or item.candidate_ref == excluded_ref
                    ):
                        raise SchemaError(
                            "forced coverage must select a non-joint-economic fused candidate",
                            path=path,
                        )
                elif (
                    item.algorithm is not SwizzleAlgorithm.UNFUSED
                    or item.mode is not MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE
                    or item.economic_selected
                ):
                    raise SchemaError(
                        "preflight non-target region must remain baseline", path=path
                    )


@dataclass(frozen=True, slots=True)
class MoeSwizzleRuntimeCasePlan:
    schema_version: str
    producer_pass: str
    id: str
    scope: MoeSwizzleRuntimeScope
    target_pattern: FusionPattern | None
    spec: MoeSwizzleScaleSpec
    oracle: MoeSwizzleScaleOracle
    execution: MoeScaleExecution
    economic_decisions: tuple[MoeSwizzleDecision, MoeSwizzleDecision]
    workload_selection: MoeSwizzleWorkloadSelection | None
    same_work: MoeSwizzleSameWorkOracle
    deployed_comp_shapes: tuple[tuple[int, int, int], ...]
    deployed_swiglu_group_shapes: tuple[tuple[int, int, int], ...]
    branches: tuple[
        MoeSwizzleRuntimeBranchPlan,
        MoeSwizzleRuntimeBranchPlan,
        MoeSwizzleRuntimeBranchPlan,
    ]

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleRuntimeCasePlan":
        result = cls(
            MOE_SWIZZLE_RUNTIME_CASE_SCHEMA_VERSION,
            "build_moe_swizzle_runtime_case",
            stable_artifact_id(
                "moe_swizzle_runtime_case",
                semantic,
                schema_version=MOE_SWIZZLE_RUNTIME_CASE_SCHEMA_VERSION,
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

    def validate(self, path: str = "moe_swizzle_runtime_case") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_RUNTIME_CASE_SCHEMA_VERSION
            or self.producer_pass != "build_moe_swizzle_runtime_case"
        ):
            raise SchemaError("unsupported runtime case schema/producer", path=path)
        if type(self.scope) is not MoeSwizzleRuntimeScope:
            raise SchemaError("requires a typed runtime scope", path=f"{path}.scope")
        if self.scope is MoeSwizzleRuntimeScope.REGION_PREFLIGHT:
            if self.target_pattern not in _PATTERNS or self.spec.name != "C2":
                raise SchemaError("region preflight is frozen to one C2 target pattern", path=path)
            if self.execution.mode is not MoeScaleExecutionMode.INFER_FORWARD:
                raise SchemaError("region preflight uses inference execution", path=f"{path}.execution")
            if self.workload_selection is not None:
                raise SchemaError(
                    "region preflight forbids workload joint selection",
                    path=f"{path}.workload_selection",
                )
        else:
            if self.target_pattern is not None:
                raise SchemaError("workload case forbids a target region", path=f"{path}.target_pattern")
            if type(self.workload_selection) is not MoeSwizzleWorkloadSelection:
                raise SchemaError(
                    "workload case requires exact typed joint selection",
                    path=f"{path}.workload_selection",
                )
        self.execution.validate_against(self.spec, self.oracle, f"{path}.execution")
        if not self.execution.execution_ready:
            raise SchemaError("runtime case requires admitted execution truth", path=f"{path}.execution")
        patterns = tuple(item.problem.region.pattern for item in self.economic_decisions)
        if patterns != _PATTERNS:
            raise SchemaError("economic decisions must use canonical pattern order", path=f"{path}.economic_decisions")
        for index, decision in enumerate(self.economic_decisions):
            decision.validate(f"{path}.economic_decisions[{index}]")
            if (
                decision.problem.source_execution_id != self.execution.id
                or not decision.performance_complete
                or not all(item.cost.calibrated for item in decision.ranked_candidates)
            ):
                raise SchemaError("runtime decision must be calibrated and execution-bound", path=f"{path}.economic_decisions[{index}]")
        if self.workload_selection is not None:
            selection = self.workload_selection
            selection.validate(f"{path}.workload_selection")
            dispatch, combine = self.economic_decisions
            if (
                selection.source_dispatch_decision_id != dispatch.id
                or selection.source_combine_decision_id != combine.id
                or selection.baseline_pair_ref != (dispatch.baseline.id, combine.baseline.id)
                or selection.selected_dispatch_candidate_ref
                    not in {item.id for item in dispatch.ranked_candidates}
                or selection.selected_combine_candidate_ref
                    not in {item.id for item in combine.ranked_candidates}
                or not selection.performance_complete
            ):
                raise SchemaError(
                    "workload selection is not exact over case decisions",
                    path=f"{path}.workload_selection",
                )
        rebuilt = MoeSwizzleSameWorkOracle.create(self.spec, self.oracle, self.execution)
        if self.same_work != rebuilt:
            raise SchemaError("same-work oracle is not production-derived", path=f"{path}.same_work")
        if tuple(item.branch for item in self.branches) != tuple(MoeSwizzleRuntimeBranch):
            raise SchemaError("case requires canonical NAIVE/AUTO/FORCED branches", path=f"{path}.branches")
        for index, branch in enumerate(self.branches):
            branch.validate_against(self, f"{path}.branches[{index}]")
        decisions = {item.problem.region.pattern: item for item in self.economic_decisions}
        rebuilt_shapes = tuple(sorted({
            shape
            for branch in self.branches
            for deployment in branch.deployments
            for shape in _candidate_comp_shapes(
                decisions[deployment.pattern], deployment.candidate_ref
            )
        }))
        if self.deployed_comp_shapes != rebuilt_shapes:
            raise SchemaError(
                "deployed NAIVE/AUTO/FORCED COMP shape set drifted",
                path=f"{path}.deployed_comp_shapes",
            )
        rebuilt_swiglu_shapes = tuple(sorted({
            shape
            for branch in self.branches
            for deployment in branch.deployments
            for shape in _candidate_swiglu_group_shapes(
                decisions[deployment.pattern], deployment.candidate_ref
            )
        }))
        if self.deployed_swiglu_group_shapes != rebuilt_swiglu_shapes:
            raise SchemaError(
                "deployed NAIVE/AUTO/FORCED SWIGLU_GROUP shape set drifted",
                path=f"{path}.deployed_swiglu_group_shapes",
            )
        profiles = {
            candidate.cost.calibration_profile
            for decision in self.economic_decisions
            for candidate in decision.ranked_candidates
        }
        if None in profiles or len(profiles) != 1:
            raise SchemaError(
                "runtime decisions require one exact MEASURED profile", path=path
            )
        profile = next(iter(profiles))
        measured_comp_shapes = {
            item.shape for item in profile.samples
            if item.kind is MoeCalibrationKind.GROUP_GEMM
        }
        measured_swiglu_shapes = {
            item.shape for item in profile.samples
            if item.kind is MoeCalibrationKind.SWIGLU_GROUP
        }
        if not set(self.deployed_comp_shapes).issubset(measured_comp_shapes):
            raise SchemaError(
                "MEASURED shape coverage must include every deployed NAIVE/AUTO/FORCED COMP shape",
                path=f"{path}.deployed_comp_shapes",
            )
        if not set(self.deployed_swiglu_group_shapes).issubset(measured_swiglu_shapes):
            raise SchemaError(
                "MEASURED shape coverage must include every deployed grouped SWIGLU shape",
                path=f"{path}.deployed_swiglu_group_shapes",
            )
        expected = stable_artifact_id(
            "moe_swizzle_runtime_case",
            self._semantic(),
            schema_version=MOE_SWIZZLE_RUNTIME_CASE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable runtime case id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeSwizzleRuntimeSuitePlan:
    schema_version: str
    producer_pass: str
    id: str
    calibration_profile: MoeSwizzleCalibrationProfile
    calibration_run_digest: str
    cases: tuple[MoeSwizzleRuntimeCasePlan, ...]

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleRuntimeSuitePlan":
        result = cls(
            MOE_SWIZZLE_RUNTIME_SUITE_SCHEMA_VERSION,
            "build_moe_swizzle_runtime_suite",
            stable_artifact_id(
                "moe_swizzle_runtime_suite",
                semantic,
                schema_version=MOE_SWIZZLE_RUNTIME_SUITE_SCHEMA_VERSION,
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

    def validate(self, path: str = "moe_swizzle_runtime_suite") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_RUNTIME_SUITE_SCHEMA_VERSION
            or self.producer_pass != "build_moe_swizzle_runtime_suite"
        ):
            raise SchemaError("unsupported runtime suite schema/producer", path=path)
        self.calibration_profile.validate(f"{path}.calibration_profile")
        if self.calibration_profile.status is not MoeCalibrationStatus.MEASURED:
            raise SchemaError("runtime suite requires actual MEASURED calibration", path=f"{path}.calibration_profile")
        validate_nonempty(self.calibration_run_digest, f"{path}.calibration_run_digest")
        if len(self.cases) != len({item.id for item in self.cases}):
            raise SchemaError("runtime suite duplicates cases", path=f"{path}.cases")
        for index, case in enumerate(self.cases):
            case.validate(f"{path}.cases[{index}]")
            refs = {
                candidate.cost.calibration_profile_ref
                for decision in case.economic_decisions
                for candidate in decision.ranked_candidates
            }
            if refs != {self.calibration_profile.id}:
                raise SchemaError("case decisions use another calibration profile", path=f"{path}.cases[{index}]")
        expected = stable_artifact_id(
            "moe_swizzle_runtime_suite",
            self._semantic(),
            schema_version=MOE_SWIZZLE_RUNTIME_SUITE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable runtime suite id", path=f"{path}.id")


def _deployment(
    decision: MoeSwizzleDecision,
    candidate_ref: str,
    mode: MoeSwizzleDeploymentMode,
    *,
    economic_selected: bool,
) -> MoeSwizzleRegionDeployment:
    candidate = next(
        (item for item in decision.ranked_candidates if item.id == candidate_ref),
        None,
    )
    if candidate is None:
        raise SchemaError("deployment candidate is absent", path="moe_runtime_suite")
    return MoeSwizzleRegionDeployment(
        decision.problem.region.pattern,
        decision.id,
        candidate.id,
        candidate.algorithm,
        mode,
        economic_selected,
    )


def _forced_candidate_ref(
    decision: MoeSwizzleDecision, excluded_ref: str
) -> str:
    fused = tuple(
        item for item in decision.ranked_candidates
        if item.algorithm is not SwizzleAlgorithm.UNFUSED
        and item.id != excluded_ref
    )
    if not fused:
        raise SchemaError(
            "forced coverage requires a non-economic fused candidate",
            path="moe_runtime_suite.economic_decisions",
        )
    return fused[0].id


def _matching_workload_selection(
    decisions: tuple[MoeSwizzleDecision, MoeSwizzleDecision],
    workload_selections: tuple[MoeSwizzleWorkloadSelection, ...],
) -> MoeSwizzleWorkloadSelection:
    for index, selection in enumerate(workload_selections):
        if type(selection) is not MoeSwizzleWorkloadSelection:
            raise SchemaError(
                "workload selections must be exact typed carriers",
                path=f"moe_runtime_suite.workload_selections[{index}]",
            )
        selection.validate(f"moe_runtime_suite.workload_selections[{index}]")
    matches = tuple(
        selection for selection in workload_selections
        if (
            selection.source_dispatch_decision_id == decisions[0].id
            and selection.source_combine_decision_id == decisions[1].id
        )
    )
    if len(matches) != 1:
        raise SchemaError(
            "WORKLOAD requires one exact typed joint selection for rebuilt decisions",
            path="moe_runtime_suite.workload_selections",
        )
    return matches[0]


def build_moe_swizzle_runtime_case_plan(
    case: MoeSwizzleScaleCase,
    mode: MoeScaleExecutionMode,
    calibration_profile: MoeSwizzleCalibrationProfile,
    *,
    scope: MoeSwizzleRuntimeScope,
    target_pattern: FusionPattern | None = None,
    workload_selections: tuple[MoeSwizzleWorkloadSelection, ...] = (),
) -> MoeSwizzleRuntimeCasePlan:
    """Build one case exclusively through production truth and planner APIs."""

    calibration_profile.validate("moe_runtime_suite.calibration_profile")
    if calibration_profile.status is not MoeCalibrationStatus.MEASURED:
        raise SchemaError("economic case construction requires MEASURED calibration", path="moe_runtime_suite.calibration_profile")
    if scope is MoeSwizzleRuntimeScope.REGION_PREFLIGHT and workload_selections:
        raise SchemaError(
            "REGION_PREFLIGHT forbids workload joint selections",
            path="moe_runtime_suite.workload_selections",
        )
    execution = build_moe_swizzle_execution(case.spec, case.oracle, mode)
    regions = discover_moe_swizzle_regions(case.spec, case.oracle, execution)
    problems = tuple(
        build_moe_swizzle_problem(
            region,
            case.spec,
            case.oracle,
            execution,
            hardware_facts=case.hardware_facts,
            endpoint_session_contract=case.endpoint_session_contract,
        )
        for region in regions
    )
    decisions = []
    for problem in problems:
        baseline = build_executable_moe_unfused_baseline(
            problem, case.spec, case.oracle, execution,
            calibration_profile=calibration_profile,
        )
        directs = build_direct_xy_moe_candidates(
            problem, case.spec, case.oracle, execution,
            calibration_profile=calibration_profile,
        )
        comets = build_comet_mesh_moe_candidate_grid(
            problem, case.spec, case.oracle, execution,
            calibration_profile=calibration_profile,
        )
        decisions.append(decide_moe_swizzle(problem, baseline, directs + comets))
    ordered = tuple(
        next(item for item in decisions if item.problem.region.pattern is pattern)
        for pattern in _PATTERNS
    )
    selection = (
        _matching_workload_selection(ordered, workload_selections)
        if scope is MoeSwizzleRuntimeScope.WORKLOAD else None
    )
    selected_refs = (
        {
            FusionPattern.MOE_DISPATCH_GEMM:
                selection.selected_dispatch_candidate_ref,
            FusionPattern.MOE_GEMM_COMBINE:
                selection.selected_combine_candidate_ref,
        }
        if selection is not None else {}
    )
    work = MoeSwizzleSameWorkOracle.create(case.spec, case.oracle, execution)
    work_digest = canonical_digest(work)

    naive = tuple(
        _deployment(
            decision,
            decision.baseline.id,
            MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE,
            economic_selected=False,
        )
        for decision in ordered
    )
    auto = []
    forced = []
    for decision in ordered:
        pattern = decision.problem.region.pattern
        deploy_economic = scope is MoeSwizzleRuntimeScope.WORKLOAD or pattern is target_pattern
        auto_ref = (
            selected_refs[pattern]
            if scope is MoeSwizzleRuntimeScope.WORKLOAD
            else decision.selected_candidate_ref
            if deploy_economic else decision.baseline.id
        )
        auto_candidate = next(item for item in decision.ranked_candidates if item.id == auto_ref)
        auto.append(_deployment(
            decision,
            auto_ref,
            MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE
            if auto_candidate.algorithm is SwizzleAlgorithm.UNFUSED
            else MoeSwizzleDeploymentMode.FUSED_AUTO,
            economic_selected=deploy_economic,
        ))
        deploy_forced = scope is MoeSwizzleRuntimeScope.WORKLOAD or pattern is target_pattern
        excluded_ref = (
            selected_refs[pattern]
            if scope is MoeSwizzleRuntimeScope.WORKLOAD
            else decision.selected_candidate_ref
        )
        forced.append(_deployment(
            decision,
            _forced_candidate_ref(decision, excluded_ref)
            if deploy_forced else decision.baseline.id,
            MoeSwizzleDeploymentMode.FUSED_FORCED
            if deploy_forced else MoeSwizzleDeploymentMode.UNFUSED_TYPED_BASELINE,
            economic_selected=False,
        ))
    branches = tuple(
        MoeSwizzleRuntimeBranchPlan(branch, work_digest, selection_ref, deployments)
        for branch, selection_ref, deployments in (
            (MoeSwizzleRuntimeBranch.NAIVE, None, naive),
            (
                MoeSwizzleRuntimeBranch.SWIZZLE_AUTO,
                selection.id if selection is not None else None,
                tuple(auto),
            ),
            (MoeSwizzleRuntimeBranch.SWIZZLE_FORCED_COVERAGE, None, tuple(forced)),
        )
    )
    decision_by_pattern = {
        item.problem.region.pattern: item for item in ordered
    }
    return MoeSwizzleRuntimeCasePlan.create(
        scope=scope,
        target_pattern=target_pattern,
        spec=case.spec,
        oracle=case.oracle,
        execution=execution,
        economic_decisions=ordered,
        workload_selection=selection,
        same_work=work,
        deployed_comp_shapes=tuple(sorted({
            shape
            for branch in branches
            for deployment in branch.deployments
            for shape in _candidate_comp_shapes(
                decision_by_pattern[deployment.pattern], deployment.candidate_ref
            )
        })),
        deployed_swiglu_group_shapes=tuple(sorted({
            shape
            for branch in branches
            for deployment in branch.deployments
            for shape in _candidate_swiglu_group_shapes(
                decision_by_pattern[deployment.pattern], deployment.candidate_ref
            )
        })),
        branches=branches,
    )


def build_moe_swizzle_runtime_suite_plan(
    calibration_profile: MoeSwizzleCalibrationProfile,
    *,
    calibration_run_digest: str,
    workload_selections: tuple[MoeSwizzleWorkloadSelection, ...] = (),
) -> MoeSwizzleRuntimeSuitePlan:
    """Build the canonical 2 W11 + 10 W12 case matrix.

    The caller must pass the digest of a validated actual 168-run calibration
    evidence carrier and one exact typed joint selection for each WORKLOAD
    execution.  The runtime provider performs the stronger binding to that
    carrier and the W9 overlay lineage before it may materialize an artifact.
    """

    if len({item.id for item in workload_selections}) != len(workload_selections):
        raise SchemaError(
            "workload selections must be canonical unique",
            path="moe_runtime_suite.workload_selections",
        )
    cases = build_moe_swizzle_scale_cases()
    c2 = next(item for item in cases if item.spec.name == "C2")
    preflight = tuple(
        build_moe_swizzle_runtime_case_plan(
            c2,
            MoeScaleExecutionMode.INFER_FORWARD,
            calibration_profile,
            scope=MoeSwizzleRuntimeScope.REGION_PREFLIGHT,
            target_pattern=pattern,
        )
        for pattern in _PATTERNS
    )
    workloads = tuple(
        build_moe_swizzle_runtime_case_plan(
            case,
            mode,
            calibration_profile,
            scope=MoeSwizzleRuntimeScope.WORKLOAD,
            workload_selections=workload_selections,
        )
        for case in cases
        for mode in (
            MoeScaleExecutionMode.INFER_FORWARD,
            MoeScaleExecutionMode.TRAIN_FORWARD,
        )
    )
    return MoeSwizzleRuntimeSuitePlan.create(
        calibration_profile=calibration_profile,
        calibration_run_digest=calibration_run_digest,
        cases=preflight + workloads,
    )


__all__ = [
    "MoeSwizzleRegionDeployment",
    "MoeSwizzleRuntimeBranch",
    "MoeSwizzleRuntimeBranchPlan",
    "MoeSwizzleRuntimeCasePlan",
    "MoeSwizzleRuntimeScope",
    "MoeSwizzleRuntimeSuitePlan",
    "MoeSwizzleSameWorkOracle",
    "build_moe_swizzle_runtime_case_plan",
    "build_moe_swizzle_runtime_suite_plan",
]
