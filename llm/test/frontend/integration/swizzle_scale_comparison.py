"""Typed S0-S3 NAIVE/AUTO scale comparison plans.

The plan freezes production ExperimentSpec-to-IR1 provenance and keeps forced
deployment visibly diagnostic.  It intentionally stops before C++ finalization
or simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.project_swizzle_plan import (
    project_swizzle_plan,
)
from llm.frontend.wafer_frontend.passes.project_unfused_comparison import (
    build_unfused_comparison_plan,
    project_unfused_comparison,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    force_swizzle_deployment,
    materialize_swizzle_decision,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize_ir1 import (
    materialize_swizzle_plan,
)
from llm.frontend.wafer_frontend.policies.swizzle.chunking import wang_tile_shape
from llm.frontend.wafer_frontend.schema.common import (
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleAlgorithm
from llm.frontend.wafer_frontend.schema.swizzle_performance_evidence import (
    SwizzleBenefitBranch,
)
from llm.frontend.wafer_frontend.schema.swizzle_plan import (
    SwizzleDeploymentReason,
)

from swizzle_scale_cases import (
    SwizzleScaleCase,
    build_first_green_swizzle_scale_cases,
)


SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_scale_comparison_plan/v1alpha1"
)


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleScaleBranchPlan:
    branch: SwizzleBenefitBranch
    same_work_digest: str
    algorithm: SwizzleAlgorithm
    economic_auto_selected: bool
    forced_deployment: bool
    deployment_selection_ref: str | None
    candidate_ref: str | None
    plan_ref: str
    projection_ref: str
    chunk_count: int
    unroll_degree: int
    tile_shape: tuple[int, int, int] | None

    def validate(self, path: str = "swizzle_scale_branch_plan") -> None:
        if type(self.branch) is not SwizzleBenefitBranch:
            raise SchemaError("must use a typed benefit branch", path=f"{path}.branch")
        _digest(self.same_work_digest, f"{path}.same_work_digest")
        if type(self.algorithm) is not SwizzleAlgorithm:
            raise SchemaError("must use a typed algorithm", path=f"{path}.algorithm")
        if (
            type(self.economic_auto_selected) is not bool
            or type(self.forced_deployment) is not bool
        ):
            raise SchemaError("selection facts must be bools", path=path)
        validate_nonempty(self.plan_ref, f"{path}.plan_ref")
        validate_nonempty(self.projection_ref, f"{path}.projection_ref")
        validate_uint64(self.chunk_count, f"{path}.chunk_count")
        validate_uint64(self.unroll_degree, f"{path}.unroll_degree")
        if self.branch is SwizzleBenefitBranch.NAIVE:
            if (
                self.algorithm is not SwizzleAlgorithm.UNFUSED
                or self.economic_auto_selected
                or self.forced_deployment
                or self.deployment_selection_ref is not None
                or self.candidate_ref is not None
                or self.chunk_count != 0
                or self.unroll_degree != 0
                or self.tile_shape is not None
            ):
                raise SchemaError("NAIVE branch must retain exact UNFUSED facts", path=path)
            return
        if self.algorithm is SwizzleAlgorithm.UNFUSED:
            raise SchemaError("Swizzle branch must select a fused algorithm", path=path)
        for name in ("deployment_selection_ref", "candidate_ref"):
            value = getattr(self, name)
            if value is None:
                raise SchemaError("Swizzle branch requires deployment provenance", path=f"{path}.{name}")
            validate_nonempty(value, f"{path}.{name}")
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
                raise SchemaError("AUTO must be economic and never forced", path=path)
        elif self.branch is SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC:
            if self.economic_auto_selected or not self.forced_deployment:
                raise SchemaError("forced coverage must remain diagnostic", path=path)
        else:
            raise SchemaError("unsupported benefit branch", path=f"{path}.branch")


@dataclass(frozen=True, slots=True)
class SwizzleScaleComparisonCasePlan:
    schema_version: str
    id: str
    scale_ref: str
    scale_name: str
    scale_ordinal: int
    pattern: FusionPattern
    source_ir0_id: str
    source_ir0_digest: str
    placed_ir1_id: str
    placed_ir1_digest: str
    partitioned_ir1_id: str
    partitioned_ir1_digest: str
    decision_ref: str
    decision_digest: str
    same_work_digest: str
    branches: tuple[SwizzleScaleBranchPlan, ...]

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleScaleComparisonCasePlan":
        result = cls(
            SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION,
            stable_artifact_id(
                "swizzle_scale_comparison_case",
                semantic,
                schema_version=SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION,
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

    @property
    def swizzle_branch(self) -> SwizzleScaleBranchPlan:
        result = next(
            (
                item
                for item in self.branches
                if item.branch is SwizzleBenefitBranch.SWIZZLE_AUTO
            ),
            None,
        )
        if result is None:
            raise SchemaError(
                "forced diagnostic case has no economic AUTO branch",
                path="swizzle_scale_comparison_case.branches",
            )
        return result

    @property
    def forced_diagnostic(self) -> SwizzleScaleBranchPlan:
        return next(
            item
            for item in self.branches
            if item.branch is SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC
        )

    def validate(self, path: str = "swizzle_scale_comparison_case") -> None:
        if self.schema_version != SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in (
            "scale_ref", "scale_name", "source_ir0_id", "placed_ir1_id",
            "partitioned_ir1_id", "decision_ref",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        validate_uint64(self.scale_ordinal, f"{path}.scale_ordinal")
        if self.scale_name != f"S{self.scale_ordinal}":
            raise SchemaError("scale name/ordinal mismatch", path=f"{path}.scale_name")
        if type(self.pattern) is not FusionPattern:
            raise SchemaError("must use a FusionPattern", path=f"{path}.pattern")
        for name in (
            "source_ir0_digest", "placed_ir1_digest", "partitioned_ir1_digest",
            "decision_digest", "same_work_digest",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        if type(self.branches) is not tuple:
            raise SchemaError("branches must be an immutable tuple", path=f"{path}.branches")
        for index, branch in enumerate(self.branches):
            branch.validate(f"{path}.branches[{index}]")
            if branch.same_work_digest != self.same_work_digest:
                raise SchemaError("all branches must retain one work digest", path=f"{path}.branches[{index}]")
        has_auto = any(
            item.branch is SwizzleBenefitBranch.SWIZZLE_AUTO
            for item in self.branches
        )
        expected_branches = (
            (
                SwizzleBenefitBranch.NAIVE,
                SwizzleBenefitBranch.SWIZZLE_AUTO,
                SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC,
            )
            if has_auto
            else (
                SwizzleBenefitBranch.NAIVE,
                SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC,
            )
        )
        if tuple(item.branch for item in self.branches) != expected_branches:
            raise SchemaError("branches are not in canonical economic order", path=f"{path}.branches")
        if has_auto and (
            self.swizzle_branch.candidate_ref
            == self.forced_diagnostic.candidate_ref
        ):
            raise SchemaError(
                "forced diagnostic must select a non-economic candidate",
                path=f"{path}.branches",
            )
        expected = stable_artifact_id(
            "swizzle_scale_comparison_case",
            self._semantic_key(),
            schema_version=SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleScaleComparisonSuitePlan:
    schema_version: str
    id: str
    cases: tuple[SwizzleScaleComparisonCasePlan, ...]

    @classmethod
    def create(
        cls,
        cases: tuple[SwizzleScaleComparisonCasePlan, ...],
    ) -> "SwizzleScaleComparisonSuitePlan":
        semantic = {"cases": cases}
        result = cls(
            SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION,
            stable_artifact_id(
                "swizzle_scale_comparison_suite",
                semantic,
                schema_version=SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION,
            ),
            cases,
        )
        result.validate()
        return result

    def validate(self, path: str = "swizzle_scale_comparison_suite") -> None:
        if self.schema_version != SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        expected_keys = tuple(
            (ordinal, pattern)
            for ordinal in range(4)
            for pattern in (FusionPattern.AG_GEMM, FusionPattern.GEMM_RS)
        )
        if tuple((item.scale_ordinal, item.pattern) for item in self.cases) != expected_keys:
            raise SchemaError("requires canonical S0-S3 AG/RS matrix", path=f"{path}.cases")
        for index, item in enumerate(self.cases):
            item.validate(f"{path}.cases[{index}]")
        expected = stable_artifact_id(
            "swizzle_scale_comparison_suite",
            {"cases": self.cases},
            schema_version=SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


def same_work_digest(case: SwizzleScaleCase, decision_index: int) -> str:
    """Digest only exact logical work/provenance shared by both deployments."""

    case.validate()
    decision = case.decisions[decision_index]
    return canonical_digest(
        {
            "scale_ref": case.point.id,
            "source_ir1_id": decision.problem.source_ir1_id,
            "fused_op_id": decision.problem.fused_op_id,
            "pattern": decision.problem.pattern,
            "gemm": decision.problem.gemm,
            "collective": decision.problem.collective,
            "group": decision.problem.group,
            "semantic_witness": decision.baseline.semantic_witness,
        }
    )


def _case_plan(
    case: SwizzleScaleCase,
    decision_index: int,
) -> SwizzleScaleComparisonCasePlan:
    decision = case.decisions[decision_index]
    work_digest = same_work_digest(case, decision_index)
    unfused_plan = build_unfused_comparison_plan(
        case.partitioned_graph, decision.problem, decision.baseline
    )
    unfused_projection = project_unfused_comparison(
        case.partitioned_graph, unfused_plan
    )
    naive = SwizzleScaleBranchPlan(
        branch=SwizzleBenefitBranch.NAIVE,
        same_work_digest=work_digest,
        algorithm=SwizzleAlgorithm.UNFUSED,
        economic_auto_selected=False,
        forced_deployment=False,
        deployment_selection_ref=None,
        candidate_ref=None,
        plan_ref=unfused_plan.id,
        projection_ref=unfused_projection.id,
        chunk_count=0,
        unroll_degree=0,
        tile_shape=None,
    )
    selected = next(
        item
        for item in decision.ranked_candidates
        if item.id == decision.selected_candidate_ref
    )
    adapters = []
    if selected.algorithm is not SwizzleAlgorithm.UNFUSED:
        adapters.append(
            (
                SwizzleBenefitBranch.SWIZZLE_AUTO,
                materialize_swizzle_decision(decision),
            )
        )
    forced_candidate = next(
        item
        for item in decision.ranked_candidates
        if item.algorithm is not SwizzleAlgorithm.UNFUSED
        and item.id != decision.selected_candidate_ref
    )
    adapters.append(
        (
            SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC,
            force_swizzle_deployment(
                decision, candidate_ref=forced_candidate.id
            ),
        )
    )
    swizzle_branches = []
    for branch_kind, adapter in adapters:
        selection = adapter.deployment_selection
        plan = materialize_swizzle_plan(
            case.partitioned_graph,
            decision,
            case.partitioned_graph.profile,
            deployment_selection=selection,
        )
        projection = project_swizzle_plan(case.partitioned_graph, plan)
        economic = selection.reason is SwizzleDeploymentReason.ECONOMIC_DECISION
        forced = selection.reason is SwizzleDeploymentReason.FORCED_BY_POLICY
        swizzle_branches.append(
            SwizzleScaleBranchPlan(
                branch=branch_kind,
                same_work_digest=work_digest,
                algorithm=plan.algorithm,
                economic_auto_selected=economic,
                forced_deployment=forced,
                deployment_selection_ref=selection.id,
                candidate_ref=plan.candidate.id,
                plan_ref=plan.id,
                projection_ref=projection.projection.id,
                chunk_count=plan.candidate.chunk_count,
                unroll_degree=plan.candidate.unroll_degree,
                tile_shape=wang_tile_shape(
                    decision.problem,
                    plan.candidate.semantic_witness,
                    plan.candidate.chunk_count,
                ),
            )
        )
    naive.validate()
    for branch in swizzle_branches:
        branch.validate()
    return SwizzleScaleComparisonCasePlan.create(
        scale_ref=case.point.id,
        scale_name=case.point.name,
        scale_ordinal=int(case.point.name[1:]),
        pattern=decision.problem.pattern,
        source_ir0_id=case.source_graph.id,
        source_ir0_digest=canonical_digest(case.source_graph),
        placed_ir1_id=case.placed_graph.id,
        placed_ir1_digest=canonical_digest(case.placed_graph),
        partitioned_ir1_id=case.partitioned_graph.id,
        partitioned_ir1_digest=canonical_digest(case.partitioned_graph),
        decision_ref=decision.id,
        decision_digest=canonical_digest(decision),
        same_work_digest=work_digest,
        branches=(naive, *swizzle_branches),
    )


def build_swizzle_scale_comparison_suite(
    cases: tuple[SwizzleScaleCase, ...] | None = None,
) -> SwizzleScaleComparisonSuitePlan:
    """Build the canonical S0-S3 AG/RS comparison matrix."""

    source_cases = build_first_green_swizzle_scale_cases() if cases is None else cases
    if tuple(item.point.name for item in source_cases) != ("S0", "S1", "S2", "S3"):
        raise SchemaError("requires exact S0-S3 scale cases", path="scale_cases")
    result = SwizzleScaleComparisonSuitePlan.create(
        tuple(
            _case_plan(case, decision_index)
            for case in source_cases
            for decision_index in range(2)
        )
    )
    result.validate()
    return result


__all__ = [
    "SWIZZLE_SCALE_COMPARISON_PLAN_SCHEMA_VERSION",
    "SwizzleScaleBranchPlan",
    "SwizzleScaleComparisonCasePlan",
    "SwizzleScaleComparisonSuitePlan",
    "build_swizzle_scale_comparison_suite",
    "same_work_digest",
]
