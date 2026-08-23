"""Python-only production lowering/linking for Swizzle scale comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.swizzle import lower_swizzle_projection
from llm.frontend.wafer_frontend.lowering.swizzle_abi import (
    allocate_swizzle_core_address_abi,
)
from llm.frontend.wafer_frontend.lowering.swizzle_standard import (
    link_swizzle_standard_program,
)
from llm.frontend.wafer_frontend.lowering.swizzle_unfused import (
    allocate_unfused_comparison_core_abi,
    build_unfused_comparison_operand_abi,
    lower_unfused_comparison_opcodes,
)
from llm.frontend.wafer_frontend.lowering.swizzle_unfused_standard import (
    link_unfused_comparison_program,
)
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
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
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.swizzle_operand_abi import (
    build_swizzle_operand_abi,
)
from llm.frontend.wafer_frontend.schema.swizzle_performance_evidence import (
    SwizzleBenefitBranch,
)
from llm.frontend.wafer_frontend.schema.swizzle_standard import (
    SwizzleStandardLinkedProgram,
)
from llm.frontend.wafer_frontend.schema.swizzle_unfused_standard import (
    UnfusedComparisonStandardLinkedProgram,
)

from swizzle_scale_cases import (
    SwizzleScaleCase,
    build_first_green_swizzle_scale_cases,
)
from swizzle_scale_comparison import (
    SwizzleScaleBranchPlan,
    SwizzleScaleComparisonCasePlan,
    SwizzleScaleComparisonSuitePlan,
    build_swizzle_scale_comparison_suite,
    same_work_digest,
)


ZERO_PROGRAM_ARTIFACT_SHA256 = "0" * 64

LinkedScaleSource = (
    SwizzleStandardLinkedProgram | UnfusedComparisonStandardLinkedProgram
)


@dataclass(frozen=True, slots=True)
class PreparedSwizzleScaleBranch:
    case_plan: SwizzleScaleComparisonCasePlan
    branch_plan: SwizzleScaleBranchPlan
    source: LinkedScaleSource
    program_io: ProgramIoContract
    stream_count: int

    def validate(self, path: str = "prepared_swizzle_scale_branch") -> None:
        self.case_plan.validate(f"{path}.case_plan")
        self.branch_plan.validate(f"{path}.branch_plan")
        if self.branch_plan not in self.case_plan.branches:
            raise SchemaError("branch does not belong to case", path=f"{path}.branch_plan")
        if self.branch_plan.same_work_digest != self.case_plan.same_work_digest:
            raise SchemaError("branch work digest drifted", path=f"{path}.branch_plan")
        if self.branch_plan.branch is SwizzleBenefitBranch.NAIVE:
            if type(self.source) is not UnfusedComparisonStandardLinkedProgram:
                raise SchemaError("NAIVE requires exact UNFUSED source", path=f"{path}.source")
        elif type(self.source) is not SwizzleStandardLinkedProgram:
            raise SchemaError("Swizzle requires exact standard source", path=f"{path}.source")
        self.source.validate(f"{path}.source")
        if (
            self.source.ir1.id,
            self.source.plan.id,
            self.source.projection.id,
        ) != (
            self.case_plan.partitioned_ir1_id,
            self.branch_plan.plan_ref,
            self.branch_plan.projection_ref,
        ):
            raise SchemaError(
                "linked source provenance does not match exact case/branch",
                path=f"{path}.source",
            )
        self.program_io.validate_against(self.source.manifest, f"{path}.program_io")
        if self.program_io.program_artifact_sha256 != ZERO_PROGRAM_ARTIFACT_SHA256:
            raise SchemaError("preflight must use explicit zero SHA", path=f"{path}.program_io")
        if type(self.stream_count) is not int or self.stream_count <= 0:
            raise SchemaError("stream count must be positive", path=f"{path}.stream_count")
        if self.stream_count != len(self.source.manifest.core_streams):
            raise SchemaError("stream count must equal linked core coverage", path=f"{path}.stream_count")


class ProductionSwizzleScaleProvider:
    """Rebuild exact scale branches exclusively through production APIs."""

    def __init__(
        self,
        *,
        cases: tuple[SwizzleScaleCase, ...] | None = None,
        suite: SwizzleScaleComparisonSuitePlan | None = None,
    ) -> None:
        self._cases = (
            build_first_green_swizzle_scale_cases() if cases is None else cases
        )
        if tuple(item.point.name for item in self._cases) != (
            "S0", "S1", "S2", "S3"
        ):
            raise SchemaError("provider requires exact S0-S3 cases", path="scale_cases")
        for index, case in enumerate(self._cases):
            case.validate(f"scale_cases[{index}]")
        self._suite = (
            build_swizzle_scale_comparison_suite(self._cases)
            if suite is None
            else suite
        )
        self._suite.validate()
        self._case_index = {
            (case.point.name, decision.problem.pattern): (case, index)
            for case in self._cases
            for index, decision in enumerate(case.decisions)
        }
        if {
            (item.scale_name, item.pattern) for item in self._suite.cases
        } != set(self._case_index):
            raise SchemaError("suite/case coverage mismatch", path="scale_suite")

    def _resolve(
        self,
        case_plan: SwizzleScaleComparisonCasePlan,
    ) -> tuple[SwizzleScaleCase, int]:
        case_plan.validate()
        expected = next(
            (
                item
                for item in self._suite.cases
                if (item.scale_name, item.pattern)
                == (case_plan.scale_name, case_plan.pattern)
            ),
            None,
        )
        resolved = self._case_index.get((case_plan.scale_name, case_plan.pattern))
        if expected is None or resolved is None or expected != case_plan:
            raise SchemaError("case is not the exact canonical scale plan", path="scale_provider.case")
        case, decision_index = resolved
        decision = case.decisions[decision_index]
        if (
            case_plan.source_ir0_id != case.source_graph.id
            or case_plan.placed_ir1_id != case.placed_graph.id
            or case_plan.partitioned_ir1_id != case.partitioned_graph.id
            or case_plan.decision_ref != decision.id
            or case_plan.same_work_digest != same_work_digest(case, decision_index)
        ):
            raise SchemaError("case production provenance drifted", path="scale_provider.case")
        return case, decision_index

    @staticmethod
    def _branch(
        case_plan: SwizzleScaleComparisonCasePlan,
        branch: SwizzleBenefitBranch,
    ) -> SwizzleScaleBranchPlan:
        if type(branch) is not SwizzleBenefitBranch:
            raise SchemaError("requires typed benefit branch", path="scale_provider.branch")
        result = next((item for item in case_plan.branches if item.branch is branch), None)
        if result is None:
            raise SchemaError("branch is unavailable for this economic decision", path="scale_provider.branch")
        return result

    def prepare(
        self,
        case_plan: SwizzleScaleComparisonCasePlan,
        branch: SwizzleBenefitBranch,
    ) -> PreparedSwizzleScaleBranch:
        case, decision_index = self._resolve(case_plan)
        decision = case.decisions[decision_index]
        branch_plan = self._branch(case_plan, branch)
        if branch is SwizzleBenefitBranch.NAIVE:
            plan = build_unfused_comparison_plan(
                case.partitioned_graph, decision.problem, decision.baseline
            )
            projection = project_unfused_comparison(case.partitioned_graph, plan)
            if (plan.id, projection.id) != (
                branch_plan.plan_ref,
                branch_plan.projection_ref,
            ):
                raise SchemaError("UNFUSED plan/projection drifted", path="scale_provider.naive")
            core_abi = allocate_unfused_comparison_core_abi(
                case.partitioned_graph, plan, projection
            )
            operand_abi = build_unfused_comparison_operand_abi(
                case.partitioned_graph, plan, projection
            )
            lowered = lower_unfused_comparison_opcodes(plan, projection)
            source: LinkedScaleSource = link_unfused_comparison_program(
                case.partitioned_graph,
                plan,
                projection,
                lowered,
                core_abi,
                operand_abi,
            )
        else:
            adapter = (
                materialize_swizzle_decision(decision)
                if branch is SwizzleBenefitBranch.SWIZZLE_AUTO
                else force_swizzle_deployment(
                    decision, candidate_ref=branch_plan.candidate_ref
                )
            )
            plan = materialize_swizzle_plan(
                case.partitioned_graph,
                decision,
                case.partitioned_graph.profile,
                deployment_selection=adapter.deployment_selection,
            )
            projected = project_swizzle_plan(case.partitioned_graph, plan)
            if (
                adapter.deployment_selection.id,
                plan.candidate.id,
                plan.id,
                projected.projection.id,
            ) != (
                branch_plan.deployment_selection_ref,
                branch_plan.candidate_ref,
                branch_plan.plan_ref,
                branch_plan.projection_ref,
            ):
                raise SchemaError("Swizzle plan/projection drifted", path="scale_provider.swizzle")
            lowered = lower_swizzle_projection(plan, projected.projection)
            core_abi = allocate_swizzle_core_address_abi(
                case.partitioned_graph, plan, projected.projection
            )
            operand_abi = build_swizzle_operand_abi(
                case.partitioned_graph, plan, projected.projection
            )
            source = link_swizzle_standard_program(
                case.partitioned_graph,
                plan,
                projected.projection,
                lowered,
                core_abi,
                operand_abi,
            )
        source.validate_against()
        program_io = build_timing_program_io(
            source, ZERO_PROGRAM_ARTIFACT_SHA256
        )
        program_io.validate_against(source.manifest)
        result = PreparedSwizzleScaleBranch(
            case_plan=case_plan,
            branch_plan=branch_plan,
            source=source,
            program_io=program_io,
            stream_count=len(source.manifest.core_streams),
        )
        result.validate()
        if result.stream_count != case.point.tp:
            raise SchemaError(
                "linked stream coverage must equal physical TP ranks",
                path="scale_provider.stream_count",
            )
        return result


def build_actual_scale_program_io(
    prepared: PreparedSwizzleScaleBranch,
    program_artifact_sha256: str,
) -> ProgramIoContract:
    """Bind one finalized non-zero artifact SHA to an exact prepared source."""

    prepared.validate("actual_scale_program_io.prepared")
    if (
        type(program_artifact_sha256) is not str
        or program_artifact_sha256 == ZERO_PROGRAM_ARTIFACT_SHA256
        or len(program_artifact_sha256) != 64
        or program_artifact_sha256.lower() != program_artifact_sha256
        or any(
            character not in "0123456789abcdef"
            for character in program_artifact_sha256
        )
    ):
        raise SchemaError(
            "actual ProgramIo requires one non-zero lowercase SHA-256",
            path="actual_scale_program_io.program_artifact_sha256",
        )
    result = build_timing_program_io(prepared.source, program_artifact_sha256)
    result.validate_against(prepared.source.manifest, "actual_scale_program_io")
    if result.program_artifact_sha256 != program_artifact_sha256:
        raise SchemaError(
            "actual artifact SHA was not retained exactly",
            path="actual_scale_program_io.program_artifact_sha256",
        )
    return result


__all__ = [
    "build_actual_scale_program_io",
    "PreparedSwizzleScaleBranch",
    "ProductionSwizzleScaleProvider",
    "ZERO_PROGRAM_ARTIFACT_SHA256",
]
