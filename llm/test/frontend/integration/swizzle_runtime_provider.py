"""Production Swizzle branch provider for the W10 runtime harness.

Both Swizzle and UNFUSED/NAIVE branches use only public production lowering APIs.
"""

from __future__ import annotations

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
from llm.frontend.wafer_frontend.passes.project_unfused_comparison import (
    build_unfused_comparison_plan,
    project_unfused_comparison,
)
from llm.frontend.wafer_frontend.passes.project_swizzle_ir2 import (
    project_swizzle_adapter,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    force_swizzle_deployment,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize_ir1 import (
    materialize_swizzle_plan,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.swizzle_evidence import (
    SwizzleComparisonBranch,
    SwizzleComparisonCasePlan,
)
from llm.frontend.wafer_frontend.schema.swizzle_operand_abi import (
    build_swizzle_operand_abi,
)
from llm.frontend.wafer_frontend.schema.swizzle_standard import (
    SwizzleStandardLinkedProgram,
)
from llm.frontend.wafer_frontend.schema.swizzle_unfused_standard import (
    UnfusedComparisonStandardLinkedProgram,
)

from run_swizzle_runtime import SwizzleRuntimeExecutable
from swizzle_cases import SwizzleIntegrationCase, build_swizzle_integration_cases
from swizzle_comparison import build_swizzle_comparison_suite


class ProductionSwizzleLowerLinkProvider:
    """Build exact production sources and actual-SHA ProgramIo for both branches."""

    def __init__(self, *, hardware_json: str, mapping_text: str) -> None:
        if not hardware_json or not mapping_text:
            raise SchemaError(
                "provider requires explicit hardware JSON and mapping text",
                path="swizzle_runtime_provider",
            )
        self._hardware_json = hardware_json
        self._mapping_text = mapping_text
        self._cases = {
            item.name: item for item in build_swizzle_integration_cases()
        }
        self._canonical_plans = {
            item.case_id: item for item in build_swizzle_comparison_suite().cases
        }
        self._sources: dict[
            str,
            SwizzleStandardLinkedProgram | UnfusedComparisonStandardLinkedProgram,
        ] = {}

    def _case(self, plan: SwizzleComparisonCasePlan) -> SwizzleIntegrationCase:
        plan.validate()
        expected = self._canonical_plans.get(plan.case_id)
        case = self._cases.get(plan.case_id)
        if expected is None or case is None or plan != expected:
            raise SchemaError(
                "comparison case is not the exact canonical production case",
                path="swizzle_runtime_provider.case",
            )
        return case

    def lower_link(
        self,
        case_plan: SwizzleComparisonCasePlan,
        branch: SwizzleComparisonBranch,
    ) -> SwizzleRuntimeExecutable:
        case = self._case(case_plan)
        if branch is SwizzleComparisonBranch.NAIVE:
            plan = build_unfused_comparison_plan(
                case.partitioned_graph,
                case.decision.problem,
                case.decision.baseline,
            )
            if (
                plan.problem != case.decision.problem
                or plan.baseline != case.decision.baseline
            ):
                raise SchemaError(
                    "UNFUSED plan drifted from the economic baseline",
                    path="swizzle_runtime_provider.naive.plan",
                )
            projection = project_unfused_comparison(
                case.partitioned_graph, plan
            )
            core_abi = allocate_unfused_comparison_core_abi(
                case.partitioned_graph, plan, projection
            )
            operand_abi = build_unfused_comparison_operand_abi(
                case.partitioned_graph, plan, projection
            )
            lowered = lower_unfused_comparison_opcodes(plan, projection)
            source = link_unfused_comparison_program(
                case.partitioned_graph,
                plan,
                projection,
                lowered,
                core_abi,
                operand_abi,
            )
            source.validate_against()
            self._sources[source.id] = source
            result = SwizzleRuntimeExecutable(
                case_plan_ref=case_plan.id,
                branch=branch,
                linked_source_ref=source.id,
                manifest=source.manifest,
                hardware_json=self._hardware_json,
                mapping_text=self._mapping_text,
            )
            result.validate()
            return result
        if branch is not SwizzleComparisonBranch.SWIZZLE:
            raise SchemaError(
                "must be a typed comparison branch",
                path="swizzle_runtime_provider.branch",
            )

        adapter = force_swizzle_deployment(case.decision)
        plan = materialize_swizzle_plan(
            case.partitioned_graph,
            case.decision,
            case.partitioned_graph.profile,
            deployment_selection=adapter.deployment_selection,
        )
        if (
            plan.id != case_plan.swizzle_plan_ref
            or plan.candidate.id != case_plan.candidate_ref
            or plan.decision.id != case_plan.decision_ref
        ):
            raise SchemaError(
                "production materialization drifted from comparison plan",
                path="swizzle_runtime_provider.plan",
            )
        projection = project_swizzle_adapter(adapter)
        lowered = lower_swizzle_projection(plan, projection)
        core_abi = allocate_swizzle_core_address_abi(
            case.partitioned_graph, plan, projection
        )
        operand_abi = build_swizzle_operand_abi(
            case.partitioned_graph, plan, projection
        )
        source = link_swizzle_standard_program(
            case.partitioned_graph,
            plan,
            projection,
            lowered,
            core_abi,
            operand_abi,
        )
        source.validate_against()
        self._sources[source.id] = source
        result = SwizzleRuntimeExecutable(
            case_plan_ref=case_plan.id,
            branch=branch,
            linked_source_ref=source.id,
            manifest=source.manifest,
            hardware_json=self._hardware_json,
            mapping_text=self._mapping_text,
        )
        result.validate()
        return result

    def build_actual_sha_program_io(
        self,
        executable: SwizzleRuntimeExecutable,
        program_artifact_sha256: str,
    ) -> ProgramIoContract:
        executable.validate()
        source = self._sources.get(executable.linked_source_ref)
        if source is None or source.manifest != executable.manifest:
            raise SchemaError(
                "executable does not resolve to the provider's exact linked source",
                path="swizzle_runtime_provider.executable",
            )
        result = build_timing_program_io(source, program_artifact_sha256)
        result.validate_against(source.manifest)
        return result


__all__ = ["ProductionSwizzleLowerLinkProvider"]
