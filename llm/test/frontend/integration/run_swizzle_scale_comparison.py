#!/usr/bin/env python3
"""Run the S0-S3 Python-only production lower/link/ProgramIo preflight."""

from __future__ import annotations

from dataclasses import dataclass
import json

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle_performance_evidence import (
    SwizzleBenefitBranch,
    SwizzleBranchTimingObservation,
    SwizzleScaleBenefitEvidence,
)

from swizzle_scale_cases import build_first_green_swizzle_scale_cases
from swizzle_scale_comparison import (
    SwizzleScaleBranchPlan,
    SwizzleScaleComparisonCasePlan,
    SwizzleScaleComparisonSuitePlan,
    build_swizzle_scale_comparison_suite,
)
from swizzle_scale_runtime_provider import (
    PreparedSwizzleScaleBranch,
    ProductionSwizzleScaleProvider,
)


@dataclass(frozen=True, slots=True)
class SwizzleScalePythonPreflight:
    suite: SwizzleScaleComparisonSuitePlan
    prepared: tuple[PreparedSwizzleScaleBranch, ...]

    def validate(self, path: str = "swizzle_scale_python_preflight") -> None:
        self.suite.validate(f"{path}.suite")
        expected_count = sum(len(item.branches) for item in self.suite.cases)
        if type(self.prepared) is not tuple or len(self.prepared) != expected_count:
            raise SchemaError("prepared branch coverage is incomplete", path=f"{path}.prepared")
        expected = tuple(
            (case.id, branch.branch)
            for case in self.suite.cases
            for branch in case.branches
        )
        actual = tuple(
            (item.case_plan.id, item.branch_plan.branch) for item in self.prepared
        )
        if actual != expected:
            raise SchemaError("prepared branch order/coverage drifted", path=f"{path}.prepared")
        for index, item in enumerate(self.prepared):
            item.validate(f"{path}.prepared[{index}]")
        for case in self.suite.cases:
            branches = tuple(
                item for item in self.prepared if item.case_plan.id == case.id
            )
            if (
                tuple(item.branch_plan.branch for item in branches)
                != tuple(item.branch for item in case.branches)
                or {item.branch_plan.same_work_digest for item in branches}
                != {case.same_work_digest}
            ):
                raise SchemaError("branch work identity is not exact", path=f"{path}.prepared")


@dataclass(frozen=True, slots=True)
class SwizzleScaleOfficialTarget:
    case_plan: SwizzleScaleComparisonCasePlan
    branch_plan: SwizzleScaleBranchPlan

    def validate(self, path: str = "swizzle_scale_official_target") -> None:
        self.case_plan.validate(f"{path}.case_plan")
        self.branch_plan.validate(f"{path}.branch_plan")
        if self.branch_plan not in self.case_plan.branches:
            raise SchemaError(
                "branch does not belong to case", path=f"{path}.branch_plan"
            )
        if self.case_plan.scale_name not in ("S1", "S2"):
            raise SchemaError(
                "official target must be S1 or S2", path=f"{path}.case_plan"
            )
        if self.branch_plan.same_work_digest != self.case_plan.same_work_digest:
            raise SchemaError(
                "official work digest drifted", path=f"{path}.branch_plan"
            )


@dataclass(frozen=True, slots=True)
class SwizzleScaleOfficialMatrix:
    official: tuple[SwizzleScaleOfficialTarget, ...]
    forced_preflight: tuple[SwizzleScaleOfficialTarget, ...]

    def validate(self, path: str = "swizzle_scale_official_matrix") -> None:
        if (
            type(self.official) is not tuple
            or type(self.forced_preflight) is not tuple
        ):
            raise SchemaError("target collections must be tuples", path=path)
        for name, targets in (
            ("official", self.official),
            ("forced_preflight", self.forced_preflight),
        ):
            for index, target in enumerate(targets):
                if type(target) is not SwizzleScaleOfficialTarget:
                    raise SchemaError(
                        "requires exact target carrier",
                        path=f"{path}.{name}[{index}]",
                    )
                target.validate(f"{path}.{name}[{index}]")
        actual_official = tuple(
            (
                target.case_plan.scale_name,
                target.case_plan.pattern,
                target.branch_plan.branch,
            )
            for target in self.official
        )
        expected_official = (
            ("S1", FusionPattern.AG_GEMM, SwizzleBenefitBranch.NAIVE),
            ("S1", FusionPattern.AG_GEMM, SwizzleBenefitBranch.SWIZZLE_AUTO),
            ("S2", FusionPattern.AG_GEMM, SwizzleBenefitBranch.NAIVE),
            ("S2", FusionPattern.AG_GEMM, SwizzleBenefitBranch.SWIZZLE_AUTO),
            ("S2", FusionPattern.GEMM_RS, SwizzleBenefitBranch.NAIVE),
            ("S2", FusionPattern.GEMM_RS, SwizzleBenefitBranch.SWIZZLE_AUTO),
        )
        if actual_official != expected_official:
            if any(
                branch is SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC
                for _, _, branch in actual_official
            ):
                raise SchemaError(
                    "forced diagnostic cannot enter official evidence",
                    path=f"{path}.official",
                )
            raise SchemaError(
                "official target coverage/order drifted", path=f"{path}.official"
            )
        if tuple(
            (target.case_plan.scale_name, target.case_plan.pattern)
            for target in self.forced_preflight
        ) != (
            ("S1", FusionPattern.AG_GEMM),
            ("S1", FusionPattern.GEMM_RS),
            ("S2", FusionPattern.AG_GEMM),
            ("S2", FusionPattern.GEMM_RS),
        ) or any(
            target.branch_plan.branch
            is not SwizzleBenefitBranch.SWIZZLE_FORCED_DIAGNOSTIC
            for target in self.forced_preflight
        ):
            raise SchemaError(
                "forced preflight target coverage/order drifted",
                path=f"{path}.forced_preflight",
            )
        grouped: dict[str, list[SwizzleScaleOfficialTarget]] = {}
        for target in self.official:
            grouped.setdefault(target.case_plan.id, []).append(target)
        if any(
            tuple(item.branch_plan.branch for item in targets)
            != (
                SwizzleBenefitBranch.NAIVE,
                SwizzleBenefitBranch.SWIZZLE_AUTO,
            )
            or len({item.branch_plan.same_work_digest for item in targets}) != 1
            for targets in grouped.values()
        ):
            raise SchemaError(
                "official cases require exact same-work NAIVE/AUTO pairs",
                path=f"{path}.official",
            )


def build_scale_official_matrix(
    suite: SwizzleScaleComparisonSuitePlan,
) -> SwizzleScaleOfficialMatrix:
    """Select only genuine S1/S2 economic pairs; retain forced as preflight."""

    suite.validate()
    eligible = tuple(
        case for case in suite.cases if case.scale_name in ("S1", "S2")
    )
    official = tuple(
        SwizzleScaleOfficialTarget(case, branch)
        for case in eligible
        if any(
            item.branch is SwizzleBenefitBranch.SWIZZLE_AUTO
            for item in case.branches
        )
        for branch in case.branches
        if branch.branch
        in (SwizzleBenefitBranch.NAIVE, SwizzleBenefitBranch.SWIZZLE_AUTO)
    )
    forced_preflight = tuple(
        SwizzleScaleOfficialTarget(case, case.forced_diagnostic)
        for case in eligible
    )
    result = SwizzleScaleOfficialMatrix(
        official=official, forced_preflight=forced_preflight
    )
    result.validate()
    return result


def build_scale_benefit_evidence(
    case_plan: SwizzleScaleComparisonCasePlan,
    *,
    naive: SwizzleBranchTimingObservation,
    swizzle_auto: SwizzleBranchTimingObservation,
) -> SwizzleScaleBenefitEvidence:
    """Assemble formal evidence only for a genuinely economic AUTO branch.

    The observation objects are retained directly, including observed send/recv
    inflight counts; no expected or analytical values are substituted.
    """

    case_plan.validate()
    auto_plan = case_plan.swizzle_branch
    if auto_plan.branch is not SwizzleBenefitBranch.SWIZZLE_AUTO:
        raise SchemaError(
            "forced diagnostic cannot be used as economic AUTO evidence",
            path="scale_benefit_evidence.swizzle_auto",
        )
    naive.validate("scale_benefit_evidence.naive")
    swizzle_auto.validate("scale_benefit_evidence.swizzle_auto")
    if (
        naive.branch is not SwizzleBenefitBranch.NAIVE
        or swizzle_auto.branch is not SwizzleBenefitBranch.SWIZZLE_AUTO
        or swizzle_auto.algorithm is not auto_plan.algorithm
        or swizzle_auto.chunk_count != auto_plan.chunk_count
        or swizzle_auto.unroll_degree != auto_plan.unroll_degree
        or swizzle_auto.tile_shape != auto_plan.tile_shape
        or not swizzle_auto.economic_auto_selected
        or swizzle_auto.forced_deployment
    ):
        raise SchemaError(
            "timing observations do not match exact NAIVE/economic AUTO plan facts",
            path="scale_benefit_evidence",
        )
    result = SwizzleScaleBenefitEvidence.create(
        scale_ref=case_plan.scale_ref,
        scale_ordinal=case_plan.scale_ordinal,
        pattern=case_plan.pattern,
        same_work_digest=case_plan.same_work_digest,
        naive=naive,
        swizzle_auto=swizzle_auto,
    )
    if (
        result.naive.observed_max_inflight_send,
        result.naive.observed_max_inflight_recv,
        result.swizzle_auto.observed_max_inflight_send,
        result.swizzle_auto.observed_max_inflight_recv,
    ) != (
        naive.observed_max_inflight_send,
        naive.observed_max_inflight_recv,
        swizzle_auto.observed_max_inflight_send,
        swizzle_auto.observed_max_inflight_recv,
    ):
        raise SchemaError(
            "actual inflight observations were not retained losslessly",
            path="scale_benefit_evidence",
        )
    return result


def run_python_scale_preflight() -> SwizzleScalePythonPreflight:
    """Build all 21 real economic/diagnostic branch products; no C++ tools."""

    cases = build_first_green_swizzle_scale_cases()
    suite = build_swizzle_scale_comparison_suite(cases)
    provider = ProductionSwizzleScaleProvider(cases=cases, suite=suite)
    result = SwizzleScalePythonPreflight(
        suite=suite,
        prepared=tuple(
            provider.prepare(case_plan, branch.branch)
            for case_plan in suite.cases
            for branch in case_plan.branches
        ),
    )
    result.validate()
    return result


def _json_summary(result: SwizzleScalePythonPreflight) -> dict[str, object]:
    rows = []
    for item in result.prepared:
        program_io = item.program_io
        rows.append(
            {
                "scale": item.case_plan.scale_name,
                "pattern": item.case_plan.pattern.value,
                "branch": item.branch_plan.branch.value,
                "algorithm": item.branch_plan.algorithm.value,
                "economic_auto_selected": item.branch_plan.economic_auto_selected,
                "forced_deployment": item.branch_plan.forced_deployment,
                "same_work_digest": item.case_plan.same_work_digest,
                "chunk_count": item.branch_plan.chunk_count,
                "unroll_degree": item.branch_plan.unroll_degree,
                "tile_shape": item.branch_plan.tile_shape,
                "stream_count": item.stream_count,
                "linked_source_ref": item.source.id,
                "manifest_ref": item.source.manifest.id,
                "program_io_ref": program_io.id,
                "program_sha256": program_io.program_artifact_sha256,
                "sram_initialization_count": len(program_io.initializations),
                "output_probe_count": len(program_io.output_probes),
                "blob_count": len(program_io.blobs),
            }
        )
    return {
        "suite_ref": result.suite.id,
        "case_count": len(result.suite.cases),
        "branch_count": len(result.prepared),
        "patterns": [FusionPattern.AG_GEMM.value, FusionPattern.GEMM_RS.value],
        "branches": rows,
    }


def main() -> int:
    result = run_python_scale_preflight()
    print(json.dumps(_json_summary(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SwizzleScaleOfficialMatrix",
    "SwizzleScaleOfficialTarget",
    "SwizzleScalePythonPreflight",
    "build_scale_official_matrix",
    "build_scale_benefit_evidence",
    "run_python_scale_preflight",
]
