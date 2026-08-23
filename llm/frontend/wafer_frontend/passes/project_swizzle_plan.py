"""Project a production SwizzleFusionPlan into the isolated timing IR2.

This pass is intentionally separate from the legacy ``ProjectToIR2`` policy.
That policy accepts only the DIRECT/NAIVE ``FusionPlan`` contract and its
downstream validators still require legacy chunk/permutation and
ReduceScatter coverage.  Converting a Swizzle plan into that carrier would
discard pattern-aware execution facts.

The bridge below instead derives the already-published W8 adapter exactly
from a production plan, projects it with the strict timing-only projector, and
retains the production plan id as an independently validated provenance edge.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..policies.swizzle.materialize import (
    SwizzleFusionActionAdapter,
    SwizzleFusionPlanAdapter,
    SwizzleFusionRankProgramAdapter,
)
from ..schema.common import stable_artifact_id, validate_nonempty
from ..schema.ir1 import IR1
from ..schema.swizzle_ir2 import SwizzleIr2Projection
from ..schema.swizzle_plan import SwizzleFusionPlan
from .project_swizzle_ir2 import (
    project_swizzle_adapter,
    validate_swizzle_projection_against_adapter,
)


SWIZZLE_PLAN_PROJECTION_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_plan_projection/v1alpha1"
)


def _adapter_from_plan(plan: SwizzleFusionPlan) -> SwizzleFusionPlanAdapter:
    """Erase no executable witness while entering the frozen W8 API."""

    return SwizzleFusionPlanAdapter.create(
        deployment_selection=plan.deployment_selection,
        rank_programs=tuple(
            SwizzleFusionRankProgramAdapter(
                rank=program.rank,
                actions=tuple(
                    SwizzleFusionActionAdapter(
                        source_action=action.source_action,
                        fusion_kind=action.fusion_kind,
                        member_ref=action.member_ref,
                        expected_route=action.expected_route,
                    )
                    for action in program.actions
                ),
            )
            for program in plan.rank_programs
        ),
        buffer_requirements=plan.buffer_requirements,
    )


def _validate_adapter_against_plan(
    adapter: SwizzleFusionPlanAdapter,
    plan: SwizzleFusionPlan,
    *,
    path: str,
) -> None:
    adapter.validate(f"{path}.adapter")
    plan.validate(f"{path}.source_plan")
    if (
        adapter.deployment_selection,
        adapter.buffer_requirements,
        adapter.source_ir1_id,
        adapter.fused_op_id,
        adapter.group_ref,
        adapter.pattern,
        adapter.algorithm,
    ) != (
        plan.deployment_selection,
        plan.buffer_requirements,
        plan.source_ir1_id,
        plan.fused_op_id,
        plan.group_ref,
        plan.pattern,
        plan.algorithm,
    ):
        raise SchemaError(
            "W8 adapter provenance disagrees with production plan",
            path=f"{path}.adapter",
        )
    if tuple(program.rank for program in adapter.rank_programs) != tuple(
        program.rank for program in plan.rank_programs
    ):
        raise SchemaError(
            "W8 adapter rank order disagrees with production plan",
            path=f"{path}.adapter.rank_programs",
        )
    for rank_index, (adapted, source) in enumerate(
        zip(adapter.rank_programs, plan.rank_programs, strict=True)
    ):
        if len(adapted.actions) != len(source.actions):
            raise SchemaError(
                "W8 adapter action count disagrees with production plan",
                path=f"{path}.adapter.rank_programs[{rank_index}]",
            )
        for action_index, (adapted_action, source_action) in enumerate(
            zip(adapted.actions, source.actions, strict=True)
        ):
            if (
                adapted_action.source_action,
                adapted_action.fusion_kind,
                adapted_action.member_ref,
                adapted_action.expected_route,
            ) != (
                source_action.source_action,
                source_action.fusion_kind,
                source_action.member_ref,
                source_action.expected_route,
            ):
                raise SchemaError(
                    "W8 adapter action binding disagrees with production plan",
                    path=(
                        f"{path}.adapter.rank_programs[{rank_index}]"
                        f".actions[{action_index}]"
                    ),
                )


@dataclass(frozen=True, slots=True)
class SwizzlePlanProjection:
    """Exact production-plan provenance around the strict W8 timing carrier."""

    schema_version: str
    producer_pass: str
    id: str
    source_plan_ref: str
    adapter: SwizzleFusionPlanAdapter
    projection: SwizzleIr2Projection

    @classmethod
    def create(
        cls,
        *,
        source_plan_ref: str,
        adapter: SwizzleFusionPlanAdapter,
        projection: SwizzleIr2Projection,
    ) -> "SwizzlePlanProjection":
        semantic = {
            "source_plan_ref": source_plan_ref,
            "adapter": adapter,
            "projection": projection,
        }
        result = cls(
            schema_version=SWIZZLE_PLAN_PROJECTION_SCHEMA_VERSION,
            producer_pass="project_swizzle_plan",
            id=stable_artifact_id(
                "swizzle_plan_projection",
                semantic,
                schema_version=SWIZZLE_PLAN_PROJECTION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_plan_ref": self.source_plan_ref,
            "adapter": self.adapter,
            "projection": self.projection,
        }

    def validate(self, path: str = "swizzle_plan_projection") -> None:
        if self.schema_version != SWIZZLE_PLAN_PROJECTION_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        if self.producer_pass != "project_swizzle_plan":
            raise SchemaError(
                "must be produced by project_swizzle_plan",
                path=f"{path}.producer_pass",
            )
        validate_nonempty(self.source_plan_ref, f"{path}.source_plan_ref")
        self.adapter.validate(f"{path}.adapter")
        validate_swizzle_projection_against_adapter(
            self.projection,
            self.adapter,
        )
        expected = stable_artifact_id(
            "swizzle_plan_projection",
            self._semantic_key(),
            schema_version=SWIZZLE_PLAN_PROJECTION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        ir1: IR1,
        plan: SwizzleFusionPlan,
        path: str = "swizzle_plan_projection",
    ) -> None:
        self.validate(path)
        if type(ir1) is not IR1 or type(plan) is not SwizzleFusionPlan:
            raise SchemaError(
                "requires typed IR1 and SwizzleFusionPlan",
                path=path,
            )
        plan.validate_against(ir1, f"{path}.source_plan")
        if self.source_plan_ref != plan.id:
            raise SchemaError(
                "projection references a different production plan",
                path=f"{path}.source_plan_ref",
            )
        _validate_adapter_against_plan(self.adapter, plan, path=path)
        if (
            self.projection.source_ir1_id,
            self.projection.source_decision_ref,
            self.projection.source_candidate_ref,
            self.projection.fused_op_id,
            self.projection.group_ref,
            self.projection.pattern,
            self.projection.algorithm,
        ) != (
            plan.source_ir1_id,
            plan.decision.id,
            plan.candidate.id,
            plan.fused_op_id,
            plan.group_ref,
            plan.pattern,
            plan.algorithm,
        ):
            raise SchemaError(
                "timing projection provenance disagrees with production plan",
                path=f"{path}.projection",
            )


def project_swizzle_plan(
    ir1: IR1,
    plan: SwizzleFusionPlan,
) -> SwizzlePlanProjection:
    """Project one validated production plan without claiming legacy IR2 support."""

    if type(ir1) is not IR1 or type(plan) is not SwizzleFusionPlan:
        raise SchemaError(
            "requires typed IR1 and SwizzleFusionPlan",
            path="project_swizzle_plan",
        )
    plan.validate_against(ir1, "swizzle_fusion_plan")
    adapter = _adapter_from_plan(plan)
    _validate_adapter_against_plan(adapter, plan, path="swizzle_plan_projection")
    result = SwizzlePlanProjection.create(
        source_plan_ref=plan.id,
        adapter=adapter,
        projection=project_swizzle_adapter(adapter),
    )
    result.validate_against(ir1, plan)
    return result


__all__ = [
    "SWIZZLE_PLAN_PROJECTION_SCHEMA_VERSION",
    "SwizzlePlanProjection",
    "project_swizzle_plan",
]
