"""Exact-provenance intra-die graph-refinement producer wrappers."""

from __future__ import annotations

from typing import Mapping, Protocol

from ..errors import SchemaError
from ..schema.intra_die_refine import (
    IntraDieRefineContext,
    RefinedIR2Bundle,
    RefinedProfileIR2,
)
from ..schema.ir1 import IR1
from ..schema.ir2 import IR2ProjectionResult, IntraDieScheduleSet
from ..schema.local_transport import LocalTransportPlan
from ..schema.split_k_refine import SplitKRefinedProjection
from ..schema.n5 import ProjectedIR2Bundle, ProjectedProfileIR2


class IntraDieRefiner(Protocol):
    def refine(self, projection: IR2ProjectionResult, ir1: IR1) -> IR2ProjectionResult: ...
    def refine_graph(
        self,
        projection: IR2ProjectionResult,
        ir1: IR1,
        context: IntraDieRefineContext,
    ) -> SplitKRefinedProjection | None: ...


def refine_profile(
    source: ProjectedProfileIR2,
    context: IntraDieRefineContext,
    policy: IntraDieRefiner | None = None,
    local_transport_plan: LocalTransportPlan | None = None,
) -> RefinedProfileIR2:
    """Refine one profile while retaining the complete projected source."""

    if type(source) is not ProjectedProfileIR2:
        raise SchemaError("must be a ProjectedProfileIR2", path="source")
    if type(context) is not IntraDieRefineContext:
        raise SchemaError("must be an IntraDieRefineContext", path="intra_die_refine_context")
    source.validate("source")
    context.validate("intra_die_refine_context")
    if policy is None:
        from ..policies.identity_intra_die_refine import IdentityIntraDieRefinePolicy

        policy = IdentityIntraDieRefinePolicy()
    projection = policy.refine(source.projection, source.graph)
    if type(projection) is not IR2ProjectionResult:
        raise SchemaError("policy must return an IR2ProjectionResult", path="projection")
    refine_graph = getattr(policy, "refine_graph", None)
    split_k_refinement = (
        refine_graph(source.projection, source.graph, context)
        if callable(refine_graph)
        else None
    )
    result = RefinedProfileIR2.create(
        source=source,
        context=context,
        projection=projection,
        local_transport_plan=local_transport_plan,
        split_k_refinement=split_k_refinement,
    )
    result.validate_against(source, context)
    return result


def refine_bundle(
    source: ProjectedIR2Bundle,
    context: IntraDieRefineContext,
    policy: IntraDieRefiner | None = None,
    local_transport_plans: Mapping[str, LocalTransportPlan] | None = None,
) -> RefinedIR2Bundle:
    """Refine every projected profile exactly once in source tuple order."""

    if type(source) is not ProjectedIR2Bundle:
        raise SchemaError("must be a ProjectedIR2Bundle", path="source")
    if type(context) is not IntraDieRefineContext:
        raise SchemaError("must be an IntraDieRefineContext", path="intra_die_refine_context")
    source.validate("source")
    context.validate("intra_die_refine_context")
    if policy is None:
        from ..policies.identity_intra_die_refine import IdentityIntraDieRefinePolicy

        policy = IdentityIntraDieRefinePolicy()
    plans = local_transport_plans or {}
    if set(plans) - {entry.id for entry in source.entries}:
        raise SchemaError("contains a plan for an unknown source profile", path="local_transport_plans")
    entries = tuple(
        refine_profile(
            entry,
            context,
            policy,
            plans.get(entry.id),
        )
        for entry in source.entries
    )
    result = RefinedIR2Bundle.create(source=source, context=context, entries=entries)
    result.validate_against(source, context)
    return result


def validate_refined_local_transport(
    refined: RefinedProfileIR2,
    schedule_set: IntraDieScheduleSet,
    ir1: IR1,
) -> None:
    """Validate a refined local plan against final schedule placement/routes."""

    if type(refined) is not RefinedProfileIR2:
        raise SchemaError("must be a RefinedProfileIR2", path="refined")
    if type(schedule_set) is not IntraDieScheduleSet:
        raise SchemaError("must be an IntraDieScheduleSet", path="schedule_set")
    if type(ir1) is not IR1:
        raise SchemaError("must be an IR1 artifact", path="ir1")
    refined.validate("refined")
    if refined.local_transport_plan is None:
        return
    refined.local_transport_plan.validate_against_schedule(
        projection=refined.projection,
        schedule_set=schedule_set,
        ir1=ir1,
        path="refined.local_transport_plan",
    )


__all__ = [
    "IntraDieRefiner", "refine_bundle", "refine_profile",
    "validate_refined_local_transport",
]
