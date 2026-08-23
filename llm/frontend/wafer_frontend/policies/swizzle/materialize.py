"""Lossless bridge from a selected Swizzle decision to FusionPlan actions.

This module intentionally stops one boundary before ``schema.action.FusionPlan``.
The published FusionPlan validator is still the DIRECT/NAIVE GEMM+RS contract,
and its ``FusionAction`` compute/reduce variants require executable
``ComputeContract``/``ReductionContract`` payloads that are not present in a
Swizzle action witness.  ``SwizzleFusionPlanAdapter`` is therefore the narrow,
typed hand-off for the action-schema migration: it retains the complete
candidate and decision while closing action kind, member and physical-route
bindings without inventing downstream execution semantics.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import SchemaError
from ...schema.action import BarrierContract, BarrierScope, FusionActionKind, SyncContract
from ...schema.common import ProfileKey, stable_artifact_id, validate_nonempty, validate_uint64
from ...schema.ir0 import CollectiveWorkload, FusionImpl, FusionPattern, GemmWorkload, OpKind, ReduceOp
from ...schema.ir1 import IR1
from ...schema.swizzle_plan import (
    SwizzleBoundAction,
    SwizzleBoundRankProgram,
    SwizzleChunkOrigin,
    SwizzleComputeOrigin,
    SwizzleDeploymentReason,
    SwizzleDeploymentSelection,
    SwizzleFusionPlan,
    SwizzleReductionOrigin,
    SwizzleReductionOriginKind,
    SwizzleValueOrigin,
    SwizzleValueUse,
)
from ...schema.swizzle import (
    SwizzleActionKind,
    SwizzleActionWitness,
    SwizzleAlgorithm,
    SwizzleBufferRequirement,
    SwizzleCandidate,
    SwizzleDecision,
    SwizzlePhase,
)


SWIZZLE_FUSION_PLAN_ADAPTER_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_fusion_plan_adapter/v1alpha1"
)


_ACTION_KIND = {
    SwizzleActionKind.COMP: FusionActionKind.COMP,
    SwizzleActionKind.SEND: FusionActionKind.SEND,
    SwizzleActionKind.RECV: FusionActionKind.RECV,
    SwizzleActionKind.WAIT: FusionActionKind.WAIT,
    SwizzleActionKind.REDUCE: FusionActionKind.REDUCE,
    SwizzleActionKind.LOCAL_COPY: FusionActionKind.LOCAL_COPY,
    SwizzleActionKind.BARRIER: FusionActionKind.BARRIER,
}


@dataclass(frozen=True, slots=True)
class SwizzleFusionActionAdapter:
    """One lossless Swizzle witness plus its FusionAction dispatch bindings."""

    source_action: SwizzleActionWitness
    fusion_kind: FusionActionKind
    member_ref: str
    expected_route: tuple[int, ...]

    def validate(
        self,
        *,
        route_index: dict[str, tuple[int, int, tuple[int, ...]]],
        path: str,
    ) -> None:
        self.source_action.validate(f"{path}.source_action")
        expected_kind = _ACTION_KIND[self.source_action.kind]
        if self.fusion_kind is not expected_kind:
            raise SchemaError(
                f"must map to {expected_kind.value!r}", path=f"{path}.fusion_kind"
            )
        validate_nonempty(self.member_ref, f"{path}.member_ref")

        action = self.source_action
        if action.kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
            route = route_index.get(action.route_ref or "")
            if route is None:
                raise SchemaError(
                    "transport action references an unknown problem route",
                    path=f"{path}.source_action.route_ref",
                )
            source_rank, destination_rank, die_path = route
            expected_pair = (
                (action.rank, action.peer_rank)
                if action.kind is SwizzleActionKind.SEND
                else (action.peer_rank, action.rank)
            )
            if (source_rank, destination_rank) != expected_pair:
                raise SchemaError(
                    "route endpoints disagree with action rank/peer",
                    path=f"{path}.source_action.route_ref",
                )
            if self.expected_route != die_path:
                raise SchemaError(
                    "expected route must exactly equal the frozen IR-1 route",
                    path=f"{path}.expected_route",
                )
        elif self.expected_route:
            raise SchemaError(
                "non-transport action cannot carry an expected route",
                path=f"{path}.expected_route",
            )


@dataclass(frozen=True, slots=True)
class SwizzleFusionRankProgramAdapter:
    rank: int
    actions: tuple[SwizzleFusionActionAdapter, ...]

    def validate(
        self,
        *,
        route_index: dict[str, tuple[int, int, tuple[int, ...]]],
        path: str,
    ) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        if type(self.actions) is not tuple or not self.actions:
            raise SchemaError("must contain actions", path=f"{path}.actions")
        for index, action in enumerate(self.actions):
            if action.source_action.rank != self.rank:
                raise SchemaError(
                    "source action rank must equal program rank",
                    path=f"{path}.actions[{index}].source_action.rank",
                )
            action.validate(
                route_index=route_index,
                path=f"{path}.actions[{index}]",
            )


@dataclass(frozen=True, slots=True)
class SwizzleFusionPlanAdapter:
    """Typed W7 carrier ready for a pattern-aware FusionPlan constructor.

    Keeping both ``decision`` and its selected ``candidate`` is intentional:
    validation proves that no candidate, ranking, cost or provenance evidence
    was dropped while materializing executable action bindings.
    """

    schema_version: str
    id: str
    deployment_selection: SwizzleDeploymentSelection
    rank_programs: tuple[SwizzleFusionRankProgramAdapter, ...]
    buffer_requirements: tuple[SwizzleBufferRequirement, ...]

    @classmethod
    def create(
        cls,
        *,
        deployment_selection: SwizzleDeploymentSelection,
        rank_programs: tuple[SwizzleFusionRankProgramAdapter, ...],
        buffer_requirements: tuple[SwizzleBufferRequirement, ...],
    ) -> "SwizzleFusionPlanAdapter":
        semantic = {
            "deployment_selection": deployment_selection,
            "rank_programs": rank_programs,
            "buffer_requirements": buffer_requirements,
        }
        result = cls(
            schema_version=SWIZZLE_FUSION_PLAN_ADAPTER_SCHEMA_VERSION,
            id=stable_artifact_id(
                "swizzle_fusion_plan_adapter",
                semantic,
                schema_version=SWIZZLE_FUSION_PLAN_ADAPTER_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "deployment_selection": self.deployment_selection,
            "rank_programs": self.rank_programs,
            "buffer_requirements": self.buffer_requirements,
        }

    @property
    def decision(self) -> SwizzleDecision:
        return self.deployment_selection.economic_decision

    @property
    def candidate(self) -> SwizzleCandidate:
        return self.deployment_selection.candidate

    @property
    def economic_decision_ref(self) -> str:
        return self.deployment_selection.economic_decision_ref

    @property
    def source_ir1_id(self) -> str:
        return self.decision.problem.source_ir1_id

    @property
    def fused_op_id(self) -> str:
        return self.decision.problem.fused_op_id

    @property
    def group_ref(self) -> str:
        return self.decision.problem.group.group_ref

    @property
    def pattern(self) -> FusionPattern:
        return self.candidate.pattern

    @property
    def algorithm(self) -> SwizzleAlgorithm:
        return self.candidate.algorithm

    def validate(self, path: str = "swizzle_fusion_plan_adapter") -> None:
        if self.schema_version != SWIZZLE_FUSION_PLAN_ADAPTER_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        self.deployment_selection.validate(f"{path}.deployment_selection")
        if self.buffer_requirements != self.candidate.buffer_requirements:
            raise SchemaError(
                "buffer requirements must be retained losslessly",
                path=f"{path}.buffer_requirements",
            )

        routes = {
            route.id: (
                route.source_rank,
                route.destination_rank,
                route.die_path,
            )
            for route in self.decision.problem.group.routes
        }
        if tuple(program.rank for program in self.rank_programs) != tuple(
            program.rank for program in self.candidate.rank_programs
        ):
            raise SchemaError(
                "rank programs must preserve candidate rank order",
                path=f"{path}.rank_programs",
            )
        for index, (materialized, source) in enumerate(
            zip(self.rank_programs, self.candidate.rank_programs, strict=True)
        ):
            materialized.validate(
                route_index=routes,
                path=f"{path}.rank_programs[{index}]",
            )
            if tuple(action.source_action for action in materialized.actions) != source.actions:
                raise SchemaError(
                    "actions must preserve the candidate program losslessly",
                    path=f"{path}.rank_programs[{index}].actions",
                )

        expected_id = stable_artifact_id(
            "swizzle_fusion_plan_adapter",
            self._semantic_key(),
            schema_version=SWIZZLE_FUSION_PLAN_ADAPTER_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )


def _member_ref(decision: SwizzleDecision, action: SwizzleActionWitness) -> str:
    problem = decision.problem
    if action.kind is SwizzleActionKind.COMP:
        return problem.gemm.node_ref
    if problem.pattern is FusionPattern.AG_GEMM and action.kind in (
        SwizzleActionKind.REDUCE,
        SwizzleActionKind.LOCAL_COPY,
    ):
        return problem.gemm.node_ref
    return problem.collective.node_ref


def materialize_swizzle_selection(
    selection: SwizzleDeploymentSelection,
) -> SwizzleFusionPlanAdapter:
    """Close one typed deployment selection into a route-bound W7 adapter."""

    if type(selection) is not SwizzleDeploymentSelection:
        raise SchemaError(
            "must be a SwizzleDeploymentSelection", path="deployment_selection"
        )
    selection.validate("deployment_selection")
    decision = selection.economic_decision
    selected = selection.candidate

    route_index = {route.id: route for route in decision.problem.group.routes}
    programs = []
    for program in selected.rank_programs:
        actions = []
        for action in program.actions:
            route = route_index.get(action.route_ref or "")
            actions.append(
                SwizzleFusionActionAdapter(
                    source_action=action,
                    fusion_kind=_ACTION_KIND[action.kind],
                    member_ref=_member_ref(decision, action),
                    expected_route=() if route is None else route.die_path,
                )
            )
        programs.append(
            SwizzleFusionRankProgramAdapter(
                rank=program.rank,
                actions=tuple(actions),
            )
        )
    return SwizzleFusionPlanAdapter.create(
        deployment_selection=selection,
        rank_programs=tuple(programs),
        buffer_requirements=selected.buffer_requirements,
    )


def materialize_swizzle_decision(
    decision: SwizzleDecision,
) -> SwizzleFusionPlanAdapter:
    """Materialize only the planner's economically selected fused candidate."""

    if type(decision) is not SwizzleDecision:
        raise SchemaError("must be a SwizzleDecision", path="decision")
    decision.validate("decision")
    selected = next(
        item
        for item in decision.ranked_candidates
        if item.id == decision.selected_candidate_ref
    )
    if selected.algorithm is SwizzleAlgorithm.UNFUSED:
        raise SchemaError(
            "UNFUSED selection must remain on the unfused pipeline",
            path="decision.selected_candidate_ref",
        )
    return materialize_swizzle_selection(
        SwizzleDeploymentSelection.create(
            economic_decision=decision,
            candidate=selected,
            reason=SwizzleDeploymentReason.ECONOMIC_DECISION,
        )
    )


def force_swizzle_deployment(
    economic_decision: SwizzleDecision,
    *,
    candidate_ref: str | None = None,
) -> SwizzleFusionPlanAdapter:
    """Explicitly deploy a ranked fused candidate without mutating economics."""

    if type(economic_decision) is not SwizzleDecision:
        raise SchemaError("must be a SwizzleDecision", path="economic_decision")
    economic_decision.validate("economic_decision")
    if candidate_ref is None:
        candidate = next(
            (
                item
                for item in economic_decision.ranked_candidates
                if item.algorithm is not SwizzleAlgorithm.UNFUSED
            ),
            None,
        )
    else:
        validate_nonempty(candidate_ref, "candidate_ref")
        candidate = next(
            (
                item
                for item in economic_decision.ranked_candidates
                if item.id == candidate_ref
            ),
            None,
        )
    if candidate is None:
        raise SchemaError(
            "economic decision has no matching fused candidate", path="candidate_ref"
        )
    return materialize_swizzle_selection(
        SwizzleDeploymentSelection.create(
            economic_decision=economic_decision,
            candidate=candidate,
            reason=SwizzleDeploymentReason.FORCED_BY_POLICY,
        )
    )


# Minimum shared-schema work needed to convert this adapter into FusionPlan.
# Kept executable/documented here so W7's boundary cannot silently drift.
FUSION_PLAN_EXTENSION_FIELDS = (
    "pattern",
    "algorithm",
    "decision_ref",
    "candidate_ref",
    "swizzle_rank_programs",
    "buffer_requirements",
)
FUSION_ACTION_EXTENSION_FIELDS = ("phase", "source_action_ref", "flops")


__all__ = [
    "FUSION_ACTION_EXTENSION_FIELDS",
    "FUSION_PLAN_EXTENSION_FIELDS",
    "SWIZZLE_FUSION_PLAN_ADAPTER_SCHEMA_VERSION",
    "SwizzleFusionActionAdapter",
    "SwizzleFusionPlanAdapter",
    "SwizzleFusionRankProgramAdapter",
    "force_swizzle_deployment",
    "materialize_swizzle_decision",
    "materialize_swizzle_selection",
]
