"""Typed integration-only forced Swizzle deployment evidence.

This carrier never rewrites an economic planner decision.  It records that the
planner selected UNFUSED, binds the best already-ranked fused candidate for
explicit integration coverage, and materializes only the route/member adapter
facts that W7 can currently prove.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    SwizzleFusionActionAdapter,
    SwizzleFusionRankProgramAdapter,
)
from llm.frontend.wafer_frontend.schema.action import FusionActionKind
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleBufferRequirement,
    SwizzleCandidate,
    SwizzleDecision,
    SwizzleDecisionReason,
)


FORCED_SWIZZLE_SELECTION_SCHEMA_VERSION = (
    "wafer_frontend.forced_swizzle_selection/v1alpha1"
)
FORCED_SWIZZLE_ADAPTER_SCHEMA_VERSION = (
    "wafer_frontend.forced_swizzle_adapter/v1alpha1"
)


class ForcedSwizzleReason(str, Enum):
    EXPLICIT_INTEGRATION_COVERAGE = "explicit_integration_coverage"


@dataclass(frozen=True, slots=True)
class ForcedSwizzleSelection:
    schema_version: str
    id: str
    economic_decision: SwizzleDecision
    candidate: SwizzleCandidate
    reason: ForcedSwizzleReason

    @classmethod
    def create(
        cls,
        *,
        economic_decision: SwizzleDecision,
        candidate: SwizzleCandidate,
        reason: ForcedSwizzleReason,
    ) -> "ForcedSwizzleSelection":
        semantic = {
            "economic_decision": economic_decision,
            "candidate": candidate,
            "reason": reason,
        }
        result = cls(
            schema_version=FORCED_SWIZZLE_SELECTION_SCHEMA_VERSION,
            id=stable_artifact_id(
                "forced_swizzle_selection",
                semantic,
                schema_version=FORCED_SWIZZLE_SELECTION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "economic_decision": self.economic_decision,
            "candidate": self.candidate,
            "reason": self.reason,
        }

    def validate(self, path: str = "forced_swizzle_selection") -> None:
        if self.schema_version != FORCED_SWIZZLE_SELECTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.economic_decision.validate(f"{path}.economic_decision")
        self.candidate.validate(f"{path}.candidate")
        if type(self.reason) is not ForcedSwizzleReason:
            raise SchemaError("must be a ForcedSwizzleReason", path=f"{path}.reason")
        economic = self.economic_decision
        selected = economic.ranked_candidates[0]
        if (
            selected.algorithm is not SwizzleAlgorithm.UNFUSED
            or economic.decision_reason is not SwizzleDecisionReason.NO_PROFITABLE_FUSION
        ):
            raise SchemaError(
                "forced coverage requires an honest NO_PROFITABLE_FUSION decision",
                path=f"{path}.economic_decision",
            )
        best_fused = next(
            (
                item
                for item in economic.ranked_candidates
                if item.algorithm is not SwizzleAlgorithm.UNFUSED
            ),
            None,
        )
        if best_fused is None or self.candidate != best_fused:
            raise SchemaError(
                "candidate must equal the economic decision's best ranked fused candidate",
                path=f"{path}.candidate",
            )
        expected = stable_artifact_id(
            "forced_swizzle_selection",
            self._semantic_key(),
            schema_version=FORCED_SWIZZLE_SELECTION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class ForcedSwizzlePlanAdapter:
    schema_version: str
    id: str
    selection: ForcedSwizzleSelection
    rank_programs: tuple[SwizzleFusionRankProgramAdapter, ...]
    buffer_requirements: tuple[SwizzleBufferRequirement, ...]

    @classmethod
    def create(
        cls,
        *,
        selection: ForcedSwizzleSelection,
        rank_programs: tuple[SwizzleFusionRankProgramAdapter, ...],
        buffer_requirements: tuple[SwizzleBufferRequirement, ...],
    ) -> "ForcedSwizzlePlanAdapter":
        semantic = {
            "selection": selection,
            "rank_programs": rank_programs,
            "buffer_requirements": buffer_requirements,
        }
        result = cls(
            schema_version=FORCED_SWIZZLE_ADAPTER_SCHEMA_VERSION,
            id=stable_artifact_id(
                "forced_swizzle_adapter",
                semantic,
                schema_version=FORCED_SWIZZLE_ADAPTER_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "selection": self.selection,
            "rank_programs": self.rank_programs,
            "buffer_requirements": self.buffer_requirements,
        }

    @property
    def decision(self) -> SwizzleDecision:
        return self.selection.economic_decision

    @property
    def candidate(self) -> SwizzleCandidate:
        return self.selection.candidate

    @property
    def pattern(self) -> FusionPattern:
        return self.candidate.pattern

    @property
    def algorithm(self) -> SwizzleAlgorithm:
        return self.candidate.algorithm

    def validate(self, path: str = "forced_swizzle_adapter") -> None:
        if self.schema_version != FORCED_SWIZZLE_ADAPTER_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.selection.validate(f"{path}.selection")
        candidate = self.candidate
        if self.buffer_requirements != candidate.buffer_requirements:
            raise SchemaError("buffer requirements were not retained", path=f"{path}.buffer_requirements")
        routes = {
            route.id: (route.source_rank, route.destination_rank, route.die_path)
            for route in self.decision.problem.group.routes
        }
        if tuple(item.rank for item in self.rank_programs) != tuple(
            item.rank for item in candidate.rank_programs
        ):
            raise SchemaError("rank order was not retained", path=f"{path}.rank_programs")
        for index, (adapted, source) in enumerate(
            zip(self.rank_programs, candidate.rank_programs, strict=True)
        ):
            adapted.validate(route_index=routes, path=f"{path}.rank_programs[{index}]")
            if tuple(item.source_action for item in adapted.actions) != source.actions:
                raise SchemaError("action witnesses were not retained", path=f"{path}.rank_programs[{index}]")
        expected = stable_artifact_id(
            "forced_swizzle_adapter",
            self._semantic_key(),
            schema_version=FORCED_SWIZZLE_ADAPTER_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


def _member_ref(decision: SwizzleDecision, kind: SwizzleActionKind) -> str:
    problem = decision.problem
    if kind is SwizzleActionKind.COMP:
        return problem.gemm.node_ref
    if problem.pattern is FusionPattern.AG_GEMM and kind in (
        SwizzleActionKind.REDUCE,
        SwizzleActionKind.LOCAL_COPY,
    ):
        return problem.gemm.node_ref
    return problem.collective.node_ref


def materialize_forced_swizzle(
    economic_decision: SwizzleDecision,
) -> ForcedSwizzlePlanAdapter:
    economic_decision.validate("economic_decision")
    candidate = next(
        item
        for item in economic_decision.ranked_candidates
        if item.algorithm is not SwizzleAlgorithm.UNFUSED
    )
    selection = ForcedSwizzleSelection.create(
        economic_decision=economic_decision,
        candidate=candidate,
        reason=ForcedSwizzleReason.EXPLICIT_INTEGRATION_COVERAGE,
    )
    route_index = {
        route.id: route for route in economic_decision.problem.group.routes
    }
    programs = tuple(
        SwizzleFusionRankProgramAdapter(
            rank=program.rank,
            actions=tuple(
                SwizzleFusionActionAdapter(
                    source_action=action,
                    fusion_kind=FusionActionKind(action.kind.value),
                    member_ref=_member_ref(economic_decision, action.kind),
                    expected_route=(
                        ()
                        if action.route_ref is None
                        else route_index[action.route_ref].die_path
                    ),
                )
                for action in program.actions
            ),
        )
        for program in candidate.rank_programs
    )
    return ForcedSwizzlePlanAdapter.create(
        selection=selection,
        rank_programs=programs,
        buffer_requirements=candidate.buffer_requirements,
    )


__all__ = [
    "FORCED_SWIZZLE_ADAPTER_SCHEMA_VERSION",
    "FORCED_SWIZZLE_SELECTION_SCHEMA_VERSION",
    "ForcedSwizzlePlanAdapter",
    "ForcedSwizzleReason",
    "ForcedSwizzleSelection",
    "materialize_forced_swizzle",
]
