"""Immutable source-bound cross-DP physical route plan, not an execution receipt."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import SchemaError
from .action import ChunkSlice, RankProgram
from .ir1 import PairRoute, PhysicalGroup

if TYPE_CHECKING:
    from .flexible_dense_train import FlexibleDenseTrainPlan
    from .placed_ir1 import TrainPlacedIR1
    from .placement import PlacementContext


@dataclass(frozen=True, slots=True)
class DenseDP2GradientRoute:
    state_ref: str
    tp_shard: int
    step: int
    gradient_bytes: int
    group_ref: str
    local_wgrad_refs: tuple[str, str]
    sync_refs: tuple[str, str]
    optimizer_refs: tuple[str, str]
    reduce_route: PairRoute
    broadcast_route: PairRoute
    chunk: ChunkSlice
    rank_programs: tuple[RankProgram, RankProgram]


@dataclass(frozen=True, slots=True)
class DenseDP2RoutePlan:
    id: str
    source_ir0_id: str
    placed_ir1_id: str
    fabric_id: str
    dp_groups: tuple[PhysicalGroup, PhysicalGroup]
    gradients: tuple[DenseDP2GradientRoute, ...]

    @property
    def schema_version(self) -> str:
        """Immutable source identity for the linked ProgramArtifact trust anchor."""
        return "wafer_frontend.dense_dp2_route_plan/v1alpha1"

    def validate_against(
        self, plan: FlexibleDenseTrainPlan, placed: TrainPlacedIR1,
        context: PlacementContext,
    ) -> None:
        """Rebuild every owner, source producer, real fabric route and byte."""
        from ..passes.full_dense_training_dp2_routes import build_dense_dp2_route_plan
        expected = build_dense_dp2_route_plan(plan, placed, context)
        if self != expected:
            raise SchemaError("DP2 gradient route/source/owner/bytes drifted",
                              path="dense_dp2_route_plan")


__all__ = ["DenseDP2GradientRoute", "DenseDP2RoutePlan"]
