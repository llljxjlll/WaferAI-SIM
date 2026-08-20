"""Frozen policy and deterministic-transform interfaces; no implementations live here."""

from __future__ import annotations

from typing import Protocol

from ..schema.action import FusionPlan, StandaloneCollectivePlan
from ..schema.common import ProfileKey
from ..schema.global_action import GlobalActionDAG
from ..schema.ir1 import FusedOpSkeleton, IR1, PhysicalNode
from ..schema.ir2 import IR2ProjectionResult, IntraDieScheduleSet
from ..schema.state_transfer import StateTransferLike


class FusionPartition(Protocol):
    def run(self, ir1: IR1) -> tuple[FusedOpSkeleton, ...]: ...


class InterDiePolicy(Protocol):
    def plan(
        self,
        ir1: IR1,
        fused_op: FusedOpSkeleton,
        profile: ProfileKey,
    ) -> FusionPlan: ...


class StandaloneCollectivePolicy(Protocol):
    def plan(
        self,
        ir1: IR1,
        collective_op: PhysicalNode,
        profile: ProfileKey,
    ) -> StandaloneCollectivePlan: ...


class ProjectToIR2(Protocol):
    def run(
        self,
        ir1: IR1,
        fusion_plans: tuple[FusionPlan, ...],
        standalone_plans: tuple[StandaloneCollectivePlan, ...],
        *,
        state_transfers: tuple[StateTransferLike, ...],
    ) -> IR2ProjectionResult: ...


class IntraDiePolicy(Protocol):
    def schedule(
        self,
        projection: IR2ProjectionResult,
        ir1: IR1,
    ) -> IntraDieScheduleSet: ...


class GlobalActionDAGBuilder(Protocol):
    def build(
        self,
        ir1: IR1,
        projection: IR2ProjectionResult,
        schedule_set: IntraDieScheduleSet,
    ) -> GlobalActionDAG: ...
