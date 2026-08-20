"""Immutable, fully cross-validated inputs shared by every lowering backend."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.action import FusionPlan, StandaloneCollectivePlan
from ..schema.global_action import GlobalActionDAG
from ..schema.ir1 import IR1
from ..schema.ir2 import IR2ProjectionResult, IntraDieScheduleSet


@dataclass(frozen=True, slots=True)
class LoweringContext:
    """Lossless lowering boundary; backends never recover data from action ids."""

    ir1: IR1
    fusion_plans: tuple[FusionPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]
    projection: IR2ProjectionResult
    schedule_set: IntraDieScheduleSet
    global_dag: GlobalActionDAG

    def validate(self, path: str = "lowering_context") -> None:
        self.ir1.validate(f"{path}.ir1")
        fusion_ids = tuple(plan.id for plan in self.fusion_plans)
        standalone_ids = tuple(plan.id for plan in self.standalone_plans)
        if len(set(fusion_ids)) != len(fusion_ids):
            raise SchemaError("fusion plans must have unique ids", path=f"{path}.fusion_plans")
        if len(set(standalone_ids)) != len(standalone_ids):
            raise SchemaError("standalone plans must have unique ids", path=f"{path}.standalone_plans")
        if fusion_ids != self.projection.fusion_plan_ids:
            raise SchemaError(
                "fusion plan tuple must exactly preserve projection id order",
                path=f"{path}.fusion_plans",
            )
        if standalone_ids != self.projection.standalone_collective_plan_ids:
            raise SchemaError(
                "standalone plan tuple must exactly preserve projection id order",
                path=f"{path}.standalone_plans",
            )
        if tuple(schedule.dag_id for schedule in self.schedule_set.schedules) != tuple(
            dag.id for dag in self.projection.dags
        ):
            raise SchemaError(
                "schedule tuple must exactly preserve projection DAG order",
                path=f"{path}.schedule_set.schedules",
            )
        for index, plan in enumerate(self.fusion_plans):
            plan.validate_against(self.ir1, f"{path}.fusion_plans[{index}]")
        for index, plan in enumerate(self.standalone_plans):
            plan.validate_against(self.ir1, f"{path}.standalone_plans[{index}]")
        self.projection.validate_against(
            self.ir1,
            self.fusion_plans,
            self.standalone_plans,
            f"{path}.projection",
        )
        self.schedule_set.validate_against(
            self.projection,
            self.ir1,
            f"{path}.schedule_set",
        )
        self.global_dag.validate_against(
            self.ir1,
            self.projection,
            self.schedule_set,
            f"{path}.global_dag",
        )
