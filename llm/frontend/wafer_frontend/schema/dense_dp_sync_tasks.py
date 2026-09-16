"""Physical N5 task sidecar for cross-replica Dense gradient synchronization.

These tasks are deliberately separate from per-replica IntraDieDAGs until the
global dependency/buffer merger can bind both replica SGD consumers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..errors import SchemaError
from .ir2 import SemanticFlow, SemanticTask

if TYPE_CHECKING:
    from .dense_dp_sync_routes import DenseDP2RoutePlan


@dataclass(frozen=True, slots=True)
class DenseDP2ProjectedTask:
    die_id: int
    replica_index: int
    state_ref: str
    step: int
    tp_shard: int
    source_sync_ref: str
    task: SemanticTask
    flow: SemanticFlow | None
    producer_task_ref: str | None
    consumer_task_ref: str | None


@dataclass(frozen=True, slots=True)
class DenseDP2ProjectedTasks:
    source_route_plan_id: str
    tasks: tuple[DenseDP2ProjectedTask, ...]

    def validate_against(self, source: DenseDP2RoutePlan) -> None:
        from ..passes.full_dense_training_dp2_tasks import project_dense_dp2_tasks

        if self != project_dense_dp2_tasks(source):
            raise SchemaError(
                "physical DP2 task/action/route coverage must be exact",
                path="dense_dp2_projected_tasks",
            )


__all__ = ["DenseDP2ProjectedTask", "DenseDP2ProjectedTasks"]
