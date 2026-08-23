"""Fail-closed local (within-one-die) transport contract.

This is deliberately separate from :mod:`ir2`: ``SemanticFlow`` remains an
inter-die abstraction and continues to reject equal die endpoints.  A future
refine pass can carry one of these plans while it materializes explicit local
send/receive tasks and the corresponding runtime ABI.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Mapping

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64, validate_unique_ids
if TYPE_CHECKING:
    from .ir1 import IR1
    from .ir2 import IR2ProjectionResult, IntraDieDAG, IntraDieScheduleSet


LOCAL_TRANSPORT_PLAN_SCHEMA_VERSION = "wafer_frontend.local_transport_plan/v1alpha1"


class LocalEventPhase(str, Enum):
    """The only two event edges allowed for one local transfer."""

    SEND_COMPLETE = "send_complete"
    RECV_READY = "recv_ready"


@dataclass(frozen=True, slots=True)
class LocalFlow:
    """One payload transfer between distinct runtime cores on one die."""

    id: str
    source_task_id: str
    destination_task_id: str
    source_core_id: int
    destination_core_id: int
    value_id: str
    tensor_slice: "TensorSlice"
    bytes: int
    dtype: DType
    route_id: str
    send_event_id: str
    recv_event_id: str

    def validate(self, path: str) -> None:
        for name in (
            "id", "source_task_id", "destination_task_id", "value_id",
            "route_id", "send_event_id", "recv_event_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if self.source_task_id == self.destination_task_id:
            raise SchemaError("local flow endpoints must be distinct tasks", path=path)
        if self.send_event_id == self.recv_event_id:
            raise SchemaError("send and receive events must be distinct", path=path)
        for name in ("source_core_id", "destination_core_id", "bytes"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.source_core_id == self.destination_core_id:
            raise SchemaError("local flow endpoints must be distinct cores", path=path)
        if self.bytes == 0:
            raise SchemaError("must be greater than zero", path=f"{path}.bytes")
        if type(self.dtype) is not DType:
            raise SchemaError("must be a DType", path=f"{path}.dtype")
        from .ir2 import TensorSlice
        if type(self.tensor_slice) is not TensorSlice:
            raise SchemaError("must be a TensorSlice", path=f"{path}.tensor_slice")
        self.tensor_slice.validate(f"{path}.tensor_slice")
        if self.tensor_slice.value_id != self.value_id:
            raise SchemaError("tensor_slice must name value_id", path=f"{path}.tensor_slice.value_id")


@dataclass(frozen=True, slots=True)
class LocalNocRoute:
    """Backend-v1 local NoC route; its exact X-then-Y form is checked in-plan."""

    id: str
    flow_id: str
    noc_path: tuple[tuple[int, int], ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.flow_id, f"{path}.flow_id")
        if not self.noc_path:
            raise SchemaError("must contain at least one NoC coordinate", path=f"{path}.noc_path")
        for index, coordinate in enumerate(self.noc_path):
            if type(coordinate) is not tuple or len(coordinate) != 2:
                raise SchemaError("must be an (x, y) tuple", path=f"{path}.noc_path[{index}]")
            for axis, value in enumerate(coordinate):
                validate_uint64(value, f"{path}.noc_path[{index}][{axis}]")
        for index, (left, right) in enumerate(zip(self.noc_path, self.noc_path[1:])):
            if abs(left[0] - right[0]) + abs(left[1] - right[1]) != 1:
                raise SchemaError("NoC hops must be Manhattan-adjacent", path=f"{path}.noc_path[{index + 1}]")


@dataclass(frozen=True, slots=True)
class LocalEvent:
    """A source completion or destination readiness event owned by one task."""

    id: str
    flow_id: str
    task_id: str
    phase: LocalEventPhase

    def validate(self, path: str) -> None:
        for name in ("id", "flow_id", "task_id"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.phase) is not LocalEventPhase:
            raise SchemaError("must be a LocalEventPhase", path=f"{path}.phase")


def _xy_path(source: tuple[int, int], destination: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, y = source
    end_x, end_y = destination
    result = [(x, y)]
    while x != end_x:
        x += 1 if end_x > x else -1
        result.append((x, y))
    while y != end_y:
        y += 1 if end_y > y else -1
        result.append((x, y))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class LocalTransportPlan:
    """A closed set of explicit local flow, route and event declarations.

    ``validate_against_placement`` is intentionally required before a consumer
    may lower this plan: basic validation cannot prove that task/core IDs name
    the refined graph being compiled.
    """

    schema_version: str
    producer_pass: str
    id: str
    source_dag_id: str
    flows: tuple[LocalFlow, ...]
    routes: tuple[LocalNocRoute, ...]
    events: tuple[LocalEvent, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic_key: object) -> "LocalTransportPlan":
        return cls(
            schema_version=LOCAL_TRANSPORT_PLAN_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("local_transport_plan", semantic_key, schema_version=LOCAL_TRANSPORT_PLAN_SCHEMA_VERSION),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {"source_dag_id": self.source_dag_id, "flows": self.flows, "routes": self.routes, "events": self.events}

    def validate(self, path: str = "local_transport_plan") -> None:
        if self.schema_version != LOCAL_TRANSPORT_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        validate_nonempty(self.source_dag_id, f"{path}.source_dag_id")
        if not self.flows:
            raise SchemaError("must contain at least one local flow", path=f"{path}.flows")
        flow_index = validate_unique_ids(self.flows, f"{path}.flows")
        route_index = validate_unique_ids(self.routes, f"{path}.routes")
        event_index = validate_unique_ids(self.events, f"{path}.events")
        for index, flow in enumerate(self.flows):
            flow.validate(f"{path}.flows[{index}]")
            route = route_index.get(flow.route_id)
            if route is None or route.flow_id != flow.id:
                raise SchemaError("flow must reference its unique route", path=f"{path}.flows[{index}].route_id")
            send = event_index.get(flow.send_event_id)
            recv = event_index.get(flow.recv_event_id)
            if send is None or (send.flow_id, send.task_id, send.phase) != (flow.id, flow.source_task_id, LocalEventPhase.SEND_COMPLETE):
                raise SchemaError("flow send event must be source SEND_COMPLETE", path=f"{path}.flows[{index}].send_event_id")
            if recv is None or (recv.flow_id, recv.task_id, recv.phase) != (flow.id, flow.destination_task_id, LocalEventPhase.RECV_READY):
                raise SchemaError("flow receive event must be destination RECV_READY", path=f"{path}.flows[{index}].recv_event_id")
        for index, route in enumerate(self.routes):
            route.validate(f"{path}.routes[{index}]")
            if route.flow_id not in flow_index:
                raise SchemaError("route references an unknown flow", path=f"{path}.routes[{index}].flow_id")
        for index, event in enumerate(self.events):
            event.validate(f"{path}.events[{index}]")
            if event.flow_id not in flow_index:
                raise SchemaError("event references an unknown flow", path=f"{path}.events[{index}].flow_id")
        if {route.flow_id for route in self.routes} != set(flow_index):
            raise SchemaError("must contain exactly one route for every local flow", path=f"{path}.routes")
        expected_event_ids = {
            event_id
            for flow in self.flows
            for event_id in (flow.send_event_id, flow.recv_event_id)
        }
        if set(event_index) != expected_event_ids:
            raise SchemaError("must contain exactly source/destination events for every local flow", path=f"{path}.events")
        if tuple(sorted(self.flows, key=lambda item: item.id)) != self.flows or tuple(sorted(self.routes, key=lambda item: item.id)) != self.routes or tuple(sorted(self.events, key=lambda item: item.id)) != self.events:
            raise SchemaError("flows, routes and events must use canonical id order", path=path)
        expected_id = stable_artifact_id("local_transport_plan", self._semantic_key(), schema_version=LOCAL_TRANSPORT_PLAN_SCHEMA_VERSION)
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against_placement(
        self,
        *,
        task_ids: set[str],
        placements: Mapping[str, int],
        core_noc_coords: Mapping[int, tuple[int, int]],
        path: str = "local_transport_plan",
    ) -> None:
        self.validate(path)
        for index, flow in enumerate(self.flows):
            flow_path = f"{path}.flows[{index}]"
            if flow.source_task_id not in task_ids or flow.destination_task_id not in task_ids:
                raise SchemaError("local flow references a task outside the refined DAG", path=flow_path)
            if placements.get(flow.source_task_id) != flow.source_core_id or placements.get(flow.destination_task_id) != flow.destination_core_id:
                raise SchemaError("local flow core endpoints disagree with task placement", path=flow_path)
            source = core_noc_coords.get(flow.source_core_id)
            destination = core_noc_coords.get(flow.destination_core_id)
            if source is None or destination is None:
                raise SchemaError("local flow references an unknown runtime core", path=flow_path)
            route = next(route for route in self.routes if route.id == flow.route_id)
            if route.noc_path != _xy_path(source, destination):
                raise SchemaError("NoC route must be exact backend-v1 X-then-Y", path=f"{flow_path}.route_id")

    def validate_against_schedule(
        self,
        *,
        projection: "IR2ProjectionResult",
        schedule_set: "IntraDieScheduleSet",
        ir1: "IR1",
        path: str = "local_transport_plan",
    ) -> None:
        """Close a local plan against the actual selected core placement.

        A graph refiner can describe a local transfer before core placement is
        known; the schedule remains the source of truth for runtime core IDs.
        """
        from .ir1 import IR1
        from .ir2 import IR2ProjectionResult, IntraDieScheduleSet

        if type(projection) is not IR2ProjectionResult:
            raise SchemaError("must be an IR2ProjectionResult", path="projection")
        if type(schedule_set) is not IntraDieScheduleSet:
            raise SchemaError("must be an IntraDieScheduleSet", path="schedule_set")
        if type(ir1) is not IR1:
            raise SchemaError("must be an IR1 artifact", path="ir1")
        projection.validate("projection")
        ir1.validate("ir1")
        schedule_set.validate_against(projection, ir1, "schedule_set")
        dag = next((item for item in projection.dags if item.id == self.source_dag_id), None)
        schedule = next(
            (item for item in schedule_set.schedules if item.dag_id == self.source_dag_id),
            None,
        )
        if dag is None or schedule is None:
            raise SchemaError("must reference one scheduled projection DAG", path=f"{path}.source_dag_id")
        die = next((item for item in ir1.fabric.dies if item.id == dag.die_id), None)
        if die is None:
            raise SchemaError("DAG references an unknown die", path=f"{path}.source_dag_id")
        self.validate_against_placement(
            task_ids={task.id for task in dag.tasks},
            placements={item.task_id: item.core_id for item in schedule.placements},
            core_noc_coords={core.runtime_core_id: core.noc_coord for core in die.cores},
            path=path,
        )

    def materialize_into_dag(self, dag: "IntraDieDAG") -> "IntraDieDAG":
        """Inject a constrained LOCAL_SEND/RECV/WAIT chain into one IR2 DAG.

        This is intentionally schema-only: local payload ownership and backend
        lowering remain unchanged.  It supports ordinary JSON-coarse endpoints
        and makes the sole legal cross-core dependency explicit.
        """
        from .ir2 import (
            IntraDieDAG, IntraDieRegion, OrdinaryNodeOrigin, RegionLowering,
            SemanticTask, SemanticTaskKind,
        )

        if type(dag) is not IntraDieDAG:
            raise SchemaError("must be an IntraDieDAG", path="dag")
        self.validate("local_transport_plan")
        if self.source_dag_id != dag.id:
            raise SchemaError("plan references a different source DAG", path="local_transport_plan.source_dag_id")
        tasks = {task.id: task for task in dag.tasks}
        regions = list(dag.regions)
        additions: list[SemanticTask] = []
        replacements: dict[str, SemanticTask] = {}
        for flow in self.flows:
            source, destination = tasks.get(flow.source_task_id), tasks.get(flow.destination_task_id)
            if source is None or destination is None:
                raise SchemaError("local flow endpoint task is absent", path=f"local_transport_plan.flows[{flow.id}]")
            if not isinstance(source.origin_ref, OrdinaryNodeOrigin) or not isinstance(destination.origin_ref, OrdinaryNodeOrigin):
                raise SchemaError("MVP local materialization requires ordinary endpoint origins", path=f"local_transport_plan.flows[{flow.id}]")
            send_id, recv_id, wait_id = (f"local_send_{flow.id}", f"local_recv_{flow.id}", f"local_wait_{flow.id}")
            region_id = f"local_transport_region_{flow.id}"
            if any(item in tasks for item in (send_id, recv_id, wait_id)) or any(region.id == region_id for region in regions):
                raise SchemaError("local transport materialization id collision", path=f"local_transport_plan.flows[{flow.id}]")
            local_flow_id = f"local.{flow.id}"
            send = SemanticTask(
                id=send_id, kind=SemanticTaskKind.LOCAL_SEND,
                origin_ref=source.origin_ref, region_id=region_id, op_kind=None,
                member_id=None, flow_id=local_flow_id, chunk_id=None,
                collective_step=None, source_rank=None, destination_rank=None,
                tensor_slice=flow.tensor_slice, bytes=flow.bytes, dtype=flow.dtype,
                shape=flow.tensor_slice.shape, read_values=(), write_values=(),
                compute=None, reduction=None, sync=None, deps=(source.id,),
            )
            recv = SemanticTask(
                id=recv_id, kind=SemanticTaskKind.LOCAL_RECV,
                origin_ref=destination.origin_ref, region_id=region_id, op_kind=None,
                member_id=None, flow_id=local_flow_id, chunk_id=None,
                collective_step=None, source_rank=None, destination_rank=None,
                tensor_slice=flow.tensor_slice, bytes=flow.bytes, dtype=flow.dtype,
                shape=flow.tensor_slice.shape, read_values=(), write_values=(),
                compute=None, reduction=None, sync=None, deps=(send_id,),
            )
            wait = SemanticTask(
                id=wait_id, kind=SemanticTaskKind.LOCAL_WAIT,
                origin_ref=destination.origin_ref, region_id=region_id, op_kind=None,
                member_id=None, flow_id=local_flow_id, chunk_id=None,
                collective_step=None, source_rank=None, destination_rank=None,
                tensor_slice=None, bytes=0, dtype=None, shape=(), read_values=(),
                write_values=(), compute=None, reduction=None, sync=None,
                deps=(recv_id,),
            )
            additions.extend((send, recv, wait))
            destination_deps = replacements.get(destination.id, destination).deps
            replacements[destination.id] = replace(destination, deps=tuple(dep for dep in destination_deps if dep != source.id) + (wait_id,))
            regions.append(IntraDieRegion(region_id, None, None, RegionLowering.JSON_COARSE, (send_id, recv_id, wait_id)))
        materialized_tasks = tuple(replacements.get(task.id, task) for task in dag.tasks) + tuple(additions)
        semantic_key = dag._semantic_key()
        semantic_key.update(tasks=materialized_tasks, regions=tuple(regions))
        result = IntraDieDAG.create(producer_pass=dag.producer_pass, **semantic_key)
        result.validate("materialized_intra_die_dag")
        self.validate_materialized_dag(result)
        return result

    def validate_materialized_dag(self, dag: "IntraDieDAG") -> None:
        """Check the exact LOCAL_SEND -> LOCAL_RECV -> LOCAL_WAIT chain."""
        from .ir2 import IntraDieDAG, SemanticTaskKind

        if type(dag) is not IntraDieDAG:
            raise SchemaError("must be an IntraDieDAG", path="dag")
        self.validate("local_transport_plan")
        dag.validate("materialized_intra_die_dag")
        tasks = {task.id: task for task in dag.tasks}
        for flow in self.flows:
            send_id, recv_id, wait_id = (
                f"local_send_{flow.id}", f"local_recv_{flow.id}",
                f"local_wait_{flow.id}",
            )
            send, recv, wait = (tasks.get(item) for item in (send_id, recv_id, wait_id))
            if (
                send is None
                or recv is None
                or wait is None
                or (send.kind, recv.kind, wait.kind)
                != (
                    SemanticTaskKind.LOCAL_SEND,
                    SemanticTaskKind.LOCAL_RECV,
                    SemanticTaskKind.LOCAL_WAIT,
                )
                or any(task.flow_id != f"local.{flow.id}" for task in (send, recv, wait))
                or send.deps != (flow.source_task_id,)
                or recv.deps != (send_id,)
                or wait.deps != (recv_id,)
                or wait_id not in (tasks.get(flow.destination_task_id).deps if tasks.get(flow.destination_task_id) is not None else ())
            ):
                raise SchemaError("local flow does not have one exact materialized task chain", path=f"local_transport_plan.flows[{flow.id}]")


__all__ = [
    "LOCAL_TRANSPORT_PLAN_SCHEMA_VERSION", "LocalEventPhase", "LocalFlow",
    "LocalNocRoute", "LocalEvent", "LocalTransportPlan",
]
