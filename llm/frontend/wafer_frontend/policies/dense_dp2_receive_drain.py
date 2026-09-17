"""Keep real DP broadcast receives ahead of cross-core HBM version waits.

The shared control input has finite credits.  A core blocked at EVENT_WAIT
cannot service an older P2P receive; source-bound DP broadcast RECV/WAIT pairs
on that core therefore drain before its first cross-core step-one parameter
LOAD.  Core order becomes an explicit GlobalAction predecessor chain.
"""
from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.dense_dp_sync_routes import DenseDP2RoutePlan
from ..schema.ir1 import IR1
from ..schema.ir2 import (
    IntraDieDAG, OrdinaryNodeOrigin, SemanticTaskKind, StandaloneNodeOrigin,
    StateIoOrigin,
)


def order_dense_dp2_receives_before_state_fences(
    dag: IntraDieDAG,
    ir1: IR1,
    route: DenseDP2RoutePlan,
    placement_by_task: dict[str, int],
    order_by_core: dict[int, tuple[str, ...]],
) -> dict[int, tuple[str, ...]]:
    """Move only independent, exact step-one DP broadcast RECV/WAIT pairs."""
    if dag.dp_gradient_plan_id != route.id:
        raise SchemaError("DP receive drain requires its exact route plan",
                          path="dag.dp_gradient_plan_id")
    tasks = {task.id: task for task in dag.tasks}
    accesses = {access.id: access for access in ir1.state_accesses}
    stores: dict[tuple[str, int], str] = {}
    loads: dict[tuple[str, int], str] = {}
    for task in dag.tasks:
        origin = task.origin_ref
        if not isinstance(origin, StateIoOrigin) or not origin.node_ref.startswith("sgd_update::"):
            continue
        access = accesses.get(origin.state_access_ref)
        if access is None:
            raise SchemaError("optimizer StateIO has no exact source access",
                              path=task.id)
        key = (access.state_ref, access.rank)
        if "::step0__dp" in origin.node_ref and task.kind is SemanticTaskKind.DMA_OUT:
            if key in stores:
                raise SchemaError("duplicate source step-zero state STORE", path=task.id)
            stores[key] = task.id
        elif "::step1__dp" in origin.node_ref and task.kind is SemanticTaskKind.DMA_IN:
            if key in loads:
                raise SchemaError("duplicate step-one state LOAD", path=task.id)
            loads[key] = task.id
    if not loads:
        return order_by_core
    pairs_by_core: dict[int, list[tuple[str, str]]] = defaultdict(list)
    loads_by_core: dict[int, list[str]] = defaultdict(list)
    for key, load_id in loads.items():
        store_id = stores.get(key)
        if store_id is None:
            raise SchemaError("step-one optimizer LOAD lacks source step-zero STORE",
                              path=load_id)
        core = placement_by_task[load_id]
        if core == placement_by_task[store_id]:
            continue
        load = tasks[load_id]
        origin = load.origin_ref
        assert isinstance(origin, StateIoOrigin)
        optimizer = tuple(task for task in dag.tasks
                          if task.kind is SemanticTaskKind.COMP
                          and isinstance(task.origin_ref, OrdinaryNodeOrigin)
                          and task.origin_ref.op_id == origin.node_ref
                          and placement_by_task.get(task.id) == core)
        if len(optimizer) != 1:
            raise SchemaError("cross-core state LOAD has no unique local optimizer",
                              path=load_id)
        waits = tuple(tasks[ref] for ref in optimizer[0].deps
                      if ref in tasks
                      and tasks[ref].kind is SemanticTaskKind.WAIT
                      and isinstance(tasks[ref].origin_ref, StandaloneNodeOrigin)
                      and tasks[ref].origin_ref.collective_plan_id == route.id
                      and tasks[ref].origin_ref.action_id.endswith(".broadcast_wait")
                      and "::step1." in tasks[ref].origin_ref.action_id)
        if len(waits) != 1 or len(waits[0].deps) != 1:
            raise SchemaError("cross-core optimizer must consume one real DP broadcast WAIT",
                              path=load_id)
        recv_id = waits[0].deps[0]
        recv = tasks.get(recv_id)
        if (recv is None or recv.kind is not SemanticTaskKind.RECV
                or not isinstance(recv.origin_ref, StandaloneNodeOrigin)
                or recv.origin_ref.collective_plan_id != route.id
                or placement_by_task.get(recv_id) != core
                or placement_by_task.get(waits[0].id) != core):
            raise SchemaError("DP broadcast WAIT lacks same-core source RECV",
                              path=load_id)
        pairs_by_core[core].append((recv_id, waits[0].id))
        loads_by_core[core].append(load_id)
    result = dict(order_by_core)
    for core, pairs in pairs_by_core.items():
        order = order_by_core[core]
        position = {task_id: index for index, task_id in enumerate(order)}
        if len(pairs) != len(set(pairs)) or len(pairs) != len(loads_by_core[core]):
            raise SchemaError("DP state fence receive pairs are not bijective",
                              path=f"core_orders[{core}]")
        pairs.sort(key=lambda pair: position[pair[0]])
        moving = {task_id for pair in pairs for task_id in pair}
        first_load = min(position[task_id] for task_id in loads_by_core[core])
        stable = [task_id for task_id in order if task_id not in moving]
        insertion = sum(position[task_id] < first_load for task_id in stable)
        reordered = tuple(stable[:insertion]
                          + [task_id for pair in pairs for task_id in pair]
                          + stable[insertion:])
        after = {task_id: index for index, task_id in enumerate(reordered)}
        if (len(reordered) != len(order) or set(reordered) != set(order)
                or max(after[pair[1]] for pair in pairs)
                   >= min(after[load_id] for load_id in loads_by_core[core])):
            raise SchemaError("all source DP receives must drain before first state fence",
                              path=f"core_orders[{core}]")
        for task_id in reordered:
            for dependency in tasks[task_id].deps:
                if dependency in after and after[dependency] >= after[task_id]:
                    raise SchemaError("DP receive drain breaks a local task dependency",
                                      path=f"core_orders[{core}]")
        result[core] = reordered
    return result


__all__ = ["order_dense_dp2_receives_before_state_fences"]
