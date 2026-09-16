"""Project each true cross-replica N4 action into a typed N5 physical task."""
from __future__ import annotations

from ..errors import SchemaError
from ..schema.action import FusionActionKind
from ..schema.dense_dp_sync_routes import DenseDP2RoutePlan
from ..schema.dense_dp_sync_tasks import DenseDP2ProjectedTask, DenseDP2ProjectedTasks
from ..schema.ir2 import (
    OriginKind, SemanticFlow, SemanticTask, SemanticTaskKind,
    StandaloneNodeOrigin, TensorSlice, canonical_semantic_flow_id,
)
from ..schema.ir0 import OpKind


def project_dense_dp2_tasks(source: DenseDP2RoutePlan) -> DenseDP2ProjectedTasks:
    """Preserve true die, rank, route, byte and producer dependencies, exactly."""
    if not source.gradients or len(source.dp_groups) != 2:
        raise SchemaError("DP2 requires two physical gradient groups and gradients",
                          path="dense_dp2_route_plan")
    group_index = {group.id: group for group in source.dp_groups}
    result: list[DenseDP2ProjectedTask] = []
    for gradient in source.gradients:
        group = group_index.get(gradient.group_ref)
        if group is None:
            raise SchemaError("unknown physical gradient group", path="gradient.group_ref")
        rank_to_die = {place.rank: place.die_id for place in group.placements}
        routes = {(route.source_rank, route.destination_rank,
                   route.die_path): route for route in group.embedding.routes}
        channel_senders = {
            action.logical_channel: (rank_program.rank, action)
            for rank_program in gradient.rank_programs
            for action in rank_program.actions
            if action.kind is FusionActionKind.SEND
        }
        for rank_program in gradient.rank_programs:
            dp = rank_program.rank
            for action in rank_program.actions:
                origin = StandaloneNodeOrigin(
                    OriginKind.STANDALONE_COLLECTIVE,
                    source.id, dp, action.id,
                )
                tensor_slice = (
                    TensorSlice(gradient.chunk.value_id, gradient.chunk.offset,
                                gradient.chunk.shape)
                    if action.slice_ref is not None else None
                )
                flow_id = None
                flow = None
                source_rank = destination_rank = None
                if action.kind in (FusionActionKind.SEND, FusionActionKind.RECV):
                    sender_rank, sender = channel_senders[action.logical_channel]
                    source_rank, destination_rank = (
                        sender_rank, sender.peer_rank,
                    )
                    route = routes.get((source_rank, destination_rank,
                                        action.expected_route))
                    if route is None or sender.bytes != action.bytes:
                        raise SchemaError("DP2 flow has no exact physical route/bytes",
                                          path=f"{gradient.group_ref}.{action.id}")
                    sender_origin = StandaloneNodeOrigin(
                        OriginKind.STANDALONE_COLLECTIVE,
                        source.id, sender_rank, sender.id,
                    )
                    flow_id = canonical_semantic_flow_id(
                        sender_origin, action.logical_channel,
                    )
                    flow = SemanticFlow(
                        flow_id, action.logical_channel, route.id,
                        source_rank, destination_rank,
                        action.expected_route[0], action.expected_route[-1],
                        action.expected_route, tensor_slice, action.bytes,
                        action.dtype, (f"task.{action.id}",),
                    )
                    flow.validate(f"dp2.tasks.{action.id}.flow")
                task = SemanticTask(
                    id=f"task.{action.id}",
                    kind=SemanticTaskKind(action.kind.value),
                    origin_ref=origin,
                    region_id=f"region.dp_gradient.{source.id}.die.{rank_to_die[dp]}",
                    op_kind=OpKind.COLLECTIVE,
                    member_id=action.member_id,
                    flow_id=flow_id,
                    chunk_id=action.chunk_id,
                    collective_step=action.collective_step,
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    tensor_slice=tensor_slice,
                    bytes=action.bytes,
                    dtype=action.dtype,
                    shape=gradient.chunk.shape if tensor_slice is not None else (),
                    read_values=action.reads,
                    write_values=action.writes,
                    compute=action.compute,
                    reduction=action.reduction,
                    sync=action.sync,
                    deps=tuple(f"task.{ref}" for ref in action.deps),
                )
                task.validate(f"dp2.tasks.{action.id}")
                producer_ref = (
                    f"task.{gradient.local_wgrad_refs[dp]}.rank."
                    f"{gradient.tp_shard}.comp"
                    if action.kind is FusionActionKind.LOCAL_COPY
                    or (action.kind is FusionActionKind.SEND and dp == 1)
                    else None
                )
                consumer_ref = (
                    f"task.{gradient.optimizer_refs[dp]}.rank."
                    f"{gradient.tp_shard}.comp"
                    if action.kind is FusionActionKind.REDUCE
                    or (action.kind is FusionActionKind.WAIT and dp == 1)
                    else None
                )
                result.append(DenseDP2ProjectedTask(
                    rank_to_die[dp], dp, gradient.state_ref,
                    gradient.step, gradient.tp_shard,
                    gradient.sync_refs[dp], task, flow,
                    producer_ref, consumer_ref,
                ))
    ids = [item.task.id for item in result]
    if len(set(ids)) != len(ids) or len(result) != 8 * len(source.gradients):
        raise SchemaError("DP2 projection must exactly cover eight actions per gradient",
                          path="dense_dp2_projected_tasks")
    return DenseDP2ProjectedTasks(source.id, tuple(result))


__all__ = ["project_dense_dp2_tasks"]
