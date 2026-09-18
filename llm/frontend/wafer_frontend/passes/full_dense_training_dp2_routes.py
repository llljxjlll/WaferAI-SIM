"""Source-bound physical DP2 gradient pair routes; no executable actions yet.

These paths join *different* TP replica groups.  They must never be placed
inside either replica's local N4 collective plan as a TP AllReduce.
"""
from __future__ import annotations


from ..errors import SchemaError
from ..schema.common import DType, MeshAxisName, RoundingMode, stable_artifact_id
from ..schema.action import (
    ChunkSlice, FusionActionKind, RankProgram, ReductionContract, SyncContract,
    _validate_rank_programs,
)
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.ir0 import CollectiveKind, OpKind, ReduceOp
from ..schema.dense_dp_sync_routes import DenseDP2GradientRoute, DenseDP2RoutePlan
from ..schema.placed_ir1 import TrainPlacedIR1
from ..schema.placement import PlacementContext
from ..schema.serde import canonical_digest
from .full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from .group_registry import _expected_group
from ..policies.naive_inter_die import _fusion_action



def _dp2_rank_programs(
    *, sync_ref: str, wgrad_value_ref: str, sync_value_ref: str,
    bytes: int, chunk: ChunkSlice, reduce_route: PairRoute,
    broadcast_route: PairRoute,
) -> tuple[RankProgram, RankProgram]:
    """Two true waves: remote contribution, rank-major FP32 SUM, return."""
    prefix = f"action.{sync_ref}"
    root_copy_ref = f"temp.{sync_ref}.dp0.local"
    root_remote_ref = f"temp.{sync_ref}.dp0.from.dp1"
    reduce_channel = f"channel.{sync_ref}.dp1.to.dp0"
    broadcast_channel = f"channel.{sync_ref}.dp0.to.dp1"
    def action(
        name: str, kind: FusionActionKind, *, dp: int,
        peer: int | None = None, route: PairRoute | None = None,
        reads: tuple[str, ...] = (), writes: tuple[str, ...] = (),
        channel: str | None = None, deps: tuple[str, ...] = (),
        wait: str | None = None,
        reduction: ReductionContract | None = None,
        step: int = 0,
    ):
        action_id = f"{prefix}.dp{dp}.{name}"
        return _fusion_action(
            action_id, kind, member_id=f"{sync_ref}__dp{dp}",
            chunk_id=0, collective_step=step,
            peer_rank=peer, expected_route=() if route is None else route.die_path,
            slice_ref=None if kind is FusionActionKind.WAIT else chunk.id,
            bytes=0 if kind is FusionActionKind.WAIT else bytes,
            dtype=None if kind is FusionActionKind.WAIT else DType.FP32,
            reads=reads, writes=writes, logical_channel=channel,
            reduction=reduction, deps=deps,
            sync=(SyncContract(f"event.{action_id}", wait, None)
                  if wait is not None else None),
        )
    root_local = action("local_copy", FusionActionKind.LOCAL_COPY,
                        dp=0, reads=(wgrad_value_ref,), writes=(root_copy_ref,))
    child_send = action("reduce_send", FusionActionKind.SEND, dp=1,
                        peer=0, route=reduce_route,
                        reads=(wgrad_value_ref,), channel=reduce_channel)
    root_recv = action("reduce_recv", FusionActionKind.RECV, dp=0,
                        peer=1, route=reduce_route,
                        writes=(root_remote_ref,), channel=reduce_channel)
    root_wait = action("reduce_wait", FusionActionKind.WAIT, dp=0,
                       deps=(root_recv.id,), wait=root_recv.sync.completion_event)
    root_reduce = action(
        "sum", FusionActionKind.REDUCE, dp=0, step=1,
        reads=(root_copy_ref, root_remote_ref), writes=(sync_value_ref,),
        deps=(root_local.id, root_wait.id),
        reduction=ReductionContract(
            ReduceOp.SUM, DType.FP32, DType.FP32, DType.FP32,
            RoundingMode.RNE, (0, 1),
        ),
    )
    root_send = action(
        "broadcast_send", FusionActionKind.SEND, dp=0, step=2,
        peer=1, route=broadcast_route, reads=(sync_value_ref,),
        channel=broadcast_channel, deps=(root_reduce.id,),
    )
    child_recv = action(
        "broadcast_recv", FusionActionKind.RECV, dp=1, step=2,
        peer=0, route=broadcast_route,
        writes=(sync_value_ref,), channel=broadcast_channel,
    )
    child_wait = action(
        "broadcast_wait", FusionActionKind.WAIT, dp=1,
        deps=(child_recv.id,), wait=child_recv.sync.completion_event,
    )
    programs = (
        RankProgram(0, (root_local, root_recv, root_wait, root_reduce, root_send)),
        RankProgram(1, (child_send, child_recv, child_wait)),
    )
    _validate_rank_programs(programs, (chunk,), path=f"dp_sync.{sync_ref}")
    return programs


def build_dense_dp2_route_plan(
    plan: FlexibleDenseTrainPlan, placed: TrainPlacedIR1,
    context: PlacementContext,
) -> DenseDP2RoutePlan:
    plan.validate("dp2_route_plan")
    placed.validate("dp2_placed")
    context.validate("dp2_placement")
    if (plan.spec.dp_degree != 2 or plan.spec.tp_degree < 1
            or len(placed.replicas) != 2
            or placed.placement_context_id != context.id):
        raise SchemaError("routes require exact TP/DP2 physical placement",
                          path="dense_dp2_route_plan")
    source = build_full_dense_training_two_step_ir0(plan)
    if (placed.source_ir0_id != source.id
            or source.producer_pass != "full_dense_training_two_step_dp2_source"):
        raise SchemaError("routes must bind the complete DP2 source IR0",
                          path="dense_dp2_route_plan.source")
    replica_graphs = tuple(item.graph for item in placed.replicas)
    for index, graph in enumerate(replica_graphs):
        group = graph.groups[0]
        if (graph.source_ir0_id != source.id
                or len(group.placements) != plan.spec.tp_degree
                or tuple(placement.rank for placement in group.placements)
                   != tuple(range(plan.spec.tp_degree))
                or group.axis is not MeshAxisName.TP
                or graph.fabric != context.fabric):
            raise SchemaError("DP2 replica TP group or fabric drifted",
                              path=f"dense_dp2_route_plan.replicas[{index}]")
    instance = source.instances[0]
    (mesh,) = instance.meshes
    groups = tuple(
        _expected_group(
            source, context, instance, mesh,
            rank_to_die_override=tuple(
                graph.groups[0].placements[tp].die_id for graph in replica_graphs
            ),
            group_id_suffix=f"__dp_gradient_tp{tp}",
            axis=MeshAxisName.DP,
        )
        for tp in range(plan.spec.tp_degree)
    )
    for tp, group in enumerate(groups):
        group.validate(f"dense_dp2_route_plan.dp_groups[{tp}]")
        for route_index, route in enumerate(group.embedding.routes):
            route.validate_against(
                context.fabric,
                {placement.rank: placement.die_id for placement in group.placements},
                f"dense_dp2_route_plan.dp_groups[{tp}].routes[{route_index}]",
            )
    graph_nodes = tuple({node.id: node for node in graph.nodes}
                        for graph in replica_graphs)
    graph_values = tuple({value.id: value for value in graph.values}
                         for graph in replica_graphs)
    states = {(state.identity.tensor_ref, state.identity.shard_index): state
              for state in source.persistent_states}
    gradients = []
    for step in (0, 1):
        for template in plan.parameter_templates:
            tp = template.tp_shard_index
            state = states[(template.tensor_ref, tp)]
            sync_ref = f"dp_sync::{state.id}::tp{tp}::step{step}"
            wgrad_ref = f"{template.wgrad_ref}::step{step}"
            optimizer_ref = f"sgd_update::{template.tensor_ref}::tp{tp}::step{step}"
            sync_output = f"dp_sync::{state.id}::tp{tp}.output::step{step}"
            wgrad_output = f"{template.wgrad_ref}.output::step{step}"
            refs = tuple(
                (f"{wgrad_ref}__dp{dp}", f"{sync_ref}__dp{dp}",
                 f"{optimizer_ref}__dp{dp}")
                for dp in (0, 1)
            )
            for dp, (local, sync, optimizer) in enumerate(refs):
                nodes = graph_nodes[dp]
                values = graph_values[dp]
                if (nodes[local].outputs != (wgrad_output,)
                        or nodes[sync].kind is not OpKind.COLLECTIVE
                        or nodes[sync].inputs != (wgrad_output,)
                        or nodes[sync].outputs != (sync_output,)
                        or nodes[sync].workload.collective is not CollectiveKind.ALL_REDUCE
                        or nodes[sync].workload.reduce_op is not ReduceOp.SUM
                        or nodes[sync].workload.rank_input_bytes != template.gradient_bytes
                        or nodes[optimizer].inputs != (template.tensor_ref, sync_output)
                        or values[wgrad_output].dtype is not DType.FP32
                        or values[sync_output].dtype is not DType.FP32):
                    raise SchemaError("DP2 route must bind WGRAD→SUM→SGD in each replica",
                                      path=f"dense_dp2_route_plan.step{step}.tp{tp}.dp{dp}")
            routes = {(route.source_rank, route.destination_rank): route
                      for route in groups[tp].embedding.routes}
            wgrad_value = graph_values[0][wgrad_output]
            local_shape = tuple(
                extent // plan.spec.tp_degree if axis is MeshAxisName.TP else extent
                for extent, axis in zip(
                    wgrad_value.shape, wgrad_value.sharding.dim_map, strict=True
                )
            )
            local_offset = tuple(
                tp * extent if axis is MeshAxisName.TP else 0
                for extent, axis in zip(
                    local_shape, wgrad_value.sharding.dim_map, strict=True
                )
            )
            chunk = ChunkSlice(
                f"chunk.{sync_ref}.tp{tp}", 0, sync_output,
                local_offset, local_shape, template.gradient_bytes, 0,
            )
            chunk.validate("dp_sync_chunk")
            programs = _dp2_rank_programs(
                sync_ref=sync_ref, wgrad_value_ref=wgrad_output,
                sync_value_ref=sync_output, bytes=template.gradient_bytes,
                chunk=chunk, reduce_route=routes[(1, 0)],
                broadcast_route=routes[(0, 1)],
            )
            gradients.append(DenseDP2GradientRoute(
                state.id, tp, step, template.gradient_bytes, groups[tp].id,
                (refs[0][0], refs[1][0]),
                (refs[0][1], refs[1][1]),
                (refs[0][2], refs[1][2]),
                routes[(1, 0)], routes[(0, 1)],
                chunk, programs,
            ))
    payload = dict(source_ir0_id=source.id, placed_ir1_id=placed.id,
                   fabric_id=canonical_digest(context.fabric), dp_groups=groups,
                   gradients=tuple(gradients))
    return DenseDP2RoutePlan(
        stable_artifact_id("dense_dp2_route_plan", payload,
                           schema_version=("dense_dp2_route_plan/v1" if
                                           plan.spec.tp_degree == 2 else
                                           "dense_dp2_route_plan/v2")),
        **payload,
    )


__all__ = ["DenseDP2GradientRoute", "DenseDP2RoutePlan",
           "build_dense_dp2_route_plan"]
