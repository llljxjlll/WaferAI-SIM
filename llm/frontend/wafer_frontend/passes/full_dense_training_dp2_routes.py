"""Source-bound physical DP2 gradient pair routes; no executable actions yet.

These paths join *different* TP replica groups.  They must never be placed
inside either replica's local N4 collective plan as a TP AllReduce.
"""
from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, MeshAxisName, stable_artifact_id
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.ir0 import CollectiveKind, OpKind, ReduceOp
from ..schema.dense_dp_sync_routes import DenseDP2GradientRoute, DenseDP2RoutePlan
from ..schema.placed_ir1 import TrainPlacedIR1
from ..schema.placement import PlacementContext
from ..schema.serde import canonical_digest
from .full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from .group_registry import _expected_group



def build_dense_dp2_route_plan(
    plan: FlexibleDenseTrainPlan, placed: TrainPlacedIR1,
    context: PlacementContext,
) -> DenseDP2RoutePlan:
    plan.validate("dp2_route_plan")
    placed.validate("dp2_placed")
    context.validate("dp2_placement")
    if (plan.spec.dp_degree != 2 or plan.spec.tp_degree != 2
            or len(placed.replicas) != 2
            or placed.placement_context_id != context.id):
        raise SchemaError("routes require exact TP2/DP2 physical placement",
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
                or len(group.placements) != 2
                or tuple(placement.rank for placement in group.placements) != (0, 1)
                or group.axis is not MeshAxisName.TP
                or graph.fabric != context.fabric):
            raise SchemaError("DP2 replica TP group or fabric drifted",
                              path=f"dense_dp2_route_plan.replicas[{index}]")
    instance = source.instances[0]
    (mesh,) = instance.meshes
    groups = tuple(
        replace(_expected_group(
            source, context, instance, mesh,
            rank_to_die_override=tuple(
                graph.groups[0].placements[tp].die_id for graph in replica_graphs
            ),
            group_id_suffix=f"__dp_gradient_tp{tp}",
        ), axis=MeshAxisName.DP)
        for tp in (0, 1)
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
            gradients.append(DenseDP2GradientRoute(
                state.id, tp, step, template.gradient_bytes, groups[tp].id,
                (refs[0][0], refs[1][0]),
                (refs[0][1], refs[1][1]),
                (refs[0][2], refs[1][2]),
                routes[(1, 0)], routes[(0, 1)],
            ))
    payload = dict(source_ir0_id=source.id, placed_ir1_id=placed.id,
                   fabric_id=canonical_digest(context.fabric), dp_groups=groups,
                   gradients=tuple(gradients))
    return DenseDP2RoutePlan(
        stable_artifact_id("dense_dp2_route_plan", payload,
                           schema_version="dense_dp2_route_plan/v1"),
        **payload,
    )


__all__ = ["DenseDP2GradientRoute", "DenseDP2RoutePlan",
           "build_dense_dp2_route_plan"]
