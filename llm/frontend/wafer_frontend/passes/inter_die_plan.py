"""N4 deterministic inter-die action-plan producer."""

from __future__ import annotations

from ..errors import SchemaError
from ..policies.interfaces import InterDiePolicy, StandaloneCollectivePolicy
from ..policies.naive_inter_die import DirectAllGatherPolicy, NaiveInterDiePolicy
from ..schema.action import FusionPlan, StandaloneCollectivePlan
from ..schema.swizzle_plan import FusedPlan
from ..policies.swizzle_defaults import production_swizzle_policy
from ..policies.swizzle_topo import SwizzlePlanner
from ..schema.ir0 import CollectiveKind, CollectiveWorkload, OpKind, ReduceOp
from ..schema.common import MeshAxisName, ProfileKey
from ..schema.ir1 import IR1, PhysicalNode
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.placement import PlacementContext
from .placement import place_train_forward_ir0
from .full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from .full_dense_training_dp2_routes import build_dense_dp2_route_plan
from ..schema.n4 import (
    FusionPartitionedIR1Bundle,
    FusionPartitionedProfileIR1,
    InterDiePlanBundle,
    InterDiePlannedProfile,
    InterDiePlanningContext,
    FusedInterDieContract,
    Stage4FusionPartitionedIR1,
    Stage4InterDiePlannedIR1,
    TrainFusionPartitionedIR1,
    TrainInterDiePlannedIR1,
    TrainReplicaInterDiePlans,
)


def _unfused_collectives(
    graph: IR1, *, dp_sync_refs: tuple[str, ...] = (),
) -> tuple[PhysicalNode, ...]:
    fused_members = {
        node_id
        for skeleton in graph.fused_op_skeletons
        for node_id in skeleton.member_node_ids
    }
    result: list[PhysicalNode] = []
    for index, node in enumerate(graph.nodes):
        if node.kind is not OpKind.COLLECTIVE or node.id in fused_members:
            continue
        if node.id in dp_sync_refs:
            if (
                not node.id.startswith("dp_sync::")
                or type(node.workload) is not CollectiveWorkload
                or node.workload.collective is not CollectiveKind.ALL_REDUCE
                or node.workload.mesh_axes != (MeshAxisName.DP,)
                or node.workload.reduce_op is not ReduceOp.SUM
            ):
                raise SchemaError("only source-bound DP2 gradient SUM may use a cross-replica plan",
                                  path=f"ir1.nodes[{index}]")
            continue
        if (
            type(node.workload) is not CollectiveWorkload
            or node.workload.collective not in (CollectiveKind.ALL_GATHER, CollectiveKind.REDUCE_SCATTER)
        ):
            raise SchemaError(
                "inter_die_plan supports only AllGather or SUM ReduceScatter as unfused collectives",
                path=f"ir1.nodes[{index}].workload.collective",
            )
        if (node.workload.collective is CollectiveKind.REDUCE_SCATTER
                and node.workload.reduce_op is not ReduceOp.SUM):
            raise SchemaError("standalone ReduceScatter requires SUM",
                              path=f"ir1.nodes[{index}].workload.reduce_op")
        result.append(node)
    return tuple(result)


def _profiles_by_instance(graph: IR1) -> dict[str, ProfileKey]:
    if not graph.instance_profiles:
        return {graph.instances[0].id: graph.profile}
    profiles: dict[str, ProfileKey] = {}
    for binding in graph.instance_profiles:
        if binding.instance_ref in profiles:
            raise SchemaError(
                "each owner instance must have exactly one profile binding",
                path="ir1.instance_profiles",
            )
        profiles[binding.instance_ref] = binding.profile
    if set(profiles) != {instance.id for instance in graph.instances}:
        raise SchemaError(
            "must bind every IR-1 instance exactly once",
            path="ir1.instance_profiles",
        )
    return profiles


def plan_ir1(
    graph: IR1,
    context: InterDiePlanningContext,
    fused_policy: InterDiePolicy | None = None,
    standalone_policy: StandaloneCollectivePolicy | None = None,
    *,
    dp_sync_refs: tuple[str, ...] = (),
) -> tuple[tuple[FusedPlan, ...], tuple[StandaloneCollectivePlan, ...]]:
    """Plan one real fusion-partitioned IR-1 without fabricating an N4 wrapper."""

    if type(graph) is not IR1:
        raise SchemaError("must be an IR1", path="ir1")
    if type(context) is not InterDiePlanningContext:
        raise SchemaError(
            "must be an InterDiePlanningContext",
            path="inter_die_planning_context",
        )
    graph.validate("ir1")
    context.validate("inter_die_planning_context")
    if graph.producer_pass != "fusion_partition":
        raise SchemaError(
            "must be produced by fusion_partition",
            path="ir1.producer_pass",
        )
    standalone_nodes = _unfused_collectives(graph, dp_sync_refs=dp_sync_refs)

    if fused_policy is None:
        selected_fused_policy: InterDiePolicy = (
            NaiveInterDiePolicy()
            if context.fused_contract is FusedInterDieContract.DIRECT_NAIVE_V1
            else production_swizzle_policy()
        )
    else:
        selected_fused_policy = fused_policy
    expected_policy_type = (
        NaiveInterDiePolicy
        if context.fused_contract is FusedInterDieContract.DIRECT_NAIVE_V1
        else SwizzlePlanner
    )
    if type(selected_fused_policy) in (NaiveInterDiePolicy, SwizzlePlanner) and type(
        selected_fused_policy
    ) is not expected_policy_type:
        raise SchemaError(
            "fused policy implementation disagrees with planning context",
            path="fused_policy",
        )
    selected_standalone_policy: StandaloneCollectivePolicy = (
        DirectAllGatherPolicy() if standalone_policy is None else standalone_policy
    )
    if not graph.fused_op_skeletons and not standalone_nodes:
        return (), ()
    profiles = _profiles_by_instance(graph)
    fusion_plans = tuple(
        selected_fused_policy.plan(
            graph,
            skeleton,
            profiles[skeleton.instance_id],
        )
        for skeleton in graph.fused_op_skeletons
    )
    standalone_plans = tuple(
        selected_standalone_policy.plan(
            graph,
            node,
            profiles[node.instance_id],
        )
        for node in standalone_nodes
    )
    for index, plan in enumerate(fusion_plans):
        plan.validate_against(graph, f"fusion_plans[{index}]")
    for index, plan in enumerate(standalone_plans):
        plan.validate_against(graph, f"standalone_plans[{index}]")
    return fusion_plans, standalone_plans


def plan_profile(
    source: FusionPartitionedProfileIR1,
    context: InterDiePlanningContext,
    fused_policy: InterDiePolicy | None = None,
    standalone_policy: StandaloneCollectivePolicy | None = None,
) -> InterDiePlannedProfile:
    """Plan one partitioned profile in skeleton/node source order."""

    if type(source) is not FusionPartitionedProfileIR1:
        raise SchemaError(
            "must be a FusionPartitionedProfileIR1",
            path="source",
        )
    if type(context) is not InterDiePlanningContext:
        raise SchemaError(
            "must be an InterDiePlanningContext",
            path="inter_die_planning_context",
        )
    source.validate("source")
    context.validate("inter_die_planning_context")

    graph = source.graph
    fusion_plans, standalone_plans = plan_ir1(
        graph,
        context,
        fused_policy,
        standalone_policy,
    )
    result = InterDiePlannedProfile.create(
        source=source,
        context=context,
        fusion_plans=fusion_plans,
        standalone_plans=standalone_plans,
    )
    result.validate_against(source, context)
    return result


def plan_bundle(
    source: FusionPartitionedIR1Bundle,
    context: InterDiePlanningContext,
    fused_policy: InterDiePolicy | None = None,
    standalone_policy: StandaloneCollectivePolicy | None = None,
) -> InterDiePlanBundle:
    """Plan every canonical profile while preserving upstream provenance."""

    if type(source) is not FusionPartitionedIR1Bundle:
        raise SchemaError(
            "must be a FusionPartitionedIR1Bundle",
            path="source",
        )
    if type(context) is not InterDiePlanningContext:
        raise SchemaError(
            "must be an InterDiePlanningContext",
            path="inter_die_planning_context",
        )
    source.validate("source")
    context.validate("inter_die_planning_context")

    selected_fused_policy: InterDiePolicy = (
        (NaiveInterDiePolicy() if context.fused_contract is FusedInterDieContract.DIRECT_NAIVE_V1 else production_swizzle_policy())
        if fused_policy is None
        else fused_policy
    )
    selected_standalone_policy: StandaloneCollectivePolicy = (
        DirectAllGatherPolicy() if standalone_policy is None else standalone_policy
    )
    entries = tuple(
        plan_profile(
            entry,
            context,
            selected_fused_policy,
            selected_standalone_policy,
        )
        for entry in source.entries
    )
    result = InterDiePlanBundle.create(
        source=source,
        context=context,
        entries=entries,
    )
    result.validate_against(source, context)
    return result


def plan_stage4(
    source: Stage4FusionPartitionedIR1,
    context: InterDiePlanningContext,
    fused_policy: InterDiePolicy | None = None,
    standalone_policy: StandaloneCollectivePolicy | None = None,
) -> Stage4InterDiePlannedIR1:
    """Plan one Stage 4 partition carrier without a profile bundle."""

    if type(source) is not Stage4FusionPartitionedIR1:
        raise SchemaError("must be a Stage4FusionPartitionedIR1", path="source")
    if type(context) is not InterDiePlanningContext:
        raise SchemaError(
            "must be an InterDiePlanningContext",
            path="inter_die_planning_context",
        )
    source.validate("source")
    context.validate("inter_die_planning_context")
    fusion_plans, standalone_plans = plan_ir1(
        source.graph,
        context,
        fused_policy,
        standalone_policy,
    )
    result = Stage4InterDiePlannedIR1.create(
        source=source,
        context=context,
        fusion_plans=fusion_plans,
        standalone_plans=standalone_plans,
    )
    result.validate_against(source, context)
    return result


def plan_train_forward(
    source: TrainFusionPartitionedIR1,
    context: InterDiePlanningContext,
    fused_policy: InterDiePolicy | None = None,
    standalone_policy: StandaloneCollectivePolicy | None = None,
    *,
    dense_dp2_plan: FlexibleDenseTrainPlan | None = None,
    dp2_placement_context: PlacementContext | None = None,
) -> TrainInterDiePlannedIR1:
    """Plan local TP replicas and authenticated physical cross-DP gradient SUM."""

    if type(source) is not TrainFusionPartitionedIR1:
        raise SchemaError("must be a TrainFusionPartitionedIR1", path="source")
    if type(context) is not InterDiePlanningContext:
        raise SchemaError("must be an InterDiePlanningContext", path="inter_die_planning_context")
    source.validate("source")
    context.validate("inter_die_planning_context")
    dp_sync_nodes = tuple(
        node for node in source.replicas[0].nodes
        if node.kind is OpKind.COLLECTIVE
        and type(node.workload) is CollectiveWorkload
        and node.workload.collective is CollectiveKind.ALL_REDUCE
        and node.workload.mesh_axes == (MeshAxisName.DP,)
    )
    dp_route_plan = None
    if dp_sync_nodes:
        if dense_dp2_plan is None or dp2_placement_context is None:
            raise SchemaError("source-backed cross-DP SUM needs its exact plan and physical N3 placement",
                              path="dense_dp2_plan")
        canonical_source = build_full_dense_training_two_step_ir0(dense_dp2_plan)
        placed = place_train_forward_ir0(canonical_source, dp2_placement_context)
        if placed.id != source.source_train_placed_id:
            raise SchemaError("DP SUM physical source placement drifted",
                              path="source.source_train_placed_id")
        dp_route_plan = build_dense_dp2_route_plan(
            dense_dp2_plan, placed, dp2_placement_context,
        )
        dp_route_plan.validate_against(dense_dp2_plan, placed, dp2_placement_context)
    elif dense_dp2_plan is not None or dp2_placement_context is not None:
        raise SchemaError("cross-DP route plan is forbidden without DP SUM nodes",
                          path="dense_dp2_plan")
    replica_plans: list[TrainReplicaInterDiePlans] = []
    for replica_index, graph in enumerate(source.replicas):
        dp_sync_refs = (
            tuple(item.sync_refs[replica_index] for item in dp_route_plan.gradients)
            if dp_route_plan is not None else ()
        )
        fusion_plans, standalone_plans = plan_ir1(
            graph, context, fused_policy, standalone_policy,
            dp_sync_refs=dp_sync_refs,
        )
        replica = TrainReplicaInterDiePlans.create(
            replica_index=replica_index,
            graph=graph,
            fusion_plans=fusion_plans,
            standalone_plans=standalone_plans,
            dp_sync_refs=dp_sync_refs,
        )
        replica.validate_against(graph, f"train_replica_plans[{replica_index}]")
        replica_plans.append(replica)
    return TrainInterDiePlannedIR1.create(
        source=source,
        context=context,
        replicas=tuple(replica_plans),
        dp_gradient_routes=dp_route_plan,
    )


__all__ = [
    "plan_bundle",
    "plan_ir1",
    "plan_profile",
    "plan_stage4",
    "plan_train_forward",
]
