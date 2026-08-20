"""N4 deterministic inter-die action-plan producer."""

from __future__ import annotations

from ..errors import SchemaError
from ..policies.interfaces import InterDiePolicy, StandaloneCollectivePolicy
from ..policies.naive_inter_die import DirectAllGatherPolicy, NaiveInterDiePolicy
from ..schema.action import FusionPlan, StandaloneCollectivePlan
from ..schema.ir0 import CollectiveKind, CollectiveWorkload, OpKind
from ..schema.common import ProfileKey
from ..schema.ir1 import IR1, PhysicalNode
from ..schema.n4 import (
    FusionPartitionedIR1Bundle,
    FusionPartitionedProfileIR1,
    InterDiePlanBundle,
    InterDiePlannedProfile,
    InterDiePlanningContext,
    Stage4FusionPartitionedIR1,
    Stage4InterDiePlannedIR1,
    TrainFusionPartitionedIR1,
    TrainInterDiePlannedIR1,
    TrainReplicaInterDiePlans,
)


def _unfused_all_gathers(graph: IR1) -> tuple[PhysicalNode, ...]:
    fused_members = {
        node_id
        for skeleton in graph.fused_op_skeletons
        for node_id in skeleton.member_node_ids
    }
    result: list[PhysicalNode] = []
    for index, node in enumerate(graph.nodes):
        if node.kind is not OpKind.COLLECTIVE or node.id in fused_members:
            continue
        if (
            type(node.workload) is not CollectiveWorkload
            or node.workload.collective is not CollectiveKind.ALL_GATHER
        ):
            raise SchemaError(
                "inter_die_plan supports only AllGather as an unfused collective",
                path=f"ir1.nodes[{index}].workload.collective",
            )
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
) -> tuple[tuple[FusionPlan, ...], tuple[StandaloneCollectivePlan, ...]]:
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

    selected_fused_policy: InterDiePolicy = (
        NaiveInterDiePolicy() if fused_policy is None else fused_policy
    )
    selected_standalone_policy: StandaloneCollectivePolicy = (
        DirectAllGatherPolicy() if standalone_policy is None else standalone_policy
    )
    standalone_nodes = _unfused_all_gathers(graph)
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
        NaiveInterDiePolicy() if fused_policy is None else fused_policy
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
) -> TrainInterDiePlannedIR1:
    """Plan every DP replica independently; no plan may cross replica groups."""

    if type(source) is not TrainFusionPartitionedIR1:
        raise SchemaError("must be a TrainFusionPartitionedIR1", path="source")
    if type(context) is not InterDiePlanningContext:
        raise SchemaError("must be an InterDiePlanningContext", path="inter_die_planning_context")
    source.validate("source")
    context.validate("inter_die_planning_context")
    replica_plans: list[TrainReplicaInterDiePlans] = []
    for replica_index, graph in enumerate(source.replicas):
        fusion_plans, standalone_plans = plan_ir1(
            graph,
            context,
            fused_policy,
            standalone_policy,
        )
        replica = TrainReplicaInterDiePlans.create(
            replica_index=replica_index,
            graph=graph,
            fusion_plans=fusion_plans,
            standalone_plans=standalone_plans,
        )
        replica.validate_against(graph, f"train_replica_plans[{replica_index}]")
        replica_plans.append(replica)
    return TrainInterDiePlannedIR1.create(
        source=source,
        context=context,
        replicas=tuple(replica_plans),
    )


__all__ = [
    "plan_bundle",
    "plan_ir1",
    "plan_profile",
    "plan_stage4",
    "plan_train_forward",
]
