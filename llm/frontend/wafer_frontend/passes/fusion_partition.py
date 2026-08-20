"""N4 deterministic fusion-partition producer.

This pass selects legal regions but deliberately does not choose an inter-die
algorithm.  The resulting IR1 therefore carries ``FusionImpl.NONE`` skeletons;
the following ``inter_die_plan`` pass is the only owner of executable action
ordering and the naive/optimized implementation choice.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..policies.interfaces import FusionPartition
from ..policies.naive_fusion_partition import NaiveFusionPartition
from ..schema.ir1 import IR1
from ..schema.n4 import (
    FusionPartitionContext,
    FusionPartitionedIR1Bundle,
    FusionPartitionedProfileIR1,
    Stage4FusionPartitionedIR1,
    TrainFusionPartitionedIR1,
)
from ..schema.placed_ir1 import PlacedIR1Bundle, Stage4PlacedIR1, TrainPlacedIR1


def partition_ir1(
    source: IR1,
    *,
    policy: FusionPartition | None = None,
) -> IR1:
    """Select all legal Dense GEMM/ReduceScatter regions in one placed graph."""

    if type(source) is not IR1:
        raise SchemaError("must be an IR1", path="source")
    source.validate("source")
    if source.producer_pass != "placement":
        raise SchemaError(
            "must be produced by placement",
            path="source.producer_pass",
        )
    if source.fused_op_skeletons:
        raise SchemaError(
            "must not already contain fusion skeletons",
            path="source.fused_op_skeletons",
        )
    selected = (
        NaiveFusionPartition() if policy is None else policy
    ).run(source)
    result = IR1.create(
        producer_pass="fusion_partition",
        source_ir0_id=source.source_ir0_id,
        profile=source.profile,
        fabric=source.fabric,
        instances=source.instances,
        groups=source.groups,
        nodes=source.nodes,
        values=source.values,
        edges=source.edges,
        fusion_candidates=source.fusion_candidates,
        fused_op_skeletons=selected,
        cross_routes=source.cross_routes,
        state_accesses=source.state_accesses,
        persistent_state_manifest=source.persistent_state_manifest,
        instance_profiles=source.instance_profiles,
        node_profiles=source.node_profiles,
        pd_plan_id=source.pd_plan_id,
    )
    result.validate("fusion_partitioned_ir1")
    return result


def partition_bundle(
    source: PlacedIR1Bundle,
    context: FusionPartitionContext,
    *,
    policy: FusionPartition | None = None,
) -> FusionPartitionedIR1Bundle:
    """Partition every canonical profile without changing placement facts."""

    if type(source) is not PlacedIR1Bundle:
        raise SchemaError("must be a PlacedIR1Bundle", path="source")
    if type(context) is not FusionPartitionContext:
        raise SchemaError(
            "must be a FusionPartitionContext",
            path="fusion_partition_context",
        )
    source.validate("source")
    context.validate("fusion_partition_context")
    selected_policy = NaiveFusionPartition() if policy is None else policy
    entries = tuple(
        FusionPartitionedProfileIR1.create(
            source=entry,
            context=context,
            graph=partition_ir1(entry.graph, policy=selected_policy),
        )
        for entry in source.entries
    )
    result = FusionPartitionedIR1Bundle.create(
        source=source,
        context=context,
        entries=entries,
    )
    result.validate_against(source, context)
    return result


def partition_stage4(
    source: Stage4PlacedIR1,
    context: FusionPartitionContext,
    *,
    policy: FusionPartition | None = None,
) -> Stage4FusionPartitionedIR1:
    """Partition one Stage 4 placed carrier without losing PD provenance."""

    if type(source) is not Stage4PlacedIR1:
        raise SchemaError("must be a Stage4PlacedIR1", path="source")
    if type(context) is not FusionPartitionContext:
        raise SchemaError(
            "must be a FusionPartitionContext",
            path="fusion_partition_context",
        )
    source.validate("source")
    context.validate("fusion_partition_context")
    graph = partition_ir1(source.graph, policy=policy)
    result = Stage4FusionPartitionedIR1.create(
        source=source,
        context=context,
        graph=graph,
    )
    result.validate_against(source, context)
    return result


def partition_train_forward(
    source: TrainPlacedIR1,
    context: FusionPartitionContext,
    *,
    policy: FusionPartition | None = None,
) -> TrainFusionPartitionedIR1:
    """Partition every disjoint DP replica with one shared policy contract."""

    if type(source) is not TrainPlacedIR1:
        raise SchemaError("must be a TrainPlacedIR1", path="source")
    if type(context) is not FusionPartitionContext:
        raise SchemaError("must be a FusionPartitionContext", path="fusion_partition_context")
    source.validate("source")
    context.validate("fusion_partition_context")
    replicas = tuple(
        partition_ir1(replica.graph, policy=policy)
        for replica in source.replicas
    )
    result = TrainFusionPartitionedIR1.create(
        source=source,
        context=context,
        replicas=replicas,
    )
    result.validate_against(source, context)
    return result


__all__ = [
    "partition_bundle",
    "partition_ir1",
    "partition_stage4",
    "partition_train_forward",
]
