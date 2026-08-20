"""N3 deterministic placement producer for the Dense naive MVP.

This pass is intentionally policy-free: it preserves every logical node,
value, edge, and fusion candidate, and only attaches the physical group chosen
by :mod:`group_registry`.  Fusion selection and action ordering belong to N4.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.ir0 import (
    IR0,
    InstanceProfileBinding,
    LogicalInstance,
    LogicalNode,
    JobKind,
    StateAccess,
)
from ..schema.common import UINT64_MAX
from ..schema.ir1 import IR1, PhysicalGroup, PhysicalInstance, PhysicalNode
from ..schema.persistent_state import (
    HbmBinding,
    PersistentStateManifest,
    StateKind,
)
from ..schema.logical import ExpandedIR0Bundle
from ..schema.placed_ir1 import (
    PlacedIR1Bundle,
    PlacedProfileIR1,
    Stage4PlacedIR1,
    TrainPlacedIR1,
    TrainPlacedReplica,
)
from ..schema.placement import PlacementContext
from ..schema.stage4_pd import Stage4PdMode, Stage4PdPlan
from .group_registry import (
    build_group_registry,
    build_train_replica_groups,
    build_stage4_cross_group_routes,
    validate_group_against,
)
from .validate_fusion import FusionSemanticValidator
from .validate_ir0 import DenseIR0Validator


def _physical_instance(
    logical: LogicalInstance,
    graph: IR0,
    groups: tuple[PhysicalGroup, ...],
    *,
    node_ids: tuple[str, ...] | None = None,
) -> PhysicalInstance:
    instance_id = logical.id
    owned_groups = tuple(
        group for group in groups if group.instance_id == instance_id
    )
    die_region: list[int] = []
    for group in owned_groups:
        for placement in getattr(group, "placements"):
            if placement.die_id not in die_region:
                die_region.append(placement.die_id)
    return PhysicalInstance(
        id=instance_id,
        origin_instance_id=instance_id,
        role=logical.role,
        die_region=tuple(die_region),
        group_ids=tuple(group.id for group in owned_groups),
        node_ids=(
            tuple(node.id for node in graph.nodes if node.instance_id == instance_id)
            if node_ids is None
            else node_ids
        ),
    )


def _physical_node(logical: LogicalNode, group_id: str) -> PhysicalNode:
    return PhysicalNode(
        id=logical.id,
        origin_node_id=logical.id,
        instance_id=logical.instance_id,
        kind=logical.kind,
        phase=logical.phase,
        stage=logical.stage,
        mesh_ref=logical.mesh_ref,
        execution_group_ref=group_id,
        inputs=logical.inputs,
        outputs=logical.outputs,
        workload=logical.workload,
        math=logical.math,
        effects=logical.effects,
        impl_ref=logical.impl_ref,
    )


def _train_replica_components(
    graph: IR0,
    group: PhysicalGroup,
    replica_index: int,
) -> tuple[
    tuple[PhysicalNode, ...],
    tuple[object, ...],
    tuple[object, ...],
    tuple[object, ...],
    tuple[StateAccess, ...],
    tuple[object, ...],
]:
    """Rewrite all node-bearing references to canonical replica-local ids."""

    node_ids = {node.id: f"{node.id}__dp{replica_index}" for node in graph.nodes}
    nodes = tuple(
        replace(_physical_node(node, group.id), id=node_ids[node.id])
        for node in graph.nodes
    )
    values = tuple(
        replace(
            value,
            producer=(None if value.producer is None else node_ids[value.producer]),
            consumers=tuple(node_ids[item] for item in value.consumers),
        )
        for value in graph.values
    )
    edges = tuple(
        replace(
            edge,
            id=f"{edge.id}__dp{replica_index}",
            source_node=node_ids[edge.source_node],
            destination_node=node_ids[edge.destination_node],
        )
        for edge in graph.edges
    )
    fusion_candidates = tuple(
        replace(
            candidate,
            id=f"{candidate.id}__dp{replica_index}",
            members=tuple(node_ids[item] for item in candidate.members),
        )
        for candidate in graph.fusion_candidates
    )
    state_accesses = tuple(
        StateAccess.create(
            node_ref=node_ids[access.node_ref],
            state_ref=access.state_ref,
            mode=access.mode,
            rank=access.rank,
            read_offset=access.read_offset,
            read_shape=access.read_shape,
            write_offset=access.write_offset,
            write_shape=access.write_shape,
        )
        for access in graph.state_accesses
    )
    node_profiles = tuple(
        replace(binding, node_ref=node_ids[binding.node_ref])
        for binding in graph.node_profiles
    )
    return (
        nodes,
        values,
        edges,
        fusion_candidates,
        state_accesses,
        node_profiles,
    )


_STATE_KIND_ORDER = {
    kind: index for index, kind in enumerate(StateKind)
}


def _place_persistent_state_manifest(
    graph: IR0,
    context: PlacementContext,
    groups: tuple[PhysicalGroup, ...],
) -> PersistentStateManifest | None:
    """Bind every logical state shard by deterministic per-die 64 B first-fit."""

    if not graph.persistent_states:
        return None
    if not context.hbm_address_spaces:
        raise SchemaError(
            "persistent state requires explicit HBM address spaces",
            path="placement_context.hbm_address_spaces",
        )

    groups_by_owner: dict[tuple[str, str], PhysicalGroup] = {}
    for group in groups:
        owner = (group.instance_id, group.mesh_ref)
        if owner in groups_by_owner:
            raise SchemaError(
                "persistent state owner must resolve to one physical group",
                path="ir1.groups",
            )
        groups_by_owner[owner] = group
    spaces_by_die = {
        space.die_id: space for space in context.hbm_address_spaces
    }

    placements: list[tuple[tuple[object, ...], object, int]] = []
    for index, declaration in enumerate(graph.persistent_states):
        identity = declaration.identity
        owner = (identity.instance_ref, identity.mesh_ref)
        group = groups_by_owner.get(owner)
        if group is None:
            raise SchemaError(
                "persistent state has no physical owner group",
                path=f"ir0.persistent_states[{index}].identity",
            )
        rank_placement = next(
            (
                item
                for item in group.placements
                if item.rank == identity.shard_index
            ),
            None,
        )
        if rank_placement is None:
            raise SchemaError(
                "persistent state shard has no physical rank placement",
                path=f"ir0.persistent_states[{index}].identity.shard_index",
            )
        home_die = rank_placement.die_id
        if home_die not in spaces_by_die:
            raise SchemaError(
                "persistent state home die has no HBM address space",
                path="placement_context.hbm_address_spaces",
            )
        key = (
            identity.instance_ref,
            identity.mesh_ref,
            identity.shard_index,
            home_die,
            _STATE_KIND_ORDER[identity.kind],
            identity.request_ref or "",
            -1 if identity.layer_index is None else identity.layer_index,
            identity.tensor_ref or "",
            identity.generation,
            declaration.id,
        )
        placements.append((key, declaration, home_die))

    cursors = {
        die_id: space.base_address for die_id, space in spaces_by_die.items()
    }
    bindings: list[HbmBinding] = []
    for _key, declaration, home_die in sorted(
        placements, key=lambda item: item[0]
    ):
        space = spaces_by_die[home_die]
        cursor = cursors[home_die]
        remainder = cursor % 64
        padding = 0 if remainder == 0 else 64 - remainder
        if cursor > UINT64_MAX - padding:
            raise SchemaError(
                "64-byte HBM alignment overflows uint64",
                path="persistent_state_placement",
            )
        address = cursor + padding
        space_end = space.base_address + space.size_bytes
        if (
            address > space_end
            or declaration.tensor_bytes > space_end - address
        ):
            raise SchemaError(
                "persistent state exceeds its home HBM capacity",
                path=f"persistent_state_placement.{declaration.id}",
            )
        binding = HbmBinding.create(
            state_ref=declaration.id,
            die_id=home_die,
            address=address,
            size_bytes=declaration.tensor_bytes,
        )
        bindings.append(binding)
        cursors[home_die] = address + declaration.tensor_bytes

    return PersistentStateManifest.create(
        address_spaces=context.hbm_address_spaces,
        declarations=graph.persistent_states,
        bindings=tuple(bindings),
    )


def place_ir0(graph: IR0, context: PlacementContext) -> IR1:
    """Place one profile-specialized IR-0 graph without selecting fusion."""

    if type(graph) is not IR0:
        raise SchemaError("must be an IR0", path="ir0")
    if type(context) is not PlacementContext:
        raise SchemaError("must be a PlacementContext", path="placement_context")
    graph.validate("ir0")
    context.validate("placement_context")
    DenseIR0Validator.validate(graph, "ir0")
    FusionSemanticValidator.validate(graph, "ir0")

    groups = build_group_registry(graph, context)
    group_by_owner = {
        (group.instance_id, group.mesh_ref): group for group in groups
    }
    persistent_state_manifest = _place_persistent_state_manifest(
        graph, context, groups
    )
    nodes = tuple(
        _physical_node(
            node,
            group_by_owner[(node.instance_id, node.mesh_ref)].id,
        )
        for node in graph.nodes
    )
    instances = tuple(
        _physical_instance(instance, graph, groups) for instance in graph.instances
    )
    result = IR1.create(
        producer_pass="placement",
        source_ir0_id=graph.id,
        profile=graph.profile,
        fabric=context.fabric,
        instances=instances,
        groups=groups,
        nodes=nodes,
        values=graph.values,
        edges=graph.edges,
        fusion_candidates=graph.fusion_candidates,
        fused_op_skeletons=(),
        cross_routes=(),
        state_accesses=graph.state_accesses,
        persistent_state_manifest=persistent_state_manifest,
        instance_profiles=graph.instance_profiles,
        node_profiles=graph.node_profiles,
        pd_plan_id=graph.pd_plan_id,
    )
    result.validate("ir1")
    for index, group in enumerate(result.groups):
        validate_group_against(
            graph,
            context,
            group,
            path=f"ir1.groups[{index}]",
        )
    return result


def place_train_forward_ir0(
    graph: IR0,
    context: PlacementContext,
) -> TrainPlacedIR1:
    """Place one forward TRAIN template into disjoint TP groups per DP replica."""

    if type(graph) is not IR0:
        raise SchemaError("must be an IR0", path="ir0")
    if type(context) is not PlacementContext:
        raise SchemaError("must be a PlacementContext", path="placement_context")
    graph.validate("ir0")
    context.validate("placement_context")
    if graph.job is not JobKind.TRAIN or graph.train is None:
        raise SchemaError("requires a TRAIN IR0", path="ir0.job")
    DenseIR0Validator.validate(graph, "ir0")
    FusionSemanticValidator.validate(graph, "ir0")

    groups = build_train_replica_groups(graph, context)
    logical_instance = graph.instances[0]
    replicas: list[TrainPlacedReplica] = []
    for replica_index, group in enumerate(groups):
        manifest = _place_persistent_state_manifest(graph, context, (group,))
        (
            nodes,
            values,
            edges,
            fusion_candidates,
            state_accesses,
            node_profiles,
        ) = _train_replica_components(graph, group, replica_index)
        instance = _physical_instance(
            logical_instance,
            graph,
            (group,),
            node_ids=tuple(node.id for node in nodes),
        )
        replica_graph = IR1.create(
            producer_pass="placement",
            source_ir0_id=graph.id,
            profile=graph.profile,
            fabric=context.fabric,
            instances=(instance,),
            groups=(group,),
            nodes=nodes,
            values=values,
            edges=edges,
            fusion_candidates=fusion_candidates,
            fused_op_skeletons=(),
            cross_routes=(),
            state_accesses=state_accesses,
            persistent_state_manifest=manifest,
            instance_profiles=graph.instance_profiles,
            node_profiles=node_profiles,
            pd_plan_id=graph.pd_plan_id,
        )
        replica_graph.validate(f"train_replica_ir1[{replica_index}]")
        replicas.append(
            TrainPlacedReplica.create(
                replica_index=replica_index,
                graph=replica_graph,
            )
        )
    result = TrainPlacedIR1.create(
        source=graph,
        placement_context=context,
        replicas=tuple(replicas),
    )
    result.validate("train_placed_ir1")
    return result


def place_stage4_ir0(
    graph: IR0,
    context: PlacementContext,
    plan: Stage4PdPlan,
) -> IR1:
    """Place one Stage 4 graph and its exact cross-group routes."""

    if type(plan) is not Stage4PdPlan:
        raise SchemaError("must be a Stage4PdPlan", path="stage4_pd_plan")
    plan.validate("stage4_pd_plan")
    if (
        plan.mode is Stage4PdMode.FUSED
        and (plan.prefill_tp != 1 or plan.decode_tp != 1)
    ):
        raise UnsupportedFeatureError(
            "Stage 4 fused placement supports TP1 only",
            path="stage4_pd_plan.mode",
        )
    if type(graph) is not IR0:
        raise SchemaError("must be an IR0", path="ir0")
    if graph.pd_plan_id != plan.id:
        raise SchemaError(
            "must equal the supplied Stage 4 plan id",
            path="ir0.pd_plan_id",
        )
    selected_instances = {
        plan.prefill_instance_ref,
        plan.decode_instance_ref,
    }
    if {instance.id for instance in graph.instances} != selected_instances:
        raise SchemaError(
            "Stage 4 graph must contain exactly the selected prefill/decode instances",
            path="ir0.instances",
        )
    expected_profiles = tuple(
        sorted(
            (
                InstanceProfileBinding(
                    plan.prefill_instance_ref,
                    plan.prefill_profile.key,
                ),
                InstanceProfileBinding(
                    plan.decode_instance_ref,
                    plan.decode_profile.key,
                ),
            ),
            key=lambda item: (
                item.instance_ref,
                item.profile.stable_id(),
            ),
        )
    )
    if graph.instance_profiles != expected_profiles:
        raise SchemaError(
            "instance profiles must exactly match the Stage 4 plan",
            path="ir0.instance_profiles",
        )

    placed = place_ir0(graph, context)
    cross_routes = (
        build_stage4_cross_group_routes(
            plan,
            context,
            placed.groups,
        )
        if plan.mode is Stage4PdMode.SEPARATED
        else ()
    )
    result = IR1.create(
        producer_pass=placed.producer_pass,
        source_ir0_id=placed.source_ir0_id,
        profile=placed.profile,
        fabric=placed.fabric,
        instances=placed.instances,
        groups=placed.groups,
        nodes=placed.nodes,
        values=placed.values,
        edges=placed.edges,
        fusion_candidates=placed.fusion_candidates,
        fused_op_skeletons=placed.fused_op_skeletons,
        cross_routes=cross_routes,
        state_accesses=placed.state_accesses,
        persistent_state_manifest=placed.persistent_state_manifest,
        instance_profiles=placed.instance_profiles,
        node_profiles=placed.node_profiles,
        pd_plan_id=placed.pd_plan_id,
    )
    result.validate("stage4_placed_ir1")
    return result


def place_stage4_carrier(
    graph: IR0,
    context: PlacementContext,
    plan: Stage4PdPlan,
) -> Stage4PlacedIR1:
    """Place a Stage 4 graph and retain its single-graph provenance."""

    placed = place_stage4_ir0(graph, context, plan)
    result = Stage4PlacedIR1.create(
        source=graph,
        context=context,
        pd_plan=plan,
        graph=placed,
    )
    result.validate_against(graph, context)
    return result


def validate_placement_against(
    result: PlacedIR1Bundle,
    source: ExpandedIR0Bundle,
    context: PlacementContext,
    *,
    path: str = "placed_ir1_bundle",
) -> None:
    """Prove source preservation and recompute every placement/embedding fact."""

    if type(result) is not PlacedIR1Bundle:
        raise SchemaError("must be a PlacedIR1Bundle", path=path)
    if type(source) is not ExpandedIR0Bundle:
        raise SchemaError("must be an ExpandedIR0Bundle", path="source")
    if type(context) is not PlacementContext:
        raise SchemaError("must be a PlacementContext", path="placement_context")
    result.validate_against(source, context, path)
    for entry_index, (placed_entry, source_entry) in enumerate(
        zip(result.entries, source.entries)
    ):
        for group_index, group in enumerate(placed_entry.graph.groups):
            validate_group_against(
                source_entry.graph,
                context,
                group,
                path=(
                    f"{path}.entries[{entry_index}].graph"
                    f".groups[{group_index}]"
                ),
            )
        expected_manifest = _place_persistent_state_manifest(
            source_entry.graph,
            context,
            placed_entry.graph.groups,
        )
        if placed_entry.graph.persistent_state_manifest != expected_manifest:
            raise SchemaError(
                "must equal canonical 64-byte first-fit HBM placement",
                path=(
                    f"{path}.entries[{entry_index}].graph"
                    ".persistent_state_manifest"
                ),
            )


def place_bundle(
    source: ExpandedIR0Bundle,
    context: PlacementContext,
) -> PlacedIR1Bundle:
    """Place all canonical profiles with one immutable hardware context."""

    if type(source) is not ExpandedIR0Bundle:
        raise SchemaError("must be an ExpandedIR0Bundle", path="source")
    if type(context) is not PlacementContext:
        raise SchemaError("must be a PlacementContext", path="placement_context")
    source.validate("source")
    context.validate("placement_context")
    entries = tuple(
        PlacedProfileIR1.create(
            source_expanded_entry_id=entry.id,
            weight=entry.weight,
            graph=place_ir0(entry.graph, context),
        )
        for entry in source.entries
    )
    result = PlacedIR1Bundle.create(
        source_expanded_bundle=source,
        placement_context=context,
        entries=entries,
    )
    validate_placement_against(result, source, context)
    return result


__all__ = [
    "place_bundle",
    "place_ir0",
    "place_stage4_carrier",
    "place_stage4_ir0",
    "validate_placement_against",
]
