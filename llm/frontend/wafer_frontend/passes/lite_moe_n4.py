"""Dedicated EP placement and production N4 bridge for S3-Lite MoE."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import MeshAxisName
from ..schema.ir0 import StateAccess, StateAccessMode
from ..schema.ir1 import IR1
from ..schema.lite_moe import LiteMoeOracle, LiteMoeSpec
from ..schema.lite_moe_graph import LiteMoeIR0Adapter
from ..schema.lite_moe_n4 import LiteMoeN4IR1, LiteMoePlacedIR1
from ..schema.n4 import FusionPartitionContext, InterDiePlanningContext
from ..schema.persistent_state import (
    HbmBinding,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateManifest,
)
from ..schema.placement import PlacementContext
from .fusion_partition import partition_ir1
from .group_registry import _expected_group
from .inter_die_plan import plan_ir1
from .lite_moe_graph import LiteMoeIR0Validator
from .placement import _physical_instance, _physical_node


def _expert_from_ref(tensor_ref: str) -> int:
    prefix = "S3L0.expert"
    if not tensor_ref.startswith(prefix):
        raise SchemaError(
            "expert parameter tensor_ref has an invalid prefix",
            path="lite_moe_placement.persistent_states",
        )
    suffix = tensor_ref[len(prefix):]
    expert_text, separator, _tail = suffix.partition(".")
    if not separator or expert_text not in ("0", "1", "2", "3"):
        raise SchemaError(
            "expert parameter tensor_ref has an invalid expert index",
            path="lite_moe_placement.persistent_states",
        )
    return int(expert_text)


def _placed_state(
    adapter: LiteMoeIR0Adapter,
    context: PlacementContext,
) -> tuple[PersistentStateManifest, tuple[StateAccess, ...]]:
    state_map: dict[str, PersistentStateDecl] = {}
    placed_declarations: list[PersistentStateDecl] = []
    for declaration in adapter.graph.persistent_states:
        tensor_ref = declaration.identity.tensor_ref
        assert tensor_ref is not None
        expert = _expert_from_ref(tensor_ref)
        identity = PersistentStateIdentity.create(
            kind=declaration.identity.kind,
            instance_ref=declaration.identity.instance_ref,
            mesh_ref=declaration.identity.mesh_ref,
            request_ref=None,
            layer_index=None,
            tensor_ref=tensor_ref,
            shard_index=expert // 2,
            generation=declaration.identity.generation,
        )
        placed = PersistentStateDecl.create(
            identity=identity,
            shape=declaration.shape,
            dtype=declaration.dtype,
            layout=declaration.layout,
            lifetime=declaration.lifetime,
            access=declaration.access,
        )
        state_map[declaration.id] = placed
        placed_declarations.append(placed)

    placed_accesses = tuple(
        StateAccess.create(
            node_ref=access.node_ref,
            state_ref=state_map[access.state_ref].id,
            mode=StateAccessMode.READ,
            rank=state_map[access.state_ref].identity.shard_index,
        )
        for access in adapter.graph.state_accesses
    )
    spaces = {space.die_id: space for space in context.hbm_address_spaces}
    if set(spaces) != {0, 1}:
        raise SchemaError(
            "S3-Lite placement requires exact HBM homes on dies 0 and 1",
            path="placement_context.hbm_address_spaces",
        )
    cursors = {die: spaces[die].base_address for die in spaces}
    bindings: list[HbmBinding] = []
    for declaration in sorted(
        placed_declarations,
        key=lambda item: (item.identity.shard_index, item.identity.tensor_ref or ""),
    ):
        die = declaration.identity.shard_index
        cursor = cursors[die]
        address = (cursor + 63) // 64 * 64
        if address + declaration.tensor_bytes > (
            spaces[die].base_address + spaces[die].size_bytes
        ):
            raise SchemaError(
                "expert parameters exceed home HBM capacity",
                path="lite_moe_placement.persistent_states",
            )
        bindings.append(
            HbmBinding.create(
                state_ref=declaration.id,
                die_id=die,
                address=address,
                size_bytes=declaration.tensor_bytes,
            )
        )
        cursors[die] = address + declaration.tensor_bytes
    return (
        PersistentStateManifest.create(
            address_spaces=context.hbm_address_spaces,
            declarations=tuple(placed_declarations),
            bindings=tuple(bindings),
        ),
        placed_accesses,
    )


def place_lite_moe_adapter(
    adapter: LiteMoeIR0Adapter,
    experiment: object,
    spec: LiteMoeSpec,
    oracle: LiteMoeOracle,
    context: PlacementContext,
) -> LiteMoePlacedIR1:
    """Place one exact EP2 graph without widening Dense N3 placement."""

    LiteMoeIR0Validator.validate(adapter, experiment, spec, oracle)
    if type(context) is not PlacementContext:
        raise SchemaError("must be a PlacementContext", path="placement_context")
    context.validate("placement_context")
    graph = adapter.graph
    instance = graph.instances[0]
    mesh = instance.meshes[0]
    group = replace(
        _expected_group(graph, context, instance, mesh),
        axis=MeshAxisName.EP,
    )
    group.validate("lite_moe_placement.group")
    manifest, state_accesses = _placed_state(adapter, context)
    physical_nodes = tuple(_physical_node(node, group.id) for node in graph.nodes)
    physical_instance = _physical_instance(instance, graph, (group,))
    placed_graph = IR1.create(
        producer_pass="placement",
        source_ir0_id=graph.id,
        profile=graph.profile,
        fabric=context.fabric,
        instances=(physical_instance,),
        groups=(group,),
        nodes=physical_nodes,
        values=graph.values,
        edges=graph.edges,
        fusion_candidates=(),
        fused_op_skeletons=(),
        cross_routes=(),
        state_accesses=state_accesses,
        persistent_state_manifest=manifest,
    )
    placed_graph.validate("lite_moe_placed_ir1.graph")
    result = LiteMoePlacedIR1.create(
        source_adapter_id=adapter.id,
        source_ir0_id=graph.id,
        placement_context_id=context.id,
        graph=placed_graph,
        p2p_bindings=adapter.p2p_bindings,
    )
    validate_lite_moe_placement(result, adapter, context)
    return result


def validate_lite_moe_placement(
    placed: LiteMoePlacedIR1,
    adapter: LiteMoeIR0Adapter,
    context: PlacementContext,
) -> None:
    placed.validate()
    if (
        placed.source_adapter_id != adapter.id
        or placed.source_ir0_id != adapter.graph.id
        or placed.placement_context_id != context.id
    ):
        raise SchemaError("placed provenance mismatch", path="lite_moe_placed_ir1")
    graph = placed.graph
    if len(graph.groups) != 1:
        raise SchemaError("must contain one EP group", path="lite_moe_placed_ir1")
    group = graph.groups[0]
    if (
        group.axis is not MeshAxisName.EP
        or group.logical_shape != (2,)
        or tuple((item.rank, item.die_id) for item in group.placements)
        != ((0, 0), (1, 1))
        or len(group.embedding.routes) != 2
    ):
        raise SchemaError("EP group placement/route mismatch", path="lite_moe_placed_ir1")
    manifest = graph.persistent_state_manifest
    if manifest is None or len(manifest.declarations) != 12:
        raise SchemaError("must place all 12 expert states", path="lite_moe_placed_ir1")
    binding_by_state = {item.state_ref: item for item in manifest.bindings}
    for declaration in manifest.declarations:
        expert = _expert_from_ref(declaration.identity.tensor_ref or "")
        home = expert // 2
        if (
            declaration.identity.shard_index != home
            or binding_by_state[declaration.id].die_id != home
        ):
            raise SchemaError("expert state home mismatch", path="lite_moe_placed_ir1")
    if len(graph.state_accesses) != 24 or any(
        access.rank
        != next(
            declaration.identity.shard_index
            for declaration in manifest.declarations
            if declaration.id == access.state_ref
        )
        for access in graph.state_accesses
    ):
        raise SchemaError("expert access home mismatch", path="lite_moe_placed_ir1")


def build_lite_moe_n4(
    placed: LiteMoePlacedIR1,
    partition_context: FusionPartitionContext,
    planning_context: InterDiePlanningContext,
) -> LiteMoeN4IR1:
    """Run production partition/planning while preserving typed MoE P2P."""

    placed.validate()
    if type(partition_context) is not FusionPartitionContext:
        raise SchemaError("must be a FusionPartitionContext", path="partition_context")
    if type(planning_context) is not InterDiePlanningContext:
        raise SchemaError("must be an InterDiePlanningContext", path="planning_context")
    partition_context.validate("partition_context")
    planning_context.validate("planning_context")
    partitioned = partition_ir1(placed.graph)
    fusion_plans, standalone_plans = plan_ir1(partitioned, planning_context)
    result = LiteMoeN4IR1.create(
        source_placed_id=placed.id,
        source_adapter_id=placed.source_adapter_id,
        source_ir0_id=placed.source_ir0_id,
        placement_context_id=placed.placement_context_id,
        partition_context_id=partition_context.id,
        planning_context_id=planning_context.id,
        graph=partitioned,
        fusion_plans=fusion_plans,
        standalone_plans=standalone_plans,
        p2p_bindings=placed.p2p_bindings,
    )
    validate_lite_moe_n4(result, placed, partition_context, planning_context)
    return result


def validate_lite_moe_n4(
    result: LiteMoeN4IR1,
    placed: LiteMoePlacedIR1,
    partition_context: FusionPartitionContext,
    planning_context: InterDiePlanningContext,
) -> None:
    result.validate()
    if (
        result.source_placed_id != placed.id
        or result.source_adapter_id != placed.source_adapter_id
        or result.source_ir0_id != placed.source_ir0_id
        or result.placement_context_id != placed.placement_context_id
        or result.partition_context_id != partition_context.id
        or result.planning_context_id != planning_context.id
        or result.p2p_bindings != placed.p2p_bindings
    ):
        raise SchemaError("N4 provenance mismatch", path="lite_moe_n4")
    if (
        result.graph.nodes != placed.graph.nodes
        or result.graph.values != placed.graph.values
        or result.graph.edges != placed.graph.edges
        or result.graph.groups != placed.graph.groups
        or result.graph.persistent_state_manifest
        != placed.graph.persistent_state_manifest
        or result.graph.state_accesses != placed.graph.state_accesses
    ):
        raise SchemaError("N4 changed placed MoE semantics", path="lite_moe_n4")


__all__ = [
    "build_lite_moe_n4",
    "place_lite_moe_adapter",
    "validate_lite_moe_n4",
    "validate_lite_moe_placement",
]
