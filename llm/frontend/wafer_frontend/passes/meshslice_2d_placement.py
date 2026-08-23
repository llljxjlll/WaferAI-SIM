"""Production placement adapter for exact 2x2 MeshSlice programs."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..policies.swizzle_topo import SwizzleFusionPartition
from ..schema.common import MeshAxisName
from ..schema.ir0 import DeviceMesh, IR0, LogicalInstance, MeshAxis, ParallelAxes
from ..schema.ir1 import IR1, PhysicalInstance, PhysicalNode
from ..schema.placement import PlacementContext
from .discover_fusion import with_discovered_fusion_candidates
from .group_registry import _expected_group


def _require_exact_mesh(source: IR0) -> tuple[LogicalInstance, DeviceMesh]:
    if len(source.instances) != 1:
        raise SchemaError(
            "MeshSlice 2D placement requires exactly one logical instance",
            path="ir0.instances",
        )
    instance = source.instances[0]
    if instance.replicas != 1 or len(instance.meshes) != 1:
        raise SchemaError(
            "MeshSlice 2D placement requires one replica and one mesh",
            path="ir0.instances[0]",
        )
    mesh = instance.meshes[0]
    axes = {axis.name: axis.size for axis in mesh.axes}
    if (
        len(mesh.axes) != 2
        or axes != {MeshAxisName.DP: 2, MeshAxisName.TP: 2}
        or instance.parallel.dp != 2
        or instance.parallel.tp != 2
        or instance.parallel.pp != 1
        or instance.parallel.ep != 1
    ):
        raise SchemaError(
            "MeshSlice 2D placement requires an exact DP2xTP2 mesh",
            path="ir0.instances[0].meshes[0].axes",
        )
    if any(node.mesh_ref != mesh.id for node in source.nodes):
        raise SchemaError(
            "every MeshSlice member must bind the exact 2D mesh",
            path="ir0.nodes",
        )
    return instance, mesh


def place_meshslice_2d_ir1(source: IR0, context: PlacementContext) -> IR1:
    """Place one exact DP2xTP2 workload on a real physical 2x2 rectangle."""

    if type(source) is not IR0 or type(context) is not PlacementContext:
        raise SchemaError(
            "requires typed IR0 and PlacementContext",
            path="meshslice_2d_placement",
        )
    source.validate("ir0")
    context.validate("placement_context")
    instance, mesh = _require_exact_mesh(source)
    if not source.fusion_candidates:
        source = with_discovered_fusion_candidates(source)
    if len(source.fusion_candidates) != 1:
        raise SchemaError(
            "MeshSlice 2D placement requires one discovered fusion candidate",
            path="ir0.fusion_candidates",
        )

    surrogate_mesh = DeviceMesh(
        mesh.id,
        (MeshAxis(MeshAxisName.TP, 4),),
    )
    surrogate_instance = replace(
        instance,
        parallel=ParallelAxes(tp=4, sp=False, dp=1, pp=1, ep=1),
        meshes=(surrogate_mesh,),
    )
    one_d = _expected_group(
        source,
        context,
        surrogate_instance,
        surrogate_mesh,
    )
    dies = {die.id: die for die in context.fabric.dies}
    coords = tuple(dies[item.die_id].coord for item in one_d.placements)
    xs = tuple(sorted({coord[0] for coord in coords}))
    ys = tuple(sorted({coord[1] for coord in coords}))
    if (
        len(one_d.placements) != 4
        or len(xs) != 2
        or len(ys) != 2
        or set(coords) != {(x, y) for x in xs for y in ys}
    ):
        raise SchemaError(
            "selected dies must form one complete physical 2x2 rectangle",
            path="placement_context.fabric.dies",
        )
    placements = tuple(
        replace(
            item,
            logical_coord=(
                ys.index(dies[item.die_id].coord[1]),
                xs.index(dies[item.die_id].coord[0]),
            ),
        )
        for item in one_d.placements
    )
    group = replace(
        one_d,
        logical_shape=(2, 2),
        placements=placements,
    )
    group.validate("meshslice_2d_group")
    pairs = {
        (route.source_rank, route.destination_rank)
        for route in group.embedding.routes
    }
    if pairs != {
        (source_rank, destination_rank)
        for source_rank in range(4)
        for destination_rank in range(4)
        if source_rank != destination_rank
    }:
        raise SchemaError(
            "MeshSlice group requires a route for every ordered rank pair",
            path="meshslice_2d_group.embedding.routes",
        )

    nodes = tuple(
        PhysicalNode(
            id=node.id,
            origin_node_id=node.id,
            instance_id=node.instance_id,
            kind=node.kind,
            phase=node.phase,
            stage=node.stage,
            mesh_ref=node.mesh_ref,
            execution_group_ref=group.id,
            inputs=node.inputs,
            outputs=node.outputs,
            workload=node.workload,
            math=node.math,
            effects=node.effects,
            impl_ref=node.impl_ref,
        )
        for node in source.nodes
    )
    physical_instance = PhysicalInstance(
        id=instance.id,
        origin_instance_id=instance.id,
        role=instance.role,
        die_region=tuple(item.die_id for item in placements),
        group_ids=(group.id,),
        node_ids=tuple(item.id for item in nodes),
    )
    placed = IR1.create(
        producer_pass="meshslice_2d_placement",
        source_ir0_id=source.id,
        profile=source.profile,
        fabric=context.fabric,
        instances=(physical_instance,),
        groups=(group,),
        nodes=nodes,
        values=source.values,
        edges=source.edges,
        fusion_candidates=source.fusion_candidates,
    )
    placed.validate("meshslice_2d_placed")
    partitioned = IR1.create(
        producer_pass="meshslice_2d_partition",
        source_ir0_id=placed.source_ir0_id,
        profile=placed.profile,
        fabric=placed.fabric,
        instances=placed.instances,
        groups=placed.groups,
        nodes=placed.nodes,
        values=placed.values,
        edges=placed.edges,
        fusion_candidates=placed.fusion_candidates,
        fused_op_skeletons=SwizzleFusionPartition().run(placed),
    )
    partitioned.validate("meshslice_2d_partitioned")
    return partitioned


__all__ = ["place_meshslice_2d_ir1"]
