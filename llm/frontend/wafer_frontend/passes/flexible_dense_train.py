"""Dense Train rectangular-mesh baseline plan and forward materializer."""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..policies.registry import RegistryKind, production_registry
from ..schema.experiment import (
    ExperimentSpec,
    InstanceRole,
    PlacementStrategy,
    TrainOptimizer,
    WorkloadMode,
)
from ..schema.flexible_dense_train import (
    DenseTrainAxisGroup,
    DenseTrainGradientTransfer,
    DenseTrainGradientWave,
    DenseTrainParameterTemplate,
    DenseTrainRankAction,
    DenseTrainTapeBinding,
    FlexibleDenseTrainActionKind,
    FlexibleDenseTrainAxis,
    FlexibleDenseTrainCapabilityStatus,
    FlexibleDenseTrainForwardCarrier,
    FlexibleDenseTrainGradientSyncRole,
    FlexibleDenseTrainPlan,
    FlexibleDenseTrainSpec,
)
from ..schema.ir1 import PhysicalFabric
from ..schema.n4 import FusionPartitionContext, InterDiePlanningContext
from ..schema.n5 import IntraDieSchedulingContext, ProjectToIR2Context
from ..schema.persistent_state import HbmAddressSpace
from ..schema.placement import PlacementContext
from ..schema.rect_mesh import RectMeshSpec
from ..schema.serde import canonical_digest
from ..schema._validation_session import builder_validation_session
from .fusion_partition import partition_train_forward
from .inter_die_plan import plan_train_forward
from .intra_die_schedule import schedule_train_forward
from .placement import place_train_forward_ir0
from .project_to_ir2 import project_train_forward
from .train_forward import build_train_forward_ir0
from .train_global_action import build_train_global_action
from .train_link_program import link_train
from .train_lower_program import lower_train


def _validate_source(spec: ExperimentSpec, mesh: RectMeshSpec) -> None:
    if type(spec) is not ExperimentSpec:
        raise SchemaError("must be an ExperimentSpec", path="spec")
    if type(mesh) is not RectMeshSpec:
        raise SchemaError("must be a RectMeshSpec", path="mesh")
    spec.validate("spec")
    mesh.validate("mesh")
    if spec.workload.mode is not WorkloadMode.TRAIN:
        raise UnsupportedFeatureError(
            "requires a forward TRAIN ExperimentSpec adapter",
            path="spec.workload.mode",
        )
    train = spec.workload.train
    assert train is not None
    if train.backward or train.optimizer is not TrainOptimizer.NONE:
        raise SchemaError(
            "v2 adapter preserves old schema: source must remain forward-only/optimizer=none",
            path="spec.workload.train",
        )
    if train.structure.micro_batch_count != 1:
        raise UnsupportedFeatureError(
            "v1 requires one microbatch", path="spec.workload.train.structure"
        )
    if len(spec.parallel.instances) != 1:
        raise UnsupportedFeatureError(
            "v1 requires one train instance", path="spec.parallel.instances"
        )
    instance = spec.parallel.instances[0]
    if (
        instance.role is not InstanceRole.TRAIN
        or instance.dp != mesh.rows
        or instance.tp != mesh.columns
        or instance.dp * instance.tp != mesh.rank_count
        or instance.pp != 1
        or instance.ep != 1
        or instance.replicas != 1
        or instance.sp != (instance.tp > 1)
    ):
        raise SchemaError(
            "requires role=TRAIN, DP=rows, TP=columns, PP=EP=replicas=1 and SP iff TP>1",
            path="spec.parallel.instances[0]",
        )
    if train.global_batch != train.micro_batch * mesh.rows:
        raise SchemaError(
            "global_batch must equal micro_batch*DP for one microbatch",
            path="spec.workload.train.global_batch",
        )
    if spec.placement.strategy is not PlacementStrategy.COMPACT:
        raise UnsupportedFeatureError(
            "v1 reuses compact row-major Train placement",
            path="spec.placement.strategy",
        )


def _action(
    rank: int,
    index: int,
    kind: FlexibleDenseTrainActionKind,
    previous: str | None,
    logical_bytes: int,
    *,
    state_ref: str | None = None,
    op_ref: str | None = None,
    send: int | None = None,
    receive: int | None = None,
    sync_role: FlexibleDenseTrainGradientSyncRole | None = None,
) -> DenseTrainRankAction:
    action_id = f"flex_train.r{rank}.s{index}.{kind.value}"
    return DenseTrainRankAction(
        id=action_id,
        rank=rank,
        index=index,
        kind=kind,
        state_ref=state_ref,
        op_ref=op_ref,
        depends_on=(() if previous is None else (previous,)),
        send_peer_rank=send,
        receive_peer_rank=receive,
        logical_bytes=logical_bytes,
        gradient_sync_role=sync_role,
    )


def _gradient_tree_edges(dp_degree: int) -> tuple[
    tuple[tuple[int, int], ...], tuple[tuple[int, int], ...],
]:
    children = tuple(range(1, dp_degree))
    reduce_edges = tuple(
        (child, (child - 1) // 2)
        for child in sorted(
            children, key=lambda item: (-((item + 1).bit_length()), -item),
        )
    )
    broadcast_edges = tuple(
        (parent, child)
        for child, parent in sorted(
            ((child, (child - 1) // 2) for child in children),
            key=lambda item: (((item[0] + 1).bit_length()), item[0]),
        )
    )
    return reduce_edges, broadcast_edges


def build_flexible_dense_train_plan(
    source: ExperimentSpec,
    mesh: RectMeshSpec,
    *,
    learning_rate: float = 1.0e-3,
) -> FlexibleDenseTrainPlan:
    """Build one deterministic finite forward/backward/sync/SGD carrier."""

    _validate_source(source, mesh)
    graph = build_train_forward_ir0(source)
    contract = FlexibleDenseTrainSpec.create(
        source_experiment_digest=canonical_digest(source),
        mesh=mesh,
        learning_rate=learning_rate,
    )
    instance_id = graph.instances[0].id
    forward_refs = tuple(node.id for node in graph.nodes)
    node_order = {ref: index for index, ref in enumerate(forward_refs)}
    states = tuple(
        sorted(
            graph.persistent_states,
            key=lambda item: (
                item.identity.shard_index,
                item.identity.tensor_ref or "",
                item.id,
            ),
        )
    )
    templates: list[DenseTrainParameterTemplate] = []
    for state in states:
        tensor_ref = state.identity.tensor_ref
        if tensor_ref is None:
            raise SchemaError(
                "Dense Train parameter requires tensor_ref",
                path="forward_graph.persistent_states",
            )
        consumers = tuple(
            sorted(
                {
                    access.node_ref
                    for access in graph.state_accesses
                    if access.state_ref == state.id
                },
                key=node_order.__getitem__,
            )
        )
        column = state.identity.shard_index
        templates.append(
            DenseTrainParameterTemplate(
                state_ref=state.id,
                tensor_ref=tensor_ref,
                tp_shard_index=column,
                owner_ranks=tuple(
                    row * mesh.columns + column for row in range(mesh.rows)
                ),
                forward_consumer_refs=consumers,
                backward_node_refs=tuple(
                    f"backward::{ref}::{state.id}"
                    for ref in reversed(consumers)
                ),
                wgrad_ref=f"wgrad::{tensor_ref}::tp{column}",
                weight_bytes=state.tensor_bytes,
                gradient_bytes=2 * state.tensor_bytes,
            )
        )
    parameter_templates = tuple(templates)
    lm_head_ref = f"{instance_id}.lm_head.weight"
    lm_head_templates = tuple(
        item for item in parameter_templates if item.tensor_ref == lm_head_ref
    )
    if len(lm_head_templates) != mesh.columns:
        raise SchemaError(
            "LM-head parameter shards must exactly cover TP",
            path="forward_graph.persistent_states",
        )
    gradient_bytes = lm_head_templates[0].gradient_bytes
    tp_groups = tuple(
        DenseTrainAxisGroup(
            FlexibleDenseTrainAxis.TP,
            row,
            tuple(row * mesh.columns + column for column in range(mesh.columns)),
        )
        for row in range(mesh.rows)
    )
    dp_groups = tuple(
        DenseTrainAxisGroup(
            FlexibleDenseTrainAxis.DP,
            column,
            tuple(row * mesh.columns + column for row in range(mesh.rows)),
        )
        for column in range(mesh.columns)
    )
    reduce_edges, broadcast_edges = _gradient_tree_edges(mesh.rows)
    wave_list: list[DenseTrainGradientWave] = []
    for template in parameter_templates:
        for edges in (reduce_edges, broadcast_edges):
            for source_row, destination_row in edges:
                wave_list.append(
                    DenseTrainGradientWave(
                        index=len(wave_list) + 1,
                        state_ref=template.state_ref,
                        tp_shard_index=template.tp_shard_index,
                        dp_offset=len(wave_list) + 1,
                        transfers=(DenseTrainGradientTransfer(
                            state_ref=template.state_ref,
                            source_rank=(source_row * mesh.columns + template.tp_shard_index),
                            destination_rank=(destination_row * mesh.columns + template.tp_shard_index),
                            logical_bytes=template.gradient_bytes,
                        ),),
                        max_sessions_per_rank=1,
                    )
                )
    waves = tuple(wave_list)
    rank_actions: list[DenseTrainRankAction] = []
    for rank in range(mesh.rank_count):
        previous = None
        action_index = 0
        row, column = divmod(rank, mesh.columns)
        local_templates = tuple(
            item
            for item in parameter_templates
            if item.tp_shard_index == column
        )

        def append_action(
            kind: FlexibleDenseTrainActionKind,
            logical_bytes: int,
            *,
            state_ref: str | None = None,
            op_ref: str | None = None,
            send: int | None = None,
            receive: int | None = None,
            sync_role: FlexibleDenseTrainGradientSyncRole | None = None,
        ) -> None:
            nonlocal previous, action_index
            action = _action(
                rank,
                action_index,
                kind,
                previous,
                logical_bytes,
                state_ref=state_ref,
                op_ref=op_ref,
                send=send,
                receive=receive,
                sync_role=sync_role,
            )
            rank_actions.append(action)
            previous = action.id
            action_index += 1

        for template in local_templates:
            append_action(
                FlexibleDenseTrainActionKind.PARAMETER_LOAD,
                template.weight_bytes,
                state_ref=template.state_ref,
            )
        for ref in forward_refs:
            append_action(FlexibleDenseTrainActionKind.FORWARD, 0, op_ref=ref)
        tape_bindings = tuple(
            DenseTrainTapeBinding(ref, f"backward::{ref}")
            for ref in reversed(forward_refs)
        )
        for binding in tape_bindings:
            append_action(
                FlexibleDenseTrainActionKind.BACKWARD,
                0,
                op_ref=binding.backward_node_ref,
            )
        for template in local_templates:
            append_action(
                FlexibleDenseTrainActionKind.WEIGHT_GRADIENT,
                template.gradient_bytes,
                state_ref=template.state_ref,
                op_ref=template.wgrad_ref,
            )
        for template in local_templates:
            for child, parent in reduce_edges:
                if row == child:
                    append_action(
                        FlexibleDenseTrainActionKind.GRADIENT_SYNC,
                        template.gradient_bytes,
                        state_ref=template.state_ref,
                        send=parent * mesh.columns + column,
                        sync_role=FlexibleDenseTrainGradientSyncRole.REDUCE_SEND,
                    )
                elif row == parent:
                    append_action(
                        FlexibleDenseTrainActionKind.GRADIENT_SYNC,
                        template.gradient_bytes,
                        state_ref=template.state_ref,
                        receive=child * mesh.columns + column,
                        sync_role=FlexibleDenseTrainGradientSyncRole.REDUCE_RECEIVE,
                    )
            for parent, child in broadcast_edges:
                if row == parent:
                    append_action(
                        FlexibleDenseTrainActionKind.GRADIENT_SYNC,
                        template.gradient_bytes,
                        state_ref=template.state_ref,
                        send=child * mesh.columns + column,
                        sync_role=FlexibleDenseTrainGradientSyncRole.BROADCAST_SEND,
                    )
                elif row == child:
                    append_action(
                        FlexibleDenseTrainActionKind.GRADIENT_SYNC,
                        template.gradient_bytes,
                        state_ref=template.state_ref,
                        receive=parent * mesh.columns + column,
                        sync_role=FlexibleDenseTrainGradientSyncRole.BROADCAST_RECEIVE,
                    )
        for template in local_templates:
            append_action(
                FlexibleDenseTrainActionKind.SGD_UPDATE,
                template.gradient_bytes,
                state_ref=template.state_ref,
            )
        for template in local_templates:
            append_action(
                FlexibleDenseTrainActionKind.PARAMETER_STORE,
                template.weight_bytes,
                state_ref=template.state_ref,
            )
    return FlexibleDenseTrainPlan.create(
        spec=contract,
        source_experiment=source,
        forward_graph=graph,
        forward_node_refs=forward_refs,
        tape_bindings=tuple(
            DenseTrainTapeBinding(ref, f"backward::{ref}")
            for ref in reversed(forward_refs)
        ),
        tp_groups=tp_groups,
        dp_groups=dp_groups,
        parameter_templates=parameter_templates,
        lm_head_gradient_bytes_per_rank=gradient_bytes,
        gradient_waves=waves,
        rank_actions=tuple(rank_actions),
        mesh_foundation=FlexibleDenseTrainCapabilityStatus.VERIFIED,
        forward_graph_status=FlexibleDenseTrainCapabilityStatus.VERIFIED,
        backward_carrier_status=FlexibleDenseTrainCapabilityStatus.VERIFIED,
        forward_lower_link_status=FlexibleDenseTrainCapabilityStatus.NOT_MEASURED,
        backward_lower_link_status=FlexibleDenseTrainCapabilityStatus.OUT_OF_SCOPE,
        program_io_status=FlexibleDenseTrainCapabilityStatus.NOT_MEASURED,
        runtime_status=FlexibleDenseTrainCapabilityStatus.NOT_MEASURED,
        full_model_backward_materialized=False,
    )


def _materialize_flexible_dense_train_forward(
    source: ExperimentSpec,
    mesh: RectMeshSpec,
    fabric: PhysicalFabric,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
    *,
    producer_pass: str = "flexible_dense_train_forward",
) -> FlexibleDenseTrainForwardCarrier:
    """Reuse the production Train forward lower/link chain for one rectangle."""

    plan = build_flexible_dense_train_plan(source, mesh)
    if type(fabric) is not PhysicalFabric:
        raise SchemaError("must be a PhysicalFabric", path="fabric")
    fabric.validate("fabric")
    if (
        fabric.die_grid != mesh.physical_shape
        or len(fabric.dies) != mesh.rank_count
    ):
        raise SchemaError("fabric does not exactly match Mesh", path="fabric")
    placement_context = PlacementContext.create(
        producer_pass=producer_pass,
        fabric=fabric,
        placement=source.placement,
        hbm_address_spaces=hbm_address_spaces,
    )
    placed = place_train_forward_ir0(plan.forward_graph, placement_context)
    partition_context = FusionPartitionContext.create(producer_pass=producer_pass)
    partitioned = partition_train_forward(placed, partition_context)
    registry = production_registry()
    planning_context = InterDiePlanningContext.create(
        producer_pass=producer_pass,
        fused_policy=registry.instantiate(
            RegistryKind.INTER_DIE, "naive"
        ).selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
        ).selection,
    )
    planned = plan_train_forward(partitioned, planning_context)
    projection_context = ProjectToIR2Context.create(
        producer_pass=producer_pass, state_transfers=()
    )
    projected = project_train_forward(planned, projection_context)
    scheduling_context = IntraDieSchedulingContext.create(
        producer_pass=producer_pass,
        policy=registry.instantiate(
            RegistryKind.INTRA_DIE, "naive"
        ).selection,
    )
    scheduled = schedule_train_forward(projected, scheduling_context)
    linked = link_train(lower_train(build_train_global_action(scheduled)))
    return FlexibleDenseTrainForwardCarrier.create(
        plan=plan,
        linked_forward=linked,
    )


def materialize_flexible_dense_train_forward(
    source: ExperimentSpec,
    mesh: RectMeshSpec,
    fabric: PhysicalFabric,
    hbm_address_spaces: tuple[HbmAddressSpace, ...],
    *,
    producer_pass: str = "flexible_dense_train_forward",
) -> FlexibleDenseTrainForwardCarrier:
    """Run the strict Train chain with one private validation session."""

    with builder_validation_session():
        return _materialize_flexible_dense_train_forward(
            source,
            mesh,
            fabric,
            hbm_address_spaces,
            producer_pass=producer_pass,
        )


__all__ = [
    "build_flexible_dense_train_plan",
    "materialize_flexible_dense_train_forward",
]
