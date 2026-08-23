"""Exact, deterministic projection of frozen inter-die plans into semantic IR-2."""

from __future__ import annotations

from dataclasses import replace
import math

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.action import (
    canonical_compute_operand_roles,
    ComputeContract,
    ComputeOperand,
    FusionActionKind,
    FusionPlan,
    ReductionContract,
    StandaloneCollectivePlan,
    SwizzleBoundActionRef,
    SyncContract,
)
from ..schema.common import DType, RoundingMode, Sharding
from ..schema.ir0 import EdgeKind, FusionPattern, OpKind, ReduceOp, StateAccessMode
from ..schema.ir1 import IR1, PhysicalNode
from ..schema.ir2 import (
    DmaContract,
    FusedNodeOrigin,
    IR2ProjectionResult,
    IntraDieDAG,
    IntraDieRegion,
    IntraDieValue,
    OrdinaryNodeOrigin,
    OriginKind,
    RegionLowering,
    SemanticFlow,
    SemanticTask,
    SemanticTaskKind,
    StandaloneNodeOrigin,
    StateIoOrigin,
    StateTransferOrigin,
    StateStagingValue,
    SwizzleIntraDieValue,
    SwizzleNodeOrigin,
    TensorSlice,
    canonical_semantic_flow_id,
    canonical_state_access_view,
    canonical_state_region_id,
    canonical_state_task_id,
    canonical_state_transfer_completion_event,
    canonical_state_transfer_flow_id,
    canonical_state_transfer_payload_id,
    canonical_state_transfer_region_id,
    canonical_state_transfer_task_id,
    canonical_transit_completion_event,
    dense_row_major_view_byte_addend,
)
from ..schema.swizzle_plan import FusedPlan, SwizzleFusionPlan
from ..schema.state_transfer import (
    SegmentedKvStateTransferContract,
    SlicedKvStateTransferContract,
    StateTransferContract,
    StateTransferLike,
)
from .swizzle_project_to_ir2 import project_swizzle_gemm_rs_to_ir2


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _ordinary_task_id(node_id: str, rank: int) -> str:
    return f"task.{node_id}.rank.{rank}.comp"


def _action_task_id(action_id: str) -> str:
    return f"task.{action_id}"


def _ordinary_region_id(node_id: str, rank: int) -> str:
    return f"region.ordinary.{node_id}.rank.{rank}"


def _planned_region_id(category: str, plan_id: str, die_id: int) -> str:
    return f"region.{category}.{plan_id}.die.{die_id}"




def _ordinary_compute(node: PhysicalNode) -> ComputeContract:
    input_roles, output_roles = canonical_compute_operand_roles(
        node.kind,
        node.workload,
        tiled=False,
        path=f"ir1.nodes[{node.id!r}].compute",
    )
    if len(node.inputs) != len(input_roles) or len(node.outputs) != len(
        output_roles
    ):
        _fail(
            "ordinary node operands do not match its exact compute carrier",
            f"ir1.nodes[{node.id!r}]",
        )
    return ComputeContract(
        op_kind=node.kind,
        workload=node.workload,
        math=node.math,
        effects=node.effects,
        impl_ref=node.impl_ref,
        inputs=tuple(
            ComputeOperand(value_id, role)
            for value_id, role in zip(node.inputs, input_roles, strict=True)
        ),
        outputs=tuple(
            ComputeOperand(value_id, role)
            for value_id, role in zip(node.outputs, output_roles, strict=True)
        ),
    )


class NaiveProjectToIR2:
    """Project IR-1 exactly once without changing any inter-die decision."""

    def run(
        self,
        ir1: IR1,
        fusion_plans: tuple[FusedPlan, ...],
        standalone_plans: tuple[StandaloneCollectivePlan, ...],
        *,
        state_transfers: tuple[StateTransferLike, ...],
    ) -> IR2ProjectionResult:
        if type(ir1) is not IR1:
            _fail("must be an IR1", "ir1")
        if type(fusion_plans) is not tuple:
            _fail("must be an immutable tuple", "fusion_plans")
        if type(standalone_plans) is not tuple:
            _fail("must be an immutable tuple", "standalone_plans")
        if type(state_transfers) is not tuple:
            _fail("must be an immutable tuple", "state_transfers")
        ir1.validate("ir1")
        if any(type(plan) not in (FusionPlan, SwizzleFusionPlan) for plan in fusion_plans):
            _fail("entries must be FusionPlan or SwizzleFusionPlan", "fusion_plans")
        if any(
            type(plan) is not StandaloneCollectivePlan
            for plan in standalone_plans
        ):
            _fail("entries must be StandaloneCollectivePlan", "standalone_plans")
        for plan in fusion_plans:
            plan.validate_against(ir1)
        if any(type(plan) is SwizzleFusionPlan for plan in fusion_plans):
            if any(type(plan) is FusionPlan for plan in fusion_plans):
                raise UnsupportedFeatureError(
                    "common IR2 cannot yet mix legacy and Swizzle fusion plans; "
                    "the bridge must preserve the full Swizzle value/buffer contract",
                    path="fusion_plans",
                )
            return project_swizzle_gemm_rs_to_ir2(
                ir1,
                tuple(
                    plan
                    for plan in fusion_plans
                    if type(plan) is SwizzleFusionPlan
                ),
                standalone_plans,
                state_transfers=state_transfers,
            )
        for plan in standalone_plans:
            plan.validate_against(ir1)
        transfer_type: type[object] | None = None
        for index, contract in enumerate(state_transfers):
            if type(contract) not in (
                StateTransferContract,
                SlicedKvStateTransferContract,
                SegmentedKvStateTransferContract,
            ):
                _fail(
                    "entries must be StateTransferLike",
                    f"state_transfers[{index}]",
                )
            contract.validate_against(ir1, f"state_transfers[{index}]")
            if transfer_type is None:
                transfer_type = type(contract)
            elif transfer_type is not type(contract):
                _fail(
                    "one projection cannot mix state transfer contract kinds",
                    f"state_transfers[{index}]",
                )
        if transfer_type is SlicedKvStateTransferContract:
            assert ir1.persistent_state_manifest is not None
            declaration_index = {
                declaration.id: declaration
                for declaration in ir1.persistent_state_manifest.declarations
            }
            access_index = {access.id: access for access in ir1.state_accesses}
            for index, contract in enumerate(state_transfers):
                assert isinstance(contract, SlicedKvStateTransferContract)
                source = declaration_index[
                    access_index[contract.source_state_access_ref].state_ref
                ]
                destination = declaration_index[
                    access_index[contract.destination_state_access_ref].state_ref
                ]
                if (
                    contract.source_local_offset != (0, 0, 0)
                    or contract.destination_local_offset != (0, 0, 0)
                    or contract.source_local_shape[1:] != source.shape[1:]
                    or contract.destination_local_shape[1:]
                    != destination.shape[1:]
                ):
                    raise UnsupportedFeatureError(
                        "Stage 4 Preview projector supports only equal-TP contiguous full-head THD transfers; gather/scatter remains a scheduler boundary",
                        path=f"state_transfers[{index}]",
                    )

        group_index = {group.id: group for group in ir1.groups}
        node_index = {node.id: node for node in ir1.nodes}
        skeleton_index = {
            skeleton.id: skeleton for skeleton in ir1.fused_op_skeletons
        }
        fusion_by_anchor = {
            skeleton_index[plan.fused_op_id].member_node_ids[0]: plan
            for plan in fusion_plans
        }
        fusion_by_member = {
            member_id: plan
            for plan in fusion_plans
            for member_id in skeleton_index[plan.fused_op_id].member_node_ids
        }
        standalone_by_node = {plan.op_id: plan for plan in standalone_plans}
        if (
            len(fusion_by_anchor) != len(fusion_plans)
            or len(fusion_by_member)
            != sum(
                len(skeleton_index[plan.fused_op_id].member_node_ids)
                for plan in fusion_plans
            )
            or len(standalone_by_node) != len(standalone_plans)
            or set(fusion_by_member).intersection(standalone_by_node)
        ):
            _fail("plans must uniquely partition covered nodes", "fusion_plans")
        if any(
            node.kind is OpKind.COLLECTIVE
            and node.id not in fusion_by_member
            and node.id not in standalone_by_node
            for node in ir1.nodes
        ):
            _fail(
                "collective nodes require a fusion or standalone plan",
                "standalone_plans",
            )

        source_order = {node.id: index for index, node in enumerate(ir1.nodes)}
        expected_fusion_order = tuple(
            plan.id
            for plan in sorted(
                fusion_plans,
                key=lambda item: source_order[
                    skeleton_index[item.fused_op_id].member_node_ids[0]
                ],
            )
        )
        expected_standalone_order = tuple(
            plan.id
            for plan in sorted(
                standalone_plans, key=lambda item: source_order[item.op_id]
            )
        )
        if tuple(plan.id for plan in fusion_plans) != expected_fusion_order:
            _fail("must follow fused skeleton source order", "fusion_plans")
        if (
            tuple(plan.id for plan in standalone_plans)
            != expected_standalone_order
        ):
            _fail("must follow collective node source order", "standalone_plans")

        tasks_by_die: dict[int, list[SemanticTask]] = {
            die.id: [] for die in ir1.fabric.dies
        }
        flows_by_die: dict[int, list[SemanticFlow]] = {
            die.id: [] for die in ir1.fabric.dies
        }
        regions_by_die: dict[int, list[IntraDieRegion]] = {
            die.id: [] for die in ir1.fabric.dies
        }
        fusion_ids_by_die: dict[int, list[str]] = {
            die.id: [] for die in ir1.fabric.dies
        }
        standalone_ids_by_die: dict[int, list[str]] = {
            die.id: [] for die in ir1.fabric.dies
        }
        ordinary_ids_by_die: dict[int, list[str]] = {
            die.id: [] for die in ir1.fabric.dies
        }
        state_access_ids_by_die: dict[int, list[str]] = {
            die.id: [] for die in ir1.fabric.dies
        }
        state_values_by_die: dict[int, list[StateStagingValue]] = {
            die.id: [] for die in ir1.fabric.dies
        }
        state_transfer_ids_by_die: dict[int, list[str]] = {
            die.id: [] for die in ir1.fabric.dies
        }

        def append_plan(
            category: str,
            plan: FusionPlan | StandaloneCollectivePlan,
        ) -> None:
            group = group_index[plan.group_ref]
            rank_to_die = {
                placement.rank: placement.die_id
                for placement in group.placements
            }
            route_index = {
                (route.source_rank, route.destination_rank, route.die_path): route
                for route in group.embedding.routes
            }
            chunk_index = {chunk.id: chunk for chunk in plan.chunk_slices}
            send_origin_by_channel: dict[
                str, FusedNodeOrigin | StandaloneNodeOrigin
            ] = {}
            for program in plan.rank_programs:
                for action in program.actions:
                    if action.kind is not FusionActionKind.SEND:
                        continue
                    assert action.logical_channel is not None
                    send_origin_by_channel[action.logical_channel] = (
                        FusedNodeOrigin(
                            OriginKind.FUSED,
                            plan.id,
                            program.rank,
                            action.id,
                        )
                        if category == "fusion"
                        else StandaloneNodeOrigin(
                            OriginKind.STANDALONE_COLLECTIVE,
                            plan.id,
                            program.rank,
                            action.id,
                        )
                    )

            plan_task_ids_by_die: dict[int, list[str]] = {}
            for program in plan.rank_programs:
                die_id = rank_to_die[program.rank]
                for action in program.actions:
                    chunk = chunk_index.get(action.slice_ref or "")
                    tensor_slice = (
                        TensorSlice(chunk.value_id, chunk.offset, chunk.shape)
                        if chunk is not None
                        else None
                    )
                    source_rank = destination_rank = None
                    flow_id = None
                    if action.kind in (
                        FusionActionKind.SEND,
                        FusionActionKind.RECV,
                    ):
                        assert action.peer_rank is not None
                        assert action.logical_channel is not None
                        source_rank, destination_rank = (
                            (program.rank, action.peer_rank)
                            if action.kind is FusionActionKind.SEND
                            else (action.peer_rank, program.rank)
                        )
                        send_origin = send_origin_by_channel[
                            action.logical_channel
                        ]
                        flow_id = canonical_semantic_flow_id(
                            send_origin, action.logical_channel
                        )
                        route = route_index[
                            (
                                source_rank,
                                destination_rank,
                                action.expected_route,
                            )
                        ]
                        flows_by_die[die_id].append(
                            SemanticFlow(
                                id=flow_id,
                                logical_channel=action.logical_channel,
                                pair_route_ref=route.id,
                                source_rank=source_rank,
                                destination_rank=destination_rank,
                                source_die=action.expected_route[0],
                                destination_die=action.expected_route[-1],
                                die_path=action.expected_route,
                                tensor_slice=tensor_slice,
                                bytes=action.bytes,
                                dtype=action.dtype,
                                task_ids=(_action_task_id(action.id),),
                            )
                        )
                    origin = (
                        FusedNodeOrigin(
                            OriginKind.FUSED,
                            plan.id,
                            program.rank,
                            action.id,
                        )
                        if category == "fusion"
                        else StandaloneNodeOrigin(
                            OriginKind.STANDALONE_COLLECTIVE,
                            plan.id,
                            program.rank,
                            action.id,
                        )
                    )
                    region_id = _planned_region_id(category, plan.id, die_id)
                    task = SemanticTask(
                        id=_action_task_id(action.id),
                        kind=SemanticTaskKind(action.kind.value),
                        origin_ref=origin,
                        region_id=region_id,
                        op_kind=(
                            action.compute.op_kind
                            if action.compute is not None
                            else OpKind.COLLECTIVE
                        ),
                        member_id=action.member_id,
                        flow_id=flow_id,
                        chunk_id=action.chunk_id,
                        collective_step=action.collective_step,
                        source_rank=source_rank,
                        destination_rank=destination_rank,
                        tensor_slice=tensor_slice,
                        bytes=action.bytes,
                        dtype=action.dtype,
                        shape=chunk.shape if chunk is not None else (),
                        read_values=action.reads,
                        write_values=action.writes,
                        compute=action.compute,
                        reduction=action.reduction,
                        sync=action.sync,
                        deps=tuple(
                            _action_task_id(dependency)
                            for dependency in action.deps
                        ),
                    )
                    tasks_by_die[die_id].append(task)
                    plan_task_ids_by_die.setdefault(die_id, []).append(task.id)

                    if action.kind is not FusionActionKind.SEND:
                        continue
                    assert action.logical_channel is not None
                    assert action.peer_rank is not None
                    assert tensor_slice is not None
                    source_origin = send_origin_by_channel[
                        action.logical_channel
                    ]
                    transit_flow_id = canonical_semantic_flow_id(
                        source_origin, action.logical_channel
                    )
                    route = route_index[
                        (program.rank, action.peer_rank, action.expected_route)
                    ]
                    for transit_die in action.expected_route[1:-1]:
                        transit_id = (
                            f"task.transit.{transit_flow_id}.die.{transit_die}"
                        )
                        transit_region = _planned_region_id(
                            category, plan.id, transit_die
                        )
                        transit = SemanticTask(
                            id=transit_id,
                            kind=SemanticTaskKind.TRANSIT,
                            origin_ref=source_origin,
                            region_id=transit_region,
                            op_kind=OpKind.COLLECTIVE,
                            member_id=action.member_id,
                            flow_id=transit_flow_id,
                            chunk_id=action.chunk_id,
                            collective_step=action.collective_step,
                            source_rank=program.rank,
                            destination_rank=action.peer_rank,
                            tensor_slice=tensor_slice,
                            bytes=action.bytes,
                            dtype=action.dtype,
                            shape=chunk.shape if chunk is not None else (),
                            read_values=(),
                            write_values=(),
                            compute=None,
                            reduction=None,
                            sync=SyncContract(
                                canonical_transit_completion_event(
                                    transit_flow_id, transit_die
                                ),
                                None,
                                None,
                            ),
                            deps=(),
                        )
                        tasks_by_die[transit_die].append(transit)
                        plan_task_ids_by_die.setdefault(transit_die, []).append(
                            transit_id
                        )
                        flows_by_die[transit_die].append(
                            SemanticFlow(
                                id=transit_flow_id,
                                logical_channel=action.logical_channel,
                                pair_route_ref=route.id,
                                source_rank=program.rank,
                                destination_rank=action.peer_rank,
                                source_die=action.expected_route[0],
                                destination_die=action.expected_route[-1],
                                die_path=action.expected_route,
                                tensor_slice=tensor_slice,
                                bytes=action.bytes,
                                dtype=action.dtype,
                                task_ids=(transit_id,),
                            )
                        )

            for die in ir1.fabric.dies:
                task_ids = tuple(plan_task_ids_by_die.get(die.id, ()))
                if not task_ids:
                    continue
                if category == "fusion":
                    fusion_ids_by_die[die.id].append(plan.id)
                else:
                    standalone_ids_by_die[die.id].append(plan.id)
                regions_by_die[die.id].append(
                    IntraDieRegion(
                        id=_planned_region_id(category, plan.id, die.id),
                        fusion_plan_id=(plan.id if category == "fusion" else None),
                        standalone_collective_plan_id=(
                            plan.id if category == "standalone" else None
                        ),
                        lowering=(
                            RegionLowering.ISA_REGION
                            if category == "fusion"
                            else RegionLowering.STRICT_ACTIONS
                        ),
                        task_ids=task_ids,
                    )
                )

        def append_ordinary(node: PhysicalNode) -> None:
            group = group_index[node.execution_group_ref]
            for placement in group.placements:
                task_id = _ordinary_task_id(node.id, placement.rank)
                region_id = _ordinary_region_id(node.id, placement.rank)
                tasks_by_die[placement.die_id].append(
                    SemanticTask(
                        id=task_id,
                        kind=SemanticTaskKind.COMP,
                        origin_ref=OrdinaryNodeOrigin(
                            OriginKind.ORDINARY, node.id, placement.rank
                        ),
                        region_id=region_id,
                        op_kind=node.kind,
                        member_id=node.id,
                        flow_id=None,
                        chunk_id=None,
                        collective_step=None,
                        source_rank=None,
                        destination_rank=None,
                        tensor_slice=None,
                        bytes=0,
                        dtype=None,
                        shape=(),
                        read_values=node.inputs,
                        write_values=node.outputs,
                        compute=_ordinary_compute(node),
                        reduction=None,
                        sync=None,
                        deps=(),
                    )
                )
                regions_by_die[placement.die_id].append(
                    IntraDieRegion(
                        id=region_id,
                        fusion_plan_id=None,
                        standalone_collective_plan_id=None,
                        lowering=RegionLowering.JSON_COARSE,
                        task_ids=(task_id,),
                    )
                )
                ordinary_ids_by_die[placement.die_id].append(node.id)

        for node in ir1.nodes:
            if node.id in fusion_by_member:
                if node.id in fusion_by_anchor:
                    append_plan("fusion", fusion_by_anchor[node.id])
                continue
            if node.id in standalone_by_node:
                append_plan("standalone", standalone_by_node[node.id])
                continue
            append_ordinary(node)

        state_specs_by_die: dict[int, list[tuple[object, object, str, str]]] = {
            die.id: [] for die in ir1.fabric.dies
        }
        task_updates: dict[str, SemanticTask] = {}
        dma_in_by_target: dict[str, list[SemanticTask]] = {}
        dma_out_by_target: dict[str, list[SemanticTask]] = {}
        state_region_by_id: dict[str, IntraDieRegion] = {}
        manifest = ir1.persistent_state_manifest
        if manifest is not None:
            declaration_index = {
                declaration.id: declaration
                for declaration in manifest.declarations
            }
            binding_index = {
                binding.state_ref: binding for binding in manifest.bindings
            }
            for access in ir1.state_accesses:
                declaration = declaration_index[access.state_ref]
                binding = binding_index[access.state_ref]
                candidates = tuple(
                    (die_id, task)
                    for die_id, die_tasks in tasks_by_die.items()
                    for task in die_tasks
                    if task.kind is SemanticTaskKind.COMP
                    and task.member_id == access.node_ref
                    and getattr(task.origin_ref, "rank", None) == access.rank
                )
                tensor_ref = declaration.identity.tensor_ref
                if (
                    not candidates
                    or (
                        len(candidates) > 1
                        and (
                            tensor_ref is None
                            or any(
                                not isinstance(task.origin_ref, FusedNodeOrigin)
                                for _die_id, task in candidates
                            )
                        )
                    )
                ):
                    _fail(
                        "state access cannot map uniquely to one local compute unit",
                        f"ir1.state_accesses.{access.id}",
                    )
                candidate_dies = {die_id for die_id, _task in candidates}
                if candidate_dies != {binding.die_id}:
                    _fail(
                        "state access targets and HBM home must be on one die",
                        f"ir1.state_accesses.{access.id}",
                    )
                die_id = binding.die_id
                targets = [
                    task_updates.get(original_target.id, original_target)
                    for _die_id, original_target in candidates
                ]
                if any(target.compute is None for target in targets):
                    _fail(
                        "state access targets must have compute contracts",
                        f"ir1.state_accesses.{access.id}",
                    )
                dma_kinds = {
                    StateAccessMode.READ: (SemanticTaskKind.DMA_IN,),
                    StateAccessMode.WRITE: (SemanticTaskKind.DMA_OUT,),
                    StateAccessMode.READ_WRITE: (
                        SemanticTaskKind.DMA_IN,
                        SemanticTaskKind.DMA_OUT,
                    ),
                }[access.mode]
                (
                    staging_shape,
                    staging_layout,
                    dma_offset,
                    dma_shape,
                ) = canonical_state_access_view(
                    ir1,
                    access,
                    declaration,
                    dma_kinds[0],
                )

                staging_shell = StateStagingValue.create(
                    state_access_ref=access.id,
                    state_ref=access.state_ref,
                    shape=staging_shape,
                    dtype=declaration.dtype,
                    logical_layout=staging_layout,
                    producer_tasks=(),
                    consumer_tasks=(),
                )
                staging_id = staging_shell.id
                rewritten_targets: list[SemanticTask] = []
                for target in targets:
                    assert target.compute is not None
                    compute_inputs = list(target.compute.inputs)
                    if tensor_ref is not None:
                        source_matches = tuple(
                            index
                            for index, operand in enumerate(compute_inputs)
                            if operand.value_id == tensor_ref
                        )
                        staging_matches = tuple(
                            index
                            for index, operand in enumerate(compute_inputs)
                            if operand.value_id == staging_id
                        )
                        if len(source_matches) == 1 and not staging_matches:
                            operand_index = source_matches[0]
                            compute_inputs[operand_index] = replace(
                                compute_inputs[operand_index],
                                value_id=staging_id,
                            )
                        elif len(staging_matches) != 1 or source_matches:
                            _fail(
                                "parameter access must identify one shared compute operand",
                                f"ir1.state_accesses.{access.id}",
                            )
                    rewritten_compute = replace(
                        target.compute,
                        inputs=tuple(compute_inputs),
                    )
                    rewritten_targets.append(
                        replace(
                            target,
                            read_values=tuple(
                                operand.value_id
                                for operand in rewritten_compute.inputs
                            ),
                            compute=rewritten_compute,
                        )
                    )
                target_ids = tuple(target.id for target in rewritten_targets)
                for dma_kind in dma_kinds:
                    (
                        _staging_shape,
                        _staging_layout,
                        dma_offset,
                        dma_shape,
                    ) = canonical_state_access_view(
                        ir1,
                        access,
                        declaration,
                        dma_kind,
                    )
                    root_slice = TensorSlice(
                        staging_id,
                        (0,) * len(staging_shape),
                        staging_shape,
                    )
                    dma_slice = TensorSlice(
                        staging_id,
                        dma_offset,
                        dma_shape,
                    )
                    state_offset_bytes = (
                        0
                        if tensor_ref is not None
                        else dense_row_major_view_byte_addend(
                            root_slice,
                            dma_slice,
                            declaration.dtype,
                            path=f"ir1.state_accesses.{access.id}",
                        )
                    )
                    element_bytes = declaration.tensor_bytes // math.prod(
                        declaration.shape
                    )
                    payload_bytes = math.prod(dma_shape) * element_bytes
                    task_id = canonical_state_task_id(access.id, dma_kind)
                    region_id = canonical_state_region_id(access.id, dma_kind)
                    dma_task = SemanticTask(
                        id=task_id,
                        kind=dma_kind,
                        origin_ref=StateIoOrigin(
                            OriginKind.STATE_IO,
                            access.id,
                            access.node_ref,
                            access.rank,
                        ),
                        region_id=region_id,
                        op_kind=None,
                        member_id=None,
                        flow_id=None,
                        chunk_id=None,
                        collective_step=None,
                        source_rank=None,
                        destination_rank=None,
                        tensor_slice=dma_slice,
                        bytes=payload_bytes,
                        dtype=declaration.dtype,
                        shape=dma_shape,
                        read_values=(
                            (staging_id,)
                            if dma_kind is SemanticTaskKind.DMA_OUT
                            else ()
                        ),
                        write_values=(
                            (staging_id,)
                            if dma_kind is SemanticTaskKind.DMA_IN
                            else ()
                        ),
                        compute=None,
                        reduction=None,
                        sync=None,
                        deps=(
                            target_ids
                            if dma_kind is SemanticTaskKind.DMA_OUT
                            else ()
                        ),
                        dma=DmaContract(
                            state_ref=access.state_ref,
                            local_value_ref=staging_id,
                            state_offset_bytes=state_offset_bytes,
                            access_task_refs=target_ids,
                        ),
                    )
                    state_region_by_id[region_id] = IntraDieRegion(
                        id=region_id,
                        fusion_plan_id=None,
                        standalone_collective_plan_id=None,
                        lowering=RegionLowering.STRICT_STATE_IO,
                        task_ids=(task_id,),
                    )
                    if dma_kind is SemanticTaskKind.DMA_IN:
                        dma_in_by_target.setdefault(
                            target_ids[0], []
                        ).append(dma_task)
                        rewritten_targets = [
                            replace(
                                target,
                                deps=tuple(
                                    dict.fromkeys(target.deps + (task_id,))
                                ),
                            )
                            for target in rewritten_targets
                        ]
                    else:
                        dma_out_by_target.setdefault(
                            target_ids[-1], []
                        ).append(dma_task)
                for target in rewritten_targets:
                    task_updates[target.id] = target
                state_access_ids_by_die[die_id].append(access.id)
                state_specs_by_die[die_id].append(
                    (
                        access,
                        declaration,
                        target_ids,
                        staging_id,
                        staging_shape,
                        staging_layout,
                    )
                )

        for die_id, die_tasks in tasks_by_die.items():
            rewritten_tasks: list[SemanticTask] = []
            for original_task in die_tasks:
                target = task_updates.get(original_task.id, original_task)
                rewritten_tasks.extend(dma_in_by_target.get(target.id, ()))
                rewritten_tasks.append(target)
                rewritten_tasks.extend(dma_out_by_target.get(target.id, ()))
            tasks_by_die[die_id] = rewritten_tasks
            rewritten_regions: list[IntraDieRegion] = []
            for region in regions_by_die[die_id]:
                target_ids = set(region.task_ids)
                inbound = tuple(
                    dma
                    for target_id in region.task_ids
                    for dma in dma_in_by_target.get(target_id, ())
                )
                outbound = tuple(
                    dma
                    for target_id in region.task_ids
                    for dma in dma_out_by_target.get(target_id, ())
                )
                rewritten_regions.extend(
                    state_region_by_id[dma.region_id]
                    for dma in inbound
                    if dma.region_id is not None
                )
                rewritten_regions.append(region)
                rewritten_regions.extend(
                    state_region_by_id[dma.region_id]
                    for dma in outbound
                    if dma.region_id is not None
                )
            regions_by_die[die_id] = rewritten_regions
        state_spec_by_access = {
            access.id: (
                die_id,
                declaration,
                target_ids,
                staging_id,
                staging_shape,
                staging_layout,
            )
            for die_id, specs in state_specs_by_die.items()
            for (
                access,
                declaration,
                target_ids,
                staging_id,
                staging_shape,
                staging_layout,
            ) in specs
        }
        pair_route_index = {
            route.id: route
            for group in ir1.groups
            for route in group.embedding.routes
        }
        cross_route_index = {route.id: route for route in ir1.cross_routes}
        base_task_position_by_die = {
            die_id: {task.id: index for index, task in enumerate(tasks)}
            for die_id, tasks in tasks_by_die.items()
        }
        transfer_task_key_by_die: dict[
            int, dict[str, tuple[int, int, int, int]]
        ] = {die.id: {} for die in ir1.fabric.dies}

        for transfer_ordinal, contract in enumerate(state_transfers):
            (
                source_die,
                source_declaration,
                source_target_ids,
                source_staging_id,
                _source_staging_shape,
                _source_staging_layout,
            ) = state_spec_by_access[contract.source_state_access_ref]
            (
                destination_die,
                destination_declaration,
                destination_target_ids,
                destination_staging_id,
                _destination_staging_shape,
                _destination_staging_layout,
            ) = state_spec_by_access[contract.destination_state_access_ref]
            if isinstance(contract, SegmentedKvStateTransferContract):
                route = cross_route_index[contract.cross_group_route_ref]
                source_region_id = canonical_state_transfer_region_id(
                    contract.id, source_die
                )
                destination_region_id = canonical_state_transfer_region_id(
                    contract.id, destination_die
                )
                source_anchor = max(
                    base_task_position_by_die[source_die][task_id]
                    for task_id in source_target_ids
                )
                destination_anchor = min(
                    base_task_position_by_die[destination_die][task_id]
                    for task_id in destination_target_ids
                )
                source_region_tasks: list[str] = []
                destination_region_tasks: list[str] = []
                destination_wait_ids: list[str] = []
                transit_region_tasks: dict[int, list[str]] = {
                    die_id: [] for die_id in route.die_path[1:-1]
                }
                for segment_index, segment in enumerate(contract.segments):
                    flow_id = canonical_state_transfer_flow_id(
                        contract.id, segment_index
                    )
                    logical_channel = (
                        f"state_transfer.{contract.id}.segment.{segment_index}"
                    )
                    logical_slice = TensorSlice(
                        canonical_state_transfer_payload_id(
                            contract.id, segment_index
                        ),
                        (0,) * len(segment.source_local_shape),
                        segment.source_local_shape,
                    )
                    send_id = canonical_state_transfer_task_id(
                        contract.id, SemanticTaskKind.SEND, source_die,
                        segment_index,
                    )
                    send = SemanticTask(
                        id=send_id,
                        kind=SemanticTaskKind.SEND,
                        origin_ref=StateTransferOrigin(
                            OriginKind.STATE_TRANSFER,
                            contract.id,
                            route.source_rank,
                            segment_index,
                        ),
                        region_id=source_region_id,
                        op_kind=OpKind.P2P,
                        member_id=None,
                        flow_id=flow_id,
                        chunk_id=None,
                        collective_step=None,
                        source_rank=route.source_rank,
                        destination_rank=route.destination_rank,
                        tensor_slice=TensorSlice(
                            source_staging_id,
                            segment.source_local_offset,
                            segment.source_local_shape,
                        ),
                        bytes=segment.bytes,
                        dtype=source_declaration.dtype,
                        shape=segment.source_local_shape,
                        read_values=(source_staging_id,),
                        write_values=(),
                        compute=None,
                        reduction=None,
                        sync=SyncContract(
                            canonical_state_transfer_completion_event(
                                contract.id, SemanticTaskKind.SEND,
                                source_die, segment_index,
                            ),
                            None,
                            None,
                        ),
                        deps=source_target_ids,
                    )
                    tasks_by_die[source_die].append(send)
                    source_region_tasks.append(send.id)
                    transfer_task_key_by_die[source_die][send.id] = (
                        source_anchor, 2, transfer_ordinal, segment_index
                    )

                    recv_id = canonical_state_transfer_task_id(
                        contract.id, SemanticTaskKind.RECV,
                        destination_die, segment_index,
                    )
                    recv_event = canonical_state_transfer_completion_event(
                        contract.id, SemanticTaskKind.RECV,
                        destination_die, segment_index,
                    )
                    recv = SemanticTask(
                        id=recv_id,
                        kind=SemanticTaskKind.RECV,
                        origin_ref=StateTransferOrigin(
                            OriginKind.STATE_TRANSFER,
                            contract.id,
                            route.destination_rank,
                            segment_index,
                        ),
                        region_id=destination_region_id,
                        op_kind=OpKind.P2P,
                        member_id=None,
                        flow_id=flow_id,
                        chunk_id=None,
                        collective_step=None,
                        source_rank=route.source_rank,
                        destination_rank=route.destination_rank,
                        tensor_slice=TensorSlice(
                            destination_staging_id,
                            segment.destination_local_offset,
                            segment.destination_local_shape,
                        ),
                        bytes=segment.bytes,
                        dtype=destination_declaration.dtype,
                        shape=segment.destination_local_shape,
                        read_values=(),
                        write_values=(destination_staging_id,),
                        compute=None,
                        reduction=None,
                        sync=SyncContract(recv_event, None, None),
                        deps=(),
                    )
                    wait_id = canonical_state_transfer_task_id(
                        contract.id, SemanticTaskKind.WAIT,
                        destination_die, segment_index,
                    )
                    wait = SemanticTask(
                        id=wait_id,
                        kind=SemanticTaskKind.WAIT,
                        origin_ref=StateTransferOrigin(
                            OriginKind.STATE_TRANSFER,
                            contract.id,
                            route.destination_rank,
                            segment_index,
                        ),
                        region_id=destination_region_id,
                        op_kind=OpKind.P2P,
                        member_id=None,
                        flow_id=None,
                        chunk_id=None,
                        collective_step=None,
                        source_rank=None,
                        destination_rank=None,
                        tensor_slice=None,
                        bytes=0,
                        dtype=None,
                        shape=(),
                        read_values=(),
                        write_values=(),
                        compute=None,
                        reduction=None,
                        sync=SyncContract(
                            canonical_state_transfer_completion_event(
                                contract.id, SemanticTaskKind.WAIT,
                                destination_die, segment_index,
                            ),
                            recv_event,
                            None,
                        ),
                        deps=(recv.id,),
                    )
                    tasks_by_die[destination_die].extend((recv, wait))
                    destination_region_tasks.extend((recv.id, wait.id))
                    destination_wait_ids.append(wait.id)
                    transfer_task_key_by_die[destination_die][recv.id] = (
                        destination_anchor, -2, transfer_ordinal,
                        segment_index * 2,
                    )
                    transfer_task_key_by_die[destination_die][wait.id] = (
                        destination_anchor, -2, transfer_ordinal,
                        segment_index * 2 + 1,
                    )

                    local_transport_ids = {
                        source_die: send.id,
                        destination_die: recv.id,
                    }
                    for hop_index, transit_die in enumerate(
                        route.die_path[1:-1], 1
                    ):
                        transit_id = canonical_state_transfer_task_id(
                            contract.id, SemanticTaskKind.TRANSIT,
                            transit_die, segment_index,
                        )
                        transit = SemanticTask(
                            id=transit_id,
                            kind=SemanticTaskKind.TRANSIT,
                            origin_ref=StateTransferOrigin(
                                OriginKind.STATE_TRANSFER,
                                contract.id,
                                route.source_rank,
                                segment_index,
                            ),
                            region_id=canonical_state_transfer_region_id(
                                contract.id, transit_die
                            ),
                            op_kind=OpKind.P2P,
                            member_id=None,
                            flow_id=flow_id,
                            chunk_id=None,
                            collective_step=None,
                            source_rank=route.source_rank,
                            destination_rank=route.destination_rank,
                            tensor_slice=logical_slice,
                            bytes=segment.bytes,
                            dtype=source_declaration.dtype,
                            shape=segment.source_local_shape,
                            read_values=(),
                            write_values=(),
                            compute=None,
                            reduction=None,
                            sync=SyncContract(
                                canonical_state_transfer_completion_event(
                                    contract.id,
                                    SemanticTaskKind.TRANSIT,
                                    transit_die,
                                    segment_index,
                                ),
                                None,
                                None,
                            ),
                            deps=(),
                        )
                        tasks_by_die[transit_die].append(transit)
                        transit_region_tasks[transit_die].append(transit.id)
                        transfer_task_key_by_die[transit_die][transit.id] = (
                            len(base_task_position_by_die[transit_die]),
                            2, transfer_ordinal, segment_index,
                        )
                        local_transport_ids[transit_die] = transit.id

                    for die_id in route.die_path:
                        flows_by_die[die_id].append(
                            SemanticFlow(
                                id=flow_id,
                                logical_channel=logical_channel,
                                pair_route_ref=route.id,
                                source_rank=route.source_rank,
                                destination_rank=route.destination_rank,
                                source_die=source_die,
                                destination_die=destination_die,
                                die_path=route.die_path,
                                tensor_slice=logical_slice,
                                bytes=segment.bytes,
                                dtype=source_declaration.dtype,
                                task_ids=(local_transport_ids[die_id],),
                            )
                        )

                tasks_by_die[destination_die] = [
                    replace(
                        task,
                        deps=tuple(
                            dict.fromkeys(
                                task.deps + tuple(destination_wait_ids)
                            )
                        ),
                    )
                    if task.id in destination_target_ids
                    else task
                    for task in tasks_by_die[destination_die]
                ]
                regions_by_die[source_die].append(
                    IntraDieRegion(
                        id=source_region_id,
                        fusion_plan_id=None,
                        standalone_collective_plan_id=None,
                        lowering=RegionLowering.STRICT_STATE_TRANSFER,
                        task_ids=tuple(source_region_tasks),
                        state_transfer_ref=contract.id,
                    )
                )
                regions_by_die[destination_die].append(
                    IntraDieRegion(
                        id=destination_region_id,
                        fusion_plan_id=None,
                        standalone_collective_plan_id=None,
                        lowering=RegionLowering.STRICT_STATE_TRANSFER,
                        task_ids=tuple(destination_region_tasks),
                        state_transfer_ref=contract.id,
                    )
                )
                for transit_die, task_ids in transit_region_tasks.items():
                    regions_by_die[transit_die].append(
                        IntraDieRegion(
                            id=canonical_state_transfer_region_id(
                                contract.id, transit_die
                            ),
                            fusion_plan_id=None,
                            standalone_collective_plan_id=None,
                            lowering=RegionLowering.STRICT_STATE_TRANSFER,
                            task_ids=tuple(task_ids),
                            state_transfer_ref=contract.id,
                        )
                    )
                for die_id in route.die_path:
                    state_transfer_ids_by_die[die_id].append(contract.id)
                continue
            if isinstance(contract, SlicedKvStateTransferContract):
                route = cross_route_index[contract.cross_group_route_ref]
                source_offset = contract.source_local_offset
                source_shape = contract.source_local_shape
                destination_offset = contract.destination_local_offset
                destination_shape = contract.destination_local_shape
                payload_bytes = contract.bytes
            else:
                route = pair_route_index[contract.pair_route_ref]
                source_offset = (0,) * len(source_declaration.shape)
                source_shape = source_declaration.shape
                destination_offset = (
                    (0,) * len(destination_declaration.shape)
                )
                destination_shape = destination_declaration.shape
                payload_bytes = source_declaration.tensor_bytes
            flow_id = canonical_state_transfer_flow_id(contract.id)
            logical_channel = f"state_transfer.{contract.id}"
            logical_slice = TensorSlice(
                canonical_state_transfer_payload_id(contract.id),
                (0,) * len(source_shape),
                source_shape,
            )
            source_region_id = canonical_state_transfer_region_id(
                contract.id, source_die
            )
            send_id = canonical_state_transfer_task_id(
                contract.id, SemanticTaskKind.SEND, source_die
            )
            send = SemanticTask(
                id=send_id,
                kind=SemanticTaskKind.SEND,
                origin_ref=StateTransferOrigin(
                    OriginKind.STATE_TRANSFER, contract.id, route.source_rank
                ),
                region_id=source_region_id,
                op_kind=OpKind.P2P,
                member_id=None,
                flow_id=flow_id,
                chunk_id=None,
                collective_step=None,
                source_rank=route.source_rank,
                destination_rank=route.destination_rank,
                tensor_slice=TensorSlice(
                    source_staging_id,
                    source_offset,
                    source_shape,
                ),
                bytes=payload_bytes,
                dtype=source_declaration.dtype,
                shape=source_shape,
                read_values=(source_staging_id,),
                write_values=(),
                compute=None,
                reduction=None,
                sync=SyncContract(
                    canonical_state_transfer_completion_event(
                        contract.id, SemanticTaskKind.SEND, source_die
                    ),
                    None,
                    None,
                ),
                deps=source_target_ids,
            )
            tasks_by_die[source_die].append(send)
            regions_by_die[source_die].append(
                IntraDieRegion(
                    id=source_region_id,
                    fusion_plan_id=None,
                    standalone_collective_plan_id=None,
                    lowering=RegionLowering.STRICT_STATE_TRANSFER,
                    task_ids=(send.id,),
                    state_transfer_ref=contract.id,
                )
            )
            source_anchor = max(
                base_task_position_by_die[source_die][task_id]
                for task_id in source_target_ids
            )
            transfer_task_key_by_die[source_die][send.id] = (
                source_anchor, 2, transfer_ordinal, 0
            )

            destination_region_id = canonical_state_transfer_region_id(
                contract.id, destination_die
            )
            recv_id = canonical_state_transfer_task_id(
                contract.id, SemanticTaskKind.RECV, destination_die
            )
            recv_event = canonical_state_transfer_completion_event(
                contract.id, SemanticTaskKind.RECV, destination_die
            )
            recv = SemanticTask(
                id=recv_id,
                kind=SemanticTaskKind.RECV,
                origin_ref=StateTransferOrigin(
                    OriginKind.STATE_TRANSFER,
                    contract.id,
                    route.destination_rank,
                ),
                region_id=destination_region_id,
                op_kind=OpKind.P2P,
                member_id=None,
                flow_id=flow_id,
                chunk_id=None,
                collective_step=None,
                source_rank=route.source_rank,
                destination_rank=route.destination_rank,
                tensor_slice=TensorSlice(
                    destination_staging_id,
                    destination_offset,
                    destination_shape,
                ),
                bytes=payload_bytes,
                dtype=destination_declaration.dtype,
                shape=destination_shape,
                read_values=(),
                write_values=(destination_staging_id,),
                compute=None,
                reduction=None,
                sync=SyncContract(recv_event, None, None),
                deps=(),
            )
            wait_id = canonical_state_transfer_task_id(
                contract.id, SemanticTaskKind.WAIT, destination_die
            )
            wait = SemanticTask(
                id=wait_id,
                kind=SemanticTaskKind.WAIT,
                origin_ref=StateTransferOrigin(
                    OriginKind.STATE_TRANSFER,
                    contract.id,
                    route.destination_rank,
                ),
                region_id=destination_region_id,
                op_kind=OpKind.P2P,
                member_id=None,
                flow_id=None,
                chunk_id=None,
                collective_step=None,
                source_rank=None,
                destination_rank=None,
                tensor_slice=None,
                bytes=0,
                dtype=None,
                shape=(),
                read_values=(),
                write_values=(),
                compute=None,
                reduction=None,
                sync=SyncContract(
                    canonical_state_transfer_completion_event(
                        contract.id, SemanticTaskKind.WAIT, destination_die
                    ),
                    recv_event,
                    None,
                ),
                deps=(recv.id,),
            )
            tasks_by_die[destination_die].extend((recv, wait))
            tasks_by_die[destination_die] = [
                replace(
                    task,
                    deps=tuple(dict.fromkeys(task.deps + (wait.id,))),
                )
                if task.id in destination_target_ids
                else task
                for task in tasks_by_die[destination_die]
            ]
            regions_by_die[destination_die].append(
                IntraDieRegion(
                    id=destination_region_id,
                    fusion_plan_id=None,
                    standalone_collective_plan_id=None,
                    lowering=RegionLowering.STRICT_STATE_TRANSFER,
                    task_ids=(recv.id, wait.id),
                    state_transfer_ref=contract.id,
                )
            )
            destination_anchor = min(
                base_task_position_by_die[destination_die][task_id]
                for task_id in destination_target_ids
            )
            transfer_task_key_by_die[destination_die][recv.id] = (
                destination_anchor, -2, transfer_ordinal, 0
            )
            transfer_task_key_by_die[destination_die][wait.id] = (
                destination_anchor, -2, transfer_ordinal, 1
            )

            local_transport_ids = {
                source_die: send.id,
                destination_die: recv.id,
            }
            for hop_index, transit_die in enumerate(route.die_path[1:-1], 1):
                transit_id = canonical_state_transfer_task_id(
                    contract.id, SemanticTaskKind.TRANSIT, transit_die
                )
                transit_region_id = canonical_state_transfer_region_id(
                    contract.id, transit_die
                )
                transit = SemanticTask(
                    id=transit_id,
                    kind=SemanticTaskKind.TRANSIT,
                    origin_ref=StateTransferOrigin(
                        OriginKind.STATE_TRANSFER,
                        contract.id,
                        route.source_rank,
                    ),
                    region_id=transit_region_id,
                    op_kind=OpKind.P2P,
                    member_id=None,
                    flow_id=flow_id,
                    chunk_id=None,
                    collective_step=None,
                    source_rank=route.source_rank,
                    destination_rank=route.destination_rank,
                    tensor_slice=logical_slice,
                    bytes=payload_bytes,
                    dtype=source_declaration.dtype,
                    shape=source_shape,
                    read_values=(),
                    write_values=(),
                    compute=None,
                    reduction=None,
                    sync=SyncContract(
                        canonical_state_transfer_completion_event(
                            contract.id,
                            SemanticTaskKind.TRANSIT,
                            transit_die,
                        ),
                        None,
                        None,
                    ),
                    deps=(),
                )
                tasks_by_die[transit_die].append(transit)
                regions_by_die[transit_die].append(
                    IntraDieRegion(
                        id=transit_region_id,
                        fusion_plan_id=None,
                        standalone_collective_plan_id=None,
                        lowering=RegionLowering.STRICT_STATE_TRANSFER,
                        task_ids=(transit.id,),
                        state_transfer_ref=contract.id,
                    )
                )
                transfer_task_key_by_die[transit_die][transit.id] = (
                    len(base_task_position_by_die[transit_die]),
                    2,
                    transfer_ordinal,
                    hop_index,
                )
                local_transport_ids[transit_die] = transit.id

            for die_id in route.die_path:
                flows_by_die[die_id].append(
                    SemanticFlow(
                        id=flow_id,
                        logical_channel=logical_channel,
                        pair_route_ref=route.id,
                        source_rank=route.source_rank,
                        destination_rank=route.destination_rank,
                        source_die=route.die_path[0],
                        destination_die=route.die_path[-1],
                        die_path=route.die_path,
                        tensor_slice=logical_slice,
                        bytes=payload_bytes,
                        dtype=source_declaration.dtype,
                        task_ids=(local_transport_ids[die_id],),
                    )
                )
                state_transfer_ids_by_die[die_id].append(contract.id)

        skeleton_by_member = {
            member_id: plan
            for member_id, plan in fusion_by_member.items()
        }

        def coverage(node_id: str) -> tuple[str, str]:
            if node_id in skeleton_by_member:
                return ("fusion", skeleton_by_member[node_id].id)
            if node_id in standalone_by_node:
                return ("standalone", standalone_by_node[node_id].id)
            return ("ordinary", node_id)

        def belongs(task: SemanticTask, node_id: str, whole: bool) -> bool:
            kind, unit_id = coverage(node_id)
            origin = task.origin_ref
            if kind == "ordinary":
                return isinstance(origin, OrdinaryNodeOrigin) and origin.op_id == unit_id
            if kind == "fusion":
                return (
                    isinstance(origin, FusedNodeOrigin)
                    and origin.plan_id == unit_id
                    and (whole or task.member_id == node_id)
                )
            return (
                isinstance(origin, StandaloneNodeOrigin)
                and origin.collective_plan_id == unit_id
                and (whole or task.member_id == node_id)
            )

        def reads_source(task: SemanticTask, value_id: str) -> bool:
            if value_id in task.read_values:
                return True
            return (
                task.kind is SemanticTaskKind.COMP
                and task.compute is not None
                and task.compute.tile is not None
                and any(
                    binding.source_value_id == value_id
                    and binding.operand_id in task.read_values
                    for binding in task.compute.tile.input_slices
                )
            )

        # Add only same-die IR-1 graph dependencies to already translated N4 deps.
        for die_id, tasks in tasks_by_die.items():
            deps_by_task = {task.id: list(task.deps) for task in tasks}
            for edge in sorted(ir1.edges, key=lambda item: item.id):
                if coverage(edge.source_node) == coverage(edge.destination_node):
                    continue
                whole = edge.kind is EdgeKind.CONTROL
                destination_kind = coverage(edge.destination_node)[0]
                expected_entry_kind = {
                    "ordinary": SemanticTaskKind.COMP,
                    "fusion": SemanticTaskKind.COMP,
                    "standalone": SemanticTaskKind.LOCAL_COPY,
                }[destination_kind]
                entries = tuple(
                    sorted(
                        (
                            task
                            for task in tasks
                            if belongs(task, edge.destination_node, whole)
                            and task.kind is expected_entry_kind
                            and (
                                whole
                                or reads_source(task, edge.value_id)
                            )
                        ),
                        key=lambda item: item.id,
                    )
                )
                source_kind = coverage(edge.source_node)[0]
                expected_completion_kind = {
                    "ordinary": SemanticTaskKind.COMP,
                    "fusion": SemanticTaskKind.REDUCE,
                    "standalone": SemanticTaskKind.BARRIER,
                }[source_kind]
                completions = tuple(
                    sorted(
                        (
                            task
                            for task in tasks
                            if belongs(task, edge.source_node, whole)
                            and task.kind is expected_completion_kind
                            and (
                                whole
                                or source_kind == "standalone"
                                or edge.value_id in task.write_values
                            )
                        ),
                        key=lambda item: item.id,
                    )
                )
                for entry in entries:
                    deps_by_task[entry.id].extend(
                        completion.id for completion in completions
                    )
            tasks_by_die[die_id] = [
                replace(
                    task,
                    deps=tuple(dict.fromkeys(deps_by_task[task.id])),
                )
                for task in tasks
            ]

        for die_id, tasks in tasks_by_die.items():
            base_positions = base_task_position_by_die[die_id]
            transfer_positions = transfer_task_key_by_die[die_id]

            def local_task_key(task: SemanticTask) -> tuple[int, int, int, int]:
                transfer_key = transfer_positions.get(task.id)
                if transfer_key is not None:
                    return transfer_key
                return (base_positions[task.id], 0, 0, 0)

            tasks_by_die[die_id] = sorted(tasks, key=local_task_key)
            final_positions = {
                task.id: index
                for index, task in enumerate(tasks_by_die[die_id])
            }
            flows_by_die[die_id] = sorted(
                flows_by_die[die_id],
                key=lambda flow: final_positions[flow.task_ids[0]],
            )
            regions_by_die[die_id] = sorted(
                regions_by_die[die_id],
                key=lambda region: min(
                    final_positions[task_id] for task_id in region.task_ids
                ),
            )
            task_tuple = tuple(tasks_by_die[die_id])
            for (
                access,
                declaration,
                _target_ids,
                staging_id,
                staging_shape,
                staging_layout,
            ) in state_specs_by_die[die_id]:
                state_values_by_die[die_id].append(
                    StateStagingValue.create(
                        state_access_ref=access.id,
                        state_ref=access.state_ref,
                        shape=staging_shape,
                        dtype=declaration.dtype,
                        logical_layout=staging_layout,
                        producer_tasks=tuple(
                            task.id
                            for task in sorted(
                                (
                                    task
                                    for task in task_tuple
                                    if staging_id in task.write_values
                                ),
                                key=lambda item: (
                                    item.tensor_slice.offset
                                    if item.tensor_slice is not None else (),
                                    item.tensor_slice.shape
                                    if item.tensor_slice is not None else (),
                                    item.id,
                                ),
                            )
                        ),
                        consumer_tasks=tuple(
                            task.id
                            for task in task_tuple
                            if staging_id in task.read_values
                        ),
                    )
                )

        value_index = {value.id: value for value in ir1.values}
        temp_origins: dict[str, str] = {}

        def bind_temp(temp_id: str, origin_id: str) -> None:
            if temp_id in value_index:
                if temp_id != origin_id:
                    _fail("planned temp conflicts with IR-1 value", "fusion_plans")
                return
            previous = temp_origins.get(temp_id)
            if previous is not None and previous != origin_id:
                _fail("planned temp has conflicting origins", "fusion_plans")
            temp_origins[temp_id] = origin_id

        for plan in fusion_plans:
            skeleton = skeleton_index[plan.fused_op_id]
            partial_id = node_index[skeleton.member_node_ids[0]].outputs[0]
            for program in plan.rank_programs:
                for action in program.actions:
                    if action.compute is not None and action.compute.tile is not None:
                        for binding in (
                            action.compute.tile.input_slices
                            + action.compute.tile.output_slices
                        ):
                            bind_temp(binding.operand_id, binding.source_value_id)
                    if action.kind is FusionActionKind.RECV:
                        for temp_id in action.writes:
                            bind_temp(temp_id, partial_id)

        dags: list[IntraDieDAG] = []
        for die in ir1.fabric.dies:
            tasks = tuple(tasks_by_die[die.id])
            state_value_ids = {
                value.id for value in state_values_by_die[die.id]
            }
            referenced_value_ids = tuple(
                dict.fromkeys(
                    value_id
                    for task in tasks
                    for value_id in task.read_values + task.write_values
                    if value_id not in state_value_ids
                )
            )
            values: list[IntraDieValue] = []
            for value_id in referenced_value_ids:
                origin_id = value_id if value_id in value_index else temp_origins.get(value_id)
                if origin_id is None:
                    _fail("task references an unknown planned temp", "tasks")
                origin = value_index[origin_id]
                producers = tuple(
                    task.id
                    for task in sorted(
                        (task for task in tasks if value_id in task.write_values),
                        key=lambda item: (
                            item.tensor_slice.offset if item.tensor_slice else (),
                            item.tensor_slice.shape if item.tensor_slice else (),
                            item.id,
                        ),
                    )
                )
                consumers = tuple(
                    task.id for task in tasks if value_id in task.read_values
                )
                values.append(
                    IntraDieValue(
                        id=value_id,
                        origin_value_id=origin_id,
                        shape=origin.shape,
                        dtype=origin.dtype,
                        logical_layout=origin.logical_layout,
                        sharding=origin.sharding,
                        alias_set=origin.alias_set,
                        producer_tasks=producers,
                        consumer_tasks=consumers,
                    )
                )
            dags.append(
                IntraDieDAG.create(
                    producer_pass="project_to_ir2",
                    source_ir1_id=ir1.id,
                    die_id=die.id,
                    fusion_plan_ids=tuple(fusion_ids_by_die[die.id]),
                    standalone_collective_plan_ids=tuple(
                        standalone_ids_by_die[die.id]
                    ),
                    ordinary_node_ids=tuple(
                        dict.fromkeys(ordinary_ids_by_die[die.id])
                    ),
                    tasks=tasks,
                    values=tuple(values),
                    flows=tuple(flows_by_die[die.id]),
                    regions=tuple(regions_by_die[die.id]),
                    source_state_manifest_id=(
                        manifest.id if manifest is not None else None
                    ),
                    state_access_ids=tuple(state_access_ids_by_die[die.id]),
                    state_staging_values=tuple(
                        sorted(
                            state_values_by_die[die.id],
                            key=lambda value: (value.state_access_ref, value.id),
                        )
                    ),
                    state_transfer_ids=tuple(
                        state_transfer_ids_by_die[die.id]
                    ),
                )
            )

        result = IR2ProjectionResult.create(
            producer_pass="project_to_ir2",
            source_ir1_id=ir1.id,
            fusion_plan_ids=tuple(plan.id for plan in fusion_plans),
            standalone_collective_plan_ids=tuple(
                plan.id for plan in standalone_plans
            ),
            dags=tuple(dags),
            source_state_manifest_id=(manifest.id if manifest is not None else None),
            state_transfers=state_transfers,
        )
        result.validate_against(ir1, fusion_plans, standalone_plans)
        return result


__all__ = ["NaiveProjectToIR2"]
