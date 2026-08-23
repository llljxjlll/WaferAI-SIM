"""Fail-closed production bridge for 2x2 MeshSlice output-stationary plans."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..errors import SchemaError
from ..passes.project_swizzle_ir2 import project_swizzle_adapter
from ..policies.swizzle.cost import build_unfused_baseline
from ..policies.swizzle.decide import decide_swizzle
from ..policies.swizzle.enumerate import materialize_drafts
from ..policies.swizzle.materialize import materialize_swizzle_selection
from ..policies.swizzle.materialize_ir1 import materialize_swizzle_plan
from ..policies.swizzle.meshslice_2d import generate_meshslice_2d_drafts
from ..policies.swizzle.problem import build_swizzle_problem
from ..schema.artifact_manifest import RecordOpcode
from ..schema.common import MeshAxisName, validate_uint64
from ..schema.ir0 import FusionPattern
from ..schema.ir1 import FusedOpSkeleton, IR1
from ..schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleConstraints,
    SwizzleDecision,
    SwizzleHardwareProfile,
    SwizzleOperand,
    SwizzleSemanticWitness,
    SwizzleTensorAxis,
    SwizzleTensorAxisRole,
    SwizzleTopologyKind,
    SwizzleUpdateKind,
)
from ..schema.swizzle_plan import (
    SwizzleDeploymentReason,
    SwizzleDeploymentSelection,
)
from ..schema.swizzle_operand_abi import build_swizzle_operand_abi
from ..schema.swizzle_standard import SwizzleStandardLinkedProgram
from .swizzle import lower_swizzle_projection
from .swizzle_abi import allocate_swizzle_core_address_abi
from .swizzle_standard import link_swizzle_standard_program


@dataclass(frozen=True, slots=True)
class MeshSlice2DStandardAudit:
    """Observable compression boundary of one standard MeshSlice manifest."""

    ranks: int
    chunks: int
    row_flows: int
    column_flows: int
    root_buffers: int
    alloc_records: int
    free_records: int
    barrier_groups: int
    barrier_actions: int
    event_records: int

    def validate(self, path: str = "meshslice_standard_audit") -> None:
        for name in (
            "ranks",
            "chunks",
            "row_flows",
            "column_flows",
            "root_buffers",
            "alloc_records",
            "free_records",
            "barrier_groups",
            "barrier_actions",
            "event_records",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.ranks != 4 or self.chunks < 2:
            raise SchemaError(
                "production MeshSlice audit requires four ranks and multiple chunks",
                path=path,
            )
        expected_direction_flows = self.ranks * self.chunks
        if (
            self.row_flows != expected_direction_flows
            or self.column_flows != expected_direction_flows
        ):
            raise SchemaError(
                "every rank/chunk must expose one row and one column flow",
                path=path,
            )
        if (
            self.root_buffers == 0
            or self.alloc_records != self.root_buffers
            or self.free_records != self.root_buffers
        ):
            raise SchemaError(
                "lifecycle compression must emit one ALLOC/FREE per root buffer",
                path=path,
            )
        if self.barrier_actions == 0:
            if self.barrier_groups != 0 or self.event_records != 0:
                raise SchemaError(
                    "barrier-free plans cannot synthesize event records",
                    path=path,
                )
        elif (
            self.barrier_actions != self.ranks * self.barrier_groups
            or self.event_records !=
                4 * (self.ranks - 1) * self.barrier_groups
        ):
            raise SchemaError(
                "group barriers must remain at the exact leader/peer event boundary",
                path=path,
            )


def decide_meshslice_2d_standard(
    ir1: IR1,
    fused_op: FusedOpSkeleton,
    hardware_profile: SwizzleHardwareProfile,
    constraints: SwizzleConstraints,
) -> SwizzleDecision:
    """Plan only the existing exact-sharding MeshSlice candidate family.

    The generic V1 semantic analyzer models an AllGather as a single-axis
    sharded-to-replicated transition.  MeshSlice instead requires the already
    materialized GEMM views to prove a closed two-axis OS layout.  This narrow
    adapter records that distinct proof without enabling the generator's
    boundary-reshard escape hatch.
    """

    problem = build_swizzle_problem(
        ir1,
        fused_op,
        hardware_profile,
        constraints,
    )
    if problem.pattern is not FusionPattern.AG_GEMM:
        raise SchemaError(
            "MeshSlice standard V1 supports only AG+GEMM",
            path="swizzle_problem.pattern",
        )
    gathered = problem.collective.output
    if gathered == problem.gemm.lhs:
        operand = SwizzleOperand.LHS
    elif gathered == problem.gemm.rhs:
        operand = SwizzleOperand.RHS
    else:
        raise SchemaError(
            "AllGather output must exactly equal one GEMM operand",
            path="swizzle_problem.collective.output",
        )
    axis = problem.collective.gather_tensor_axis
    if axis is None or axis >= len(gathered.shape):
        raise SchemaError(
            "AllGather gather axis is outside the GEMM operand",
            path="swizzle_problem.collective.gather_tensor_axis",
        )
    if (
        gathered.axis_roles[axis] is not SwizzleTensorAxisRole.CONTRACT
        or problem.collective.input.sharding_dim_map !=
            gathered.sharding_dim_map
        or problem.collective.input.partial_mesh_axes
        or gathered.partial_mesh_axes
    ):
        raise SchemaError(
            "MeshSlice AG boundary must preserve a closed sharded K view",
            path="swizzle_problem.collective",
        )
    witness = SwizzleSemanticWitness(
        pattern=problem.pattern,
        member_refs=fused_op.member_node_ids,
        boundary_input_refs=fused_op.boundary_inputs,
        boundary_output_refs=fused_op.boundary_outputs,
        intermediate_value_ref=gathered.value_ref,
        gemm_operand=operand,
        split_axis=SwizzleTensorAxis(
            tensor_ref=gathered.value_ref,
            index=axis,
            name=f"{gathered.axis_roles[axis].value}_{axis}",
            extent=gathered.shape[axis],
            role=gathered.axis_roles[axis],
        ),
        update_kind=SwizzleUpdateKind.PARTIAL_ACCUMULATION,
        gather_axis=axis,
        reduction_axis=None,
        has_reduction_phase=False,
        has_replication_phase=False,
        input_layout_closed=True,
        output_layout_closed=True,
        sharding_transition_closed=True,
    )
    witness.validate("meshslice_semantic_witness")
    drafts = generate_meshslice_2d_drafts(problem, witness)
    candidates = materialize_drafts(problem, drafts)
    baseline = build_unfused_baseline(problem, witness)
    decision = decide_swizzle(problem, baseline, candidates)
    decision.validate("meshslice_decision")
    return decision


def _selected_meshslice(
    decision: SwizzleDecision,
    candidate_ref: str,
) -> SwizzleCandidate:
    decision.validate("decision")
    candidate = next(
        (
            item
            for item in decision.ranked_candidates
            if item.id == candidate_ref
        ),
        None,
    )
    if candidate is None:
        raise SchemaError(
            "candidate is absent from the economic ranking",
            path="candidate_ref",
        )
    if candidate.algorithm is not SwizzleAlgorithm.MESHSLICE_2D_OS:
        raise SchemaError(
            "candidate must use MeshSlice 2D output-stationary",
            path="candidate.algorithm",
        )
    return candidate


def _validate_2d_contract(
    ir1: IR1,
    decision: SwizzleDecision,
    candidate: SwizzleCandidate,
) -> None:
    topology = candidate.topology_witness
    if (
        topology.kind is not SwizzleTopologyKind.RECTANGLE_2D
        or len(topology.row_orders) != 2
        or any(len(row) != 2 for row in topology.row_orders)
        or len(topology.column_orders) != 2
        or any(len(column) != 2 for column in topology.column_orders)
        or set(topology.rank_order) != set(range(4))
        or candidate.chunk_count < 2
    ):
        raise SchemaError(
            "production MeshSlice requires an exact 2x2 rectangle and multiple K slices",
            path="candidate.topology_witness",
        )
    group = next(
        (
            item
            for item in ir1.groups
            if item.id == decision.problem.group.group_ref
        ),
        None,
    )
    if group is None or group.logical_shape != (2, 2):
        raise SchemaError(
            "production MeshSlice requires a typed 2x2 physical group",
            path="ir1.groups",
        )
    dies = {item.id: item for item in ir1.fabric.dies}
    physical = {
        dies[item.die_id].coord: item.rank for item in group.placements
    }
    xs = tuple(sorted({coord[0] for coord in physical}))
    ys = tuple(sorted({coord[1] for coord in physical}))
    rows = tuple(tuple(physical[(x, y)] for x in xs) for y in ys)
    columns = tuple(tuple(physical[(x, y)] for y in ys) for x in xs)
    if (
        len(physical) != 4
        or len(xs) != 2
        or len(ys) != 2
        or topology.row_orders != rows
        or topology.column_orders != columns
    ):
        raise SchemaError(
            "candidate row/column lines must equal real placement coordinates",
            path="candidate.topology_witness",
        )
    lhs = decision.problem.gemm.lhs
    rhs = decision.problem.gemm.rhs
    output = decision.problem.gemm.output
    row_axis, column_axis = output.sharding_dim_map[-2:]
    if (
        row_axis is None
        or column_axis is None
        or row_axis is column_axis
        or {row_axis, column_axis} !=
            {MeshAxisName.DP, MeshAxisName.TP}
        or lhs.sharding_dim_map[-2:] != (row_axis, column_axis)
        or rhs.sharding_dim_map[-2:] != (row_axis, column_axis)
        or lhs.partial_mesh_axes
        or rhs.partial_mesh_axes
        or output.partial_mesh_axes
    ):
        raise SchemaError(
            "production MeshSlice requires explicit closed DPxTP tensor sharding",
            path="decision.problem.gemm",
        )

    routes = {
        (item.source_rank, item.destination_rank): item
        for item in decision.problem.group.routes
    }
    actions_by_rank = {
        program.rank: program.actions for program in candidate.rank_programs
    }
    for rank in range(4):
        row = next(item for item in rows if rank in item)
        column = next(item for item in columns if rank in item)
        row_peer = next(item for item in row if item != rank)
        column_peer = next(item for item in column if item != rank)
        output_ref = f"buffer.meshslice.rank.{rank}.output"
        requirements = {
            item.buffer_ref: item
            for item in candidate.buffer_requirements
            if item.rank == rank
        }
        if (
            not requirements[f"buffer.meshslice.rank.{rank}.lhs"].double_buffered
            or not requirements[f"buffer.meshslice.rank.{rank}.rhs"].double_buffered
            or requirements[output_ref].double_buffered
        ):
            raise SchemaError(
                "MeshSlice must retain two input slots and one stationary output",
                path="candidate.buffer_requirements",
            )
        prior_compute = None
        for chunk in range(candidate.chunk_count):
            chunk_actions = tuple(
                action
                for action in actions_by_rank[rank]
                if action.chunk_index == chunk
            )
            sends = tuple(
                action
                for action in chunk_actions
                if action.kind is SwizzleActionKind.SEND
            )
            receives = tuple(
                action
                for action in chunk_actions
                if action.kind is SwizzleActionKind.RECV
            )
            waits = tuple(
                action
                for action in chunk_actions
                if action.kind is SwizzleActionKind.WAIT
            )
            computes = tuple(
                action
                for action in chunk_actions
                if action.kind is SwizzleActionKind.COMP
            )
            if (
                len(sends) != 2
                or {item.peer_rank for item in sends} !=
                    {row_peer, column_peer}
                or len(receives) != 2
                or {item.peer_rank for item in receives} !=
                    {row_peer, column_peer}
                or len(waits) != 2
                or len(computes) != 1
                or sends[0].deps != sends[1].deps
            ):
                raise SchemaError(
                    "every rank/chunk must expose independent row/column traffic",
                    path="candidate.rank_programs",
                )
            row_resources = set(routes[(rank, row_peer)].resource_ids)
            column_resources = set(routes[(rank, column_peer)].resource_ids)
            if row_resources.intersection(column_resources):
                raise SchemaError(
                    "row and column sends must use disjoint physical resources",
                    path="decision.problem.group.routes",
                )
            compute = computes[0]
            expected_inputs = (
                (f"buffer.meshslice.rank.{rank}.lhs",
                 f"buffer.meshslice.rank.{rank}.rhs")
                if chunk == 0
                else (f"buffer.meshslice.rank.{rank}.lhs",
                      f"buffer.meshslice.rank.{rank}.rhs", output_ref)
            )
            if (
                not {item.id for item in waits}.issubset(compute.deps)
                or compute.input_refs != expected_inputs
                or compute.output_refs != (output_ref,)
                or (
                    prior_compute is not None
                    and prior_compute not in compute.deps
                )
            ):
                raise SchemaError(
                    "compute must first-write then carry the output accumulator",
                    path="candidate.rank_programs",
                )
            prior_compute = compute.id


def audit_meshslice_2d_standard_program(
    source: SwizzleStandardLinkedProgram,
) -> MeshSlice2DStandardAudit:
    source.validate_against()
    candidate = source.plan.candidate
    rows = candidate.topology_witness.row_orders
    columns = candidate.topology_witness.column_orders
    row_pairs = {
        (left, right)
        for row in rows
        for left in row
        for right in row
        if left != right
    }
    column_pairs = {
        (top, bottom)
        for column in columns
        for top in column
        for bottom in column
        if top != bottom
    }
    row_flows = sum(
        (flow.source_rank, flow.destination_rank) in row_pairs
        for flow in source.projection.flows
    )
    column_flows = sum(
        (flow.source_rank, flow.destination_rank) in column_pairs
        for flow in source.projection.flows
    )
    opcodes = Counter(
        record.opcode
        for stream in source.fragment.core_streams
        for record in stream.records
    )
    barriers = tuple(
        action
        for program in source.plan.rank_programs
        for action in program.actions
        if action.source_action.kind is SwizzleActionKind.BARRIER
    )
    barrier_groups = {
        action.sync.barrier.id
        for action in barriers
        if action.sync.barrier is not None
    }
    result = MeshSlice2DStandardAudit(
        ranks=len(source.projection.rank_dags),
        chunks=candidate.chunk_count,
        row_flows=row_flows,
        column_flows=column_flows,
        root_buffers=sum(
            item.alias_of is None for item in source.fragment.buffer_abi
        ),
        alloc_records=opcodes[RecordOpcode.SRAM_ALLOC_AT],
        free_records=opcodes[RecordOpcode.SRAM_FREE],
        barrier_groups=len(barrier_groups),
        barrier_actions=len(barriers),
        event_records=(
            opcodes[RecordOpcode.EVENT_SET] +
            opcodes[RecordOpcode.EVENT_WAIT]
        ),
    )
    result.validate()
    return result


def link_meshslice_2d_standard_program(
    ir1: IR1,
    decision: SwizzleDecision,
    *,
    candidate_ref: str,
) -> tuple[SwizzleStandardLinkedProgram, MeshSlice2DStandardAudit]:
    """Close one ranked 2x2 MeshSlice candidate through the standard manifest."""

    ir1.validate("ir1")
    candidate = _selected_meshslice(decision, candidate_ref)
    _validate_2d_contract(ir1, decision, candidate)
    selection = SwizzleDeploymentSelection.create(
        economic_decision=decision,
        candidate=candidate,
        reason=(
            SwizzleDeploymentReason.ECONOMIC_DECISION
            if candidate.id == decision.selected_candidate_ref
            else SwizzleDeploymentReason.FORCED_BY_POLICY
        ),
    )
    adapter = materialize_swizzle_selection(selection)
    plan = materialize_swizzle_plan(
        ir1,
        decision,
        ir1.profile,
        deployment_selection=selection,
    )
    projection = project_swizzle_adapter(adapter)
    lowered = lower_swizzle_projection(plan, projection)
    core_abi = allocate_swizzle_core_address_abi(
        ir1, plan, projection
    )
    operand_abi = build_swizzle_operand_abi(
        ir1, plan, projection
    )
    source = link_swizzle_standard_program(
        ir1, plan, projection, lowered, core_abi, operand_abi
    )
    return source, audit_meshslice_2d_standard_program(source)


__all__ = [
    "decide_meshslice_2d_standard",
    "MeshSlice2DStandardAudit",
    "audit_meshslice_2d_standard_program",
    "link_meshslice_2d_standard_program",
]
