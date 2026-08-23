"""Deterministic Dense GEMM/ReduceScatter fusion partition policy."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import stable_artifact_id
from ..schema.ir0 import (
    CollectiveKind,
    CollectiveWorkload,
    EdgeKind,
    FusionCandidate,
    FusionImpl,
    FusionPattern,
    GemmPartition,
    GemmWorkload,
    OpKind,
    ReduceOp,
)
from ..schema.ir1 import FusedOpSkeleton, IR1
from ..schema.n4 import FUSED_OP_SKELETON_SCHEMA_VERSION


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _validate_candidate(
    ir1: IR1,
    candidate: FusionCandidate,
    *,
    path: str,
) -> tuple[object, object]:
    nodes = {node.id: node for node in ir1.nodes}
    values = {value.id: value for value in ir1.values}
    if len(candidate.members) != 2:
        _fail(
            "candidate must contain exactly row-GEMM then SUM ReduceScatter",
            f"{path}.members",
        )

    gemm = nodes[candidate.members[0]]
    reduce_scatter = nodes[candidate.members[1]]
    if (
        gemm.kind is OpKind.GEMM
        and type(gemm.workload) is GemmWorkload
        and gemm.workload.partition
        is GemmPartition.SEQUENCE_PARALLEL_REPLICATED_WEIGHT
    ):
        _fail(
            "sequence-parallel replicated-weight GEMM cannot enter a fusion candidate",
            f"{path}.members[0]",
        )
    if (
        gemm.kind is not OpKind.GEMM
        or type(gemm.workload) is not GemmWorkload
        or gemm.workload.partition is not GemmPartition.ROW_PARALLEL
    ):
        _fail("first member must be a row-parallel GEMM", f"{path}.members[0]")
    if (
        reduce_scatter.kind is not OpKind.COLLECTIVE
        or type(reduce_scatter.workload) is not CollectiveWorkload
        or reduce_scatter.workload.collective is not CollectiveKind.REDUCE_SCATTER
        or reduce_scatter.workload.reduce_op is not ReduceOp.SUM
    ):
        _fail("second member must be SUM ReduceScatter", f"{path}.members[1]")

    if (
        gemm.instance_id != reduce_scatter.instance_id
        or gemm.stage != reduce_scatter.stage
        or gemm.phase is not reduce_scatter.phase
        or gemm.execution_group_ref != reduce_scatter.execution_group_ref
    ):
        _fail(
            "members must share instance, stage, phase, and execution group",
            f"{path}.members",
        )

    internal_data_edges = tuple(
        edge
        for edge in ir1.edges
        if edge.kind is EdgeKind.DATA
        and edge.source_node in candidate.members
        and edge.destination_node in candidate.members
    )
    expected_edge = (
        gemm.id,
        reduce_scatter.id,
        gemm.outputs[0] if len(gemm.outputs) == 1 else None,
    )
    if (
        len(internal_data_edges) != 1
        or (
            internal_data_edges[0].source_node,
            internal_data_edges[0].destination_node,
            internal_data_edges[0].value_id,
        )
        != expected_edge
    ):
        _fail(
            "members must have exactly one direct GEMM-to-ReduceScatter DATA edge",
            f"{path}.members",
        )
    if (
        len(gemm.outputs) != 1
        or len(reduce_scatter.inputs) != 1
        or len(reduce_scatter.outputs) != 1
        or gemm.outputs[0] != reduce_scatter.inputs[0]
    ):
        _fail(
            "GEMM sole output must be the ReduceScatter sole input",
            f"{path}.members",
        )
    partial = values[gemm.outputs[0]]
    if partial.consumers != (reduce_scatter.id,):
        _fail(
            "GEMM partial output must have only the ReduceScatter consumer",
            f"{path}.members",
        )

    if candidate.boundary_inputs != gemm.inputs:
        _fail(
            "boundary_inputs must exactly preserve GEMM input order",
            f"{path}.boundary_inputs",
        )
    if candidate.boundary_outputs != reduce_scatter.outputs:
        _fail(
            "boundary_outputs must exactly preserve ReduceScatter output order",
            f"{path}.boundary_outputs",
        )

    contract = candidate.semantic_contract
    if contract.pattern is not FusionPattern.GEMM_RS:
        _fail(
            "naive fusion only accepts GEMM_RS candidates",
            f"{path}.semantic_contract.pattern",
        )
    if contract.tile_domain != ("M", "N"):
        _fail("tile_domain must be exactly ('M', 'N')", f"{path}.semantic_contract.tile_domain")
    if contract.reduction_axes != (2,):
        _fail(
            "reduction_axes must be exactly GEMM K axis (2,)",
            f"{path}.semantic_contract.reduction_axes",
        )
    expected_input_layouts = tuple(values[value_id].logical_layout for value_id in gemm.inputs)
    if contract.input_layouts != expected_input_layouts:
        _fail(
            "input_layouts must exactly match ordered boundary inputs",
            f"{path}.semantic_contract.input_layouts",
        )
    expected_output_layout = values[reduce_scatter.outputs[0]].logical_layout
    if contract.output_layout != expected_output_layout:
        _fail(
            "output_layout must exactly match the ReduceScatter output",
            f"{path}.semantic_contract.output_layout",
        )
    if (
        gemm.math.numerical_policy is not reduce_scatter.math.numerical_policy
        or contract.numerical_policy is not gemm.math.numerical_policy
    ):
        _fail(
            "numerical policy must exactly match both members",
            f"{path}.semantic_contract.numerical_policy",
        )
    return gemm, reduce_scatter


def _skeleton_id(candidate: FusionCandidate) -> str:
    impl = FusionImpl.NONE
    semantic_key = {
        "fusion_ref": candidate.id,
        "member_node_ids": candidate.members,
        "boundary_inputs": candidate.boundary_inputs,
        "boundary_outputs": candidate.boundary_outputs,
        "semantic_contract": candidate.semantic_contract,
        "impl": impl,
    }
    return stable_artifact_id(
        "fused_op_skeleton",
        semantic_key,
        schema_version=FUSED_OP_SKELETON_SCHEMA_VERSION,
    )


class NaiveFusionPartition:
    """Select every legal Dense row-GEMM/SUM-ReduceScatter candidate."""

    def run(self, ir1: IR1) -> tuple[FusedOpSkeleton, ...]:
        if type(ir1) is not IR1:
            raise SchemaError("must be an IR1", path="ir1")
        ir1.validate("ir1")
        if ir1.fused_op_skeletons:
            raise SchemaError(
                "must be empty before fusion partition",
                path="ir1.fused_op_skeletons",
            )
        claimed_members: set[str] = set()
        result: list[FusedOpSkeleton] = []
        for index, candidate in enumerate(ir1.fusion_candidates):
            candidate_path = f"ir1.fusion_candidates[{index}]"
            overlap = claimed_members.intersection(candidate.members)
            if candidate.semantic_contract.pattern is not FusionPattern.GEMM_RS:
                continue
            if overlap:
                _fail(
                    "fusion candidates must not share member nodes",
                    f"{candidate_path}.members",
                )
            gemm, _ = _validate_candidate(ir1, candidate, path=candidate_path)
            claimed_members.update(candidate.members)
            result.append(
                FusedOpSkeleton(
                    id=_skeleton_id(candidate),
                    fusion_ref=candidate.id,
                    instance_id=gemm.instance_id,
                    member_node_ids=candidate.members,
                    boundary_inputs=candidate.boundary_inputs,
                    boundary_outputs=candidate.boundary_outputs,
                    semantic_contract=candidate.semantic_contract,
                    impl=FusionImpl.NONE,
                )
            )
        return tuple(result)


__all__ = ["NaiveFusionPartition"]
