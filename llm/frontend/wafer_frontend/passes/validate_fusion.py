"""Fail-closed semantic validation for Dense GEMM/ReduceScatter candidates."""

from __future__ import annotations

from collections import deque

from ..errors import SchemaError
from ..schema.ir0 import (
    CollectiveKind,
    CollectiveWorkload,
    EdgeKind,
    EffectKind,
    FusionImpl,
    FusionOrigin,
    GemmPartition,
    GemmWorkload,
    IR0,
    OpKind,
    ReduceOp,
)
from .validate_ir0 import DenseIR0Validator


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _is_row_gemm(node: object) -> bool:
    return (
        getattr(node, "kind", None) is OpKind.GEMM
        and isinstance(getattr(node, "workload", None), GemmWorkload)
        and node.workload.partition is GemmPartition.ROW_PARALLEL
    )


def _is_sum_rs(node: object) -> bool:
    return (
        getattr(node, "kind", None) is OpKind.COLLECTIVE
        and isinstance(getattr(node, "workload", None), CollectiveWorkload)
        and node.workload.collective is CollectiveKind.REDUCE_SCATTER
        and node.workload.reduce_op is ReduceOp.SUM
    )


def _validate_convexity(
    graph: IR0,
    members: set[str],
    *,
    path: str,
) -> None:
    adjacency: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    reverse: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.source_node].append(edge.destination_node)
        reverse[edge.destination_node].append(edge.source_node)

    reachable_from_members: set[str] = set()
    ready = deque(members)
    while ready:
        current = ready.popleft()
        for destination in adjacency[current]:
            if destination not in reachable_from_members:
                reachable_from_members.add(destination)
                ready.append(destination)

    can_reach_members: set[str] = set()
    ready = deque(members)
    while ready:
        current = ready.popleft()
        for source in reverse[current]:
            if source not in can_reach_members:
                can_reach_members.add(source)
                ready.append(source)

    if (reachable_from_members & can_reach_members) - members:
        _fail(
            "candidate members do not form a convex dependency subgraph",
            f"{path}.members",
        )


class FusionSemanticValidator:
    """Validate exact Dense row-GEMM plus SUM ReduceScatter candidates."""

    @staticmethod
    def validate(graph: IR0, path: str = "ir0") -> None:
        DenseIR0Validator.validate(graph, path)
        from .discover_fusion import discover_fusion_candidates

        expected = discover_fusion_candidates(graph)
        seen_member_sets: set[tuple[str, ...]] = set()
        for candidate in graph.fusion_candidates:
            if candidate.members in seen_member_sets:
                _fail(
                    "fusion candidates must not share member nodes",
                    f"{path}.fusion_candidates",
                )
            seen_member_sets.add(candidate.members)
        if len(graph.fusion_candidates) != len(expected):
            _fail(
                "every direct supported fusion pair must have exactly one candidate",
                f"{path}.fusion_candidates",
            )
        nodes = {node.id: node for node in graph.nodes}
        values = {value.id: value for value in graph.values}
        claimed_members: set[str] = set()
        for index, (candidate, canonical) in enumerate(
            zip(graph.fusion_candidates, expected, strict=True)
        ):
            candidate_path = f"{path}.fusion_candidates[{index}]"
            if candidate.impl is not FusionImpl.NONE:
                _fail(
                    "candidate implementation must remain NONE before partitioning",
                    f"{candidate_path}.impl",
                )
            if candidate.origin is not FusionOrigin.DISCOVERED:
                _fail(
                    "generated Dense candidates must have DISCOVERED origin",
                    f"{candidate_path}.origin",
                )
            if candidate.members != canonical.members:
                if len(candidate.members) == 2:
                    source = nodes[candidate.members[0]]
                    destination = nodes[candidate.members[1]]
                    if (
                        len(source.outputs) != 1
                        or source.outputs[0] not in destination.inputs
                    ):
                        _fail(
                            "fusion source sole partial output must be a destination input",
                            f"{candidate_path}.members",
                        )
                _fail(
                    "candidate members must preserve canonical ordered fusion pairs",
                    f"{candidate_path}.members",
                )
            overlap = claimed_members.intersection(candidate.members)
            if overlap:
                _fail(
                    "fusion candidates must not share member nodes",
                    f"{candidate_path}.members",
                )
            claimed_members.update(candidate.members)
            if candidate.boundary_inputs != canonical.boundary_inputs:
                _fail(
                    "candidate boundary input order is not canonical",
                    f"{candidate_path}.boundary_inputs",
                )
            if candidate.boundary_outputs != canonical.boundary_outputs:
                _fail(
                    "candidate boundary_outputs are not canonical",
                    f"{candidate_path}.boundary_outputs",
                )
            actual_contract = candidate.semantic_contract
            expected_contract = canonical.semantic_contract
            for field_name in (
                "pattern",
                "tile_domain",
                "reduction_axes",
                "input_layouts",
                "output_layout",
                "numerical_policy",
            ):
                if getattr(actual_contract, field_name) != getattr(
                    expected_contract, field_name
                ):
                    label = (
                        "numerical policy"
                        if field_name == "numerical_policy"
                        else field_name
                    )
                    _fail(
                        f"candidate {label} is not canonical",
                        f"{candidate_path}.semantic_contract.{field_name}",
                    )
            if candidate.id != canonical.id:
                _fail(
                    "candidate id does not match its canonical semantic content",
                    f"{candidate_path}.id",
                )
            _validate_convexity(
                graph,
                set(candidate.members),
                path=candidate_path,
            )
        return

        nodes = {node.id: node for node in graph.nodes}
        values = {value.id: value for value in graph.values}
        data_edges = tuple(edge for edge in graph.edges if edge.kind is EdgeKind.DATA)
        claimed_members: set[str] = set()
        candidate_pairs: list[tuple[str, str]] = []

        for index, candidate in enumerate(graph.fusion_candidates):
            candidate_path = f"{path}.fusion_candidates[{index}]"
            if candidate.impl is not FusionImpl.NONE:
                _fail(
                    "candidate implementation must remain NONE before partitioning",
                    f"{candidate_path}.impl",
                )
            if candidate.origin is not FusionOrigin.DISCOVERED:
                _fail(
                    "generated Dense candidates must have DISCOVERED origin",
                    f"{candidate_path}.origin",
                )
            if len(candidate.members) != 2:
                _fail(
                    "candidate must contain exactly row-GEMM then ReduceScatter",
                    f"{candidate_path}.members",
                )
            gemm = nodes[candidate.members[0]]
            rs = nodes[candidate.members[1]]
            if not _is_row_gemm(gemm) or not _is_sum_rs(rs):
                _fail(
                    "candidate members must be ordered row-GEMM then SUM ReduceScatter",
                    f"{candidate_path}.members",
                )
            overlap = claimed_members.intersection(candidate.members)
            if overlap:
                _fail(
                    "fusion candidates must not share member nodes",
                    f"{candidate_path}.members",
                )
            claimed_members.update(candidate.members)

            if not (
                gemm.instance_id == rs.instance_id
                and gemm.stage == rs.stage
                and gemm.phase is rs.phase
                and gemm.mesh_ref == rs.mesh_ref
            ):
                _fail(
                    "candidate members must share instance, stage, phase and mesh",
                    f"{candidate_path}.members",
                )
            if (
                len(gemm.outputs) != 1
                or len(rs.inputs) != 1
                or gemm.outputs[0] != rs.inputs[0]
            ):
                _fail(
                    "row-GEMM sole partial output must be the ReduceScatter sole input",
                    f"{candidate_path}.members",
                )
            partial = values[gemm.outputs[0]]
            if partial.consumers != (rs.id,):
                _fail(
                    "candidate partial value cannot have an external consumer",
                    f"{candidate_path}.members",
                )
            internal_edges = tuple(
                edge
                for edge in data_edges
                if edge.source_node in candidate.members
                and edge.destination_node in candidate.members
            )
            expected_edge = (gemm.id, rs.id, partial.id)
            if (
                len(internal_edges) != 1
                or (
                    internal_edges[0].source_node,
                    internal_edges[0].destination_node,
                    internal_edges[0].value_id,
                )
                != expected_edge
            ):
                _fail(
                    "candidate must contain exactly one direct GEMM-to-RS DATA edge",
                    f"{candidate_path}.members",
                )
            if candidate.boundary_inputs != gemm.inputs:
                _fail(
                    "candidate boundary_inputs must exactly preserve GEMM input order",
                    f"{candidate_path}.boundary_inputs",
                )
            if candidate.boundary_outputs != rs.outputs:
                _fail(
                    "candidate boundary_outputs must exactly equal RS outputs",
                    f"{candidate_path}.boundary_outputs",
                )

            contract = candidate.semantic_contract
            if contract.tile_domain != ("M", "N"):
                _fail(
                    "candidate tile_domain must be exactly ('M', 'N')",
                    f"{candidate_path}.semantic_contract.tile_domain",
                )
            if contract.reduction_axes != (2,):
                _fail(
                    "candidate reduction_axes must be exactly GEMM K axis (2,)",
                    f"{candidate_path}.semantic_contract.reduction_axes",
                )
            expected_input_layouts = tuple(
                values[value_id].logical_layout for value_id in gemm.inputs
            )
            if contract.input_layouts != expected_input_layouts:
                _fail(
                    "candidate input_layouts must exactly match ordered boundary inputs",
                    f"{candidate_path}.semantic_contract.input_layouts",
                )
            expected_output_layout = values[rs.outputs[0]].logical_layout
            if contract.output_layout != expected_output_layout:
                _fail(
                    "candidate output_layout must exactly match the RS output",
                    f"{candidate_path}.semantic_contract.output_layout",
                )
            if (
                gemm.math.numerical_policy is not rs.math.numerical_policy
                or contract.numerical_policy is not gemm.math.numerical_policy
            ):
                _fail(
                    "candidate numerical policy must exactly match both members",
                    f"{candidate_path}.semantic_contract.numerical_policy",
                )
            if any(
                node.effects.kind is not EffectKind.PURE
                or node.effects.effect_token is not None
                or node.effects.alias_set is not None
                for node in (gemm, rs)
            ):
                _fail(
                    "candidate members must be effect-free PURE nodes",
                    f"{candidate_path}.members",
                )
            candidate_values = (
                partial,
                *(values[value_id] for value_id in candidate.boundary_inputs),
                *(values[value_id] for value_id in candidate.boundary_outputs),
            )
            if any(value.alias_set is not None for value in candidate_values):
                _fail(
                    "candidate internal and boundary values must be alias-safe",
                    f"{candidate_path}.members",
                )

            _validate_convexity(
                graph,
                set(candidate.members),
                path=candidate_path,
            )
            candidate_pairs.append((gemm.id, rs.id))

        direct_pairs = []
        for gemm in graph.nodes:
            if not _is_row_gemm(gemm) or len(gemm.outputs) != 1:
                continue
            output = values[gemm.outputs[0]]
            if len(output.consumers) != 1:
                continue
            rs = nodes[output.consumers[0]]
            if _is_sum_rs(rs) and rs.inputs == (output.id,):
                direct_pairs.append((gemm.id, rs.id))

        if sorted(candidate_pairs) != sorted(direct_pairs):
            _fail(
                "every direct row-GEMM-to-RS pair must have exactly one candidate",
                f"{path}.fusion_candidates",
            )


def validate(graph: IR0, path: str = "ir0") -> None:
    """Validate ``graph`` using the public functional pass API."""

    FusionSemanticValidator.validate(graph, path)


__all__ = ["FusionSemanticValidator", "validate"]
