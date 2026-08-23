"""Deterministic discovery of direct GEMM/collective fusion candidates."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import stable_artifact_id
from ..schema.ir0 import (
    CollectiveKind,
    CollectiveWorkload,
    EdgeKind,
    EffectKind,
    FusionCandidate,
    FusionImpl,
    FusionOrigin,
    FusionPattern,
    FusionSemanticContract,
    GemmPartition,
    GemmWorkload,
    IR0,
    LogicalNode,
    NumericalPolicy,
    OpKind,
    ReduceOp,
)


DISCOVER_FUSION_SCHEMA_VERSION = (
    "wafer_frontend.discover_fusion/v1alpha1"
)


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _pure(node: LogicalNode) -> bool:
    return (
        node.effects.kind is EffectKind.PURE
        and node.effects.effect_token is None
        and node.effects.alias_set is None
    )


def _pattern(source: LogicalNode, destination: LogicalNode) -> FusionPattern | None:
    if (
        source.kind is OpKind.COLLECTIVE
        and type(source.workload) is CollectiveWorkload
        and source.workload.collective is CollectiveKind.ALL_GATHER
        and source.workload.reduce_op is None
        and destination.kind is OpKind.GEMM
        and type(destination.workload) is GemmWorkload
        and destination.workload.partition is GemmPartition.COLUMN_PARALLEL
    ):
        return FusionPattern.AG_GEMM
    if (
        source.kind is OpKind.GEMM
        and type(source.workload) is GemmWorkload
        and source.workload.partition is GemmPartition.ROW_PARALLEL
        and destination.kind is OpKind.COLLECTIVE
        and type(destination.workload) is CollectiveWorkload
        and destination.workload.reduce_op is ReduceOp.SUM
    ):
        if destination.workload.collective is CollectiveKind.REDUCE_SCATTER:
            return FusionPattern.GEMM_RS
        if destination.workload.collective is CollectiveKind.ALL_REDUCE:
            return FusionPattern.GEMM_AR
    return None


def _boundary_inputs(
    graph: IR0,
    source: LogicalNode,
    destination: LogicalNode,
) -> tuple[str, ...]:
    members = {source.id, destination.id}
    values = {value.id: value for value in graph.values}
    ordered: list[str] = []
    for node in (source, destination):
        for value_ref in node.inputs:
            value = values[value_ref]
            if value.producer in members or value_ref in ordered:
                continue
            ordered.append(value_ref)
    return tuple(ordered)


def _boundary_outputs(
    graph: IR0,
    source: LogicalNode,
    destination: LogicalNode,
) -> tuple[str, ...]:
    members = {source.id, destination.id}
    return tuple(
        value.id
        for value in graph.values
        if value.producer in members
        and (
            not value.consumers
            or any(consumer not in members for consumer in value.consumers)
        )
    )


def _candidate(
    graph: IR0,
    source: LogicalNode,
    destination: LogicalNode,
    value_id: str,
    pattern: FusionPattern,
) -> FusionCandidate:
    values = {value.id: value for value in graph.values}
    shared = values[value_id]
    if source.outputs != (value_id,) or destination.inputs.count(value_id) != 1:
        _fail(
            "fusion members require one direct sole-output operand",
            "fusion_discovery.members",
        )
    if shared.producer != source.id or shared.consumers != (destination.id,):
        _fail(
            "fusion intermediate must have exactly one destination consumer",
            "fusion_discovery.intermediate",
        )
    if (
        source.instance_id != destination.instance_id
        or source.stage != destination.stage
        or source.phase is not destination.phase
        or source.mesh_ref != destination.mesh_ref
    ):
        _fail(
            "fusion members must share instance, stage, phase, and mesh",
            "fusion_discovery.members",
        )
    if not _pure(source) or not _pure(destination):
        _fail("fusion members must be PURE", "fusion_discovery.members")
    if source.math.numerical_policy is not destination.math.numerical_policy:
        _fail(
            "fusion members must share one numerical policy",
            "fusion_discovery.members",
        )
    boundary_inputs = _boundary_inputs(graph, source, destination)
    boundary_outputs = _boundary_outputs(graph, source, destination)
    if len(boundary_outputs) != 1:
        _fail(
            "V1 fusion requires exactly one boundary output",
            "fusion_discovery.boundary_outputs",
        )
    candidate_values = (
        shared,
        *(values[value_ref] for value_ref in boundary_inputs),
        *(values[value_ref] for value_ref in boundary_outputs),
    )
    if any(value.alias_set is not None for value in candidate_values):
        _fail(
            "fusion intermediate and boundaries must be alias-safe",
            "fusion_discovery.members",
        )
    semantic_contract = FusionSemanticContract(
        pattern=pattern,
        tile_domain=("M", "N"),
        reduction_axes=()
        if pattern is FusionPattern.AG_GEMM
        else (2,),
        input_layouts=tuple(values[ref].logical_layout for ref in boundary_inputs),
        output_layout=values[boundary_outputs[0]].logical_layout,
        numerical_policy=source.math.numerical_policy,
    )
    semantic_key = {
        "members": (source.id, destination.id),
        "boundary_inputs": boundary_inputs,
        "boundary_outputs": boundary_outputs,
        "semantic_contract": semantic_contract,
        "origin": FusionOrigin.DISCOVERED,
    }
    return FusionCandidate(
        id=stable_artifact_id(
            "fusion_candidate",
            semantic_key,
            schema_version=DISCOVER_FUSION_SCHEMA_VERSION,
        ),
        members=(source.id, destination.id),
        boundary_inputs=boundary_inputs,
        boundary_outputs=boundary_outputs,
        semantic_contract=semantic_contract,
        impl=FusionImpl.NONE,
        origin=FusionOrigin.DISCOVERED,
    )


def discover_fusion_candidates(graph: IR0) -> tuple[FusionCandidate, ...]:
    """Discover every supported direct pair in canonical DATA-edge order."""

    if type(graph) is not IR0:
        raise SchemaError("must be an IR0", path="graph")
    graph.validate("graph")
    nodes = {node.id: node for node in graph.nodes}
    result: list[FusionCandidate] = []
    claimed: set[str] = set()
    for edge in graph.edges:
        if edge.kind is not EdgeKind.DATA or edge.value_id is None:
            continue
        source = nodes[edge.source_node]
        destination = nodes[edge.destination_node]
        pattern = _pattern(source, destination)
        if pattern is None:
            continue
        overlap = claimed.intersection((source.id, destination.id))
        if overlap:
            _fail(
                "V1 does not select overlapping fusion candidates",
                "fusion_discovery.members",
            )
        candidate = _candidate(
            graph,
            source,
            destination,
            edge.value_id,
            pattern,
        )
        result.append(candidate)
        claimed.update(candidate.members)
    return tuple(result)


def with_discovered_fusion_candidates(graph: IR0) -> IR0:
    """Return an exact IR0 copy carrying canonical discovered candidates."""

    if type(graph) is not IR0:
        raise SchemaError("must be an IR0", path="graph")
    graph.validate("graph")
    if graph.fusion_candidates:
        _fail(
            "discovery requires a graph with no pre-existing fusion candidates",
            "graph.fusion_candidates",
        )
    result = IR0.create(
        producer_pass=graph.producer_pass,
        job=graph.job,
        instances=graph.instances,
        nodes=graph.nodes,
        values=graph.values,
        edges=graph.edges,
        fusion_candidates=discover_fusion_candidates(graph),
        profile=graph.profile,
        train=graph.train,
        instance_profiles=graph.instance_profiles,
        node_profiles=graph.node_profiles,
        pd_plan_id=graph.pd_plan_id,
        persistent_states=graph.persistent_states,
        state_accesses=graph.state_accesses,
    )
    result.validate("discovered_graph")
    return result


__all__ = [
    "DISCOVER_FUSION_SCHEMA_VERSION",
    "discover_fusion_candidates",
    "with_discovered_fusion_candidates",
]
