"""Build namespaced static Stage 4 separated or TP1 fused PD IR0 graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.experiment import ExperimentSpec, InferOutput
from ..schema.ir0 import (
    EdgeKind,
    FusionCandidate,
    GraphEdge,
    IR0,
    InstanceProfileBinding,
    LogicalNode,
    NodeProfileBinding,
    NodeEffects,
    StateAccess,
    StateAccessMode,
)
from ..schema.common import TensorValue
from ..schema.persistent_state import (
    PersistentStateDecl,
    PersistentStateIdentity,
    StateKind,
)
from ..schema.serde import canonical_digest
from ..schema.stage4_pd import Stage4PdMode, Stage4PdPlan
from .build_ir0 import build_ir0_template_for_profile
from .logical_expand import logical_expand
from .validate_ir0 import DenseIR0Validator


@dataclass(frozen=True, slots=True)
class _NamespacedGraph:
    nodes: tuple[LogicalNode, ...]
    values: tuple[TensorValue, ...]
    edges: tuple[GraphEdge, ...]
    fusion_candidates: tuple[FusionCandidate, ...]
    persistent_states: tuple[PersistentStateDecl, ...]
    state_accesses: tuple[StateAccess, ...]


def _scoped(prefix: str, value: str | None) -> str | None:
    return None if value is None else f"{prefix}.{value}"


def _namespace_graph(
    graph: IR0,
    prefix: str,
    *,
    request_profile_id: str,
    suppress_kv_reads: bool = False,
    shared_value_ids: frozenset[str] = frozenset(),
    share_stateful_effects: bool = False,
) -> _NamespacedGraph:
    graph.validate(f"{prefix}.source_graph")
    node_ids = {node.id: f"{prefix}.{node.id}" for node in graph.nodes}
    value_ids = {
        value.id: (
            value.id if value.id in shared_value_ids else f"{prefix}.{value.id}"
        )
        for value in graph.values
    }

    nodes = tuple(
        replace(
            node,
            id=node_ids[node.id],
            inputs=tuple(value_ids[value_id] for value_id in node.inputs),
            outputs=tuple(value_ids[value_id] for value_id in node.outputs),
            effects=NodeEffects(
                kind=node.effects.kind,
                effect_token=(
                    node.effects.effect_token
                    if share_stateful_effects and node.effects.effect_token is not None
                    else _scoped(prefix, node.effects.effect_token)
                ),
                alias_set=(
                    node.effects.alias_set
                    if share_stateful_effects and node.effects.alias_set is not None
                    else _scoped(prefix, node.effects.alias_set)
                ),
            ),
        )
        for node in graph.nodes
    )
    values = tuple(
        replace(
            value,
            id=value_ids[value.id],
            producer=(
                None if value.producer is None else node_ids[value.producer]
            ),
            consumers=tuple(node_ids[item] for item in value.consumers),
            alias_set=_scoped(prefix, value.alias_set),
        )
        for value in graph.values
    )
    edges = tuple(
        replace(
            edge,
            id=f"{prefix}.{edge.id}",
            source_node=node_ids[edge.source_node],
            destination_node=node_ids[edge.destination_node],
            value_id=(
                None if edge.value_id is None else value_ids[edge.value_id]
            ),
        )
        for edge in graph.edges
    )
    candidates = tuple(
        replace(
            candidate,
            id=f"{prefix}.{candidate.id}",
            members=tuple(node_ids[item] for item in candidate.members),
            boundary_inputs=tuple(
                value_ids[item] for item in candidate.boundary_inputs
            ),
            boundary_outputs=tuple(
                value_ids[item] for item in candidate.boundary_outputs
            ),
        )
        for candidate in graph.fusion_candidates
    )

    state_ids: dict[str, str] = {}
    states: list[PersistentStateDecl] = []
    for declaration in graph.persistent_states:
        identity = declaration.identity
        request_ref = identity.request_ref
        if request_ref is not None:
            expected_prefix = f"{request_profile_id}:"
            if not request_ref.startswith(expected_prefix):
                raise SchemaError(
                    "KV request lineage must carry the exact source profile",
                    path=f"{prefix}.source_graph.persistent_states",
                )
            request_ref = request_ref[len(expected_prefix):]
        rebuilt_identity = PersistentStateIdentity.create(
            kind=identity.kind,
            instance_ref=identity.instance_ref,
            mesh_ref=identity.mesh_ref,
            request_ref=request_ref,
            layer_index=identity.layer_index,
            tensor_ref=(
                None
                if identity.tensor_ref is None
                else value_ids[identity.tensor_ref]
            ),
            shard_index=identity.shard_index,
            generation=identity.generation,
        )
        rebuilt = PersistentStateDecl.create(
            identity=rebuilt_identity,
            shape=declaration.shape,
            dtype=declaration.dtype,
            layout=declaration.layout,
            lifetime=declaration.lifetime,
            access=declaration.access,
        )
        state_ids[declaration.id] = rebuilt.id
        states.append(rebuilt)
    state_kind_by_id = {
        declaration.id: declaration.identity.kind
        for declaration in graph.persistent_states
    }
    accesses = []
    for access in graph.state_accesses:
        suppress_read = (
            suppress_kv_reads
            and state_kind_by_id[access.state_ref]
            in (StateKind.KV_KEY, StateKind.KV_VALUE)
            and access.mode is StateAccessMode.READ_WRITE
        )
        accesses.append(
            StateAccess.create(
                node_ref=node_ids[access.node_ref],
                state_ref=state_ids[access.state_ref],
                mode=(StateAccessMode.WRITE if suppress_read else access.mode),
                rank=access.rank,
                read_offset=None if suppress_read else access.read_offset,
                read_shape=None if suppress_read else access.read_shape,
                write_offset=access.write_offset,
                write_shape=access.write_shape,
            )
        )
    return _NamespacedGraph(
        nodes=nodes,
        values=values,
        edges=edges,
        fusion_candidates=candidates,
        persistent_states=tuple(states),
        state_accesses=tuple(accesses),
    )


def _parameter_value_ids(graph: IR0) -> frozenset[str]:
    result = {
        declaration.identity.tensor_ref
        for declaration in graph.persistent_states
        if declaration.identity.kind is StateKind.PARAMETER
    }
    if None in result:
        raise SchemaError(
            "parameter state requires tensor_ref",
            path="stage4_fused_ir0.persistent_states",
        )
    return frozenset(result)


def _merge_fused_values(
    prefill: tuple[TensorValue, ...],
    decode: tuple[TensorValue, ...],
) -> tuple[TensorValue, ...]:
    ordered: list[TensorValue] = []
    indices: dict[str, int] = {}
    for value in prefill + decode:
        previous_index = indices.get(value.id)
        if previous_index is None:
            indices[value.id] = len(ordered)
            ordered.append(value)
            continue
        previous = ordered[previous_index]
        if replace(previous, consumers=()) != replace(value, consumers=()):
            raise SchemaError(
                "shared parameter values must have identical metadata",
                path=f"stage4_fused_ir0.values[{value.id!r}]",
            )
        consumers = previous.consumers + value.consumers
        if len(set(consumers)) != len(consumers):
            raise SchemaError(
                "shared parameter consumers must be unique",
                path=f"stage4_fused_ir0.values[{value.id!r}].consumers",
            )
        ordered[previous_index] = replace(previous, consumers=consumers)
    return tuple(ordered)


def _merge_fused_states(
    prefill: tuple[PersistentStateDecl, ...],
    decode: tuple[PersistentStateDecl, ...],
) -> tuple[PersistentStateDecl, ...]:
    declarations: dict[str, PersistentStateDecl] = {}
    for declaration in prefill + decode:
        previous = declarations.get(declaration.id)
        if previous is not None and previous != declaration:
            raise SchemaError(
                "shared state identity has incompatible declarations",
                path="stage4_fused_ir0.persistent_states",
            )
        declarations[declaration.id] = declaration
    return tuple(
        sorted(declarations.values(), key=lambda item: (item.identity.id, item.id))
    )


def build_stage4_fused_ir0(
    spec: ExperimentSpec,
    plan: Stage4PdPlan,
) -> IR0:
    """Materialize TP1 fused prefill then decode over one shared state domain."""

    spec.validate("spec")
    plan.validate("plan")
    if canonical_digest(spec) != plan.source_spec_digest:
        raise SchemaError(
            "does not match the Stage 4 plan source spec",
            path="plan.source_spec_digest",
        )
    if plan.mode is not Stage4PdMode.FUSED:
        raise SchemaError("requires a fused PD plan", path="plan.mode")
    if plan.prefill_tp != 1 or plan.decode_tp != 1:
        raise UnsupportedFeatureError(
            "Stage 4 fused logical preview supports TP1 only",
            path="plan.prefill_tp",
        )
    if spec.workload.infer.output is not InferOutput.LOGITS:
        raise UnsupportedFeatureError(
            "Stage 4 fused logical preview supports logits output only",
            path="spec.workload.infer.output",
        )

    prefill_graph = logical_expand(
        build_ir0_template_for_profile(
            spec,
            instance_ref=plan.prefill_instance_ref,
            exact_profile=plan.prefill_profile,
        )
    ).entries[0].graph
    decode_graph = logical_expand(
        build_ir0_template_for_profile(
            spec,
            instance_ref=plan.decode_instance_ref,
            exact_profile=plan.decode_profile,
        )
    ).entries[0].graph
    if prefill_graph.instances != decode_graph.instances:
        raise SchemaError(
            "fused phases must use one identical logical instance",
            path="stage4_fused_ir0.instances",
        )
    parameter_ids = _parameter_value_ids(prefill_graph)
    if parameter_ids != _parameter_value_ids(decode_graph):
        raise SchemaError(
            "fused phases must expose identical parameter tensors",
            path="stage4_fused_ir0.values",
        )
    prefill = _namespace_graph(
        prefill_graph,
        "prefill",
        request_profile_id=plan.prefill_profile.id,
        shared_value_ids=parameter_ids,
        share_stateful_effects=True,
    )
    decode = _namespace_graph(
        decode_graph,
        "decode",
        request_profile_id=plan.decode_profile.id,
        shared_value_ids=parameter_ids,
        share_stateful_effects=True,
    )
    control = GraphEdge(
        id=f"{plan.id}.prefill_complete_to_decode_start",
        kind=EdgeKind.CONTROL,
        source_node=prefill.nodes[-1].id,
        destination_node=decode.nodes[0].id,
        value_id=None,
    )
    bindings = tuple(
        sorted(
            (
                InstanceProfileBinding(
                    plan.prefill_instance_ref, plan.prefill_profile.key
                ),
                InstanceProfileBinding(
                    plan.decode_instance_ref, plan.decode_profile.key
                ),
            ),
            key=lambda item: (item.instance_ref, item.profile.stable_id()),
        )
    )
    nodes = prefill.nodes + decode.nodes
    node_profiles = tuple(
        NodeProfileBinding(node.id, plan.prefill_profile.key)
        for node in prefill.nodes
    ) + tuple(
        NodeProfileBinding(node.id, plan.decode_profile.key)
        for node in decode.nodes
    )
    result = IR0.create(
        producer_pass="stage4_logical_expand",
        job=prefill_graph.job,
        instances=prefill_graph.instances,
        nodes=nodes,
        values=_merge_fused_values(prefill.values, decode.values),
        edges=prefill.edges + decode.edges + (control,),
        fusion_candidates=(
            prefill.fusion_candidates + decode.fusion_candidates
        ),
        profile=bindings[0].profile,
        instance_profiles=bindings,
        node_profiles=node_profiles,
        pd_plan_id=plan.id,
        persistent_states=_merge_fused_states(
            prefill.persistent_states, decode.persistent_states
        ),
        state_accesses=prefill.state_accesses + decode.state_accesses,
    )
    result.validate("stage4_fused_ir0")
    DenseIR0Validator.validate(result, "stage4_fused_ir0")
    return result


def build_stage4_separated_ir0(
    spec: ExperimentSpec,
    plan: Stage4PdPlan,
) -> IR0:
    """Materialize the selected prefill and decode phases into one IR0."""

    spec.validate("spec")
    plan.validate("plan")
    if canonical_digest(spec) != plan.source_spec_digest:
        raise SchemaError(
            "does not match the Stage 4 plan source spec",
            path="plan.source_spec_digest",
        )
    if plan.mode is not Stage4PdMode.SEPARATED:
        raise UnsupportedFeatureError(
            "fused PD state unification is not implemented in this slice",
            path="plan.mode",
        )
    prefill_template = build_ir0_template_for_profile(
        spec,
        instance_ref=plan.prefill_instance_ref,
        exact_profile=plan.prefill_profile,
    )
    decode_template = build_ir0_template_for_profile(
        spec,
        instance_ref=plan.decode_instance_ref,
        exact_profile=plan.decode_profile,
    )
    prefill_graph = logical_expand(prefill_template).entries[0].graph
    decode_graph = logical_expand(decode_template).entries[0].graph
    prefill = _namespace_graph(
        prefill_graph,
        "prefill",
        request_profile_id=plan.prefill_profile.id,
    )
    decode = _namespace_graph(
        decode_graph,
        "decode",
        request_profile_id=plan.decode_profile.id,
        suppress_kv_reads=True,
    )
    control = GraphEdge(
        id=f"{plan.id}.prefill_complete_to_decode_start",
        kind=EdgeKind.CONTROL,
        source_node=prefill.nodes[-1].id,
        destination_node=decode.nodes[0].id,
        value_id=None,
    )
    bindings = tuple(
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
            key=lambda item: (item.instance_ref, item.profile.stable_id()),
        )
    )
    result = IR0.create(
        producer_pass="stage4_logical_expand",
        job=prefill_graph.job,
        instances=(
            prefill_graph.instances[0],
            decode_graph.instances[0],
        ),
        nodes=prefill.nodes + decode.nodes,
        values=prefill.values + decode.values,
        edges=prefill.edges + decode.edges + (control,),
        fusion_candidates=(
            prefill.fusion_candidates + decode.fusion_candidates
        ),
        profile=bindings[0].profile,
        instance_profiles=bindings,
        pd_plan_id=plan.id,
        persistent_states=(
            prefill.persistent_states + decode.persistent_states
        ),
        state_accesses=prefill.state_accesses + decode.state_accesses,
    )
    result.validate("stage4_ir0")
    return result


__all__ = ["build_stage4_fused_ir0", "build_stage4_separated_ir0"]
