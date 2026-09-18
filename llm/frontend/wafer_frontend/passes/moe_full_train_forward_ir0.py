"""Construct real TP1/EP1-or-EP2 two-layer MoE forward TRAIN IR0 from shared Dense.

This replaces the old Dense gate/up/SwiGLU/down values and physical state
declarations; it does not add backward, source projection, or executable
training.  Materialization needs dedicated public MOE_* IR0 OpKind support.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import DType, MeshAxisName, Sharding, TensorValue
from ..schema.e2e_workload_graph import E2EStateKind
from ..schema.ir0 import (
    DeviceMesh, EdgeKind, GraphEdge, IR0, JobKind, LogicalNode,
    MeshAxis, NodeEffects, OpKind, OpPhase, EffectKind,
    StateAccess, StateAccessMode,
)
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.moe_full_training_block_workload import (
    MoeForwardBlockKind, MoeFullTrainingBlockWorkload,
)
from ..schema.persistent_state import (
    PersistentStateAccess, PersistentStateDecl, PersistentStateIdentity,
    PersistentStateLifetime, StateKind,
)
from ..schema.serde import canonical_digest
from .moe_full_forward_ir0_coverage import (
    require_moe_full_forward_ir0_coverage,
)


_MOE_FORWARD_KINDS = {
    MoeForwardBlockKind.ROUTER: "MOE_ROUTER",
    MoeForwardBlockKind.ROUTE_FREEZE: "MOE_ROUTE_FREEZE",
    MoeForwardBlockKind.DISPATCH: "MOE_DISPATCH",
    MoeForwardBlockKind.EXPERT: "MOE_EXPERT_FORWARD",
    MoeForwardBlockKind.COMBINE: "MOE_COMBINE",
}


@dataclass(frozen=True, slots=True)
class MoeForwardEpStateOwner:
    source_state_decl_ref: str
    source_e2e_state_ref: str
    source_e2e_parameter_view_ref: str
    source_e2e_parameter_name: str
    ep_owner: int
    tp_shard: int


@dataclass(frozen=True, slots=True)
class FullMoeForwardIr0Phase:
    graph: IR0
    dense_forward_ir0_ref: str
    source_moe_sequence_ref: str
    source_route_trace_refs: tuple[str, ...]
    removed_dense_op_refs: tuple[str, ...]
    removed_dense_state_refs: tuple[str, ...]
    shared_source_state_refs: tuple[tuple[str, str], ...]
    ep_state_owners: tuple[MoeForwardEpStateOwner, ...]
    route_state_refs: tuple[str, ...]
    step: int

    def validate(self) -> None:
        self.graph.validate("full_moe_forward_ir0")
        nodes = {node.id: node for node in self.graph.nodes}
        states = {state.id: state for state in self.graph.persistent_states}
        ep_degree = self.graph.instances[0].parallel.ep
        expected_ep_owners = 2 * (ep_degree + 3 * ep_degree)
        if (set(self.removed_dense_op_refs) & nodes.keys()
                or set(self.removed_dense_state_refs) & states.keys()
                or len(self.source_route_trace_refs) != 2
                or len(self.shared_source_state_refs) != 11
                or len({old for old, new in self.shared_source_state_refs}) != 11
                or len({new for old, new in self.shared_source_state_refs}) != 11
                or any(new not in states for old, new in self.shared_source_state_refs)
                or ep_degree not in (1, 2)
                or len(self.ep_state_owners) != expected_ep_owners
                or len(self.route_state_refs) != 2
                or len(set(self.route_state_refs)) != 2
                or any(ref not in states for ref in self.route_state_refs)
                or {owner.source_state_decl_ref for owner in self.ep_state_owners}
                != {owner.source_state_decl_ref for owner in self.ep_state_owners
                    if owner.source_state_decl_ref in states}
                or any(owner.tp_shard != 0
                       or not 0 <= owner.ep_owner < ep_degree
                       for owner in self.ep_state_owners)):
            raise SchemaError("MoE source op/state removal or EP physical owner lost",
                              path="full_moe_forward_ir0")
        values = {value.id: value for value in self.graph.values}
        accesses = self.graph.state_accesses
        instance = self.graph.instances[0]
        for layer, state_ref in enumerate(self.route_state_refs):
            state = states[state_ref]
            route_ref = f"{instance.id}.layer{layer}.moe.route_table_source"
            route = values.get(route_ref)
            matched = tuple(access for access in accesses
                            if access.state_ref == state_ref)
            if (route is None
                    or state.identity.kind is not StateKind.MOE_STATIC_ROUTE
                    or state.identity.instance_ref != instance.id
                    or state.identity.mesh_ref != instance.meshes[0].id
                    or state.identity.layer_index != layer
                    or state.identity.generation != self.step
                    or state.identity.tensor_ref != route_ref
                    or state.shape != route.shape
                    or state.layout != route.logical_layout
                    or route.producer is not None
                    or route.consumers != (f"{instance.id}.layer{layer}.moe.route_freeze",)
                    or len(matched) != 1
                    or matched[0].node_ref != route.consumers[0]
                    or matched[0].mode is not StateAccessMode.READ
                    or matched[0].rank != 0):
                raise SchemaError("static route HBM state must feed exact layer freeze",
                                  path=f"full_moe_forward_ir0.route_state[{layer}]")

    def validate_against(self, dense: IR0,
                         sequence: MoeCompileSequence) -> None:
        """Require owner1 genuine E2E view and production Die1 state home."""
        self.validate()
        require_moe_full_forward_ir0_coverage(
            self.graph, sequence, step=self.step,
        )
        dense.validate("full_moe_forward_dense_source")
        sequence.validate("full_moe_forward_sequence_source")
        if (self.dense_forward_ir0_ref != dense.id
                or self.source_moe_sequence_ref != sequence.id
                or self.source_route_trace_refs != tuple(
                    unit.route_trace_ref for unit in sequence.units
                    if unit.step == self.step)):
            raise SchemaError("forward source Dense/MoE identity or route digest drifted",
                              path="full_moe_forward_ir0.source")
        states = {state.id: state for state in self.graph.persistent_states}
        values = {value.id: value for value in self.graph.values}
        source_states = {state.id: state for state in
                         sequence.materialization.logical_graph.state_versions}
        source_views = {view.id: view for view in
                        sequence.materialization.logical_graph.tensor_values}
        units = {unit.layer:unit for unit in sequence.units
                 if unit.step == self.step}
        if len(units) != 2:
            raise SchemaError("two source layer units absent",
                              path="full_moe_forward_ir0.units")
        for owner in self.ep_state_owners:
            state = states[owner.source_state_decl_ref]
            origin = source_states.get(owner.source_e2e_state_ref)
            view = source_views.get(owner.source_e2e_parameter_view_ref)
            if (origin is None or origin.version != self.step
                    or origin.kind is not E2EStateKind.PARAMETER
                    or origin.logical_name != owner.source_e2e_parameter_name
                    or view is None or view.state_ref != origin.id
                    or view.logical_rank != owner.ep_owner
                    or view.shape != state.shape
                    or view.dtype is not state.dtype
                    or state.identity.generation != 0
                    or state.identity.shard_index != owner.tp_shard
                    or state.identity.ep_owner_rank != owner.ep_owner
                    or state.identity.tensor_ref not in values):
                raise SchemaError("EP owner is not the actual source E2E tensor home",
                                  path=f"full_moe_forward_ir0.owner[{owner.source_state_decl_ref}]")
            unit = units[origin.layer]
            group = next((group for group in unit.parameter_bindings
                          if owner.source_e2e_parameter_name in group.parameter_refs),None)
            previous_unit = next((unit for unit in sequence.units
                                  if (unit.step, unit.layer) ==
                                     (self.step - 1, origin.layer)), None)
            previous_group = (next((item for item in
                                   previous_unit.parameter_bindings
                                   if owner.source_e2e_parameter_name in
                                   item.parameter_refs), None)
                              if previous_unit is not None else None)
            position = (group.parameter_refs.index(
                owner.source_e2e_parameter_name) if group is not None else -1)
            previous_position = (previous_group.parameter_refs.index(
                owner.source_e2e_parameter_name)
                                 if previous_group is not None else -1)
            if (group is None or group.expert != origin.expert
                    or origin.id != group.input_parameter_state_refs[position]
                    or (self.step == 0 and origin.producer_op_id is not None)
                    or (self.step == 1 and
                        (previous_group is None or
                         previous_group.output_parameter_state_refs[
                             previous_position] != origin.id or
                         previous_group.sgd_operation_refs[
                             previous_position] != origin.producer_op_id))
                    or (origin.expert is not None and
                        owner.ep_owner != origin.expert)):
                raise SchemaError("expert owner/parameter group differs",
                                  path=f"full_moe_forward_ir0.owner[{state.id}]")
            homes = {abi.state_ref: abi for fragment in unit.linked_manifest.fragments
                     for abi in fragment.state_abi
                     if abi.state_ref in group.production_parameter_state_refs}
            if (set(homes) != set(group.production_parameter_state_refs)
                    or owner.ep_owner not in
                       {abi.die_id for abi in homes.values()}):
                raise SchemaError("EP owner lacks corresponding production physical StateABI Die",
                                  path=f"full_moe_forward_ir0.owner[{state.id}]")


def _resign_parameter_state(
    declaration: PersistentStateDecl, *,
    new_tensor_ref: str | None = None,
    shape: tuple[int, ...] | None = None,
    layout: str | None = None,
    ep_owner_rank: int | None = None,
) -> PersistentStateDecl:
    identity = PersistentStateIdentity.create(
        kind=StateKind.TRAINABLE_PARAMETER,
        instance_ref=declaration.identity.instance_ref,
        mesh_ref=declaration.identity.mesh_ref,
        request_ref=None,
        layer_index=None,
        tensor_ref=(declaration.identity.tensor_ref if new_tensor_ref is None
                    else new_tensor_ref),
        shard_index=0,
        generation=declaration.identity.generation,
        ep_owner_rank=ep_owner_rank,
    )
    return PersistentStateDecl.create(
        identity=identity, shape=(declaration.shape if shape is None else shape),
        dtype=declaration.dtype,
        layout=(declaration.layout if layout is None else layout),
        lifetime=PersistentStateLifetime.PERSISTENT,
        access=PersistentStateAccess.READ_WRITE,
    )


def _source_moe_operation_workload(
    sequence: MoeCompileSequence, *, step: int, layer: int,
    kind: MoeForwardBlockKind, expert: int | None,
) -> MoeFullTrainingBlockWorkload:
    unit = next(unit for unit in sequence.units
                if (unit.step, unit.layer) == (step, layer))
    trace = next(trace for trace in sequence.materialization.logical_graph.route_traces
                 if (trace.step, trace.layer) == (step, layer))
    refs = unit.operation_binding
    if kind is MoeForwardBlockKind.ROUTER:
        operation_ref = refs.router_operation_ref
    elif kind is MoeForwardBlockKind.ROUTE_FREEZE:
        operation_ref = refs.route_freeze_operation_ref
    elif kind is MoeForwardBlockKind.DISPATCH:
        operation_ref = refs.dispatch_operation_ref
    elif kind is MoeForwardBlockKind.EXPERT:
        operation_ref = refs.expert_forward_operation_refs[expert]
    else:
        operation_ref = refs.combine_operation_ref
    operations = {operation.id: operation for operation
                  in sequence.materialization.logical_graph.operations}
    source_op = operations.get(operation_ref)
    if (source_op is None or source_op.kind.value != kind.value
            or source_op.step != step or source_op.layer != layer
            or source_op.expert != expert or trace.id != unit.route_trace_ref
            or canonical_digest(trace) != unit.route_trace_digest
            or unit.spec.trace.expert_histogram != trace.expert_token_counts
            or tuple(item.expert_index for item in unit.spec.trace.assignments)
                != trace.expert_by_token):
        raise SchemaError("MoE block operation is not bound to true frozen per-layer route",
                          path=f"moe_full_train_forward_ir0.step{step}.layer{layer}")
    model = sequence.materialization.request.model
    result = MoeFullTrainingBlockWorkload(
        kind, sequence.materialization.request.case_id,
        operation_ref, trace.id, canonical_digest(trace),
        step, layer, expert, trace.token_count,
        model.hidden_size, model.intermediate_size,
        model.num_experts, trace.expert_token_counts, trace.expert_by_token,
        tuple(item.slot_index for item in sorted(
            unit.spec.trace.assignments, key=lambda item: item.token_index)),
    )
    result.validate()
    return result


def build_moe_full_train_forward_ir0(
    dense_forward: IR0, sequence: MoeCompileSequence, *, step: int = 0,
) -> FullMoeForwardIr0Phase:
    """Replace exactly both original Dense MLPs and preserve both residuals.

    Step1 binds P2 version1 parameter lineage, but a forward phase alone
    cannot prove an executed optimizer or complete two-step TRAIN timeline.
    """
    dense_forward.validate("dense_forward")
    sequence.validate("moe_sequence")
    if (type(step) is not int or step not in (0, 1)
            or dense_forward.producer_pass != "train_forward_expand"
            or dense_forward.job is not JobKind.TRAIN
            or len(dense_forward.instances) != 1
            or dense_forward.instances[0].parallel.tp != 1
            or dense_forward.instances[0].parallel.ep != 1
            or len(dense_forward.instances[0].meshes) != 1
            or sequence.materialization.request.mesh.rank_count not in (1, 2)
            or sequence.materialization.request.parallel.ep
                != sequence.materialization.request.mesh.rank_count
            or sequence.materialization.request.model.num_experts
                != sequence.materialization.request.mesh.rank_count
            or sequence.materialization.request.parallel.tp != 1
            or sequence.materialization.request.model.num_layers != 2
            or len({(unit.step, unit.layer) for unit in sequence.units}) != 4
            or sequence.materialization.request.steps.training is None
            or sequence.materialization.request.steps.training.sequence_length
                != dense_forward.profile.prefill_tokens):
        raise SchemaError("requires true TP1→EP1/EP2 two-layer step0/step1 TRAIN source with identical rows",
                          path="moe_full_train_forward_ir0.source")
    model = sequence.materialization.request.model
    vocab = dense_forward.values
    embedding_weight = next((value for value in vocab
                             if value.id == f"{dense_forward.instances[0].id}.tok_embeddings.weight"),None)
    lm_head_weight = next((value for value in vocab
                           if value.id == f"{dense_forward.instances[0].id}.lm_head.weight"),None)
    attention = [node for node in dense_forward.nodes
                 if node.kind is OpKind.ATTENTION]
    if (embedding_weight is None or lm_head_weight is None
            or embedding_weight.shape != (model.vocabulary_size,model.hidden_size)
            or lm_head_weight.shape != (model.hidden_size,model.vocabulary_size)
            or len(attention) != model.num_layers
            or any((node.workload.hidden_size,node.workload.num_heads,
                    node.workload.num_kv_heads,node.workload.head_dim)
                   != (model.hidden_size,model.num_attention_heads,
                       model.num_kv_heads,model.head_dim) for node in attention)):
        raise SchemaError("Dense full model vocab/head/attention does not match MoE source",
                          path="moe_full_train_forward_ir0.model")
    missing_op_kinds = tuple(name for name in _MOE_FORWARD_KINDS.values()
                             if not hasattr(OpKind,name))
    if missing_op_kinds:
        raise UnsupportedFeatureError(
            "canonical IR0 lacks typed forward MoE operations: "+
            ",".join(missing_op_kinds),
            path="moe_full_train_forward_ir0.op_kind",
        )
    initial = dense_forward.instances[0]
    mesh = initial.meshes[0]
    if any(axis.name is MeshAxisName.EP for axis in mesh.axes):
        raise SchemaError("original Dense mesh already carries EP",
                          path="moe_full_train_forward_ir0.mesh")
    ep_degree = sequence.materialization.request.parallel.ep
    instance = replace(
        initial, parallel=replace(initial.parallel, ep=ep_degree),
        meshes=(DeviceMesh(mesh.id, (
            *mesh.axes, MeshAxis(MeshAxisName.EP, ep_degree)
        )),),
    )
    old_nodes = {node.id: node for node in dense_forward.nodes}
    old_values = {value.id: value for value in dense_forward.values}
    removed_ops, removed_values, removed_states = set(), set(), set()
    states = list(dense_forward.persistent_states)
    owner_bindings: list[MoeForwardEpStateOwner] = []
    route_state_refs: list[str] = []
    new_nodes: list[LogicalNode] = []
    new_values: list[TensorValue] = []
    altered_values: dict[str, TensorValue] = {}
    altered_nodes: dict[str, LogicalNode] = {}
    new_accesses: list[StateAccess] = []
    source_versions = {(state.logical_name, state.version): state for state in
                       sequence.materialization.logical_graph.state_versions
                       if state.kind is E2EStateKind.PARAMETER}
    if len(source_versions) != len([state for state in
                                    sequence.materialization.logical_graph.state_versions
                                    if state.kind is E2EStateKind.PARAMETER]):
        raise SchemaError("MoE parameter state versions are ambiguous",
                          path="moe_full_train_forward_ir0.parameters")
    source_parameter_views = {(value.state_ref, value.logical_rank): value
                              for value in sequence.materialization.logical_graph.tensor_values
                              if value.state_ref in {state.id for state in
                                                     source_versions.values()}}
    if len(source_parameter_views) != len([value for value in
                                           sequence.materialization.logical_graph.tensor_values
                                           if value.state_ref in {state.id for state in
                                                                  source_versions.values()}]):
        raise SchemaError("MoE parameter owner views are ambiguous",
                          path="moe_full_train_forward_ir0.parameters")

    def add_value(value_id: str, shape: tuple[int, ...], dtype: DType,
                  producer: str | None, consumers: tuple[str, ...]) -> TensorValue:
        value = TensorValue(
            id=value_id, shape=shape, dtype=dtype, logical_layout="MoE_"+value_id,
            sharding=Sharding(mesh.id, (None,) * len(shape), ()),
            producer=producer, consumers=consumers, alias_set=None,
        )
        new_values.append(value)
        return value

    def add_parameter(value: TensorValue, *, logical_name: str,
                      owner_ranks: tuple[int, ...],
                      read_node_ref: str, template: PersistentStateDecl,
                      layer: int) -> None:
        source = source_versions.get((logical_name, step))
        previous_unit = next((unit for unit in sequence.units
                              if (unit.step, unit.layer) == (step - 1, layer)), None)
        current_unit = next(unit for unit in sequence.units
                            if (unit.step, unit.layer) == (step, layer))
        group = next((group for group in current_unit.parameter_bindings
                      if logical_name in group.parameter_refs), None)
        if (source is None or group is None
                or source.id != group.input_parameter_state_refs[
                    group.parameter_refs.index(logical_name)]
                or (step == 0 and source.producer_op_id is not None)
                or (step == 1 and (previous_unit is None
                    or source.producer_op_id is None
                    or not any(logical_name in previous.parameter_refs
                               and previous.output_parameter_state_refs[
                                   previous.parameter_refs.index(logical_name)] == source.id
                               and previous.sgd_operation_refs[
                                   previous.parameter_refs.index(logical_name)]
                                   == source.producer_op_id
                               for previous in previous_unit.parameter_bindings)))):
            raise SchemaError("source owner lacks exact step-version parameter",
                              path=f"moe_full_train_forward_ir0.{logical_name}")
        for owner in owner_ranks:
            view = source_parameter_views.get((source.id, owner))
            if (view is None or view.shape != value.shape
                    or view.dtype is not value.dtype):
                raise SchemaError("source actual EP parameter view differs from IR0 tensor",
                                  path=f"moe_full_train_forward_ir0.{logical_name}.rank{owner}")
            declaration = _resign_parameter_state(
                template, new_tensor_ref=value.id,
                shape=value.shape,
                layout=value.logical_layout,
                ep_owner_rank=owner,
            )
            states.append(declaration)
            new_accesses.append(StateAccess.create(
                node_ref=read_node_ref, state_ref=declaration.id,
                mode=StateAccessMode.READ, rank=owner,
            ))
            owner_bindings.append(MoeForwardEpStateOwner(
                declaration.id, source.id, view.id, logical_name, owner, 0,
            ))

    for layer in range(2):
        prefix = f"{initial.id}.layer{layer}."
        old_names = ("gate_up", "swiglu", "down")
        dense_mlps = tuple(prefix+name for name in old_names)
        norm_id, residual_id = prefix+"norm2", prefix+"residual2"
        norm = old_values.get(prefix+"norm2_out")
        residual = old_nodes.get(residual_id)
        old_down = old_values.get(prefix+"down_out")
        old_weight_refs = (prefix+"w_gate_up", prefix+"w_down")
        if (any(ref not in old_nodes for ref in dense_mlps)
                or norm is None or norm.producer != norm_id
                or norm.consumers != (dense_mlps[0],)
                or old_down is None or old_down.producer != dense_mlps[2]
                or residual is None or residual.inputs[1] != old_down.id
                or norm.shape != (dense_forward.profile.prefill_tokens,
                                  model.hidden_size)
                or any(ref not in old_values for ref in old_weight_refs)
                or old_values[old_weight_refs[0]].shape !=
                   (model.hidden_size,2*model.intermediate_size)
                or old_values[old_weight_refs[1]].shape !=
                   (model.intermediate_size,model.hidden_size)):
            raise SchemaError("true Dense norm2→three MLP→residual2 spine not found",
                              path=f"moe_full_train_forward_ir0.layer{layer}")
        old_weight_states = tuple(state for state in states
                                  if state.identity.tensor_ref in old_weight_refs)
        if len(old_weight_states) != 2:
            raise SchemaError("Dense MLP must remove exactly two persistent parameters",
                              path=f"moe_full_train_forward_ir0.layer{layer}")
        removed_states.update(state.id for state in old_weight_states)
        removed_ops.update(dense_mlps)
        removed_values.update((prefix+"gate_up_out", prefix+"swiglu_out",
                               old_down.id, *old_weight_refs))
        router_ref = prefix+"moe.router"
        freeze_ref = prefix+"moe.route_freeze"
        dispatch_ref = prefix+"moe.dispatch"
        combine_ref = prefix+"moe.combine"
        expert_refs = tuple(prefix+f"moe.expert{expert}"
                            for expert in range(model.num_experts))
        hidden = model.hidden_size
        tokens = dense_forward.profile.prefill_tokens
        route_scores = add_value(prefix+"moe.router_scores",
                                 (tokens,model.num_experts), DType.FP16,
                                 router_ref, (freeze_ref, combine_ref))
        route_source = add_value(prefix+"moe.route_table_source",
                                 (tokens,5), DType.INT32, None, (freeze_ref,))
        route = add_value(prefix+"moe.route_ids", (tokens,5), DType.INT32,
                          freeze_ref, (dispatch_ref, combine_ref))
        route_identity = PersistentStateIdentity.create(
            kind=StateKind.MOE_STATIC_ROUTE, instance_ref=initial.id, mesh_ref=mesh.id,
            request_ref=sequence.materialization.request.case_id,
            layer_index=layer, tensor_ref=route_source.id, shard_index=0,
            generation=step,
        )
        route_state = PersistentStateDecl.create(
            identity=route_identity, shape=route_source.shape, dtype=DType.INT32,
            layout=route_source.logical_layout, lifetime=PersistentStateLifetime.STEP,
            access=PersistentStateAccess.READ_ONLY,
        )
        states.append(route_state)
        route_state_refs.append(route_state.id)
        new_accesses.append(StateAccess.create(
            node_ref=freeze_ref, state_ref=route_state.id,
            mode=StateAccessMode.READ, rank=0,
        ))
        router_weights = []
        for rank in range(ep_degree):
            weight = add_value(prefix+f"moe.router.weight.ep{rank}",
                               (hidden,model.num_experts), DType.FP16,
                               None,(router_ref,))
            router_weights.append(weight)
            add_parameter(weight,
                          logical_name=f"layer.{layer}.router.weight",
                          owner_ranks=(rank,),read_node_ref=router_ref,
                          template=old_weight_states[0],layer=layer)
        histogram = next(trace for trace in
                         sequence.materialization.logical_graph.route_traces
                         if (trace.step, trace.layer) == (step, layer)).expert_token_counts
        if any(count == 0 for count in histogram):
            raise UnsupportedFeatureError("positive-size canonical TensorValue cannot represent zero-token expert; implement explicit zero-work physical degeneracy first",
                                          path=f"moe_full_train_forward_ir0.layer{layer}")
        dispatches, expert_outputs = [], []
        for expert, count in enumerate(histogram):
            exp_ref = expert_refs[expert]
            dispatches.append(add_value(
                prefix+f"moe.dispatch{expert}", (count,hidden), DType.FP16,
                dispatch_ref, (exp_ref,),
            ))
            expert_outputs.append(add_value(
                prefix+f"moe.expert{expert}.output", (count,hidden), DType.FP16,
                exp_ref, (combine_ref,),
            ))
            weights = []
            for projection, shape in (
                ("gate",(hidden,model.intermediate_size)),
                ("up",(hidden,model.intermediate_size)),
                ("down",(model.intermediate_size,hidden)),
            ):
                value = add_value(prefix+f"moe.expert{expert}.{projection}.weight",
                                  shape, DType.FP16, None, (exp_ref,))
                weights.append(value)
                add_parameter(value,
                              logical_name=f"layer.{layer}.expert.{expert}.{projection}.weight",
                              owner_ranks=(expert,), read_node_ref=exp_ref,
                              template=old_weight_states[0], layer=layer)
            first_workload = _source_moe_operation_workload(
                sequence, step=step,layer=layer,
                kind=MoeForwardBlockKind.EXPERT,expert=expert,
            )
            new_nodes.append(LogicalNode(
                id=exp_ref, instance_id=initial.id,
                kind=OpKind.MOE_EXPERT_FORWARD, phase=OpPhase.FWD, stage=0,
                mesh_ref=mesh.id,
                inputs=(dispatches[-1].id, *(weight.id for weight in weights)),
                outputs=(expert_outputs[-1].id,), workload=first_workload,
                math=residual.math,
                effects=NodeEffects(EffectKind.PURE,None,None),
                impl_ref="moe_expert_forward",
            ))
        combine = add_value(prefix+"moe.combine_out", (tokens,hidden),
                            DType.FP16, combine_ref, (residual_id,))
        altered_values[norm.id] = replace(norm,
            consumers=(router_ref,dispatch_ref))
        altered_nodes[residual_id] = replace(
            residual, inputs=(residual.inputs[0],combine.id),
        )
        sources = (
            (MoeForwardBlockKind.ROUTER, router_ref,
             (norm.id,*(weight.id for weight in router_weights)),
             (route_scores.id,)),
            (MoeForwardBlockKind.ROUTE_FREEZE,freeze_ref,
             (route_scores.id,route_source.id), (route.id,)),
            (MoeForwardBlockKind.DISPATCH,dispatch_ref,
             (norm.id,route.id), tuple(value.id for value in dispatches)),
            (MoeForwardBlockKind.COMBINE,combine_ref,
             (*[value.id for value in expert_outputs], route.id,
              route_scores.id), (combine.id,)),
        )
        for kind, node_ref, inputs, outputs in sources:
            new_nodes.append(LogicalNode(
                id=node_ref, instance_id=initial.id,
                kind=getattr(OpKind,_MOE_FORWARD_KINDS[kind]),
                phase=OpPhase.FWD, stage=0, mesh_ref=mesh.id,
                inputs=inputs, outputs=outputs,
                workload=_source_moe_operation_workload(
                    sequence,step=step,layer=layer,kind=kind,expert=None),
                math=residual.math,
                effects=NodeEffects(EffectKind.PURE,None,None),
                impl_ref="moe_"+kind.value,
            ))
    removed = set(removed_ops)
    states = [state for state in states if state.id not in removed_states]
    old_to_new = {}
    for state in states:
        if state in dense_forward.persistent_states:
            rebuilt = _resign_parameter_state(state)
            old_to_new[state.id] = rebuilt.id
            states[states.index(state)] = rebuilt
    accesses = [StateAccess.create(
        node_ref=access.node_ref,
        state_ref=old_to_new[access.state_ref],
        mode=access.mode, rank=access.rank,
        read_offset=access.read_offset,read_shape=access.read_shape,
        write_offset=access.write_offset,write_shape=access.write_shape,
    ) for access in dense_forward.state_accesses
        if access.state_ref not in removed_states
        and access.node_ref not in removed]
    accesses.extend(new_accesses)
    values = tuple(altered_values.get(value.id,value) for value in
                   dense_forward.values if value.id not in removed_values)
    values += tuple(new_values)
    nodes = tuple(altered_nodes.get(node.id,node) for node in
                  dense_forward.nodes if node.id not in removed)
    nodes += tuple(new_nodes)
    data_edges = tuple(GraphEdge(
        f"moe_data::{value.id}::{consumer}",EdgeKind.DATA,
        value.producer,consumer,value.id,
    ) for value in values if value.producer is not None
        for consumer in value.consumers)
    old_controls = tuple(edge for edge in dense_forward.edges
                         if edge.kind is EdgeKind.CONTROL
                         and edge.source_node not in removed
                         and edge.destination_node not in removed)
    ir0 = IR0.create(
        producer_pass="moe_full_train_forward_replacement_ir0",
        job=JobKind.TRAIN, instances=(instance,), nodes=nodes,
        values=values,edges=(*data_edges,*old_controls),
        fusion_candidates=(), profile=dense_forward.profile,
        train=dense_forward.train,
        persistent_states=tuple(sorted(states,key=lambda state:
            (state.identity.id,state.id))),
        state_accesses=tuple(sorted(accesses,key=lambda access:
            (access.node_ref,access.state_ref,access.rank,access.id))),
    )
    result = FullMoeForwardIr0Phase(
        ir0,dense_forward.id,sequence.id,
        tuple(unit.route_trace_ref for unit in sequence.units
              if unit.step == step),
        tuple(sorted(removed_ops)),tuple(sorted(removed_states)),
        tuple(sorted(old_to_new.items())),tuple(owner_bindings),
        tuple(route_state_refs),step,
    )
    result.validate()
    result.validate_against(dense_forward,sequence)
    return result
