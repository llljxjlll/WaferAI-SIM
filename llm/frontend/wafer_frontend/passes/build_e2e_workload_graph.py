"""Build typed multi-layer, multi-step logical graphs for four E2E families."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import DType
from ..schema.e2e_workload_graph import (
    E2EOperationKind,
    E2ERouteTrace,
    E2EStateKind,
    E2EStateVersion,
    E2ETensorValue,
    E2EWorkloadGraph,
    E2EWorkloadOperation,
)
from ..schema.ir0 import ReduceOp
from ..schema.parallel_placement import (
    ParallelPlacement,
    ParameterOwnershipKind,
    build_dense_parallel_placement,
    build_moe_parallel_placement,
)
from ..schema.rect_mesh import RectMeshSpec
from ..schema.workload_run import WorkloadOptimizerKind, WorkloadRunRequest


def _placement(request: WorkloadRunRequest) -> ParallelPlacement:
    common = {
        "mesh": RectMeshSpec(request.mesh.rows, request.mesh.columns),
        "tp_degree": request.parallel.tp,
        "dp_degree": request.parallel.dp,
        "pp_degree": request.parallel.pp,
        "active_die_ids": request.parallel.active_die_ids or None,
    }
    if request.family.is_moe:
        return build_moe_parallel_placement(
            ep_degree=request.parallel.ep,
            num_experts=request.model.num_experts,
            **common,
        )
    return build_dense_parallel_placement(**common)


def _tokens(request: WorkloadRunRequest, step: int) -> int:
    if request.family.is_training:
        assert request.steps.training is not None
        return (
            request.steps.training.global_batch_size
            * request.steps.training.sequence_length
        )
    assert request.steps.inference is not None
    if step == 0:
        return (
            request.steps.inference.request_count
            * request.steps.inference.prefill_tokens
        )
    return request.steps.inference.request_count


def _parameter_shape(request: WorkloadRunRequest, name: str) -> tuple[int, ...]:
    model = request.model
    if name == "embedding.weight":
        return (model.vocabulary_size, model.hidden_size)
    if name == "final_norm.weight":
        return (model.hidden_size,)
    if name == "lm_head.weight":
        return (model.hidden_size, model.vocabulary_size)
    if name.endswith("input_norm.weight") or name.endswith("post_norm.weight"):
        return (model.hidden_size,)
    if name.endswith("qkv.weight"):
        return (
            model.hidden_size,
            (model.num_attention_heads + 2 * model.num_kv_heads)
            * model.head_dim,
        )
    if name.endswith("attention_out.weight"):
        return (model.hidden_size, model.hidden_size)
    if name.endswith("router.weight"):
        return (model.hidden_size, model.num_experts)
    if name.endswith("gate.weight") or name.endswith("up.weight"):
        return (model.hidden_size, model.intermediate_size)
    if name.endswith("down.weight"):
        return (model.intermediate_size, model.hidden_size)
    raise SchemaError("unknown parameter tensor", path="logical_name")


def _state_tensor_spec(
    request: WorkloadRunRequest,
    logical_name: str,
    kind: E2EStateKind,
    version: int,
    *,
    shape: tuple[int, ...] | None,
    dtype: DType | None,
) -> tuple[tuple[int, ...], DType]:
    if shape is not None:
        return shape, request.model.dtype if dtype is None else dtype
    if kind is E2EStateKind.PARAMETER:
        return _parameter_shape(request, logical_name), request.model.dtype
    if kind is E2EStateKind.GRADIENT:
        parameter = logical_name.removeprefix("gradient.raw.").removeprefix(
            "gradient.synced."
        )
        return _parameter_shape(request, parameter), DType.FP32
    if kind in (
        E2EStateKind.OPTIMIZER_MASTER,
        E2EStateKind.OPTIMIZER_MOMENT1,
        E2EStateKind.OPTIMIZER_MOMENT2,
    ):
        parameter = logical_name.split(".", 3)[-1]
        return _parameter_shape(request, parameter), DType.FP32
    if kind is E2EStateKind.OPTIMIZER_STEP:
        return (1,), DType.INT32
    if kind is E2EStateKind.KV:
        assert request.steps.inference is not None
        cached_tokens = (
            0
            if version == 0
            else request.steps.inference.prefill_tokens
            + (version - 1) * request.steps.inference.request_count
        )
        return (
            request.steps.inference.request_count,
            request.model.num_kv_heads,
            cached_tokens,
            request.model.head_dim,
        ), request.model.dtype
    raise SchemaError("tensor shape must be explicit", path="shape")


class _GraphBuilder:
    def __init__(self, request: WorkloadRunRequest, placement: ParallelPlacement) -> None:
        self.request = request
        self.placement = placement
        self.operations: list[E2EWorkloadOperation] = []
        self.states: dict[
            tuple[str, E2EStateKind, int, int | None, int | None],
            E2EStateVersion,
        ] = {}
        self.values: dict[str, E2ETensorValue] = {}
        self.state_values: dict[str, tuple[str, ...]] = {}
        self.route_traces: list[E2ERouteTrace] = []

    def _owners(self, expert: int | None):
        kind = (
            ParameterOwnershipKind.EXPERT
            if expert is not None
            else ParameterOwnershipKind.SHARED
        )
        return tuple(
            owner
            for owner in self.placement.ownership_domains
            if owner.kind is kind and owner.expert_id == expert
        )

    def _rank_shape(
        self,
        shape: tuple[int, ...],
        tp_shard: int,
        preferred_axis: int | None = None,
    ) -> tuple[int, ...]:
        del tp_shard
        if self.placement.tp_degree == 1:
            return shape
        dimensions = list(shape)
        if (
            preferred_axis is not None
            and dimensions[preferred_axis] % self.placement.tp_degree == 0
        ):
            dimensions[preferred_axis] //= self.placement.tp_degree
            return tuple(dimensions)
        for axis in reversed(range(len(dimensions))):
            if dimensions[axis] % self.placement.tp_degree == 0:
                dimensions[axis] //= self.placement.tp_degree
                return tuple(dimensions)
        return shape

    def state(
        self,
        logical_name: str,
        kind: E2EStateKind,
        version: int,
        *,
        layer: int | None = None,
        expert: int | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DType | None = None,
        owned: bool | None = None,
    ) -> str:
        key = (logical_name, kind, version, layer, expert)
        if key not in self.states:
            self.states[key] = E2EStateVersion.create(
                case_id=self.request.case_id,
                logical_name=logical_name,
                kind=kind,
                version=version,
                layer=layer,
                expert=expert,
            )
            state = self.states[key]
            logical_shape, logical_dtype = _state_tensor_spec(
                self.request,
                logical_name,
                kind,
                version,
                shape=shape,
                dtype=dtype,
            )
            if owned is None:
                owned = kind in (
                    E2EStateKind.PARAMETER,
                    E2EStateKind.GRADIENT,
                    E2EStateKind.OPTIMIZER_MASTER,
                    E2EStateKind.OPTIMIZER_MOMENT1,
                    E2EStateKind.OPTIMIZER_MOMENT2,
                    E2EStateKind.OPTIMIZER_STEP,
                )
            bindings: list[tuple[int, int, str | None]] = []
            if owned:
                for owner in self._owners(expert):
                    bindings.extend(
                        (rank, owner.tp_shard, owner.id)
                        for rank in owner.replica_ranks
                    )
            else:
                bindings.extend(
                    (
                        rank.logical_rank,
                        rank.coordinate.tp,
                        None,
                    )
                    for rank in self.placement.rank_placements
                )
            value_ids: list[str] = []
            for rank, tp_shard, owner_ref in bindings:
                value = E2ETensorValue.create(
                    case_id=self.request.case_id,
                    logical_name=f"{logical_name}.rank{rank}",
                    shape=self._rank_shape(
                        logical_shape,
                        tp_shard,
                        preferred_axis=1 if kind is E2EStateKind.KV else None,
                    ),
                    dtype=logical_dtype,
                    state_ref=state.id,
                    owner_domain_ref=owner_ref,
                    logical_rank=rank,
                    tp_shard=tp_shard,
                )
                self.values[value.id] = value
                value_ids.append(value.id)
            self.state_values[state.id] = tuple(value_ids)
        return self.states[key].id

    def add(
        self,
        kind: E2EOperationKind,
        *,
        phase: str,
        step: int,
        layer: int | None = None,
        expert: int | None = None,
        parameter_ref: str | None = None,
        reads: tuple[str, ...] = (),
        writes: tuple[str, ...] = (),
        input_value_refs: tuple[str, ...] = (),
        output_value_refs: tuple[str, ...] = (),
        group_refs: tuple[str, ...] = (),
        reduce_op: ReduceOp | None = None,
        normalization_denominator: int | None = None,
        deps: tuple[str, ...] | None = None,
    ) -> E2EWorkloadOperation:
        if deps is None:
            deps = (self.operations[-1].id,) if self.operations else ()
        result = E2EWorkloadOperation.create(
            case_id=self.request.case_id,
            sequence_index=len(self.operations),
            kind=kind,
            phase=phase,
            step=step,
            layer=layer,
            expert=expert,
            parameter_ref=parameter_ref,
            reads=reads,
            writes=writes,
            input_value_refs=(
                *input_value_refs,
                *(value for state in reads for value in self.state_values[state]),
            ),
            output_value_refs=(
                *output_value_refs,
                *(value for state in writes for value in self.state_values[state]),
            ),
            group_refs=group_refs,
            reduce_op=reduce_op,
            normalization_denominator=normalization_denominator,
            deps=deps,
        )
        self.operations.append(result)
        return result

    def finish(self) -> E2EWorkloadGraph:
        producer: dict[str, str] = {}
        consumers: dict[str, list[str]] = {
            state.id: [] for state in self.states.values()
        }
        value_producer: dict[str, str] = {}
        value_consumers: dict[str, list[str]] = {
            value.id: [] for value in self.values.values()
        }
        for operation in self.operations:
            for state_id in operation.reads:
                consumers[state_id].append(operation.id)
            for state_id in operation.writes:
                if state_id in producer:
                    raise SchemaError(
                        "builder generated multiple state producers",
                        path="operations",
                    )
                producer[state_id] = operation.id
            for value_id in operation.input_value_refs:
                value_consumers[value_id].append(operation.id)
            for value_id in operation.output_value_refs:
                if value_id in value_producer:
                    raise SchemaError("builder generated multiple value producers", path="operations")
                value_producer[value_id] = operation.id
        states = tuple(
            replace(
                state,
                producer_op_id=producer.get(state.id),
                consumer_op_ids=tuple(consumers[state.id]),
            )
            for state in self.states.values()
        )
        values = tuple(
            replace(
                value,
                producer_op_id=value_producer.get(value.id),
                consumer_op_ids=tuple(value_consumers[value.id]),
            )
            for value in self.values.values()
        )
        return E2EWorkloadGraph.create(
            request=self.request,
            placement=self.placement,
            operations=tuple(self.operations),
            state_versions=states,
            tensor_values=values,
            route_traces=tuple(self.route_traces),
        )


def _parameter_specs(
    request: WorkloadRunRequest,
) -> tuple[tuple[str, int | None, int | None], ...]:
    specs: list[tuple[str, int | None, int | None]] = [
        ("embedding.weight", None, None)
    ]
    for layer in range(request.model.num_layers):
        prefix = f"layer.{layer}"
        specs.extend(
            (
                (f"{prefix}.input_norm.weight", layer, None),
                (f"{prefix}.qkv.weight", layer, None),
                (f"{prefix}.attention_out.weight", layer, None),
                (f"{prefix}.post_norm.weight", layer, None),
            )
        )
        if request.family.is_moe:
            specs.append((f"{prefix}.router.weight", layer, None))
            for expert in range(request.model.num_experts):
                specs.extend(
                    (
                        (f"{prefix}.expert.{expert}.gate.weight", layer, expert),
                        (f"{prefix}.expert.{expert}.up.weight", layer, expert),
                        (f"{prefix}.expert.{expert}.down.weight", layer, expert),
                    )
                )
        else:
            specs.extend(
                (
                    (f"{prefix}.mlp_gate.weight", layer, None),
                    (f"{prefix}.mlp_up.weight", layer, None),
                    (f"{prefix}.mlp_down.weight", layer, None),
                )
            )
    specs.extend((("final_norm.weight", None, None), ("lm_head.weight", None, None)))
    return tuple(specs)


def _parameter_states(
    builder: _GraphBuilder,
    specs: tuple[tuple[str, int | None, int | None], ...],
    version: int,
) -> dict[str, str]:
    return {
        name: builder.state(
            name,
            E2EStateKind.PARAMETER,
            version,
            layer=layer,
            expert=expert,
        )
        for name, layer, expert in specs
    }


def _adamw_states(
    builder: _GraphBuilder,
    specs: tuple[tuple[str, int | None, int | None], ...],
    version: int,
) -> dict[str, tuple[str, str, str, str]]:
    """Return exact FP32 master/m/v and INT32 step state per parameter."""

    return {
        name: (
            builder.state(
                f"optimizer.adamw.master.{name}",
                E2EStateKind.OPTIMIZER_MASTER,
                version,
                layer=layer,
                expert=expert,
            ),
            builder.state(
                f"optimizer.adamw.m.{name}",
                E2EStateKind.OPTIMIZER_MOMENT1,
                version,
                layer=layer,
                expert=expert,
            ),
            builder.state(
                f"optimizer.adamw.v.{name}",
                E2EStateKind.OPTIMIZER_MOMENT2,
                version,
                layer=layer,
                expert=expert,
            ),
            builder.state(
                f"optimizer.adamw.step.{name}",
                E2EStateKind.OPTIMIZER_STEP,
                version,
                layer=layer,
                expert=expert,
            ),
        )
        for name, layer, expert in specs
    }


def _activation(
    builder: _GraphBuilder,
    name: str,
    *,
    step: int,
    layer: int | None = None,
    expert: int | None = None,
    shape: tuple[int, ...] | None = None,
    kind: E2EStateKind = E2EStateKind.ACTIVATION,
    dtype: DType | None = None,
    owned: bool = False,
) -> str:
    return builder.state(
        name,
        kind,
        step,
        layer=layer,
        expert=expert,
        shape=shape or (_tokens(builder.request, step), builder.request.model.hidden_size),
        dtype=dtype,
        owned=owned,
    )


def _add_attention_forward(
    builder: _GraphBuilder,
    parameter_states: dict[str, str],
    *,
    phase: str,
    step: int,
    layer: int,
    kv_version: int | None,
) -> None:
    prefix = f"layer.{layer}"
    builder.add(
        E2EOperationKind.INPUT_NORM,
        phase=phase,
        step=step,
        layer=layer,
        parameter_ref=f"{prefix}.input_norm.weight",
        reads=(parameter_states[f"{prefix}.input_norm.weight"],),
    )
    builder.add(
        E2EOperationKind.QKV,
        phase=phase,
        step=step,
        layer=layer,
        parameter_ref=f"{prefix}.qkv.weight",
        reads=(parameter_states[f"{prefix}.qkv.weight"],),
    )
    builder.add(E2EOperationKind.ROPE, phase=phase, step=step, layer=layer)
    if kv_version is not None:
        current_kv = builder.state(
            f"layer.{layer}.kv",
            E2EStateKind.KV,
            kv_version,
            layer=layer,
        )
        builder.add(
            E2EOperationKind.KV_LOAD,
            phase=phase,
            step=step,
            layer=layer,
            reads=(current_kv,),
        )
    attention = _activation(
        builder,
        f"activation.{phase}.{step}.layer.{layer}.attention",
        step=step,
        layer=layer,
    )
    builder.add(
        E2EOperationKind.ATTENTION,
        phase=phase,
        step=step,
        layer=layer,
        writes=(attention,),
    )
    if kv_version is not None:
        next_kv = builder.state(
            f"layer.{layer}.kv",
            E2EStateKind.KV,
            kv_version + 1,
            layer=layer,
        )
        builder.add(
            E2EOperationKind.KV_APPEND,
            phase=phase,
            step=step,
            layer=layer,
            writes=(next_kv,),
        )
    builder.add(
        E2EOperationKind.ATTENTION_OUT,
        phase=phase,
        step=step,
        layer=layer,
        parameter_ref=f"{prefix}.attention_out.weight",
        reads=(parameter_states[f"{prefix}.attention_out.weight"],),
    )
    builder.add(E2EOperationKind.RESIDUAL, phase=phase, step=step, layer=layer)
    builder.add(
        E2EOperationKind.POST_NORM,
        phase=phase,
        step=step,
        layer=layer,
        parameter_ref=f"{prefix}.post_norm.weight",
        reads=(parameter_states[f"{prefix}.post_norm.weight"],),
    )


def _add_dense_mlp(
    builder: _GraphBuilder,
    parameter_states: dict[str, str],
    *,
    phase: str,
    step: int,
    layer: int,
) -> None:
    prefix = f"layer.{layer}"
    for parameter in ("mlp_gate.weight", "mlp_up.weight"):
        name = f"{prefix}.{parameter}"
        builder.add(
            E2EOperationKind.MLP_UP,
            phase=phase,
            step=step,
            layer=layer,
            parameter_ref=name,
            reads=(parameter_states[name],),
        )
    builder.add(
        E2EOperationKind.MLP_ACTIVATION,
        phase=phase,
        step=step,
        layer=layer,
    )
    name = f"{prefix}.mlp_down.weight"
    builder.add(
        E2EOperationKind.MLP_DOWN,
        phase=phase,
        step=step,
        layer=layer,
        parameter_ref=name,
        reads=(parameter_states[name],),
    )
    output = _activation(
        builder,
        f"activation.{phase}.{step}.layer.{layer}.dense_output",
        step=step,
        layer=layer,
    )
    builder.add(
        E2EOperationKind.RESIDUAL,
        phase=phase,
        step=step,
        layer=layer,
        writes=(output,),
    )


def _add_moe_forward(
    builder: _GraphBuilder,
    parameter_states: dict[str, str],
    *,
    phase: str,
    step: int,
    layer: int,
) -> None:
    prefix = f"layer.{layer}"
    router = f"{prefix}.router.weight"
    router_scores = _activation(
        builder,
        f"activation.{phase}.{step}.layer.{layer}.router_scores",
        step=step,
        layer=layer,
        shape=(_tokens(builder.request, step), builder.request.model.num_experts),
        dtype=DType.FP32,
    )
    builder.add(
        E2EOperationKind.ROUTER,
        phase=phase,
        step=step,
        layer=layer,
        parameter_ref=router,
        reads=(parameter_states[router],),
        writes=(router_scores,),
    )
    token_count = _tokens(builder.request, step)
    expert_by_token = tuple(
        token % builder.request.model.num_experts for token in range(token_count)
    )
    expert_counts = tuple(
        expert_by_token.count(expert)
        for expert in range(builder.request.model.num_experts)
    )
    route_state = _activation(
        builder,
        f"route.{phase}.{step}.layer.{layer}",
        step=step,
        layer=layer,
        shape=(token_count,),
        kind=E2EStateKind.ROUTE_TOKEN,
        dtype=DType.INT32,
    )
    builder.add(
        E2EOperationKind.ROUTE_FREEZE,
        phase=phase,
        step=step,
        layer=layer,
        reads=(router_scores,),
        writes=(route_state,),
    )
    dispatch_states = tuple(
        _activation(
            builder,
            f"dispatch.{phase}.{step}.layer.{layer}.expert.{expert}",
            step=step,
            layer=layer,
            expert=expert,
            shape=(expert_counts[expert], builder.request.model.hidden_size),
            kind=E2EStateKind.DISPATCH_PAYLOAD,
            owned=True,
        )
        for expert in range(builder.request.model.num_experts)
    )
    builder.add(
        E2EOperationKind.DISPATCH,
        phase=phase,
        step=step,
        layer=layer,
        reads=(route_state,),
        writes=dispatch_states,
    )
    expert_output_states: list[str] = []
    for expert in range(builder.request.model.num_experts):
        names = tuple(
            f"{prefix}.expert.{expert}.{projection}.weight"
            for projection in ("gate", "up", "down")
        )
        expert_output = _activation(
            builder,
            f"expert_output.{phase}.{step}.layer.{layer}.expert.{expert}",
            step=step,
            layer=layer,
            expert=expert,
            shape=(expert_counts[expert], builder.request.model.hidden_size),
            kind=E2EStateKind.DISPATCH_PAYLOAD,
            owned=True,
        )
        expert_output_states.append(expert_output)
        builder.add(
            E2EOperationKind.EXPERT_FORWARD,
            phase=phase,
            step=step,
            layer=layer,
            expert=expert,
            reads=(
                dispatch_states[expert],
                *(parameter_states[name] for name in names),
            ),
            writes=(expert_output,),
        )
    combine_state = _activation(
        builder,
        f"activation.{phase}.{step}.layer.{layer}.combine",
        step=step,
        layer=layer,
    )
    builder.add(
        E2EOperationKind.COMBINE,
        phase=phase,
        step=step,
        layer=layer,
        reads=tuple(expert_output_states),
        writes=(combine_state,),
    )
    builder.add(E2EOperationKind.RESIDUAL, phase=phase, step=step, layer=layer)
    builder.route_traces.append(
        E2ERouteTrace.create(
            case_id=builder.request.case_id,
            phase=phase,
            step=step,
            layer=layer,
            token_count=token_count,
            expert_by_token=expert_by_token,
            expert_token_counts=expert_counts,
            route_value_refs=builder.state_values[route_state],
            dispatch_value_refs=tuple(
                value
                for state in dispatch_states
                for value in builder.state_values[state]
            ),
            expert_output_value_refs=tuple(
                value
                for state in expert_output_states
                for value in builder.state_values[state]
            ),
            combine_value_refs=builder.state_values[combine_state],
        )
    )


def _add_forward(
    builder: _GraphBuilder,
    parameter_states: dict[str, str],
    *,
    phase: str,
    step: int,
    with_kv: bool,
) -> None:
    embedding = _activation(
        builder,
        f"activation.{phase}.{step}.embedding",
        step=step,
    )
    builder.add(
        E2EOperationKind.EMBEDDING,
        phase=phase,
        step=step,
        parameter_ref="embedding.weight",
        reads=(parameter_states["embedding.weight"],),
        writes=(embedding,),
    )
    for layer in range(builder.request.model.num_layers):
        _add_attention_forward(
            builder,
            parameter_states,
            phase=phase,
            step=step,
            layer=layer,
            kv_version=step if with_kv else None,
        )
        if builder.request.family.is_moe:
            _add_moe_forward(
                builder,
                parameter_states,
                phase=phase,
                step=step,
                layer=layer,
            )
        else:
            _add_dense_mlp(
                builder,
                parameter_states,
                phase=phase,
                step=step,
                layer=layer,
            )
    builder.add(
        E2EOperationKind.FINAL_NORM,
        phase=phase,
        step=step,
        parameter_ref="final_norm.weight",
        reads=(parameter_states["final_norm.weight"],),
    )
    builder.add(
        E2EOperationKind.LM_HEAD,
        phase=phase,
        step=step,
        parameter_ref="lm_head.weight",
        reads=(parameter_states["lm_head.weight"],),
    )
    logits = _activation(
        builder,
        f"logits.{phase}.{step}",
        step=step,
        shape=(_tokens(builder.request, step), builder.request.model.vocabulary_size),
        kind=E2EStateKind.LOGITS,
    )
    builder.add(
        E2EOperationKind.LOGITS,
        phase=phase,
        step=step,
        writes=(logits,),
    )


def _build_inference(
    request: WorkloadRunRequest,
    placement: ParallelPlacement,
) -> E2EWorkloadGraph:
    assert request.steps.inference is not None
    builder = _GraphBuilder(request, placement)
    parameters = _parameter_specs(request)
    parameter_states = _parameter_states(builder, parameters, 0)
    for step in range(request.steps.inference.decode_steps + 1):
        phase = "prefill" if step == 0 else "decode"
        _add_forward(
            builder,
            parameter_states,
            phase=phase,
            step=step,
            with_kv=True,
        )
    return builder.finish()


def _gradient_kind(parameter: str) -> E2EOperationKind:
    if ".expert." in parameter:
        return E2EOperationKind.EXPERT_GRADIENT
    if parameter.endswith(".router.weight"):
        return E2EOperationKind.ROUTER_GRADIENT
    return E2EOperationKind.WGRAD


def _build_training(
    request: WorkloadRunRequest,
    placement: ParallelPlacement,
) -> E2EWorkloadGraph:
    assert request.steps.training is not None
    builder = _GraphBuilder(request, placement)
    specs = _parameter_specs(request)
    for step in range(request.steps.training.step_count):
        parameter_states = _parameter_states(builder, specs, step)
        adamw_states = (
            _adamw_states(builder, specs, step)
            if request.optimizer is not None
            and request.optimizer.kind is WorkloadOptimizerKind.ADAMW
            else None
        )
        for name, layer, expert in specs:
            builder.add(
                E2EOperationKind.PARAMETER_LOAD,
                phase="train",
                step=step,
                layer=layer,
                expert=expert,
                parameter_ref=name,
                reads=(parameter_states[name],),
            )
            if adamw_states is not None:
                builder.add(
                    E2EOperationKind.OPTIMIZER_LOAD,
                    phase="optimizer_load",
                    step=step,
                    layer=layer,
                    expert=expert,
                    parameter_ref=name,
                    reads=adamw_states[name],
                )
        _add_forward(
            builder,
            parameter_states,
            phase="train_forward",
            step=step,
            with_kv=False,
        )
        loss_state = _activation(
            builder,
            f"loss.{step}",
            step=step,
            shape=(1,),
            kind=E2EStateKind.LOSS,
            dtype=DType.FP32,
        )
        builder.add(
            E2EOperationKind.LOSS,
            phase="loss",
            step=step,
            writes=(loss_state,),
        )
        for layer in reversed(range(request.model.num_layers)):
            if request.family.is_moe:
                builder.add(
                    E2EOperationKind.GRAD_DISPATCH,
                    phase="train_backward",
                    step=step,
                    layer=layer,
                )
                for expert in range(request.model.num_experts):
                    builder.add(
                        E2EOperationKind.EXPERT_BACKWARD,
                        phase="train_backward",
                        step=step,
                        layer=layer,
                        expert=expert,
                    )
                builder.add(
                    E2EOperationKind.DX_COMBINE,
                    phase="train_backward",
                    step=step,
                    layer=layer,
                )
                builder.add(
                    E2EOperationKind.SHARED_BACKWARD,
                    phase="train_backward",
                    step=step,
                    layer=layer,
                )
            else:
                backward_state = _activation(
                    builder,
                    f"activation.train_backward.{step}.layer.{layer}",
                    step=step,
                    layer=layer,
                    dtype=DType.FP32,
                )
                builder.add(
                    E2EOperationKind.DENSE_BACKWARD,
                    phase="train_backward",
                    step=step,
                    layer=layer,
                    writes=(backward_state,),
                )
        next_parameters = _parameter_states(builder, specs, step + 1)
        next_adamw_states = (
            _adamw_states(builder, specs, step + 1)
            if adamw_states is not None
            else None
        )
        stores: list[str] = []
        for name, layer, expert in specs:
            raw_gradient = builder.state(
                f"gradient.raw.{name}",
                E2EStateKind.GRADIENT,
                step,
                layer=layer,
                expert=expert,
            )
            builder.add(
                _gradient_kind(name),
                phase="train_backward",
                step=step,
                layer=layer,
                expert=expert,
                parameter_ref=name,
                writes=(raw_gradient,),
            )
            synced_gradient = builder.state(
                f"gradient.synced.{name}",
                E2EStateKind.GRADIENT,
                step,
                layer=layer,
                expert=expert,
            )
            sync_groups = tuple(
                owner.synchronization_group_id
                for owner in builder._owners(expert)
            )
            group_by_id = {group.id: group for group in placement.groups}
            normalization_denominator = len(group_by_id[sync_groups[0]].ranks)
            builder.add(
                E2EOperationKind.GRADIENT_SYNC,
                phase="gradient_sync",
                step=step,
                layer=layer,
                expert=expert,
                parameter_ref=name,
                reads=(raw_gradient,),
                writes=(synced_gradient,),
                group_refs=sync_groups,
                reduce_op=ReduceOp.SUM,
                normalization_denominator=normalization_denominator,
            )
            update_kind = (
                E2EOperationKind.ADAMW_UPDATE
                if adamw_states is not None
                else E2EOperationKind.SGD_UPDATE
            )
            optimizer_reads = adamw_states[name] if adamw_states is not None else ()
            optimizer_writes = (
                next_adamw_states[name] if next_adamw_states is not None else ()
            )
            builder.add(
                update_kind,
                phase="optimizer",
                step=step,
                layer=layer,
                expert=expert,
                parameter_ref=name,
                reads=(parameter_states[name], synced_gradient, *optimizer_reads),
                writes=(next_parameters[name], *optimizer_writes),
            )
            store = builder.add(
                E2EOperationKind.PARAMETER_STORE,
                phase="state_store",
                step=step,
                layer=layer,
                expert=expert,
                parameter_ref=name,
                reads=(next_parameters[name],),
            )
            stores.append(store.id)
            if next_adamw_states is not None:
                optimizer_store = builder.add(
                    E2EOperationKind.OPTIMIZER_STORE,
                    phase="optimizer_store",
                    step=step,
                    layer=layer,
                    expert=expert,
                    parameter_ref=name,
                    reads=next_adamw_states[name],
                )
                stores.append(optimizer_store.id)
        builder.add(
            E2EOperationKind.STEP_COMMIT,
            phase="step_commit",
            step=step,
            deps=tuple(stores),
        )
    return builder.finish()


def build_e2e_workload_graph(
    request: WorkloadRunRequest,
    placement: ParallelPlacement | None = None,
) -> E2EWorkloadGraph:
    """Build logical coverage only; lowering/runtime remain not materialized."""

    if type(request) is not WorkloadRunRequest:
        raise SchemaError("must be a WorkloadRunRequest", path="request")
    request.validate("request")
    if placement is None:
        placement = _placement(request)
    if type(placement) is not ParallelPlacement:
        raise SchemaError("must be a ParallelPlacement", path="placement")
    if request.model.num_layers < 2:
        raise UnsupportedFeatureError(
            "P3 coverage requires at least two model layers",
            path="request.model.num_layers",
        )
    if request.family.is_training:
        assert request.steps.training is not None
        if request.steps.training.step_count < 2:
            raise UnsupportedFeatureError(
                "P3 training coverage requires at least two steps",
                path="request.steps.training.step_count",
            )
        assert request.optimizer is not None
        return _build_training(request, placement)
    assert request.steps.inference is not None
    if request.steps.inference.prefill_tokens == 0:
        raise UnsupportedFeatureError(
            "P3 inference coverage requires a prefill",
            path="request.steps.inference.prefill_tokens",
        )
    if request.steps.inference.decode_steps < 2:
        raise UnsupportedFeatureError(
            "P3 inference coverage requires at least two decode steps",
            path="request.steps.inference.decode_steps",
        )
    return _build_inference(request, placement)


__all__ = ["build_e2e_workload_graph"]
