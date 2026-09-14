"""Independent coverage and state-lineage oracle for P3 logical graphs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.e2e_workload_graph import (
    E2EArtifactStatus,
    E2EOperationKind,
    E2EStateKind,
    E2EStateVersion,
    E2EWorkloadGraph,
    E2EWorkloadOperation,
)
from ..schema.ir0 import ReduceOp
from ..schema.parallel_placement import ParameterOwnershipKind


_OperationKey = tuple[
    E2EOperationKind,
    str,
    int,
    int | None,
    int | None,
    str | None,
]


@dataclass(frozen=True, slots=True)
class E2ECoverageReport:
    family: str
    layer_count: int
    logical_step_count: int
    operation_count: int
    state_version_count: int
    tensor_value_count: int
    route_trace_count: int
    lowering_verified: bool = False
    runtime_verified: bool = False


def _parameters(
    graph: E2EWorkloadGraph,
) -> tuple[tuple[str, int | None, int | None], ...]:
    result: list[tuple[str, int | None, int | None]] = [
        ("embedding.weight", None, None)
    ]
    for layer in range(graph.request.model.num_layers):
        prefix = f"layer.{layer}"
        result.extend(
            (
                (f"{prefix}.input_norm.weight", layer, None),
                (f"{prefix}.qkv.weight", layer, None),
                (f"{prefix}.attention_out.weight", layer, None),
                (f"{prefix}.post_norm.weight", layer, None),
            )
        )
        if graph.request.family.is_moe:
            result.append((f"{prefix}.router.weight", layer, None))
            for expert in range(graph.request.model.num_experts):
                for projection in ("gate", "up", "down"):
                    result.append(
                        (
                            f"{prefix}.expert.{expert}.{projection}.weight",
                            layer,
                            expert,
                        )
                    )
        else:
            for projection in ("gate", "up", "down"):
                result.append(
                    (f"{prefix}.mlp_{projection}.weight", layer, None)
                )
    result.extend((("final_norm.weight", None, None), ("lm_head.weight", None, None)))
    return tuple(result)


def _tokens(graph: E2EWorkloadGraph, step: int) -> int:
    if graph.request.family.is_training:
        assert graph.request.steps.training is not None
        return (
            graph.request.steps.training.global_batch_size
            * graph.request.steps.training.sequence_length
        )
    assert graph.request.steps.inference is not None
    if step == 0:
        return (
            graph.request.steps.inference.request_count
            * graph.request.steps.inference.prefill_tokens
        )
    return graph.request.steps.inference.request_count


def _parameter_shape(graph: E2EWorkloadGraph, name: str) -> tuple[int, ...]:
    model = graph.request.model
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
    raise SchemaError("unknown parameter tensor", path="graph.tensor_values")


def _rank_shape(
    graph: E2EWorkloadGraph,
    shape: tuple[int, ...],
    *,
    preferred_axis: int | None = None,
) -> tuple[int, ...]:
    degree = graph.placement.tp_degree
    if degree == 1:
        return shape
    dimensions = list(shape)
    if preferred_axis is not None and dimensions[preferred_axis] % degree == 0:
        dimensions[preferred_axis] //= degree
        return tuple(dimensions)
    for axis in reversed(range(len(dimensions))):
        if dimensions[axis] % degree == 0:
            dimensions[axis] //= degree
            return tuple(dimensions)
    return shape


def _add(
    expected: Counter[_OperationKey],
    kind: E2EOperationKind,
    phase: str,
    step: int,
    layer: int | None = None,
    expert: int | None = None,
    parameter: str | None = None,
    count: int = 1,
) -> None:
    expected[(kind, phase, step, layer, expert, parameter)] += count


def _forward_expected(
    graph: E2EWorkloadGraph,
    expected: Counter[_OperationKey],
    *,
    phase: str,
    step: int,
    with_kv: bool,
) -> None:
    _add(
        expected,
        E2EOperationKind.EMBEDDING,
        phase,
        step,
        parameter="embedding.weight",
    )
    for layer in range(graph.request.model.num_layers):
        prefix = f"layer.{layer}"
        _add(
            expected,
            E2EOperationKind.INPUT_NORM,
            phase,
            step,
            layer,
            parameter=f"{prefix}.input_norm.weight",
        )
        _add(
            expected,
            E2EOperationKind.QKV,
            phase,
            step,
            layer,
            parameter=f"{prefix}.qkv.weight",
        )
        _add(expected, E2EOperationKind.ROPE, phase, step, layer)
        if with_kv:
            _add(expected, E2EOperationKind.KV_LOAD, phase, step, layer)
        _add(expected, E2EOperationKind.ATTENTION, phase, step, layer)
        if with_kv:
            _add(expected, E2EOperationKind.KV_APPEND, phase, step, layer)
        _add(
            expected,
            E2EOperationKind.ATTENTION_OUT,
            phase,
            step,
            layer,
            parameter=f"{prefix}.attention_out.weight",
        )
        _add(expected, E2EOperationKind.RESIDUAL, phase, step, layer)
        _add(
            expected,
            E2EOperationKind.POST_NORM,
            phase,
            step,
            layer,
            parameter=f"{prefix}.post_norm.weight",
        )
        if graph.request.family.is_moe:
            _add(
                expected,
                E2EOperationKind.ROUTER,
                phase,
                step,
                layer,
                parameter=f"{prefix}.router.weight",
            )
            _add(expected, E2EOperationKind.ROUTE_FREEZE, phase, step, layer)
            _add(expected, E2EOperationKind.DISPATCH, phase, step, layer)
            for expert in range(graph.request.model.num_experts):
                _add(
                    expected,
                    E2EOperationKind.EXPERT_FORWARD,
                    phase,
                    step,
                    layer,
                    expert,
                )
            _add(expected, E2EOperationKind.COMBINE, phase, step, layer)
        else:
            _add(
                expected,
                E2EOperationKind.MLP_UP,
                phase,
                step,
                layer,
                parameter=f"{prefix}.mlp_gate.weight",
            )
            _add(
                expected,
                E2EOperationKind.MLP_UP,
                phase,
                step,
                layer,
                parameter=f"{prefix}.mlp_up.weight",
            )
            _add(expected, E2EOperationKind.MLP_ACTIVATION, phase, step, layer)
            _add(
                expected,
                E2EOperationKind.MLP_DOWN,
                phase,
                step,
                layer,
                parameter=f"{prefix}.mlp_down.weight",
            )
        _add(expected, E2EOperationKind.RESIDUAL, phase, step, layer)
    _add(
        expected,
        E2EOperationKind.FINAL_NORM,
        phase,
        step,
        parameter="final_norm.weight",
    )
    _add(
        expected,
        E2EOperationKind.LM_HEAD,
        phase,
        step,
        parameter="lm_head.weight",
    )
    _add(expected, E2EOperationKind.LOGITS, phase, step)


def _expected_operations(graph: E2EWorkloadGraph) -> Counter[_OperationKey]:
    expected: Counter[_OperationKey] = Counter()
    if not graph.request.family.is_training:
        assert graph.request.steps.inference is not None
        for step in range(graph.request.steps.inference.decode_steps + 1):
            _forward_expected(
                graph,
                expected,
                phase="prefill" if step == 0 else "decode",
                step=step,
                with_kv=True,
            )
        return expected

    assert graph.request.steps.training is not None
    parameters = _parameters(graph)
    for step in range(graph.request.steps.training.step_count):
        for parameter, layer, expert in parameters:
            _add(
                expected,
                E2EOperationKind.PARAMETER_LOAD,
                "train",
                step,
                layer,
                expert,
                parameter,
            )
        _forward_expected(
            graph,
            expected,
            phase="train_forward",
            step=step,
            with_kv=False,
        )
        _add(expected, E2EOperationKind.LOSS, "loss", step)
        for layer in range(graph.request.model.num_layers):
            if graph.request.family.is_moe:
                _add(
                    expected,
                    E2EOperationKind.GRAD_DISPATCH,
                    "train_backward",
                    step,
                    layer,
                )
                for expert in range(graph.request.model.num_experts):
                    _add(
                        expected,
                        E2EOperationKind.EXPERT_BACKWARD,
                        "train_backward",
                        step,
                        layer,
                        expert,
                    )
                _add(
                    expected,
                    E2EOperationKind.DX_COMBINE,
                    "train_backward",
                    step,
                    layer,
                )
                _add(
                    expected,
                    E2EOperationKind.SHARED_BACKWARD,
                    "train_backward",
                    step,
                    layer,
                )
            else:
                _add(
                    expected,
                    E2EOperationKind.DENSE_BACKWARD,
                    "train_backward",
                    step,
                    layer,
                )
        for parameter, layer, expert in parameters:
            gradient_kind = E2EOperationKind.WGRAD
            if ".expert." in parameter:
                gradient_kind = E2EOperationKind.EXPERT_GRADIENT
            elif parameter.endswith(".router.weight"):
                gradient_kind = E2EOperationKind.ROUTER_GRADIENT
            _add(
                expected,
                gradient_kind,
                "train_backward",
                step,
                layer,
                expert,
                parameter,
            )
            _add(
                expected,
                E2EOperationKind.GRADIENT_SYNC,
                "gradient_sync",
                step,
                layer,
                expert,
                parameter,
            )
            _add(
                expected,
                E2EOperationKind.SGD_UPDATE,
                "optimizer",
                step,
                layer,
                expert,
                parameter,
            )
            _add(
                expected,
                E2EOperationKind.PARAMETER_STORE,
                "state_store",
                step,
                layer,
                expert,
                parameter,
            )
        _add(expected, E2EOperationKind.STEP_COMMIT, "step_commit", step)
    return expected


def _actual_operations(graph: E2EWorkloadGraph) -> Counter[_OperationKey]:
    return Counter(
        (
            operation.kind,
            operation.phase,
            operation.step,
            operation.layer,
            operation.expert,
            operation.parameter_ref,
        )
        for operation in graph.operations
    )


def _one_operation(
    graph: E2EWorkloadGraph,
    *,
    kind: E2EOperationKind,
    step: int,
    parameter: str | None = None,
    layer: int | None = None,
) -> E2EWorkloadOperation:
    matches = tuple(
        operation
        for operation in graph.operations
        if operation.kind is kind
        and operation.step == step
        and (parameter is None or operation.parameter_ref == parameter)
        and (layer is None or operation.layer == layer)
    )
    if len(matches) != 1:
        raise SchemaError(
            f"expected one {kind.value} operation, found {len(matches)}",
            path="e2e_workload_graph.operations",
        )
    return matches[0]


def _one_state(
    graph: E2EWorkloadGraph,
    *,
    logical_name: str,
    kind: E2EStateKind,
    version: int,
) -> E2EStateVersion:
    matches = tuple(
        state
        for state in graph.state_versions
        if state.logical_name == logical_name
        and state.kind is kind
        and state.version == version
    )
    if len(matches) != 1:
        raise SchemaError(
            f"expected one {kind.value} state version, found {len(matches)}",
            path="e2e_workload_graph.state_versions",
        )
    return matches[0]


def _happens_before(
    graph: E2EWorkloadGraph,
    source: E2EWorkloadOperation,
    target: E2EWorkloadOperation,
) -> bool:
    """Return whether target transitively depends on source."""

    dependencies = {operation.id: operation.deps for operation in graph.operations}
    pending = list(target.deps)
    visited: set[str] = set()
    while pending:
        operation_id = pending.pop()
        if operation_id == source.id:
            return True
        if operation_id in visited:
            continue
        visited.add(operation_id)
        pending.extend(dependencies[operation_id])
    return False


def _require_happens_before(
    graph: E2EWorkloadGraph,
    source: E2EWorkloadOperation,
    target: E2EWorkloadOperation,
    message: str,
) -> None:
    if not _happens_before(graph, source, target):
        raise SchemaError(message, path="graph.operations")


def _validate_inference_lineage(graph: E2EWorkloadGraph) -> None:
    assert graph.request.steps.inference is not None
    step_count = graph.request.steps.inference.decode_steps + 1
    for layer in range(graph.request.model.num_layers):
        name = f"layer.{layer}.kv"
        for version in range(step_count + 1):
            state = _one_state(
                graph,
                logical_name=name,
                kind=E2EStateKind.KV,
                version=version,
            )
            if version == 0:
                if state.producer_op_id is not None:
                    raise SchemaError(
                        "initial KV version must have no producer",
                        path="e2e_workload_graph.state_versions",
                    )
            else:
                append = _one_operation(
                    graph,
                    kind=E2EOperationKind.KV_APPEND,
                    step=version - 1,
                    layer=layer,
                )
                if state.producer_op_id != append.id or state.id not in append.writes:
                    raise SchemaError(
                        "KV append does not produce the next version",
                        path="e2e_workload_graph.state_versions",
                    )
            if version < step_count:
                load = _one_operation(
                    graph,
                    kind=E2EOperationKind.KV_LOAD,
                    step=version,
                    layer=layer,
                )
                if state.id not in load.reads:
                    raise SchemaError(
                        "KV load does not consume the current version",
                        path="e2e_workload_graph.operations",
                    )
                if version > 0:
                    previous_append = _one_operation(
                        graph,
                        kind=E2EOperationKind.KV_APPEND,
                        step=version - 1,
                        layer=layer,
                    )
                    _require_happens_before(
                        graph,
                        previous_append,
                        load,
                        "KV append must happen before the next decode KV load",
                    )
    for step in range(1, step_count):
        previous_logits = _one_operation(
            graph,
            kind=E2EOperationKind.LOGITS,
            step=step - 1,
        )
        next_embedding = _one_operation(
            graph,
            kind=E2EOperationKind.EMBEDDING,
            step=step,
        )
        _require_happens_before(
            graph,
            previous_logits,
            next_embedding,
            "prefill/decode steps must form a happens-before chain",
        )


def _validate_training_lineage(graph: E2EWorkloadGraph) -> None:
    assert graph.request.steps.training is not None
    step_count = graph.request.steps.training.step_count
    for step in range(step_count):
        loss = _one_operation(
            graph,
            kind=E2EOperationKind.LOSS,
            step=step,
        )
        backward = tuple(
            operation
            for operation in graph.operations
            if operation.step == step and operation.phase == "train_backward"
        )
        for operation in backward:
            _require_happens_before(
                graph,
                loss,
                operation,
                "loss must happen before every backward operation",
            )
        higher_anchor_kind = (
            E2EOperationKind.SHARED_BACKWARD
            if graph.request.family.is_moe
            else E2EOperationKind.DENSE_BACKWARD
        )
        lower_anchor_kind = (
            E2EOperationKind.GRAD_DISPATCH
            if graph.request.family.is_moe
            else E2EOperationKind.DENSE_BACKWARD
        )
        for layer in reversed(range(1, graph.request.model.num_layers)):
            higher_layer = _one_operation(
                graph,
                kind=higher_anchor_kind,
                step=step,
                layer=layer,
            )
            lower_layer = _one_operation(
                graph,
                kind=lower_anchor_kind,
                step=step,
                layer=layer - 1,
            )
            _require_happens_before(
                graph,
                higher_layer,
                lower_layer,
                "backward layers must follow reverse layer order",
            )
    for parameter, _, _ in _parameters(graph):
        for version in range(step_count + 1):
            state = _one_state(
                graph,
                logical_name=parameter,
                kind=E2EStateKind.PARAMETER,
                version=version,
            )
            if version == 0:
                if state.producer_op_id is not None:
                    raise SchemaError(
                        "initial parameter version must have no producer",
                        path="e2e_workload_graph.state_versions",
                    )
            else:
                update = _one_operation(
                    graph,
                    kind=E2EOperationKind.SGD_UPDATE,
                    step=version - 1,
                    parameter=parameter,
                )
                if state.producer_op_id != update.id or state.id not in update.writes:
                    raise SchemaError(
                        "optimizer does not produce the next parameter version",
                        path="e2e_workload_graph.state_versions",
                    )
            if version < step_count:
                load = _one_operation(
                    graph,
                    kind=E2EOperationKind.PARAMETER_LOAD,
                    step=version,
                    parameter=parameter,
                )
                if state.id not in load.reads:
                    raise SchemaError(
                        "parameter load reads a stale version",
                        path="e2e_workload_graph.operations",
                    )
                gradient_kind = E2EOperationKind.WGRAD
                if ".expert." in parameter:
                    gradient_kind = E2EOperationKind.EXPERT_GRADIENT
                elif parameter.endswith(".router.weight"):
                    gradient_kind = E2EOperationKind.ROUTER_GRADIENT
                gradient = _one_operation(
                    graph,
                    kind=gradient_kind,
                    step=version,
                    parameter=parameter,
                )
                sync = _one_operation(
                    graph,
                    kind=E2EOperationKind.GRADIENT_SYNC,
                    step=version,
                    parameter=parameter,
                )
                update = _one_operation(
                    graph,
                    kind=E2EOperationKind.SGD_UPDATE,
                    step=version,
                    parameter=parameter,
                )
                store = _one_operation(
                    graph,
                    kind=E2EOperationKind.PARAMETER_STORE,
                    step=version,
                    parameter=parameter,
                )
                raw = _one_state(
                    graph,
                    logical_name=f"gradient.raw.{parameter}",
                    kind=E2EStateKind.GRADIENT,
                    version=version,
                )
                synced = _one_state(
                    graph,
                    logical_name=f"gradient.synced.{parameter}",
                    kind=E2EStateKind.GRADIENT,
                    version=version,
                )
                next_parameter = _one_state(
                    graph,
                    logical_name=parameter,
                    kind=E2EStateKind.PARAMETER,
                    version=version + 1,
                )
                if (
                    raw.producer_op_id != gradient.id
                    or raw.id not in sync.reads
                    or synced.producer_op_id != sync.id
                    or synced.id not in update.reads
                    or state.id not in update.reads
                    or next_parameter.id not in update.writes
                    or next_parameter.id not in store.reads
                ):
                    raise SchemaError(
                        "parameter gradient/sync/update/store lineage is incomplete",
                        path="e2e_workload_graph.state_versions",
                    )
                _require_happens_before(
                    graph,
                    gradient,
                    sync,
                    "gradient must happen before gradient sync",
                )
                _require_happens_before(
                    graph,
                    sync,
                    update,
                    "gradient sync must happen before optimizer update",
                )
                _require_happens_before(
                    graph,
                    update,
                    store,
                    "optimizer update must happen before parameter store",
                )
                if version + 1 < step_count:
                    next_load = _one_operation(
                        graph,
                        kind=E2EOperationKind.PARAMETER_LOAD,
                        step=version + 1,
                        parameter=parameter,
                    )
                    _require_happens_before(
                        graph,
                        store,
                        next_load,
                        "parameter store must happen before next-step load",
                    )


def _expected_state_tensor(
    graph: E2EWorkloadGraph,
    state: E2EStateVersion,
) -> tuple[tuple[int, ...], DType, bool]:
    model = graph.request.model
    if state.kind is E2EStateKind.PARAMETER:
        return _parameter_shape(graph, state.logical_name), model.dtype, True
    if state.kind is E2EStateKind.GRADIENT:
        parameter = state.logical_name.removeprefix(
            "gradient.raw."
        ).removeprefix("gradient.synced.")
        return _parameter_shape(graph, parameter), DType.FP32, True
    if state.kind is E2EStateKind.KV:
        assert graph.request.steps.inference is not None
        cached_tokens = (
            0
            if state.version == 0
            else graph.request.steps.inference.prefill_tokens
            + (state.version - 1) * graph.request.steps.inference.request_count
        )
        return (
            graph.request.steps.inference.request_count,
            model.num_kv_heads,
            cached_tokens,
            model.head_dim,
        ), model.dtype, False
    if state.kind is E2EStateKind.LOSS:
        return (1,), DType.FP32, False
    if state.kind is E2EStateKind.LOGITS:
        return (_tokens(graph, state.version), model.vocabulary_size), model.dtype, False
    if state.kind is E2EStateKind.ROUTE_TOKEN:
        return (_tokens(graph, state.version),), DType.INT32, False
    if state.kind is E2EStateKind.DISPATCH_PAYLOAD:
        trace = next(
            item
            for item in graph.route_traces
            if item.step == state.version and item.layer == state.layer
        )
        assert state.expert is not None
        return (
            trace.expert_token_counts[state.expert],
            model.hidden_size,
        ), model.dtype, True
    if state.kind is E2EStateKind.ACTIVATION:
        width = (
            model.num_experts
            if state.logical_name.endswith("router_scores")
            else model.hidden_size
        )
        dtype = DType.FP32 if (
            state.logical_name.endswith("router_scores")
            or "train_backward" in state.logical_name
        ) else model.dtype
        return (_tokens(graph, state.version), width), dtype, False
    raise SchemaError("unsupported state tensor kind", path="graph.state_versions")


def _validate_tensor_values(graph: E2EWorkloadGraph) -> None:
    values_by_state: dict[str, list[object]] = {
        state.id: [] for state in graph.state_versions
    }
    for value in graph.tensor_values:
        values_by_state[value.state_ref].append(value)
    for state in graph.state_versions:
        values = values_by_state[state.id]
        if not values:
            raise SchemaError("state has no tensor values", path="graph.tensor_values")
        logical_shape, dtype, owned = _expected_state_tensor(graph, state)
        preferred_axis = 1 if state.kind is E2EStateKind.KV else None
        expected_shape = _rank_shape(
            graph,
            logical_shape,
            preferred_axis=preferred_axis,
        )
        if owned:
            owner_kind = (
                ParameterOwnershipKind.EXPERT
                if state.expert is not None
                else ParameterOwnershipKind.SHARED
            )
            owners = tuple(
                owner
                for owner in graph.placement.ownership_domains
                if owner.kind is owner_kind and owner.expert_id == state.expert
            )
            expected_bindings = {
                (rank, owner.tp_shard, owner.id)
                for owner in owners
                for rank in owner.replica_ranks
            }
        else:
            expected_bindings = {
                (rank.logical_rank, rank.coordinate.tp, None)
                for rank in graph.placement.rank_placements
            }
        actual_bindings = {
            (value.logical_rank, value.tp_shard, value.owner_domain_ref)
            for value in values
        }
        if actual_bindings != expected_bindings or len(values) != len(expected_bindings):
            raise SchemaError("tensor value owner/rank coverage is incomplete", path="graph.tensor_values")
        for value in values:
            if value.shape != expected_shape or value.dtype is not dtype:
                raise SchemaError("tensor value shape/dtype is incorrect", path="graph.tensor_values")
    values_by_id = {value.id: value for value in graph.tensor_values}
    for operation in graph.operations:
        expected_inputs = {
            value.id
            for state_ref in operation.reads
            for value in values_by_state[state_ref]
        }
        expected_outputs = {
            value.id
            for state_ref in operation.writes
            for value in values_by_state[state_ref]
        }
        if not expected_inputs.issubset(operation.input_value_refs):
            raise SchemaError("state reads lack tensor value inputs", path="graph.operations")
        if not expected_outputs.issubset(operation.output_value_refs):
            raise SchemaError("state writes lack tensor value outputs", path="graph.operations")
        if any(item not in values_by_id for item in operation.input_value_refs):
            raise SchemaError("operation reads an unknown tensor value", path="graph.operations")


def _validate_route_traces(graph: E2EWorkloadGraph) -> None:
    if not graph.request.family.is_moe:
        if graph.route_traces:
            raise SchemaError("Dense graph cannot carry route traces", path="graph.route_traces")
        return
    step_count = (
        graph.request.steps.training.step_count
        if graph.request.family.is_training
        else graph.request.steps.inference.decode_steps + 1
    )
    assert step_count is not None
    expected_keys = {
        (step, layer)
        for step in range(step_count)
        for layer in range(graph.request.model.num_layers)
    }
    if {(trace.step, trace.layer) for trace in graph.route_traces} != expected_keys:
        raise SchemaError("route traces do not cover every step/layer", path="graph.route_traces")
    values = {value.id: value for value in graph.tensor_values}
    for trace in graph.route_traces:
        expected_tokens = _tokens(graph, trace.step)
        expected_route = tuple(
            token % graph.request.model.num_experts
            for token in range(expected_tokens)
        )
        if trace.token_count != expected_tokens or trace.expert_by_token != expected_route:
            raise SchemaError("frozen token-to-expert trace is incorrect", path="graph.route_traces")
        if expected_tokens < graph.request.model.num_experts and not trace.zero_token_experts:
            raise SchemaError("trace must retain zero-token experts", path="graph.route_traces")
        for value_ref in trace.dispatch_value_refs:
            value = values[value_ref]
            if value.size_bytes != value.shape[0] * value.shape[1] * 2:
                raise SchemaError("dispatch payload bytes are incorrect", path="graph.route_traces")


def _validate_gradient_sync_groups(graph: E2EWorkloadGraph) -> None:
    if not graph.request.family.is_training:
        return
    group_by_id = {group.id: group for group in graph.placement.groups}
    for operation in graph.operations:
        if operation.kind is not E2EOperationKind.GRADIENT_SYNC:
            continue
        expert = operation.expert if operation.parameter_ref and ".expert." in operation.parameter_ref else None
        owner_kind = (
            ParameterOwnershipKind.EXPERT
            if expert is not None
            else ParameterOwnershipKind.SHARED
        )
        owners = tuple(
            owner
            for owner in graph.placement.ownership_domains
            if owner.kind is owner_kind and owner.expert_id == expert
        )
        expected_groups = tuple(owner.synchronization_group_id for owner in owners)
        if operation.group_refs != expected_groups or operation.reduce_op is not ReduceOp.SUM:
            raise SchemaError("gradient sync is not bound to its P1 SUM groups", path="graph.operations")
        group_sizes = {len(group_by_id[group].ranks) for group in expected_groups}
        if group_sizes != {operation.normalization_denominator}:
            raise SchemaError("gradient SUM-to-MEAN normalization is incorrect", path="graph.operations")


def validate_e2e_workload_coverage(
    graph: E2EWorkloadGraph,
) -> E2ECoverageReport:
    """Validate required semantics without calling the graph producer."""

    if type(graph) is not E2EWorkloadGraph:
        raise SchemaError("must be an E2EWorkloadGraph", path="graph")
    graph.request.validate("graph.request")
    expected = _expected_operations(graph)
    actual = _actual_operations(graph)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        if missing:
            key, count = next(iter(missing.items()))
            raise SchemaError(
                f"missing {count} required {key[0].value} operation(s)",
                path="graph.operations",
            )
        key, count = next(iter(extra.items()))
        raise SchemaError(
            f"contains {count} unexpected {key[0].value} operation(s)",
            path="graph.operations",
        )
    graph.validate("graph")
    _validate_tensor_values(graph)
    _validate_route_traces(graph)
    _validate_gradient_sync_groups(graph)
    if (
        graph.lowering_status is not E2EArtifactStatus.NOT_MATERIALIZED
        or graph.runtime_status is not E2EArtifactStatus.NOT_MATERIALIZED
    ):
        raise SchemaError(
            "coverage report cannot imply lowering/runtime success", path="graph"
        )
    if graph.request.family.is_training:
        _validate_training_lineage(graph)
        assert graph.request.steps.training is not None
        logical_steps = graph.request.steps.training.step_count
    else:
        _validate_inference_lineage(graph)
        assert graph.request.steps.inference is not None
        logical_steps = graph.request.steps.inference.decode_steps + 1
    return E2ECoverageReport(
        family=graph.family.value,
        layer_count=graph.request.model.num_layers,
        logical_step_count=logical_steps,
        operation_count=len(graph.operations),
        state_version_count=len(graph.state_versions),
        tensor_value_count=len(graph.tensor_values),
        route_trace_count=len(graph.route_traces),
    )


__all__ = ["E2ECoverageReport", "validate_e2e_workload_coverage"]
