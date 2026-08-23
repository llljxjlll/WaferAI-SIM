"""Independent analytic oracle for N6.1 forward-only dense training."""

from __future__ import annotations

from math import prod

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import DType, MeshAxisName
from ..schema.experiment import ExperimentSpec, WorkloadMode
from ..schema.ir0 import (
    AttentionMode,
    AttentionWorkload,
    CollectiveWorkload,
    CrossEntropyForwardWorkload,
    EmbeddingWorkload,
    GemmWorkload,
    IR0,
    OpKind,
    RmsNormWorkload,
    RopeQkWorkload,
)
from ..schema.persistent_state import StateKind
from ..schema.serde import canonical_digest
from ..schema.train_forward_oracle import (
    TrainForwardCeMetrics,
    TrainForwardCollectiveMetrics,
    TrainForwardGraphMetrics,
    TrainForwardOracle,
    TrainForwardParameterMetrics,
)


_DTYPE_BYTES = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}


def _train_inputs(spec: ExperimentSpec) -> tuple[object, object]:
    spec.validate("spec")
    if spec.workload.mode is not WorkloadMode.TRAIN:
        raise UnsupportedFeatureError(
            "train-forward oracle requires a TRAIN workload",
            path="spec.workload.mode",
        )
    train = spec.workload.train
    assert train is not None
    instance = spec.parallel.instances[0]
    if instance.sp != (instance.tp > 1):
        raise UnsupportedFeatureError(
            "N6.1 requires sequence parallel exactly when TP > 1",
            path="spec.parallel.instances[0].sp",
        )
    return train, instance


def build_train_forward_oracle(spec: ExperimentSpec) -> TrainForwardOracle:
    """Derive all metrics from the ExperimentSpec, before any graph exists."""

    train, instance = _train_inputs(spec)
    model = spec.model
    tp = instance.tp  # type: ignore[attr-defined]
    dp = instance.dp  # type: ignore[attr-defined]
    tokens = train.micro_batch * train.seq_len  # type: ignore[attr-defined]
    qkv = (model.NH + 2 * model.KVH) * model.DH
    per_layer_linear = (
        model.H * qkv + model.H * model.H + 3 * model.H * model.I
    )
    replicated_per_rank = (
        model.V * model.H
        + (2 * model.L + 1) * model.H
        + model.H * model.V
    )
    tp_placed_elements = model.L * per_layer_linear + tp * replicated_per_rank
    parameter_count = 6 * model.L + 3

    gemm_flops = (
        model.L
        * (
            2 * tokens * model.H * qkv
            + 2 * tokens * model.H * model.H
            + 4 * tokens * model.H * model.I
            + 2 * tokens * model.I * model.H
        )
        + 2 * tokens * model.V * model.H
    )
    pairs = model.L * train.micro_batch * train.seq_len * (train.seq_len + 1) // 2  # type: ignore[attr-defined]
    attention_flops = 4 * pairs * model.NH * model.DH
    logical_flops = gemm_flops + attention_flops

    collective_count = 4 * model.L if tp > 1 else 0
    tensor_bytes = 2 * tokens * model.H if tp > 1 else 0
    group_payload = collective_count * tensor_bytes * (tp - 1)
    rank_rows = tokens // tp if tp > 1 else tokens
    unique_elements = model.parameter_elements()
    return TrainForwardOracle.create(
        source_spec_digest=canonical_digest(spec),
        train_structure=train.structure,  # type: ignore[attr-defined]
        tp_degree=tp,
        dp_degree=dp,
        sequence_parallel=instance.sp,  # type: ignore[attr-defined]
        tokens_per_microbatch=tokens,
        parameters=TrainForwardParameterMetrics(
            unique_tensor_count=parameter_count,
            unique_elements=unique_elements,
            unique_bytes=2 * unique_elements,
            tp_placed_elements=tp_placed_elements,
            tp_placed_bytes=2 * tp_placed_elements,
            dp_replicated_bytes=2 * tp_placed_elements * dp,
        ),
        graph=TrainForwardGraphMetrics(
            node_count=(11 + (4 if tp > 1 else 0)) * model.L + 4,
            value_count=(18 + (4 if tp > 1 else 0)) * model.L + 7,
            edge_count=(13 + (4 if tp > 1 else 0)) * model.L + 3,
            fusion_candidate_count=4 * model.L if tp > 1 else 0,
            collective_node_count=collective_count,
            parameter_declaration_count=parameter_count * tp,
            kv_declaration_count=0,
        ),
        gemm_flops_per_microbatch=gemm_flops,
        attention_query_key_pairs_per_microbatch=pairs,
        attention_flops_per_microbatch=attention_flops,
        logical_forward_flops_per_microbatch=logical_flops,
        rank_forward_flops_per_microbatch=logical_flops // tp,
        cluster_forward_flops_per_step=(
            logical_flops * dp * train.structure.micro_batch_count  # type: ignore[attr-defined]
        ),
        collectives=TrainForwardCollectiveMetrics(
            node_count=collective_count,
            logical_tensor_bytes_per_node=tensor_bytes,
            group_payload_bytes_per_microbatch=group_payload,
            cluster_step_group_payload_bytes=(
                group_payload * dp * train.structure.micro_batch_count  # type: ignore[attr-defined]
            ),
        ),
        ce=TrainForwardCeMetrics(
            logical_rows=tokens,
            rank_rows=rank_rows,
            label_dtype=DType.INT32,
            loss_dtype=DType.FP32,
            logical_label_bytes=4 * tokens,
            rank_label_bytes=4 * rank_rows,
            logical_loss_bytes=4 * tokens,
            rank_loss_bytes=4 * rank_rows,
        ),
    )


def _validate_source_geometry(spec: ExperimentSpec, graph: IR0) -> None:
    train, source_instance = _train_inputs(spec)
    if graph.train != train.structure:  # type: ignore[attr-defined]
        raise SchemaError("does not preserve TrainStructure", path="graph.train")
    if len(graph.instances) != 1:
        raise SchemaError("requires one train instance", path="graph.instances")
    instance = graph.instances[0]
    if (
        instance.id != source_instance.id  # type: ignore[attr-defined]
        or instance.parallel.tp != source_instance.tp  # type: ignore[attr-defined]
        or instance.parallel.sp != source_instance.sp  # type: ignore[attr-defined]
        or instance.parallel.dp != source_instance.dp  # type: ignore[attr-defined]
        or instance.parallel.pp != source_instance.pp  # type: ignore[attr-defined]
        or instance.parallel.ep != source_instance.ep  # type: ignore[attr-defined]
    ):
        raise SchemaError("does not preserve train instance geometry", path="graph.instances[0]")
    tokens = train.micro_batch * train.seq_len  # type: ignore[attr-defined]
    if (
        graph.profile.prefill_tokens != tokens
        or graph.profile.decode_tokens != 0
        or graph.profile.num_seqs != train.micro_batch  # type: ignore[attr-defined]
        or graph.profile.context_sum != tokens
        or graph.profile.context_max != train.seq_len  # type: ignore[attr-defined]
        or graph.profile.kv_pages != 0
    ):
        raise SchemaError("does not preserve train token geometry", path="graph.profile")
    model = spec.model
    for index, node in enumerate(graph.nodes):
        workload = node.workload
        node_path = f"graph.nodes[{index}].workload"
        if type(workload) is EmbeddingWorkload and workload.logical_table_shape != (model.V, model.H):
            raise SchemaError("embedding geometry disagrees with model", path=node_path)
        if type(workload) is RmsNormWorkload and (
            workload.logical_weight_shape != (model.H,)
            or workload.epsilon != model.rms_norm_epsilon
        ):
            raise SchemaError("RMSNorm geometry disagrees with model", path=node_path)
        if type(workload) is RopeQkWorkload and (
            workload.num_heads,
            workload.num_kv_heads,
            workload.head_dim,
            workload.rotary_dim,
            workload.rope_theta,
            workload.max_position_embeddings,
        ) != (
            model.NH,
            model.KVH,
            model.DH,
            model.rotary_dim,
            model.rope_theta,
            model.max_position_embeddings,
        ):
            raise SchemaError("RoPE geometry disagrees with model", path=node_path)
        if type(workload) is AttentionWorkload and (
            workload.mode is not AttentionMode.TRAIN_FORWARD
            or workload.hidden_size != model.H
            or workload.num_heads != model.NH
            or workload.num_kv_heads != model.KVH
            or workload.head_dim != model.DH
        ):
            raise SchemaError("attention geometry disagrees with model", path=node_path)


def _derive_from_graph(spec: ExperimentSpec, graph: IR0) -> TrainForwardOracle:
    train, source_instance = _train_inputs(spec)
    tp = source_instance.tp  # type: ignore[attr-defined]
    dp = source_instance.dp  # type: ignore[attr-defined]
    values = {value.id: value for value in graph.values}
    parameters = tuple(
        state
        for state in graph.persistent_states
        if state.identity.kind is StateKind.PARAMETER
    )
    kv = tuple(
        state
        for state in graph.persistent_states
        if state.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
    )
    parameter_refs = {state.identity.tensor_ref for state in parameters}
    if None in parameter_refs:
        raise SchemaError("parameter tensor_ref is required", path="graph.persistent_states")
    refs = tuple(sorted(ref for ref in parameter_refs if ref is not None))
    unique_elements = sum(prod(values[ref].shape) for ref in refs)
    unique_bytes = sum(
        prod(values[ref].shape) * _DTYPE_BYTES[values[ref].dtype] for ref in refs
    )
    placed_elements = sum(prod(state.shape) for state in parameters)
    placed_bytes = sum(state.tensor_bytes for state in parameters)

    logical_gemm = rank_gemm = 0
    logical_pairs = logical_attention = rank_attention = 0
    collectives: list[CollectiveWorkload] = []
    ce_nodes: list[object] = []
    for node in graph.nodes:
        workload = node.workload
        if type(workload) is GemmWorkload:
            logical_gemm += 2 * prod(workload.logical_shape)
            rank_gemm += 2 * prod(workload.rank_shape)
        elif type(workload) is AttentionWorkload:
            logical_pairs += workload.query_key_pairs
            logical_attention += (
                4 * workload.query_key_pairs * workload.num_heads * workload.head_dim
            )
            rank_attention += (
                4 * workload.query_key_pairs * workload.rank_num_heads * workload.head_dim
            )
        elif type(workload) is CollectiveWorkload:
            collectives.append(workload)
        elif type(workload) is CrossEntropyForwardWorkload:
            ce_nodes.append(node)
    if len(ce_nodes) != 1:
        raise SchemaError("requires exactly one CE node", path="graph.nodes")
    ce_node = ce_nodes[0]
    ce_work = ce_node.workload  # type: ignore[attr-defined]
    labels = values[ce_node.inputs[1]]  # type: ignore[attr-defined]
    loss = values[ce_node.outputs[0]]  # type: ignore[attr-defined]
    logical_sizes = {work.logical_tensor_bytes for work in collectives}
    if len(logical_sizes) > 1:
        raise SchemaError("collective logical bytes must be uniform", path="graph.nodes")
    per_node = next(iter(logical_sizes), 0)
    group_payload = sum(work.group_logical_payload_bytes for work in collectives)
    logical_flops = logical_gemm + logical_attention
    rank_flops = rank_gemm + rank_attention
    return TrainForwardOracle.create(
        source_spec_digest=canonical_digest(spec),
        train_structure=train.structure,  # type: ignore[attr-defined]
        tp_degree=tp,
        dp_degree=dp,
        sequence_parallel=source_instance.sp,  # type: ignore[attr-defined]
        tokens_per_microbatch=graph.profile.prefill_tokens,
        parameters=TrainForwardParameterMetrics(
            len(refs),
            unique_elements,
            unique_bytes,
            placed_elements,
            placed_bytes,
            placed_bytes * dp,
        ),
        graph=TrainForwardGraphMetrics(
            len(graph.nodes),
            len(graph.values),
            len(graph.edges),
            len(graph.fusion_candidates),
            len(collectives),
            len(parameters),
            len(kv),
        ),
        gemm_flops_per_microbatch=logical_gemm,
        attention_query_key_pairs_per_microbatch=logical_pairs,
        attention_flops_per_microbatch=logical_attention,
        logical_forward_flops_per_microbatch=logical_flops,
        rank_forward_flops_per_microbatch=rank_flops,
        cluster_forward_flops_per_step=(
            logical_flops * dp * train.structure.micro_batch_count  # type: ignore[attr-defined]
        ),
        collectives=TrainForwardCollectiveMetrics(
            len(collectives),
            per_node,
            group_payload,
            group_payload * dp * train.structure.micro_batch_count,  # type: ignore[attr-defined]
        ),
        ce=TrainForwardCeMetrics(
            ce_work.logical_label_shape[0],
            ce_work.rank_label_shape[0],
            labels.dtype,
            loss.dtype,
            prod(labels.shape) * _DTYPE_BYTES[labels.dtype],
            prod(ce_work.rank_label_shape) * _DTYPE_BYTES[labels.dtype],
            prod(loss.shape) * _DTYPE_BYTES[loss.dtype],
            prod(ce_work.rank_loss_shape) * _DTYPE_BYTES[loss.dtype],
        ),
    )


def validate_train_forward_oracle_against_ir0(
    oracle: TrainForwardOracle,
    spec: ExperimentSpec,
    graph: IR0,
    *,
    path: str = "train_forward_oracle",
) -> None:
    from .validate_ir0 import DenseIR0Validator

    oracle.validate_against_spec(spec, path=path)
    if type(graph) is not IR0:
        raise SchemaError("must be an IR0", path="graph")
    DenseIR0Validator.validate(graph, "graph")
    _validate_source_geometry(spec, graph)
    actual = _derive_from_graph(spec, graph)
    if actual != oracle:
        for name in oracle.__dataclass_fields__:
            if getattr(actual, name) != getattr(oracle, name):
                raise SchemaError(
                    "graph-derived metric does not match the oracle",
                    path=f"{path}.{name}",
                )
        raise SchemaError("graph does not match oracle", path=path)


__all__ = [
    "build_train_forward_oracle",
    "validate_train_forward_oracle_against_ir0",
]
