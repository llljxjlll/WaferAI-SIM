"""Pure Stage 2 Dense forward analytic-oracle producer."""

from __future__ import annotations

from math import prod

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import DType, MeshAxisName, ProfileKey, TensorValue, validate_uint64
from ..schema.experiment import InferOutput
from ..schema.ir0 import (
    AttentionWorkload,
    CollectiveKind,
    CollectiveWorkload,
    EmbeddingWorkload,
    GemmWorkload,
    GreedySampleWorkload,
    IR0,
    OpKind,
    ResidualWorkload,
    RmsNormWorkload,
    RopeQkWorkload,
    SwiGluWorkload,
)
from ..schema.ir1 import IR1
from ..schema.logical import IR0Template
from ..schema.persistent_state import StateKind
from ..schema.stage2_dense_forward_oracle import (
    DenseAttentionMetrics,
    DenseCollectiveKindMetrics,
    DenseCollectiveMetrics,
    DenseEmbeddingMetrics,
    DenseForwardWorkMetrics,
    DenseGemmMetrics,
    DenseGemmOpMetrics,
    DenseGraphMetrics,
    DenseGreedyMetrics,
    DenseKvMetrics,
    DenseParameterMetrics,
    DenseResidualMetrics,
    DenseRmsNormMetrics,
    DenseRopeQkMetrics,
    DenseSwiGluMetrics,
    Stage2DenseForwardOracle,
)


def _unsupported(message: str, *, path: str) -> None:
    raise UnsupportedFeatureError(message, path=path)


def _gemm(*, count: int, m: int, n: int, k: int) -> DenseGemmOpMetrics:
    return DenseGemmOpMetrics(
        flops=count * 2 * m * n * k,
        memory_read_bytes=count * 2 * (m * k + k * n),
        memory_write_bytes=count * 2 * m * n,
    )


def _gemms(
    *,
    layers: int,
    m: int,
    hidden: int,
    intermediate: int,
    qkv: int,
    vocab: int,
    tp: int,
    rank: bool,
) -> DenseGemmMetrics:
    if not rank:
        return DenseGemmMetrics(
            qkv=_gemm(count=layers, m=m, n=qkv, k=hidden),
            attention_output=_gemm(count=layers, m=m, n=hidden, k=hidden),
            gate_up=_gemm(count=layers, m=m, n=2 * intermediate, k=hidden),
            down=_gemm(count=layers, m=m, n=hidden, k=intermediate),
            lm_head=_gemm(count=1, m=m, n=vocab, k=hidden),
        )
    return DenseGemmMetrics(
        qkv=_gemm(count=layers, m=m, n=qkv // tp, k=hidden),
        attention_output=_gemm(count=layers, m=m, n=hidden, k=hidden // tp),
        gate_up=_gemm(count=layers, m=m, n=2 * intermediate // tp, k=hidden),
        down=_gemm(count=layers, m=m, n=hidden, k=intermediate // tp),
        lm_head=_gemm(count=1, m=m // tp, n=vocab, k=hidden),
    )


def _work(
    template: IR0Template, profile: ProfileKey, *, tp: int, rank: bool
) -> DenseForwardWorkMetrics:
    model = template.model
    m = profile.prefill_tokens
    rank_tokens = m // tp if rank else m
    rank_heads = model.num_heads // tp if rank else model.num_heads
    rank_kv_heads = model.num_kv_heads // tp if rank else model.num_kv_heads
    qkv_width = (model.num_heads + 2 * model.num_kv_heads) * model.head_dim
    rank_qkv_width = (
        (rank_heads + 2 * rank_kv_heads) * model.head_dim if rank else qkv_width
    )
    pairs_per_layer = m * (m + 1) // 2
    pair_count = model.num_layers * pairs_per_layer
    softmax_elements = pair_count * rank_heads
    rms_rows = (2 * model.num_layers + 1) * rank_tokens
    rotated_elements = (
        model.num_layers
        * m
        * (rank_heads + rank_kv_heads)
        * model.rotary_dim
    )
    swiglu_elements = model.num_layers * m * model.intermediate_size
    residual_elements = 2 * model.num_layers * m * model.hidden_size
    if rank:
        swiglu_elements //= tp
        residual_elements //= tp
    if template.infer_output is InferOutput.GREEDY_SAMPLE:
        sample_count = profile.num_seqs
        comparisons = sample_count * (model.vocab_size - 1)
        greedy_read_bytes = sample_count * model.vocab_size * 2
        greedy_write_bytes = sample_count * 4
    else:
        sample_count = comparisons = greedy_read_bytes = greedy_write_bytes = 0
    return DenseForwardWorkMetrics(
        gemm=_gemms(
            layers=model.num_layers,
            m=m,
            hidden=model.hidden_size,
            intermediate=model.intermediate_size,
            qkv=qkv_width,
            vocab=model.vocab_size,
            tp=tp,
            rank=rank,
        ),
        embedding=DenseEmbeddingMetrics(
            rows=rank_tokens,
            vector_ops=0,
            sfu_ops=0,
            memory_read_bytes=rank_tokens * (4 + 2 * model.hidden_size),
            memory_write_bytes=2 * rank_tokens * model.hidden_size,
        ),
        rms_norm=DenseRmsNormMetrics(
            rows=rms_rows,
            vector_ops=rms_rows * (4 * model.hidden_size + 1),
            sfu_ops=0,
            memory_read_bytes=(2 * model.num_layers + 1)
            * 2
            * (rank_tokens * model.hidden_size + model.hidden_size),
            memory_write_bytes=2 * rms_rows * model.hidden_size,
        ),
        rope_qk=DenseRopeQkMetrics(
            rotated_elements=rotated_elements,
            vector_ops=3 * rotated_elements,
            sfu_ops=0,
            memory_read_bytes=2 * model.num_layers * m * rank_qkv_width,
            memory_write_bytes=2 * model.num_layers * m * rank_qkv_width,
        ),
        attention=DenseAttentionMetrics(
            query_key_pairs=pair_count,
            softmax_elements=softmax_elements,
            qk_matmul_flops=2 * softmax_elements * model.head_dim,
            value_matmul_flops=2 * softmax_elements * model.head_dim,
            vector_ops=2 * softmax_elements,
            sfu_ops=softmax_elements,
            activation_memory_read_bytes=2 * model.num_layers * m * rank_qkv_width,
            activation_memory_write_bytes=(
                2 * model.num_layers * m * model.hidden_size // tp
                if rank
                else 2 * model.num_layers * m * model.hidden_size
            ),
        ),
        swiglu=DenseSwiGluMetrics(
            output_elements=swiglu_elements,
            vector_ops=4 * swiglu_elements,
            sfu_ops=swiglu_elements,
            memory_read_bytes=4 * swiglu_elements,
            memory_write_bytes=2 * swiglu_elements,
        ),
        residual=DenseResidualMetrics(
            output_elements=residual_elements,
            vector_ops=residual_elements,
            sfu_ops=0,
            memory_read_bytes=4 * residual_elements,
            memory_write_bytes=2 * residual_elements,
        ),
        greedy=DenseGreedyMetrics(
            sample_count=sample_count,
            comparisons=comparisons,
            vector_ops=0,
            sfu_ops=0,
            memory_read_bytes=greedy_read_bytes,
            memory_write_bytes=greedy_write_bytes,
        ),
    )


def build_stage2_dense_forward_oracle(
    template: IR0Template,
    profile: ProfileKey,
    *,
    tp_degree: int,
) -> Stage2DenseForwardOracle:
    """Build a Dense oracle from source shapes, before any graph exists."""

    template.validate("template")
    profile.validate("profile")
    validate_uint64(tp_degree, "tp_degree")
    if not any(entry.key == profile for entry in template.profiles):
        raise SchemaError("profile is not present in template", path="profile")
    if tp_degree != template.instance.parallel.tp:
        raise SchemaError("must equal template.instance.parallel.tp", path="tp_degree")
    if profile.decode_tokens or not profile.prefill_tokens:
        _unsupported(
            "the first Dense forward oracle supports pure prefill only", path="profile"
        )
    if (
        profile.num_seqs != 1
        or profile.context_sum != profile.prefill_tokens
        or profile.context_max != profile.prefill_tokens
    ):
        _unsupported(
            "the first Dense forward oracle requires one exact causal prefill",
            path="profile",
        )
    model = template.model
    for field_name, value in (
        ("prefill_tokens", profile.prefill_tokens),
        ("num_heads", model.num_heads),
        ("num_kv_heads", model.num_kv_heads),
        ("intermediate_size", model.intermediate_size),
    ):
        if value % tp_degree:
            raise SchemaError("must divide evenly by TP", path=f"oracle.{field_name}")
    if template.infer_output is InferOutput.GREEDY_SAMPLE and tp_degree != 1:
        _unsupported(
            "TP-sharded greedy sampling is not implemented",
            path="template.infer_output",
        )

    hidden = model.hidden_size
    qkv = (model.num_heads + 2 * model.num_kv_heads) * model.head_dim
    per_layer_linear = (
        hidden * qkv
        + hidden * hidden
        + 3 * hidden * model.intermediate_size
    )
    replicated_elements_per_rank = (
        model.vocab_size * hidden
        + 2 * model.num_layers * hidden
        + hidden
        + hidden * model.vocab_size
    )
    placed_elements = (
        model.num_layers * per_layer_linear
        + tp_degree * replicated_elements_per_rank
    )
    unique_elements = model.parameter_elements()
    parameter_count = 6 * model.num_layers + 3
    collective_nodes_per_kind = 2 * model.num_layers if tp_degree > 1 else 0
    activation_bytes = 2 * profile.prefill_tokens * hidden
    if tp_degree > 1:
        rank_payload_per_node = activation_bytes * (tp_degree - 1) // tp_degree
        collective_kind = DenseCollectiveKindMetrics(
            node_count=collective_nodes_per_kind,
            logical_tensor_bytes_per_node=activation_bytes,
            rank_payload_bytes_total=collective_nodes_per_kind
            * rank_payload_per_node,
            group_payload_bytes_total=collective_nodes_per_kind
            * rank_payload_per_node
            * tp_degree,
        )
    else:
        collective_kind = DenseCollectiveKindMetrics(0, 0, 0, 0)
    logical_kv_write = (
        model.num_layers
        * 4
        * profile.prefill_tokens
        * model.num_kv_heads
        * model.head_dim
    )
    return Stage2DenseForwardOracle.create(
        source_template_id=template.id,
        profile=profile,
        tp_degree=tp_degree,
        infer_output=template.infer_output,
        parameters=DenseParameterMetrics(
            unique_tensor_count=parameter_count,
            unique_elements=unique_elements,
            unique_bytes=2 * unique_elements,
            placed_elements=placed_elements,
            placed_bytes=2 * placed_elements,
        ),
        graph=DenseGraphMetrics(
            node_count=11 * model.num_layers
            + 3
            + 2 * collective_nodes_per_kind
            + (1 if template.infer_output is InferOutput.GREEDY_SAMPLE else 0),
            collective_node_count=2 * collective_nodes_per_kind,
            parameter_declaration_count=parameter_count * tp_degree,
            kv_declaration_count=2 * model.num_layers * tp_degree,
        ),
        logical_work=_work(template, profile, tp=tp_degree, rank=False),
        rank_work=_work(template, profile, tp=tp_degree, rank=True),
        collectives=DenseCollectiveMetrics(
            all_gather=collective_kind,
            reduce_scatter=collective_kind,
        ),
        kv=DenseKvMetrics(
            logical_read_bytes=0,
            logical_write_bytes=logical_kv_write,
            rank_read_bytes=0,
            rank_write_bytes=logical_kv_write // tp_degree,
        ),
    )


_DTYPE_BYTES = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}


def _rank_shape(value: TensorValue, tp: int) -> tuple[int, ...]:
    return tuple(
        extent // tp if axis is MeshAxisName.TP else extent
        for extent, axis in zip(value.shape, value.sharding.dim_map)
    )


def _sum_gemm_metrics(
    nodes: tuple[object, ...], *, rank: bool
) -> DenseGemmMetrics:
    grouped: dict[str, list[DenseGemmOpMetrics]] = {
        "qkv": [],
        "attention_output": [],
        "gate_up": [],
        "down": [],
        "lm_head": [],
    }
    for node in nodes:
        if node.kind is not OpKind.GEMM:  # type: ignore[attr-defined]
            continue
        workload = node.workload  # type: ignore[attr-defined]
        assert type(workload) is GemmWorkload
        shape = workload.rank_shape if rank else workload.logical_shape
        m, n, k = shape
        metric = _gemm(count=1, m=m, n=n, k=k)
        node_id = node.id  # type: ignore[attr-defined]
        if node_id.endswith(".qkv"):
            key = "qkv"
        elif node_id.endswith(".o"):
            key = "attention_output"
        elif node_id.endswith(".gate_up"):
            key = "gate_up"
        elif node_id.endswith(".down"):
            key = "down"
        elif node_id.endswith(".lm_head"):
            key = "lm_head"
        else:
            raise SchemaError("unknown Dense GEMM family", path="graph.nodes")
        grouped[key].append(metric)

    def total(key: str) -> DenseGemmOpMetrics:
        items = grouped[key]
        if not items:
            raise SchemaError(f"missing {key} GEMM", path="graph.nodes")
        return DenseGemmOpMetrics(
            sum(item.flops for item in items),
            sum(item.memory_read_bytes for item in items),
            sum(item.memory_write_bytes for item in items),
        )

    return DenseGemmMetrics(
        qkv=total("qkv"),
        attention_output=total("attention_output"),
        gate_up=total("gate_up"),
        down=total("down"),
        lm_head=total("lm_head"),
    )


def _derive_work(
    nodes: tuple[object, ...],
    values: tuple[TensorValue, ...],
    *,
    tp: int,
    rank: bool,
) -> DenseForwardWorkMetrics:
    value_index = {value.id: value for value in values}

    def shape(value: TensorValue) -> tuple[int, ...]:
        return _rank_shape(value, tp) if rank else value.shape

    embedding_nodes = tuple(
        node for node in nodes if node.kind is OpKind.EMBEDDING  # type: ignore[attr-defined]
    )
    if len(embedding_nodes) != 1:
        raise SchemaError("requires exactly one embedding", path="graph.nodes")
    embedding = embedding_nodes[0]
    embedding_work = embedding.workload  # type: ignore[attr-defined]
    assert type(embedding_work) is EmbeddingWorkload
    embedding_rows = (
        embedding_work.rank_index_shape[0]
        if rank
        else embedding_work.logical_index_shape[0]
    )
    hidden = embedding_work.logical_table_shape[1]

    norm_rows = norm_vector = norm_read = norm_write = 0
    rope_rotated = rope_read = rope_write = 0
    attention_pairs = attention_softmax = attention_read = attention_write = 0
    attention_qk = attention_value = 0
    swiglu_elements = residual_elements = 0
    greedy_count = greedy_comparisons = greedy_read = greedy_write = 0
    kv_read = kv_write = 0
    for node in nodes:
        workload = node.workload  # type: ignore[attr-defined]
        if type(workload) is RmsNormWorkload:
            activation_shape = (
                workload.rank_activation_shape
                if rank
                else workload.logical_activation_shape
            )
            weight_shape = (
                workload.rank_weight_shape
                if rank
                else workload.logical_weight_shape
            )
            rows, width = activation_shape
            norm_rows += rows
            norm_vector += rows * (4 * width + 1)
            norm_read += 2 * (prod(activation_shape) + prod(weight_shape))
            norm_write += 2 * prod(
                workload.rank_output_shape if rank else workload.logical_output_shape
            )
        elif type(workload) is RopeQkWorkload:
            heads = workload.rank_num_heads if rank else workload.num_heads
            kv_heads = (
                workload.rank_num_kv_heads if rank else workload.num_kv_heads
            )
            input_shape = (
                workload.rank_input_shape if rank else workload.logical_input_shape
            )
            output_shape = (
                workload.rank_output_shape if rank else workload.logical_output_shape
            )
            rotated = input_shape[0] * (heads + kv_heads) * workload.rotary_dim
            rope_rotated += rotated
            rope_read += 2 * prod(input_shape)
            rope_write += 2 * prod(output_shape)
        elif type(workload) is AttentionWorkload:
            heads = workload.rank_num_heads if rank else workload.num_heads
            softmax = workload.query_key_pairs * heads
            attention_pairs += workload.query_key_pairs
            attention_softmax += softmax
            attention_qk += 2 * softmax * workload.head_dim
            attention_value += 2 * softmax * workload.head_dim
            input_value = value_index[node.inputs[0]]  # type: ignore[attr-defined]
            output_value = value_index[node.outputs[0]]  # type: ignore[attr-defined]
            attention_read += 2 * prod(shape(input_value))
            attention_write += 2 * prod(shape(output_value))
            kv_read += (
                workload.rank_kv_read_bytes
                if rank
                else workload.logical_kv_read_bytes
            )
            kv_write += (
                workload.rank_kv_write_bytes
                if rank
                else workload.logical_kv_write_bytes
            )
        elif type(workload) is SwiGluWorkload:
            swiglu_elements += prod(
                workload.rank_output_shape if rank else workload.logical_output_shape
            )
        elif type(workload) is ResidualWorkload:
            residual_elements += prod(
                workload.rank_shape if rank else workload.logical_shape
            )
        elif type(workload) is GreedySampleWorkload:
            greedy_count += workload.sample_count
            greedy_comparisons += workload.comparisons
            greedy_read += workload.sample_count * workload.logical_logits_shape[1] * 2
            greedy_write += workload.sample_count * 4

    result = DenseForwardWorkMetrics(
        gemm=_sum_gemm_metrics(nodes, rank=rank),
        embedding=DenseEmbeddingMetrics(
            embedding_rows,
            0,
            0,
            embedding_rows * (4 + 2 * hidden),
            2 * embedding_rows * hidden,
        ),
        rms_norm=DenseRmsNormMetrics(
            norm_rows, norm_vector, 0, norm_read, norm_write
        ),
        rope_qk=DenseRopeQkMetrics(
            rope_rotated,
            3 * rope_rotated,
            0,
            rope_read,
            rope_write,
        ),
        attention=DenseAttentionMetrics(
            attention_pairs,
            attention_softmax,
            attention_qk,
            attention_value,
            2 * attention_softmax,
            attention_softmax,
            attention_read,
            attention_write,
        ),
        swiglu=DenseSwiGluMetrics(
            swiglu_elements,
            4 * swiglu_elements,
            swiglu_elements,
            4 * swiglu_elements,
            2 * swiglu_elements,
        ),
        residual=DenseResidualMetrics(
            residual_elements,
            residual_elements,
            0,
            4 * residual_elements,
            2 * residual_elements,
        ),
        greedy=DenseGreedyMetrics(
            greedy_count,
            greedy_comparisons,
            0,
            0,
            greedy_read,
            greedy_write,
        ),
    )
    return result


def _derive_collectives(nodes: tuple[object, ...]) -> DenseCollectiveMetrics:
    def derive(kind: CollectiveKind) -> DenseCollectiveKindMetrics:
        workloads = tuple(
            node.workload  # type: ignore[attr-defined]
            for node in nodes
            if node.kind is OpKind.COLLECTIVE  # type: ignore[attr-defined]
            and type(node.workload) is CollectiveWorkload  # type: ignore[attr-defined]
            and node.workload.collective is kind  # type: ignore[attr-defined]
        )
        if not workloads:
            return DenseCollectiveKindMetrics(0, 0, 0, 0)
        logical_sizes = {work.logical_tensor_bytes for work in workloads}
        if len(logical_sizes) != 1:
            raise SchemaError(
                "collective logical tensor bytes must be uniform",
                path="graph.nodes",
            )
        return DenseCollectiveKindMetrics(
            len(workloads),
            next(iter(logical_sizes)),
            sum(work.rank_logical_payload_bytes for work in workloads),
            sum(work.group_logical_payload_bytes for work in workloads),
        )

    return DenseCollectiveMetrics(
        all_gather=derive(CollectiveKind.ALL_GATHER),
        reduce_scatter=derive(CollectiveKind.REDUCE_SCATTER),
    )


def _derive_from_graph(
    template: IR0Template,
    graph: IR0 | IR1,
    *,
    placed_parameter_bytes: int | None = None,
) -> Stage2DenseForwardOracle:
    tp = template.instance.parallel.tp
    if type(graph) is IR0:
        declarations = graph.persistent_states
    else:
        manifest = graph.persistent_state_manifest
        if manifest is None:
            raise SchemaError("IR1 requires persistent state manifest", path="ir1.persistent_state_manifest")
        declarations = manifest.declarations
    parameters = tuple(
        declaration
        for declaration in declarations
        if declaration.identity.kind is StateKind.PARAMETER
    )
    kv = tuple(
        declaration
        for declaration in declarations
        if declaration.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
    )
    parameter_refs = {item.identity.tensor_ref for item in parameters}
    if None in parameter_refs:
        raise SchemaError("parameter state requires tensor_ref", path="graph.persistent_states")
    tensor_refs = tuple(sorted(ref for ref in parameter_refs if ref is not None))
    values = {value.id: value for value in graph.values}
    unique_elements = sum(prod(values[ref].shape) for ref in tensor_refs if ref is not None)
    placed_elements = sum(prod(item.shape) for item in parameters)
    if placed_parameter_bytes is None:
        placed_parameter_bytes = sum(item.tensor_bytes for item in parameters)
    collective_count = sum(node.kind is OpKind.COLLECTIVE for node in graph.nodes)
    logical_work = _derive_work(graph.nodes, graph.values, tp=tp, rank=False)
    rank_work = _derive_work(graph.nodes, graph.values, tp=tp, rank=True)
    return Stage2DenseForwardOracle.create(
        source_template_id=template.id,
        profile=graph.profile,
        tp_degree=tp,
        infer_output=template.infer_output,
        parameters=DenseParameterMetrics(
            len(tensor_refs),
            unique_elements,
            sum(_DTYPE_BYTES[values[ref].dtype] * prod(values[ref].shape) for ref in tensor_refs if ref is not None),
            placed_elements,
            placed_parameter_bytes,
        ),
        graph=DenseGraphMetrics(
            len(graph.nodes), collective_count, len(parameters), len(kv)
        ),
        logical_work=logical_work,
        rank_work=rank_work,
        collectives=_derive_collectives(graph.nodes),
        kv=DenseKvMetrics(
            sum(
                node.workload.logical_kv_read_bytes
                for node in graph.nodes
                if type(node.workload) is AttentionWorkload
            ),
            sum(
                node.workload.logical_kv_write_bytes
                for node in graph.nodes
                if type(node.workload) is AttentionWorkload
            ),
            sum(
                node.workload.rank_kv_read_bytes
                for node in graph.nodes
                if type(node.workload) is AttentionWorkload
            ),
            sum(
                node.workload.rank_kv_write_bytes
                for node in graph.nodes
                if type(node.workload) is AttentionWorkload
            ),
        ),
    )


def _compare(actual: Stage2DenseForwardOracle, expected: Stage2DenseForwardOracle, path: str) -> None:
    for field_name in actual.__dataclass_fields__:
        if getattr(actual, field_name) != getattr(expected, field_name):
            raise SchemaError(
                "graph-derived metric does not match the oracle",
                path=f"{path}.{field_name}",
            )


def validate_stage2_oracle_against_ir0(
    oracle: Stage2DenseForwardOracle,
    template: IR0Template,
    graph: IR0,
    *,
    path: str,
) -> None:
    from .validate_logical_bundle import DenseLogicalBundleValidator
    from .validate_fusion import FusionSemanticValidator
    from .validate_ir0 import DenseIR0Validator

    oracle.validate_against_template(template, path=path)
    if type(graph) is not IR0:
        raise SchemaError("must be an IR0", path="graph")
    if not any(entry.key == graph.profile for entry in template.profiles):
        raise SchemaError("graph profile is absent from template", path="graph.profile")
    if (
        graph.job is not template.job
        or graph.instances != (template.instance,)
    ):
        raise SchemaError("graph does not preserve template ownership", path="graph")
    DenseLogicalBundleValidator._validate_graph_structure(
        template, graph, path="graph"
    )
    DenseIR0Validator.validate(graph, "graph")
    FusionSemanticValidator.validate(graph, "graph")
    _compare(_derive_from_graph(template, graph), oracle, path)


def validate_stage2_oracle_against_ir1(
    oracle: Stage2DenseForwardOracle,
    template: IR0Template,
    source: IR0,
    graph: IR1,
    *,
    path: str,
) -> None:
    validate_stage2_oracle_against_ir0(oracle, template, source, path=path)
    graph.validate("ir1")
    if graph.source_ir0_id != source.id:
        raise SchemaError("IR1 source_ir0_id does not match", path="ir1.source_ir0_id")
    if (
        graph.profile != source.profile
        or graph.values != source.values
        or graph.edges != source.edges
        or graph.fusion_candidates != source.fusion_candidates
        or graph.state_accesses != source.state_accesses
        or graph.fused_op_skeletons
        or graph.cross_routes
    ):
        raise SchemaError("IR1 does not exactly preserve IR0 graph provenance", path="ir1")
    if len(graph.nodes) != len(source.nodes):
        raise SchemaError("IR1 node coverage is not exact", path="ir1.nodes")
    group_by_owner = {
        (group.instance_id, group.mesh_ref): group for group in graph.groups
    }
    for physical, logical in zip(graph.nodes, source.nodes):
        expected_group = group_by_owner.get(
            (logical.instance_id, logical.mesh_ref)
        )
        if expected_group is None:
            raise SchemaError("IR1 node has no unique owner group", path="ir1.groups")
        actual = (
            physical.id,
            physical.origin_node_id,
            physical.instance_id,
            physical.kind,
            physical.phase,
            physical.stage,
            physical.mesh_ref,
            physical.execution_group_ref,
            physical.inputs,
            physical.outputs,
            physical.workload,
            physical.math,
            physical.effects,
            physical.impl_ref,
        )
        expected = (
            logical.id,
            logical.id,
            logical.instance_id,
            logical.kind,
            logical.phase,
            logical.stage,
            logical.mesh_ref,
            expected_group.id,
            logical.inputs,
            logical.outputs,
            logical.workload,
            logical.math,
            logical.effects,
            logical.impl_ref,
        )
        if actual != expected:
            raise SchemaError("IR1 physical node does not exactly preserve IR0", path="ir1.nodes")
    manifest = graph.persistent_state_manifest
    if manifest is None or manifest.declarations != source.persistent_states:
        raise SchemaError("IR1 manifest declarations do not match IR0", path="ir1.persistent_state_manifest")
    binding_by_state = {binding.state_ref: binding for binding in manifest.bindings}
    parameter_bytes = 0
    for declaration in manifest.declarations:
        binding = binding_by_state[declaration.id]
        group = group_by_owner[
            (declaration.identity.instance_ref, declaration.identity.mesh_ref)
        ]
        expected_die = next(
            placement.die_id
            for placement in group.placements
            if placement.rank == declaration.identity.shard_index
        )
        if binding.die_id != expected_die or binding.size_bytes != declaration.tensor_bytes:
            raise SchemaError(
                "IR1 HBM binding size/home does not match the state declaration",
                path="ir1.persistent_state_manifest.bindings",
            )
        if declaration.identity.kind is StateKind.PARAMETER:
            parameter_bytes += binding.size_bytes
    _compare(
        _derive_from_graph(
            template, graph, placed_parameter_bytes=parameter_bytes
        ),
        oracle,
        path,
    )


__all__ = [
    "build_stage2_dense_forward_oracle",
    "validate_stage2_oracle_against_ir0",
    "validate_stage2_oracle_against_ir1",
]
