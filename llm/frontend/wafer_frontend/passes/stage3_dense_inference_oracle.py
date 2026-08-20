"""Pure analytic oracle for exact Stage 3 static Dense inference profiles."""

from __future__ import annotations

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.experiment import InferOutput
from ..schema.ir0 import LogicalRole
from ..schema.logical import IR0Template
from ..schema.stage2_dense_forward_oracle import (
    DenseAttentionMetrics,
    DenseCollectiveKindMetrics,
    DenseCollectiveMetrics,
    DenseEmbeddingMetrics,
    DenseForwardWorkMetrics,
    DenseGraphMetrics,
    DenseGreedyMetrics,
    DenseParameterMetrics,
    DenseResidualMetrics,
    DenseRmsNormMetrics,
    DenseRopeQkMetrics,
    DenseSwiGluMetrics,
)
from ..schema.stage3_dense_inference_oracle import (
    Stage3DenseInferenceOracle,
    Stage3KvMetrics,
)
from ..schema.stage3_profile import Stage3ProfileMode, Stage3StaticProfile
from ..schema.common import validate_uint64
from .stage2_dense_forward_oracle import _gemms


def _unsupported(message: str, *, path: str) -> None:
    raise UnsupportedFeatureError(message, path=path)


def _work(
    template: IR0Template,
    profile: Stage3StaticProfile,
    *,
    tp: int,
    rank: bool,
) -> DenseForwardWorkMetrics:
    model = template.model
    m = profile.key.prefill_tokens + profile.key.decode_tokens
    rank_tokens = m // tp if rank else m
    rank_heads = model.num_heads // tp if rank else model.num_heads
    rank_kv_heads = model.num_kv_heads // tp if rank else model.num_kv_heads
    qkv_width = (model.num_heads + 2 * model.num_kv_heads) * model.head_dim
    rank_qkv_width = (
        (rank_heads + 2 * rank_kv_heads) * model.head_dim if rank else qkv_width
    )
    pairs_per_layer = profile.capacity.query_key_pairs
    pair_count = model.num_layers * pairs_per_layer
    softmax_elements = pair_count * rank_heads
    rms_rows = (2 * model.num_layers + 1) * rank_tokens
    rotated_elements = (
        model.num_layers
        * m
        * (rank_heads + rank_kv_heads)
        * model.rotary_dim
    )
    swiglu_elements = model.num_layers * rank_tokens * model.intermediate_size
    residual_elements = 2 * model.num_layers * rank_tokens * model.hidden_size
    if template.infer_output is InferOutput.GREEDY_SAMPLE:
        sample_count = profile.key.num_seqs
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
            activation_memory_read_bytes=2
            * model.num_layers
            * m
            * rank_qkv_width,
            activation_memory_write_bytes=2
            * model.num_layers
            * m
            * (model.hidden_size // tp if rank else model.hidden_size),
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


def build_stage3_dense_inference_oracle(
    template: IR0Template,
    static_profile: Stage3StaticProfile,
    *,
    tp_degree: int,
) -> Stage3DenseInferenceOracle:
    """Build Stage 3 metrics from model and exact request shapes only."""

    template.validate("template")
    static_profile.validate("static_profile")
    validate_uint64(tp_degree, "tp_degree")
    if tp_degree == 0:
        raise SchemaError("must be greater than zero", path="tp_degree")
    if tp_degree != template.instance.parallel.tp:
        raise SchemaError("must equal template TP", path="tp_degree")
    if not any(entry.key == static_profile.key for entry in template.profiles):
        raise SchemaError(
            "exact profile key is absent from template", path="static_profile.key"
        )
    role = template.instance.role
    if (
        static_profile.mode is Stage3ProfileMode.PREFILL
        and role not in (LogicalRole.PREFILL, LogicalRole.BOTH)
    ):
        raise SchemaError("prefill profile is incompatible with role", path="template.instance.role")
    if (
        static_profile.mode is Stage3ProfileMode.DECODE
        and role not in (LogicalRole.DECODE, LogicalRole.BOTH)
    ):
        raise SchemaError("decode profile is incompatible with role", path="template.instance.role")
    if static_profile.mode is Stage3ProfileMode.MIXED and role is not LogicalRole.BOTH:
        raise SchemaError("mixed profile requires role=both", path="template.instance.role")
    if static_profile.key.context_max > template.model.max_position_embeddings:
        raise SchemaError(
            "must not exceed max_position_embeddings",
            path="static_profile.key.context_max",
        )
    model = template.model
    m = static_profile.key.prefill_tokens + static_profile.key.decode_tokens
    for field_name, value in (
        ("query_tokens", m),
        ("num_heads", model.num_heads),
        ("num_kv_heads", model.num_kv_heads),
        ("intermediate_size", model.intermediate_size),
    ):
        if value % tp_degree:
            raise SchemaError(
                "must divide evenly by TP", path=f"oracle.{field_name}"
            )
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
    activation_bytes = 2 * m * hidden
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

    kv_bytes_per_token = 4 * model.num_kv_heads * model.head_dim
    logical_kv_read = (
        model.num_layers
        * static_profile.capacity.kv_read_tokens
        * kv_bytes_per_token
    )
    logical_kv_write = (
        model.num_layers
        * static_profile.capacity.kv_write_tokens
        * kv_bytes_per_token
    )
    page_size_tokens = static_profile.requests[0].kv_span.page_size_tokens
    reserved_tokens = sum(
        request.kv_span.capacity_tokens for request in static_profile.requests
    )
    logical_reserved = model.num_layers * reserved_tokens * kv_bytes_per_token
    return Stage3DenseInferenceOracle.create(
        source_template_id=template.id,
        static_profile=static_profile,
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
            kv_declaration_count=(
                2
                * model.num_layers
                * len(static_profile.requests)
                * tp_degree
            ),
        ),
        logical_work=_work(
            template, static_profile, tp=tp_degree, rank=False
        ),
        rank_work=_work(template, static_profile, tp=tp_degree, rank=True),
        collectives=DenseCollectiveMetrics(
            all_gather=collective_kind,
            reduce_scatter=collective_kind,
        ),
        kv=Stage3KvMetrics(
            page_count=static_profile.key.kv_pages,
            page_size_tokens=page_size_tokens,
            logical_read_bytes=logical_kv_read,
            logical_write_bytes=logical_kv_write,
            rank_read_bytes=logical_kv_read // tp_degree,
            rank_write_bytes=logical_kv_write // tp_degree,
            logical_reserved_bytes=logical_reserved,
            rank_reserved_bytes=logical_reserved // tp_degree,
        ),
    )


__all__ = ["build_stage3_dense_inference_oracle"]
