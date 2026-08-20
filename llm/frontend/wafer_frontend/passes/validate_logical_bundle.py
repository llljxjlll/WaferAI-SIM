"""Exact template-to-bundle provenance checks for the Dense naive MVP."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.experiment import InferOutput
from ..schema.ir0 import (
    AttentionWorkload,
    CollectiveKind,
    EffectKind,
    EmbeddingWorkload,
    GemmPartition,
    GemmWorkload,
    IR0,
    OpKind,
    RopeQkWorkload,
    GreedySampleWorkload,
    SampleRowSelection,
    SamplingMode,
)
from ..schema.logical import ExpandedIR0Bundle, IR0Template
from ..schema.persistent_state import StateKind
from .validate_fusion import FusionSemanticValidator
from .validate_ir0 import DenseIR0Validator


_LOCAL_NODE_SUFFIXES = (
    "norm1",
    "qkv",
    "rope",
    "attention",
    "o",
    "residual1",
    "norm2",
    "gate_up",
    "swiglu",
    "down",
    "residual2",
)
_DISTRIBUTED_NODE_SUFFIXES = (
    "norm1",
    "ag1",
    "qkv",
    "rope",
    "attention",
    "o",
    "rs1",
    "residual1",
    "norm2",
    "ag2",
    "gate_up",
    "swiglu",
    "down",
    "rs2",
    "residual2",
)
_LOCAL_VALUE_SUFFIXES = (
    "w_norm1",
    "w_norm2",
    "w_qkv",
    "w_o",
    "w_gate_up",
    "w_down",
    "norm1_out",
    "qkv_out",
    "qkv_rope",
    "attention_out",
    "o_out",
    "residual1_out",
    "norm2_out",
    "gate_up_out",
    "swiglu_out",
    "down_out",
    "output",
)
_DISTRIBUTED_VALUE_SUFFIXES = (
    "w_norm1",
    "w_norm2",
    "w_qkv",
    "w_o",
    "w_gate_up",
    "w_down",
    "norm1_out",
    "ag1_out",
    "qkv_out",
    "qkv_rope",
    "attention_out",
    "o_partial",
    "rs1_out",
    "residual1_out",
    "norm2_out",
    "ag2_out",
    "gate_up_out",
    "swiglu_out",
    "down_partial",
    "rs2_out",
    "output",
)


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


class DenseLogicalBundleValidator:
    """Prove that every profile graph is the exact expansion of a template."""

    @staticmethod
    def validate(
        template: IR0Template,
        bundle: ExpandedIR0Bundle,
        path: str = "expanded_ir0_bundle",
    ) -> None:
        template.validate("template")
        bundle.validate(path)
        if bundle.source_template_id != template.id:
            _fail("bundle source_template_id does not match the supplied template", f"{path}.source_template_id")
        if bundle.source_profiles != template.profiles:
            _fail("bundle source_profiles do not exactly match the supplied template", f"{path}.source_profiles")

        tp = template.instance.parallel.tp
        if (tp == 1 and template.instance.parallel.sp) or (
            tp > 1 and not template.instance.parallel.sp
        ):
            _fail("Dense expansion requires TP=1 without SP or TP>1 with SP", "template.instance.parallel.sp")

        graph_ids: set[str] = set()
        reference_node_ids: set[str] | None = None
        reference_value_ids: set[str] | None = None
        reference_parameter_ids: tuple[str, ...] | None = None
        seen_kv_identity_ids: set[str] = set()
        for index, (profile, entry) in enumerate(zip(template.profiles, bundle.entries)):
            entry_path = f"{path}.entries[{index}]"
            graph = entry.graph
            if entry.source_template_id != template.id:
                _fail("entry source_template_id does not match the supplied template", f"{entry_path}.source_template_id")
            if entry.profile_id != profile.profile_id or entry.weight != profile.weight:
                _fail("entry profile identity/weight does not match the template", entry_path)
            if graph.job is not template.job:
                _fail("graph job does not match the template", f"{entry_path}.graph.job")
            if graph.instances != (template.instance,):
                _fail("graph instances do not exactly match the template instance", f"{entry_path}.graph.instances")
            if graph.profile != profile.key:
                _fail("graph profile does not match the template entry", f"{entry_path}.graph.profile")
            if graph.id in graph_ids:
                _fail("profile graphs must have unique graph IDs", f"{entry_path}.graph.id")
            graph_ids.add(graph.id)

            DenseLogicalBundleValidator._validate_graph_structure(
                template, graph, path=f"{entry_path}.graph"
            )
            DenseIR0Validator.validate(graph, f"{entry_path}.graph")
            FusionSemanticValidator.validate(graph, f"{entry_path}.graph")

            request_count = (
                len(profile.exact_profile.requests)
                if profile.exact_profile is not None
                else 1
            )
            expected_state_count = (
                (6 * template.model.num_layers + 3)
                + 2 * template.model.num_layers * request_count
            ) * tp
            if (
                len(graph.persistent_states) != expected_state_count
                or len(graph.state_accesses) != expected_state_count
            ):
                _fail(
                    "Dense graph must contain exact block/global parameter and KV state per rank",
                    f"{entry_path}.graph.persistent_states",
                )
            if any(
                declaration.identity.kind is StateKind.OPTIMIZER_RESERVED
                for declaration in graph.persistent_states
            ):
                _fail(
                    "Dense infer expansion cannot invent optimizer reservations",
                    f"{entry_path}.graph.persistent_states",
                )
            parameter_ids = tuple(
                declaration.id
                for declaration in graph.persistent_states
                if declaration.identity.kind is StateKind.PARAMETER
            )
            kv_identity_ids = {
                declaration.identity.id
                for declaration in graph.persistent_states
                if declaration.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
            }
            if reference_parameter_ids is None:
                reference_parameter_ids = parameter_ids
            elif parameter_ids != reference_parameter_ids:
                _fail(
                    "parameter state IDs must remain stable across profiles",
                    f"{entry_path}.graph.persistent_states",
                )
            overlap = seen_kv_identity_ids.intersection(kv_identity_ids)
            if overlap:
                _fail(
                    "KV state identities must be profile-local",
                    f"{entry_path}.graph.persistent_states",
                )
            seen_kv_identity_ids.update(kv_identity_ids)
            node_ids = {node.id for node in graph.nodes}
            value_ids = {value.id for value in graph.values}
            if reference_node_ids is None:
                reference_node_ids = node_ids
                reference_value_ids = value_ids
            elif node_ids != reference_node_ids or value_ids != reference_value_ids:
                _fail(
                    "all profile graphs must use the same canonical node/value ID sets",
                    f"{entry_path}.graph",
                )

    @staticmethod
    def _validate_graph_structure(
        template: IR0Template,
        graph: IR0,
        *,
        path: str,
    ) -> None:
        instance_id = template.instance.id
        layers = template.model.num_layers
        tp = template.instance.parallel.tp
        distributed = tp > 1
        node_suffixes = (
            _DISTRIBUTED_NODE_SUFFIXES if distributed else _LOCAL_NODE_SUFFIXES
        )
        value_suffixes = (
            _DISTRIBUTED_VALUE_SUFFIXES if distributed else _LOCAL_VALUE_SUFFIXES
        )
        expected_node_ids = (
            f"{instance_id}.embedding",
            *(
                f"{instance_id}.layer{layer}.{suffix}"
                for layer in range(layers)
                for suffix in node_suffixes
            ),
            f"{instance_id}.final_norm",
            f"{instance_id}.lm_head",
            *(
                (f"{instance_id}.greedy_sample",)
                if template.infer_output is InferOutput.GREEDY_SAMPLE
                else ()
            ),
        )
        if tuple(node.id for node in graph.nodes) != expected_node_ids:
            _fail(
                "graph nodes must exactly match canonical per-layer IDs and order",
                f"{path}.nodes",
            )
        expected_value_ids = (
            f"{instance_id}.token_ids",
            f"{instance_id}.tok_embeddings.weight",
            f"{instance_id}.embedding_out",
            *(
                f"{instance_id}.layer{layer}.{suffix}"
                for layer in range(layers)
                for suffix in value_suffixes
            ),
            f"{instance_id}.final_norm.weight",
            f"{instance_id}.final_norm_out",
            f"{instance_id}.lm_head.weight",
            f"{instance_id}.logits",
            *(
                (f"{instance_id}.sampled_ids",)
                if template.infer_output is InferOutput.GREEDY_SAMPLE
                else ()
            ),
        )
        if tuple(value.id for value in graph.values) != expected_value_ids:
            _fail(
                "graph values must exactly match canonical per-layer IDs and order",
                f"{path}.values",
            )

        expected_collectives = 4 * layers if distributed else 0
        actual_collectives = sum(
            node.kind is OpKind.COLLECTIVE for node in graph.nodes
        )
        expected_candidates = 2 * layers if distributed else 0
        expected_edges = (17 * layers + 2) if distributed else (13 * layers + 2)
        if template.infer_output is InferOutput.GREEDY_SAMPLE:
            expected_edges += 1
        if actual_collectives != expected_collectives:
            _fail("graph collective count does not match TP/SP expansion", f"{path}.nodes")
        if len(graph.fusion_candidates) != expected_candidates:
            _fail("graph fusion candidate count does not match TP/SP expansion", f"{path}.fusion_candidates")
        if len(graph.edges) != expected_edges:
            _fail("graph edge count does not match the exact layer chain", f"{path}.edges")

        values = {value.id: value for value in graph.values}
        nodes = {node.id: node for node in graph.nodes}
        qkv_width = (
            template.model.num_heads + 2 * template.model.num_kv_heads
        ) * template.model.head_dim
        tokens = graph.profile.prefill_tokens + graph.profile.decode_tokens
        hidden = template.model.hidden_size
        intermediate = template.model.intermediate_size

        first_input_id = f"{instance_id}.embedding_out"
        first_input = values[first_input_id]
        if (
            first_input.producer != f"{instance_id}.embedding"
            or first_input.consumers
            != (
                f"{instance_id}.layer0.norm1",
                f"{instance_id}.layer0.residual1",
            )
            or first_input.shape != (tokens, hidden)
            or first_input.dtype is not template.model.dtype
        ):
            _fail("embedding output provenance/shape/consumers are not exact", f"{path}.values")

        embedding = nodes[f"{instance_id}.embedding"]
        embedding_work = embedding.workload
        if (
            type(embedding_work) is not EmbeddingWorkload
            or embedding.inputs
            != (
                f"{instance_id}.token_ids",
                f"{instance_id}.tok_embeddings.weight",
            )
            or embedding.outputs != (first_input_id,)
            or embedding_work.logical_table_shape
            != (template.model.vocab_size, hidden)
            or embedding_work.profile != graph.profile
        ):
            _fail("embedding topology/model/profile is not exact", f"{path}.nodes[0]")

        expected_candidate_ids: list[str] = []
        for layer in range(layers):
            prefix = f"{instance_id}.layer{layer}"
            layer_input_id = (
                first_input_id
                if layer == 0
                else f"{instance_id}.layer{layer - 1}.output"
            )
            layer_input = values[layer_input_id]
            expected_input_consumers = (
                f"{prefix}.norm1",
                f"{prefix}.residual1",
            )
            if layer_input.consumers != expected_input_consumers:
                _fail(
                    "previous layer output must directly feed next norm1 and residual1",
                    f"{path}.values",
                )
            norm1 = nodes[f"{prefix}.norm1"]
            residual1 = nodes[f"{prefix}.residual1"]
            if norm1.inputs != (
                layer_input_id,
                f"{prefix}.w_norm1",
            ) or residual1.inputs[0] != layer_input_id:
                _fail(
                    "layer norm1/residual1 must consume the exact previous layer output",
                    f"{path}.nodes",
                )
            if layer > 0 and f"{prefix}.input" in values:
                _fail("later layers cannot introduce a pseudo input", f"{path}.values")

            final_output = values[f"{prefix}.output"]
            expected_output_consumers = (
                (
                    f"{instance_id}.layer{layer + 1}.norm1",
                    f"{instance_id}.layer{layer + 1}.residual1",
                )
                if layer + 1 < layers
                else (f"{instance_id}.final_norm",)
            )
            if final_output.producer != f"{prefix}.residual2" or final_output.consumers != expected_output_consumers:
                _fail("layer output producer/next-layer consumers are not exact", f"{path}.values")

            weight_contracts = (
                ("w_norm1", (hidden,), "norm1"),
                ("w_norm2", (hidden,), "norm2"),
                ("w_qkv", (hidden, qkv_width), "qkv"),
                ("w_o", (hidden, hidden), "o"),
                ("w_gate_up", (hidden, 2 * intermediate), "gate_up"),
                ("w_down", (intermediate, hidden), "down"),
            )
            for suffix, shape, consumer_suffix in weight_contracts:
                weight = values[f"{prefix}.{suffix}"]
                if (
                    weight.producer is not None
                    or weight.shape != shape
                    or weight.dtype is not template.model.dtype
                    or weight.consumers != (f"{prefix}.{consumer_suffix}",)
                ):
                    _fail(
                        "weight producer/shape/dtype/consumer does not match the model",
                        f"{path}.values",
                    )

            expected_gemms = (
                ("qkv", (tokens, qkv_width, hidden), GemmPartition.COLUMN_PARALLEL),
                ("o", (tokens, hidden, hidden), GemmPartition.ROW_PARALLEL),
                ("gate_up", (tokens, 2 * intermediate, hidden), GemmPartition.COLUMN_PARALLEL),
                ("down", (tokens, hidden, intermediate), GemmPartition.ROW_PARALLEL),
            )
            if not distributed:
                expected_gemms = tuple(
                    (suffix, shape, GemmPartition.REPLICATED)
                    for suffix, shape, _partition in expected_gemms
                )
            for suffix, shape, partition in expected_gemms:
                node = nodes[f"{prefix}.{suffix}"]
                if (
                    not isinstance(node.workload, GemmWorkload)
                    or node.workload.logical_shape != shape
                    or node.workload.partition is not partition
                ):
                    _fail("GEMM shape/partition does not match the template", f"{path}.nodes")

            attention = nodes[f"{prefix}.attention"]
            rope = nodes[f"{prefix}.rope"]
            work = attention.workload
            if (
                type(rope.workload) is not RopeQkWorkload
                or rope.inputs != (f"{prefix}.qkv_out",)
                or rope.outputs != (f"{prefix}.qkv_rope",)
                or rope.workload.profile != graph.profile
                or (
                    rope.workload.num_heads,
                    rope.workload.num_kv_heads,
                    rope.workload.head_dim,
                    rope.workload.rotary_dim,
                    rope.workload.rope_theta,
                    rope.workload.max_position_embeddings,
                    rope.workload.dtype,
                )
                != (
                    template.model.num_heads,
                    template.model.num_kv_heads,
                    template.model.head_dim,
                    template.model.rotary_dim,
                    template.model.rope_theta,
                    template.model.max_position_embeddings,
                    template.model.dtype,
                )
                or attention.inputs != (f"{prefix}.qkv_rope",)
                or
                not isinstance(work, AttentionWorkload)
                or work.profile != graph.profile
                or work.exact_profile
                != next(
                    item.exact_profile
                    for item in template.profiles
                    if item.key == graph.profile
                )
                or (
                    work.hidden_size,
                    work.num_heads,
                    work.num_kv_heads,
                    work.head_dim,
                    work.dtype,
                )
                != (
                    hidden,
                    template.model.num_heads,
                    template.model.num_kv_heads,
                    template.model.head_dim,
                    template.model.dtype,
                )
                or attention.effects.kind is not EffectKind.STATEFUL
                or attention.effects.effect_token != f"kv_effect_layer_{layer}"
                or attention.effects.alias_set != f"kv_alias_layer_{layer}"
            ):
                _fail("attention model/profile/KV provenance does not match its layer", f"{path}.nodes")

            if distributed:
                expected_candidate_ids.extend(
                    (
                        f"{prefix}.candidate.o_rs1",
                        f"{prefix}.candidate.down_rs2",
                    )
                )

        if tuple(candidate.id for candidate in graph.fusion_candidates) != tuple(expected_candidate_ids):
            _fail("fusion candidate IDs/order are not the canonical per-layer set", f"{path}.fusion_candidates")

        final_norm = nodes[f"{instance_id}.final_norm"]
        lm_head = nodes[f"{instance_id}.lm_head"]
        expected_partition = (
            GemmPartition.SEQUENCE_PARALLEL_REPLICATED_WEIGHT
            if distributed
            else GemmPartition.REPLICATED
        )
        if (
            final_norm.inputs
            != (
                f"{instance_id}.layer{layers - 1}.output",
                f"{instance_id}.final_norm.weight",
            )
            or final_norm.outputs != (f"{instance_id}.final_norm_out",)
            or lm_head.inputs
            != (
                f"{instance_id}.final_norm_out",
                f"{instance_id}.lm_head.weight",
            )
            or lm_head.outputs != (f"{instance_id}.logits",)
            or type(lm_head.workload) is not GemmWorkload
            or lm_head.workload.logical_shape
            != (tokens, template.model.vocab_size, hidden)
            or lm_head.workload.partition is not expected_partition
        ):
            _fail("final norm/LM-head topology and partition are not exact", f"{path}.nodes")

        logits = values[f"{instance_id}.logits"]
        if template.infer_output is InferOutput.LOGITS:
            if logits.consumers:
                _fail("LOGITS output requires logits as the unique terminal", f"{path}.values")
        else:
            sample = nodes[f"{instance_id}.greedy_sample"]
            sampled = values[f"{instance_id}.sampled_ids"]
            if (
                type(sample.workload) is not GreedySampleWorkload
                or sample.inputs != (logits.id,)
                or sample.outputs != (sampled.id,)
                or logits.consumers != (sample.id,)
                or sampled.consumers
                or sample.workload.profile != graph.profile
                or sample.workload.mode is not SamplingMode.GREEDY
                or sample.workload.row_selection
                is not SampleRowSelection.LAST_PER_SEQUENCE
                or sample.workload.sample_count != graph.profile.num_seqs
                or sample.workload.logical_logits_shape
                != (tokens, template.model.vocab_size)
                or sample.workload.comparisons
                != graph.profile.num_seqs * (template.model.vocab_size - 1)
            ):
                _fail("GREEDY_SAMPLE requires the canonical sampling terminal", f"{path}.nodes")


__all__ = ["DenseLogicalBundleValidator"]
