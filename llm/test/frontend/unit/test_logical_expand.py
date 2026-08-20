from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import prod
import unittest

from llm.frontend.wafer_frontend.errors import UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes import logical_expand as public_logical_expand
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.schema.common import MeshAxisName, ProfileKey
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import (
    AttentionMode,
    AttentionWorkload,
    CollectiveKind,
    CollectiveWorkload,
    EffectKind,
    GemmPartition,
    GemmWorkload,
    LogicalRole,
    OpKind,
    NumericalPolicy,
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.logical import (
    ExpandedIR0Bundle,
    IR0Template,
    ProfileEntry,
    SequenceParallelSpec,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)

from _fixtures import valid_spec


def decode(raw: dict[str, object]) -> ExperimentSpec:
    return from_data(ExperimentSpec, raw, path="spec")


def n2_template() -> IR0Template:
    raw = valid_spec()
    raw["parallel"]["instances"][0].update(tp=1, sp=False)  # type: ignore[index]
    return build_ir0(decode(raw))


def rebuild(template: IR0Template, **changes: object) -> IR0Template:
    fields = {
        "job": template.job,
        "model": template.model,
        "instance": template.instance,
        "sequence_parallel": template.sequence_parallel,
        "layer": template.layer,
        "infer_output": template.infer_output,
        "profiles": template.profiles,
    }
    fields.update(changes)
    return IR0Template.create(**fields)  # type: ignore[arg-type]


class LogicalExpandTest(unittest.TestCase):
    def test_tiny_current_block_state_and_attention_matrix_is_exact(self) -> None:
        expected_parameter_bytes = {1: 12448, 2: 14656, 4: 19072}
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                raw = valid_spec()
                raw["model"].update(  # type: ignore[index]
                    V=32,
                    H=16,
                    I=32,
                    NH=4,
                    KVH=4,
                    DH=4,
                    rotary_dim=4,
                    L=2,
                    max_position_embeddings=128,
                )
                raw["parallel"]["instances"][0].update(  # type: ignore[index]
                    tp=tp,
                    sp=tp > 1,
                )
                raw["workload"]["infer"]["profile"].update(  # type: ignore[index]
                    prefill_tokens=8,
                    decode_tokens=0,
                    num_seqs=1,
                    context_sum=8,
                    context_max=8,
                    kv_pages=1,
                )
                graph = logical_expand(build_ir0(decode(raw))).entries[0].graph
                parameters = tuple(
                    declaration
                    for declaration in graph.persistent_states
                    if declaration.identity.kind is StateKind.PARAMETER
                )
                self.assertEqual(len(parameters), 15 * tp)
                self.assertEqual(
                    sum(2 * prod(declaration.shape) for declaration in parameters),
                    expected_parameter_bytes[tp],
                )
                attention = tuple(
                    node.workload
                    for node in graph.nodes
                    if node.kind is OpKind.ATTENTION
                )
                self.assertEqual(len(attention), 2)
                self.assertTrue(
                    all(
                        work.mode is AttentionMode.PREFILL
                        and work.logical_kv_read_bytes == 0
                        and work.logical_kv_write_bytes == 512
                        and work.rank_kv_read_bytes == 0
                        and work.rank_kv_write_bytes == 512 // tp
                        for work in attention
                    )
                )
                declarations = {
                    declaration.id: declaration
                    for declaration in graph.persistent_states
                }
                kv_accesses = tuple(
                    access
                    for access in graph.state_accesses
                    if declarations[access.state_ref].identity.kind
                    in (StateKind.KV_KEY, StateKind.KV_VALUE)
                )
                self.assertEqual(len(kv_accesses), 4 * tp)
                self.assertTrue(
                    all(access.mode is StateAccessMode.WRITE for access in kv_accesses)
                )

    def test_complete_prefill_graph_is_stable_closed_and_roundtrips(self) -> None:
        self.assertIs(public_logical_expand, logical_expand)
        template = n2_template()
        before = canonical_digest(template)
        bundle = logical_expand(template)
        bundle.validate()
        self.assertEqual(canonical_digest(template), before)
        self.assertEqual(logical_expand(template), bundle)
        self.assertEqual(
            loads_dataclass(ExpandedIR0Bundle, canonical_json(bundle)), bundle
        )

        self.assertEqual(len(bundle.entries), 1)
        graph = bundle.entries[0].graph
        self.assertEqual(
            tuple(node.id.rsplit(".", 1)[-1] for node in graph.nodes),
            (
                "embedding",
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
                "final_norm",
                "lm_head",
            ),
        )
        self.assertEqual(len(graph.values), 24)
        self.assertEqual(len(graph.edges), 15)
        self.assertFalse(graph.fusion_candidates)
        self.assertFalse(any(node.kind is OpKind.COLLECTIVE for node in graph.nodes))

        value_index = {value.id.rsplit(".", 1)[-1]: value for value in graph.values}
        weights = tuple(value_index[name] for name in ("w_qkv", "w_o", "w_gate_up", "w_down"))
        self.assertTrue(all(value.producer is None for value in weights))
        self.assertEqual(
            tuple(value.shape for value in weights),
            ((256, 512), (256, 256), (256, 1024), (512, 256)),
        )
        self.assertEqual(
            sum(value.shape[0] * value.shape[1] for value in weights), 589824
        )
        for value in graph.values:
            self.assertEqual(value.sharding.dim_map, (None,) * len(value.shape))
            self.assertEqual(value.sharding.partial, ())
            self.assertEqual(value.sharding.mesh_ref, graph.instances[0].meshes[0].id)

        expected_edges = {
            (value.producer, consumer, value.id)
            for value in graph.values
            if value.producer is not None
            for consumer in value.consumers
        }
        self.assertEqual(
            {(edge.source_node, edge.destination_node, edge.value_id) for edge in graph.edges},
            expected_edges,
        )

    def test_exact_sample_work_and_attention_state(self) -> None:
        graph = logical_expand(n2_template()).entries[0].graph
        gemms = [node.workload for node in graph.nodes if node.kind is OpKind.GEMM]
        self.assertEqual(len(gemms), 5)
        self.assertTrue(all(isinstance(work, GemmWorkload) for work in gemms))
        self.assertTrue(all(work.partition is GemmPartition.REPLICATED for work in gemms))
        self.assertEqual(
            sum(2 * m * n * k for work in gemms for m, n, k in (work.logical_shape,)),
            46137344,
        )

        attention = next(node for node in graph.nodes if node.kind is OpKind.ATTENTION)
        self.assertIsInstance(attention.workload, AttentionWorkload)
        work = attention.workload
        self.assertEqual(work.query_key_pairs, 528)
        self.assertEqual(4 * work.query_key_pairs * work.num_heads * work.head_dim, 540672)
        self.assertEqual(work.query_key_pairs * work.num_heads, 2112)
        self.assertIs(attention.effects.kind, EffectKind.STATEFUL)
        self.assertEqual(attention.effects.effect_token, "kv_effect_layer_0")
        self.assertEqual(attention.effects.alias_set, "kv_alias_layer_0")
        self.assertEqual(
            tuple(node.impl_ref for node in graph.nodes),
            (
                "embedding_lookup",
                "rms_norm",
                "matmul_forward",
                "rope_qk_exact",
                "attention_forward",
                "matmul_forward",
                "residual",
                "rms_norm",
                "matmul_forward",
                "swiglu",
                "matmul_forward",
                "residual",
                "rms_norm",
                "matmul_forward",
            ),
        )

    def test_decode_uses_context_sum_for_query_key_pairs(self) -> None:
        raw = valid_spec()
        raw["parallel"]["instances"][0].update(  # type: ignore[index]
            tp=1, sp=False, role="decode"
        )
        raw["workload"]["infer"]["profile"] = {  # type: ignore[index]
            "prefill_tokens": 0,
            "decode_tokens": 32,
            "num_seqs": 32,
            "context_sum": 131072,
            "context_max": 4096,
            "kv_pages": 128,
            "expert_load": None,
        }
        template = build_ir0(decode(raw))
        bundle = logical_expand(template)
        graph = bundle.entries[0].graph
        graph.validate("graph")
        self.assertEqual((len(graph.nodes), len(graph.values), len(graph.edges)), (14, 24, 15))
        attention = next(
            node for node in graph.nodes if node.kind is OpKind.ATTENTION
        )
        work = attention.workload
        self.assertIsInstance(work, AttentionWorkload)
        self.assertIs(work.mode, AttentionMode.DECODE)
        self.assertEqual(work.query_tokens, 32)
        self.assertEqual(work.query_key_pairs, 131072)
        self.assertEqual(work.logical_kv_read_bytes, 67108864)
        self.assertEqual(work.logical_kv_write_bytes, 16384)
        kv_states = tuple(
            state
            for state in graph.persistent_states
            if state.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        self.assertEqual(len(kv_states), 2)
        self.assertEqual(
            tuple(state.shape for state in kv_states),
            ((131072, 2, 64), (131072, 2, 64)),
        )
        kv_refs = {state.id for state in kv_states}
        self.assertEqual(
            {
                access.mode
                for access in graph.state_accesses
                if access.state_ref in kv_refs
            },
            {StateAccessMode.READ_WRITE},
        )

    def test_direct_ambiguous_profiles_fail_closed(self) -> None:
        base = n2_template()

        def with_profile(key: ProfileKey, *, role: LogicalRole) -> IR0Template:
            return rebuild(
                base,
                instance=replace(base.instance, role=role),
                profiles=(ProfileEntry.create(key=key, weight=1.0),),
            )

        mixed = with_profile(
            ProfileKey(16, 1, 1, 16, 16, 1, None),
            role=LogicalRole.BOTH,
        )
        with self.assertRaisesRegex(
            UnsupportedFeatureError, "mixed prefill/decode"
        ):
            logical_expand(mixed)

        zero_context_decode = with_profile(
            ProfileKey(0, 1, 1, 0, 0, 1, None),
            role=LogicalRole.DECODE,
        )
        with self.assertRaisesRegex(UnsupportedFeatureError, "context_sum"):
            logical_expand(zero_context_decode)

        ambiguous_prefill = with_profile(
            ProfileKey(32, 0, 1, 64, 64, 1, None),
            role=LogicalRole.PREFILL,
        )
        with self.assertRaisesRegex(UnsupportedFeatureError, "prefill requires"):
            logical_expand(ambiguous_prefill)

    def test_tp2_sp_graph_and_accounting_are_exact(self) -> None:
        graph = logical_expand(build_ir0(decode(valid_spec()))).entries[0].graph
        graph.validate()
        self.assertEqual(len(graph.nodes), 18)
        self.assertEqual(len(graph.values), 28)
        self.assertEqual(len(graph.edges), 19)
        self.assertEqual(
            tuple(node.id.rsplit(".", 1)[-1] for node in graph.nodes),
            (
                "embedding", "norm1", "ag1", "qkv", "rope", "attention", "o", "rs1",
                "residual1", "norm2", "ag2", "gate_up", "swiglu",
                "down", "rs2", "residual2", "final_norm", "lm_head",
            ),
        )
        gemms = [node.workload for node in graph.nodes if node.kind is OpKind.GEMM]
        self.assertEqual(
            sum(2 * m * n * k for work in gemms for m, n, k in (work.rank_shape,)),
            23068672,
        )
        collectives = [
            node.workload
            for node in graph.nodes
            if node.kind is OpKind.COLLECTIVE
        ]
        self.assertEqual(len(collectives), 4)
        self.assertTrue(all(isinstance(work, CollectiveWorkload) for work in collectives))
        self.assertEqual(
            tuple(work.collective for work in collectives),
            (
                CollectiveKind.ALL_GATHER,
                CollectiveKind.REDUCE_SCATTER,
                CollectiveKind.ALL_GATHER,
                CollectiveKind.REDUCE_SCATTER,
            ),
        )
        self.assertTrue(
            all(
                (work.logical_tensor_bytes, work.rank_logical_payload_bytes,
                 work.group_logical_payload_bytes) == (16384, 8192, 16384)
                for work in collectives
            )
        )
        self.assertEqual(sum(work.rank_logical_payload_bytes for work in collectives), 32768)
        self.assertEqual(sum(work.group_logical_payload_bytes for work in collectives), 65536)
        self.assertEqual(len(graph.fusion_candidates), 2)
        self.assertTrue(
            all(
                candidate.semantic_contract.numerical_policy
                is graph.nodes[0].math.numerical_policy
                for candidate in graph.fusion_candidates
            )
        )
        values = {value.id.rsplit(".", 1)[-1]: value for value in graph.values}
        for name in (
            "embedding_out", "norm1_out", "rs1_out", "residual1_out", "norm2_out",
            "rs2_out", "output",
        ):
            self.assertEqual(values[name].sharding.dim_map, (MeshAxisName.TP, None))
            self.assertEqual(values[name].sharding.partial, ())
        for name in ("ag1_out", "ag2_out"):
            self.assertEqual(values[name].sharding.dim_map, (None, None))
            self.assertEqual(values[name].sharding.partial, ())
        for name in ("qkv_out", "qkv_rope", "attention_out", "gate_up_out", "swiglu_out"):
            self.assertEqual(values[name].sharding.dim_map, (None, MeshAxisName.TP))
            self.assertEqual(values[name].sharding.partial, ())
        for name in ("o_partial", "down_partial"):
            self.assertEqual(values[name].sharding.dim_map, (None, None))
            self.assertEqual(values[name].sharding.partial, (MeshAxisName.TP,))
        self.assertEqual(values["w_qkv"].sharding.dim_map, (None, MeshAxisName.TP))
        self.assertEqual(values["w_gate_up"].sharding.dim_map, (None, MeshAxisName.TP))
        self.assertEqual(values["w_o"].sharding.dim_map, (MeshAxisName.TP, None))
        self.assertEqual(values["w_down"].sharding.dim_map, (MeshAxisName.TP, None))

        nodes = {node.id.rsplit(".", 1)[-1]: node for node in graph.nodes}
        self.assertEqual(values["o_partial"].consumers, (nodes["rs1"].id,))
        self.assertEqual(values["down_partial"].consumers, (nodes["rs2"].id,))
        self.assertIn(values["rs1_out"].id, nodes["residual1"].inputs)
        self.assertNotIn(values["o_partial"].id, nodes["residual1"].inputs)
        self.assertIn(values["rs2_out"].id, nodes["residual2"].inputs)
        self.assertNotIn(values["down_partial"].id, nodes["residual2"].inputs)

        expected_candidates = (
            (
                (nodes["o"].id, nodes["rs1"].id),
                (values["attention_out"].id, values["w_o"].id),
                (values["rs1_out"].id,),
            ),
            (
                (nodes["down"].id, nodes["rs2"].id),
                (values["swiglu_out"].id, values["w_down"].id),
                (values["rs2_out"].id,),
            ),
        )
        self.assertEqual(
            tuple(
                (candidate.members, candidate.boundary_inputs, candidate.boundary_outputs)
                for candidate in graph.fusion_candidates
            ),
            expected_candidates,
        )
        for candidate in graph.fusion_candidates:
            self.assertEqual(candidate.semantic_contract.tile_domain, ("M", "N"))
            self.assertEqual(candidate.semantic_contract.reduction_axes, (2,))
            self.assertIs(
                candidate.semantic_contract.numerical_policy,
                NumericalPolicy.BITWISE,
            )

    def test_tp4_sp_rank_shapes_and_payloads_scale_exactly(self) -> None:
        raw = valid_spec()
        raw["parallel"]["instances"][0].update(tp=4, sp=True)  # type: ignore[index]
        raw["model"]["KVH"] = 4  # type: ignore[index]
        graph = logical_expand(build_ir0(decode(raw))).entries[0].graph
        graph.validate()

        gemms = [node.workload for node in graph.nodes if node.kind is OpKind.GEMM]
        self.assertEqual(len(gemms), 5)
        for work in gemms:
            logical_flops = 2
            rank_flops = 2
            for logical_extent, rank_extent in zip(work.logical_shape, work.rank_shape):
                logical_flops *= logical_extent
                rank_flops *= rank_extent
            self.assertEqual(rank_flops * 4, logical_flops)

        attention = next(node for node in graph.nodes if node.kind is OpKind.ATTENTION)
        self.assertEqual(attention.workload.rank_num_heads, 1)
        self.assertEqual(attention.workload.rank_num_kv_heads, 1)
        collectives = [
            node.workload for node in graph.nodes if node.kind is OpKind.COLLECTIVE
        ]
        self.assertTrue(
            all(
                (work.participant_count, work.logical_tensor_bytes,
                 work.rank_logical_payload_bytes, work.group_logical_payload_bytes)
                == (4, 16384, 12288, 49152)
                for work in collectives
            )
        )

    def test_tp2_two_layers_are_directly_chained_and_kv_is_isolated(self) -> None:
        raw = valid_spec()
        raw["model"]["L"] = 2  # type: ignore[index]
        graph = logical_expand(build_ir0(decode(raw))).entries[0].graph
        graph.validate()
        self.assertEqual(len(graph.nodes), 33)
        self.assertEqual(len(graph.values), 49)
        self.assertEqual(len(graph.edges), 36)
        self.assertEqual(
            sum(node.kind is OpKind.COLLECTIVE for node in graph.nodes), 8
        )
        self.assertEqual(len(graph.fusion_candidates), 4)

        values = {value.id: value for value in graph.values}
        layer0_output = values["P0.layer0.output"]
        self.assertEqual(
            layer0_output.consumers,
            ("P0.layer1.norm1", "P0.layer1.residual1"),
        )
        layer1_norm1 = next(node for node in graph.nodes if node.id == "P0.layer1.norm1")
        layer1_residual1 = next(
            node for node in graph.nodes if node.id == "P0.layer1.residual1"
        )
        self.assertEqual(
            layer1_norm1.inputs,
            (layer0_output.id, "P0.layer1.w_norm1"),
        )
        self.assertEqual(layer1_residual1.inputs[0], layer0_output.id)
        self.assertNotIn("P0.layer1.input", values)

        attention = [node for node in graph.nodes if node.kind is OpKind.ATTENTION]
        self.assertEqual(
            tuple(node.effects.effect_token for node in attention),
            ("kv_effect_layer_0", "kv_effect_layer_1"),
        )
        self.assertEqual(
            tuple(node.effects.alias_set for node in attention),
            ("kv_alias_layer_0", "kv_alias_layer_1"),
        )

    def test_multiple_profiles_are_canonical_independent_and_provenanced(self) -> None:
        def profile_raw(tokens: int) -> dict[str, object]:
            return {
                "prefill_tokens": tokens,
                "decode_tokens": 0,
                "num_seqs": 1,
                "context_sum": tokens,
                "context_max": tokens,
                "kv_pages": tokens // 16,
                "expert_load": None,
            }

        def distribution(order: tuple[int, int]) -> IR0Template:
            weights = {32: 0.25, 64: 0.75}
            raw = valid_spec()
            raw["workload"]["infer"] = {  # type: ignore[index]
                "source": "shape_dist",
                "output": "logits",
                "shape_dist": {
                    "profiles": [
                        {"key": profile_raw(tokens), "weight": weights[tokens]}
                        for tokens in order
                    ]
                },
            }
            return build_ir0(decode(raw))

        template = distribution((64, 32))
        reversed_template = distribution((32, 64))
        self.assertEqual(template, reversed_template)
        bundle = logical_expand(template)
        self.assertEqual(bundle, logical_expand(reversed_template))
        self.assertEqual(
            tuple(entry.profile_id for entry in bundle.entries),
            tuple(profile.profile_id for profile in template.profiles),
        )
        self.assertEqual(len({entry.graph.id for entry in bundle.entries}), 2)

        by_tokens = {
            entry.graph.profile.prefill_tokens: entry.graph
            for entry in bundle.entries
        }
        graph32 = by_tokens[32]
        graph64 = by_tokens[64]
        self.assertEqual(graph32.values[2].shape, (32, 256))
        self.assertEqual(graph64.values[2].shape, (64, 256))

        def rank_gemm_flops(graph: object) -> int:
            return sum(
                2 * m * n * k
                for node in graph.nodes  # type: ignore[attr-defined]
                if node.kind is OpKind.GEMM
                for m, n, k in (node.workload.rank_shape,)
            )

        self.assertEqual(rank_gemm_flops(graph64), 2 * rank_gemm_flops(graph32))
        attention_pairs = {
            tokens: next(
                node.workload.query_key_pairs
                for node in graph.nodes
                if node.kind is OpKind.ATTENTION
            )
            for tokens, graph in by_tokens.items()
        }
        self.assertEqual(attention_pairs, {32: 528, 64: 2080})

        for tokens, graph in by_tokens.items():
            raw = valid_spec()
            raw["workload"]["infer"]["profile"] = profile_raw(tokens)  # type: ignore[index]
            standalone = logical_expand(build_ir0(decode(raw))).entries[0].graph
            self.assertEqual(standalone, graph)

        reweighted_profiles = tuple(
            replace(profile, weight=0.6 if index == 0 else 0.4)
            for index, profile in enumerate(template.profiles)
        )
        reweighted = rebuild(template, profiles=reweighted_profiles)
        reweighted_bundle = logical_expand(reweighted)
        self.assertEqual(
            tuple(entry.graph for entry in reweighted_bundle.entries),
            tuple(entry.graph for entry in bundle.entries),
        )
        self.assertNotEqual(
            tuple(entry.id for entry in reweighted_bundle.entries),
            tuple(entry.id for entry in bundle.entries),
        )
        self.assertNotEqual(reweighted_bundle.source_template_id, bundle.source_template_id)
        self.assertNotEqual(reweighted_bundle.id, bundle.id)

    def test_out_of_scope_sp_fails_closed(self) -> None:
        base = n2_template()

        parallel = replace(base.instance.parallel, sp=True)
        mesh = replace(
            base.instance.meshes[0],
            axes=(replace(base.instance.meshes[0].axes[0], size=1),),
        )
        sp_instance = replace(base.instance, parallel=parallel, meshes=(mesh,))
        sp_template = rebuild(
            base,
            instance=sp_instance,
            sequence_parallel=SequenceParallelSpec(True, MeshAxisName.TP),
        )
        with self.assertRaisesRegex(UnsupportedFeatureError, "sequence parallelism"):
            logical_expand(sp_template)

        original = deepcopy(base)
        with self.assertRaises(UnsupportedFeatureError):
            logical_expand(sp_template)
        self.assertEqual(base, original)


if __name__ == "__main__":
    unittest.main()
