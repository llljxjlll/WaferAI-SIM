from __future__ import annotations

import json
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes import (
    build_stage2_dense_forward_oracle as public_build_oracle,
)
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.stage2_dense_forward_oracle import (
    build_stage2_dense_forward_oracle,
)
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.stage2_dense_forward_oracle import (
    STAGE2_DENSE_FORWARD_ORACLE_SCHEMA_VERSION,
    DenseAttentionMetrics,
    DenseCollectiveKindMetrics,
    DenseCollectiveMetrics,
    DenseEmbeddingMetrics,
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

from _fixtures import valid_spec


def _template(tp: int, *, output: str = "logits"):
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
    raw["workload"]["infer"].update(  # type: ignore[index]
        output=output,
        profile={
            "prefill_tokens": 8,
            "decode_tokens": 0,
            "num_seqs": 1,
            "context_sum": 8,
            "context_max": 8,
            "kv_pages": 1,
            "expert_load": None,
        },
    )
    return build_ir0(from_data(ExperimentSpec, raw, path="spec"))


def _gemm(
    qkv: tuple[int, int, int],
    attention_output: tuple[int, int, int],
    gate_up: tuple[int, int, int],
    down: tuple[int, int, int],
    lm_head: tuple[int, int, int],
) -> DenseGemmMetrics:
    return DenseGemmMetrics(
        qkv=DenseGemmOpMetrics(*qkv),
        attention_output=DenseGemmOpMetrics(*attention_output),
        gate_up=DenseGemmOpMetrics(*gate_up),
        down=DenseGemmOpMetrics(*down),
        lm_head=DenseGemmOpMetrics(*lm_head),
    )


class Stage2DenseForwardOracleTest(unittest.TestCase):
    def test_four_case_schema_roundtrip_determinism_and_template_gate(self) -> None:
        self.assertIs(public_build_oracle, build_stage2_dense_forward_oracle)
        self.assertEqual(
            STAGE2_DENSE_FORWARD_ORACLE_SCHEMA_VERSION,
            "wafer_frontend.stage2_dense_forward_oracle/v1alpha1",
        )
        for tp, output in ((1, "logits"), (2, "logits"), (4, "logits"), (1, "greedy_sample")):
            with self.subTest(tp=tp, output=output):
                template = _template(tp, output=output)
                before = canonical_digest(template)
                profile = template.profiles[0].key
                oracle = build_stage2_dense_forward_oracle(
                    template, profile, tp_degree=tp
                )
                oracle.validate_against_template(template)
                self.assertEqual(
                    loads_dataclass(
                        Stage2DenseForwardOracle,
                        canonical_json(oracle),
                        path="oracle",
                    ),
                    oracle,
                )
                self.assertEqual(
                    build_stage2_dense_forward_oracle(template, profile, tp_degree=tp),
                    oracle,
                )
                self.assertEqual(canonical_digest(template), before)

    def test_logits_tp1_tp2_tp4_exact_metrics(self) -> None:
        expected_parameters = {
            1: DenseParameterMetrics(15, 6224, 12448, 6224, 12448),
            2: DenseParameterMetrics(15, 6224, 12448, 7328, 14656),
            4: DenseParameterMetrics(15, 6224, 12448, 9536, 19072),
        }
        expected_graph = {
            1: DenseGraphMetrics(25, 0, 15, 4),
            2: DenseGraphMetrics(33, 8, 30, 8),
            4: DenseGraphMetrics(33, 8, 60, 16),
        }
        logical_gemm = _gemm(
            (24576, 3584, 1536),
            (8192, 1536, 512),
            (32768, 4608, 2048),
            (16384, 3072, 512),
            (8192, 1280, 512),
        )
        rank_gemm = {
            1: logical_gemm,
            2: _gemm(
                (12288, 2048, 768),
                (4096, 768, 512),
                (16384, 2560, 1024),
                (8192, 1536, 512),
                (4096, 1152, 256),
            ),
            4: _gemm(
                (6144, 1280, 384),
                (2048, 384, 512),
                (8192, 1536, 512),
                (4096, 768, 512),
                (2048, 1088, 128),
            ),
        }
        logical_non_gemm = (
            DenseEmbeddingMetrics(8, 0, 0, 288, 256),
            DenseRmsNormMetrics(40, 2600, 0, 1440, 1280),
            DenseRopeQkMetrics(512, 1536, 0, 1536, 1536),
            DenseAttentionMetrics(72, 288, 2304, 2304, 576, 288, 1536, 512),
            DenseSwiGluMetrics(512, 2048, 512, 2048, 1024),
            DenseResidualMetrics(512, 512, 0, 2048, 1024),
            DenseGreedyMetrics(0, 0, 0, 0, 0, 0),
        )
        rank_non_gemm = {
            1: logical_non_gemm,
            2: (
                DenseEmbeddingMetrics(4, 0, 0, 144, 128),
                DenseRmsNormMetrics(20, 1300, 0, 800, 640),
                DenseRopeQkMetrics(256, 768, 0, 768, 768),
                DenseAttentionMetrics(72, 144, 1152, 1152, 288, 144, 768, 256),
                DenseSwiGluMetrics(256, 1024, 256, 1024, 512),
                DenseResidualMetrics(256, 256, 0, 1024, 512),
                DenseGreedyMetrics(0, 0, 0, 0, 0, 0),
            ),
            4: (
                DenseEmbeddingMetrics(2, 0, 0, 72, 64),
                DenseRmsNormMetrics(10, 650, 0, 480, 320),
                DenseRopeQkMetrics(128, 384, 0, 384, 384),
                DenseAttentionMetrics(72, 72, 576, 576, 144, 72, 384, 128),
                DenseSwiGluMetrics(128, 512, 128, 512, 256),
                DenseResidualMetrics(128, 128, 0, 512, 256),
                DenseGreedyMetrics(0, 0, 0, 0, 0, 0),
            ),
        }
        expected_collective = {
            1: DenseCollectiveKindMetrics(0, 0, 0, 0),
            2: DenseCollectiveKindMetrics(4, 256, 512, 1024),
            4: DenseCollectiveKindMetrics(4, 256, 768, 3072),
        }
        expected_kv = {
            1: DenseKvMetrics(0, 1024, 0, 1024),
            2: DenseKvMetrics(0, 1024, 0, 512),
            4: DenseKvMetrics(0, 1024, 0, 256),
        }

        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                template = _template(tp)
                oracle = build_stage2_dense_forward_oracle(
                    template, template.profiles[0].key, tp_degree=tp
                )
                self.assertEqual(oracle.parameters, expected_parameters[tp])
                self.assertEqual(oracle.graph, expected_graph[tp])
                self.assertEqual(oracle.logical_work.gemm, logical_gemm)
                self.assertEqual(oracle.rank_work.gemm, rank_gemm[tp])
                self.assertEqual(
                    (
                        oracle.logical_work.embedding,
                        oracle.logical_work.rms_norm,
                        oracle.logical_work.rope_qk,
                        oracle.logical_work.attention,
                        oracle.logical_work.swiglu,
                        oracle.logical_work.residual,
                        oracle.logical_work.greedy,
                    ),
                    logical_non_gemm,
                )
                self.assertEqual(
                    (
                        oracle.rank_work.embedding,
                        oracle.rank_work.rms_norm,
                        oracle.rank_work.rope_qk,
                        oracle.rank_work.attention,
                        oracle.rank_work.swiglu,
                        oracle.rank_work.residual,
                        oracle.rank_work.greedy,
                    ),
                    rank_non_gemm[tp],
                )
                self.assertEqual(
                    oracle.collectives,
                    DenseCollectiveMetrics(
                        expected_collective[tp], expected_collective[tp]
                    ),
                )
                self.assertEqual(oracle.kv, expected_kv[tp])

    def test_tp1_greedy_is_a_separate_one_sample_terminal(self) -> None:
        template = _template(1, output="greedy_sample")
        oracle = build_stage2_dense_forward_oracle(
            template, template.profiles[0].key, tp_degree=1
        )
        self.assertEqual(oracle.graph, DenseGraphMetrics(26, 0, 15, 4))
        expected = DenseGreedyMetrics(1, 31, 0, 0, 64, 4)
        self.assertEqual(oracle.logical_work.greedy, expected)
        self.assertEqual(oracle.rank_work.greedy, expected)

    def test_strict_serde_source_recompute_and_fail_closed_boundaries(self) -> None:
        template = _template(1)
        profile = template.profiles[0].key
        oracle = build_stage2_dense_forward_oracle(template, profile, tp_degree=1)
        raw = json.loads(canonical_json(oracle))
        raw["unexpected"] = 1
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(Stage2DenseForwardOracle, json.dumps(raw), path="oracle")
        del raw["unexpected"]
        del raw["kv"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(Stage2DenseForwardOracle, json.dumps(raw), path="oracle")
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(oracle, schema_version="wafer_frontend.stage2_dense_forward_oracle/v1alpha0").validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(oracle, id="oracle_tampered").validate()

        self_consistent_tampers = (
            (
                "parameters",
                replace(
                    oracle.parameters,
                    placed_bytes=oracle.parameters.placed_bytes + 2,
                ),
            ),
            (
                "graph",
                replace(
                    oracle.graph,
                    parameter_declaration_count=(
                        oracle.graph.parameter_declaration_count + 1
                    ),
                ),
            ),
            (
                "logical_work",
                replace(
                    oracle.logical_work,
                    rms_norm=replace(
                        oracle.logical_work.rms_norm,
                        vector_ops=oracle.logical_work.rms_norm.vector_ops + 1,
                    ),
                ),
            ),
            (
                "rank_work",
                replace(
                    oracle.rank_work,
                    attention=replace(
                        oracle.rank_work.attention,
                        vector_ops=oracle.rank_work.attention.vector_ops + 1,
                    ),
                ),
            ),
            (
                "collectives",
                replace(
                    oracle.collectives,
                    all_gather=replace(
                        oracle.collectives.all_gather,
                        rank_payload_bytes_total=(
                            oracle.collectives.all_gather.rank_payload_bytes_total
                            + 1
                        ),
                    ),
                ),
            ),
            (
                "kv",
                replace(
                    oracle.kv,
                    logical_write_bytes=oracle.kv.logical_write_bytes + 2,
                ),
            ),
        )
        for field_name, tampered_value in self_consistent_tampers:
            with self.subTest(self_consistent_tamper=field_name):
                semantic_key = oracle._semantic_key()
                semantic_key[field_name] = tampered_value
                self_consistent_tamper = Stage2DenseForwardOracle.create(
                    **semantic_key
                )
                self_consistent_tamper.validate()
                with self.assertRaisesRegex(SchemaError, "source template"):
                    self_consistent_tamper.validate_against_template(template)

        with self.assertRaisesRegex(SchemaError, "unsigned 64-bit"):
            build_stage2_dense_forward_oracle(template, profile, tp_degree=True)
        with self.assertRaisesRegex(SchemaError, "not present"):
            build_stage2_dense_forward_oracle(
                template, replace(profile, kv_pages=2), tp_degree=1
            )
        greedy_tp2 = _template(2, output="greedy_sample")
        with self.assertRaisesRegex(UnsupportedFeatureError, "TP-sharded"):
            build_stage2_dense_forward_oracle(
                greedy_tp2, greedy_tp2.profiles[0].key, tp_degree=2
            )


if __name__ == "__main__":
    unittest.main()
