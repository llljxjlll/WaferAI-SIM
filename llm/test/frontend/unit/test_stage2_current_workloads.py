from __future__ import annotations

import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType, ProfileKey
from llm.frontend.wafer_frontend.schema.ir0 import (
    AttentionMode,
    AttentionWorkload,
    EffectKind,
    GemmPartition,
    GemmWorkload,
    LogicalNode,
    NodeEffects,
    NodeMath,
    NormWorkload,
    NumericalPolicy,
    OpKind,
    OpPhase,
    ResidualWorkload,
    RmsNormWorkload,
    SwiGluWorkload,
)


class Stage2CurrentWorkloadSchemaTest(unittest.TestCase):
    def test_sequence_parallel_gemm_shards_only_m(self) -> None:
        workload = GemmWorkload(
            logical_shape=(8, 16, 32),
            rank_shape=(4, 16, 32),
            partition=GemmPartition.SEQUENCE_PARALLEL_REPLICATED_WEIGHT,
            dtype=DType.FP16,
        )
        workload.validate("gemm")
        with self.assertRaisesRegex(SchemaError, "shard only M"):
            replace(workload, rank_shape=(4, 8, 32)).validate("gemm")
        with self.assertRaisesRegex(SchemaError, "exactly divide"):
            replace(workload, rank_shape=(3, 16, 32)).validate("gemm")

    def test_exact_current_block_workloads(self) -> None:
        norm = RmsNormWorkload(
            logical_activation_shape=(8, 16),
            logical_output_shape=(8, 16),
            rank_activation_shape=(4, 16),
            rank_output_shape=(4, 16),
            logical_weight_shape=(16,),
            rank_weight_shape=(16,),
            epsilon=1e-5,
            dtype=DType.FP16,
        )
        swiglu = SwiGluWorkload(
            logical_input_shape=(8, 64),
            logical_output_shape=(8, 32),
            rank_input_shape=(8, 32),
            rank_output_shape=(8, 16),
            dtype=DType.FP16,
        )
        residual = ResidualWorkload(
            logical_shape=(8, 16), rank_shape=(4, 16), dtype=DType.FP16
        )
        for workload in (norm, swiglu, residual):
            workload.validate("workload")
        with self.assertRaisesRegex(SchemaError, "replicated"):
            replace(norm, rank_weight_shape=(8,)).validate("norm")
        with self.assertRaisesRegex(SchemaError, "finite positive"):
            replace(norm, epsilon=0.0).validate("norm")
        with self.assertRaisesRegex(SchemaError, "halve"):
            replace(swiglu, logical_output_shape=(8, 31)).validate("swiglu")

    def test_attention_prefill_and_decode_bytes_are_exact(self) -> None:
        prefill = AttentionWorkload(
            profile=ProfileKey(8, 0, 1, 8, 8, 1, None),
            mode=AttentionMode.PREFILL,
            causal=True,
            query_tokens=8,
            context_sum=8,
            context_max=8,
            hidden_size=16,
            num_heads=4,
            num_kv_heads=4,
            head_dim=4,
            rank_num_heads=2,
            rank_num_kv_heads=2,
            query_key_pairs=36,
            logical_kv_read_bytes=0,
            logical_kv_write_bytes=512,
            rank_kv_read_bytes=0,
            rank_kv_write_bytes=256,
            dtype=DType.FP16,
        )
        prefill.validate("prefill")
        decode = AttentionWorkload(
            profile=ProfileKey(0, 2, 2, 10, 6, 1, None),
            mode=AttentionMode.DECODE,
            causal=True,
            query_tokens=2,
            context_sum=10,
            context_max=6,
            hidden_size=16,
            num_heads=4,
            num_kv_heads=4,
            head_dim=4,
            rank_num_heads=2,
            rank_num_kv_heads=2,
            query_key_pairs=10,
            logical_kv_read_bytes=640,
            logical_kv_write_bytes=128,
            rank_kv_read_bytes=320,
            rank_kv_write_bytes=64,
            dtype=DType.FP16,
        )
        decode.validate("decode")
        with self.assertRaisesRegex(SchemaError, "must equal 640"):
            replace(decode, logical_kv_read_bytes=639).validate("decode")
        with self.assertRaisesRegex(SchemaError, "pure prefill"):
            replace(decode, mode=AttentionMode.PREFILL).validate("decode")

    def test_legacy_generic_norm_payload_is_rejected_by_current_ir0_node(self) -> None:
        legacy = NormWorkload((8, 16), (8, 16), (8, 16), (8, 16), DType.FP16)
        node = LogicalNode(
            id="norm",
            instance_id="instance",
            kind=OpKind.NORM,
            phase=OpPhase.FWD,
            stage=0,
            mesh_ref="mesh",
            inputs=(),
            outputs=(),
            workload=legacy,
            math=NodeMath(DType.FP32, NumericalPolicy.BITWISE),
            effects=NodeEffects(EffectKind.PURE, None, None),
            impl_ref="rms_norm",
        )
        with self.assertRaisesRegex(SchemaError, "RmsNormWorkload"):
            node.validate("node")


if __name__ == "__main__":
    unittest.main()
