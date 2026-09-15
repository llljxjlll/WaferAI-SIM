"""Exact source shape and byte ABI for one real GEMM parameter-gradient tile."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.gemm_weight_wgrad_workload import (
    GemmWeightWgradWorkload,
)


class GemmWeightWgradWorkloadTest(unittest.TestCase):
    def test_fp16_source_shapes_and_fp32_parameter_tile(self) -> None:
        tile = GemmWeightWgradWorkload(
            8, 16, 4, "layer0.qkv.forward", "layer0.qkv.weight")
        tile.validate()
        self.assertEqual((tile.activation_bytes, tile.upstream_bytes,
                          tile.gradient_bytes, tile.fma_ops),
                         (64, 128, 512, 512))
        self.assertIs(tile.gradient_dtype, DType.FP32)
        self.assertIs(tile.activation_dtype, DType.FP16)
        self.assertIs(tile.upstream_dtype, DType.FP16)

    def test_rejects_fp16_output_or_unsound_source_or_oversized_tile(self) -> None:
        tile = GemmWeightWgradWorkload(
            8, 16, 4, "layer0.qkv.forward", "layer0.qkv.weight")
        for altered in (
            replace(tile, gradient_dtype=DType.FP16),
            replace(tile, upstream_dtype=DType.FP32),
            replace(tile, source_forward_op_ref=""),
            replace(tile, source_parameter_state_ref=""),
            replace(tile, k=0),
            replace(tile, m=512, n=512),
        ):
            with self.subTest(altered=altered), self.assertRaises(SchemaError):
                altered.validate()


if __name__ == "__main__":
    unittest.main()
