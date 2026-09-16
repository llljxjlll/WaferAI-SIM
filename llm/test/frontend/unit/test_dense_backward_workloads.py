from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.schema.action import canonical_compute_operand_roles
from llm.frontend.wafer_frontend.schema.ir0 import OpKind

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.dense_backward_workloads import (
    AttentionBackwardWorkload, ResidualBackwardWorkload,
    RmsNormBackwardWorkload, RopeBackwardWorkload,
    SwiGluBackwardWorkload,
)


class DenseBackwardWorkloadsTest(unittest.TestCase):
    def test_exact_small_dense_profiles(self) -> None:
        rms = RmsNormBackwardWorkload(1, 4, 1)
        attention = AttentionBackwardWorkload(1, 1, 1, 4, 1, 1, 1)
        rope = RopeBackwardWorkload(1, 1, 1, 1, 1, 1, 1, 4, 4, 64)
        residual = ResidualBackwardWorkload(1, 1, 1, 4)
        swiglu = SwiGluBackwardWorkload(1, 8)
        for workload in (rms, attention, rope, residual, swiglu):
            workload.validate()
        self.assertEqual((rms.tensor_bytes, attention.input_bytes,
                          attention.upstream_bytes, rope.position_bytes,
                          rope.packed_bytes, residual.tensor_bytes,
                          swiglu.input_bytes, swiglu.upstream_bytes),
                         (8, 24, 8, 4, 24, 8, 32, 16))
        self.assertGreater(rope.position_trace_tag, 0)

    def test_opkind_registration_has_exact_multi_output_roles(self) -> None:
        cases = (
            (OpKind.RMSNORM_BACKWARD, RmsNormBackwardWorkload(1, 4, 1),
             (("forward_activation", "upstream_gradient"), ("input_gradient",))),
            (OpKind.ATTENTION_BACKWARD,
             AttentionBackwardWorkload(1, 1, 1, 4, 1, 1, 1),
             (("forward_packed_qkv", "upstream_gradient"), ("input_gradient",))),
            (OpKind.ROPE_BACKWARD,
             RopeBackwardWorkload(1, 1, 1, 1, 1, 1, 1, 4, 4, 64),
             (("position_ids", "upstream_gradient"), ("input_gradient",))),
            (OpKind.RESIDUAL_BACKWARD, ResidualBackwardWorkload(1, 1, 1, 4),
             (("forward_output", "upstream_gradient"),
              ("left_gradient", "right_gradient"))),
            (OpKind.SWIGLU_BACKWARD, SwiGluBackwardWorkload(1, 8),
             (("forward_gate_up", "upstream_gradient"), ("input_gradient",))),
        )
        for kind, workload, expected in cases:
            with self.subTest(kind=kind.value):
                self.assertEqual(canonical_compute_operand_roles(
                    kind, workload, tiled=False), expected)

    def test_invalid_geometry_and_dtype_fail_closed(self) -> None:
        fixtures = (
            replace(RmsNormBackwardWorkload(1, 4, 1), mode=1),
            replace(AttentionBackwardWorkload(2, 2, 1, 4, 1, 1, 3), pairs=2),
            replace(RopeBackwardWorkload(1, 1, 1, 1, 1, 1, 1, 4, 4, 64),
                    logical_query_heads=2),
            replace(ResidualBackwardWorkload(2, 1, 1, 4), logical_rows=2),
            SwiGluBackwardWorkload(0, 8),
        )
        for workload in fixtures:
            with self.subTest(workload=type(workload).__name__), self.assertRaises(SchemaError):
                workload.validate()


if __name__ == "__main__":
    unittest.main()
