from __future__ import annotations

from dataclasses import replace
import unittest

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
