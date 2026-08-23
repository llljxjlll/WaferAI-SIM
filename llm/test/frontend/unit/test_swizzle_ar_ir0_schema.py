from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.schema.common import DType, MeshAxisName
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    CollectiveRole,
    CollectiveWorkload,
    ReduceOp,
)


def _all_reduce() -> CollectiveWorkload:
    return CollectiveWorkload(
        collective=CollectiveKind.ALL_REDUCE,
        reduce_op=ReduceOp.SUM,
        mesh_axes=(MeshAxisName.TP,),
        participant_count=2,
        reduction_mesh_axes=(MeshAxisName.TP,),
        scatter_tensor_axis=None,
        gather_tensor_axis=None,
        logical_tensor_bytes=128,
        rank_input_bytes=128,
        rank_output_bytes=128,
        rank_logical_payload_bytes=128,
        group_logical_payload_bytes=256,
        dtype=DType.FP16,
        role=CollectiveRole.ACTIVATION,
        input_layout="MN_partial_tp",
        output_layout="MN_replicated",
    )


class SwizzleAllReduceIr0SchemaTest(unittest.TestCase):
    def test_sum_all_reduce_has_exact_ring_payload_contract(self) -> None:
        _all_reduce().validate("all_reduce")

    def test_axes_bytes_payload_and_reduce_op_fail_closed(self) -> None:
        cases = (
            (replace(_all_reduce(), scatter_tensor_axis=0), SchemaError, "axes"),
            (replace(_all_reduce(), rank_output_bytes=64), SchemaError, "rank input/output"),
            (replace(_all_reduce(), rank_logical_payload_bytes=64), SchemaError, "payloads"),
            (replace(_all_reduce(), reduction_mesh_axes=()), SchemaError, "exactly equal"),
            (replace(_all_reduce(), reduce_op=ReduceOp.MAX), UnsupportedFeatureError, "SUM only"),
        )
        for workload, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    workload.validate("all_reduce")


if __name__ == "__main__":
    unittest.main()
