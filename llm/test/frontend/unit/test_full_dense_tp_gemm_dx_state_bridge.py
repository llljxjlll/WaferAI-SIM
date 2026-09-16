"""TP dX weights must come from every actual forward GEMM shard READ."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
)
from llm.frontend.wafer_frontend.passes.full_dense_tp_gemm_dx_state_bridge import (
    dense_tp_gemm_dx_state_refs,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class DenseTpGemmDxStateBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_flexible_dense_train_plan(
            _spec(1, 4), RectMeshSpec(1, 4),
        )

    def test_all_nine_real_tp4_gemms_bind_four_distinct_forward_reads(self) -> None:
        plan = self.plan
        graph = plan.forward_graph
        states = {item.id: item for item in graph.persistent_states}
        gemms = tuple(node for node in graph.nodes if node.kind is OpKind.GEMM)
        self.assertEqual(len(gemms), 9)
        all_refs = []
        for node in gemms:
            refs = dense_tp_gemm_dx_state_refs(plan, node.id)
            self.assertEqual(len(refs), 4)
            self.assertEqual(len(set(refs)), 4)
            for rank, ref in enumerate(refs):
                with self.subTest(gemm=node.id, rank=rank):
                    self.assertEqual(states[ref].identity.shard_index, rank)
                    self.assertEqual(states[ref].identity.tensor_ref,
                                     node.inputs[1])
                    self.assertTrue(any(
                        access.node_ref == node.id
                        and access.rank == rank and access.state_ref == ref
                        for access in graph.state_accesses
                    ))
            all_refs.extend(refs)
        self.assertEqual(len(set(all_refs)), 36)

    def test_missing_tp_shard_read_and_non_gemm_fail_closed(self) -> None:
        plan = self.plan
        graph = plan.forward_graph
        node = next(node for node in graph.nodes if node.kind is OpKind.GEMM)
        corrupted = replace(graph, state_accesses=tuple(
            access for access in graph.state_accesses
            if not (access.node_ref == node.id and access.rank == 2)
        ))
        with self.assertRaises(SchemaError):
            dense_tp_gemm_dx_state_refs(
                replace(plan, forward_graph=corrupted), node.id,
            )
        with self.assertRaisesRegex(SchemaError, "real forward GEMM"):
            dense_tp_gemm_dx_state_refs(plan, "not_a_real_gemm")


if __name__ == "__main__":
    unittest.main()
