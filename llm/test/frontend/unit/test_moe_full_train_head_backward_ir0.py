"""MoE LM-head reverse edge must consume real CE dLogits and saved head tape."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ce_backward_ir0 import (
    append_moe_full_train_ce_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_head_backward_ir0 import (
    append_moe_full_train_head_backward_ir0,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeHeadBackwardSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.phases = tuple(build_single_die_moe_train_physical_source(
            Fixture, step=step)[0] for step in (0, 1))

    def test_both_steps_connect_real_ce_to_head_gradients(self):
        graphs = tuple(append_moe_full_train_head_backward_ir0(
            append_moe_full_train_ce_backward_ir0(phase))
            for phase in self.phases)
        self.assertNotEqual(graphs[0].id, graphs[1].id)
        for graph in graphs:
            nodes = {node.kind: node for node in graph.nodes if node.kind in (
                OpKind.CE_BACKWARD, OpKind.GEMM_WEIGHT_WGRAD,
                OpKind.GEMM_INPUT_DX)}
            values = {value.id: value for value in graph.values}
            dlogits = nodes[OpKind.CE_BACKWARD].outputs[0]
            self.assertIn(dlogits, nodes[OpKind.GEMM_WEIGHT_WGRAD].inputs)
            self.assertIn(dlogits, nodes[OpKind.GEMM_INPUT_DX].inputs)
            self.assertIs(values[nodes[OpKind.GEMM_WEIGHT_WGRAD].outputs[0]].dtype,
                          DType.FP32)
            self.assertIs(values[nodes[OpKind.GEMM_INPUT_DX].outputs[0]].dtype,
                          DType.FP16)
            self.assertEqual(len(graph.nodes), 33)
            graph.validate()

    def test_missing_ce_reverse_cannot_start_head_backward(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_head_backward_ir0(self.phases[0].graph)


if __name__ == "__main__":
    unittest.main()
