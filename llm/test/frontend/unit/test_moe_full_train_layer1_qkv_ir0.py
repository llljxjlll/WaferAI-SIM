"""Two-step MoE layer1 QKV/RoPE/norm1 reverse reaches layer0 source."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_attention_ir0 import (
    append_moe_full_train_layer1_attention_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_qkv_ir0 import (
    append_moe_full_train_layer1_qkv_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_layer1_attention_ir0 import (
    MoeLayer1AttentionSourceTest as Fixture,
)


class MoeLayer1QkvSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.sources = tuple(append_moe_full_train_layer1_attention_ir0(source)
                            for source in Fixture.sources)

    def test_two_steps_reach_layer0_with_owned_gradients(self):
        for source in self.sources:
            graph = append_moe_full_train_layer1_qkv_ir0(source)
            rope, wgrad, qkv_dx, gamma, norm_dx, merge = graph.nodes[-6:]
            self.assertEqual(tuple(node.kind for node in graph.nodes[-6:]), (
                OpKind.ROPE_BACKWARD,
                OpKind.GEMM_WEIGHT_WGRAD,
                OpKind.GEMM_INPUT_DX,
                OpKind.NORM_GAMMA_WGRAD,
                OpKind.RMSNORM_BACKWARD,
                OpKind.ELEMENTWISE,
            ))
            self.assertEqual(wgrad.inputs[1], rope.outputs[0])
            self.assertEqual(qkv_dx.inputs[1], rope.outputs[0])
            self.assertEqual(gamma.inputs[1], qkv_dx.outputs[0])
            self.assertEqual(norm_dx.inputs[1], qkv_dx.outputs[0])
            self.assertEqual(merge.inputs[1], norm_dx.outputs[0])
            self.assertEqual(merge.inputs[0],
                             "backward::T0.layer1.residual1.left_gradient")
            self.assertEqual(next(value for value in graph.values
                                  if value.id == merge.outputs[0]).producer,
                             merge.id)
            self.assertIn(qkv_dx.id,
                          {access.node_ref for access in graph.state_accesses})
            graph.validate()

    def test_wrong_source_and_duplicate_fail_closed(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer1_qkv_ir0(Fixture.sources[0])
        graph = append_moe_full_train_layer1_qkv_ir0(self.sources[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer1_qkv_ir0(graph)


if __name__ == "__main__":
    unittest.main()
