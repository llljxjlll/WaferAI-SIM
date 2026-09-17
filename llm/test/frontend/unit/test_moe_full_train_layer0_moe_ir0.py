"""Layer0 MoE source reverse is bound to layer1's true input gradient."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_attention_ir0 import (
    append_moe_full_train_layer0_attention_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_qkv_ir0 import (
    append_moe_full_train_layer0_qkv_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer0_moe_ir0 import (
    append_moe_full_train_layer0_moe_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_qkv_ir0 import (
    append_moe_full_train_layer1_qkv_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_layer1_qkv_ir0 import (
    MoeLayer1QkvSourceTest as Fixture,
)


class MoeLayer0SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.sources = tuple(append_moe_full_train_layer1_qkv_ir0(source)
                            for source in Fixture.sources)

    def test_both_steps_connect_layer0_moe_and_shared_norm2(self):
        for source in self.sources:
            graph = append_moe_full_train_layer0_moe_ir0(source)
            reverse = graph.nodes[-9:]
            self.assertEqual(tuple(node.kind for node in reverse), (
                OpKind.RESIDUAL_BACKWARD,
                OpKind.MOE_COMBINE_BACKWARD,
                OpKind.GEMM_WEIGHT_WGRAD,
                OpKind.MOE_EXPERT_BACKWARD,
                OpKind.GEMM_INPUT_DX,
                OpKind.ELEMENTWISE,
                OpKind.NORM_GAMMA_WGRAD,
                OpKind.RMSNORM_BACKWARD,
                OpKind.ELEMENTWISE,
            ))
            self.assertEqual(reverse[0].inputs[1],
                             "backward::T0.layer1.norm1.merge_layer0.input_gradient")
            self.assertEqual(reverse[1].inputs[3], reverse[0].outputs[1])
            self.assertEqual(reverse[3].inputs[4], reverse[1].outputs[1])
            self.assertEqual(reverse[5].inputs,
                             (reverse[3].outputs[0], reverse[4].outputs[0]))
            self.assertEqual(reverse[8].inputs,
                             (reverse[0].outputs[0], reverse[7].outputs[0]))
            self.assertIn(reverse[4].id,
                          {access.node_ref for access in graph.state_accesses})
            graph.validate()

    def test_both_steps_continue_to_embedding_gradient(self):
        for source in self.sources:
            graph = append_moe_full_train_layer0_moe_ir0(source)
            graph = append_moe_full_train_layer0_attention_ir0(graph)
            graph = append_moe_full_train_layer0_qkv_ir0(graph)
            self.assertEqual(graph.nodes[-1].outputs,
                             ("backward::T0.layer0.norm1.merge_layer0.input_gradient",))
            self.assertEqual(graph.nodes[-1].kind, OpKind.ELEMENTWISE)
            self.assertEqual(graph.nodes[-1].inputs[0],
                             "backward::T0.layer0.residual1.left_gradient")
            graph.validate()

    def test_wrong_source_and_duplicate_fail_closed(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer0_moe_ir0(Fixture.sources[0])
        graph = append_moe_full_train_layer0_moe_ir0(self.sources[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer0_moe_ir0(graph)


if __name__ == "__main__":
    unittest.main()
