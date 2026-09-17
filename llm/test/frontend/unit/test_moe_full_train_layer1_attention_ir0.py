"""Source-bound layer1 residual1, output projection, and attention reverse."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_attention_ir0 import (
    append_moe_full_train_layer1_attention_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_backbone_ir0 import (
    append_moe_full_train_layer1_backbone_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_layer1_backbone_ir0 import (
    MoeLayer1BackboneSourceTest as Fixture,
)


class MoeLayer1AttentionSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.sources = tuple(append_moe_full_train_layer1_backbone_ir0(source)
                            for source in Fixture.sources)

    def test_two_steps_bind_attention_and_projection_reverse(self):
        for source in self.sources:
            graph = append_moe_full_train_layer1_attention_ir0(source)
            residual, wgrad, projection_dx, attention_dx = graph.nodes[-4:]
            self.assertEqual(tuple(node.kind for node in graph.nodes[-4:]), (
                OpKind.RESIDUAL_BACKWARD,
                OpKind.GEMM_WEIGHT_WGRAD,
                OpKind.GEMM_INPUT_DX,
                OpKind.ATTENTION_BACKWARD,
            ))
            self.assertEqual(residual.inputs[1],
                             "backward::T0.layer1.residual1.merge.input_gradient")
            self.assertEqual(wgrad.inputs[1], residual.outputs[1])
            self.assertEqual(projection_dx.inputs[1], residual.outputs[1])
            self.assertEqual(attention_dx.inputs[1], projection_dx.outputs[0])
            self.assertIn(projection_dx.id,
                          {access.node_ref for access in graph.state_accesses})
            graph.validate()

    def test_wrong_source_and_duplicate_fail_closed(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer1_attention_ir0(Fixture.sources[0])
        graph = append_moe_full_train_layer1_attention_ir0(self.sources[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer1_attention_ir0(graph)


if __name__ == "__main__":
    unittest.main()
