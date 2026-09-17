"""Layer1 MoE dX must propagate through real norm2 and residual1 skip."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_input_gradient_ir0 import (
    append_moe_full_train_input_gradient_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_backbone_ir0 import (
    append_moe_full_train_layer1_backbone_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_input_gradient_ir0 import (
    MoeInputGradientSourceTest as Fixture,
)


class MoeLayer1BackboneSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.sources = tuple(append_moe_full_train_input_gradient_ir0(source)
                            for source in Fixture.sources)

    def test_two_steps_bind_norm2_gamma_dx_and_skip_merge(self):
        for source in self.sources:
            graph = append_moe_full_train_layer1_backbone_ir0(source)
            gamma, norm_dx, merge = graph.nodes[-3:]
            self.assertIs(gamma.kind, OpKind.NORM_GAMMA_WGRAD)
            self.assertIs(norm_dx.kind, OpKind.RMSNORM_BACKWARD)
            self.assertIs(merge.kind, OpKind.ELEMENTWISE)
            self.assertEqual(norm_dx.inputs[1],
                             "backward::T0.layer1.moe.input_sum.norm2_gradient")
            self.assertEqual(merge.inputs[0],
                             "backward::T0.layer1.residual2.left_gradient")
            self.assertEqual(merge.inputs[1], norm_dx.outputs[0])
            graph.validate()

    def test_wrong_source_and_duplicate_fail_closed(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer1_backbone_ir0(Fixture.sources[0])
        graph = append_moe_full_train_layer1_backbone_ir0(self.sources[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer1_backbone_ir0(graph)


if __name__ == "__main__":
    unittest.main()
