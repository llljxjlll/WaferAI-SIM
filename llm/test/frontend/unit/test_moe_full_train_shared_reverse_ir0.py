"""MoE shared dCombined must originate in the real head→norm→residual chain."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ce_backward_ir0 import (
    append_moe_full_train_ce_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_head_backward_ir0 import (
    append_moe_full_train_head_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_shared_reverse_ir0 import (
    append_moe_full_train_shared_reverse_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeSharedReverseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.heads = tuple(append_moe_full_train_head_backward_ir0(
            append_moe_full_train_ce_backward_ir0(
                build_single_die_moe_train_physical_source(Fixture, step=step)[0]))
            for step in (0, 1))

    def test_each_step_has_real_norm_dx_gamma_wgrad_and_dcombined(self):
        outputs = tuple(append_moe_full_train_shared_reverse_ir0(source)
                        for source in self.heads)
        self.assertNotEqual(outputs[0].id, outputs[1].id)
        for graph in outputs:
            nodes = {node.id: node for node in graph.nodes}
            values = {value.id: value for value in graph.values}
            norm = nodes["T0.final_norm"]
            residual = nodes["T0.layer1.residual2"]
            combine = nodes["T0.layer1.moe.combine"]
            gamma = next(node for node in graph.nodes
                         if node.kind is OpKind.NORM_GAMMA_WGRAD)
            norm_dx = next(node for node in graph.nodes
                           if node.kind is OpKind.RMSNORM_BACKWARD)
            residual_dx = next(node for node in graph.nodes
                               if node.kind is OpKind.RESIDUAL_BACKWARD)
            head_dx = nodes["backward::T0.lm_head"]
            self.assertEqual(gamma.inputs, (norm.inputs[0], head_dx.outputs[0]))
            self.assertEqual(norm_dx.inputs, gamma.inputs)
            self.assertEqual(residual_dx.inputs,
                             (residual.outputs[0], norm_dx.outputs[0]))
            self.assertEqual(values[residual_dx.outputs[1]].shape,
                             values[combine.outputs[0]].shape)
            self.assertEqual(values[residual_dx.outputs[1]].producer,
                             residual_dx.id)
            self.assertEqual(len(graph.nodes), 36)
            graph.validate()

    def test_requires_real_head_source_and_no_duplicate(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_shared_reverse_ir0(
                append_moe_full_train_ce_backward_ir0(
                    build_single_die_moe_train_physical_source(
                        Fixture, step=0)[0]))
        graph = append_moe_full_train_shared_reverse_ir0(self.heads[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_shared_reverse_ir0(graph)


if __name__ == "__main__":
    unittest.main()
