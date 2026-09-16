"""Native expert reverse source needs real 0x28 dExpert and three FP32 gradients."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ce_backward_ir0 import append_moe_full_train_ce_backward_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_head_backward_ir0 import append_moe_full_train_head_backward_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_shared_reverse_ir0 import append_moe_full_train_shared_reverse_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_combine_backward_ir0 import append_moe_full_train_combine_backward_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_router_wgrad_ir0 import append_moe_full_train_router_wgrad_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_expert_backward_ir0 import append_moe_full_train_expert_backward_ir0
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeExpertBackwardSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.combine = []
        for step in (0, 1):
            phase = build_single_die_moe_train_physical_source(Fixture, step=step)[0]
            cls.combine.append(append_moe_full_train_combine_backward_ir0(
                append_moe_full_train_shared_reverse_ir0(
                    append_moe_full_train_head_backward_ir0(
                        append_moe_full_train_ce_backward_ir0(phase)))))

    def test_real_same_layer_dexpert_and_three_projection_gradients(self):
        for combine in self.combine:
            source = append_moe_full_train_router_wgrad_ir0(combine)
            graph = append_moe_full_train_expert_backward_ir0(source)
            node = graph.nodes[-1]
            forward = next(item for item in graph.nodes
                           if item.id == "T0.layer1.moe.expert0")
            score_backward = next(item for item in graph.nodes
                                  if item.id == "backward::T0.layer1.moe.combine")
            values = {item.id: item for item in graph.values}
            self.assertEqual(node.kind, OpKind.MOE_EXPERT_BACKWARD)
            self.assertEqual(node.inputs, (*forward.inputs,
                                           score_backward.outputs[1]))
            self.assertEqual(tuple(values[ref].shape for ref in node.outputs),
                             ((4, 4), (4, 8), (4, 8), (8, 4)))
            self.assertEqual(tuple(values[ref].dtype for ref in node.outputs),
                             (DType.FP16, DType.FP32, DType.FP32, DType.FP32))
            self.assertEqual(node.workload.source_forward_op_ref, forward.id)
            self.assertEqual(node.workload.source_combine_backward_op_ref,
                             score_backward.id)
            graph.validate()

    def test_no_fake_upstream_or_duplicate_reverse(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_expert_backward_ir0(self.combine[0])
        source = append_moe_full_train_router_wgrad_ir0(self.combine[0])
        graph = append_moe_full_train_expert_backward_ir0(source)
        with self.assertRaises(SchemaError):
            append_moe_full_train_expert_backward_ir0(graph)


if __name__ == "__main__":
    unittest.main()
