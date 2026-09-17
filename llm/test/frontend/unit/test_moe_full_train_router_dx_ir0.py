"""EP1 router dX must consume same-layer native 0x28 dScore and real gate state."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_router_wgrad_ir0 import (
    append_moe_full_train_router_wgrad_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_expert_backward_ir0 import (
    append_moe_full_train_expert_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_dx_ir0 import (
    append_moe_full_train_router_dx_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind, StateAccessMode
from llm.test.frontend.unit.test_moe_full_train_expert_backward_ir0 import (
    MoeExpertBackwardSourceTest as Fixture,
)


class MoeRouterDxSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.sources = tuple(append_moe_full_train_expert_backward_ir0(
            append_moe_full_train_router_wgrad_ir0(source))
            for source in Fixture.combine)

    def test_two_steps_bind_real_dscore_and_gate_state(self):
        for source in self.sources:
            graph = append_moe_full_train_router_dx_ir0(source)
            node = graph.nodes[-1]
            forward = next(item for item in graph.nodes
                           if item.id == "T0.layer1.moe.router")
            combine = next(item for item in graph.nodes
                           if item.id == "backward::T0.layer1.moe.combine")
            output = next(item for item in graph.values
                          if item.id == node.outputs[0])
            self.assertIs(node.kind, OpKind.GEMM_INPUT_DX)
            self.assertEqual(node.inputs, (forward.inputs[1], combine.outputs[0]))
            self.assertEqual(output.shape, (4, 4))
            self.assertEqual(node.workload.source_forward_op_ref, forward.id)
            self.assertEqual(sum(item.node_ref == node.id and
                                 item.state_ref == node.workload.source_parameter_state_ref
                                 and item.mode is StateAccessMode.READ
                                 for item in graph.state_accesses), 1)
            graph.validate()

    def test_wrong_stage_and_duplicate_router_reverse_fail_closed(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_router_dx_ir0(Fixture.combine[0])
        graph = append_moe_full_train_router_dx_ir0(self.sources[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_router_dx_ir0(graph)


if __name__ == "__main__":
    unittest.main()
