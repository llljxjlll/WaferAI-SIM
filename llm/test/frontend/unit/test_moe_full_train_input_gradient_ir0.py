"""EP1 expert/router dX sum must prove the signed identity route."""

import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_input_gradient_ir0 import (
    append_moe_full_train_input_gradient_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_dx_ir0 import (
    append_moe_full_train_router_dx_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import IR0, OpKind
from llm.test.frontend.unit.test_moe_full_train_router_dx_ir0 import (
    MoeRouterDxSourceTest as Fixture,
)


class MoeInputGradientSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.sources = tuple(append_moe_full_train_router_dx_ir0(source)
                            for source in Fixture.sources)

    def test_two_steps_sum_exact_expert_and_router_derivatives(self):
        for source in self.sources:
            graph = append_moe_full_train_input_gradient_ir0(source)
            node = graph.nodes[-1]
            self.assertIs(node.kind, OpKind.ELEMENTWISE)
            self.assertEqual(node.inputs, (
                "backward::T0.layer1.moe.expert0.activation_gradient",
                "backward::T0.layer1.moe.router.input.gradient",
            ))
            self.assertEqual(node.outputs,
                             ("backward::T0.layer1.moe.input_sum.norm2_gradient",))
            graph.validate()

    def test_missing_router_stage_and_duplicate_fail_closed(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_input_gradient_ir0(Fixture.sources[0])
        graph = append_moe_full_train_input_gradient_ir0(self.sources[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_input_gradient_ir0(graph)

    def test_tampered_gradient_order_rejected_by_ir0(self):
        graph = append_moe_full_train_input_gradient_ir0(self.sources[0])
        node = graph.nodes[-1]
        tampered = replace(node, inputs=tuple(reversed(node.inputs)))
        altered = IR0.create(
            producer_pass=graph.producer_pass, job=graph.job,
            instances=graph.instances, nodes=(*graph.nodes[:-1], tampered),
            values=graph.values, edges=graph.edges,
            fusion_candidates=graph.fusion_candidates, profile=graph.profile,
            train=graph.train, persistent_states=graph.persistent_states,
            state_accesses=graph.state_accesses,
        )
        with self.assertRaisesRegex(SchemaError, "MoE input gradient sum"):
            altered.validate()


if __name__ == "__main__":
    unittest.main()
