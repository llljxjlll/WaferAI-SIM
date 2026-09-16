"""Router FP32 gate WGRAD must consume same-layer native 0x28 dScore."""

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
from llm.frontend.wafer_frontend.passes.moe_full_train_combine_backward_ir0 import (
    append_moe_full_train_combine_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_wgrad_ir0 import (
    append_moe_full_train_router_wgrad_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeRouterWgradSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.combine = tuple(append_moe_full_train_combine_backward_ir0(
            append_moe_full_train_shared_reverse_ir0(
                append_moe_full_train_head_backward_ir0(
                    append_moe_full_train_ce_backward_ir0(
                        build_single_die_moe_train_physical_source(
                            Fixture, step=step)[0]))))
            for step in (0, 1))

    def test_two_steps_bind_real_gate_state_and_same_layer_dscore(self):
        graphs = tuple(append_moe_full_train_router_wgrad_ir0(source)
                       for source in self.combine)
        self.assertNotEqual(graphs[0].id, graphs[1].id)
        for graph in graphs:
            nodes = {node.id: node for node in graph.nodes}
            router = nodes["T0.layer1.moe.router"]
            combined = nodes["backward::T0.layer1.moe.combine"]
            wgrad = graph.nodes[-1]
            self.assertEqual(wgrad.kind, OpKind.GEMM_WEIGHT_WGRAD)
            self.assertEqual(wgrad.inputs,
                             (router.inputs[0], combined.outputs[0]))
            states = {item.id: item for item in graph.persistent_states}
            state = states[wgrad.workload.source_parameter_state_ref]
            self.assertEqual(state.identity.tensor_ref, router.inputs[1])
            self.assertEqual((wgrad.workload.m, wgrad.workload.n,
                              wgrad.workload.k), (4, 1, 4))
            self.assertEqual(len(graph.nodes), 38)
            graph.validate()

    def test_cannot_create_router_gradient_without_0x28(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_router_wgrad_ir0(
                append_moe_full_train_shared_reverse_ir0(
                    append_moe_full_train_head_backward_ir0(
                        append_moe_full_train_ce_backward_ir0(
                            build_single_die_moe_train_physical_source(
                                Fixture, step=0)[0]))))
        graph = append_moe_full_train_router_wgrad_ir0(self.combine[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_router_wgrad_ir0(graph)


if __name__ == "__main__":
    unittest.main()
