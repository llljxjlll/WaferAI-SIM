"""Router SGD must consume native dScore-derived FP32 WGRAD and write gate state."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_router_sgd_ir0 import (
    append_moe_full_train_router_sgd_ir0,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import (
    OpKind, StateAccessMode,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ce_backward_ir0 import append_moe_full_train_ce_backward_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_head_backward_ir0 import append_moe_full_train_head_backward_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_shared_reverse_ir0 import append_moe_full_train_shared_reverse_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_combine_backward_ir0 import append_moe_full_train_combine_backward_ir0


class MoeRouterSgdSourceTest(unittest.TestCase):
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
        cls.sequences = tuple(
            build_single_die_moe_train_physical_source(Fixture, step=step)[1]
            for step in (0, 1)
        )

    def test_two_steps_update_exact_gate_state_from_fp32_wgrad(self):
        from llm.frontend.wafer_frontend.passes.moe_full_train_router_wgrad_ir0 import (
            append_moe_full_train_router_wgrad_ir0,
        )
        for step, (combine, sequence) in enumerate(zip(
                self.combine, self.sequences, strict=True)):
            gradient_graph = append_moe_full_train_router_wgrad_ir0(combine)
            graph = append_moe_full_train_router_sgd_ir0(
                gradient_graph, sequence)
            update = graph.nodes[-1]
            gradient_node = gradient_graph.nodes[-1]
            self.assertEqual(update.kind, OpKind.OPTIMIZER_UPDATE)
            self.assertEqual(update.inputs[1], gradient_node.outputs[0])
            self.assertEqual(update.workload.gradient_dtype, DType.FP32)
            self.assertEqual(update.workload.learning_rate, 0.001)
            self.assertEqual(update.workload.element_count, 4)
            self.assertEqual(len(graph.nodes), 39)
            state = gradient_node.workload.source_parameter_state_ref
            self.assertEqual(sum(
                access.node_ref == update.id
                and access.state_ref == state
                and access.mode is StateAccessMode.READ_WRITE
                for access in graph.state_accesses
            ), 1)
            self.assertEqual(gradient_graph.nodes[-1].workload.source_forward_op_ref,
                             f"T0.layer1.moe.router")
            self.assertEqual(step, next(node for node in graph.nodes if node.id == "T0.layer1.moe.router").workload.step)
            graph.validate()

    def test_missing_native_router_gradient_and_duplicate_update_fail_closed(self):
        from llm.frontend.wafer_frontend.passes.moe_full_train_router_wgrad_ir0 import (
            append_moe_full_train_router_wgrad_ir0,
        )
        with self.assertRaises(SchemaError):
            append_moe_full_train_router_sgd_ir0(
                self.combine[0], self.sequences[0])
        gradient = append_moe_full_train_router_wgrad_ir0(self.combine[0])
        update = append_moe_full_train_router_sgd_ir0(
            gradient, self.sequences[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_router_sgd_ir0(update, self.sequences[0])


if __name__ == "__main__":
    unittest.main()
