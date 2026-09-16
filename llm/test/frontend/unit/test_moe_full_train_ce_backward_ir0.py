"""The first reverse edge must use each real MoE step's CE tape."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ce_backward_ir0 import (
    append_moe_full_train_ce_backward_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import IR0, OpKind, OpPhase
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeCeBackwardSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.phases = tuple(build_single_die_moe_train_physical_source(
            Fixture, step=step)[0] for step in (0, 1))

    def test_each_step_has_distinct_independent_dloss_and_ce_dlogits(self):
        graphs = tuple(append_moe_full_train_ce_backward_ir0(phase)
                       for phase in self.phases)
        self.assertNotEqual(graphs[0].id, graphs[1].id)
        for graph in graphs:
            ce = next(node for node in graph.nodes
                      if node.kind is OpKind.CE_FORWARD)
            backward = next(node for node in graph.nodes
                            if node.kind is OpKind.CE_BACKWARD)
            values = {value.id: value for value in graph.values}
            self.assertEqual(backward.phase, OpPhase.DGRAD)
            self.assertEqual(backward.inputs[:2], ce.inputs)
            self.assertNotEqual(backward.inputs[2], ce.outputs[0])
            self.assertIsNone(values[backward.inputs[2]].producer)
            self.assertEqual(values[backward.outputs[0]].producer, backward.id)
            self.assertEqual(len(graph.nodes), 31)
            graph.validate()

    def test_rejects_non_moe_source_and_replayed_reverse_node(self):
        phase = self.phases[0]
        forged = IR0.create(
            producer_pass="train_forward_expand", **phase.graph._semantic_key(),
        )
        with self.assertRaises(SchemaError):
            append_moe_full_train_ce_backward_ir0(
                replace(phase, graph=forged))
        graph = append_moe_full_train_ce_backward_ir0(phase)
        with self.assertRaises(SchemaError):
            append_moe_full_train_ce_backward_ir0(
                replace(phase, graph=graph))


if __name__ == "__main__":
    unittest.main()
