"""Four EP1 MoE parameter updates must consume their exact physical dW."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_input_gradient_ir0 import (
    append_moe_full_train_input_gradient_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_backbone_ir0 import (
    append_moe_full_train_layer1_backbone_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_parameter_sgd_ir0 import (
    append_moe_full_train_layer1_parameter_sgd_ir0,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import OpKind, StateAccessMode
from llm.test.frontend.unit.test_moe_full_train_input_gradient_ir0 import (
    MoeInputGradientSourceTest as InputFixture,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeLayer1ParameterSgdSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        InputFixture.setUpClass()
        cls.sources = tuple(append_moe_full_train_layer1_backbone_ir0(
            append_moe_full_train_input_gradient_ir0(graph))
            for graph in InputFixture.sources)
        cls.sequences = tuple(build_single_die_moe_train_physical_source(
            Fixture, step=step)[1] for step in (0, 1))

    def test_two_steps_bind_router_and_three_expert_fp32_gradients(self):
        for source, sequence in zip(self.sources, self.sequences, strict=True):
            graph = append_moe_full_train_layer1_parameter_sgd_ir0(
                source, sequence)
            updates = graph.nodes[-4:]
            self.assertEqual(len(updates), 4)
            self.assertEqual(tuple(node.kind for node in updates),
                             (OpKind.OPTIMIZER_UPDATE,) * 4)
            self.assertEqual(tuple(node.workload.element_count for node in updates),
                             (4, 32, 32, 32))
            for node in updates:
                self.assertIs(node.workload.gradient_dtype, DType.FP32)
                self.assertEqual(sum(access.node_ref == node.id
                                     and access.mode is StateAccessMode.READ_WRITE
                                     for access in graph.state_accesses), 1)
            graph.validate()

    def test_missing_source_and_duplicate_fail_closed(self):
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer1_parameter_sgd_ir0(
                InputFixture.sources[0], self.sequences[0])
        graph = append_moe_full_train_layer1_parameter_sgd_ir0(
            self.sources[0], self.sequences[0])
        with self.assertRaises(SchemaError):
            append_moe_full_train_layer1_parameter_sgd_ir0(
                graph, self.sequences[0])


if __name__ == "__main__":
    unittest.main()
