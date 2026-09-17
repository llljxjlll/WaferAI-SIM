"""Every EP1 MoE trainable state has a distinct sourced two-step SGD."""

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_all_parameter_sgd_ir0 import (
    append_moe_full_train_all_parameter_sgd_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_embedding_wgrad_ir0 import (
    append_moe_full_train_embedding_wgrad_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer0_moe_ir0 import (
    append_moe_full_train_layer0_moe_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_attention_ir0 import (
    append_moe_full_train_layer0_attention_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_qkv_ir0 import (
    append_moe_full_train_layer0_qkv_ir0,
    append_moe_full_train_layer1_qkv_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind, StateAccessMode
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)
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
from llm.frontend.wafer_frontend.passes.moe_full_train_expert_backward_ir0 import (
    append_moe_full_train_expert_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_dx_ir0 import (
    append_moe_full_train_router_dx_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_input_gradient_ir0 import (
    append_moe_full_train_input_gradient_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_backbone_ir0 import (
    append_moe_full_train_layer1_backbone_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_layer1_attention_ir0 import (
    append_moe_full_train_layer1_attention_ir0,
)


_REVERSE = (
    append_moe_full_train_ce_backward_ir0,
    append_moe_full_train_head_backward_ir0,
    append_moe_full_train_shared_reverse_ir0,
    append_moe_full_train_combine_backward_ir0,
    append_moe_full_train_router_wgrad_ir0,
    append_moe_full_train_expert_backward_ir0,
    append_moe_full_train_router_dx_ir0,
    append_moe_full_train_input_gradient_ir0,
    append_moe_full_train_layer1_backbone_ir0,
    append_moe_full_train_layer1_attention_ir0,
    append_moe_full_train_layer1_qkv_ir0,
    append_moe_full_train_layer0_moe_ir0,
    append_moe_full_train_layer0_attention_ir0,
    append_moe_full_train_layer0_qkv_ir0,
    append_moe_full_train_embedding_wgrad_ir0,
)


class MoeAllParameterSgdSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()

    def test_both_steps_cover_exact_19_owned_states(self):
        for step in (0, 1):
            phase, sequence, _placement, _physical = (
                build_single_die_moe_train_physical_source(Fixture, step=step)
            )
            graph = phase
            for transform in _REVERSE:
                graph = transform(graph)
            self.assertEqual(graph.nodes[-1].kind,
                             OpKind.EMBEDDING_TABLE_WGRAD)
            result = append_moe_full_train_all_parameter_sgd_ir0(
                graph, sequence)
            updates = result.nodes[-19:]
            self.assertEqual({node.kind for node in updates},
                             {OpKind.OPTIMIZER_UPDATE})
            self.assertEqual(len({node.inputs[0] for node in updates}), 19)
            self.assertEqual(len({node.inputs[1] for node in updates}), 19)
            self.assertEqual(len({access.state_ref for access in
                                  result.state_accesses
                                  if access.mode is StateAccessMode.READ_WRITE}), 19)
            result.validate()

    def test_wrong_source_and_duplicate_fail_closed(self):
        phase, sequence, _placement, _physical = (
            build_single_die_moe_train_physical_source(Fixture, step=0)
        )
        with self.assertRaises(SchemaError):
            append_moe_full_train_all_parameter_sgd_ir0(phase.graph, sequence)
        graph = phase
        for transform in _REVERSE:
            graph = transform(graph)
        result = append_moe_full_train_all_parameter_sgd_ir0(graph, sequence)
        with self.assertRaises(SchemaError):
            append_moe_full_train_all_parameter_sgd_ir0(result, sequence)


if __name__ == "__main__":
    unittest.main()
