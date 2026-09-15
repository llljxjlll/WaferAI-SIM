"""P2 expert gradient owner, dynamic dExpert source and old physical failure."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_router_dexpert_handoff import (
    build_moe_router_dexpert_handoff,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_native_protocol import (
    build_moe_router_native_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_return_protocol import (
    build_moe_router_signed_return_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_score_source import (
    build_moe_trainable_signed_router_requirements,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeFullTrainRouterDexpertHandoffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.score = build_moe_trainable_signed_router_requirements(
            Fixture.sequence)
        cls.returned = build_moe_router_signed_return_protocol(
            cls.score, Fixture.sequence)
        cls.native = build_moe_router_native_protocol(
            cls.score, cls.returned, Fixture.sequence)
        cls.handoff = build_moe_router_dexpert_handoff(
            cls.score, cls.returned, cls.native, Fixture.sequence)

    def test_four_l2_two_step_grouped_gradient_handoffs(self):
        self.handoff.validate_against(self.score, self.returned,
                                      self.native, Fixture.sequence)
        self.assertEqual(len(self.handoff.segments), 8)
        for step in (0, 1):
            for layer in (0, 1):
                pair = [segment for segment in self.handoff.segments
                        if (segment.step, segment.layer) == (step, layer)]
                self.assertEqual([(segment.source_rank,
                                   segment.expert_home_rank,
                                   segment.expert_index,
                                   segment.source_offset_bytes,
                                   segment.size_bytes,
                                   segment.assignment_refs)
                                  for segment in pair], [
                    (0, 0, 0, 0, 16, ("assignment.0", "assignment.2")),
                    (0, 1, 1, 16, 16, ("assignment.1", "assignment.3")),
                ])
                self.assertIsNone(pair[0].backward_flow_ref)
                self.assertIsNone(pair[0].send_action_ref)
                self.assertIsNone(pair[0].recv_action_ref)
                self.assertIsNotNone(pair[1].backward_flow_ref)
                self.assertIsNotNone(pair[1].send_action_ref)
                self.assertIsNotNone(pair[1].recv_action_ref)
                self.assertNotEqual(pair[0].dgrad_action_ref,
                                    pair[1].dgrad_action_ref)

    def test_old_static_leaf_remains_physical_fail(self):
        with self.assertRaisesRegex(SchemaError,
                                    "public opcode 0x28 is not implemented"):
            self.handoff.require_physical_dexpert_consumption(
                self.score, self.returned, self.native, Fixture.sequence)

    def test_remote_gradient_overlap_or_short_bytes_fail_source_gate(self):
        segments = list(self.handoff.segments)
        remote = segments[1]
        for incorrect in (replace(remote, source_offset_bytes=0),
                          replace(remote, size_bytes=8),
                          replace(remote, backward_flow_ref=None),
                          replace(remote, dgrad_action_ref=segments[0].dgrad_action_ref)):
            with self.subTest(incorrect=incorrect), self.assertRaisesRegex(
                    SchemaError, "expert dY route"):
                segments[1] = incorrect
                replace(self.handoff,
                        segments=tuple(segments)).validate_against(
                            self.score, self.returned, self.native,
                            Fixture.sequence)

    def test_dual_output_missing_expert_bytes_fail_source_gate(self):
        pair = self.native.pairs[0]
        incorrect = replace(pair.score_backward,
                            hidden_size=pair.score_backward.hidden_size - 1)
        with self.assertRaisesRegex(SchemaError,
                                    "router FWD/BWD op geometry"):
            build_moe_router_dexpert_handoff(self.score, self.returned,
                replace(self.native, pairs=(replace(pair,
                    score_backward=incorrect), *self.native.pairs[1:])),
                Fixture.sequence)


if __name__ == "__main__":
    unittest.main()
