"""Strict source-bound 0x27/0x28 workload shape and two-output contract."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_router_native_protocol import (
    build_moe_router_native_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_return_protocol import (
    build_moe_router_signed_return_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_score_source import (
    build_moe_trainable_signed_router_requirements,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeFullTrainRouterNativeProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.score = build_moe_trainable_signed_router_requirements(
            Fixture.sequence)
        cls.returned = build_moe_router_signed_return_protocol(
            cls.score, Fixture.sequence)
        cls.native = build_moe_router_native_protocol(
            cls.score, cls.returned, Fixture.sequence)

    def test_every_layer_and_step_has_exact_source_bounded_dual_output_work(self):
        self.native.validate_against(self.score, self.returned,
                                     Fixture.sequence)
        self.assertEqual(len(self.native.pairs), 4)
        for pair in self.native.pairs:
            forward, backward = pair.weighted_forward, pair.score_backward
            self.assertEqual(pair.source_rank, 0)
            self.assertEqual(forward.rank_rows, 4)
            self.assertEqual((forward.hidden_size, forward.expert_count),
                             (4, 2))
            self.assertEqual(forward.source_dynamic_case_ref,
                             self.score.dynamic_score_case_ref)
            self.assertEqual(backward.source_weighted_forward_ref,
                             forward.id)
            self.assertEqual((forward.score_bytes,
                              forward.expert_return_bytes,
                              forward.combined_bytes), (16, 32, 32))
            self.assertEqual(backward.operand_bytes,
                             (16, 32, 32, 16, 32))
            self.assertEqual((forward.logical_flops,
                              backward.logical_flops), (32, 48))
            self.assertEqual(forward.byte_lanes(),
                             ((0, 0, 0), (6, 16, 8),
                              (8, 8, 16), (14, 24, 24)))
            self.assertEqual(backward.byte_lanes(),
                             forward.byte_lanes())

    def test_forward_expert_group_overlap_or_last_token_drop_rejected(self):
        forward = self.native.pairs[0].weighted_forward
        groups = forward.expert_groups
        with self.assertRaisesRegex(SchemaError, "contiguous distinct"):
            replace(forward, expert_groups=(
                groups[0], replace(groups[1], offset_bytes=8))).validate()
        with self.assertRaisesRegex(SchemaError,
                                    "one positive rank-local source"):
            replace(forward, rank_rows=3).validate()
        with self.assertRaisesRegex(SchemaError,
                                    "one ordered frozen selected expert"):
            replace(forward, routes=(replace(
                forward.routes[0], selected_expert=2), *forward.routes[1:])
                ).validate()

    def test_backward_must_write_two_owned_fp16_score_and_expert_gradients(self):
        backward = self.native.pairs[0].score_backward
        with self.assertRaisesRegex(SchemaError,
                                    "three FP16 input and two different FP16 output"):
            replace(backward, dexpert_dtype=DType.FP32).validate()
        with self.assertRaisesRegex(SchemaError,
                                    "one real weighted forward"):
            replace(backward, source_weighted_forward_ref="").validate()
        with self.assertRaisesRegex(SchemaError,
                                    "all source token expert outputs"):
            replace(backward, expert_groups=(
                backward.expert_groups[0],
                replace(backward.expert_groups[1], size_bytes=8)
                )).validate()

    def test_pair_source_case_action_and_return_identity_recomputed(self):
        pair = self.native.pairs[0]
        wrong_forward = replace(
            pair.weighted_forward,
            source_dynamic_case_ref=self.score.original_static_case_ref)
        with self.assertRaisesRegex(SchemaError,
                                    "geometry/dScore\+dExpert ownership"):
            replace(self.native, pairs=(
                replace(pair, weighted_forward=wrong_forward),
                *self.native.pairs[1:])
                ).validate_against(self.score, self.returned,
                                   Fixture.sequence)


if __name__ == "__main__":
    unittest.main()
