"""Signed top1 score source geometry and real old-router physical rejection."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_router_score_source import (
    build_moe_trainable_signed_router_requirements,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeFullTrainRouterScoreSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.source = build_moe_trainable_signed_router_requirements(
            Fixture.sequence)

    def test_new_model_case_exact_source_score_and_frozen_choice(self):
        self.source.validate_against(Fixture.sequence)
        self.assertNotEqual(self.source.dynamic_score_case_ref,
                            self.source.original_static_case_ref)
        self.assertEqual(self.source.assignment_derivative,
                         "stop_gradient_top1_selection")
        self.assertEqual(self.source.score_weight,
                         "selected_raw_signed_score")
        self.assertEqual(len(self.source.paths), 8)
        self.assertEqual([(item.step, item.layer, item.source_rank)
                          for item in self.source.paths],
                         [(step, layer, rank) for step in (0, 1)
                          for layer in (0, 1) for rank in (0, 1)])
        for path in self.source.paths:
            if path.source_rank == 0:
                self.assertEqual(len(path.routes), 4)
                self.assertEqual(path.score_tape_bytes, 16)
                self.assertEqual(path.dscore_bytes, 16)
                self.assertEqual(path.forward_expert_bytes, 32)
                self.assertEqual(path.backward_dcombined_bytes, 32)
                self.assertEqual(path.fp32_gate_gradient_bytes, 32)
                self.assertEqual([(route.token_index, route.selected_expert)
                                  for route in path.routes], [(0, 0), (1, 1),
                                                              (2, 0), (3, 1)])
            else:
                self.assertEqual(path.routes, ())
                self.assertEqual(path.score_tape_bytes, 0)
                self.assertEqual(path.dscore_bytes, 0)
                self.assertEqual(path.fp32_gate_gradient_bytes, 32)

    def test_static_router_matmul_has_no_distinct_score_tape(self):
        with self.assertRaisesRegex(SchemaError,
                                    "independently retained signed score tape"):
            self.source.require_signed_gate_score_tape(Fixture.sequence)

    def test_static_copy_weighted_combine_never_reads_router_score(self):
        with self.assertRaisesRegex(SchemaError,
                                    "weighted combine does not consume genuine router score"):
            self.source.require_score_weighted_combine(Fixture.sequence)

    def test_static_backward_does_not_produce_router_score_derivative(self):
        with self.assertRaisesRegex(SchemaError,
                                    "lacks selected-score derivative"):
            self.source.require_signed_dscore_producer(Fixture.sequence)

    def test_router_old_fp16_matmul_cast_is_not_native_fp32_score_wgrad(self):
        with self.assertRaisesRegex(SchemaError,
                                    "old router FP16 MATMUL then cast"):
            self.source.require_native_fp32_router_wgrad(Fixture.sequence)

    def test_new_model_identity_or_any_token_route_change_is_rejected(self):
        with self.assertRaisesRegex(SchemaError,
                                    "source/hardware/route/score contract"):
            replace(self.source, dynamic_score_case_ref=
                    self.source.original_static_case_ref).validate_against(
                    Fixture.sequence)
        path = self.source.paths[0]
        route = path.routes[0]
        forged = replace(path, routes=(replace(
            route, selected_expert=1), *path.routes[1:]))
        with self.assertRaisesRegex(SchemaError,
                                    "source/hardware/route/score contract"):
            replace(self.source, paths=(forged, *self.source.paths[1:])
                    ).validate_against(Fixture.sequence)


if __name__ == "__main__":
    unittest.main()
