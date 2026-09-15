"""Old late-combine dExpert causal cycle and versioned early DAG faults."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_router_dexpert_handoff import (
    build_moe_router_dexpert_handoff,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_native_protocol import (
    build_moe_router_native_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_pre_dispatch_dag import (
    build_moe_router_pre_dispatch_proposal,
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


class MoeFullTrainRouterPreDispatchDagTest(unittest.TestCase):
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
        cls.producers = {(step, layer):
                         f"UNBOUND.shared.dcombined.step{step}.layer{layer}"
                         for step in (0, 1) for layer in (0, 1)}
        cls.proposal = build_moe_router_pre_dispatch_proposal(
            cls.score, cls.returned, cls.native, cls.handoff,
            Fixture.sequence, dcombined_producer_refs=cls.producers)

    def test_two_steps_two_layers_have_early_before_both_local_and_remote_experts(self):
        self.proposal.validate_topology()
        self.assertEqual(len(self.proposal.paths), 4)
        self.assertEqual(len(self.proposal.nodes_by_step_layer), 4)
        for path in self.proposal.paths:
            self.assertNotEqual(path.early_action_ref,
                                path.late_original_combine_backward_ref)
            self.assertEqual(path.logical_flops, 48)
            self.assertEqual(len(path.backward_send_refs), 1)
            self.assertEqual(len(path.local_expert_dgrad_refs), 1)
            self.assertEqual(len(path.remote_expert_dgrad_refs), 1)
            nodes = {item.action_ref: item for step, layer, values in
                     self.proposal.nodes_by_step_layer
                     if (step, layer) == (path.step, path.layer)
                     for item in values}
            self.assertIn(path.early_action_ref,
                          nodes[path.backward_send_refs[0]].predecessors)
            self.assertIn(path.early_action_ref,
                          nodes[path.local_expert_dgrad_refs[0]].predecessors)
            self.assertIn(path.required_dcombined_producer_ref,
                          nodes[path.early_action_ref].predecessors)

    def test_old_p2_has_only_late_combine_and_must_fail_production_source(self):
        with self.assertRaisesRegex(SchemaError,
                                    "old P2 has no official source-backed pre-dispatch"):
            self.proposal.require_production_source(Fixture.sequence)

    def test_reusing_original_late_action_as_pre_dispatch_creates_actual_cycle(self):
        step, layer, nodes = self.proposal.nodes_by_step_layer[0]
        path = self.proposal.paths[0]
        forged = tuple(replace(item, predecessors=(*item.predecessors,
                    path.late_original_combine_backward_ref))
                    if item.action_ref == path.early_action_ref else item
                    for item in nodes)
        changed = replace(self.proposal,
            nodes_by_step_layer=((step, layer, forged),
                                 *self.proposal.nodes_by_step_layer[1:]))
        with self.assertRaisesRegex(SchemaError, "physical dependency cycle"):
            changed.validate_topology()

    def test_early_missing_send_local_dgrad_or_real_upstream_edge_fails(self):
        step, layer, nodes = self.proposal.nodes_by_step_layer[0]
        path = self.proposal.paths[0]
        for missing in (path.backward_send_refs[0],
                        path.local_expert_dgrad_refs[0],
                        path.early_action_ref):
            dependency = (path.required_dcombined_producer_ref
                          if missing == path.early_action_ref
                          else path.early_action_ref)
            forged = tuple(replace(item, predecessors=tuple(
                    pred for pred in item.predecessors if pred != dependency))
                    if item.action_ref == missing else item
                    for item in nodes)
            changed = replace(self.proposal,
                nodes_by_step_layer=((step, layer, forged),
                                     *self.proposal.nodes_by_step_layer[1:]))
            with self.subTest(missing=missing), self.assertRaisesRegex(
                    SchemaError, "must dominate|must consume actual"):
                changed.validate_topology()

    def test_same_shared_producer_across_two_training_steps_is_rejected(self):
        forged = dict(self.producers)
        forged[(1, 1)] = forged[(0, 0)]
        with self.assertRaisesRegex(SchemaError,
                                    "each step/layer needs its own exact"):
            build_moe_router_pre_dispatch_proposal(
                self.score, self.returned, self.native,
                self.handoff, Fixture.sequence,
                dcombined_producer_refs=forged)


if __name__ == "__main__":
    unittest.main()
