"""Official EP2 P2 signed pre-dispatch source; runtime is not implied."""

from dataclasses import replace
import hashlib
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.flexible_moe_production import (
    lower_link_flexible_moe_production,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_score_source import (
    build_moe_trainable_signed_router_requirements,
)
from llm.frontend.wafer_frontend.passes.flexible_moe import (
    compile_flexible_moe_baseline,
    compile_flexible_moe_signed_top1_train_source,
)
from llm.frontend.wafer_frontend.schema.flexible_moe import (
    MoeRectActionKind, MoeRectFlowStage, MoeRectSignedTop1TrainSource,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeSignedRouterTrainSourcePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.unit = Fixture.sequence.units[0]
        cls.spec = cls.unit.spec
        cls.original = compile_flexible_moe_baseline(cls.spec)
        cls.score_requirements = build_moe_trainable_signed_router_requirements(
            Fixture.sequence,
        )
        cls.source = MoeRectSignedTop1TrainSource.create(
            cls.spec,
            dynamic_case_ref=cls.score_requirements.dynamic_score_case_ref,
            shared_dcombined_producer_ref="REQUIRED.real_shared_spine.backward.dcombined.step0.layer0",
            shared_full_model_manifest_ref="REQUIRED.real_shared_dense_global_training_timeline",
        )
        cls.plan = compile_flexible_moe_signed_top1_train_source(
            cls.spec, cls.source,
        )

    def test_old_p2_exact_baseline_and_opt_in_default_rejection(self):
        self.original.validate_against(self.spec)
        self.assertEqual(self.original, self.unit.plan)
        self.assertEqual(compile_flexible_moe_baseline(self.spec).id,
                         self.original.id)
        self.assertNotEqual(self.original.id, self.plan.id)
        with self.assertRaisesRegex(SchemaError, "exact versioned source binding"):
            self.plan.validate_against(self.spec)
        with self.assertRaisesRegex(SchemaError, "baseline plan cannot carry"):
            self.original.validate_against(self.spec,
                                           signed_source=self.source)
        self.plan.validate_against_signed(self.spec, self.source)
        with self.assertRaisesRegex(SchemaError, "baseline cannot carry versioned"):
            replace(self.plan, producer_pass="compile_flexible_moe_baseline").validate_against(self.spec)
        with self.assertRaisesRegex(SchemaError, "exact versioned source binding"):
            lower_link_flexible_moe_production(self.plan, self.spec,
                                               full_model_dataflow=True)

    def test_two_steps_two_layers_keep_source_assignment_and_distinct_signed_ids(self):
        self.assertEqual(len(Fixture.sequence.units), 4)
        ids = set()
        for unit in Fixture.sequence.units:
            source = MoeRectSignedTop1TrainSource.create(
                unit.spec,
                dynamic_case_ref=self.score_requirements.dynamic_score_case_ref,
                shared_dcombined_producer_ref=(
                    f"REQUIRED.real_shared_spine.backward.dcombined.step{unit.step}.layer{unit.layer}"
                ),
                shared_full_model_manifest_ref=(
                    "REQUIRED.real_shared_dense_global_training_timeline"
                ),
            )
            plan = compile_flexible_moe_signed_top1_train_source(
                unit.spec, source,
            )
            ids.add(source.id)
            plan.validate_against_signed(unit.spec, source)
            self.assertEqual(source.route_bytes, 80)
            self.assertEqual(plan.flows, unit.plan.flows)
            self.assertEqual(plan.state_bindings, unit.plan.state_bindings)
        self.assertEqual(len(ids), 4)

    def test_route_blob_is_exact_nonzero_frozen_p2_assignment_and_80_bytes(self):
        expected = tuple(value for assignment in self.spec.trace.assignments
                         for value in (assignment.token_index,
                                       assignment.source_rank,
                                       assignment.expert_index,
                                       assignment.expert_home_rank,
                                       assignment.slot_index))
        self.assertEqual(self.source.route_words, expected)
        self.assertEqual(self.source.route_bytes, 80)
        self.assertEqual(len(self.source.route_blob), 80)
        self.assertTrue(any(self.source.route_blob))
        self.assertEqual(self.source.route_sha256,
                         hashlib.sha256(self.source.route_blob).hexdigest())
        self.assertEqual(self.plan.flows, self.original.flows)
        self.assertEqual(self.plan.state_bindings, self.original.state_bindings)
        self.assertEqual(self.plan.gate_all_reduce,
                         self.original.gate_all_reduce)

    def test_early_dependency_dominates_transport_and_local_expert_reverse(self):
        index = {item.id: item for item in self.plan.actions}
        flow_index = {item.id: item for item in self.plan.flows}
        shared, = (item for item in self.plan.actions if item.kind is
                   MoeRectActionKind.SHARED_DCOMBINED_IMPORT)
        score, = (item for item in self.plan.actions if item.kind is
                  MoeRectActionKind.SCORE_WEIGHT_BACKWARD_PRE_DISPATCH)
        self.assertEqual((shared.rank, score.rank), (0, 0))
        self.assertEqual((shared.logical_bytes, score.logical_bytes,
                          score.flops), (32, 48, 48))
        self.assertEqual(score.deps, (shared.deps[0], shared.id))
        remote_send = (item for item in self.plan.actions
                       if item.kind is MoeRectActionKind.SEND
                       and item.rank == 0 and item.flow_ref is not None
                       and flow_index[item.flow_ref].stage is
                       MoeRectFlowStage.BACKWARD_GRADIENT)
        local_reverse = (item for item in self.plan.actions
                         if item.kind in (MoeRectActionKind.EXPERT_DGRAD,
                                          MoeRectActionKind.EXPERT_WGRAD)
                         and item.rank == 0 and item.assignment_refs)
        self.assertTrue(remote_send)
        self.assertTrue(local_reverse)
        for action in (*remote_send, *local_reverse):
            self.assertIn(score.id, action.deps)
        self.assertEqual(len([item for item in self.plan.actions
                              if item.kind is MoeRectActionKind.COMBINE_BACKWARD]),
                         len([item for item in self.original.actions
                              if item.kind is MoeRectActionKind.COMBINE_BACKWARD]))
        for late in (item for item in self.plan.actions
                     if item.kind is MoeRectActionKind.COMBINE_BACKWARD):
            self.assertNotIn(score.id, late.deps)
            self.assertTrue(any(index[dep].kind is MoeRectActionKind.EXPERT_DGRAD
                                for dep in late.deps) or late.rank != 0)

    def test_fake_shared_import_or_score_edge_and_cycle_fail(self):
        score, = (item for item in self.plan.actions if item.kind is
                  MoeRectActionKind.SCORE_WEIGHT_BACKWARD_PRE_DISPATCH)
        late, = (item for item in self.plan.actions if item.kind is
                 MoeRectActionKind.COMBINE_BACKWARD and item.rank == 0)
        for changed, reason in (
            (replace(score, deps=(score.deps[0],)), "source binding|work and forward dependency"),
            (replace(score, deps=(*score.deps, late.id)), "cycle"),
        ):
            forged = replace(self.plan, actions=tuple(changed if item.id ==
                             score.id else item for item in self.plan.actions))
            from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
            from llm.frontend.wafer_frontend.schema.flexible_moe import FLEXIBLE_MOE_PLAN_SCHEMA_VERSION
            forged = replace(forged, id=stable_artifact_id(
                "flexible_moe_plan", forged._semantic(),
                schema_version=FLEXIBLE_MOE_PLAN_SCHEMA_VERSION))
            with self.subTest(reason=reason), self.assertRaisesRegex(
                    SchemaError, reason):
                forged.validate_against(self.spec, signed_source=self.source)

    def test_forged_route_or_named_source_is_rejected(self):
        forged = replace(self.source, route_words=(
            *self.source.route_words[:2], 1 - self.source.route_words[2],
            *self.source.route_words[3:],
        ))
        with self.assertRaisesRegex(SchemaError,
                                    "route table|artifact identity"):
            forged.validate_against(self.spec)
        with self.assertRaisesRegex(SchemaError, "action/flow/state DAG"):
            forged_plan = replace(self.plan, symbolic_file_bytes=
                                  self.plan.symbolic_file_bytes + 1)
            # Change plan ID to independently satisfy the ordinary stable hash.
            from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
            from llm.frontend.wafer_frontend.schema.flexible_moe import FLEXIBLE_MOE_PLAN_SCHEMA_VERSION
            forged_plan = replace(forged_plan, id=stable_artifact_id(
                "flexible_moe_plan", forged_plan._semantic(),
                schema_version=FLEXIBLE_MOE_PLAN_SCHEMA_VERSION))
            forged_plan.validate_against_signed(self.spec, self.source)


if __name__ == "__main__":
    unittest.main()
