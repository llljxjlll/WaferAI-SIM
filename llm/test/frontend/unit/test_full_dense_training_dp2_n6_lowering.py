"""True 2x2 Dense training DP gradients must lower to source-bound native records."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.dense_dp_sync import lower_dense_dp_gradient
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import schedule_train_forward
from llm.frontend.wafer_frontend.passes.train_global_action import build_train_global_action
from llm.frontend.wafer_frontend.passes.train_lower_program import lower_train
from llm.frontend.wafer_frontend.policies.registry import RegistryKind, production_registry
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.ir2 import StandaloneNodeOrigin
from llm.frontend.wafer_frontend.schema.n5 import ProjectToIR2Context, IntraDieSchedulingContext
from llm.frontend.wafer_frontend.schema.train_n6 import train_replica_lowering_context
from llm.test.frontend.unit import test_full_dense_training_dp2_n4 as fixture


class FullDenseDP2N6LoweringTest(unittest.TestCase):
    @classmethod
    @builder_validation_session()
    def setUpClass(cls) -> None:
        fixture.FullDenseDP2N4Test.setUpClass()
        projected = project_train_forward(
            fixture.FullDenseDP2N4Test.result,
            ProjectToIR2Context.create(
                producer_pass="dense_dp2_n6_source_test", state_transfers=(),
            ),
        )
        scheduled = schedule_train_forward(
            projected,
            IntraDieSchedulingContext.create(
                producer_pass="dense_dp2_n6_source_test",
                policy=production_registry().instantiate(
                    RegistryKind.INTRA_DIE, "naive",
                ).selection,
            ),
        )
        cls.source = build_train_global_action(scheduled)
        cls.native = lower_train(cls.source)

    def test_every_gradient_has_real_native_dte_wait_copy_and_fp32_sum(self) -> None:
        self.assertEqual([len(replica.fragments) for replica in self.native.replicas],
                         [644, 644])
        dp_plan = self.source.replicas[0].scheduled.projected.dp_gradient_routes
        for dp, replica in enumerate(self.native.replicas):
            fragments = tuple(fragment for fragment in replica.fragments
                              if fragment.producer_pass == "dense_dp_gradient_lowering")
            self.assertEqual(len(fragments), len(dp_plan.gradients))
            actions = {action.id: action for action in
                       self.source.replicas[dp].global_dag.actions}
            claimed = [ref for fragment in fragments for ref in fragment.claimed_action_ids]
            self.assertEqual(len(claimed), 300 if dp == 0 else 180)
            self.assertEqual(len(claimed), len(set(claimed)))
            self.assertTrue(all(isinstance(actions[ref].origin_ref,
                                           StandaloneNodeOrigin)
                                and actions[ref].origin_ref.collective_plan_id == dp_plan.id
                                for ref in claimed))
            opcodes = [record.opcode for fragment in fragments
                       for stream in fragment.core_streams for record in stream.records]
            self.assertEqual(opcodes.count(RecordOpcode.DTE_SEND), 60)
            self.assertEqual(opcodes.count(RecordOpcode.DTE_RECV), 60)
            self.assertEqual(opcodes.count(RecordOpcode.DTE_WAIT),
                             120 if dp == 0 else 60)
            self.assertEqual(opcodes.count(RecordOpcode.LOCAL_REDUCE),
                             60 if dp == 0 else 0)
            if dp == 0:
                self.assertEqual(opcodes.count(RecordOpcode.DTE_ISSUE), 60)

    @builder_validation_session()
    def test_deleted_recv_and_tampered_source_bytes_fail_closed(self) -> None:
        replica = self.source.replicas[0]
        context = train_replica_lowering_context(replica)
        gradient = context.dp_route_plan.gradients[0]
        source_ids = {source.id for source in gradient.rank_programs[0].actions}
        actions = tuple(action for action in replica.global_dag.actions
                        if isinstance(action.origin_ref, StandaloneNodeOrigin)
                        and action.origin_ref.collective_plan_id == context.dp_route_plan.id
                        and action.origin_ref.action_id in source_ids)
        recv = next(action for action in actions if action.task_kind.value == "recv")
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            lower_dense_dp_gradient(tuple(action for action in actions
                                          if action is not recv), context, 0)
        with self.assertRaisesRegex(SchemaError, "drifted"):
            lower_dense_dp_gradient(tuple(replace(action, bytes=action.bytes + 4)
                                          if action is recv else action
                                          for action in actions), context, 0)


if __name__ == "__main__":
    unittest.main()
