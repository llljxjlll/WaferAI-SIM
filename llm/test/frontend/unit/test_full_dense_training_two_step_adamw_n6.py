"""Every two-step AdamW state DMA and compute must reach native N6."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.train_global_action import build_train_global_action
from llm.frontend.wafer_frontend.passes.train_lower_program import lower_train
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind, StateUseAccess
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragments
from llm.test.frontend.unit.test_full_dense_training_two_step_adamw_n5 import (
    FullDenseTwoStepAdamwN5Test,
)


class FullDenseTwoStepAdamwN6Test(unittest.TestCase):
    @classmethod
    @builder_validation_session()
    def setUpClass(cls) -> None:
        FullDenseTwoStepAdamwN5Test.setUpClass()
        cls.actions = build_train_global_action(
            FullDenseTwoStepAdamwN5Test.scheduled,
        )
        cls.replica = cls.actions.replicas[0]
        cls.native = lower_train(cls.actions)

    def test_real_520_action_native_timeline_carries_thirty_adamw_records(self) -> None:
        self.assertEqual(len(self.replica.global_dag.actions), 520)
        self.assertEqual(len(self.native.replicas[0].fragments), 520)
        self.assertEqual(sum(
            record.opcode is RecordOpcode.ADAMW_UPDATE
            for fragment in _leaf_fragments(self.native.replicas[0].fragments)
            for stream in fragment.core_streams
            for record in stream.records
        ), 30)
        self.assertEqual(sum(
            action.task_kind is SemanticTaskKind.DMA_OUT
            and action.state_uses[0].access is StateUseAccess.WRITE
            for action in self.replica.global_dag.actions
        ), 150)

    def test_losing_one_step_counter_store_cannot_pass_action_coverage(self) -> None:
        action = next(action for action in self.replica.global_dag.actions
                      if action.task_kind is SemanticTaskKind.DMA_OUT)
        forged = replace(self.replica, global_dag=replace(
            self.replica.global_dag,
            actions=tuple(item for item in self.replica.global_dag.actions
                          if item.id != action.id),
        ))
        graph = self.replica.scheduled.projected.graph
        manifest = graph.persistent_state_manifest
        with self.assertRaisesRegex(SchemaError, "state actions must cover"):
            forged._validate_full_dense_sgd_actions(
                graph, manifest, "missing_optimizer_store",
            )


if __name__ == "__main__":
    unittest.main()
