"""Real two-replica native Dense gradients and HBM state versions fail closed."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.full_dense_gradient_physical_gate import (
    require_full_dense_physical_gradient_paths,
)
from llm.frontend.wafer_frontend.lowering.full_dense_two_step_physical_dag import (
    build_full_dense_two_step_physical_dag, dense_two_step_native_opcode_contract,
)
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.dense_state_version_fence import (
    dense_dp2_state_version_fences,
)
from llm.frontend.wafer_frontend.schema.full_dense_gradient_requirements import (
    build_dense_full_train_requirements,
)
from llm.test.frontend.unit import test_full_dense_training_dp2_native_link as linked_fixture
from llm.test.frontend.unit import test_full_dense_training_dp2_n4 as plan_fixture


class FullDenseDP2PhysicalGradientGateTest(unittest.TestCase):
    @classmethod
    @builder_validation_session()
    def setUpClass(cls) -> None:
        linked_fixture.FullDenseDP2NativeLinkTest.setUpClass()
        cls.linked = linked_fixture.FullDenseDP2NativeLinkTest.linked
        cls.plan = plan_fixture.FullDenseDP2N4Test.plan
        cls.requirements = build_dense_full_train_requirements(cls.plan, steps=2)
        cls.physical = build_full_dense_two_step_physical_dag(
            cls.linked, cls.plan, cls.requirements,
        )
        cls.backward, cls.wgrad = dense_two_step_native_opcode_contract(cls.plan)
        cls.fences = tuple(fence for replica in cls.linked.source.replicas
                           for fence in dense_dp2_state_version_fences(
                               replica.lowering_context.global_dag))

    @builder_validation_session()
    def test_all_120_gradients_and_22_real_cross_core_state_fences(self) -> None:
        self.assertEqual(len(self.requirements.paths), 120)
        self.assertEqual(len(self.physical.actions), 2176)
        self.assertEqual(len(self.physical.transport_edges), 248)
        self.assertEqual(len(self.physical.state_version_edges), 82)
        self.assertEqual(len(self.fences), 22)
        self.assertEqual(sum(RecordOpcode.EVENT_SET in {
            opcode for action in self.physical.actions
            if action.id == fence.store_action_id
            for _fragment, _index, opcode in action.executable_records
        } and RecordOpcode.EVENT_WAIT in {
            opcode for action in self.physical.actions
            if action.id == fence.load_action_id
            for _fragment, _index, opcode in action.executable_records
        } for fence in self.fences), 22)
        require_full_dense_physical_gradient_paths(
            self.linked.manifest, self.plan, self.requirements, self.physical,
            required_backward_opcodes=self.backward,
            required_wgrad_opcodes=self.wgrad,
        )

    @builder_validation_session()
    def test_missing_event_or_version_edge_fails_closed(self) -> None:
        fence = self.fences[0]
        store = next(action for action in self.physical.actions
                     if action.id == fence.store_action_id)
        forged = replace(store, executable_records=tuple(
            record for record in store.executable_records
            if record[2] is not RecordOpcode.EVENT_SET))
        missing_event = replace(self.physical, actions=tuple(
            forged if action.id == store.id else action
            for action in self.physical.actions))
        with self.assertRaisesRegex(SchemaError, "witness every carrier record"):
            require_full_dense_physical_gradient_paths(
                self.linked.manifest, self.plan, self.requirements, missing_event,
                required_backward_opcodes=self.backward,
                required_wgrad_opcodes=self.wgrad,
            )
        missing_version = replace(self.physical, state_version_edges=tuple(
            edge for edge in self.physical.state_version_edges
            if edge != (fence.store_action_id, fence.load_action_id)))
        with self.assertRaisesRegex(SchemaError, "next-step HBM LOAD version"):
            require_full_dense_physical_gradient_paths(
                self.linked.manifest, self.plan, self.requirements, missing_version,
                required_backward_opcodes=self.backward,
                required_wgrad_opcodes=self.wgrad,
            )

    def test_missing_source_store_cannot_define_a_fence(self) -> None:
        replica = self.linked.source.replicas[1]
        fence = self.fences[0]
        forged = replace(replica.lowering_context.global_dag, actions=tuple(
            action for action in replica.lowering_context.global_dag.actions
            if action.id != fence.store_action_id))
        with self.assertRaisesRegex(SchemaError, "lacks a store/read pair"):
            dense_dp2_state_version_fences(forged)


if __name__ == '__main__':
    unittest.main()
