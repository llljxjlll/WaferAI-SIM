from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import build_global_action_dag
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_global_action_schema import _create_global, _with_same_core_predecessor
from test_ir2_route_schedule import _two_by_two_case, _two_die_case


class GlobalActionBuilderTest(unittest.TestCase):
    def test_two_die_build_is_exact_canonical_and_deterministic(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        built = build_global_action_dag(ir1, projection, schedule_set)
        built.validate_against(ir1, projection, schedule_set)
        expected = _create_global(ir1, projection, schedule_set)
        self.assertEqual(built.id, expected.id)
        self.assertEqual(built._semantic_key(), expected._semantic_key())
        self.assertEqual(
            tuple(action.source.task_id for action in built.actions),
            tuple(task.id for dag in projection.dags for task in dag.tasks),
        )
        self.assertEqual(
            canonical_digest(build_global_action_dag(ir1, projection, schedule_set)),
            canonical_digest(built),
        )

    def test_semantic_and_same_core_predecessor_are_ordered_unique(self) -> None:
        ir1, projection, schedule_set, expected = _with_same_core_predecessor()
        built = build_global_action_dag(ir1, projection, schedule_set)
        self.assertEqual(built.id, expected.id)
        self.assertEqual(built._semantic_key(), expected._semantic_key())
        predecessor, send, _recv = built.actions
        self.assertEqual(send.deps, (predecessor.id,))

    def test_transit_remains_coreless_but_preserves_route(self) -> None:
        ir1, projection, schedule_set = _two_by_two_case()
        built = build_global_action_dag(ir1, projection, schedule_set)
        transit = next(
            action for action in built.actions
            if action.task_kind is SemanticTaskKind.TRANSIT
        )
        self.assertIsNone(transit.logical_core)
        self.assertIsNone(transit.core_order_index)
        self.assertIsNotNone(transit.flow)
        self.assertIsNotNone(transit.flow_route)

    def test_input_type_and_provenance_are_fail_closed(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        for arguments, message in (
            ((object(), projection, schedule_set), "ir1"),
            ((ir1, object(), schedule_set), "projection"),
            ((ir1, projection, object()), "schedule_set"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(SchemaError, message):
                build_global_action_dag(*arguments)
        forged = replace(schedule_set, source_ir1_id="forged")
        with self.assertRaises(SchemaError):
            build_global_action_dag(ir1, projection, forged)

    def test_public_result_is_the_global_action_dag_contract(self) -> None:
        ir1, projection, schedule_set = _two_die_case()
        self.assertIsInstance(
            build_global_action_dag(ir1, projection, schedule_set),
            GlobalActionDAG,
        )


if __name__ == "__main__":
    unittest.main()
