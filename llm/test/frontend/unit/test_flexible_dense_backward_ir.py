from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_backward_projection import (
    build_flexible_dense_backward_lineage,
)
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
)
from llm.frontend.wafer_frontend.schema.flexible_dense_train import (
    FlexibleDenseTrainActionKind,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FlexibleDenseBackwardLineageTest(unittest.TestCase):
    def test_all_rectangles_have_exact_rank_major_lineage(self) -> None:
        for rows in range(1, 11):
            for columns in range(1, 11):
                with self.subTest(rows=rows, columns=columns):
                    plan = build_flexible_dense_train_plan(
                        _spec(rows, columns), RectMeshSpec(rows, columns)
                    )
                    ir, projection, schedule, global_dag = (
                        build_flexible_dense_backward_lineage(plan)
                    )
                    self.assertEqual(ir.source_plan_id, plan.id)
                    self.assertEqual(ir.rank_count, rows * columns)
                    self.assertEqual(
                        tuple(action.action_id for action in ir.actions),
                        projection.action_ids,
                    )
                    self.assertEqual(
                        tuple(
                            item
                            for stream in schedule.rank_action_ids
                            for item in stream
                        ),
                        global_dag.action_ids,
                    )
                    expected_edges = {
                        (dependency, action.id)
                        for action in plan.rank_actions
                        for dependency in action.depends_on
                    }
                    self.assertEqual(
                        set(global_dag.dependency_edges), expected_edges
                    )
                    expected_owners = {
                        (template.state_ref, rank)
                        for template in plan.parameter_templates
                        for rank in template.owner_ranks
                    }
                    self.assertEqual(
                        set(projection.state_owner_pairs), expected_owners
                    )
                    backward = tuple(
                        action
                        for action in ir.actions
                        if action.kind is FlexibleDenseTrainActionKind.BACKWARD
                    )
                    self.assertTrue(backward)
                    self.assertTrue(
                        all(action.tape_origin_ref for action in backward)
                    )

    def test_digest_and_dependency_tampering_fail_closed(self) -> None:
        plan = build_flexible_dense_train_plan(_spec(2, 3), RectMeshSpec(2, 3))
        ir, projection, schedule, global_dag = (
            build_flexible_dense_backward_lineage(plan)
        )
        with self.assertRaisesRegex(SchemaError, "unstable"):
            replace(ir, source_plan_digest="0" * 64).validate()
        with self.assertRaisesRegex(SchemaError, "escapes"):
            replace(
                global_dag,
                dependency_edges=(("forged", global_dag.action_ids[0]),),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unstable"):
            replace(schedule, source_projection_id="forged").validate()
        projection.validate()


if __name__ == "__main__":
    unittest.main()
