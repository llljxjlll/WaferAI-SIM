"""Real cross-DP N4 routes produce typed N5 physical tasks and true SGD edges."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.full_dense_training_dp2_tasks import project_dense_dp2_tasks
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind
from llm.test.frontend.unit import test_full_dense_training_dp2_n4 as n4_fixture


class FullDenseDP2N5PhysicalTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        n4_fixture.FullDenseDP2N4Test.setUpClass()
        cls.source = n4_fixture.FullDenseDP2N4Test.result.dp_gradient_routes
        cls.projected = project_dense_dp2_tasks(cls.source)

    def test_exact_task_route_and_sgd_boundaries(self) -> None:
        self.projected.validate_against(self.source)
        self.assertEqual(len(self.projected.tasks), 480)
        self.assertEqual({item.die_id for item in self.projected.tasks}, {0, 1, 2, 3})
        self.assertEqual(sum(item.flow is not None for item in self.projected.tasks), 240)
        self.assertEqual(sum(item.task.kind is SemanticTaskKind.REDUCE
                             for item in self.projected.tasks), 60)
        self.assertEqual(sum(item.producer_task_ref is not None
                             for item in self.projected.tasks), 120)
        self.assertEqual(sum(item.consumer_task_ref is not None
                             for item in self.projected.tasks), 120)
        self.assertEqual({item.task.bytes for item in self.projected.tasks
                          if item.flow is not None}, {32, 128, 256, 384, 512})

    def test_dropped_sum_or_fake_sgd_dependency_fails_closed(self) -> None:
        shortened = replace(self.projected, tasks=self.projected.tasks[1:])
        with self.assertRaisesRegex(SchemaError, "coverage must be exact"):
            shortened.validate_against(self.source)
        victim = next(i for i, item in enumerate(self.projected.tasks)
                      if item.consumer_task_ref is not None)
        edited = list(self.projected.tasks)
        edited[victim] = replace(edited[victim], consumer_task_ref="task.fake.sgd")
        with self.assertRaisesRegex(SchemaError, "coverage must be exact"):
            replace(self.projected, tasks=tuple(edited)).validate_against(self.source)


if __name__ == "__main__":
    unittest.main()
