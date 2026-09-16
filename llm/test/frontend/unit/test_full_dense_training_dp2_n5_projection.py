"""Complete real TP2/DP2 source through official cross-replica N5 carrier."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import schedule_train_forward
from llm.frontend.wafer_frontend.policies.registry import RegistryKind, production_registry
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind, StandaloneNodeOrigin
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.n5 import ProjectToIR2Context, IntraDieSchedulingContext
from llm.test.frontend.unit import test_full_dense_training_dp2_n4 as n4_fixture


class FullDenseDP2N5ProjectionTest(unittest.TestCase):
    @classmethod
    @builder_validation_session()
    def setUpClass(cls) -> None:
        n4_fixture.FullDenseDP2N4Test.setUpClass()
        cls.source = n4_fixture.FullDenseDP2N4Test.result
        cls.context = ProjectToIR2Context.create(
            producer_pass="full_dense_dp2_n5_projection_test",
            state_transfers=(),
        )
        cls.projected = project_train_forward(cls.source, cls.context)

    @builder_validation_session()
    def test_60_sync_nodes_have_exact_physical_route_and_sgd_edges(self) -> None:
        projected = self.projected
        projected.validate_against(self.source, self.context)
        self.assertEqual(len(projected.replicas), 2)
        self.assertEqual(len(projected.dp_projected_tasks.tasks), 480)
        self.assertEqual([len(replica.projection.dp_sync_refs)
                          for replica in projected.replicas], [60, 60])
        self.assertEqual([
            sum(1 for dag in replica.projection.dags for task in dag.tasks
                if task.kind is SemanticTaskKind.REDUCE
                and isinstance(task.origin_ref, StandaloneNodeOrigin)
                and task.origin_ref.collective_plan_id
                == projected.dp_gradient_routes.id)
            for replica in projected.replicas
        ], [60, 0])
        for physical in projected.dp_projected_tasks.tasks:
            replica = projected.replicas[physical.replica_index]
            dag = next(dag for dag in replica.projection.dags
                       if dag.die_id == physical.die_id)
            tasks = {task.id: task for task in dag.tasks}
            task = tasks[physical.task.id]
            self.assertEqual(task.origin_ref.collective_plan_id,
                             projected.dp_gradient_routes.id)
            if physical.producer_task_ref is not None:
                self.assertIn(physical.producer_task_ref, task.deps)
            if physical.consumer_task_ref is not None:
                self.assertIn(task.id, tasks[physical.consumer_task_ref].deps)

    @builder_validation_session()
    def test_dp2_schedule_has_four_physical_dies_and_real_reduce_buffers(self) -> None:
        registry = production_registry()
        context = IntraDieSchedulingContext.create(
            producer_pass="full_dense_dp2_n5_projection_test",
            policy=registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection,
        )
        scheduled = schedule_train_forward(self.projected, context)
        scheduled.validate_against(self.projected, context)
        active = tuple(
            schedule for replica in scheduled.replicas
            for schedule in replica.schedule_set.schedules
            if schedule.placements
        )
        self.assertEqual({item.die_id for item in active}, {0, 1, 2, 3})
        dp_plan = self.projected.dp_gradient_routes
        dp_flow_ids = {task.flow.id for task in self.projected.dp_projected_tasks.tasks
                       if task.flow is not None}
        self.assertEqual(sum(binding.flow_id in dp_flow_ids
                             for schedule in active
                             for binding in schedule.flow_routes), 240)
        self.assertTrue(all(binding.pair_route_ref in {
            route.id for group in dp_plan.dp_groups
            for route in group.embedding.routes
        } for schedule in active for binding in schedule.flow_routes
            if binding.flow_id in dp_flow_ids))

    def test_missing_cross_replica_task_proof_fails_closed(self) -> None:
        with self.assertRaises(SchemaError):
            replace(self.projected, dp_projected_tasks=None).validate()
        replica = self.projected.replicas[1]
        forged = replace(replica, dp_projected_tasks=None)
        with self.assertRaises(SchemaError):
            replace(self.projected,
                    replicas=(self.projected.replicas[0], forged)).validate()


if __name__ == "__main__":
    unittest.main()
