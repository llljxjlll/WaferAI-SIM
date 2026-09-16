"""Complete two-step Dense source task inventory before physical lowering."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
)
from llm.frontend.wafer_frontend.schema.full_dense_training_source_pipeline import (
    DenseFullTrainingTaskKind,
    build_full_dense_training_source_ir,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FullDenseTrainingSourcePipelineTest(unittest.TestCase):
    def _build(self, rows: int, columns: int):
        plan = build_flexible_dense_train_plan(
            _spec(rows, columns), RectMeshSpec(rows, columns)
        )
        return plan, build_full_dense_training_source_ir(plan)

    def test_1x1_has_every_two_layer_two_step_source_task(self) -> None:
        plan, source = self._build(1, 1)
        source.validate_against(plan)
        self.assertEqual(source.mesh_shape, (1, 1))
        self.assertEqual(len(source.tasks), 254)
        self.assertEqual(len(source.state_version_edges), 15)
        expected = {
            DenseFullTrainingTaskKind.PARAMETER_LOAD: 15,
            DenseFullTrainingTaskKind.FORWARD: 26,
            DenseFullTrainingTaskKind.LOSS_SEED: 1,
            DenseFullTrainingTaskKind.CE_BACKWARD: 1,
            DenseFullTrainingTaskKind.BACKBONE_BACKWARD: 24,
            DenseFullTrainingTaskKind.PARAMETER_WGRAD: 15,
            DenseFullTrainingTaskKind.GRADIENT_SYNC: 15,
            DenseFullTrainingTaskKind.SGD_UPDATE: 15,
            DenseFullTrainingTaskKind.PARAMETER_STORE: 15,
        }
        for step in (0, 1):
            current = tuple(task for task in source.tasks if task.step == step)
            for kind, count in expected.items():
                self.assertEqual(sum(task.kind is kind for task in current), count)
            forward = {task.operation_ref for task in current
                       if task.kind is DenseFullTrainingTaskKind.FORWARD}
            self.assertIn("T0.layer0.attention", forward)
            self.assertIn("T0.layer1.attention", forward)
            self.assertIn("T0.cross_entropy", forward)
            loads = tuple(task for task in current if task.kind is
                          DenseFullTrainingTaskKind.PARAMETER_LOAD)
            updates = tuple(task for task in current if task.kind is
                            DenseFullTrainingTaskKind.SGD_UPDATE)
            stores = tuple(task for task in current if task.kind is
                           DenseFullTrainingTaskKind.PARAMETER_STORE)
            self.assertTrue(all((task.read_version, task.write_version)
                                == (step, None) for task in loads))
            self.assertTrue(all((task.read_version, task.write_version)
                                == (step, step + 1) for task in updates))
            self.assertTrue(all((task.read_version, task.write_version)
                                == (step + 1, None) for task in stores))

    def test_rectangular_2x3_is_rank_complete_and_versioned(self) -> None:
        plan, source = self._build(2, 3)
        source.validate_against(plan)
        self.assertEqual(source.mesh_shape, (2, 3))
        self.assertEqual(len(source.state_version_edges), 90)
        for step in (0, 1):
            for rank in range(6):
                current = tuple(task for task in source.tasks
                                if (task.step, task.rank) == (step, rank))
                self.assertTrue(current)
                self.assertEqual(tuple(task.ordinal for task in current),
                                 tuple(range(len(current))))
                self.assertEqual(sum(task.kind is
                                     DenseFullTrainingTaskKind.PARAMETER_WGRAD
                                     for task in current), 15)
                self.assertTrue(all(
                    task.rank_instance_ref.startswith(f"step{step}.rank{rank}::")
                    for task in current
                ))
        attention_instances = {
            task.rank_instance_ref
            for task in source.tasks
            if task.step == 0
            and task.kind is DenseFullTrainingTaskKind.FORWARD
            and task.operation_ref == "T0.layer0.attention"
        }
        self.assertEqual(len(attention_instances), 6)

    def test_missing_reverse_and_forged_version_edges_fail_closed(self) -> None:
        plan, source = self._build(1, 1)
        victim = next(task for task in source.tasks
                      if task.kind is
                      DenseFullTrainingTaskKind.BACKBONE_BACKWARD)
        with self.assertRaisesRegex(SchemaError, "dangling|task order"):
            replace(source, tasks=tuple(task for task in source.tasks
                                        if task.id != victim.id)).validate()
        load = next(task for task in source.tasks
                    if task.kind is DenseFullTrainingTaskKind.PARAMETER_LOAD)
        with self.assertRaisesRegex(SchemaError, "LOAD must read current"):
            replace(load, write_version=load.step + 1).validate("load")
        store = next(task for task in source.tasks
                     if task.kind is DenseFullTrainingTaskKind.PARAMETER_STORE)
        with self.assertRaisesRegex(SchemaError, "STORE must persist next"):
            replace(store, read_version=store.step).validate("store")
        with self.assertRaisesRegex(SchemaError, "rank instance"):
            replace(load, rank_instance_ref="global").validate("load")
        source_edge, target_edge = source.state_version_edges[0]
        with self.assertRaisesRegex(SchemaError, "canonical|STORE0 to LOAD1"):
            replace(source, state_version_edges=((target_edge, source_edge),
                                                  *source.state_version_edges[1:])).validate()
        with self.assertRaisesRegex(SchemaError, "drifted"):
            replace(source, requirements_digest="0" * 64).validate_against(plan)


if __name__ == "__main__":
    unittest.main()
