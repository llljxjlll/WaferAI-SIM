from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.full_dense_gradient_requirements import (
    build_dense_full_train_requirements,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FullDenseGradientRequirementsTest(unittest.TestCase):
    def _plan(self, rows: int, columns: int):
        return build_flexible_dense_train_plan(
            _spec(rows, columns), RectMeshSpec(rows, columns)
        )

    def test_two_steps_cover_every_real_parameter_shard_and_owner(self) -> None:
        for rows, columns in ((1, 1), (2, 2), (2, 3)):
            with self.subTest(mesh=(rows, columns)):
                plan = self._plan(rows, columns)
                oracle = build_dense_full_train_requirements(plan)
                oracle.validate_against(plan)
                self.assertEqual(
                    len(oracle.required_gradient_producers),
                    2 * rows * len(plan.parameter_templates),
                )
                self.assertEqual(
                    {path.parameter_state_ref for path in oracle.paths},
                    {state.id for state in plan.forward_graph.persistent_states},
                )
                self.assertEqual(
                    oracle.required_forward_refs,
                    tuple(node.id for node in plan.forward_graph.nodes),
                )
                self.assertEqual(
                    len(oracle.required_backbone_backward_refs),
                    len(plan.forward_graph.nodes) - 1,
                )
                self.assertNotEqual(
                    oracle.loss_gradient_seed_ref, oracle.forward_loss_value_ref
                )
                self.assertIs(oracle.loss_gradient_seed_dtype, DType.FP32)
                for step in range(2):
                    for template in plan.parameter_templates:
                        for rank in template.owner_ranks:
                            path = oracle.required_gradient_producers[
                                (step, template.state_ref, rank)
                            ]
                            self.assertEqual((path.read_version, path.write_version),
                                             (step, step + 1))
                            self.assertEqual(path.backward_producer_refs,
                                             template.backward_node_refs)
                            self.assertEqual(path.gradient_bytes,
                                             2 * path.weight_bytes)
                            self.assertEqual(path.dp_group_ranks,
                                             template.owner_ranks)
                            self.assertIn(
                                "dp_sync::" if rows > 1 else "local_sync::",
                                path.named_sync_op_ref,
                            )

    def test_missing_real_parameter_path_and_old_state_version_fail(self) -> None:
        plan = self._plan(2, 2)
        oracle = build_dense_full_train_requirements(plan)
        with self.assertRaisesRegex(SchemaError, "contract drifted"):
            replace(oracle, paths=oracle.paths[1:]).validate_against(plan)
        corrupt = replace(oracle.paths[0], write_version=0)
        with self.assertRaisesRegex(SchemaError, "contract drifted"):
            replace(oracle, paths=(corrupt, *oracle.paths[1:])).validate_against(plan)

    def test_wrong_dp_sync_group_or_unrelated_seed_rejected(self) -> None:
        plan = self._plan(2, 3)
        oracle = build_dense_full_train_requirements(plan)
        bad_sync = replace(oracle.paths[0], dp_group_ranks=(oracle.paths[0].rank,))
        with self.assertRaises(SchemaError):
            replace(oracle, paths=(bad_sync, *oracle.paths[1:])).validate_against(plan)
        with self.assertRaises(SchemaError):
            replace(oracle, loss_gradient_seed_ref=oracle.forward_loss_value_ref).validate_against(plan)
        with self.assertRaisesRegex(SchemaError, "two steps"):
            build_dense_full_train_requirements(plan, steps=1)


if __name__ == "__main__":
    unittest.main()
