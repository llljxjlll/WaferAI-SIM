"""Real L2/EP2 TRAIN source oracle refuses doubled Dense MLP and wrong owners."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.full_moe_shared_train_source_gate import (
    require_moe_shared_train_no_dense_mlp,
)
from llm.frontend.wafer_frontend.schema.full_dense_gradient_requirements import (
    build_dense_full_train_requirements,
)
from llm.frontend.wafer_frontend.schema.full_moe_shared_train_requirements import (
    build_full_moe_shared_train_requirements,
)
from llm.test.frontend.unit.test_full_training_timeline_linker import (
    FullTrainingTimelineLinkerTest,
)


class FullMoeSharedTrainRequirementsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        FullTrainingTimelineLinkerTest.setUpClass()
        cls.source = FullTrainingTimelineLinkerTest
        cls.dense = build_dense_full_train_requirements(
            cls.source.forward.plan, steps=2,
        )
        cls.requirements = build_full_moe_shared_train_requirements(
            cls.source.forward.plan, cls.dense, cls.source.moe,
        )

    def test_exact_mlp_operation_and_state_replacement_not_double_computation(self) -> None:
        req = self.requirements
        self.assertEqual(len(req.layer_replacements), 2)
        self.assertEqual(len(req.excluded_dense_parameter_state_refs), 4)
        self.assertEqual(len(req.shared_dense_parameter_state_refs), 11)
        self.assertEqual(len(req.shared_dense_gradient_paths), 22)
        self.assertEqual(len(req.shared_parameter_version_bindings), 11)
        self.assertEqual({item.owner_rank for item
                          in req.shared_parameter_version_bindings}, {0})
        self.assertEqual(len({state for item in req.shared_parameter_version_bindings
                              for state in item.source_e2e_state_versions}), 33)
        self.assertEqual(len(req.moe_parameter_requirements), 32)
        for layer in req.layer_replacements:
            self.assertEqual(tuple(ref.rsplit(".", 1)[-1] for ref
                                   in layer.replaced_dense_forward_refs),
                             ("gate_up", "swiglu", "down"))
            self.assertEqual(len(layer.removed_dense_parameter_state_refs), 2)
            self.assertEqual(len(layer.moe_forward_refs_by_step), 2)
            self.assertEqual(len(layer.moe_backward_refs_by_step), 2)
        self.assertFalse(req.source_ir0_replacement_materialized)

    def test_router_gate_has_two_real_physical_owners_and_expert_fused_homes(self) -> None:
        for step in (0, 1):
            for layer in (0, 1):
                paths = [path for path in self.requirements.moe_parameter_requirements
                         if (path.step, path.layer) == (step, layer)]
                self.assertEqual(len(paths), 8)
                self.assertEqual({path.owner_rank for path in paths
                                  if path.expert is None}, {0, 1})
                for expert in (0, 1):
                    self.assertEqual({path.owner_rank for path in paths
                                      if path.expert == expert}, {expert})
                    self.assertEqual(len([path for path in paths
                                          if path.expert == expert]), 3)
                self.assertTrue(all(path.read_version == step
                                    and path.write_version == step + 1
                                    for path in paths))

    def test_doubled_dense_mlp_state_inventory_rejected(self) -> None:
        doubled = replace(
            self.requirements,
            shared_dense_parameter_state_refs=tuple(sorted((
                *self.requirements.shared_dense_parameter_state_refs,
                self.requirements.excluded_dense_parameter_state_refs[0],
            ))),
        )
        with self.assertRaisesRegex(SchemaError, "replacement or parameter lineage"):
            doubled.validate_against(self.source.forward.plan,
                                     self.dense, self.source.moe)

    def test_real_dense_15_state_backward_carrier_cannot_impersonate_moe_train(self) -> None:
        with self.assertRaisesRegex(SchemaError, "displaced Dense MLP HBM state"):
            require_moe_shared_train_no_dense_mlp(
                self.source.backward.manifest, self.source.forward.plan,
                self.dense, self.source.moe, self.requirements,
            )

    def test_missing_router_die_replica_or_expert_projection_rejected(self) -> None:
        wrong = replace(
            self.requirements,
            moe_parameter_requirements=self.requirements.moe_parameter_requirements[1:],
        )
        with self.assertRaisesRegex(SchemaError, "replacement or parameter lineage"):
            wrong.validate_against(self.source.forward.plan,
                                   self.dense, self.source.moe)

    def test_source_explicitly_remains_partial_not_full_ir0(self) -> None:
        wrong = replace(self.requirements,
                        source_ir0_replacement_materialized=True)
        with self.assertRaisesRegex(SchemaError, "replacement or parameter lineage"):
            wrong.validate_against(self.source.forward.plan,
                                   self.dense, self.source.moe)


if __name__ == "__main__":
    unittest.main()
