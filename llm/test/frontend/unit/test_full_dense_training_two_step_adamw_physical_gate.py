"""A full AdamW physical receipt must close all gradients and HBM versions."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.full_dense_adamw_physical_gate import (
    require_full_dense_adamw_physical_gradient_paths,
)
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.test.frontend.integration.run_full_dense_adamw_two_step_native_fresh import _compile


class FullDenseAdamwPhysicalGateTest(unittest.TestCase):
    @classmethod
    @builder_validation_session()
    def setUpClass(cls):
        cls.program, cls.physical = _compile()
        cls.plan = cls.program.source.replicas[0].lowering_context.ir1  # source identity witness

    @builder_validation_session()
    def test_full_gradient_and_version_receipt(self):
        self.assertEqual(len(self.physical.actions), 520)
        self.assertEqual(len(self.physical.state_version_edges), 75)
        self.assertEqual(len(self.program.manifest.fragments), 520)
        from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
        from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
        from llm.test.frontend.unit.test_flexible_dense_train import _spec
        plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
        require_full_dense_adamw_physical_gradient_paths(self.physical, self.program, plan)

    @builder_validation_session()
    def test_missing_state_version_or_wgrad_action_is_rejected(self):
        from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
        from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
        from llm.test.frontend.unit.test_flexible_dense_train import _spec
        plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
        missing_edge = replace(self.physical,
                               state_version_edges=self.physical.state_version_edges[:-1])
        with self.assertRaisesRegex(SchemaError, "version receipt differs"):
            require_full_dense_adamw_physical_gradient_paths(missing_edge,
                                                               self.program, plan)
        omitted = replace(self.physical, actions=self.physical.actions[:-1])
        with self.assertRaises(SchemaError):
            require_full_dense_adamw_physical_gradient_paths(omitted,
                                                               self.program, plan)


if __name__ == "__main__":
    unittest.main()
