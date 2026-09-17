"""Low-HBM necessary window from the complete 520-fragment AdamW source."""
from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_adamw_offload_preflight import (
    derive_full_dense_adamw_offload_window,
    require_full_dense_adamw_resident_capacity,
)
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.integration.run_full_dense_adamw_two_step_native_fresh import _compile
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FullDenseAdamwOffloadPreflightTest(unittest.TestCase):
    @classmethod
    @builder_validation_session()
    def setUpClass(cls):
        cls.program, _physical = _compile()
        cls.plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))

    @builder_validation_session()
    def test_same_model_low_hbm_resident_rejected_but_candidate_window_exists(self):
        window = derive_full_dense_adamw_offload_window(
            self.program, self.plan, hbm_capacity_bytes=4096,
            external_capacity_bytes=8192, sram_capacity_bytes=1 << 20)
        self.assertEqual((window.state_count, window.state_payload_bytes,
                          window.resident_hbm_highwater_bytes,
                          window.atomic_update_hbm_slot_bytes),
                         (75, 5716, 7684, 960))
        self.assertEqual((window.saved_activation_values,
                          window.saved_activation_payload_bytes,
                          window.sram_allocated_highwater_bytes,
                          window.sram_live_storage_peak_bytes),
                         (38, 512, 31556, 2372))
        self.assertTrue(window.blocking_offload_window_necessary)
        with self.assertRaisesRegex(SchemaError, "resident StateABI"):
            require_full_dense_adamw_resident_capacity(window)

    @builder_validation_session()
    def test_minimum_real_update_external_and_activation_windows(self):
        defaults = dict(program=self.program, plan=self.plan,
                        hbm_capacity_bytes=4096,
                        external_capacity_bytes=8192,
                        sram_capacity_bytes=1 << 20)
        for change, message in (
            ({"hbm_capacity_bytes": 959}, "five-state AdamW"),
            ({"external_capacity_bytes": 7683}, "external backing"),
            ({"sram_capacity_bytes": 31555}, "SRAM activation"),
        ):
            with self.subTest(change=change):
                with self.assertRaisesRegex(SchemaError, message):
                    derive_full_dense_adamw_offload_window(**(defaults | change))


if __name__ == "__main__":
    unittest.main()
