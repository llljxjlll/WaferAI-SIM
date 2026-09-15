"""Same-model two-layer AdamW capacity pair must not claim startup offload success."""

import unittest

from llm.frontend.wafer_frontend.passes.dense_adamw_offload_preflight import (
    preflight_dense_adamw_offload_window,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_offload_plan import (
    plan_dense_adamw_source_offload,
)
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadMemoryMode
from llm.test.frontend.unit.test_dense_adamw_compile_sequence import _adamw_case


class DenseAdamwOffloadPreflightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, _physical = _adamw_case()

    def test_real_same_model_bounded_reject_and_offload_not_runtime_pass(self):
        evidence = preflight_dense_adamw_offload_window(self.source)
        self.assertEqual(evidence.resident_rejection_code, "memory_capacity_exceeded")
        self.assertEqual(evidence.materialization.request.model, self.source.request.model)
        self.assertEqual(evidence.materialization.request.steps, self.source.request.steps)
        self.assertEqual(
            evidence.materialization.request.memory.mode,
            WorkloadMemoryMode.EXTERNAL_OFFLOAD,
        )
        self.assertEqual(evidence.external_state_bytes, 32100)
        self.assertEqual(evidence.external_peak_bytes, 32112)
        self.assertEqual(evidence.hbm_workspace_peak_bytes, 9248)
        self.assertEqual(evidence.startup_combined_peak_bytes, 41348)
        self.assertFalse(evidence.startup_window_sufficient)
        self.assertGreater(
            evidence.startup_combined_peak_bytes,
            evidence.resident_hbm_capacity_bytes,
        )
        plan = plan_dense_adamw_source_offload(evidence.materialization)
        self.assertEqual(len(plan.state_mappings), 5)
        self.assertEqual(plan.stats.bring_in_count, 5)
        self.assertEqual(plan.stats.dirty_writeback_count, 5)
        self.assertEqual(plan.stats.transfer_bytes, 64200)
        self.assertEqual(plan.stats.hbm_peak_bytes, 32112)
        self.assertEqual(plan.stats.final_pin_count, 0)
        self.assertEqual(plan.stats.final_dirty_chunks, 0)
        self.assertEqual(plan.stats.final_resident_chunks, 0)
        self.assertGreater(
            plan.stats.hbm_peak_bytes + evidence.hbm_workspace_peak_bytes,
            evidence.resident_hbm_capacity_bytes,
        )


if __name__ == "__main__":
    unittest.main()
