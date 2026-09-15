"""Real source workspace and full two-step 83-state timed HBM slot oracle."""

import unittest

from llm.frontend.wafer_frontend.passes.dense_adamw_compile_sequence import (
    compile_dense_adamw_step,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_mid_program_residency import (
    assign_dense_adamw_bounded_slots,
    derive_dense_adamw_mid_program_residency,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_offload_preflight import (
    preflight_dense_adamw_offload_window,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_paged_compile_sequence import (
    compile_dense_adamw_paged_step,
)
from llm.frontend.wafer_frontend.schema.workload_run import WorkloadMemoryMode
from llm.test.frontend.unit.test_dense_adamw_compile_sequence import _adamw_case


class DenseAdamwMidProgramResidencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source, physical = _adamw_case()
        cls.window = preflight_dense_adamw_offload_window(source)
        cls.physical = physical
        cls.linked = tuple(compile_dense_adamw_step(source, physical, i)
                           for i in (0, 1))

    def test_actual_332_lsu_boundaries_and_workspace_disjoint_timed_slots(self):
        schedule = derive_dense_adamw_mid_program_residency(
            self.window, self.linked,
        )
        slots = assign_dense_adamw_bounded_slots(self.window, schedule)
        self.assertEqual(len(schedule.spans), 83)
        self.assertEqual(len(schedule.events), 332)
        self.assertEqual(
            sum(item.size_bytes for item in schedule.spans), 32100,
        )
        self.assertEqual(schedule.peak_active_state_bytes, 13796)
        self.assertEqual(schedule.peak_active_plus_workspace_bytes, 23044)
        self.assertEqual(slots.workspace_end_bytes, 9248)
        self.assertEqual(len(slots.state_addresses), 83)
        self.assertEqual(len({addr for _ref, addr in slots.state_addresses}), 26)
        self.assertEqual(slots.highest_state_end_bytes, 23300)
        self.assertGreaterEqual(min(addr for _ref, addr in slots.state_addresses), 9248)
        self.assertLess(slots.highest_state_end_bytes, 36864)
        seen = {}
        for event in schedule.events:
            key = (event.step_index, event.state_ref)
            seen.setdefault(key, set()).add(event.kind)
        self.assertEqual(len(seen), 166)
        self.assertTrue(all(
            kinds == {"restore_before_lsu_load", "writeback_after_lsu_store"}
            for kinds in seen.values()
        ))
        paged = tuple(
            compile_dense_adamw_paged_step(self.window, self.physical, i, slots)
            for i in (0, 1)
        )
        self.assertTrue(all(
            item.materialization.request.memory.mode is WorkloadMemoryMode.EXTERNAL_OFFLOAD
            for item in paged
        ))
        state0 = tuple(sorted(
            (item.state_ref, item.address, item.size_bytes, item.id)
            for item in paged[0].manifest.fragments[0].state_abi
        ))
        state1 = tuple(sorted(
            (item.state_ref, item.address, item.size_bytes, item.id)
            for item in paged[1].manifest.fragments[0].state_abi
        ))
        self.assertEqual(state0, state1)
        self.assertEqual(len(state0), 83)
        self.assertEqual(max(address + size for _ref, address, size, _id in state0), 23300)
        self.assertEqual(sum(size for _ref, _address, size, _id in state0), 32100)


if __name__ == "__main__":
    unittest.main()
