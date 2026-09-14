from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest


EXP4 = Path(__file__).resolve().parents[1]
if str(EXP4) not in sys.path:
    sys.path.insert(0, str(EXP4))

import action_replay  # noqa: E402
from candidate_loader import load_candidates  # noqa: E402


class CorrectedActionReplayTests(unittest.TestCase):
    def test_all_reference_rows_reproduce_and_respect_lower_bound(self):
        audit = action_replay.audit_reference_reproduction()
        self.assertEqual(audit["row_count"], 72)
        self.assertEqual(audit["failure_count"], 0)

    def test_exact_prefill_ledger_inventory(self):
        document = json.loads((EXP4 / "inputs/prefill_resource_ledgers.json").read_text())
        self.assertEqual(document["row_count"], 12)
        for row in document["rows"]:
            self.assertEqual(len(row["base_resource_service_cycles"]), 234)
            self.assertEqual(row["base_makespan_cycles"], row["base_makespan_cycles"])
            self.assertLessEqual(row["base_theory_lower_cycles"], row["base_makespan_cycles"])
            self.assertLessEqual(row["sw_opt_theory_lower_cycles"], row["sw_opt_makespan_cycles"])

    def test_increasing_noc_bandwidth_cannot_slow_fixed_profile(self):
        candidate = next(item for item in load_candidates() if item.B_GBs == 64)
        faster = replace(candidate, B_GBs=128)
        row = {"case_id": "prefill_pd__llama2_7b__s2304"}
        for state in ("naive", "sw_opt"):
            slow = action_replay.estimate(row, "prefill", state, candidate)
            fast = action_replay.estimate(row, "prefill", state, faster)
            self.assertLessEqual(fast.estimate_cycles, slow.estimate_cycles)


if __name__ == "__main__":
    unittest.main()
