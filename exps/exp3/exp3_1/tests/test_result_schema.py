from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from case_matrix import build_logical_cases
from legacy_alignment import dense_native
from resource_replay import paired_replay
from result_schema import build_case_states, validate_state_rows


class ResultSchemaTests(unittest.TestCase):
    def test_six_states_and_gpu_pairing(self) -> None:
        case = next(c for c in build_logical_cases() if c.operator_family == "gemm_rs")
        native = dense_native(case)
        pair = paired_replay(
            algorithm="fixture", lookup_key=case.ring_key,
            lookup_latency_ns=100, execution_count=case.D,
            communication_off_ns=50, communication_on_ns=55,
            overlap_factor=0.1, evidence="fixture",
        )
        rows = build_case_states(case, native, pair, {"config_sha256": "x"})
        self.assertEqual({row["state"] for row in rows}, {"W00", "W11", "C00", "C10", "G00", "G10"})
        by_state = {row["state"]: row for row in rows}
        self.assertEqual(by_state["G00"]["algorithm"], by_state["G10"]["algorithm"])
        self.assertEqual(by_state["G00"]["compute_time_ns"], by_state["G10"]["compute_time_ns"])
        self.assertEqual(by_state["G10"]["baseline_state"], "G00")
        self.assertEqual(by_state["C10"]["baseline_state"], "C00")
        self.assertEqual(by_state["W11"]["baseline_state"], "W00")
        self.assertAlmostEqual(
            by_state["W11"]["speedup"],
            native["state_cycles"]["W00"] / native["state_cycles"]["W11"],
        )

        self.assertAlmostEqual(
            by_state["C10"]["speedup"],
            native["controlled_state_cycles"]["C00"] / native["controlled_state_cycles"]["C10"],
        )
        self.assertGreater(by_state["C00"]["inter_port_time_ns"], 0)
        self.assertGreater(by_state["C10"]["inter_port_time_ns"], 0)
        self.assertGreater(by_state["W00"]["inter_port_time_ns"], 0)
        self.assertEqual(by_state["C00"]["hbm_time_ns"], by_state["C10"]["hbm_time_ns"])
        self.assertNotIn("hbm_intermediate_materialization_applied", by_state["C00"])
    def test_missing_state_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_state_rows([{"case_id": "bad", "state": "W00"}])


if __name__ == "__main__":
    unittest.main()
