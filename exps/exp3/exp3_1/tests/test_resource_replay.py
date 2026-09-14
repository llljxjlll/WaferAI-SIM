from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from resource_replay import paired_replay


class ResourceReplayTests(unittest.TestCase):
    def test_serial_and_overlap_use_same_compute_decomposition(self) -> None:
        pair = paired_replay(
            algorithm="fixture", lookup_key=(2, 3, 4), lookup_latency_ns=10,
            execution_count=4, communication_off_ns=20,
            communication_on_ns=20, overlap_factor=0.1, evidence="fixture",
        )
        self.assertEqual(pair.compute_time_ns, 40)
        self.assertEqual(pair.off_time_ns, 60)
        self.assertEqual(pair.on_time_ns, 42)
        self.assertAlmostEqual(pair.speedup, 60 / 42)
        self.assertGreaterEqual(pair.on_time_ns, max(40, 20))

    def test_zero_communication_has_exact_unit_speedup(self) -> None:
        pair = paired_replay(
            algorithm="fixture", lookup_key=(2, 3, 4), lookup_latency_ns=10,
            execution_count=1, communication_off_ns=0,
            communication_on_ns=0, overlap_factor=0.5, evidence="fixture",
        )
        self.assertEqual(pair.off_time_ns, pair.on_time_ns)
        self.assertEqual(pair.speedup, 1.0)

    def test_invalid_duration_or_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_replay(
                algorithm="fixture", lookup_key=(2, 3, 4), lookup_latency_ns=10,
                execution_count=1, communication_off_ns=1,
                communication_on_ns=1, overlap_factor=1.1, evidence="fixture",
            )


if __name__ == "__main__":
    unittest.main()
