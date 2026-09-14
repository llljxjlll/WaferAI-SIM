from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_results
import result_adapter


def _record(**updates):
    value = {
        "case_id": "case0",
        "mesh": "1x4",
        "operator": "AG_GEMM",
        "model": "model",
        "layer": "mlp",
        "seq_len": 2048,
        "logical_flops": 1_000_000,
        "T00_cycles": 100,
        "T10_cycles": 80,
        "T01_cycles": 50,
        "T11_cycles": 40,
        "normal_cycles": 40,
        "congested_upper_bound_cycles": 48,
        "T00_T10_T01_source": "cycle_accurate_trace_replay",
        "estimate_source": "cycle_accurate_trace_calibrated_estimate",
        "status": "estimated_via_tiling_and_padding",
    }
    value.update(updates)
    return value


class ResultAdapterTest(unittest.TestCase):
    def test_derives_complete_factorial_metrics_and_source_aliases(self) -> None:
        result = result_adapter.enrich_record(_record())
        self.assertEqual(result["inter_speedup_without_intra"], 1.25)
        self.assertEqual(result["inter_speedup_with_intra"], 1.25)
        self.assertEqual(result["intra_speedup_without_inter"], 2.0)
        self.assertEqual(result["intra_speedup_with_inter"], 2.0)
        self.assertEqual(result["total_speedup"], 2.5)
        self.assertEqual(result["synergy"], 1.0)
        self.assertEqual(result["congestion_factor"], 1.2)
        self.assertEqual(result["T00_source"], "cycle_accurate_trace_replay")
        self.assertEqual(
            result["result_source"], "cycle_accurate_trace_calibrated_estimate"
        )
        self.assertTrue(set(result_adapter.OUTPUT_FIELDS).issubset(result))

    def test_accepts_sub_ppm_integer_cycle_rounding(self) -> None:
        value = _record(inter_speedup_without_intra=1.2500005)
        self.assertAlmostEqual(
            result_adapter.enrich_record(value)["inter_speedup_without_intra"],
            1.25,
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            result_adapter.enrich_record(
                _record(inter_speedup_without_intra=1.251)
            )

    def test_plot_reads_enriched_schema_and_emits_metric_svg(self) -> None:
        row = {key: str(value) for key, value in result_adapter.enrich_record(_record()).items()}
        self.assertEqual(plot_results.result_series([row]), plot_results.T_SERIES)
        self.assertEqual(len(plot_results.values_for(row)), 4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.svg"
            plot_results.plot_svg(
                [row], "metrics", output,
                series=plot_results.METRIC_SERIES, metrics=True,
            )
            text = output.read_text(encoding="utf-8")
        self.assertIn("Synergy", text)
        self.assertIn(">Ratio<", text)


if __name__ == "__main__":
    unittest.main()
