from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from plot_dual_axis import build_chart, read_rows, write_svg


BASE = Path(__file__).resolve().parents[1]


class DualAxisPlotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = read_rows(BASE / "results" / "results.csv")

    def test_eight_charts_have_expected_hierarchy_and_normalization(self) -> None:
        for mesh in ("1x4", "2x3", "3x3", "6x6"):
            for operator in ("AG_GEMM", "GEMM_RS"):
                chart = build_chart(self.rows, mesh, operator)
                self.assertEqual(len(chart.points), 22)
                self.assertEqual([span.label for span in chart.layer_spans],
                                 ["Attn", "MLP"])
                self.assertEqual(len(chart.model_spans), 11)
                self.assertAlmostEqual(
                    max(point.optimized_normalized for point in chart.points), 1.0
                )
                self.assertTrue(all(
                    point.theoretical_speedup >= point.actual_speedup
                    for point in chart.points
                ))
                self.assertTrue(all(
                    0 < point.attainment_rate <= 1
                    for point in chart.points
                ))
                for point in chart.points:
                    self.assertAlmostEqual(
                        point.attainment_rate,
                        point.actual_speedup / point.theoretical_speedup,
                    )

    def test_spacing_orders_seq_model_and_layer_gaps(self) -> None:
        chart = build_chart(self.rows, "1x4", "AG_GEMM")
        xs = [point.x for point in chart.points]
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        seq_gap = min(gaps)
        layer_gap = max(gaps)
        model_gap = next(gap for gap in gaps if seq_gap < gap < layer_gap)
        self.assertAlmostEqual(seq_gap, 1.0)
        self.assertAlmostEqual(model_gap, 1.65)
        self.assertAlmostEqual(layer_gap, 1.9)
        self.assertLess(seq_gap, model_gap)
        self.assertLess(model_gap, layer_gap)

    def test_svg_contains_both_axes_and_layer_separated_attainment_lines(self) -> None:
        chart = build_chart(self.rows, "1x4", "GEMM_RS")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chart.svg"
            write_svg(chart, output)
            text = output.read_text(encoding="utf-8")
            root = ET.parse(output).getroot()
        self.assertIn("Normalized performance", text)
        self.assertIn("Actual / theoretical speedup", text)
        self.assertEqual(text.count("<polyline"), 2)
        svg_ns = "{http://www.w3.org/2000/svg}"
        polylines = list(root.iter(f"{svg_ns}polyline"))
        self.assertEqual({line.get("data-layer") for line in polylines},
                         {"attention", "mlp"})
        self.assertTrue(all(line.get("stroke-width") == "6.75"
                            for line in polylines))
        self.assertIn("100%", text)
        self.assertNotIn("Architecture-aware upper speedup", text)
        self.assertNotIn("Actual estimated speedup", text)
        rects = list(root.iter(f"{svg_ns}rect"))
        by_role = {
            role: [rect for rect in rects if rect.get("data-role") == role]
            for role in ("optimized-gain", "naive-baseline", "bar-outline")
        }
        self.assertTrue(all(len(items) == 22 for items in by_role.values()))
        for gain, naive, outline in zip(
                by_role["optimized-gain"], by_role["naive-baseline"],
                by_role["bar-outline"]):
            self.assertEqual(gain.get("width"), naive.get("width"))
            self.assertEqual(naive.get("width"), outline.get("width"))
            self.assertEqual(outline.get("stroke"), "#000")
            self.assertEqual(outline.get("stroke-width"), "1.2")
        self.assertEqual(len([rect for rect in rects
                              if rect.get("data-role") == "plot-frame"]), 1)
        self.assertEqual(len([rect for rect in rects
                              if rect.get("data-role") == "figure-frame"]), 1)
        circles = list(root.iter(f"{svg_ns}circle"))
        self.assertTrue(all(circle.get("fill") == "white" for circle in circles))
        self.assertTrue(all(circle.get("r") == "8.68" for circle in circles))
        self.assertTrue(all(circle.get("stroke-width") == "5.1"
                            for circle in circles))
        self.assertTrue(all(circle.get("stroke") == "#a23b72"
                            for circle in circles))
        self.assertIn("#24557a", text)
        self.assertIn("#9ecae1", text)
        self.assertIn("#28735a", text)
        self.assertIn("#a8dcc8", text)
        self.assertIn('stroke="#b8b8b8"', text)


if __name__ == "__main__":
    unittest.main()
