from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from case_matrix import emit_artifacts
from gpu_lut import write_placeholder_yaml
from run_experiment import run


class EndToEndTests(unittest.TestCase):
    def test_placeholder_run_emits_complete_auditable_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated"
            required = emit_artifacts(generated)["required_gpu_shapes"]
            measurements = write_placeholder_yaml(required, root / "placeholder.yaml")
            output = root / "results"
            result = run(
                gpu_yaml=measurements, output_dir=output,
                generated_dir=generated, allow_placeholder=True,
            )
            self.assertEqual(result["status"], "placeholder_smoke_test")
            self.assertEqual(result["summary"]["logical_cases"], 48)
            self.assertEqual(result["summary"]["state_rows"], 288)
            self.assertEqual(result["summary"]["paired_comparisons"], 144)
            self.assertEqual(result["summary"]["placeholder_shapes"], 120)
            self.assertTrue(result["summary"]["history_compatibility_passed"])
            definitions = {
                (row["comparison"], row["baseline_state"], row["optimized_state"])
                for row in result["comparisons"]
            }
            self.assertEqual(definitions, {
                ("native_full", "W00", "W11"),
                ("native_inter_only", "C00", "C10"),
                ("gpu_inter", "G00", "G10"),
            })
            for name in ("exp3_1_results.json", "exp3_1_results.csv", "audit.json"):
                self.assertTrue((output / name).is_file())
            audit = json.loads((output / "audit.json").read_text())
            self.assertEqual(len(audit["cases"]), 48)
            self.assertEqual(
                audit["calibration_evidence"]["strategy"],
                "small_cycle_exact_motifs_plus_large_analytical_extrapolation",
            )
            names = ("exp3_1_results.json", "exp3_1_results.csv", "audit.json")
            first = {name: (output / name).read_bytes() for name in names}
            run(
                gpu_yaml=measurements, output_dir=output,
                generated_dir=generated, allow_placeholder=True,
            )
            second = {name: (output / name).read_bytes() for name in names}
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
