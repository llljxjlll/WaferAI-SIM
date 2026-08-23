from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from run_intra_die_ablation import (
    CASES,
    build_comparison,
    derive_spec,
    run_matrix,
)


def _fake_report(name: str, cycles: int, projection: str) -> dict[str, object]:
    return {
        "id": f"report-{name}",
        "inputs": {
            "spec_digest": f"spec-{name}",
            "fabric_digest": f"fabric-{name}",
            "hardware_sha256": f"hardware-{name}",
            "simulation_sha256": f"simulation-{name}",
            "mapping_sha256": f"mapping-{name}",
        },
        "runtime": {"makespan_cycles": cycles},
        "_intra_calibration_evidence": [{"calibrated": True, "id": f"calibration-{name}"}],
        "provenance": {
            "profile_id": f"profile-{name}",
            "policy_selections": [{"id": f"policy-{name}"}],
            "pass_receipts": [{"pass_name": "inter_die_plan", "id": f"receipt-{name}"}],
            "stage_digests": [f"pre-{name}"] * 6 + [projection],
        },
    }


class IntraDieAblationTest(unittest.TestCase):
    def test_independent_policy_derivation_does_not_mutate_base(self) -> None:
        base = {"workload": {"name": "tiny"}, "policy": {"partition": "gemm_coll"}}
        derived = {
            name: derive_spec(base, inter, intra)
            for name, (inter, intra) in CASES.items()
        }
        self.assertEqual(base["policy"], {"partition": "gemm_coll"})
        self.assertEqual(
            {(value["policy"]["inter_die"], value["policy"]["intra_die"]) for value in derived.values()},
            {(inter, "optimized") for inter, _mode in CASES.values()},
        )

    def test_matrix_formulas_and_machine_outputs(self) -> None:
        cycles = {"A00": 120, "A10": 90, "A01": 80, "A11": 50}

        def fake(name: str, spec: dict[str, object], path: Path) -> object:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, spec)
            projection = "naive-projection" if name in ("A00", "A01") else "swizzle-projection"
            return _fake_report(name, cycles[name], projection)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            matrix = run_matrix({"policy": {}}, root, fake)
            disk = json.loads((root / "matrix.json").read_text(encoding="utf-8"))
            self.assertEqual(disk, matrix)
            self.assertIn("| A11 | swizzle_topo | auto | 50 |", (root / "comparison.md").read_text(encoding="utf-8"))
            self.assertEqual(matrix["schema_version"], "wafer_frontend.intra_die_ablation/v3")
            self.assertTrue(matrix["calibrated"])
            case = matrix["cases"]["A11"]
            self.assertEqual(case["config_digests"]["hardware_sha256"], "hardware-A11")
            self.assertEqual(case["policy_selections"], [{"id": "policy-A11"}])
            self.assertEqual(case["pass_receipts"], [{"pass_name": "inter_die_plan", "id": "receipt-A11"}])
            self.assertEqual(case["profile_id"], "profile-A11")
            self.assertEqual(case["stage_digests"][-1], "swizzle-projection")
            metrics = matrix["metrics"]
            self.assertAlmostEqual(metrics["combined_speedup"], 2.4)
            self.assertAlmostEqual(metrics["interaction"], 50 - 90 - 80 + 120)

    def test_missing_optional_evidence_is_not_required(self) -> None:
        reports = {
            name: {
                "id": name,
                "runtime": {"makespan_cycles": 100},
                "provenance": {"stage_digests": ["0"] * 6 + [inter]},
            }
            for name, (inter, _intra) in CASES.items()
        }
        matrix = build_comparison(reports, scope="test")
        self.assertNotIn("config_digests", matrix["cases"]["A00"])
        self.assertNotIn("policy_selections", matrix["cases"]["A00"])
        self.assertNotIn("pass_receipts", matrix["cases"]["A00"])

    def test_fixed_inter_projection_digest_must_match(self) -> None:
        reports = {
            name: _fake_report(name, 100, name)
            for name in CASES
        }
        with self.assertRaisesRegex(ValueError, "fixed-inter projection digest changed"):
            build_comparison(reports, scope="test")


if __name__ == "__main__":
    unittest.main()
