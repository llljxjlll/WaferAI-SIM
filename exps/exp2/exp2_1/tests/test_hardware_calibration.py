from __future__ import annotations

import json
import sys
from pathlib import Path
import unittest


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
sys.path.insert(0, str(EXP_ROOT))

import hardware_binding  # noqa: E402


class HardwareCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hardware_path = EXP_ROOT / "configs" / "target_hardware.json"
        cls.simulation_path = EXP_ROOT / "configs" / "target_simulation.json"
        cls.hardware = json.loads(cls.hardware_path.read_text(encoding="utf-8"))
        cls.closure = hardware_binding.build_unit_closure(
            cls.hardware_path, cls.simulation_path
        )

    def test_geometry_control_and_compute_arithmetic(self) -> None:
        structural = self.closure["structural_closure"]
        self.assertEqual(structural["wafer"]["die_mesh"], [6, 6])
        self.assertEqual(structural["wafer"]["die_count"], 36)
        self.assertEqual(structural["wafer"]["core_mesh_per_die"], [4, 4])
        self.assertEqual(structural["wafer"]["worker_cores_per_die"], 16)
        self.assertTrue(structural["control"]["gate_pass"])
        self.assertTrue(structural["control"]["dedicated_control_is_not_addressed_as_worker"])

        compute = self.closure["unit_gates"]["compute"]
        self.assertEqual(compute["logical_flops_per_cycle"], 16000.0)
        self.assertAlmostEqual(compute["derived_core_TFLOPs"], 8.0)
        self.assertAlmostEqual(compute["derived_die_TFLOPs"], 128.0)
        self.assertTrue(compute["configuration_arithmetic_match"])
        self.assertFalse(compute["target_bound_isolated_gemm_validated"])
        self.assertFalse(compute["gate_pass"])

    def test_sram_dte_and_noc_config_arithmetic_is_not_direct_validation(self) -> None:
        gates = self.closure["unit_gates"]
        self.assertEqual(gates["sram_read_port"]["capacity_bytes"], 3 * 1024**2)
        self.assertTrue(gates["sram_read_port"]["regions_nonoverlap_and_in_bounds"])
        for name in ("sram_read_port", "dte_injection", "noc_logical_payload"):
            self.assertAlmostEqual(gates[name]["derived_Bps"] / 1e9, 256.0)
            self.assertTrue(gates[name]["arithmetic_match"])
            self.assertFalse(gates[name]["direct_saturation_sweep_validated"])
            self.assertFalse(gates[name]["gate_pass"])

    def test_d2d_rate_cap_forces_unit_closure_failure(self) -> None:
        d2d = self.closure["unit_gates"]["d2d_single_lane"]
        self.assertEqual(d2d["packet_payload_bytes"], 16)
        self.assertEqual(d2d["packets_per_cycle"], 1.0)
        self.assertAlmostEqual(d2d["derived_Bps"] / 1e9, 8.0)
        self.assertEqual(d2d["expected_Bps"], 1.0e12)
        self.assertAlmostEqual(d2d["target_to_realized_ratio"], 125.0)
        self.assertFalse(d2d["arithmetic_match"])
        self.assertFalse(d2d["gate_pass"])
        self.assertFalse(self.closure["simulator_unit_closure"])
        self.assertFalse(self.closure["publish_target_absolute_cycles"])

    def test_four_edge_hbm_stacks_have_exact_capacity_and_address_ranges(self) -> None:
        hbm = self.closure["unit_gates"]["hbm"]
        self.assertEqual(hbm["stack_count"], 4)
        self.assertEqual(hbm["home_die_ids"], [1, 4, 31, 34])
        self.assertEqual(
            self.closure["structural_closure"]["hbm_edge_placement"][
                "home_die_coordinates_xy"
            ],
            [[1, 0], [4, 0], [1, 5], [4, 5]],
        )
        self.assertEqual(hbm["per_stack_capacity_bytes"], [16 * 1024**3] * 4)
        self.assertEqual(hbm["aggregate_capacity_bytes"], 64 * 1024**3)
        self.assertEqual(hbm["profile_derived_stack_Bps"], 256.0e9)
        self.assertTrue(hbm["address_ranges_nonoverlap"])
        self.assertTrue(hbm["address_ranges_contiguous"])
        self.assertTrue(hbm["configuration_arithmetic_match"])
        self.assertFalse(hbm["direct_saturation_sweep_validated"])
        self.assertFalse(hbm["gate_pass"])

    def test_calibration_evidence_never_promotes_tiny_cycles(self) -> None:
        source = json.loads(
            (EXP_ROOT / "calibration" / "source_evidence.json").read_text(
                encoding="utf-8"
            )
        )
        applicability = source["target_applicability"]
        self.assertFalse(applicability["simulator_unit_closure"])
        self.assertEqual(applicability["direct_target_cycle_anchors"], 0)
        self.assertFalse(applicability["target_absolute_cycles_publishable"])
        encoded = json.dumps(source, sort_keys=True)
        self.assertIn("structural_pd_lifecycle_prior_only", encoded)
        self.assertIn("primitive_execution_smoke_only", encoded)

        stage4 = json.loads(
            (
                EXP_ROOT
                / "calibration"
                / "stage4_pds_current_build_evidence.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(stage4["repeatability"]["makespan_cycles"], [6187, 6187])
        self.assertTrue(stage4["drain"]["program_io_all_probes_passed"])
        self.assertTrue(
            all(value == 0 for key, value in stage4["drain"].items() if key.endswith("_residual"))
        )
        self.assertFalse(stage4["applicability"]["absolute_cycle_anchor_eligible"])

    def test_frozen_source_files_exist_and_digests_match(self) -> None:
        for item in self.closure["generated_from"]["implementation_sources"].values():
            path = WORKSPACE / item["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(hardware_binding.sha256_file(path), item["sha256"])


if __name__ == "__main__":
    unittest.main()
