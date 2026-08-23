from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from run_intra_die_performance import (
    CORE16_MODES,
    REPEAT,
    _calibration_summary,
    build_comparison,
    build_mode_options,
    build_request,
    require_performance_spec,
    run_comparison,
)


def _receipt(name: str, input_digest: str, output_digest: str) -> dict[str, object]:
    return {"pass_name": name, "input_digest": input_digest, "output_digest": output_digest}


def _report(mode: str, cycles: int, *, simulation: str = "sim") -> dict[str, object]:
    return {
        "id": f"report-{mode}",
        "inputs": {
            "spec_digest": "spec", "fabric_digest": "fabric",
            "hardware_sha256": "hardware", "simulation_sha256": simulation,
            "mapping_sha256": "mapping",
        },
        "tools": {"finalizer_sha256": "finalizer", "npusim_sha256": "npusim"},
        "runtime": {
            "makespan_cycles": cycles, "repeat": REPEAT,
            "repeat_signature_stable": True,
        },
        "artifact": {"artifact_sha256": mode[0] * 64, "record_count": 10},
        "validation": {
            "timing": "pass", "address_lifecycle": "pass", "transport_control": "pass",
        },
        "provenance": {
            "profile_id": "prefill",
            "policy_selections": [{"kind": "inter_die", "name": "naive", "id": "inter"}],
            "pass_receipts": [
                _receipt("placement", "logical", "ir1"),
                _receipt("project_to_ir2", "planned", "projection"),
                _receipt("intra_die_refine", "projection", f"refined-{mode}"),
            ],
        },
    }


def _search(kind: str, *, calls: int = 0) -> list[dict[str, object]]:
    candidate = {"id": f"candidate-{kind}", "kind": kind}
    return [{
        "id": f"decision-{kind}", "selected_candidate_ref": candidate["id"],
        "selection_reason": "test", "candidates": [candidate],
        "generated_candidate_count": 1, "simulator_calls_during_search": calls,
    }]


class IntraDiePerformanceCliTest(unittest.TestCase):
    def test_16_core_naive_auto_options_and_core_gate(self) -> None:
        naive = build_mode_options("naive", compute_groups_per_die=16)
        auto = build_mode_options("auto", compute_groups_per_die=16)
        self.assertEqual(naive.mode.value, "force")
        self.assertEqual(naive.force_candidate, "split_k_barrier")
        self.assertEqual(naive.split_k_parts, (16,))
        self.assertEqual(auto.mode.value, "auto")
        self.assertEqual(auto.compute_groups_per_die, 16)
        self.assertIn("split_k_tree_direct_dma", auto.allowed_candidates)

        reports = {
            "naive": _report("naive", 120),
            "auto": _report("auto", 100),
        }
        active = [[core, 1] for core in range(32)]
        for report in reports.values():
            report["artifact"]["core_count"] = 32
            report["runtime"]["done_by_core"] = active
            report["runtime"]["ack_by_core"] = active
        comparison = build_comparison(
            reports,
            workload_sha256="w" * 64,
            searches={
                "naive": _search("split_k_fallback"),
                "auto": _search("split_k_fallback"),
            },
            modes=CORE16_MODES,
            cores_per_die=16,
            expected_die_count=2,
            expectation="gain",
        )
        self.assertAlmostEqual(comparison["metrics"]["gain"], 1 / 6)
        reports["naive"]["artifact"]["core_count"] = 31
        with self.assertRaisesRegex(ValueError, "16 cores on every die"):
            build_comparison(
                reports,
                workload_sha256="w" * 64,
                searches={
                    "naive": _search("split_k_fallback"),
                    "auto": _search("split_k_fallback"),
                },
                modes=CORE16_MODES,
                cores_per_die=16,
                expected_die_count=2,
            )

    def _args(self, root: Path, spec: Path, *, expectation: str = "no-regression") -> Namespace:
        return Namespace(
            spec=spec, hardware=root / "hardware.json",
            simulation=root / "simulation.json", mapping=root / "mapping.spec",
            npusim=root / "npusim", finalizer=root / "finalizer",
            output=root / "output", case="E1", profile_id=None,
            timeout_seconds=300, expect=expectation, minimum_gain=0.05,
        )

    def test_request_uses_same_inputs_and_three_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); spec = root / "spec.yaml"
            spec.write_text(yaml.safe_dump({"policy": {"intra_die": "optimized"}}))
            args = self._args(root, spec)
            off = build_request(args, "off", option_factory=lambda mode: f"options-{mode}")
            auto = build_request(args, "auto", option_factory=lambda mode: f"options-{mode}")
            self.assertEqual(off.repeat, 3)
            self.assertEqual(auto.repeat, 3)
            self.assertEqual(off.spec_path, auto.spec_path)
            self.assertEqual(off.hardware_config_path, auto.hardware_config_path)
            self.assertNotEqual(off.output_dir, auto.output_dir)

    def test_unified_off_auto_options_are_bounded(self) -> None:
        off = build_mode_options("off")
        auto = build_mode_options("auto")
        self.assertEqual(off.mode.value, "off")
        self.assertEqual(auto.mode.value, "auto")
        self.assertEqual(auto.allowed_candidates, ("identity", "split_k"))
        self.assertEqual(auto.max_candidates, 8)
        off.validate()
        auto.validate()

    def test_comparison_computes_gain_and_closes_budget(self) -> None:
        comparison = build_comparison(
            {"off": _report("off", 100), "auto": _report("auto", 80)},
            workload_sha256="w" * 64,
            searches={"off": _search("identity"), "auto": _search("split_k_fallback")},
            expectation="gain",
        )
        self.assertEqual(comparison["status"], "pass")
        self.assertAlmostEqual(comparison["metrics"]["speedup"], 1.25)
        self.assertAlmostEqual(comparison["metrics"]["gain"], 0.2)
        self.assertEqual(comparison["budget"]["final_simulator_calls"], 6)
        self.assertEqual(comparison["equivalence"]["pre_refine_projection_digest"], "projection")

    def test_input_or_search_budget_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "simulation_sha256"):
            build_comparison(
                {"off": _report("off", 100), "auto": _report("auto", 90, simulation="other")},
                workload_sha256="w", searches={"off": _search("identity"), "auto": _search("identity")},
            )
        with self.assertRaisesRegex(ValueError, "must not call"):
            build_comparison(
                {"off": _report("off", 100), "auto": _report("auto", 90)},
                workload_sha256="w", searches={"off": _search("identity"), "auto": _search("identity", calls=1)},
            )

    def test_calibration_must_match_measured_cycles_and_budget(self) -> None:
        row = {
            "id": "cal", "selected_candidate_ref": "candidate",
            "predicted_makespan_cycles": 100,
            "simulator_measured_makespan_cycles": 100,
            "relative_error": 0.0, "calibrated": True,
            "simulator_calls_used": REPEAT,
            "reserved_simulator_calls_for_final_evidence": REPEAT,
            "repeat_signature_stable": True,
        }
        self.assertEqual(_calibration_summary([row], 100)["status"], "pass")
        with self.assertRaisesRegex(ValueError, "disagree"):
            _calibration_summary([{**row, "simulator_measured_makespan_cycles": 99}], 100)
        with self.assertRaisesRegex(ValueError, "three-call"):
            _calibration_summary([{**row, "simulator_calls_used": 2}], 100)

    def test_sync_bound_requires_identity_and_no_regression(self) -> None:
        with self.assertRaisesRegex(ValueError, "select identity"):
            build_comparison(
                {"off": _report("off", 100), "auto": _report("auto", 90)},
                workload_sha256="w", searches={"off": _search("identity"), "auto": _search("split_k_fallback")},
                expectation="identity",
            )
        with self.assertRaisesRegex(ValueError, "regressed"):
            build_comparison(
                {"off": _report("off", 100), "auto": _report("auto", 101)},
                workload_sha256="w", searches={"off": _search("identity"), "auto": _search("identity")},
                expectation="identity",
            )

    def test_run_writes_versioned_report_and_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); spec = root / "spec.yaml"
            spec.write_text(yaml.safe_dump({"policy": {"intra_die": "optimized"}}))
            args = self._args(root, spec, expectation="identity")
            workspace = Path(__file__).resolve().parents[4]
            args.hardware = workspace / "notes" / "frontend" / "examples" / "hardware_2x1.json"
            args.simulation = workspace / "llm" / "test" / "sram" / "simulation.json"

            def fake(request: object) -> object:
                mode = Path(request.output_dir).name
                Path(request.output_dir, "compile").mkdir(parents=True)
                Path(request.output_dir, "compile", "intra_die_v2_search_decisions.json").write_text(
                    json.dumps(_search("identity")), encoding="utf-8"
                )
                run_dir = Path(request.output_dir, "run")
                run_dir.mkdir()
                resource_log = (
                    "[PRIM] Core 0 start compute primitive Matmul_f. | 2 ns\n"
                    "[PRIM] Core 0 end compute primitive Matmul_f. | 4 ns\n"
                    "[PROGRAM_MEMORY] core=0 lsu_issued=0 lsu_completed=0\n"
                    "[D2D] busy_cycles=0 stall_cycles=0\n"
                )
                for index in range(REPEAT):
                    (run_dir / f"stdout.{index}.log").write_text(resource_log)
                calibration = [{
                    "id": f"calibration-{mode}", "selected_candidate_ref": "candidate-identity",
                    "predicted_makespan_cycles": 100, "simulator_measured_makespan_cycles": 100,
                    "relative_error": 0.0, "calibrated": True, "simulator_calls_used": REPEAT,
                    "reserved_simulator_calls_for_final_evidence": REPEAT,
                    "repeat_signature_stable": True,
                }]
                Path(request.output_dir, "compile", "intra_die_v2_calibration_evidence.json").write_text(
                    json.dumps(calibration), encoding="utf-8"
                )
                return _report(mode, 100)

            comparison = run_comparison(
                args, runner=fake, option_factory=lambda mode: f"options-{mode}"
            )
            self.assertTrue(comparison["id"].startswith("intra_die_performance_compare_"))
            self.assertTrue((args.output / "comparison.json").is_file())
            self.assertEqual((args.output / "SUCCESS").read_text().strip(), comparison["id"])

    def test_request_binds_actual_timing_input_digests(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); spec = root / "spec.yaml"
            spec.write_text(yaml.safe_dump({"policy": {"intra_die": "optimized"}}))
            args = self._args(root, spec)
            args.hardware.write_text("hardware")
            args.simulation.write_text("simulation")
            request = build_request(args, "auto")
            import hashlib
            self.assertEqual(
                request.intra_die_refine_options.timing_hardware_digest,
                hashlib.sha256(b"hardware").hexdigest(),
            )
            self.assertEqual(
                request.intra_die_refine_options.timing_simulation_digest,
                hashlib.sha256(b"simulation").hexdigest(),
            )

    def test_nonoptimized_spec_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            spec = Path(raw) / "spec.yaml"
            spec.write_text(yaml.safe_dump({"policy": {"intra_die": "naive"}}))
            with self.assertRaisesRegex(ValueError, "intra_die=optimized"):
                require_performance_spec(spec)


if __name__ == "__main__":
    unittest.main()
