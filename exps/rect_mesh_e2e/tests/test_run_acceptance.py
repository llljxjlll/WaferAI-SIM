from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from exps.rect_mesh_e2e import run_acceptance


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "exps" / "rect_mesh_e2e" / "configs" / "cases.json"


class RunAcceptanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = run_acceptance.load_config(CONFIG)
        self.by_id = {case["case_id"]: case for case in self.config["cases"]}

    def test_shipped_config_has_bounded_evidence_classes(self) -> None:
        self.assertEqual(len(self.by_id), 8)
        self.assertEqual(
            {case["classification"] for case in self.config["cases"]},
            {
                "preflight_pass",
                "runtime_pass",
                "compile_only",
                "known_blocker",
            },
        )
        self.assertNotIn(
            "dense_inference_runtime_2x2_session_blocker",
            self.config["default_case_ids"],
        )
        self.assertEqual(
            self.by_id["p5_preflight_400"]["classification"],
            "preflight_pass",
        )
        self.assertEqual(
            self.by_id["moe_full_inference_compile_1x2"]["classification"],
            "compile_only",
        )

    def test_all_entrypoints_are_existing_allowlisted_modules(self) -> None:
        for case in self.config["cases"]:
            self.assertIn(case["entrypoint"], run_acceptance.ALLOWED_MODULES)
            if case["entrypoint"] != "unittest":
                module_path = ROOT / (case["entrypoint"].replace(".", "/") + ".py")
                self.assertTrue(module_path.is_file(), module_path)

    def test_classification_requires_exact_exit_and_markers(self) -> None:
        runtime_case = self.by_id["dense_inference_resident_1x1"]
        marker = runtime_case["expected"]["required_output_substrings"][0]
        self.assertEqual(
            run_acceptance.classify_observation(
                runtime_case, return_code=0, combined_output=marker
            ),
            ("runtime_pass", []),
        )
        outcome, problems = run_acceptance.classify_observation(
            runtime_case, return_code=1, combined_output=marker
        )
        self.assertEqual(outcome, "unexpected_result")
        self.assertTrue(any("return_code" in item for item in problems))

    def test_known_blocker_is_never_reported_as_runtime_pass(self) -> None:
        case = self.by_id["dense_inference_runtime_2x2_session_blocker"]
        marker = case["expected"]["required_output_substrings"][0]
        self.assertEqual(
            run_acceptance.classify_observation(
                case, return_code=1, combined_output=marker
            ),
            ("known_blocker_reproduced", []),
        )
        outcome, _ = run_acceptance.classify_observation(
            case,
            return_code=0,
            combined_output="Dense sequence runtime canary PASS mesh=2x2",
        )
        self.assertEqual(outcome, "unexpected_result")

    def test_config_rejects_known_blocker_as_default(self) -> None:
        changed = json.loads(json.dumps(self.config))
        changed["default_case_ids"].append(
            "dense_inference_runtime_2x2_session_blocker"
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cases.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selected explicitly"):
                run_acceptance.load_config(path)

    def test_command_expansion_uses_argument_vector_and_tmp_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(
                python="/usr/bin/python3",
                finalizer=ROOT / "build-debug-final" / "npusim_program_finalizer",
                npusim=ROOT / "build-debug-final" / "npusim",
                simulation=ROOT / "llm/test/program/p5_behavioral_simulation.json",
                timeout=321,
                output=Path(temp),
            )
            command, environment = run_acceptance.build_command(
                self.by_id["dense_training_offload_1x1"], args
            )
        self.assertEqual(
            command[:4],
            [
                str(Path("/usr/bin/python3").resolve()),
                "-B",
                "-m",
                self.by_id["dense_training_offload_1x1"]["entrypoint"],
            ],
        )
        self.assertIn("--external-offload", command)
        self.assertIn("321", command)
        output_index = command.index("--output") + 1
        self.assertEqual(
            command[output_index],
            self.by_id["dense_training_offload_1x1"]["artifact_output"],
        )
        self.assertIn(
            "llm/test/program/p5_behavioral_simulation.json",
            command,
        )
        self.assertNotIn(
            str(ROOT / "llm/test/program/p5_behavioral_simulation.json"),
            command,
        )
        self.assertIn(str(ROOT), environment["PYTHONPATH"])
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")


if __name__ == "__main__":
    unittest.main()
