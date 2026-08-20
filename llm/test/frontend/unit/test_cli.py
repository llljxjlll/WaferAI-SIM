from __future__ import annotations

import builtins
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from llm.frontend.wafer_frontend.cli import main

from _fixtures import valid_spec


class CliTest(unittest.TestCase):
    def _invoke(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_validate_spec_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.yaml"
            path.write_text(json.dumps(valid_spec()), encoding="utf-8")
            first = self._invoke(["validate-spec", str(path)])
            second = self._invoke(["validate-spec", str(path)])
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        self.assertEqual(json.loads(first[1])["status"], "valid")
        self.assertEqual(first[2], "")

    def test_validate_spec_reports_precise_failure(self) -> None:
        raw = valid_spec()
        del raw["workload"]["infer"]["profile"]["kv_pages"]  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.yaml"
            path.write_text(json.dumps(raw), encoding="utf-8")
            status, stdout, stderr = self._invoke(["validate-spec", str(path)])
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("spec.workload.infer.profile.kv_pages", stderr)

    def test_dump_empty_pipeline_is_stable(self) -> None:
        first = self._invoke(["dump-empty-pipeline"])
        second = self._invoke(["dump-empty-pipeline"])
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0)
        output = json.loads(first[1])
        self.assertEqual(output["initial_phase"], "spec_validated")
        self.assertEqual(output["passes"][0]["name"], "build_ir0")

    def test_validate_spec_rejects_top_level_and_nested_duplicate_yaml_keys(self) -> None:
        cases = (
            (
                "schema_version: first\nschema_version: second\n",
                "spec.schema_version",
                "line 2, column 1",
            ),
            (
                "workload:\n  infer:\n    source: static_profile\n"
                "    source: shape_dist\n",
                "spec.workload.infer.source",
                "line 4, column 5",
            ),
        )
        for content, error_path, location in cases:
            with self.subTest(error_path=error_path):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "spec.yaml"
                    path.write_text(content, encoding="utf-8")
                    status, stdout, stderr = self._invoke(
                        ["validate-spec", str(path)]
                    )
                self.assertEqual(status, 2)
                self.assertEqual(stdout, "")
                self.assertIn(error_path, stderr)
                self.assertIn("duplicate YAML mapping key", stderr)
                self.assertIn(location, stderr)

    def test_validate_spec_rejects_nonfinite_yaml_weights(self) -> None:
        raw = valid_spec()
        profile = raw["workload"]["infer"]["profile"]  # type: ignore[index]
        raw["workload"]["infer"] = {  # type: ignore[index]
            "source": "shape_dist",
            "output": "logits",
            "shape_dist": {
                "profiles": (
                    {"key": profile, "weight": "NONFINITE"},
                ),
            },
        }
        template = json.dumps(raw)
        for token in (".nan", ".inf", "-.inf"):
            with self.subTest(token=token):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "spec.yaml"
                    path.write_text(
                        template.replace('"NONFINITE"', token),
                        encoding="utf-8",
                    )
                    status, stdout, stderr = self._invoke(
                        ["validate-spec", str(path)]
                    )
                self.assertEqual(status, 2)
                self.assertEqual(stdout, "")
                self.assertIn(
                    "spec.workload.infer.shape_dist.profiles[0].weight",
                    stderr,
                )
                self.assertIn("non-finite YAML floats", stderr)

    def test_pyyaml_is_lazy_and_missing_dependency_is_actionable(self) -> None:
        original_import = builtins.__import__

        def import_without_yaml(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("yaml deliberately unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_yaml):
            dump_status, dump_stdout, dump_stderr = self._invoke(
                ["dump-empty-pipeline"]
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "spec.yaml"
                path.write_text(json.dumps(valid_spec()), encoding="utf-8")
                validate_status, validate_stdout, validate_stderr = self._invoke(
                    ["validate-spec", str(path)]
                )
        self.assertEqual(dump_status, 0)
        self.assertNotEqual(dump_stdout, "")
        self.assertEqual(dump_stderr, "")
        self.assertEqual(validate_status, 2)
        self.assertEqual(validate_stdout, "")
        self.assertIn("PyYAML is required", validate_stderr)


if __name__ == "__main__":
    unittest.main()
