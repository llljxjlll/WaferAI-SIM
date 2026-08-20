from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

from freeze_stage1a_baseline import (
    _copy_regression,
    _raw_report_evidence,
    _review,
    _reviewed_diff,
    _stage1a_case_command,
    _tree_digest,
    _validate_negative,
)
from llm.frontend.wafer_frontend.passes import build_stage1a_capability_manifest
from llm.frontend.wafer_frontend.schema import (
    CapabilityManifest,
    CaseMatrix,
    Stage1aOracle,
    Stage1aRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json, load_json_dataclass
from llm.frontend.wafer_frontend.schema.stage1a_evidence import Stage1aCase


_STAGE0 = Path("notes/frontend/baselines/stage0-policy-provenance-v1")
_STAGE1A = Path("notes/frontend/baselines/stage1a-persistent-state-v1")


class Stage1aBaselineFreezerTest(unittest.TestCase):
    @staticmethod
    def _raw_report_root(root: Path, case: Stage1aCase) -> None:
        stem = case.value.lower()
        values = {
            f"{stem}.oracle.json": "{}",
            f"{stem}.runtime.json": "{}",
            f"{stem}.finalizer.0.log": "",
            f"{stem}.finalizer.1.log": "",
            f"{stem}.resolver.log": "resolved",
            f"{stem}.runtime.0.log": "[SIM_RESULT] x\n[D2D_TYPE] x\n",
            f"{stem}.runtime.1.log": "[SIM_RESULT] x\n[D2D_TYPE] x\n",
        }
        if case is Stage1aCase.PD1:
            values[f"{stem}.dramsys-negative.log"] = (
                "HBM backend does not support debug peeking"
            )
        for name, value in values.items():
            (root / name).write_text(value, encoding="utf-8")

    def test_raw_report_exact_set_and_pd_negative_are_strict(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage1a-raw-report-") as raw:
            root = Path(raw)
            self._raw_report_root(root, Stage1aCase.PD1)
            observed = _raw_report_evidence(Stage1aCase.PD1, root)
            self.assertEqual(len(observed), 8)
            (root / "extra.log").write_text("forged", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "evidence set changed"):
                _raw_report_evidence(Stage1aCase.PD1, root)

    def test_real_stage1a_runner_argv_has_one_exact_finalizer_value(self) -> None:
        args = Namespace(
            finalizer=Path("/build/finalizer"),
            hardware=Path("/inputs/hardware.json"),
            mapping=Path("/inputs/mapping.spec"),
            npusim=Path("/build/npusim"),
            resolver=Path("/build/resolver"),
            runtime_root=Path("/runtime"),
            simulation=Path("/inputs/simulation.json"),
        )
        command = _stage1a_case_command(
            Stage1aCase.P1, args, Path("/reports")
        )
        self.assertEqual(command.count(str(args.finalizer)), 1)
        position = command.index("--finalizer")
        self.assertEqual(command[position + 1], str(args.finalizer))
        self.assertEqual(command.count("--finalizer"), 1)

    def test_identity_e1_e2_diff_is_stable_and_reviewed(self) -> None:
        observed = _reviewed_diff(
            _STAGE0,
            {"E1": _STAGE0 / "e1", "E2": _STAGE0 / "e2"},
        )
        repeated = _reviewed_diff(
            _STAGE0,
            {"E1": _STAGE0 / "e1", "E2": _STAGE0 / "e2"},
        )
        self.assertEqual(observed, repeated)
        self.assertEqual(
            tuple((row["case"], row["changed_fields"]) for row in observed["cases"]),
            (("E1", ()), ("E2", ())),
        )

    def test_regression_copy_never_persists_program_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage1a-freezer-test-") as raw:
            destination = Path(raw) / "e1"
            _copy_regression(_STAGE0 / "e1", destination)
            self.assertTrue((destination / "run_report.json").is_file())
            self.assertTrue((destination / "program/program_io.json").is_file())
            self.assertFalse(any(destination.rglob("*.npup")))

    def test_checked_baseline_rebuilds_from_strict_evidence(self) -> None:
        self.assertTrue(_STAGE1A.is_dir())
        self.assertFalse(any(_STAGE1A.rglob("*.npup")))
        args = Namespace(
            finalizer=Path("build/npusim_program_finalizer"),
            hardware=Path("llm/test/sram/hardware_numa.json"),
            mapping=Path("llm/test/default/mapping.spec"),
            npusim=Path("build/npusim"),
            resolver=Path("build/npusim_program_io_selftest"),
            simulation=Path("llm/test/sram/simulation.json"),
        )
        negative = _validate_negative(
            json.loads(
                (_STAGE1A / "negative/negative_evidence.json").read_text(
                    encoding="utf-8"
                )
            ),
            args,
        )
        cases = (Stage1aCase.P1, Stage1aCase.K1, Stage1aCase.PD1)
        oracles = {
            case: load_json_dataclass(
                Stage1aOracle,
                _STAGE1A / case.value.lower() / "oracle.json",
                path=f"{case.value}.oracle",
            )
            for case in cases
        }
        reports = {
            case: load_json_dataclass(
                Stage1aRuntimeReport,
                _STAGE1A / case.value.lower() / "runtime_report.json",
                path=f"{case.value}.runtime_report",
            )
            for case in cases
        }
        matrix = load_json_dataclass(
            CaseMatrix, _STAGE1A / "case_matrix.json", path="stage1a.matrix"
        )
        manifest = load_json_dataclass(
            CapabilityManifest,
            _STAGE1A / "capability_manifest.json",
            path="stage1a.manifest",
        )
        prior_matrix = load_json_dataclass(
            CaseMatrix, _STAGE0 / "case_matrix.json", path="stage0.matrix"
        )
        prior_manifest = load_json_dataclass(
            CapabilityManifest,
            _STAGE0 / "capability_manifest.json",
            path="stage0.manifest",
        )
        rebuilt = build_stage1a_capability_manifest(
            prior_matrix,
            prior_manifest,
            p1_oracle=oracles[Stage1aCase.P1],
            p1_report=reports[Stage1aCase.P1],
            k1_oracle=oracles[Stage1aCase.K1],
            k1_report=reports[Stage1aCase.K1],
            pd1_oracle=oracles[Stage1aCase.PD1],
            pd1_report=reports[Stage1aCase.PD1],
            state_fail_closed_evidence_digest=canonical_digest(negative),
        )
        self.assertEqual(rebuilt, (matrix, manifest))
        diff = _reviewed_diff(
            _STAGE0, {"E1": _STAGE1A / "e1", "E2": _STAGE1A / "e2"}
        )
        self.assertEqual(
            canonical_json(diff),
            (_STAGE1A / "e1_e2_reviewed_diff.json")
            .read_text(encoding="utf-8")
            .strip(),
        )
        review = _review(
            prior_root=_STAGE0.resolve(),
            diff=diff,
            negative=negative,
            oracles=oracles,
            reports=reports,
        )
        self.assertEqual(
            canonical_json(review),
            (_STAGE1A / "baseline_review.json")
            .read_text(encoding="utf-8")
            .strip(),
        )
        self.assertEqual(
            review["prior_baseline_tree_digest"], _tree_digest(_STAGE0)
        )


if __name__ == "__main__":
    unittest.main()
