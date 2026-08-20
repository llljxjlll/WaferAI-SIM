from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import freeze_stage2_dense_forward_baseline as freezer
from freeze_stage2_dense_forward_baseline import (
    _BASELINE_EPOCH,
    _build_stage2_negative_evidence,
    _EXACT_REASON_MAP,
    _EXPECTED_CHANGE_CONTRACT,
    _FROZEN_NORMALIZED_GOLDENS,
    _NEGATIVE_BINDING_SHAPE,
    _OFFICIAL_NAIVE_MARKER_SUFFIX,
    _ROOT,
    _SOURCE_PATHS,
    _STAGE2_NEGATIVE_KEYS,
    _STAGE2_NEGATIVE_SCHEMA_VERSION,
    _STAGE2_RAW_SUFFIXES,
    _STAGE1A_CASE_FILES,
    _STAGE2_CASE_FILES,
    _NEGATIVE_FILES,
    _NAIVE_RESULT_FILES,
    _CURRENT_REGRESSION_GOLDENS,
    _atomic_publish_noreplace,
    _checked_summary,
    _recursive_diff,
    _rename_noreplace,
    _require_canonical_file,
    _require_exact_subtree,
    _sha256,
    _tree_digest,
    _validate_official_naive_gate,
    _validate_checked_summary,
    _validate_exact_change_contract,
    _validate_publish_target,
    _validate_stage2_negative,
    _validate_stage2_provenance,
    _validate_staging,
    _write_files,
    _write_new,
)
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.frontend.wafer_frontend.schema.stage2_dense_forward_evidence import (
    Stage2DenseForwardToolEvidence,
)
from run_stage2_dense_forward_negative_evidence import _valid_runtime_report
from stage2_dense_forward_cases import build_stage2_dense_forward_case


_PRIOR = Path("notes/frontend/baselines/stage1a-persistent-state-v1")
_DIGEST = "1" * 64


def _negative_evidence() -> dict[str, object]:
    witnesses = []
    for key in _STAGE2_NEGATIVE_KEYS:
        tools, sources, inputs = _NEGATIVE_BINDING_SHAPE[key]
        bindings = {
            "tools": [
                {
                    "name": name,
                    "sha256": (
                        _sha256(Path(sys.executable).resolve())
                        if name == "python"
                        else _DIGEST
                    ),
                }
                for name in tools
            ],
            "sources": [
                {"path": path, "sha256": _sha256(_ROOT / path)}
                for path in sources
            ],
            "inputs": [
                {"name": name, "sha256": _DIGEST}
                for name in inputs
            ],
        }
        witnesses.append(
            {
                "key": key,
                "passed": True,
                "expected_message": f"{key}: rejected",
                "observed_error": f"{key}: rejected",
                "bindings": bindings,
            }
        )
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "command": ("python3", "negative.py"),
        "source_digests": tuple(
            {"path": path, "sha256": _sha256(_ROOT / path)}
            for path in _SOURCE_PATHS
        ),
        "witnesses": tuple(witnesses),
    }
    return {
        "schema_version": _STAGE2_NEGATIVE_SCHEMA_VERSION,
        "producer_pass": "stage2_dense_forward_negative_runner",
        "id": stable_artifact_id(
            "stage2_dense_forward_negative_evidence",
            semantic_key,
            schema_version=_STAGE2_NEGATIVE_SCHEMA_VERSION,
        ),
        **json.loads(json.dumps(semantic_key)),
    }


class Stage2DenseForwardBaselineFreezerTest(unittest.TestCase):
    @staticmethod
    def _minimal_staging_fixture(root: Path) -> None:
        for case in ("p1", "k1", "pd1"):
            paths = set(_STAGE1A_CASE_FILES)
            if case == "pd1":
                paths.add("run/dramsys-negative.log")
            for relative in paths:
                path = root / "stage1a" / case / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            (root / "stage1a" / case / "SUCCESS").write_text(
                f"report-{case}\n", encoding="utf-8"
            )
        for relative in _NEGATIVE_FILES:
            path = root / "stage1a/negative" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

        for label in ("e1", "e2"):
            paths = set(_NAIVE_RESULT_FILES) | {
                "inputs/freezer_input_summary.json",
                "official_wrapper.stderr.log",
                "official_wrapper.stdout.log",
            }
            for relative in paths:
                path = root / "stage1a" / label / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            upper = label.upper()
            marker = (
                f"[NAIVE N7 {upper}] PASS: "
                + (
                    "report=naive_run_report_fixture"
                    if upper == "E1"
                    else ""
                )
                + _OFFICIAL_NAIVE_MARKER_SUFFIX[upper]
                + "\n"
            )
            (root / "stage1a" / label / "official_wrapper.stdout.log").write_text(
                marker, encoding="utf-8"
            )
            (root / "stage1a" / label / "official_wrapper.stderr.log").write_text(
                "", encoding="utf-8"
            )
            (root / "stage1a" / label / "SUCCESS").write_text(
                f"report-{label}\n", encoding="utf-8"
            )

        for tp in (1, 2, 4):
            stem = f"stage2-tp{tp}"
            paths = set(_STAGE2_CASE_FILES) | {
                f"raw/{stem}.{suffix}" for suffix in _STAGE2_RAW_SUFFIXES
            }
            for relative in paths:
                path = root / "stage2" / f"tp{tp}" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            tp_root = root / "stage2" / f"tp{tp}"
            (tp_root / "SUCCESS").write_text(
                f"report-tp{tp}\n", encoding="utf-8"
            )
            (tp_root / f"raw/{stem}.resolver.log").write_text(
                "resolved\n", encoding="utf-8"
            )
            for index in range(2):
                (tp_root / f"raw/{stem}.runtime.{index}.log").write_text(
                    "[SIM_RESULT] ok\n[D2D_TYPE] ok\n", encoding="utf-8"
                )
        for relative in _NEGATIVE_FILES:
            path = root / "stage2/negative" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        for name in (
            "baseline_review.json",
            "capability_manifest.json",
            "case_matrix.json",
            "checked_rebuild_summary.json",
            "stage1a_regression_reviewed_diff.json",
        ):
            (root / name).write_text("{}\n", encoding="utf-8")

    @unittest.skipUnless(
        Path("build/npusim_program_finalizer").is_file()
        and Path("build/npusim_program_io_selftest").is_file(),
        "production negative binaries are not built",
    )
    def test_production_negative_rebuild_uses_real_binary_args(self) -> None:
        args = Namespace(
            finalizer=Path("build/npusim_program_finalizer").resolve(),
            resolver=Path("build/npusim_program_io_selftest").resolve(),
            runtime_root=Path("build").resolve(),
        )
        evidence = json.loads(
            canonical_json(_build_stage2_negative_evidence(args))
        )
        self.assertEqual(
            _validate_stage2_negative(evidence, args=args),
            evidence,
        )

    @unittest.skipUnless(
        Path("build/npusim").is_file()
        and Path("build/npusim_program_finalizer").is_file()
        and Path("build/npusim_program_io_selftest").is_file(),
        "runtime binaries are not built",
    )
    def test_stage2_provenance_rejects_tool_and_simulation_drift(self) -> None:
        args = Namespace(
            finalizer=Path("build/npusim_program_finalizer").resolve(),
            resolver=Path("build/npusim_program_io_selftest").resolve(),
            npusim=Path("build/npusim").resolve(),
            simulation=Path("llm/test/sram/simulation.json").resolve(),
        )
        case = build_stage2_dense_forward_case(1)
        report = _valid_runtime_report(case)
        report = replace(
            report,
            tools=Stage2DenseForwardToolEvidence(
                _sha256(args.finalizer),
                _sha256(args.resolver),
                _sha256(args.npusim),
            ),
            simulation_digest=_sha256(args.simulation),
        )
        _validate_stage2_provenance(1, report, args)
        with self.assertRaisesRegex(RuntimeError, "provenance changed"):
            _validate_stage2_provenance(
                1,
                replace(
                    report,
                    tools=replace(
                        report.tools, finalizer_sha256="0" * 64
                    ),
                ),
                args,
            )
        with self.assertRaisesRegex(RuntimeError, "provenance changed"):
            _validate_stage2_provenance(
                1,
                replace(report, simulation_digest="0" * 64),
                args,
            )

    def test_full_staging_validator_rejects_derived_tamper_and_extra_sibling(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-full-staging-") as raw:
            staging = Path(raw) / "staging"
            staging.mkdir()
            self._minimal_staging_fixture(staging)
            args = Namespace(
                prior_root=_PRIOR.resolve(),
                finalizer=Path("build/npusim_program_finalizer").resolve(),
                resolver=Path("build/npusim_program_io_selftest").resolve(),
                npusim=Path("build/npusim").resolve(),
                hardware=Path("llm/test/sram/hardware_numa.json").resolve(),
                simulation=Path("llm/test/sram/simulation.json").resolve(),
                mapping=Path("llm/test/default/mapping.spec").resolve(),
                e1_spec=Path("notes/frontend/examples/naive_dense_tp2.yaml").resolve(),
                e1_hardware=Path("notes/frontend/examples/hardware_2x1.json").resolve(),
                e2_spec=Path("notes/frontend/examples/naive_dense_tp4.yaml").resolve(),
                e2_hardware=Path("notes/frontend/examples/hardware_2x2.json").resolve(),
            )
            extra = staging / "stage2/forged-sibling"
            extra.mkdir()
            with self.assertRaisesRegex(RuntimeError, "immediate child set"):
                _validate_staging(
                    staging,
                    args=args,
                    prior_tree_digest=_tree_digest(_PRIOR),
                )
            extra.rmdir()

            stage1_reports = {
                label: SimpleNamespace(
                    id=f"report-{label}",
                    hardware_digest="hardware",
                    validate_against=MagicMock(),
                )
                for label in ("p1", "k1", "pd1")
            }
            naive_reports = {
                label: SimpleNamespace(id=f"report-{label}")
                for label in ("e1", "e2")
            }
            stage2_reports = {
                tp: SimpleNamespace(
                    id=f"report-tp{tp}",
                    validate_against=MagicMock(),
                    artifact=SimpleNamespace(
                        linked_manifest_id=None,
                        linked_manifest_digest=None,
                        program_artifact_sha256=None,
                        artifact_size_bytes=None,
                        record_count=None,
                        relocation_count=None,
                    ),
                    sidecar=SimpleNamespace(
                        contract_id=None, contract_digest=None
                    ),
                    tools=SimpleNamespace(),
                    compile=SimpleNamespace(
                        template_digest=None,
                        ir1_digest=None,
                        global_dag_digest=None,
                        lowered_digest=None,
                    ),
                    hardware_digest=None,
                    simulation_digest=None,
                    mapping_digest=None,
                )
                for tp in (1, 2, 4)
            }
            fake_manifest = SimpleNamespace(id=None, validate_against=MagicMock())
            fake_sidecar = SimpleNamespace(id=None, validate_against=MagicMock())
            fake_matrix = SimpleNamespace()
            fake_capability = SimpleNamespace(
                validate_against=MagicMock(),
                coverage_score=lambda stage: (
                    Fraction(7, 2)
                    if stage is freezer.CapabilityStage.S1
                    else Fraction(0, 1)
                ),
                acceptance_score=lambda stage: (1, 3),
            )

            def fake_load(cls, _path, *, path):
                if cls is freezer.Stage1aOracle:
                    return SimpleNamespace()
                if cls is freezer.Stage1aRuntimeReport:
                    return stage1_reports[path.split(".")[1]]
                if cls is freezer.NaiveRunReport:
                    return naive_reports[path.split(".")[1].lower()]
                if cls is freezer.Stage2DenseForwardOracle:
                    return SimpleNamespace()
                if cls is freezer.Stage2DenseForwardRuntimeReport:
                    tp = int(path.split(".")[1][2:])
                    return stage2_reports[tp]
                if cls is freezer.LinkedProgramManifest:
                    return fake_manifest
                if cls is freezer.ProgramIoContract:
                    return fake_sidecar
                if cls is freezer.CaseMatrix:
                    return fake_matrix
                if cls is freezer.CapabilityManifest:
                    return fake_capability
                raise AssertionError((cls, path))

            original_require = freezer._require_canonical_file

            def selective_require(path, expected, *, label, trailing_newline=True):
                if label in {
                    "case matrix",
                    "capability manifest",
                    "Stage1a reviewed diff",
                    "baseline review",
                }:
                    return original_require(
                        path, expected, label=label, trailing_newline=trailing_newline
                    )
                return None

            with (
                patch.object(freezer, "load_json_dataclass", side_effect=fake_load),
                patch.object(freezer, "canonical_digest", return_value=None),
                patch.object(freezer, "_validate_stage1a_disk_closure"),
                patch.object(freezer, "_validate_naive_disk_closure"),
                patch.object(freezer, "_validate_stage2_provenance"),
                patch.object(
                    freezer,
                    "_stage1a_summary",
                    side_effect=(
                        _CURRENT_REGRESSION_GOLDENS["P1"],
                        _CURRENT_REGRESSION_GOLDENS["K1"],
                        _CURRENT_REGRESSION_GOLDENS["PD1"],
                    ),
                ),
                patch.object(
                    freezer,
                    "_naive_summary",
                    side_effect=(
                        _CURRENT_REGRESSION_GOLDENS["E1"],
                        _CURRENT_REGRESSION_GOLDENS["E2"],
                    ),
                ),
                patch.object(freezer, "_validate_stage1a_negative"),
                patch.object(
                    freezer, "_validate_stage2_negative", return_value={}
                ),
                patch.object(
                    freezer,
                    "_build_capability",
                    return_value=({"matrix": "derived"}, {"manifest": "derived"}),
                ),
                patch.object(freezer, "_require_canonical_file", side_effect=selective_require),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "case matrix differs from the disk-derived rebuild"
                ):
                    _validate_staging(
                        staging,
                        args=args,
                        prior_tree_digest=_tree_digest(_PRIOR),
                    )

    def test_official_naive_wrapper_marker_is_exact(self) -> None:
        e1 = subprocess.CompletedProcess(
            args=("wrapper",),
            returncode=0,
            stdout=(
                "[NAIVE N7 E1] PASS: report=naive_run_report_exact"
                + _OFFICIAL_NAIVE_MARKER_SUFFIX["E1"]
                + "\n"
            ),
            stderr="",
        )
        self.assertIn(
            "dma_in=18", _validate_official_naive_gate("E1", e1)
        )
        missing = subprocess.CompletedProcess(
            args=("wrapper",), returncode=0, stdout="", stderr=""
        )
        with self.assertRaisesRegex(RuntimeError, "wrapper gate failed"):
            _validate_official_naive_gate("E1", missing)
        drift = subprocess.CompletedProcess(
            args=("wrapper",),
            returncode=0,
            stdout=e1.stdout.replace("dma_in=18", "dma_in=19"),
            stderr="",
        )
        with self.assertRaisesRegex(RuntimeError, "static marker changed"):
            _validate_official_naive_gate("E1", drift)

    def test_subprocess_runner_prepends_root_to_existing_pythonpath(self) -> None:
        completed = subprocess.CompletedProcess(
            args=("python",), returncode=0, stdout="", stderr=""
        )
        with (
            patch.dict(freezer.os.environ, {"PYTHONPATH": "/existing"}),
            patch.object(freezer.subprocess, "run", return_value=completed) as run,
        ):
            self.assertIs(
                freezer._run(["python"], cwd=_ROOT, timeout=1), completed
            )
        self.assertEqual(
            run.call_args.kwargs["env"]["PYTHONPATH"],
            str(_ROOT) + freezer.os.pathsep + "/existing",
        )
        self.assertEqual(run.call_args.kwargs["cwd"], _ROOT)

    def test_exact_leaf_contract_has_no_wildcards_and_rejects_nested_added(
        self,
    ) -> None:
        expected_counts = {"P1": 16, "K1": 63, "PD1": 27, "E1": 82, "E2": 84}
        for label, golden in _FROZEN_NORMALIZED_GOLDENS.items():
            with self.subTest(label=label):
                changes = _recursive_diff(
                    golden["old_changed_leaves"],
                    golden["current_changed_leaves"],
                )
                _validate_exact_change_contract(label, changes)
                self.assertEqual(len(changes), expected_counts[label])
                self.assertFalse(
                    any("*" in path for path in _EXPECTED_CHANGE_CONTRACT[label])
                )
                self.assertEqual(
                    set(_EXACT_REASON_MAP[label]),
                    set(_EXPECTED_CHANGE_CONTRACT[label]),
                )

        for label, path in (
            ("E1", "runtime.forged_nested"),
            ("E1", "static_metrics.forged_nested"),
            ("E1", "artifact.forged_nested"),
            ("K1", "artifact.forged_nested"),
            ("K1", "sidecar.forged_nested"),
        ):
            changes = list(
                _recursive_diff(
                    _FROZEN_NORMALIZED_GOLDENS[label]["old_changed_leaves"],
                    _FROZEN_NORMALIZED_GOLDENS[label]["current_changed_leaves"],
                )
            )
            changes.append(
                {
                    "kind": "added",
                    "new": 1,
                    "new_present": True,
                    "old": None,
                    "old_present": False,
                    "path": path,
                }
            )
            with self.subTest(label=label, path=path):
                with self.assertRaisesRegex(RuntimeError, "unknown="):
                    _validate_exact_change_contract(label, tuple(changes))

    def test_negative_evidence_has_exact_witness_and_binding_closure(self) -> None:
        evidence = _negative_evidence()
        self.assertEqual(
            _validate_stage2_negative(
                evidence, expected_evidence=evidence
            ),
            evidence,
        )
        bad = json.loads(json.dumps(evidence))
        bad["witnesses"][0]["bindings"]["extra"] = [
            {"name": "forged", "sha256": _DIGEST}
        ]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            _validate_stage2_negative(bad)
        bad = json.loads(json.dumps(evidence))
        bad["witnesses"][0]["observed_error"] = "accepted"
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            _validate_stage2_negative(bad)

        restable = json.loads(json.dumps(evidence))
        restable["witnesses"][0]["bindings"]["inputs"][0][
            "sha256"
        ] = "2" * 64
        semantic_key = {
            key: restable[key]
            for key in restable
            if key not in ("schema_version", "producer_pass", "id")
        }
        restable["id"] = stable_artifact_id(
            "stage2_dense_forward_negative_evidence",
            semantic_key,
            schema_version=_STAGE2_NEGATIVE_SCHEMA_VERSION,
        )
        with self.assertRaisesRegex(RuntimeError, "independently rebuilt"):
            _validate_stage2_negative(
                restable, expected_evidence=evidence
            )

    def test_checked_summary_is_byte_exact_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-summary-") as raw:
            root = Path(raw)
            _write_new(root / "payload.json", {"value": 1})
            summary = _checked_summary(root, "2" * 64)
            _write_new(root / "checked_rebuild_summary.json", summary)
            self.assertEqual(
                canonical_json(_validate_checked_summary(root)),
                canonical_json(summary),
            )
            with self.assertRaisesRegex(RuntimeError, "prior tree digest"):
                _validate_checked_summary(root, "3" * 64)
            (root / "payload.json").write_text("{\"value\":2}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inventory changed"):
                _validate_checked_summary(root)

    def test_writes_reject_overwrite_symlink_npup_and_traversal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-write-") as raw:
            root = Path(raw)
            _write_files(root, {"nested/value.txt": "value"})
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                _write_files(root, {"nested/value.txt": "new"})
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                _write_files(root, {"program.npup": b"bytes"})
            with self.assertRaisesRegex(RuntimeError, "invalid"):
                _write_files(root, {"../escape.txt": "escape"})
            target = root / "target.txt"
            target.write_text("keep", encoding="utf-8")
            link = root / "link.txt"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                _write_files(root, {"link.txt": "forged"})
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

            subtree = root / "subtree"
            _write_files(subtree, {"expected.txt": "value"})
            _require_exact_subtree(
                subtree, {"expected.txt"}, label="focused"
            )
            (subtree / "extra.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "subtree shape changed"):
                _require_exact_subtree(
                    subtree, {"expected.txt"}, label="focused"
                )

            derived = root / "derived.json"
            _write_new(derived, {"value": 1})
            _require_canonical_file(
                derived, {"value": 1}, label="derived"
            )
            derived.write_text('{"value":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "disk-derived rebuild"):
                _require_canonical_file(
                    derived, {"value": 1}, label="derived"
                )

            raw = root / "raw.json"
            raw.write_text('{"value":1}', encoding="utf-8")
            _require_canonical_file(
                raw, {"value": 1}, label="raw", trailing_newline=False
            )
            raw.write_text('{"value":2}', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "disk-derived rebuild"):
                _require_canonical_file(
                    raw, {"value": 1}, label="raw", trailing_newline=False
                )

    def test_tree_and_publish_preflight_reject_symlinks_and_existing_target(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-target-") as raw:
            parent = Path(raw)
            checked = parent / "checked"
            checked.mkdir()
            (checked / "value.txt").write_text("value", encoding="utf-8")
            digest = _tree_digest(checked)
            self.assertEqual(digest, _tree_digest(checked))
            (checked / "link.txt").symlink_to(checked / "value.txt")
            with self.assertRaisesRegex(RuntimeError, "symlinks"):
                _tree_digest(checked)

            target = parent / _BASELINE_EPOCH
            _validate_publish_target(target)
            target.mkdir()
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                _validate_publish_target(target)

    def test_symlink_ancestor_and_kernel_publish_race_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage2-race-") as raw:
            parent = Path(raw)
            real_parent = parent / "real"
            real_parent.mkdir()
            linked_parent = parent / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink ancestor"):
                _validate_publish_target(linked_parent / _BASELINE_EPOCH)
            with self.assertRaisesRegex(RuntimeError, "symlink ancestor"):
                _write_files(linked_parent / "evidence", {"value.txt": "x"})

            staging = parent / "staging"
            staging.mkdir()
            (staging / "ours.txt").write_text("ours", encoding="utf-8")
            raced = parent / _BASELINE_EPOCH
            _validate_publish_target(raced)
            raced.mkdir()
            (raced / "theirs.txt").write_text("theirs", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "raced publish target"):
                _rename_noreplace(staging, raced)
            self.assertTrue((staging / "ours.txt").is_file())
            self.assertEqual(
                (raced / "theirs.txt").read_text(encoding="utf-8"),
                "theirs",
            )

            raced.rmdir() if not any(raced.iterdir()) else None
            (raced / "theirs.txt").unlink()
            raced.rmdir()
            _atomic_publish_noreplace(staging, raced)
            self.assertFalse(staging.exists())
            self.assertTrue((raced / "ours.txt").is_file())


if __name__ == "__main__":
    unittest.main()
