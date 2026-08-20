from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from freeze_stage3_static_profile_baseline import (
    _BASELINE_EPOCH,
    _REGRESSION_CASES,
    _atomic_publish_noreplace,
    _case_files,
    _checked_summary,
    _load_prior,
    _require_exact_directories,
    _require_exact_files,
    _tree_digest,
    _validate_publish_target,
    _validate_summary,
    _write_new,
)
from llm.frontend.wafer_frontend.schema.stage2_dense_forward_evidence import (
    _CASE_GOLDENS as _STAGE2_CASE_GOLDENS,
)
from run_naive_e1 import _ARTIFACT_SHA256 as _E1_ARTIFACT_SHA256
from run_naive_e2 import _ARTIFACT_SHA256 as _E2_ARTIFACT_SHA256
from run_stage1a_state_cases import _INTEGRATION_GOLDENS
from run_stage2_dense_forward import _REVIEWED_RUNTIME_GOLDENS
from stage3_decode_cases import Stage3StaticCaseKind


_ROOT = Path(__file__).resolve().parents[4]


class Stage3StaticProfileFreezerTest(unittest.TestCase):
    def test_regression_review_matches_current_runner_goldens(self) -> None:
        rows = {row["case"]: row for row in _REGRESSION_CASES}
        for case, golden in _INTEGRATION_GOLDENS.items():
            row = rows[case.value]
            self.assertEqual(
                (
                    row["new_artifact_sha256"],
                    row["artifact_size_bytes"],
                    row["record_count"],
                    row["relocation_count"],
                    row["makespan_cycles"],
                ),
                (
                    golden.artifact_sha256,
                    golden.artifact_size_bytes,
                    golden.finalizer_record_count,
                    golden.relocation_count,
                    golden.makespan_cycles,
                ),
            )
        for tp_degree, golden in _REVIEWED_RUNTIME_GOLDENS.items():
            row = rows[f"STAGE2_TP{tp_degree}"]
            schema_artifact = _STAGE2_CASE_GOLDENS[tp_degree]["artifact"]
            self.assertEqual(
                (
                    row["new_artifact_sha256"],
                    row["artifact_size_bytes"],
                    row["record_count"],
                    row["relocation_count"],
                    row["makespan_cycles"],
                ),
                (
                    golden.artifact_sha256,
                    golden.artifact_bytes,
                    golden.record_count,
                    golden.relocation_count,
                    golden.makespan_cycles,
                ),
            )
            self.assertEqual(schema_artifact[6], golden.artifact_sha256)
        self.assertEqual(
            rows["E1"]["new_artifact_sha256"], _E1_ARTIFACT_SHA256
        )
        self.assertEqual(
            rows["E2"]["new_artifact_sha256"], _E2_ARTIFACT_SHA256
        )

    def test_prior_is_strict_and_stage3_case_shape_is_exact(self) -> None:
        prior = _ROOT / "notes/frontend/baselines/stage2-dense-forward-v1"
        matrix, manifest = _load_prior(prior)
        manifest.validate_against(matrix)
        self.assertEqual(manifest.baseline_epoch, "stage2-dense-forward-v1")
        self.assertEqual(len(_tree_digest(prior)), 64)
        for kind in Stage3StaticCaseKind:
            names = _case_files(kind)
            self.assertEqual(len(names), 16)
            self.assertFalse(any(name.endswith(".npup") for name in names))

    def test_checked_summary_is_rebuilt_from_disk_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_new(root / "payload.json", {"value": 1})
            prior_digest = "1" * 64
            summary = _checked_summary(root, prior_digest)
            _write_new(root / "checked_rebuild_summary.json", summary)
            _validate_summary(root, prior_digest)

            changed = json.loads(
                (root / "checked_rebuild_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            changed["payload_tree_digest"] = "2" * 64
            (root / "checked_rebuild_summary.json").write_text(
                json.dumps(changed), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "summary changed"):
                _validate_summary(root, prior_digest)

    def test_exact_files_and_directories_reject_extras(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_new(root / "raw/evidence.json", {})
            _require_exact_files(root, {"raw/evidence.json"}, "fixture")
            _require_exact_directories(root, {"raw"}, "fixture")
            (root / "extra").mkdir()
            with self.assertRaisesRegex(RuntimeError, "directory set"):
                _require_exact_directories(root, {"raw"}, "fixture")
            with self.assertRaisesRegex(RuntimeError, "ProgramArtifact"):
                _write_new(root / "artifact.npup", "bytes")

    def test_atomic_publish_is_no_clobber_and_symlink_safe(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            target = parent / _BASELINE_EPOCH
            staging = parent / ".staging-one"
            staging.mkdir()
            _write_new(staging / "payload", "first")
            _atomic_publish_noreplace(staging, target)
            self.assertEqual((target / "payload").read_text(), "first")

            second = parent / ".staging-two"
            second.mkdir()
            _write_new(second / "payload", "second")
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                _atomic_publish_noreplace(second, target)
            self.assertEqual((target / "payload").read_text(), "first")
            self.assertEqual((second / "payload").read_text(), "second")

            real_parent = parent / "real"
            real_parent.mkdir()
            linked_parent = parent / "linked"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "symlink ancestor"):
                _validate_publish_target(linked_parent / _BASELINE_EPOCH)


if __name__ == "__main__":
    unittest.main()
