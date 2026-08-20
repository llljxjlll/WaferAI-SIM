from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


_HERE = Path(__file__).resolve().parent
sys.path[:0] = (str(_HERE), str(_HERE.parent / "unit"))

import freeze_stage4_pd_baseline as freezer  # noqa: E402
from llm.frontend.wafer_frontend.errors import SchemaError  # noqa: E402
from llm.frontend.wafer_frontend.schema import (  # noqa: E402
    CapabilityManifest,
    CapabilityStage,
    Stage4PdCaseMatrix,
)
from llm.frontend.wafer_frontend.schema.stage4_pd_evidence import (  # noqa: E402
    Stage4PdNamedDigest,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_json,
    load_json_dataclass,
)
import run_stage4_pd_runtime as runtime  # noqa: E402
from stage4_pd_cases import Stage4PdCaseKind  # noqa: E402
import test_run_stage4_pd_runtime as runtime_test  # noqa: E402


class Stage4PdBaselineFreezerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.official = self.root / "official"
        self.prior = (
            freezer._ROOT / "notes/frontend/baselines/stage3-static-profile-v1"
        )
        tools = {}
        for name in ("finalizer", "npusim", "resolver", "simulation"):
            path = self.root / name
            path.write_text(f"{name}-phase1\n", encoding="utf-8")
            tools[name] = path
        self.args = argparse.Namespace(
            prior_root=self.prior,
            official_root=self.official,
            finalizer=tools["finalizer"],
            npusim=tools["npusim"],
            resolver=tools["resolver"],
            simulation=tools["simulation"],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _reports(self):
        actual_tools = freezer._tool_digests(self.args)
        simulation_digest = freezer._sha256(self.args.simulation)
        result = []
        artifact = b"stage4-runtime-parser"
        for kind in (
            Stage4PdCaseKind.FUSED,
            Stage4PdCaseKind.PDS,
            Stage4PdCaseKind.PDR,
        ):
            case_name = next(
                name
                for name, candidate in freezer._CASE_KINDS.items()
                if candidate is kind
            )
            case, oracle = freezer._production_case(case_name)
            static = runtime._validate_static_case(case, oracle)
            contract, memory = freezer._production_runtime_inputs(
                case_name,
                hashlib.sha256(b"stage4-runtime-parser").hexdigest(),
            )
            with patch.object(
                runtime_test,
                "_case",
                return_value=(case, oracle, static, contract, memory),
            ):
                output = runtime_test._synthetic_output(kind)
            observation = runtime._observe_runtime(
                case,
                oracle,
                output,
                contract.program_artifact_sha256,
                contract,
                memory,
            )
            finalization = {
                "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
                "artifact_bytes": len(artifact),
                "linked_manifest_id": case.manifest.id,
                "linked_manifest_digest": freezer.canonical_digest(
                    case.manifest
                ),
                "record_count": static.record_count,
                "relocation_count": static.address_relocation_count,
            }
            report = runtime._build_report(
                case,
                oracle,
                static,
                finalization,
                artifact,
                contract,
                (observation, observation),
                tool_digests=tuple(
                    Stage4PdNamedDigest(name, digest)
                    for name, digest in sorted(actual_tools.items())
                ),
                hardware_digest=hashlib.sha256(
                    case.runtime_hardware_inputs.hardware_json.encode("utf-8")
                ).hexdigest(),
                simulation_digest=simulation_digest,
                mapping_digest=hashlib.sha256(
                    case.runtime_hardware_inputs.mapping_text.encode("utf-8")
                ).hexdigest(),
            )
            resolver = (
                f"initializations={len(contract.initializations)} "
                f"probes={len(contract.output_probes)}\n"
            )
            entries = runtime._report_text_entries(
                kind,
                case=case,
                oracle=oracle,
                runtime_report=report,
                contract=contract,
                finalization=finalization,
                finalizer_logs=("finalizer pass\n", "finalizer pass\n"),
                resolver_log=resolver,
                runtime_logs=(output, output),
            )
            result.append((report, entries))
        return tuple(result)

    def _write_official(self, count: int = 3) -> None:
        self.official.mkdir()
        for kind, (report, entries) in zip(
            freezer._CASE_ORDER[:count], self._reports()[:count], strict=True
        ):
            root = self.official / kind
            root.mkdir()
            for name, value in entries:
                (root / name).write_text(value, encoding="utf-8")

    def _stage(self, name: str) -> tuple[Path, str]:
        staging = self.root / name
        staging.mkdir()
        prior_digest = freezer._tree_digest(self.prior)
        freezer._stage_from_official(staging, self.args, prior_digest)
        freezer._validate_staging(staging, self.args, prior_digest)
        return staging, prior_digest

    def test_disk_rebuild_determinism_scores_and_truthful_p1(self) -> None:
        self._write_official()
        prior_before = freezer._tree_digest(self.prior)
        first, _ = self._stage("staging-a")
        second, _ = self._stage("staging-b")
        self.assertEqual(freezer._tree_digest(first), freezer._tree_digest(second))
        self.assertEqual(freezer._tree_digest(self.prior), prior_before)
        self.assertFalse(any(first.rglob("*.npup")))
        matrix = load_json_dataclass(
            Stage4PdCaseMatrix,
            first / "stage4_pd_case_matrix.json",
            path="stage4.matrix",
        )
        manifest = load_json_dataclass(
            CapabilityManifest,
            first / "capability_manifest.json",
            path="stage4.capability",
        )
        self.assertTrue(matrix.stage4_ready)
        self.assertEqual(manifest.coverage_score(CapabilityStage.S1), 11 / 2)
        self.assertEqual(manifest.acceptance_score(CapabilityStage.S1), (1, 3))
        review = json.loads(
            (first / "baseline_review.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("deferred_raw_evidence", review)
        self.assertTrue(
            any("complete current runner" in item for item in review["caveats"])
        )
        for kind in freezer._CASE_ORDER:
            raw = json.loads(
                (first / "stage4" / kind / "inputs/input_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                raw["raw_evidence_status"],
                "complete-current-runner-report-root",
            )

    def test_missing_pdr_and_unreviewed_outputs_fail_closed(self) -> None:
        self._write_official(2)
        staging = self.root / "missing"
        staging.mkdir()
        with self.assertRaisesRegex(RuntimeError, "directory set changed"):
            freezer._stage_from_official(
                staging, self.args, freezer._tree_digest(self.prior)
            )
        self.assertEqual(tuple(staging.iterdir()), ())
        for index, kind in enumerate(("tamper", "extra", "symlink")):
            self.official = self.root / f"official-{index}"
            self.args.official_root = self.official
            self._write_official()
            if kind == "tamper":
                path = self.official / "pdr/runtime_report.json"
                raw = json.loads(path.read_text(encoding="utf-8"))
                raw["model_functional"] = True
                path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            elif kind == "extra":
                (self.official / "pdr/unreviewed.log").write_text(
                    "unexpected\n", encoding="utf-8"
                )
            else:
                (self.official / "pdr/alias.log").symlink_to(
                    self.official / "pdr/runtime.0.log"
                )
            rejected = self.root / f"rejected-{index}"
            rejected.mkdir()
            with self.subTest(kind), self.assertRaises(
                (RuntimeError, SchemaError)
            ):
                freezer._stage_from_official(
                    rejected, self.args, freezer._tree_digest(self.prior)
                )

    def test_current_production_rebuild_rejects_forged_inputs_and_logs(
        self,
    ) -> None:
        self._write_official()
        fused_plan = self.official / "fused/plan.json"
        original_plan = fused_plan.read_text(encoding="utf-8")
        fused_plan.write_text(
            (self.official / "pds/plan.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "formal inputs differ"):
            freezer._load_official_case(
                self.official / "fused", "fused", self.args
            )
        fused_plan.write_text(original_plan, encoding="utf-8")

        runtime_log = self.official / "fused/runtime.0.log"
        original_log = runtime_log.read_text(encoding="utf-8")
        self.assertIn("makespan_cycles=1234", original_log)
        runtime_log.write_text(
            original_log.replace(
                "makespan_cycles=1234", "makespan_cycles=1235", 1
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(RuntimeError, "does not rebuild its report"):
            freezer._load_official_case(
                self.official / "fused", "fused", self.args
            )

    def test_npup_overwrite_and_atomic_no_clobber(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ProgramArtifact"):
            freezer._write_new(self.root / "forbidden.npup", "bytes")
        exclusive = self.root / "exclusive.json"
        freezer._write_new(exclusive, {"first": True})
        with self.assertRaisesRegex(RuntimeError, "overwrite"):
            freezer._write_new(exclusive, {"second": True})
        staging = self.root / "publish-source"
        staging.mkdir()
        (staging / "marker").write_text("reviewed\n", encoding="utf-8")
        target = self.root / freezer._BASELINE_EPOCH
        freezer._atomic_publish_noreplace(staging, target)
        self.assertEqual((target / "marker").read_text(), "reviewed\n")
        raced = self.root / "publish-raced"
        raced.mkdir()
        with self.assertRaisesRegex(RuntimeError, "overwrite"):
            freezer._atomic_publish_noreplace(raced, target)


if __name__ == "__main__":
    unittest.main()
