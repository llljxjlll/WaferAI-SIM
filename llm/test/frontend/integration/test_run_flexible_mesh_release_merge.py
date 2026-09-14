from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseFamily,
    generate_flexible_mesh_release_cases,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_representative_completion import (
    FLEXIBLE_MESH_REPRESENTATIVE_SHAPES,
    select_flexible_mesh_representative_cases,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.test.frontend.unit.test_flexible_mesh_release import (
    _binding,
    _cases,
    _execution,
)

from run_flexible_mesh_release_merge import (
    _summary,
    audit_flexible_mesh_release_roots,
    run,
)
from flexible_mesh_release_profiles import release_trace_model_digests


def _evidence(case, binding) -> FlexibleMeshReleaseCaseEvidence:
    return FlexibleMeshReleaseCaseEvidence.create(
        case=case,
        binding=binding,
        executions=(
            _execution(case, binding, 0),
            _execution(case, binding, 1),
        ),
    )


def _write(root: Path, evidence: FlexibleMeshReleaseCaseEvidence) -> None:
    directory = root / evidence.case.id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "case_evidence.json").write_text(
        canonical_json(evidence), encoding="utf-8",
    )


class FlexibleMeshReleaseMergeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.binding = _binding()
        self.cases = _cases()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.roots = tuple(self.root / f"runtime-{index}" for index in range(3))
        for root in self.roots:
            root.mkdir()
            (root / "release_binding.json").write_text(
                canonical_json(self.binding), encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_missing_matrix_is_an_audit_not_completion(self) -> None:
        audit, rows = audit_flexible_mesh_release_roots(
            binding=self.binding,
            cases=self.cases,
            runtime_roots=self.roots,
        )
        self.assertFalse(audit["clean"])
        self.assertEqual(audit["missing_case_count"], 600)
        self.assertEqual(audit["valid_unique_cases"], 0)
        self.assertEqual(rows, ())

    def test_canonical_profiles_cover_six_families_without_caller_input(self) -> None:
        profiles = release_trace_model_digests()
        self.assertEqual(tuple(family for family, _digest in profiles), tuple(
            FlexibleMeshReleaseFamily
        ))
        self.assertEqual(len({digest for _family, digest in profiles}), 6)

    def test_duplicate_case_is_rejected_even_when_identical(self) -> None:
        evidence = _evidence(self.cases[0], self.binding)
        _write(self.roots[0], evidence)
        _write(self.roots[1], evidence)
        audit, _rows = audit_flexible_mesh_release_roots(
            binding=self.binding,
            cases=self.cases,
            runtime_roots=self.roots,
        )
        self.assertFalse(audit["clean"])
        self.assertEqual(audit["duplicate_count"], 1)
        self.assertEqual(audit["valid_unique_cases"], 1)
        self.assertEqual(audit["missing_case_count"], 599)

    def test_family_is_owned_by_exact_workload_root(self) -> None:
        evidence = _evidence(self.cases[0], self.binding)
        _write(self.roots[1], evidence)
        audit, _rows = audit_flexible_mesh_release_roots(
            binding=self.binding,
            cases=self.cases,
            runtime_roots=self.roots,
        )
        self.assertFalse(audit["clean"])
        self.assertEqual(audit["drifted_count"], 1)
        self.assertIn("different workload root", audit["drifted"][0]["error"])

    def test_unknown_trace_case_is_reported_as_drift(self) -> None:
        expected = self.cases[0]
        drifted = FlexibleMeshReleaseCase.create(
            family=expected.family,
            mesh=expected.mesh,
            trace_model_digest=hashlib.sha256(b"drifted-trace").hexdigest(),
            runtime_profile_version=expected.runtime_profile_version,
        )
        _write(self.roots[0], _evidence(drifted, self.binding))
        audit, _rows = audit_flexible_mesh_release_roots(
            binding=self.binding,
            cases=self.cases,
            runtime_roots=self.roots,
        )
        self.assertFalse(audit["clean"])
        self.assertEqual(audit["drifted_count"], 1)
        self.assertEqual(audit["valid_unique_cases"], 0)

    def test_root_binding_bytes_must_match(self) -> None:
        (self.roots[1] / "release_binding.json").write_text(
            canonical_json(self.binding) + "\n", encoding="utf-8",
        )
        audit, _rows = audit_flexible_mesh_release_roots(
            binding=self.binding,
            cases=self.cases,
            runtime_roots=self.roots,
        )
        self.assertFalse(audit["clean"])
        self.assertEqual(audit["invalid_count"], 1)
        self.assertIn("bytes/digest drifted", audit["invalid"][0]["error"])

    def test_execution_config_marker_and_repeat_corruption_fail_closed(self) -> None:
        base = _evidence(self.cases[0], self.binding)
        corruptions = (
            "hardware_config_sha256",
            "simulation_config_sha256",
            "mapping_config_sha256",
            "marker",
            "repeat",
        )
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt):
                case_root = self.roots[0] / base.case.id
                case_root.mkdir(parents=True, exist_ok=True)
                raw = json.loads(canonical_json(base))
                if corrupt.endswith("_sha256"):
                    del raw["executions"][0][corrupt]
                elif corrupt == "marker":
                    raw["executions"][0]["marker_digest"] = "0" * 64
                else:
                    raw["executions"][1]["makespan_cycles"] += 1
                (case_root / "case_evidence.json").write_text(
                    json.dumps(raw), encoding="utf-8",
                )
                audit, _rows = audit_flexible_mesh_release_roots(
                    binding=self.binding,
                    cases=self.cases,
                    runtime_roots=self.roots,
                )
                self.assertFalse(audit["clean"])
                self.assertEqual(audit["invalid_count"], 1)
                (case_root / "case_evidence.json").unlink()

    def test_cli_failure_writes_only_missing_audit(self) -> None:
        binding_path = self.root / "binding.json"
        binding_path.write_text(canonical_json(self.binding), encoding="utf-8")
        output = self.root / "output"
        code = run(argparse.Namespace(
            binding=binding_path,
            runtime_root=list(self.roots),
            output_dir=output,
        ))
        self.assertEqual(code, 2)
        self.assertTrue((output / "missing_audit.json").is_file())
        self.assertFalse((output / "completion.json").exists())
        self.assertFalse((output / "completion.toml").exists())
        self.assertFalse((output / "release_summary.json").exists())

    def test_representative_cli_derives_explicit_non_exhaustive_report(self) -> None:
        cases = select_flexible_mesh_representative_cases(
            generate_flexible_mesh_release_cases(
                trace_model_digests=release_trace_model_digests(),
                runtime_profile_version=self.binding.runtime_profile_version,
            )
        )
        root_by_family = {
            family: root
            for families, root in zip(
                (
                    (FlexibleMeshReleaseFamily.DENSE_TRAIN,),
                    (
                        FlexibleMeshReleaseFamily.MOE_INFERENCE,
                        FlexibleMeshReleaseFamily.MOE_TRAIN,
                    ),
                    (
                        FlexibleMeshReleaseFamily.MESHSLICE_AG,
                        FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,
                        FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,
                    ),
                ),
                self.roots,
            )
            for family in families
        }
        for case in cases:
            _write(root_by_family[case.family], _evidence(case, self.binding))
        binding_path = self.root / "representative-binding.json"
        binding_path.write_text(canonical_json(self.binding), encoding="utf-8")
        output = self.root / "representative-output"
        code = run(argparse.Namespace(
            binding=binding_path,
            runtime_root=list(self.roots),
            output_dir=output,
            validation_scope="representative",
        ))
        self.assertEqual(code, 0)
        completion = json.loads((output / "completion.json").read_text())
        summary = json.loads((output / "release_summary.json").read_text())
        state = (output / "completion.toml").read_text()
        self.assertEqual(completion["validation_scope"], "representative")
        self.assertFalse(completion["exhaustive_runtime"])
        self.assertEqual(
            tuple((mesh["rows"], mesh["columns"]) for mesh in completion["tested_meshes"]),
            FLEXIBLE_MESH_REPRESENTATIVE_SHAPES,
        )
        self.assertEqual(summary["case_count"], 48)
        self.assertEqual(summary["execution_count"], 96)
        self.assertEqual(summary["validation_scope"], "representative")
        self.assertFalse(summary["exhaustive_runtime"])
        self.assertFalse(summary["exhaustive_rect_runtime_complete"])
        self.assertTrue(summary["flexible_mesh_workloads_complete"])
        self.assertIn('validation_scope = "representative"', state)
        self.assertIn("exhaustive_runtime = false", state)
        self.assertIn("exhaustive_rect_runtime_complete = false", state)

    def test_summary_exact_runtime_capacity_and_zero_failure_accounting(self) -> None:
        one_case_per_family = tuple(
            next(case for case in self.cases if case.family is family)
            for family in FlexibleMeshReleaseFamily
        )
        one_row_per_family = tuple(
            _evidence(case, self.binding) for case in one_case_per_family
        )
        rows = tuple(
            row for row in one_row_per_family for _index in range(100)
        )
        completion = SimpleNamespace(
            id="test-completion",
            evidence_matrix=SimpleNamespace(
                case_evidence=rows,
                release_binding=self.binding,
            ),
        )
        summary = _summary(completion)
        self.assertEqual(summary["case_count"], 600)
        self.assertEqual(summary["execution_count"], 1200)
        self.assertEqual(summary["runtime_verified_cases"], 600)
        self.assertEqual(summary["repeatability_verified_cases"], 600)
        self.assertEqual(summary["failed_case_ids"], [])
        self.assertEqual(summary["failure_count"], 0)
        self.assertEqual(summary["stage_failure_count"], 0)
        self.assertEqual(summary["residual_nonzero_execution_count"], 0)
        self.assertEqual(
            summary["runtime_verified_cases_by_family"],
            {family.value: 100 for family in FlexibleMeshReleaseFamily},
        )
        self.assertEqual(
            summary["repeatability_verified_cases_by_family"],
            {family.value: 100 for family in FlexibleMeshReleaseFamily},
        )
        self.assertEqual(
            set(summary["capacity_limits"]), set(summary["capacity_max"]),
        )
        self.assertEqual(summary["capacity_limits"]["symbolic_record_count"], 1_048_576)
        self.assertEqual(
            summary["capacity_policy_version"],
            "wafer_frontend.flexible_mesh_release_capacity/v2",
        )
        self.assertEqual(
            summary["capacity_limits"]["linked_manifest_file_bytes"],
            256 * 1024 * 1024,
        )
        self.assertEqual(
            summary["capacity_headroom"]["symbolic_record_count"],
            1_048_574,
        )
        self.assertEqual(
            set(summary["residual_max"]),
            {
                "active_endpoints", "active_sessions", "outstanding_tags",
                "incomplete_barriers", "pending_state_writes", "proto_wait_count",
                "lsu_residual", "dte_residual", "router_residual", "credit_residual",
            },
        )
        self.assertEqual(set(summary["residual_max"].values()), {0})
        self.assertEqual(set(summary["residual_nonzero_counts"].values()), {0})
        first_execution = one_row_per_family[0].executions[0]
        self.assertEqual(
            summary["program_io_phase_counts"],
            {phase.value: 1200 for phase in first_execution.program_io_phases},
        )
        self.assertEqual(
            summary["runtime_stage_counts"],
            {
                stage.value: {"success": 1200, "failure": 0}
                for stage, _code in first_execution.stage_exit_codes
            },
        )
        expected_markers = {}
        for row in one_row_per_family:
            for marker in row.executions[0].completion_markers:
                expected_markers[marker.value] = (
                    expected_markers.get(marker.value, 0) + 200
                )
        self.assertEqual(
            summary["completion_marker_execution_counts"],
            dict(sorted(expected_markers.items())),
        )
        self.assertTrue(summary["timing_execution_only"])


if __name__ == "__main__":
    unittest.main()
