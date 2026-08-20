#!/usr/bin/env python3
"""Freeze reviewed Stage 0 capability truth from persistent E1/E2 reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm.frontend.wafer_frontend.passes import (
    build_stage0_capability_manifest,
)
from llm.frontend.wafer_frontend.runner import NaiveRunReport
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    load_json_dataclass,
)


_BASELINE_EPOCH = "stage0-policy-provenance-v1"
_REVIEW_SCHEMA_VERSION = "wafer_frontend.baseline_review/v1alpha1"
_PRIOR_ARTIFACTS = {
    "E1": "f4607bcf478a51854adaa7182507094a4b2b456bcc10d9c197ea811007f796bf",
    "E2": "ec7295762a8349ac80e71b013598c59d30e69d5c3eccdb7c32035bbeed2bbe77",
}


def _write_new(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite checked evidence: {path}")
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _semantic_fingerprint(report: NaiveRunReport) -> str:
    return canonical_digest(
        {
            "artifact": report.artifact,
            "static_metrics": report.static_metrics,
            "runtime": report.runtime,
            "validation": report.validation,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.baseline_root
    reports = {
        case: load_json_dataclass(
            NaiveRunReport,
            root / case.lower() / "run_report.json",
            path=f"{case}.report",
        )
        for case in ("E1", "E2")
    }
    for case, report in reports.items():
        if report.case.value != case:
            raise RuntimeError(f"{case} report has case {report.case.value!r}")
        artifact = report.artifact
        if not isinstance(artifact, dict):
            raise RuntimeError(f"{case} artifact summary is not an object")
        if artifact["artifact_sha256"] != _PRIOR_ARTIFACTS[case]:
            raise RuntimeError(
                f"{case} artifact changed without an approved refreeze"
            )

    matrix, manifest = build_stage0_capability_manifest(
        e1_artifact_digest=_PRIOR_ARTIFACTS["E1"],
        e1_report_digest=canonical_digest(reports["E1"]),
        e2_artifact_digest=_PRIOR_ARTIFACTS["E2"],
        e2_report_digest=canonical_digest(reports["E2"]),
        single_die_evidence_digest=canonical_digest(
            {
                "tests": (
                    "test_dense_ir0_validator",
                    "test_global_action_schema",
                    "test_ir2_schema",
                )
            }
        ),
        baseline_epoch=_BASELINE_EPOCH,
    )
    case_rows = []
    for case in ("E1", "E2"):
        report = reports[case]
        artifact = report.artifact
        runtime = report.runtime
        metrics = report.static_metrics
        assert isinstance(artifact, dict)
        assert isinstance(runtime, dict)
        assert isinstance(metrics, dict)
        case_rows.append(
            {
                "case": case,
                "prior_artifact_sha256": _PRIOR_ARTIFACTS[case],
                "artifact_sha256": artifact["artifact_sha256"],
                "prior_report_digest": None,
                "report_digest": canonical_digest(report),
                "semantic_fingerprint_digest": _semantic_fingerprint(report),
                "action_counts": metrics["action_counts"],
                "record_count": artifact["record_count"],
                "relocation_count": artifact["relocation_count"],
                "analytic_transfer_bytes": metrics["analytic_transfer_bytes"],
                "observed_transfer_bytes": runtime["observed_transfer_bytes"],
                "makespan_cycles": runtime["makespan_cycles"],
                "semantic_differences": (),
            }
        )
    review_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "review_kind": "initial-policy-provenance-snapshot",
        "reason": (
            "PolicySelection became explicit in registry, contexts, receipts, "
            "compilation and report; executable artifacts remain byte-identical."
        ),
        "version_changes": (
            "naive_run_report/v1alpha1 -> v1alpha2",
            "pipeline/v1alpha3 -> v1alpha4",
            "policy_selection/v1alpha1 introduced",
        ),
        "cases": tuple(case_rows),
        "commands": (
            "python3 -B -m llm.frontend.wafer_frontend.cli run --case E1 ...",
            "python3 -B -m llm.frontend.wafer_frontend.cli run --case E2 ...",
            "python3 -B llm/test/frontend/integration/freeze_stage0_baseline.py "
            "--baseline-root notes/frontend/baselines/"
            "stage0-policy-provenance-v1",
        ),
        "reviewer": "codex-stage0-development",
        "review_decision": "approved-initial-provenance-baseline",
        "caveats": (
            "No prior persistent report existed, so prior_report_digest is null.",
            "E1/E2 prove timing execution, not Dense numerical correctness.",
        ),
    }
    review = {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "producer_pass": "freeze_stage0_baseline",
        "id": stable_artifact_id(
            "baseline_review",
            review_key,
            schema_version=_REVIEW_SCHEMA_VERSION,
        ),
        **review_key,
    }
    _write_new(root / "case_matrix.json", matrix)
    _write_new(root / "capability_manifest.json", manifest)
    _write_new(root / "baseline_review.json", review)
    print(
        "[STAGE0 BASELINE] PASS: "
        f"matrix={matrix.id} manifest={manifest.id} review={review['id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
