#!/usr/bin/env python3
"""Freeze reviewed Stage4 PD timing reports without persisting NPUP bytes."""

from __future__ import annotations

import argparse
import ctypes
import errno
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from llm.frontend.wafer_frontend.passes import (  # noqa: E402
    build_stage4_capability_manifest,
    build_stage4_pd_case_matrix,
    build_stage4_pd_oracle,
)
from llm.frontend.wafer_frontend.schema import (  # noqa: E402
    CapabilityManifest,
    CapabilityStage,
    CaseMatrix,
    ExperimentSpec,
    ProgramIoContract,
    Stage4PdOracle,
    Stage4PdPlan,
    Stage4KvReshardKind,
    Stage4PdCaseMatrix,
    Stage4PdMode,
    Stage4PdRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (  # noqa: E402
    LinkedProgramManifest,
)
from llm.frontend.wafer_frontend.schema.common import (  # noqa: E402
    stable_artifact_id,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
    load_json_dataclass,
)
from llm.frontend.wafer_frontend.schema.stage3_static_profile_evidence import (  # noqa: E402
    STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
)
from llm.frontend.wafer_frontend.schema.stage4_pd_evidence import (  # noqa: E402
    STAGE4_PD_BASELINE_EPOCH,
)
from stage4_pd_cases import (  # noqa: E402
    Stage4PdCase,
    Stage4PdCaseKind,
    build_stage4_pd_case,
)
import run_stage4_pd_runtime as runtime  # noqa: E402


_BASELINE_EPOCH = STAGE4_PD_BASELINE_EPOCH
_PRIOR_EPOCH = STAGE3_STATIC_PROFILE_BASELINE_EPOCH
_PRODUCER = "freeze_stage4_pd_baseline"
_SUMMARY_VERSION = "wafer_frontend.stage4_checked_rebuild_summary/v1alpha1"
_REVIEW_VERSION = "wafer_frontend.baseline_review/v1alpha5"
_INPUT_VERSION = "wafer_frontend.stage4_freezer_inputs/v1alpha1"
_CASE_ORDER = ("fused", "pds", "pdr")
_CASE_KEYS = {
    "fused": (Stage4PdMode.FUSED, Stage4KvReshardKind.NONE),
    "pds": (Stage4PdMode.SEPARATED, Stage4KvReshardKind.ONE_TO_ONE),
    "pdr": (Stage4PdMode.SEPARATED, Stage4KvReshardKind.GATHER),
}
_CASE_KINDS = {
    "fused": Stage4PdCaseKind.FUSED,
    "pds": Stage4PdCaseKind.PDS,
    "pdr": Stage4PdCaseKind.PDR,
}
_OFFICIAL_FILES = {
    "actual_sha_program_io.json",
    "finalization.json",
    "finalizer.0.log",
    "finalizer.1.log",
    "input_digests.json",
    "linked_manifest.json",
    "mapping.spec",
    "model_spec.json",
    "oracle.json",
    "plan.json",
    "resolved_hardware.json",
    "resolver.log",
    "runtime.0.log",
    "runtime.1.log",
    "runtime_report.json",
}
_STAGED_CASE_FILES = {
    *_OFFICIAL_FILES,
    "SUCCESS",
    "inputs/input_summary.json",
}
_STAGED_ROOT_FILES = {
    "baseline_review.json",
    "capability_manifest.json",
    "case_matrix.json",
    "checked_rebuild_summary.json",
    "stage4_pd_case_matrix.json",
    *(
        f"stage4/{kind}/{name}"
        for kind in _CASE_ORDER
        for name in _STAGED_CASE_FILES
    ),
}
_SOURCE_PATHS = (
    "llm/frontend/wafer_frontend/passes/capability_manifest.py",
    "llm/frontend/wafer_frontend/passes/stage4_pd_case_matrix.py",
    "llm/frontend/wafer_frontend/policies/registry.py",
    "llm/frontend/wafer_frontend/schema/policy.py",
    "llm/frontend/wafer_frontend/schema/s1_naive_evidence.py",
    "llm/frontend/wafer_frontend/schema/stage4_pd_case_matrix.py",
    "llm/frontend/wafer_frontend/schema/stage4_pd_evidence.py",
    "llm/test/frontend/integration/freeze_stage4_pd_baseline.py",
    "llm/test/frontend/integration/run_stage4_pd_runtime.py",
    "llm/test/frontend/integration/stage4_pd_cases.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_rows(root: Path) -> tuple[tuple[str, str, int], ...]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"evidence root must be a real directory: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"evidence contains symlink: {path}")
        if path.is_file():
            rows.append(
                (str(path.relative_to(root)), _sha256(path), path.stat().st_size)
            )
    return tuple(rows)


def _tree_digest(root: Path) -> str:
    return canonical_digest(_tree_rows(root))


def _reject_symlink_ancestors(path: Path) -> None:
    candidate = path.absolute()
    for ancestor in (candidate, *candidate.parents):
        if ancestor.is_symlink():
            raise RuntimeError(f"evidence path has symlink ancestor: {ancestor}")


def _write_new(path: Path, value: object) -> None:
    _reject_symlink_ancestors(path.parent)
    if path.suffix == ".npup":
        raise RuntimeError(f"refusing to persist ProgramArtifact bytes: {path}")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if type(value) is str else canonical_json(value) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _copy_new(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"official output must be a real file: {source}")
    if source.suffix == ".npup" or destination.suffix == ".npup":
        raise RuntimeError("refusing to persist ProgramArtifact bytes")
    _reject_symlink_ancestors(destination.parent)
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to overwrite evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)


def _relative_files(root: Path) -> set[str]:
    return {row[0] for row in _tree_rows(root)}


def _require_exact_files(root: Path, expected: set[str], label: str) -> None:
    observed = _relative_files(root)
    if observed != expected:
        raise RuntimeError(
            f"{label} file set changed: observed={sorted(observed)!r} "
            f"expected={sorted(expected)!r}"
        )


def _require_exact_directories(
    root: Path, expected: set[str], label: str
) -> None:
    observed = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if observed != expected:
        raise RuntimeError(
            f"{label} directory set changed: observed={sorted(observed)!r} "
            f"expected={sorted(expected)!r}"
        )


def _tool_digests(args: argparse.Namespace) -> dict[str, str]:
    return {
        "finalizer": _sha256(args.finalizer),
        "npusim": _sha256(args.npusim),
        "resolver": _sha256(args.resolver),
        "runner": _sha256(
            _ROOT / "llm/test/frontend/integration/run_stage4_pd_runtime.py"
        ),
    }


def _validate_report_provenance(
    report: Stage4PdRuntimeReport,
    kind: str,
    args: argparse.Namespace,
) -> None:
    report.validate(f"stage4.{kind}.runtime_report")
    if (report.mode, report.reshard) != _CASE_KEYS[kind]:
        raise RuntimeError(f"Stage4 {kind} report case key changed")
    if report.baseline_epoch != _BASELINE_EPOCH:
        raise RuntimeError(f"Stage4 {kind} report epoch changed")
    if {item.name: item.digest for item in report.tool_digests} != _tool_digests(
        args
    ):
        raise RuntimeError(f"Stage4 {kind} tool digests changed")
    inputs = {item.name: item.digest for item in report.input_digests}
    if inputs["simulation"] != _sha256(args.simulation):
        raise RuntimeError(f"Stage4 {kind} simulation digest changed")


@lru_cache(maxsize=3)
def _production_case(kind: str) -> tuple[Stage4PdCase, Stage4PdOracle]:
    case = build_stage4_pd_case(_CASE_KINDS[kind])
    oracle = build_stage4_pd_oracle(case.pd_plan)
    oracle.validate_against(case.pd_plan)
    runtime._validate_static_case(case, oracle)
    return case, oracle


@lru_cache(maxsize=6)
def _production_runtime_inputs(
    kind: str,
    artifact_sha256: str,
) -> tuple[ProgramIoContract, object]:
    case, _oracle = _production_case(kind)
    program_io = runtime._rebuild_sidecar(case, artifact_sha256)
    memory_expected = runtime._memory_expected(
        case, runtime._leaf_fragments(case)
    )
    return program_io, memory_expected


def _validate_current_case_evidence(
    root: Path,
    kind: str,
    report: Stage4PdRuntimeReport,
    args: argparse.Namespace,
) -> None:
    case, oracle = _production_case(kind)
    spec = load_json_dataclass(
        ExperimentSpec,
        root / "model_spec.json",
        path=f"stage4.{kind}.spec",
    )
    plan = load_json_dataclass(
        Stage4PdPlan,
        root / "plan.json",
        path=f"stage4.{kind}.plan",
    )
    persisted_oracle = load_json_dataclass(
        Stage4PdOracle,
        root / "oracle.json",
        path=f"stage4.{kind}.oracle",
    )
    manifest = load_json_dataclass(
        LinkedProgramManifest,
        root / "linked_manifest.json",
        path=f"stage4.{kind}.manifest",
    )
    program_io = load_json_dataclass(
        ProgramIoContract,
        root / "actual_sha_program_io.json",
        path=f"stage4.{kind}.program_io",
    )
    if (
        spec != case.spec
        or plan != case.pd_plan
        or persisted_oracle != oracle
        or manifest != case.manifest
    ):
        raise RuntimeError(
            f"Stage4 {kind} formal inputs differ from current production rebuild"
        )
    expected_program_io, memory_expected = _production_runtime_inputs(
        kind, report.artifact.program_artifact_sha256
    )
    if program_io != expected_program_io:
        raise RuntimeError(
            f"Stage4 {kind} ProgramIo differs from current production rebuild"
        )
    program_io.validate_against(case.manifest)
    report.validate_against(
        case.pd_plan,
        oracle,
        case.manifest,
        case.planning_context,
        case.scheduling_context,
    )
    if (
        _sha256(root / "resolved_hardware.json") != report.hardware_digest
        or _sha256(root / "mapping.spec") != report.mapping_digest
        or (root / "resolved_hardware.json").read_text(encoding="utf-8")
        != case.runtime_hardware_inputs.hardware_json
        or (root / "mapping.spec").read_text(encoding="utf-8")
        != case.runtime_hardware_inputs.mapping_text
    ):
        raise RuntimeError(f"Stage4 {kind} hardware/mapping evidence changed")
    finalization = json.loads(
        (root / "finalization.json").read_text(encoding="utf-8")
    )
    if (
        finalization.get("artifact_sha256")
        != report.artifact.program_artifact_sha256
        or finalization.get("artifact_bytes")
        != report.artifact.artifact_size_bytes
        or finalization.get("linked_manifest_id") != case.manifest.id
        or finalization.get("linked_manifest_digest")
        != canonical_digest(case.manifest)
    ):
        raise RuntimeError(f"Stage4 {kind} finalization evidence changed")
    expected_inputs = {
        "schema_version": runtime._REPORT_INPUT_SCHEMA_VERSION,
        "case_kind": _CASE_KINDS[kind].value,
        "report_id": report.id,
        "report_digest": canonical_digest(report),
        "tool_digests": report.tool_digests,
        "input_digests": report.input_digests,
    }
    observed_inputs = json.loads(
        (root / "input_digests.json").read_text(encoding="utf-8")
    )
    if canonical_json(observed_inputs) != canonical_json(expected_inputs):
        raise RuntimeError(f"Stage4 {kind} input digest evidence changed")
    resolver = (root / "resolver.log").read_text(encoding="utf-8")
    witness = (
        f"initializations={len(program_io.initializations)} "
        f"probes={len(program_io.output_probes)}"
    )
    if witness not in resolver:
        raise RuntimeError(f"Stage4 {kind} resolver evidence changed")

    for run_index in range(2):
        observation = runtime._observe_runtime(
            case,
            oracle,
            (root / f"runtime.{run_index}.log").read_text(encoding="utf-8"),
            report.artifact.program_artifact_sha256,
            program_io,
            memory_expected,
        )
        repeat = report.repeats[run_index]
        if (
            observation.makespan_cycles != report.makespan_cycles
            or observation.marker_digest != repeat.marker_digest
            or observation.memory != report.memory
            or observation.program_io != report.program_io
            or observation.control != report.control
            or observation.d2d != report.d2d
        ):
            raise RuntimeError(
                f"Stage4 {kind} runtime log {run_index} does not rebuild its report"
            )


def _load_official_case(
    root: Path,
    kind: str,
    args: argparse.Namespace,
) -> Stage4PdRuntimeReport:
    _require_exact_files(root, _OFFICIAL_FILES, f"Stage4 {kind} official")
    _require_exact_directories(root, set(), f"Stage4 {kind} official")
    report = load_json_dataclass(
        Stage4PdRuntimeReport,
        root / "runtime_report.json",
        path=f"stage4.{kind}.runtime_report",
    )
    _validate_report_provenance(report, kind, args)
    _validate_current_case_evidence(root, kind, report, args)
    return report


def _input_summary(
    root: Path,
    kind: str,
    report: Stage4PdRuntimeReport,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "schema_version": _INPUT_VERSION,
        "case_kind": kind,
        "report_id": report.id,
        "report_digest": canonical_digest(report),
        "artifact_sha256": report.artifact.program_artifact_sha256,
        "official_outputs": tuple(
            {
                "path": name,
                "sha256": _sha256(root / name),
                "size_bytes": (root / name).stat().st_size,
            }
            for name in sorted(_OFFICIAL_FILES)
        ),
        "sources": tuple(
            {"path": name, "sha256": _sha256(_ROOT / name)}
            for name in _SOURCE_PATHS
        ),
        "tools": tuple(
            {"name": name, "sha256": digest}
            for name, digest in sorted(_tool_digests(args).items())
        ),
        "inputs": tuple(
            {"name": item.name, "sha256": item.digest}
            for item in report.input_digests
        ),
        "raw_evidence_status": "complete-current-runner-report-root",
    }


def _load_prior(root: Path) -> tuple[CaseMatrix, CapabilityManifest]:
    matrix = load_json_dataclass(
        CaseMatrix, root / "case_matrix.json", path="stage3.case_matrix"
    )
    manifest = load_json_dataclass(
        CapabilityManifest,
        root / "capability_manifest.json",
        path="stage3.capability_manifest",
    )
    manifest.validate_against(matrix)
    if manifest.baseline_epoch != _PRIOR_EPOCH:
        raise RuntimeError("prior Stage3 capability epoch changed")
    return matrix, manifest


def _build_from_reports(
    prior_root: Path,
    reports: tuple[Stage4PdRuntimeReport, ...],
) -> tuple[Stage4PdCaseMatrix, CaseMatrix, CapabilityManifest]:
    stage4_matrix = build_stage4_pd_case_matrix(reports)
    if not stage4_matrix.stage4_ready:
        raise RuntimeError("Stage4 freeze requires exact PD-F/PDS/PDR reports")
    prior_matrix, prior_manifest = _load_prior(prior_root)
    matrix, manifest = build_stage4_capability_manifest(
        prior_matrix, prior_manifest, stage4_matrix
    )
    if (
        manifest.coverage_score(CapabilityStage.S1) != Fraction(11, 2)
        or manifest.acceptance_score(CapabilityStage.S1) != (1, 3)
        or manifest.coverage_score(CapabilityStage.S2) != Fraction(0, 1)
        or manifest.coverage_score(CapabilityStage.S3) != Fraction(0, 1)
    ):
        raise RuntimeError("Stage4 capability score overclaim")
    return stage4_matrix, matrix, manifest


def _review(
    prior_root: Path,
    prior_digest: str,
    stage4_matrix: Stage4PdCaseMatrix,
) -> dict[str, object]:
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_baseline_epoch": _PRIOR_EPOCH,
        "prior_baseline_path": str(prior_root.relative_to(_ROOT)),
        "prior_baseline_tree_digest": prior_digest,
        "stage4_case_matrix_id": stage4_matrix.id,
        "stage4_case_matrix_digest": canonical_digest(stage4_matrix),
        "review_kind": "stage4-pd-timing-baseline",
        "review_decision": "approved-stage4-pd-timing-baseline",
        "reviewer": "codex-stage4-pd-development",
        "reason": (
            "PD-F, PDS and PDR production evidence, raw-log rebuild, "
            "anti-forgery negatives and no-NPUP publication are complete."
        ),
        "cases": tuple(
            {
                "case_id": report.case_id,
                "mode": report.mode,
                "reshard": report.reshard,
                "report_id": report.id,
                "report_digest": canonical_digest(report),
                "artifact_sha256": report.artifact.program_artifact_sha256,
            }
            for report in stage4_matrix.reports
        ),
        "caveats": (
            "The complete current runner report-root text set is persisted.",
            "Evidence proves timing, state transport and ProgramIo boundaries, "
            "not model-functional correctness.",
            "S1 score is 5.5/6 and acceptance is 1/3; S1-N is not complete.",
            "No ProgramArtifact .npup bytes are persisted.",
        ),
    }
    return {
        "schema_version": _REVIEW_VERSION,
        "producer_pass": _PRODUCER,
        "id": stable_artifact_id(
            "baseline_review",
            semantic_key,
            schema_version=_REVIEW_VERSION,
        ),
        **semantic_key,
    }


def _summary(staging: Path, prior_digest: str) -> dict[str, object]:
    files = tuple(
        {"path": name, "sha256": digest, "size_bytes": size}
        for name, digest, size in _tree_rows(staging)
    )
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_tree_digest": prior_digest,
        "files": files,
        "payload_tree_digest": canonical_digest(files),
    }
    return {
        "schema_version": _SUMMARY_VERSION,
        "producer_pass": _PRODUCER,
        "id": stable_artifact_id(
            "stage4_checked_rebuild_summary",
            semantic_key,
            schema_version=_SUMMARY_VERSION,
        ),
        **semantic_key,
    }


def _validate_summary(staging: Path, prior_digest: str) -> None:
    observed = json.loads(
        (staging / "checked_rebuild_summary.json").read_text(encoding="utf-8")
    )
    files = tuple(
        {"path": name, "sha256": digest, "size_bytes": size}
        for name, digest, size in _tree_rows(staging)
        if name != "checked_rebuild_summary.json"
    )
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_tree_digest": prior_digest,
        "files": files,
        "payload_tree_digest": canonical_digest(files),
    }
    expected = {
        "schema_version": _SUMMARY_VERSION,
        "producer_pass": _PRODUCER,
        "id": stable_artifact_id(
            "stage4_checked_rebuild_summary",
            semantic_key,
            schema_version=_SUMMARY_VERSION,
        ),
        **semantic_key,
    }
    if canonical_json(observed) != canonical_json(expected):
        raise RuntimeError("Stage4 checked rebuild summary changed")


def _stage_from_official(
    staging: Path,
    args: argparse.Namespace,
    prior_digest: str,
) -> None:
    _require_exact_directories(
        args.official_root, set(_CASE_ORDER), "Stage4 official root"
    )
    if _relative_files(args.official_root) != {
        f"{kind}/{name}" for kind in _CASE_ORDER for name in _OFFICIAL_FILES
    }:
        raise RuntimeError("Stage4 official root file matrix changed")
    reports = []
    for kind in _CASE_ORDER:
        source = args.official_root / kind
        report = _load_official_case(source, kind, args)
        destination = staging / "stage4" / kind
        for name in sorted(_OFFICIAL_FILES):
            _copy_new(source / name, destination / name)
        _write_new(destination / "SUCCESS", report.id + "\n")
        _write_new(
            destination / "inputs/input_summary.json",
            _input_summary(destination, kind, report, args),
        )
        reports.append(report)
    stage4_matrix, matrix, manifest = _build_from_reports(
        args.prior_root, tuple(reports)
    )
    _write_new(staging / "stage4_pd_case_matrix.json", stage4_matrix)
    _write_new(staging / "case_matrix.json", matrix)
    _write_new(staging / "capability_manifest.json", manifest)
    _write_new(
        staging / "baseline_review.json",
        _review(args.prior_root, prior_digest, stage4_matrix),
    )
    _write_new(
        staging / "checked_rebuild_summary.json",
        _summary(staging, prior_digest),
    )


def _validate_staging(
    staging: Path,
    args: argparse.Namespace,
    prior_digest: str,
) -> None:
    if any(path.suffix == ".npup" for path in staging.rglob("*")):
        raise RuntimeError("Stage4 checked baseline must not persist NPUP")
    _require_exact_directories(
        staging,
        {
            "stage4",
            *(f"stage4/{kind}" for kind in _CASE_ORDER),
            *(f"stage4/{kind}/inputs" for kind in _CASE_ORDER),
        },
        "Stage4 staging",
    )
    _require_exact_files(staging, _STAGED_ROOT_FILES, "Stage4 staging")
    reports = []
    for kind in _CASE_ORDER:
        root = staging / "stage4" / kind
        _require_exact_files(root, _STAGED_CASE_FILES, f"Stage4 {kind} staged")
        _require_exact_directories(root, {"inputs"}, f"Stage4 {kind} staged")
        report = load_json_dataclass(
            Stage4PdRuntimeReport,
            root / "runtime_report.json",
            path=f"stage4.{kind}.runtime_report",
        )
        _validate_report_provenance(report, kind, args)
        _validate_current_case_evidence(root, kind, report, args)
        if (root / "SUCCESS").read_text(encoding="utf-8").strip() != report.id:
            raise RuntimeError(f"Stage4 {kind} SUCCESS changed")
        expected_inputs = _input_summary(root, kind, report, args)
        observed_inputs = json.loads(
            (root / "inputs/input_summary.json").read_text(encoding="utf-8")
        )
        if canonical_json(observed_inputs) != canonical_json(expected_inputs):
            raise RuntimeError(f"Stage4 {kind} input summary changed")
        reports.append(report)
    stage4_matrix, matrix, manifest = _build_from_reports(
        args.prior_root, tuple(reports)
    )
    expected_files = {
        "stage4_pd_case_matrix.json": stage4_matrix,
        "case_matrix.json": matrix,
        "capability_manifest.json": manifest,
        "baseline_review.json": _review(
            args.prior_root, prior_digest, stage4_matrix
        ),
    }
    for name, expected in expected_files.items():
        wanted = (canonical_json(expected) + "\n").encode("utf-8")
        if (staging / name).read_bytes() != wanted:
            raise RuntimeError(f"{name} differs from disk-derived rebuild")
    _validate_summary(staging, prior_digest)


def _validate_publish_target(path: Path) -> None:
    _reject_symlink_ancestors(path)
    if path.name != _BASELINE_EPOCH:
        raise RuntimeError(f"baseline root must end in {_BASELINE_EPOCH}")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite evidence: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(f"baseline parent must be a real directory: {path.parent}")


def _atomic_publish_noreplace(staging: Path, target: Path) -> None:
    _reject_symlink_ancestors(staging)
    _validate_publish_target(target)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-clobber publish requires Linux renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(staging), -100, os.fsencode(target), 1) != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise RuntimeError(f"refusing to overwrite raced target: {target}")
        raise OSError(number, f"atomic publish failed: {staging} -> {target}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--npusim", required=True, type=Path)
    parser.add_argument("--finalizer", required=True, type=Path)
    parser.add_argument("--resolver", required=True, type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    args = parser.parse_args()
    for name in (
        "prior_root",
        "official_root",
        "npusim",
        "finalizer",
        "resolver",
        "simulation",
    ):
        supplied = getattr(args, name)
        if supplied.is_symlink():
            parser.error(f"--{name.replace('_', '-')} must be a real path")
        resolved = supplied.resolve()
        if not resolved.exists():
            parser.error(f"--{name.replace('_', '-')} must exist: {resolved}")
        setattr(args, name, resolved)
    args.baseline_root = args.baseline_root.absolute()
    _validate_publish_target(args.baseline_root)
    if args.prior_root.name != _PRIOR_EPOCH or not args.prior_root.is_dir():
        parser.error(f"--prior-root must identify {_PRIOR_EPOCH}")
    if not args.official_root.is_dir():
        parser.error("--official-root must be a directory")
    return args


def main() -> int:
    args = _parse_args()
    prior_digest = _tree_digest(args.prior_root)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{_BASELINE_EPOCH}.staging-",
            dir=args.baseline_root.parent,
        )
    )
    published = False
    try:
        _stage_from_official(staging, args, prior_digest)
        _validate_staging(staging, args, prior_digest)
        if _tree_digest(args.prior_root) != prior_digest:
            raise RuntimeError("prior Stage3 baseline changed during freeze")
        _atomic_publish_noreplace(staging, args.baseline_root)
        published = True
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
    print(
        "[STAGE4 PD FREEZE] PASS: cases=3 score=5.5/6 "
        "acceptance=1/3 model_functional=0 raw_persistence=complete"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
