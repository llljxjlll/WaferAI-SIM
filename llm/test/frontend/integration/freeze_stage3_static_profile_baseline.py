#!/usr/bin/env python3
"""Freeze the reviewed Stage3 static-profile timing evidence epoch."""

from __future__ import annotations

import argparse
import ctypes
import errno
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from llm.frontend.wafer_frontend.passes import (  # noqa: E402
    build_stage3_capability_manifest,
)
from llm.frontend.wafer_frontend.schema import (  # noqa: E402
    CapabilityManifest,
    CapabilityStage,
    CaseMatrix,
    LinkedProgramManifest,
    ProgramIoContract,
    Stage3DenseInferenceOracle,
    Stage3StaticProfileRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.common import (  # noqa: E402
    stable_artifact_id,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
    load_json_dataclass,
)
from llm.frontend.wafer_frontend.schema.stage3_profile import (  # noqa: E402
    Stage3ProfileMode,
)
from llm.frontend.wafer_frontend.schema.stage3_static_profile_evidence import (  # noqa: E402
    STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
)
from run_stage3_static_profile_negative_evidence import (  # noqa: E402
    _validate_evidence as _validate_negative_evidence,
)
from stage3_decode_cases import (  # noqa: E402
    Stage3StaticCaseKind,
    build_stage3_static_case,
)


_BASELINE_EPOCH = STAGE3_STATIC_PROFILE_BASELINE_EPOCH
_PRIOR_EPOCH = "stage2-dense-forward-v1"
_SUMMARY_SCHEMA_VERSION = (
    "wafer_frontend.stage3_checked_rebuild_summary/v1alpha1"
)
_REVIEW_SCHEMA_VERSION = "wafer_frontend.baseline_review/v1alpha4"
_PRODUCER = "freeze_stage3_static_profile_baseline"
_MODES = (
    Stage3StaticCaseKind.PREFILL,
    Stage3StaticCaseKind.DECODE,
    Stage3StaticCaseKind.MIXED,
)
_RAW_SUFFIXES = (
    "finalization.json",
    "finalizer.0.log",
    "finalizer.1.log",
    "input_digests.json",
    "linked_manifest.json",
    "markers.json",
    "oracle.json",
    "program_io.json",
    "resolver.log",
    "runtime.0.log",
    "runtime.1.log",
    "runtime.json",
)
_CASE_FILES = {
    "SUCCESS",
    "inputs/input_summary.json",
    "runner.stderr.log",
    "runner.stdout.log",
    *{f"raw/stage3-{{mode}}.{suffix}" for suffix in _RAW_SUFFIXES},
}
_NEGATIVE_FILES = {
    "SUCCESS",
    "negative_evidence.json",
    "runner.stderr.log",
    "runner.stdout.log",
}
_REGRESSION_CASES = (
    {
        "case": "P1",
        "old_artifact_sha256": (
            "3f6bccb95bfc1f0d63fe720dac8bf98f67b7522d803e79410ca279ea536c8363"
        ),
        "new_artifact_sha256": (
            "944657529885a9e4a34ade89aeba7bac0ffbbc265456df71da482d54931f9dfe"
        ),
        "artifact_size_bytes": 1694,
        "record_count": 9,
        "relocation_count": 16,
        "makespan_cycles": 164,
    },
    {
        "case": "K1",
        "old_artifact_sha256": (
            "42ae61bb31936950d74860172ece3c6602d7620fc948109e0a4f6f3c48d31a14"
        ),
        "new_artifact_sha256": (
            "d83e0b498882fd0bdb93d6df56a92c2900620e792c542331c369d34006dc3911"
        ),
        "artifact_size_bytes": 21230,
        "record_count": 156,
        "relocation_count": 287,
        "makespan_cycles": 1633,
    },
    {
        "case": "PD1",
        "old_artifact_sha256": (
            "a1adac8f41d5f10d7db5f0660ecff88f97b75029e6437b3d0de52fa0a2dd58e2"
        ),
        "new_artifact_sha256": (
            "eb14f21bcb9a20d47e0a54904587007253ef0eceee7d42a4e374b124c852c52f"
        ),
        "artifact_size_bytes": 4448,
        "record_count": 30,
        "relocation_count": 44,
        "makespan_cycles": 473,
    },
    {
        "case": "STAGE2_TP1",
        "old_artifact_sha256": (
            "b7a8548c7b938da31a59dd8d457dadd85965999626f64e2f1fbe2502be152efa"
        ),
        "new_artifact_sha256": (
            "308152e15697e45329660c160a3428a809015e408079019693e8d7c7d4d3fbad"
        ),
        "artifact_size_bytes": 22258,
        "record_count": 159,
        "relocation_count": 297,
        "makespan_cycles": 5805,
    },
    {
        "case": "STAGE2_TP2",
        "old_artifact_sha256": (
            "d86f3622a7cf2ad2000b885bde8b0d905fb978bd76a8385d781a2b040329fd05"
        ),
        "new_artifact_sha256": (
            "1824ceeea8a23150489df9a0e41f4571ac90edbcf86ff82da0feb786888a830c"
        ),
        "artifact_size_bytes": 65832,
        "record_count": 510,
        "relocation_count": 842,
        "makespan_cycles": 6628,
    },
    {
        "case": "STAGE2_TP4",
        "old_artifact_sha256": (
            "343ba30a2c10af3bead6d08f855527ec2d0f305737e13a1987ccef566114cead"
        ),
        "new_artifact_sha256": (
            "fad49b3356701b6e5e65139f7e91104b0693c020710536b97af84bd177fae187"
        ),
        "artifact_size_bytes": 179316,
        "record_count": 1452,
        "relocation_count": 2260,
        "makespan_cycles": 7503,
    },
    {
        "case": "E1",
        "old_artifact_sha256": (
            "95bb582bc852b35bc701ed41562664173fda95268e736363990ee0061bff6ef5"
        ),
        "new_artifact_sha256": (
            "17f40263ea05b2591935a08361abccd3ea2680ce9872372f8dfdd1a1ff2c3c8a"
        ),
        "artifact_size_bytes": 36392,
        "record_count": 278,
        "relocation_count": 464,
        "makespan_cycles": 8958,
    },
    {
        "case": "E2",
        "old_artifact_sha256": (
            "fe6ed061db11e54c182e64ca97c5a1741a5006df2d16e66e9d542ae13661edfe"
        ),
        "new_artifact_sha256": (
            "5476081c966f4d63ea68166d84090f96bce230b6330085b4b91d6aa65e01a5c3"
        ),
        "artifact_size_bytes": 179280,
        "record_count": 1452,
        "relocation_count": 2260,
        "makespan_cycles": 12952,
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tree_rows(root: Path) -> tuple[tuple[str, str, int], ...]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"checked evidence root must be real: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"checked evidence contains symlink: {path}")
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
            raise RuntimeError(
                f"checked evidence path has symlink ancestor: {ancestor}"
            )


def _write_new(path: Path, value: object) -> None:
    _reject_symlink_ancestors(path.parent)
    if path.suffix == ".npup":
        raise RuntimeError(f"refusing to persist ProgramArtifact bytes: {path}")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite checked evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if type(value) is str else canonical_json(value) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


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
        if path.is_dir()
    }
    if observed != expected:
        raise RuntimeError(
            f"{label} directory set changed: observed={sorted(observed)!r} "
            f"expected={sorted(expected)!r}"
        )


def _run(
    command: list[str], *, cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_ROOT)
    if previous:
        env["PYTHONPATH"] += os.pathsep + previous
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def _input_summary(
    kind: Stage3StaticCaseKind, args: argparse.Namespace
) -> dict[str, object]:
    case = build_stage3_static_case(kind, 1)
    sources = (
        "llm/frontend/wafer_frontend/passes/capability_manifest.py",
        "llm/frontend/wafer_frontend/schema/stage3_static_profile_evidence.py",
        "llm/test/frontend/integration/run_stage3_static_profiles.py",
        "llm/test/frontend/integration/stage3_decode_cases.py",
        "llm/test/frontend/integration/freeze_stage3_static_profile_baseline.py",
    )
    return {
        "schema_version": "wafer_frontend.stage3_freezer_inputs/v1alpha1",
        "profile_mode": kind.value,
        "tools": {
            "finalizer": _sha256(args.finalizer),
            "resolver": _sha256(args.resolver),
            "npusim": _sha256(args.npusim),
            "python": _sha256(Path(sys.executable)),
        },
        "sources": tuple(
            {"path": path, "sha256": _sha256(_ROOT / path)}
            for path in sources
        ),
        "simulation": {
            "path": str(args.simulation.relative_to(_ROOT)),
            "sha256": _sha256(args.simulation),
        },
        "hardware": {
            "source_path": str(
                case.runtime_hardware_inputs.source_hardware_path.relative_to(
                    _ROOT
                )
            ),
            "generated_sha256": _sha256_text(
                case.runtime_hardware_inputs.hardware_json
            ),
        },
        "mapping": {
            "source_path": str(
                case.runtime_hardware_inputs.source_mapping_path.relative_to(
                    _ROOT
                )
            ),
            "generated_sha256": _sha256_text(
                case.runtime_hardware_inputs.mapping_text
            ),
        },
    }


def _raw_names(kind: Stage3StaticCaseKind) -> set[str]:
    stem = f"stage3-{kind.value}"
    return {f"{stem}.{suffix}" for suffix in _RAW_SUFFIXES}


def _case_files(kind: Stage3StaticCaseKind) -> set[str]:
    return {
        name.format(mode=kind.value) if "{mode}" in name else name
        for name in _CASE_FILES
    }


def _validate_markers(
    path: Path, report: Stage3StaticProfileRuntimeReport
) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != report.marker_schema_version:
        raise RuntimeError("Stage3 marker schema changed")
    runs = raw.get("runs")
    if type(runs) is not list or len(runs) != 2:
        raise RuntimeError("Stage3 markers require two runs")
    for index, (run, repeat) in enumerate(zip(runs, report.repeats)):
        if (
            run.get("run_index") != index
            or run.get("makespan_cycles") != repeat.makespan_cycles
            or run.get("marker_digest") != repeat.marker_digest
            or canonical_digest(run.get("memory")) != repeat.memory_digest
            or canonical_digest(run.get("probes")) != repeat.probe_digest
            or canonical_digest(run.get("control")) != repeat.control_digest
            or canonical_digest(run.get("d2d")) != repeat.d2d_digest
        ):
            raise RuntimeError(f"Stage3 marker run {index} changed")


def _validate_case(
    root: Path,
    kind: Stage3StaticCaseKind,
    args: argparse.Namespace,
) -> tuple[Stage3DenseInferenceOracle, Stage3StaticProfileRuntimeReport]:
    _require_exact_files(root, _case_files(kind), f"Stage3 {kind.value}")
    _require_exact_directories(
        root, {"inputs", "raw"}, f"Stage3 {kind.value}"
    )
    raw_root = root / "raw"
    _require_exact_files(raw_root, _raw_names(kind), "Stage3 raw evidence")
    stem = f"stage3-{kind.value}"
    oracle = load_json_dataclass(
        Stage3DenseInferenceOracle,
        raw_root / f"{stem}.oracle.json",
        path=f"stage3.{kind.value}.oracle",
    )
    report = load_json_dataclass(
        Stage3StaticProfileRuntimeReport,
        raw_root / f"{stem}.runtime.json",
        path=f"stage3.{kind.value}.runtime",
    )
    report.validate_against(oracle)
    expected_mode = Stage3ProfileMode(kind.value)
    if report.profile_mode is not expected_mode:
        raise RuntimeError(f"Stage3 {kind.value} report mode changed")

    manifest = load_json_dataclass(
        LinkedProgramManifest,
        raw_root / f"{stem}.linked_manifest.json",
        path=f"stage3.{kind.value}.manifest",
    )
    sidecar = load_json_dataclass(
        ProgramIoContract,
        raw_root / f"{stem}.program_io.json",
        path=f"stage3.{kind.value}.program_io",
    )
    sidecar.validate_against(manifest)
    manifest_digest = canonical_digest(manifest)
    if (
        report.artifact.linked_manifest_id != manifest.id
        or report.artifact.linked_manifest_digest != manifest_digest
        or sidecar.source_linked_manifest_id != manifest.id
        or sidecar.source_linked_manifest_digest != manifest_digest
        or sidecar.program_artifact_sha256
        != report.artifact.program_artifact_sha256
        or sidecar.id != report.sidecar.contract_id
        or canonical_digest(sidecar) != report.sidecar.contract_digest
    ):
        raise RuntimeError(f"Stage3 {kind.value} manifest/sidecar closure changed")

    finalization = json.loads(
        (raw_root / f"{stem}.finalization.json").read_text(encoding="utf-8")
    )
    if (
        finalization.get("artifact_sha256")
        != report.artifact.program_artifact_sha256
        or finalization.get("artifact_bytes")
        != report.artifact.artifact_size_bytes
        or finalization.get("record_count") != report.artifact.record_count
        or finalization.get("relocation_count")
        != report.artifact.relocation_count
        or finalization.get("linked_manifest_id") != manifest.id
        or finalization.get("linked_manifest_digest") != manifest_digest
    ):
        raise RuntimeError(f"Stage3 {kind.value} finalization closure changed")

    built = build_stage3_static_case(kind, 1)
    if (
        canonical_digest(built.oracle) != canonical_digest(oracle)
        or canonical_digest(built.manifest) != manifest_digest
        or report.compile.template_id != built.template.id
        or report.compile.template_digest != canonical_digest(built.template)
        or report.compile.ir1_id != built.graph.id
        or report.compile.ir1_digest != canonical_digest(built.graph)
        or report.compile.global_dag_id != built.global_dag.id
        or report.compile.global_dag_digest != canonical_digest(built.global_dag)
        or report.compile.lowered_id != built.lowered.id
        or report.compile.lowered_digest != canonical_digest(built.lowered)
    ):
        raise RuntimeError(f"Stage3 {kind.value} production rebuild changed")

    expected_tools = (
        _sha256(args.finalizer),
        _sha256(args.resolver),
        _sha256(args.npusim),
    )
    if (
        (
            report.tools.finalizer_sha256,
            report.tools.resolver_sha256,
            report.tools.npusim_sha256,
        )
        != expected_tools
        or report.simulation_digest != _sha256(args.simulation)
        or report.hardware_digest
        != _sha256_text(built.runtime_hardware_inputs.hardware_json)
        or report.mapping_digest
        != _sha256_text(built.runtime_hardware_inputs.mapping_text)
    ):
        raise RuntimeError(f"Stage3 {kind.value} tool/input provenance changed")

    expected_inputs = {
        "schema_version": "wafer_frontend.stage3_static_profile_inputs/v1",
        "tools": report.tools,
        "hardware_digest": report.hardware_digest,
        "simulation_digest": report.simulation_digest,
        "mapping_digest": report.mapping_digest,
        "static_profile_digest": report.static_profile_digest,
        "template_digest": report.compile.template_digest,
        "ir1_digest": report.compile.ir1_digest,
        "global_dag_digest": report.compile.global_dag_digest,
        "lowered_digest": report.compile.lowered_digest,
    }
    observed_inputs = json.loads(
        (raw_root / f"{stem}.input_digests.json").read_text(encoding="utf-8")
    )
    if canonical_json(observed_inputs) != canonical_json(expected_inputs):
        raise RuntimeError(f"Stage3 {kind.value} input digests changed")
    if json.loads(
        (root / "inputs/input_summary.json").read_text(encoding="utf-8")
    ) != json.loads(canonical_json(_input_summary(kind, args))):
        raise RuntimeError(f"Stage3 {kind.value} freezer inputs changed")

    _validate_markers(raw_root / f"{stem}.markers.json", report)
    for index in range(2):
        log = (raw_root / f"{stem}.runtime.{index}.log").read_text(
            encoding="utf-8"
        )
        if "[SIM_RESULT] " not in log or "[D2D_TYPE] " not in log:
            raise RuntimeError(f"Stage3 runtime log {index} lacks markers")
    if not (raw_root / f"{stem}.resolver.log").read_text(encoding="utf-8"):
        raise RuntimeError("Stage3 resolver log is empty")
    if (root / "SUCCESS").read_text(encoding="utf-8").strip() != report.id:
        raise RuntimeError(f"Stage3 {kind.value} SUCCESS changed")
    stdout = (root / "runner.stdout.log").read_text(encoding="utf-8")
    if (
        f"[STAGE3 {kind.value.upper()}] PASS" not in stdout
        or f"report_id={report.id}" not in stdout
    ):
        raise RuntimeError(f"Stage3 {kind.value} runner PASS changed")
    return oracle, report


def _run_case(
    staging: Path,
    kind: Stage3StaticCaseKind,
    args: argparse.Namespace,
) -> tuple[Stage3DenseInferenceOracle, Stage3StaticProfileRuntimeReport]:
    root = staging / "stage3" / kind.value
    raw_root = root / "raw"
    command = [
        sys.executable,
        "-B",
        str(_ROOT / "llm/test/frontend/integration/run_stage3_static_profiles.py"),
        "--case",
        kind.value,
        "--npusim",
        str(args.npusim),
        "--finalizer",
        str(args.finalizer),
        "--resolver",
        str(args.resolver),
        "--simulation",
        str(args.simulation),
        "--runtime-root",
        str(args.runtime_root),
        "--timeout",
        str(args.timeout),
        "--report-root",
        str(raw_root),
    ]
    completed = _run(command, cwd=_ROOT, timeout=args.timeout + 120)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage3 {kind.value} runner failed ({completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    _write_new(root / "runner.stdout.log", completed.stdout)
    _write_new(root / "runner.stderr.log", completed.stderr)
    report = load_json_dataclass(
        Stage3StaticProfileRuntimeReport,
        raw_root / f"stage3-{kind.value}.runtime.json",
        path=f"stage3.{kind.value}.runtime",
    )
    _write_new(root / "SUCCESS", report.id + "\n")
    _write_new(root / "inputs/input_summary.json", _input_summary(kind, args))
    return _validate_case(root, kind, args)


def _run_negative(staging: Path) -> dict[str, object]:
    root = staging / "stage3/negative"
    output = root / "negative_evidence.json"
    command = [
        sys.executable,
        "-B",
        str(
            _ROOT
            / "llm/test/frontend/integration/"
            "run_stage3_static_profile_negative_evidence.py"
        ),
        "--output",
        str(output),
    ]
    completed = _run(command, cwd=_ROOT, timeout=60)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage3 negative runner failed ({completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    _validate_negative_evidence(evidence)
    _write_new(root / "runner.stdout.log", completed.stdout)
    _write_new(root / "runner.stderr.log", completed.stderr)
    _write_new(root / "SUCCESS", str(evidence["id"]) + "\n")
    _require_exact_files(root, _NEGATIVE_FILES, "Stage3 negative")
    _require_exact_directories(root, set(), "Stage3 negative")
    if "[STAGE3 NEGATIVE] PASS" not in completed.stdout:
        raise RuntimeError("Stage3 negative PASS marker changed")
    return evidence


def _load_prior(root: Path) -> tuple[CaseMatrix, CapabilityManifest]:
    matrix = load_json_dataclass(
        CaseMatrix, root / "case_matrix.json", path="stage2.case_matrix"
    )
    manifest = load_json_dataclass(
        CapabilityManifest,
        root / "capability_manifest.json",
        path="stage2.capability_manifest",
    )
    manifest.validate_against(matrix)
    if manifest.baseline_epoch != _PRIOR_EPOCH:
        raise RuntimeError("prior capability epoch changed")
    return matrix, manifest


def _build_capability(
    prior_root: Path,
    evidence: dict[
        Stage3StaticCaseKind,
        tuple[Stage3DenseInferenceOracle, Stage3StaticProfileRuntimeReport],
    ],
    negative: dict[str, object],
) -> tuple[CaseMatrix, CapabilityManifest]:
    prior_matrix, prior_manifest = _load_prior(prior_root)
    matrix, manifest = build_stage3_capability_manifest(
        prior_matrix,
        prior_manifest,
        prefill_oracle=evidence[Stage3StaticCaseKind.PREFILL][0],
        prefill_report=evidence[Stage3StaticCaseKind.PREFILL][1],
        decode_oracle=evidence[Stage3StaticCaseKind.DECODE][0],
        decode_report=evidence[Stage3StaticCaseKind.DECODE][1],
        mixed_oracle=evidence[Stage3StaticCaseKind.MIXED][0],
        mixed_report=evidence[Stage3StaticCaseKind.MIXED][1],
        static_profile_fail_closed_evidence_digest=canonical_digest(negative),
    )
    if (
        manifest.coverage_score(CapabilityStage.S1) != Fraction(7, 2)
        or manifest.acceptance_score(CapabilityStage.S1) != (1, 3)
        or manifest.coverage_score(CapabilityStage.S2) != Fraction(0, 1)
        or manifest.coverage_score(CapabilityStage.S3) != Fraction(0, 1)
    ):
        raise RuntimeError("Stage3 capability score overclaim")
    return matrix, manifest


def _baseline_review(
    prior_root: Path,
    prior_digest: str,
    evidence: dict[
        Stage3StaticCaseKind,
        tuple[Stage3DenseInferenceOracle, Stage3StaticProfileRuntimeReport],
    ],
    negative: dict[str, object],
) -> dict[str, object]:
    cases = tuple(
        {
            "profile_mode": kind.value,
            "oracle_id": oracle.id,
            "oracle_digest": canonical_digest(oracle),
            "report_id": report.id,
            "report_digest": canonical_digest(report),
            "artifact_sha256": report.artifact.program_artifact_sha256,
            "artifact_size_bytes": report.artifact.artifact_size_bytes,
            "record_count": report.artifact.record_count,
            "relocation_count": report.artifact.relocation_count,
            "makespan_cycles": report.makespan_cycles,
            "marker_digest": report.repeats[0].marker_digest,
        }
        for kind, (oracle, report) in sorted(
            evidence.items(), key=lambda item: item[0].value
        )
    )
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_baseline_epoch": _PRIOR_EPOCH,
        "prior_baseline_path": str(prior_root.relative_to(_ROOT)),
        "prior_baseline_tree_digest": prior_digest,
        "review_kind": "static-profile-timing-baseline",
        "review_decision": "approved-stage3-static-profile-timing-baseline",
        "reviewer": "codex-stage3-static-profile-development",
        "reason": (
            "Stage3 adds exact prefill, decode and mixed/ragged static-profile "
            "timing, KV/HBM accounting and machine fail-closed evidence."
        ),
        "stage3_cases": cases,
        "stage3_negative_evidence_id": negative["id"],
        "stage3_negative_evidence_digest": canonical_digest(negative),
        "prior_stage_regression_review": {
            "reason": (
                "Program ISA minor 1.1 to 1.2 adds exact static-profile mode "
                "semantics; legacy case structure and timing are unchanged."
            ),
            "changed_fields": (
                "program_artifact_sha256",
                "derived finalizer, ProgramIo and report identities",
            ),
            "unchanged_fields": (
                "artifact_size_bytes",
                "record_count",
                "relocation_count",
                "makespan_cycles",
                "opcode, memory, D2D, control and drain evidence",
            ),
            "cases": _REGRESSION_CASES,
        },
        "caveats": (
            "Evidence proves timing execution and accounting, not "
            "compute-functional or model-functional correctness.",
            "Profile fallback eager/recompile remains declared unavailable.",
            "PD separation, KV handoff/reshard, training and optimized policies "
            "remain unsupported.",
            "Stage3 changes zero scored coverage: S1 remains 3.5/6 and 1/3; "
            "S2 and S3 scores remain zero.",
            "No ProgramArtifact .npup bytes are persisted in this epoch.",
            "The Stage2 checked tree is an immutable opaque prior, not copied "
            "or rewritten under current schemas.",
        ),
    }
    return {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "producer_pass": _PRODUCER,
        "id": stable_artifact_id(
            "baseline_review",
            semantic_key,
            schema_version=_REVIEW_SCHEMA_VERSION,
        ),
        **semantic_key,
    }


def _checked_summary(staging: Path, prior_digest: str) -> dict[str, object]:
    files = tuple(
        {"path": path, "sha256": digest, "size_bytes": size}
        for path, digest, size in _tree_rows(staging)
    )
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_tree_digest": prior_digest,
        "files": files,
        "payload_tree_digest": canonical_digest(files),
    }
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "producer_pass": _PRODUCER,
        "id": stable_artifact_id(
            "stage3_checked_rebuild_summary",
            semantic_key,
            schema_version=_SUMMARY_SCHEMA_VERSION,
        ),
        **semantic_key,
    }


def _validate_summary(staging: Path, prior_digest: str) -> None:
    summary = json.loads(
        (staging / "checked_rebuild_summary.json").read_text(encoding="utf-8")
    )
    files = tuple(
        {"path": path, "sha256": digest, "size_bytes": size}
        for path, digest, size in _tree_rows(staging)
        if path != "checked_rebuild_summary.json"
    )
    expected = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_tree_digest": prior_digest,
        "files": files,
        "payload_tree_digest": canonical_digest(files),
    }
    expected_full = {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "producer_pass": _PRODUCER,
        "id": stable_artifact_id(
            "stage3_checked_rebuild_summary",
            expected,
            schema_version=_SUMMARY_SCHEMA_VERSION,
        ),
        **expected,
    }
    if canonical_json(summary) != canonical_json(expected_full):
        raise RuntimeError("Stage3 checked rebuild summary changed")


def _validate_staging(
    staging: Path,
    args: argparse.Namespace,
    prior_digest: str,
) -> None:
    if {path.name for path in staging.iterdir()} != {
        "baseline_review.json",
        "capability_manifest.json",
        "case_matrix.json",
        "checked_rebuild_summary.json",
        "stage3",
    }:
        raise RuntimeError("Stage3 baseline top-level shape changed")
    if {path.name for path in (staging / "stage3").iterdir()} != {
        "prefill",
        "decode",
        "mixed",
        "negative",
    }:
        raise RuntimeError("Stage3 baseline case directory set changed")
    if any(path.suffix == ".npup" for path in staging.rglob("*")):
        raise RuntimeError("Stage3 checked baseline must not persist NPUP")
    evidence = {
        kind: _validate_case(staging / "stage3" / kind.value, kind, args)
        for kind in _MODES
    }
    negative_root = staging / "stage3/negative"
    _require_exact_files(negative_root, _NEGATIVE_FILES, "Stage3 negative")
    _require_exact_directories(negative_root, set(), "Stage3 negative")
    negative = json.loads(
        (negative_root / "negative_evidence.json").read_text(encoding="utf-8")
    )
    _validate_negative_evidence(negative)
    if (
        (negative_root / "SUCCESS").read_text(encoding="utf-8").strip()
        != negative["id"]
        or "[STAGE3 NEGATIVE] PASS"
        not in (negative_root / "runner.stdout.log").read_text(encoding="utf-8")
    ):
        raise RuntimeError("Stage3 negative persisted marker changed")
    matrix, manifest = _build_capability(args.prior_root, evidence, negative)
    for name, expected in (
        ("case_matrix.json", matrix),
        ("capability_manifest.json", manifest),
        (
            "baseline_review.json",
            _baseline_review(args.prior_root, prior_digest, evidence, negative),
        ),
    ):
        observed = (staging / name).read_bytes()
        wanted = (canonical_json(expected) + "\n").encode("utf-8")
        if observed != wanted:
            raise RuntimeError(f"{name} differs from disk-derived rebuild")
    _validate_summary(staging, prior_digest)


def _validate_publish_target(path: Path) -> None:
    _reject_symlink_ancestors(path)
    if path.name != _BASELINE_EPOCH:
        raise RuntimeError(f"baseline root must end in {_BASELINE_EPOCH}")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite checked evidence: {path}")
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
    parser.add_argument("--npusim", required=True, type=Path)
    parser.add_argument("--finalizer", required=True, type=Path)
    parser.add_argument("--resolver", required=True, type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    for name in ("prior_root", "npusim", "finalizer", "resolver", "simulation"):
        supplied = getattr(args, name)
        if supplied.is_symlink():
            parser.error(f"--{name.replace('_', '-')} must be a real path")
        resolved = supplied.resolve()
        if not resolved.exists():
            parser.error(f"--{name.replace('_', '-')} must exist: {resolved}")
        setattr(args, name, resolved)
    args.runtime_root = args.runtime_root.resolve()
    if not args.runtime_root.is_dir() or args.runtime_root.is_symlink():
        parser.error("--runtime-root must be a real directory")
    args.baseline_root = args.baseline_root.absolute()
    _validate_publish_target(args.baseline_root)
    if args.prior_root.name != _PRIOR_EPOCH or not args.prior_root.is_dir():
        parser.error(f"--prior-root must identify {_PRIOR_EPOCH}")
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
        evidence = {kind: _run_case(staging, kind, args) for kind in _MODES}
        negative = _run_negative(staging)
        matrix, manifest = _build_capability(
            args.prior_root, evidence, negative
        )
        review = _baseline_review(
            args.prior_root, prior_digest, evidence, negative
        )
        _write_new(staging / "case_matrix.json", matrix)
        _write_new(staging / "capability_manifest.json", manifest)
        _write_new(staging / "baseline_review.json", review)
        _write_new(
            staging / "checked_rebuild_summary.json",
            _checked_summary(staging, prior_digest),
        )
        _validate_staging(staging, args, prior_digest)
        if _tree_digest(args.prior_root) != prior_digest:
            raise RuntimeError("prior baseline changed during Stage3 freeze")
        _atomic_publish_noreplace(staging, args.baseline_root)
        published = True
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
    print(
        f"[STAGE3 FREEZE] PASS: baseline={args.baseline_root} "
        f"prior_digest={prior_digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
