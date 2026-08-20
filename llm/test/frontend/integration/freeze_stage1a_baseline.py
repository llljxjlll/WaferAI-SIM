#!/usr/bin/env python3
"""Freeze the reviewed Stage1a persistent-state evidence epoch."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from llm.frontend.wafer_frontend.passes import (  # noqa: E402
    build_stage1a_capability_manifest,
)
from llm.frontend.wafer_frontend.runner import NaiveRunReport  # noqa: E402
from llm.frontend.wafer_frontend.schema import (  # noqa: E402
    CapabilityManifest,
    CaseMatrix,
    LinkedProgramManifest,
    ProgramIoContract,
    Stage1aCase,
    Stage1aOracle,
    Stage1aRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.capability import (  # noqa: E402
    CapabilityStage,
    CapabilityStatus,
)
from llm.frontend.wafer_frontend.schema.common import (  # noqa: E402
    stable_artifact_id,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
    load_json_dataclass,
    load_json_value,
)
from run_stage1a_state_cases import (  # noqa: E402
    _case_data,
    _rebuild_sidecar,
    _runtime_hardware,
)


_BASELINE_EPOCH = "stage1a-persistent-state-v1"
_PRIOR_EPOCH = "stage0-policy-provenance-v1"
_REVIEW_SCHEMA_VERSION = "wafer_frontend.baseline_review/v1alpha2"
_DIFF_SCHEMA_VERSION = "wafer_frontend.reviewed_baseline_diff/v1alpha1"
_NEGATIVE_SCHEMA_VERSION = (
    "wafer_frontend.stage1a_negative_evidence/v1alpha1"
)
_CASES = (Stage1aCase.P1, Stage1aCase.K1, Stage1aCase.PD1)
_NEGATIVE_KEYS = (
    "dma_completion_dependency_missing",
    "dma_consumer_refs_incomplete",
    "dramsys_debug_peek_preflight",
    "hbm_address_uint64_overflow",
    "hbm_binding_misaligned",
    "hbm_binding_wrong_home_range",
    "program_io_sram_out_of_range",
    "read_only_state_writeback",
    "state_identity_duplicate",
)
_E1_E2_REQUIRED = (
    "SUCCESS",
    "compile/linked_manifest.json",
    "compile/pass_receipts.json",
    "compile/stage_digests.json",
    "inputs/experiment.json",
    "inputs/hardware.json",
    "inputs/mapping.spec",
    "inputs/simulation.json",
    "program/finalization_report.json",
    "program/finalizer.stderr.log",
    "program/finalizer.stdout.log",
    "program/program_io.json",
    "run/parsed_markers.json",
    "run/stderr.0.log",
    "run/stderr.1.log",
    "run/stdout.0.log",
    "run/stdout.1.log",
    "run_report.json",
)
_E1_E2_INVARIANTS = (
    "case",
    "inputs",
    "static_metrics.analytic_transfer_bytes",
    "static_metrics.op_counts",
    "static_metrics.rank_attention_matmul_flops",
    "static_metrics.rank_gemm_flops",
    "static_metrics.unique_flow_count",
    "runtime.ack_by_core",
    "runtime.ack_total",
    "runtime.credit_balanced",
    "runtime.d2d_link_packets",
    "runtime.done_by_core",
    "runtime.done_total",
    "runtime.drain_residuals",
    "runtime.observed_transfer_bytes",
    "validation",
    "validation_mode",
)
_ALLOWED_CHANGED_PREFIXES = (
    "artifact.",
    "id",
    "producer_pass",
    "provenance.",
    "runtime.makespan_cycles",
    "runtime.program_io_",
    "runtime.repeat",
    "runtime.repeat_signature_stable",
    "schema_version",
    "static_metrics.action_counts",
    "static_metrics.fragment_count",
    "static_metrics.opcode_counts",
    "static_metrics.per_core_sram_max_end",
    "static_metrics.record_count",
    "static_metrics.scheduled_binding_count",
    "static_metrics.task_counts",
    "status",
    "tools.",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_ROOT))
    except ValueError:
        return path.name


def _run(
    command: list[str], *, cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def _write_new(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite checked evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if type(value) is str else canonical_json(value) + "\n"
    path.write_text(text, encoding="utf-8")


def _tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"checked evidence must not contain symlinks: {path}")
        if path.is_file():
            rows.append((str(path.relative_to(root)), _sha256(path)))
    return canonical_digest(tuple(rows))


def _regression_tree_digest(root: Path) -> str:
    """Digest only the regression evidence that the baseline persists."""
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"regression evidence contains a symlink: {path}")
        if path.is_file() and path.suffix != ".npup":
            rows.append((str(path.relative_to(root)), _sha256(path)))
    return canonical_digest(tuple(rows))


def _file_summary(paths: tuple[Path, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "path": _relative(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    )


def _stage1a_case_command(
    case: Stage1aCase,
    args: argparse.Namespace,
    report_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(_ROOT / "llm/test/frontend/integration/run_stage1a_state_cases.py"),
        "--case",
        case.value.lower(),
        "--npusim",
        str(args.npusim),
        "--finalizer",
        str(args.finalizer),
        "--resolver",
        str(args.resolver),
        "--hardware",
        str(args.hardware),
        "--simulation",
        str(args.simulation),
        "--mapping",
        str(args.mapping),
        "--runtime-root",
        str(args.runtime_root),
        "--report-root",
        str(report_root),
    ]


def _raw_report_evidence(
    case: Stage1aCase, report_root: Path
) -> dict[str, str]:
    stem = case.value.lower()
    names = {
        f"{stem}.oracle.json",
        f"{stem}.runtime.json",
        f"{stem}.finalizer.0.log",
        f"{stem}.finalizer.1.log",
        f"{stem}.resolver.log",
        f"{stem}.runtime.0.log",
        f"{stem}.runtime.1.log",
    }
    if case is Stage1aCase.PD1:
        names.add(f"{stem}.dramsys-negative.log")
    paths = tuple(sorted(report_root.iterdir(), key=lambda item: item.name))
    if (
        {path.name for path in paths} != names
        or any(path.is_symlink() or not path.is_file() for path in paths)
        or any(path.suffix == ".npup" for path in paths)
    ):
        raise RuntimeError(
            f"{case.value} report-root evidence set changed: "
            f"{tuple(path.name for path in paths)}"
        )
    result = {
        path.name: path.read_text(encoding="utf-8") for path in paths
    }
    if not result[f"{stem}.resolver.log"]:
        raise RuntimeError(f"{case.value} resolver log must be non-empty")
    for index in range(2):
        runtime_log = result[f"{stem}.runtime.{index}.log"]
        if "[SIM_RESULT]" not in runtime_log or "[D2D_TYPE]" not in runtime_log:
            raise RuntimeError(
                f"{case.value} runtime.{index} lacks reviewed markers"
            )
    if case is Stage1aCase.PD1:
        negative = result[f"{stem}.dramsys-negative.log"]
        if (
            "HBM backend does not support debug peeking" not in negative
            or "[SIM_RESULT]" in negative
            or "[PROGRAM_MEMORY]" in negative
        ):
            raise RuntimeError("PD1 raw DRAMSys negative log changed")
    return result


def _run_stage1a_case(
    case: Stage1aCase,
    args: argparse.Namespace,
    temporary: Path,
) -> dict[str, str]:
    report_root = temporary / f"{case.value.lower()}-reports"
    command = _stage1a_case_command(case, args, report_root)
    completed = _run(command, cwd=_ROOT, timeout=300)
    pass_marker = f"[STAGE1A {case.value}] PASS"
    if completed.returncode != 0 or pass_marker not in completed.stdout:
        raise RuntimeError(
            f"{case.value} runtime failed: exit={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    stem = case.value.lower()
    raw_evidence = _raw_report_evidence(case, report_root)
    oracle = load_json_dataclass(
        Stage1aOracle,
        report_root / f"{stem}.oracle.json",
        path=f"{stem}.oracle",
    )
    report = load_json_dataclass(
        Stage1aRuntimeReport,
        report_root / f"{stem}.runtime.json",
        path=f"{stem}.runtime",
    )
    report.validate_against(oracle)
    data = _case_data(case)
    if (
        report.artifact.linked_manifest_id != data.manifest.id
        or report.artifact.linked_manifest_digest
        != canonical_digest(data.manifest)
    ):
        raise RuntimeError(f"{case.value} report does not close rebuilt manifest")

    manifest_path = temporary / f"{stem}.linked.json"
    manifest_path.write_text(canonical_json(data.manifest), encoding="utf-8")
    artifact_paths = (
        temporary / f"{stem}.0.npup",
        temporary / f"{stem}.1.npup",
    )
    finalization_paths = (
        temporary / f"{stem}.finalization.0.json",
        temporary / f"{stem}.finalization.1.json",
    )
    finalizer_logs = []
    for artifact_path, finalization_path in zip(
        artifact_paths, finalization_paths, strict=True
    ):
        finalized = _run(
            [
                str(args.finalizer),
                "--input",
                str(manifest_path),
                "--output",
                str(artifact_path),
                "--report",
                str(finalization_path),
            ],
            cwd=args.runtime_root,
            timeout=60,
        )
        if finalized.returncode != 0:
            raise RuntimeError(
                f"{case.value} finalizer failed: {finalized.stdout}"
                f"{finalized.stderr}"
            )
        finalizer_logs.append(finalized)
    if artifact_paths[0].read_bytes() != artifact_paths[1].read_bytes():
        raise RuntimeError(f"{case.value} finalizer bytes are not deterministic")
    finalizations = tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in finalization_paths
    )
    if finalizations[0] != finalizations[1]:
        raise RuntimeError(f"{case.value} finalizer report is not deterministic")
    finalization = finalizations[0]
    artifact_sha = _sha256(artifact_paths[0])
    if (
        artifact_sha != report.artifact.program_artifact_sha256
        or artifact_paths[0].stat().st_size
        != report.artifact.artifact_size_bytes
        or int(finalization["record_count"]) != report.artifact.record_count
        or int(finalization["relocation_count"])
        != report.artifact.relocation_count
        or finalization["linked_manifest_id"] != data.manifest.id
        or finalization["linked_manifest_digest"]
        != canonical_digest(data.manifest)
    ):
        raise RuntimeError(f"{case.value} finalization/report closure changed")
    sidecar = _rebuild_sidecar(data, artifact_sha)
    if (
        sidecar.id != report.sidecar.contract_id
        or canonical_digest(sidecar) != report.sidecar.contract_digest
    ):
        raise RuntimeError(f"{case.value} sidecar/report closure changed")
    runtime_hardware = temporary / f"{stem}.hardware.json"
    _runtime_hardware(case, data, args.hardware, runtime_hardware)
    if (
        _sha256(runtime_hardware) != report.hardware_digest
        or _sha256(args.simulation) != report.simulation_digest
        or _sha256(args.mapping) != report.mapping_digest
    ):
        raise RuntimeError(f"{case.value} input/report digest closure changed")

    stage_digests = {
        "global_action": canonical_digest(data.global_dag),
        "ir1": canonical_digest(data.graph),
        "linked_manifest": canonical_digest(data.manifest),
        "lowering_context": canonical_digest(data.lowering_context),
        "projection": canonical_digest(data.projection),
        "schedule_set": canonical_digest(data.schedule_set),
    }
    input_summary = {
        "binary_inputs": _file_summary(
            (args.finalizer, args.npusim, args.resolver)
        ),
        "case_builder": {
            "path": "llm/test/frontend/integration/stage1a_state_cases.py",
            "sha256": _sha256(
                _ROOT / "llm/test/frontend/integration/stage1a_state_cases.py"
            ),
        },
        "runtime_hardware_digest": _sha256(runtime_hardware),
        "source_inputs": _file_summary(
            (args.hardware, args.mapping, args.simulation)
        ),
    }
    parsed = {
        "control": report.control,
        "makespan_cycles": report.makespan_cycles,
        "memory": report.memory,
        "observed_d2d_bytes": report.observed_d2d_bytes,
        "observed_hbm_read_bytes": report.observed_hbm_read_bytes,
        "observed_hbm_write_bytes": report.observed_hbm_write_bytes,
        "probes": report.probes,
        "repeats": report.repeats,
    }
    return {
        "SUCCESS": report.id + "\n",
        "compile/linked_manifest.json": canonical_json(data.manifest) + "\n",
        "compile/stage_digests.json": canonical_json(stage_digests) + "\n",
        "inputs/input_summary.json": canonical_json(input_summary) + "\n",
        "oracle.json": canonical_json(oracle) + "\n",
        "program/finalization_report.json": canonical_json(finalization) + "\n",
        "program/finalizer.0.log": raw_evidence[f"{stem}.finalizer.0.log"],
        "program/finalizer.1.log": raw_evidence[f"{stem}.finalizer.1.log"],
        "program/program_io.json": canonical_json(sidecar) + "\n",
        "program/resolver.log": raw_evidence[f"{stem}.resolver.log"],
        "run/parsed_markers.json": canonical_json(parsed) + "\n",
        "run/runtime.0.log": raw_evidence[f"{stem}.runtime.0.log"],
        "run/runtime.1.log": raw_evidence[f"{stem}.runtime.1.log"],
        "runtime_report.json": canonical_json(report) + "\n",
        **(
            {"run/dramsys-negative.log": raw_evidence[f"{stem}.dramsys-negative.log"]}
            if case is Stage1aCase.PD1
            else {}
        ),
    }


def _validate_negative(
    value: object, args: argparse.Namespace
) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError("negative evidence must be an object")
    evidence = value
    if set(evidence) != {
        "baseline_epoch",
        "command",
        "external_inputs",
        "id",
        "producer_pass",
        "schema_version",
        "source_digests",
        "witnesses",
    }:
        raise RuntimeError("negative evidence has an unexpected wire shape")
    if (
        evidence["schema_version"] != _NEGATIVE_SCHEMA_VERSION
        or evidence["producer_pass"] != "stage1a_negative_runner"
        or evidence["baseline_epoch"] != _BASELINE_EPOCH
    ):
        raise RuntimeError("negative evidence identity/version changed")
    witnesses = evidence["witnesses"]
    if type(witnesses) is not list:
        raise RuntimeError("negative witnesses must be a canonical array")
    keys = tuple(sorted(item["key"] for item in witnesses))
    if keys != _NEGATIVE_KEYS or not all(item.get("passed") is True for item in witnesses):
        raise RuntimeError(f"negative witness closure changed: {keys}")
    semantic_key = {
        key: evidence[key]
        for key in evidence
        if key not in ("schema_version", "producer_pass", "id")
    }
    expected_id = stable_artifact_id(
        "stage1a_negative_evidence",
        semantic_key,
        schema_version=_NEGATIVE_SCHEMA_VERSION,
    )
    if evidence["id"] != expected_id:
        raise RuntimeError("negative evidence has a stale stable id")
    expected_external = {
        name: _sha256(getattr(args, name))
        for name in (
            "finalizer",
            "hardware",
            "mapping",
            "npusim",
            "resolver",
            "simulation",
        )
    }
    observed_external = {
        item["name"]: item["sha256"] for item in evidence["external_inputs"]
    }
    if observed_external != expected_external:
        raise RuntimeError("negative evidence external input digests changed")
    return evidence


def _run_negative(args: argparse.Namespace, temporary: Path) -> dict[str, str]:
    output = temporary / "negative.json"
    completed = _run(
        [
            sys.executable,
            "-B",
            str(
                _ROOT
                / "llm/test/frontend/integration/"
                "run_stage1a_negative_evidence.py"
            ),
            "--npusim",
            str(args.npusim),
            "--finalizer",
            str(args.finalizer),
            "--resolver",
            str(args.resolver),
            "--hardware",
            str(args.hardware),
            "--simulation",
            str(args.simulation),
            "--mapping",
            str(args.mapping),
            "--runtime-root",
            str(args.runtime_root),
            "--output",
            str(output),
        ],
        cwd=_ROOT,
        timeout=180,
    )
    if completed.returncode != 0 or "[STAGE1A NEGATIVE] PASS" not in completed.stdout:
        raise RuntimeError(
            f"negative runner failed: {completed.stdout}{completed.stderr}"
        )
    evidence = _validate_negative(
        load_json_value(output, path="stage1a_negative_evidence"), args
    )
    return {
        "negative_evidence.json": canonical_json(evidence) + "\n",
        "stderr.log": completed.stderr,
        "stdout.log": completed.stdout,
    }


def _read_path(data: object, path: str) -> object:
    value = data
    for component in path.split("."):
        if type(value) is not dict or component not in value:
            raise RuntimeError(f"review path is absent: {path}")
        value = value[component]
    return value


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if type(value) is dict:
        result: dict[str, object] = {}
        for key in sorted(value):
            nested = f"{prefix}.{key}" if prefix else key
            result.update(_flatten(value[key], nested))
        return result
    return {prefix: value}


def _program_memory_summary(
    root: Path, report: NaiveRunReport
) -> dict[str, object]:
    repeats = []
    for run_index in range(2):
        rows = []
        text = (root / f"run/stdout.{run_index}.log").read_text(
            encoding="utf-8"
        )
        for line in text.splitlines():
            position = line.find("[PROGRAM_MEMORY] ")
            if position < 0:
                continue
            fields = {}
            for token in line[position + len("[PROGRAM_MEMORY] ") :].split():
                if "=" in token:
                    key, value = token.rstrip(".").split("=", 1)
                    fields[key] = int(value, 10)
            required = {
                "core",
                "dte_residual",
                "lsu_completed",
                "lsu_hbm_read_bytes",
                "lsu_hbm_write_bytes",
                "lsu_issued",
                "lsu_residual",
            }
            if not required.issubset(fields):
                raise RuntimeError("PROGRAM_MEMORY marker is incomplete")
            if (
                fields["lsu_issued"] != fields["lsu_completed"]
                or fields["lsu_residual"] != 0
                or fields["dte_residual"] != 0
            ):
                raise RuntimeError("PROGRAM_MEMORY engines did not drain")
            rows.append(fields)
        rows.sort(key=lambda item: item["core"])
        if len({item["core"] for item in rows}) != len(rows):
            raise RuntimeError("PROGRAM_MEMORY core rows are not unique")
        repeats.append(tuple(rows))
    if repeats[0] != repeats[1]:
        raise RuntimeError("PROGRAM_MEMORY evidence changed across repeats")
    opcode_counts = report.static_metrics["opcode_counts"]
    has_lsu = any(
        opcode_counts.get(name, 0) for name in ("LSU_LOAD", "LSU_STORE")
    )
    if bool(repeats[0]) is not has_lsu:
        raise RuntimeError("PROGRAM_MEMORY presence disagrees with LSU opcodes")
    return {
        "observed_hbm_read_bytes": sum(
            item["lsu_hbm_read_bytes"] for item in repeats[0]
        ),
        "observed_hbm_write_bytes": sum(
            item["lsu_hbm_write_bytes"] for item in repeats[0]
        ),
        "per_core": repeats[0],
        "repeat_stable": True,
    }


def _regression_case(root: Path, expected_case: str) -> tuple[NaiveRunReport, dict[str, object]]:
    for relative in _E1_E2_REQUIRED:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{expected_case} evidence is incomplete: {path}")
    report = load_json_dataclass(
        NaiveRunReport,
        root / "run_report.json",
        path=f"{expected_case}.run_report",
    )
    if report.case.value != expected_case:
        raise RuntimeError(f"expected {expected_case}, observed {report.case.value}")
    if (root / "SUCCESS").read_text(encoding="utf-8").strip() != report.id:
        raise RuntimeError(f"{expected_case} SUCCESS does not bind its report")
    raw = json.loads((root / "run_report.json").read_text(encoding="utf-8"))
    summary = {
        "artifact": raw["artifact"],
        "linked_manifest_digest": _sha256(root / "compile/linked_manifest.json"),
        "linked_manifest_schema_version": json.loads(
            (root / "compile/linked_manifest.json").read_text(encoding="utf-8")
        )["schema_version"],
        "program_memory": _program_memory_summary(root, report),
        "report": raw,
        "report_digest": canonical_digest(report),
        "semantic_fingerprint_digest": canonical_digest(
            {
                "artifact": report.artifact,
                "runtime": report.runtime,
                "static_metrics": report.static_metrics,
                "validation": report.validation,
            }
        ),
        "persisted_tree_digest": _regression_tree_digest(root),
    }
    return report, summary


def _reviewed_diff(
    prior_root: Path,
    current_roots: dict[str, Path],
) -> dict[str, object]:
    rows = []
    for case in ("E1", "E2"):
        prior_report, prior = _regression_case(prior_root / case.lower(), case)
        current_report, current = _regression_case(current_roots[case], case)
        prior_raw = json.loads(canonical_json(prior_report))
        current_raw = json.loads(canonical_json(current_report))
        for invariant in _E1_E2_INVARIANTS:
            if _read_path(prior_raw, invariant) != _read_path(current_raw, invariant):
                raise RuntimeError(f"{case} invariant changed: {invariant}")
        old_flat = _flatten(prior_raw)
        new_flat = _flatten(current_raw)
        changed = tuple(
            {
                "field": field,
                "new": new_flat.get(field),
                "old": old_flat.get(field),
            }
            for field in sorted(set(old_flat) | set(new_flat))
            if old_flat.get(field) != new_flat.get(field)
        )
        unexpected = tuple(
            item["field"]
            for item in changed
            if not any(
                item["field"] == prefix or item["field"].startswith(prefix)
                for prefix in _ALLOWED_CHANGED_PREFIXES
            )
        )
        if unexpected:
            raise RuntimeError(f"{case} has unreviewed changes: {unexpected}")
        artifact_changed = (
            prior_report.artifact["artifact_sha256"]
            != current_report.artifact["artifact_sha256"]
        )
        structure_changed = any(
            _read_path(prior_raw, path) != _read_path(current_raw, path)
            for path in (
                "artifact.artifact_bytes",
                "artifact.record_count",
                "artifact.relocation_count",
                "static_metrics.opcode_counts",
            )
        )
        schema_changed = (
            prior["linked_manifest_schema_version"]
            != current["linked_manifest_schema_version"]
            or prior_report.schema_version != current_report.schema_version
        )
        if artifact_changed and not (structure_changed or schema_changed):
            raise RuntimeError(
                f"{case} artifact changed without schema or structural reason"
            )
        rows.append(
            {
                "case": case,
                "changed_fields": changed,
                "current": current,
                "prior": prior,
                "required_invariants": _E1_E2_INVARIANTS,
                "reviewed_change_classes": (
                    "persistent StateABI and LSU records",
                    "explicit HBM traffic and state ProgramIo",
                    "state DMA dependencies and SRAM lifecycle",
                    "breaking command/linker/finalizer versions",
                ),
            }
        )
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_baseline_epoch": _PRIOR_EPOCH,
        "cases": tuple(rows),
    }
    return {
        "schema_version": _DIFF_SCHEMA_VERSION,
        "producer_pass": "freeze_stage1a_baseline",
        "id": stable_artifact_id(
            "reviewed_baseline_diff",
            semantic_key,
            schema_version=_DIFF_SCHEMA_VERSION,
        ),
        **semantic_key,
    }


def _copy_regression(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"regression evidence contains a symlink: {path}")
        if not path.is_file() or path.suffix == ".npup":
            continue
        target = destination / path.relative_to(source)
        if target.exists():
            raise RuntimeError(f"duplicate regression evidence path: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _review(
    *,
    prior_root: Path,
    diff: dict[str, object],
    negative: dict[str, object],
    oracles: dict[Stage1aCase, Stage1aOracle],
    reports: dict[Stage1aCase, Stage1aRuntimeReport],
) -> dict[str, object]:
    cases = tuple(
        {
            "artifact_sha256": reports[case].artifact.program_artifact_sha256,
            "case": case.value,
            "linked_manifest_digest": reports[case].artifact.linked_manifest_digest,
            "makespan_cycles": reports[case].makespan_cycles,
            "observed_d2d_bytes": reports[case].observed_d2d_bytes,
            "observed_hbm_read_bytes": reports[case].observed_hbm_read_bytes,
            "observed_hbm_write_bytes": reports[case].observed_hbm_write_bytes,
            "oracle_digest": canonical_digest(oracles[case]),
            "report_digest": canonical_digest(reports[case]),
        }
        for case in _CASES
    )
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_baseline_epoch": _PRIOR_EPOCH,
        "prior_baseline_path": str(prior_root.relative_to(_ROOT)),
        "prior_baseline_tree_digest": _tree_digest(prior_root),
        "review_kind": "persistent-state-breaking-baseline",
        "reason": (
            "Stage1a introduced HBM-backed persistent parameter/KV identity, "
            "explicit state DMA, StateABI/LSU lowering and byte-exact ProgramIo."
        ),
        "version_changes": (
            "persistent state, IR2, schedule and GlobalAction contracts introduced",
            "CommandFragment v1alpha7 and LinkedProgramManifest v1alpha8",
            "ProgramIo tagged HBM/SRAM targets v1alpha2",
            "Stage1a oracle/runtime evidence v1alpha1 introduced",
        ),
        "cases": cases,
        "negative_evidence_digest": canonical_digest(negative),
        "negative_evidence_id": negative["id"],
        "reviewed_diff_digest": canonical_digest(diff),
        "reviewed_diff_id": diff["id"],
        "commands": (
            "python3 -B llm/test/frontend/integration/run_stage1a_negative_evidence.py ...",
            "python3 -B llm/test/frontend/integration/run_stage1a_state_cases.py --case {p1,k1,pd1} ...",
            "python3 -B llm/test/frontend/integration/freeze_stage1a_baseline.py ...",
        ),
        "reviewer": "codex-stage1a-development",
        "review_decision": "approved-stage1a-persistent-state-baseline",
        "caveats": (
            "P1/K1/PD1 prove timing execution and exact state transport, not model-functional outputs.",
            "PD1 is synthetic and does not upgrade s1.pd or s1.kv_handoff.",
            "No ProgramArtifact .npup bytes are persisted in this epoch.",
        ),
    }
    return {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "producer_pass": "freeze_stage1a_baseline",
        "id": stable_artifact_id(
            "baseline_review",
            semantic_key,
            schema_version=_REVIEW_SCHEMA_VERSION,
        ),
        **semantic_key,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--e1-root", required=True, type=Path)
    parser.add_argument("--e2-root", required=True, type=Path)
    parser.add_argument("--npusim", required=True, type=Path)
    parser.add_argument("--finalizer", required=True, type=Path)
    parser.add_argument("--resolver", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()
    for name in (
        "prior_root",
        "e1_root",
        "e2_root",
        "npusim",
        "finalizer",
        "resolver",
        "hardware",
        "simulation",
        "mapping",
    ):
        path = getattr(args, name).resolve()
        if not path.exists():
            parser.error(f"--{name} does not exist: {path}")
        setattr(args, name, path)
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    baseline_root = args.baseline_root.resolve()
    if baseline_root.name != _BASELINE_EPOCH:
        parser.error(f"--baseline-root must end in {_BASELINE_EPOCH}")
    if baseline_root.exists() or baseline_root.is_symlink():
        raise RuntimeError(f"refusing to overwrite checked evidence: {baseline_root}")
    if args.prior_root.name != _PRIOR_EPOCH:
        parser.error(f"--prior-root must identify {_PRIOR_EPOCH}")

    prior_before = _tree_digest(args.prior_root)
    with tempfile.TemporaryDirectory(
        prefix="stage1a-freeze-", dir=args.runtime_root
    ) as raw:
        temporary = Path(raw)
        case_files = {
            case: _run_stage1a_case(case, args, temporary)
            for case in _CASES
        }
        negative_files = _run_negative(args, temporary)
        negative = _validate_negative(
            json.loads(negative_files["negative_evidence.json"]), args
        )
        diff = _reviewed_diff(
            args.prior_root,
            {"E1": args.e1_root, "E2": args.e2_root},
        )
        oracles = {
            case: load_json_dataclass(
                Stage1aOracle,
                temporary
                / f"{case.value.lower()}-reports"
                / f"{case.value.lower()}.oracle.json",
                path=f"{case.value}.oracle",
            )
            for case in _CASES
        }
        reports = {
            case: load_json_dataclass(
                Stage1aRuntimeReport,
                temporary
                / f"{case.value.lower()}-reports"
                / f"{case.value.lower()}.runtime.json",
                path=f"{case.value}.runtime",
            )
            for case in _CASES
        }
        stage0_matrix = load_json_dataclass(
            CaseMatrix,
            args.prior_root / "case_matrix.json",
            path="stage0.case_matrix",
        )
        stage0_manifest = load_json_dataclass(
            CapabilityManifest,
            args.prior_root / "capability_manifest.json",
            path="stage0.capability_manifest",
        )
        matrix, manifest = build_stage1a_capability_manifest(
            stage0_matrix,
            stage0_manifest,
            p1_oracle=oracles[Stage1aCase.P1],
            p1_report=reports[Stage1aCase.P1],
            k1_oracle=oracles[Stage1aCase.K1],
            k1_report=reports[Stage1aCase.K1],
            pd1_oracle=oracles[Stage1aCase.PD1],
            pd1_report=reports[Stage1aCase.PD1],
            state_fail_closed_evidence_digest=canonical_digest(negative),
        )
        if (
            manifest.coverage_score(CapabilityStage.S1) != Fraction(7, 2)
            or manifest.acceptance_score(CapabilityStage.S1) != (1, 3)
        ):
            raise RuntimeError("Stage1a foundation evidence changed S1 scoring")
        claims = {claim.key: claim for claim in manifest.claims}
        for key in ("s1.kv_handoff", "s1.pd"):
            if claims[key].status is not CapabilityStatus.UNSUPPORTED:
                raise RuntimeError(f"Stage1a foundation overclaims {key}")
        review = _review(
            prior_root=args.prior_root,
            diff=diff,
            negative=negative,
            oracles=oracles,
            reports=reports,
        )

        baseline_root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{_BASELINE_EPOCH}.", dir=baseline_root.parent
            )
        )
        try:
            for case, files in case_files.items():
                for relative, text in files.items():
                    _write_new(staging / case.value.lower() / relative, text)
            for relative, text in negative_files.items():
                _write_new(staging / "negative" / relative, text)
            _copy_regression(args.e1_root, staging / "e1")
            _copy_regression(args.e2_root, staging / "e2")
            _write_new(staging / "case_matrix.json", matrix)
            _write_new(staging / "capability_manifest.json", manifest)
            _write_new(staging / "e1_e2_reviewed_diff.json", diff)
            _write_new(staging / "baseline_review.json", review)
            forbidden = tuple(staging.rglob("*.npup"))
            if forbidden:
                raise RuntimeError(f"baseline must not persist ProgramArtifact: {forbidden}")
            if _tree_digest(args.prior_root) != prior_before:
                raise RuntimeError(
                    "Stage0 baseline was modified during Stage1a freeze"
                )
            baseline_root.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(baseline_root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    print(
        "[STAGE1A BASELINE] PASS: "
        f"matrix={matrix.id} manifest={manifest.id} review={review['id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
