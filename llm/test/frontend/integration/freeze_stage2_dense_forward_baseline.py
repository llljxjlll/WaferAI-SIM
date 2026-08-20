#!/usr/bin/env python3
"""Freeze the reviewed Stage2 dense-forward timing evidence epoch."""

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

from llm.frontend.wafer_frontend import (  # noqa: E402
    NaiveRunCase,
    NaiveRunReport,
    NaiveRunRequest,
    NaiveRunValidation,
    run_naive,
)
from llm.frontend.wafer_frontend.passes import (  # noqa: E402
    build_stage2_capability_manifest,
)
from llm.frontend.wafer_frontend.schema import (  # noqa: E402
    CapabilityManifest,
    CaseMatrix,
    LinkedProgramManifest,
    ProgramIoContract,
    Stage1aCase,
    Stage1aOracle,
    Stage1aRuntimeReport,
    Stage2DenseForwardOracle,
    Stage2DenseForwardRuntimeReport,
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
)
from freeze_stage1a_baseline import (  # noqa: E402
    _file_summary as _stage1a_file_summary,
    _run_negative as _run_stage1a_negative,
    _run_stage1a_case,
    _validate_negative as _validate_stage1a_negative,
)
from run_stage2_dense_forward_negative_evidence import (  # noqa: E402
    build_negative_evidence as _build_stage2_negative_evidence,
)
from stage2_dense_forward_cases import (  # noqa: E402
    build_stage2_dense_forward_case,
)


_BASELINE_EPOCH = "stage2-dense-forward-v1"
_PRIOR_EPOCH = "stage1a-persistent-state-v1"
_SUMMARY_SCHEMA_VERSION = (
    "wafer_frontend.stage2_checked_rebuild_summary/v1alpha1"
)
_DIFF_SCHEMA_VERSION = (
    "wafer_frontend.stage1a_regression_reviewed_diff/v1alpha1"
)
_REVIEW_SCHEMA_VERSION = "wafer_frontend.baseline_review/v1alpha3"
_STAGE2_NEGATIVE_SCHEMA_VERSION = (
    "wafer_frontend.stage2_dense_forward_negative_evidence/v1alpha1"
)
_STAGE2_NEGATIVE_KEYS = (
    "ce_projection_unsupported",
    "resolver_artifact_sha_mismatch",
    "runtime_artifact_sha_tamper",
    "runtime_d2d_packet_tamper",
    "runtime_functional_overclaim",
    "runtime_makespan_tamper",
    "runtime_repeat_marker_mismatch",
    "tp2_greedy_unsupported",
    "tp4_greedy_unsupported",
)
_SOURCE_PATHS = (
    "llm/frontend/wafer_frontend/passes/logical_expand.py",
    "llm/frontend/wafer_frontend/policies/naive_project_to_ir2.py",
    "llm/frontend/wafer_frontend/schema/stage2_dense_forward_evidence.py",
    "llm/src/frontend/program_io.cpp",
    "llm/test/frontend/integration/stage2_dense_forward_cases.py",
    "llm/test/frontend/integration/run_stage2_dense_forward_negative_evidence.py",
    "llm/unittest/program_io_selftest_main.cpp",
)
_NEGATIVE_BINDING_SHAPE = {
    "ce_projection_unsupported": (
        ("python",),
        (
            "llm/frontend/wafer_frontend/policies/naive_project_to_ir2.py",
            "llm/test/frontend/integration/stage2_dense_forward_cases.py",
        ),
        ("ir1_with_ce",),
    ),
    "resolver_artifact_sha_mismatch": (
        ("finalizer", "resolver"),
        (
            "llm/src/frontend/program_io.cpp",
            "llm/unittest/program_io_selftest_main.cpp",
        ),
        ("linked_manifest", "program_artifact", "program_io"),
    ),
    **{
        key: (
            ("python",),
            (
                "llm/frontend/wafer_frontend/schema/"
                "stage2_dense_forward_evidence.py",
            ),
            ("mutated_runtime_semantic_key",),
        )
        for key in (
            "runtime_artifact_sha_tamper",
            "runtime_d2d_packet_tamper",
            "runtime_functional_overclaim",
            "runtime_makespan_tamper",
            "runtime_repeat_marker_mismatch",
        )
    },
    **{
        f"tp{tp}_greedy_unsupported": (
            ("python",),
            (
                "llm/frontend/wafer_frontend/passes/logical_expand.py",
                "llm/test/frontend/integration/stage2_dense_forward_cases.py",
            ),
            ("ir0_template",),
        )
        for tp in (2, 4)
    },
}
_STAGE2_RAW_SUFFIXES = (
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
def _summary(
    artifact_sha256: str,
    artifact_size_bytes: int,
    record_count: int,
    relocation_count: int,
    makespan_cycles: int,
    initialization_count: int,
    probe_count: int,
) -> dict[str, object]:
    return {
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": artifact_size_bytes,
        "record_count": record_count,
        "relocation_count": relocation_count,
        "makespan_cycles": makespan_cycles,
        "initialization_count": initialization_count,
        "probe_count": probe_count,
    }


_PRIOR_REGRESSION_GOLDENS = {
    "P1": _summary(
        "bc35aa29893675ff413c4566c54a3c54286cc6a84a962ffa74c9c76e302917c0",
        1694, 9, 16, 163, 4, 1,
    ),
    "K1": _summary(
        "e9c28d7d891bed9d4c2210cd217ac303da05b2b0588f60f6d4e33b43afbc1c72",
        16266, 122, 223, 1291, 37, 9,
    ),
    "PD1": _summary(
        "b7776e8798b10d7931f32806d6f5dcd170496f873c6dbe3f804b3ae4ab32b7c0",
        4272, 30, 44, 466, 10, 4,
    ),
    "E1": _summary(
        "fdd1f19839881c285e6f5a7a4deb4db8f4be663be92c39ef32a175658be7a7be",
        27676, 220, 354, 4292, 70, 6,
    ),
    "E2": _summary(
        "1caba2e72b7e3888d54e5f61097f54098434c8e81c42e846aff33f678ba815ed",
        156256, 1304, 1980, 8273, 372, 20,
    ),
}

_CURRENT_REGRESSION_GOLDENS = {
    "P1": _summary(
        "3f6bccb95bfc1f0d63fe720dac8bf98f67b7522d803e79410ca279ea536c8363",
        1694, 9, 16, 164, 4, 1,
    ),
    "K1": _summary(
        "42ae61bb31936950d74860172ece3c6602d7620fc948109e0a4f6f3c48d31a14",
        21230, 156, 287, 1633, 49, 9,
    ),
    "PD1": _summary(
        "a1adac8f41d5f10d7db5f0660ecff88f97b75029e6437b3d0de52fa0a2dd58e2",
        4448, 30, 44, 473, 10, 4,
    ),
    "E1": _summary(
        "95bb582bc852b35bc701ed41562664173fda95268e736363990ee0061bff6ef5",
        36392, 278, 464, 8958, 94, 2,
    ),
    "E2": _summary(
        "fe6ed061db11e54c182e64ca97c5a1741a5006df2d16e66e9d542ae13661edfe",
        179280, 1452, 2260, 12952, 432, 4,
    ),
}
_NAIVE_RESULT_FILES = {
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
}
_STAGE1A_CASE_FILES = {
    "SUCCESS",
    "compile/linked_manifest.json",
    "compile/stage_digests.json",
    "inputs/input_summary.json",
    "oracle.json",
    "program/finalization_report.json",
    "program/finalizer.0.log",
    "program/finalizer.1.log",
    "program/program_io.json",
    "program/resolver.log",
    "run/parsed_markers.json",
    "run/runtime.0.log",
    "run/runtime.1.log",
    "runtime_report.json",
}
_NEGATIVE_FILES = {
    "negative_evidence.json",
    "stderr.log",
    "stdout.log",
}
_STAGE2_CASE_FILES = {
    "SUCCESS",
    "inputs/input_summary.json",
    "runner.stderr.log",
    "runner.stdout.log",
}
_OFFICIAL_NAIVE_MARKER_SUFFIX = {
    "E1": (
        " artifact=36392B records=278 relocs=464 "
        "initializations=94 probes=2 ACK=4 DONE=2 drains=0 "
        "dma_in=18 dma_out=4 LSU_LOAD=18 LSU_STORE=4 fragments=52 "
        "makespan_cycles=8958 "
        "sha256=95bb582bc852b35bc701ed41562664173fda95268e736363990ee0061bff6ef5"
    ),
    "E2": (
        "artifact=179280B records=1452 relocs=2260 "
        "initializations=432 probes=4 ACK=8 DONE=4 drains=0 "
        "dma_in=60 dma_out=16 LSU_LOAD=60 LSU_STORE=16 fragments=180 "
        "makespan_cycles=12952 "
        "sha256=fe6ed061db11e54c182e64ca97c5a1741a5006df2d16e66e9d542ae13661edfe"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_rows(root: Path) -> tuple[tuple[str, str, int], ...]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"checked evidence root must be a real directory: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"checked evidence must not contain symlinks: {path}")
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
                f"checked evidence path has a symlink ancestor: {ancestor}"
            )


def _run(
    command: list[str], *, cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_ROOT)
    if existing_pythonpath:
        env["PYTHONPATH"] += os.pathsep + existing_pythonpath
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


def _write_new(path: Path, value: object) -> None:
    _reject_symlink_ancestors(path.parent)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite/symlink checked evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if type(value) is str else canonical_json(value) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _file_summary(paths: tuple[Path, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    )


def _relative_file_set(root: Path) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"evidence subtree must be a real directory: {root}")
    result = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"evidence subtree contains a symlink: {path}")
        if path.is_file():
            result.add(str(path.relative_to(root)))
    return result


def _require_exact_subtree(
    root: Path, expected: set[str], *, label: str
) -> None:
    observed = _relative_file_set(root)
    if observed != expected:
        raise RuntimeError(
            f"{label} subtree shape changed: {tuple(sorted(observed))}"
        )


def _stage2_input_summary(
    tp_degree: int, args: argparse.Namespace
) -> dict[str, object]:
    stem = f"stage2-tp{tp_degree}"
    return {
        "binary_inputs": _file_summary(
            (args.finalizer, args.npusim, args.resolver)
        ),
        "source_inputs": _file_summary((args.simulation,)),
        "case_builder": {
            "path": "llm/test/frontend/integration/stage2_dense_forward_cases.py",
            "sha256": _sha256(
                _ROOT
                / "llm/test/frontend/integration/stage2_dense_forward_cases.py"
            ),
        },
        "runner": {
            "path": "llm/test/frontend/integration/run_stage2_dense_forward.py",
            "sha256": _sha256(
                _ROOT
                / "llm/test/frontend/integration/run_stage2_dense_forward.py"
            ),
        },
        "evidence_shape": "official-strict-report-root/v1",
        "raw_file_names": tuple(
            sorted(f"{stem}.{suffix}" for suffix in _STAGE2_RAW_SUFFIXES)
        ),
    }


def _stage1a_input_summary(
    report: Stage1aRuntimeReport, args: argparse.Namespace
) -> dict[str, object]:
    return {
        "binary_inputs": _stage1a_file_summary(
            (args.finalizer, args.npusim, args.resolver)
        ),
        "case_builder": {
            "path": "llm/test/frontend/integration/stage1a_state_cases.py",
            "sha256": _sha256(
                _ROOT / "llm/test/frontend/integration/stage1a_state_cases.py"
            ),
        },
        "runtime_hardware_digest": report.hardware_digest,
        "source_inputs": _stage1a_file_summary(
            (args.hardware, args.mapping, args.simulation)
        ),
    }


def _validate_stage2_provenance(
    tp_degree: int,
    report: Stage2DenseForwardRuntimeReport,
    args: argparse.Namespace,
) -> None:
    rebuilt_case = build_stage2_dense_forward_case(tp_degree)
    expected_hardware_digest = hashlib.sha256(
        rebuilt_case.runtime_hardware_inputs.hardware_json.encode("utf-8")
    ).hexdigest()
    expected_mapping_digest = hashlib.sha256(
        rebuilt_case.runtime_hardware_inputs.mapping_text.encode("utf-8")
    ).hexdigest()
    if (
        report.tools.finalizer_sha256 != _sha256(args.finalizer)
        or report.tools.resolver_sha256 != _sha256(args.resolver)
        or report.tools.npusim_sha256 != _sha256(args.npusim)
        or report.simulation_digest != _sha256(args.simulation)
        or report.hardware_digest != expected_hardware_digest
        or report.mapping_digest != expected_mapping_digest
        or report.oracle_id != rebuilt_case.oracle.id
        or report.oracle_digest != canonical_digest(rebuilt_case.oracle)
    ):
        raise RuntimeError(
            f"Stage2 TP{tp_degree} tool/input/oracle provenance changed"
        )


def _stage2_case_command(
    tp_degree: int,
    args: argparse.Namespace,
    report_root: Path,
) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(
            _ROOT
            / "llm/test/frontend/integration/run_stage2_dense_forward.py"
        ),
        "--tp-degree",
        str(tp_degree),
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
        "--report-root",
        str(report_root),
        "--timeout",
        str({1: 300, 2: 600, 4: 1200}[tp_degree]),
    ]


def _run_stage2_case(
    tp_degree: int,
    args: argparse.Namespace,
    temporary: Path,
) -> tuple[dict[str, str], Stage2DenseForwardOracle, Stage2DenseForwardRuntimeReport]:
    report_root = temporary / f"stage2-tp{tp_degree}-reports"
    completed = _run(
        _stage2_case_command(tp_degree, args, report_root),
        cwd=_ROOT,
        timeout={1: 360, 2: 660, 4: 1260}[tp_degree],
    )
    pass_marker = f"[STAGE2 TP{tp_degree}] PASS"
    if completed.returncode != 0 or pass_marker not in completed.stdout:
        raise RuntimeError(
            f"Stage2 TP{tp_degree} runtime failed: exit={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    stem = f"stage2-tp{tp_degree}"
    expected_names = tuple(
        sorted(f"{stem}.{suffix}" for suffix in _STAGE2_RAW_SUFFIXES)
    )
    paths = tuple(sorted(report_root.iterdir(), key=lambda path: path.name))
    if (
        tuple(path.name for path in paths) != expected_names
        or any(path.is_symlink() or not path.is_file() for path in paths)
        or any(path.suffix == ".npup" for path in paths)
    ):
        raise RuntimeError(
            f"Stage2 TP{tp_degree} report-root shape changed: "
            f"{tuple(path.name for path in paths)}"
        )
    oracle_path = report_root / f"{stem}.oracle.json"
    runtime_path = report_root / f"{stem}.runtime.json"
    manifest_path = report_root / f"{stem}.linked_manifest.json"
    sidecar_path = report_root / f"{stem}.program_io.json"
    finalization_path = report_root / f"{stem}.finalization.json"
    oracle = load_json_dataclass(
        Stage2DenseForwardOracle,
        oracle_path,
        path=f"stage2.tp{tp_degree}.oracle",
    )
    report = load_json_dataclass(
        Stage2DenseForwardRuntimeReport,
        runtime_path,
        path=f"stage2.tp{tp_degree}.runtime",
    )
    report.validate_against(oracle)
    manifest = load_json_dataclass(
        LinkedProgramManifest,
        manifest_path,
        path=f"stage2.tp{tp_degree}.manifest",
    )
    sidecar = load_json_dataclass(
        ProgramIoContract,
        sidecar_path,
        path=f"stage2.tp{tp_degree}.program_io",
    )
    sidecar.validate_against(manifest)
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    if (
        report.artifact.linked_manifest_id != manifest.id
        or report.artifact.linked_manifest_digest != canonical_digest(manifest)
        or report.sidecar.contract_id != sidecar.id
        or report.sidecar.contract_digest != canonical_digest(sidecar)
        or finalization.get("artifact_sha256")
        != report.artifact.program_artifact_sha256
        or finalization.get("artifact_bytes")
        != report.artifact.artifact_size_bytes
        or finalization.get("record_count") != report.artifact.record_count
        or finalization.get("relocation_count")
        != report.artifact.relocation_count
    ):
        raise RuntimeError(
            f"Stage2 TP{tp_degree} report/manifest/sidecar/finalization closure changed"
        )
    raw = {
        f"raw/{path.name}": path.read_text(encoding="utf-8")
        for path in paths
    }
    input_summary = _stage2_input_summary(tp_degree, args)
    return (
        {
            "SUCCESS": report.id + "\n",
            "inputs/input_summary.json": canonical_json(input_summary) + "\n",
            "runner.stderr.log": completed.stderr,
            "runner.stdout.log": completed.stdout,
            **raw,
        },
        oracle,
        report,
    )


def _validate_stage2_negative(
    value: object,
    *,
    args: argparse.Namespace | None = None,
    expected_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    if type(value) is not dict:
        raise RuntimeError("Stage2 negative evidence must be an object")
    evidence = value
    expected_fields = {
        "baseline_epoch",
        "command",
        "id",
        "producer_pass",
        "schema_version",
        "source_digests",
        "witnesses",
    }
    if set(evidence) != expected_fields:
        raise RuntimeError("Stage2 negative evidence wire shape changed")
    if (
        evidence["schema_version"] != _STAGE2_NEGATIVE_SCHEMA_VERSION
        or evidence["producer_pass"]
        != "stage2_dense_forward_negative_runner"
        or evidence["baseline_epoch"] != _BASELINE_EPOCH
    ):
        raise RuntimeError("Stage2 negative evidence identity changed")
    witnesses = evidence["witnesses"]
    if type(witnesses) is not list or tuple(
        item.get("key") for item in witnesses
    ) != _STAGE2_NEGATIVE_KEYS:
        raise RuntimeError("Stage2 negative witness set/order changed")
    source_digests = evidence["source_digests"]
    expected_source_digests = [
        {"path": path, "sha256": _sha256(_ROOT / path)}
        for path in _SOURCE_PATHS
    ]
    if source_digests != expected_source_digests:
        raise RuntimeError("Stage2 negative top-level source provenance changed")
    for witness in witnesses:
        key = witness.get("key")
        bindings = witness.get("bindings")
        if (
            witness.get("passed") is not True
            or witness.get("expected_message") != witness.get("observed_error")
            or type(bindings) is not dict
            or set(bindings) != {"tools", "sources", "inputs"}
            or any(
                type(bindings.get(name)) is not list or not bindings[name]
                for name in ("tools", "sources", "inputs")
            )
        ):
            raise RuntimeError(
                f"Stage2 negative witness is incomplete: {witness}"
            )
        expected_tools, expected_sources, expected_inputs = (
            _NEGATIVE_BINDING_SHAPE[str(key)]
        )
        if (
            tuple(item.get("name") for item in bindings["tools"])
            != expected_tools
            or tuple(item.get("path") for item in bindings["sources"])
            != expected_sources
            or tuple(item.get("name") for item in bindings["inputs"])
            != expected_inputs
        ):
            raise RuntimeError(
                f"Stage2 negative witness provenance shape changed: {key}"
            )
        for source in bindings["sources"]:
            source_path = _ROOT / source["path"]
            if source.get("sha256") != _sha256(source_path):
                raise RuntimeError(
                    f"Stage2 negative source digest changed: {source['path']}"
                )
        expected_tool_paths = {
            "python": Path(sys.executable).resolve(),
            **(
                {
                    "finalizer": args.finalizer,
                    "resolver": args.resolver,
                }
                if args is not None
                else {}
            ),
        }
        for tool in bindings["tools"]:
            tool_path = expected_tool_paths.get(tool["name"])
            if tool_path is not None and tool.get("sha256") != _sha256(tool_path):
                raise RuntimeError(
                    f"Stage2 negative tool digest changed: {tool['name']}"
                )
        for group in bindings.values():
            for item in group:
                digest = item.get("sha256")
                if (
                    type(digest) is not str
                    or len(digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in digest
                    )
                ):
                    raise RuntimeError(
                        f"Stage2 negative binding digest changed: {item}"
                    )
    semantic_key = {
        key: evidence[key]
        for key in evidence
        if key not in ("schema_version", "producer_pass", "id")
    }
    expected_id = stable_artifact_id(
        "stage2_dense_forward_negative_evidence",
        semantic_key,
        schema_version=_STAGE2_NEGATIVE_SCHEMA_VERSION,
    )
    if evidence["id"] != expected_id:
        raise RuntimeError("Stage2 negative evidence stable id changed")
    if expected_evidence is None and args is not None:
        expected_evidence = _build_stage2_negative_evidence(args)
    if (
        expected_evidence is not None
        and canonical_json(evidence) != canonical_json(expected_evidence)
    ):
        raise RuntimeError(
            "Stage2 negative evidence differs from independently rebuilt evidence"
        )
    return evidence


def _run_stage2_negative(
    args: argparse.Namespace, temporary: Path
) -> tuple[dict[str, str], dict[str, object]]:
    output = temporary / "stage2-negative.json"
    command = [
        sys.executable,
        "-B",
        str(
            _ROOT
            / "llm/test/frontend/integration/"
            "run_stage2_dense_forward_negative_evidence.py"
        ),
        "--finalizer",
        str(args.finalizer),
        "--resolver",
        str(args.resolver),
        "--runtime-root",
        str(args.runtime_root),
        "--output",
        str(output),
    ]
    completed = _run(command, cwd=_ROOT, timeout=180)
    if (
        completed.returncode != 0
        or "[STAGE2 NEGATIVE] PASS: witnesses=9 deterministic=1"
        not in completed.stdout
    ):
        raise RuntimeError(
            "Stage2 negative runner failed: "
            f"{completed.stdout}{completed.stderr}"
        )
    evidence = _validate_stage2_negative(
        json.loads(output.read_text(encoding="utf-8")),
        args=args,
    )
    return (
        {
            "negative_evidence.json": canonical_json(evidence) + "\n",
            "stderr.log": completed.stderr,
            "stdout.log": completed.stdout,
        },
        evidence,
    )


def _read_result_tree(root: Path) -> dict[str, bytes]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"production result contains a symlink: {path}")
        if not path.is_file() or path.suffix == ".npup":
            continue
        files[str(path.relative_to(root))] = path.read_bytes()
    if set(files) != _NAIVE_RESULT_FILES:
        raise RuntimeError(
            "production result evidence shape changed: "
            f"{tuple(sorted(files))}"
        )
    return files


def _naive_summary(raw: dict[str, object]) -> dict[str, object]:
    artifact = raw["artifact"]
    runtime = raw["runtime"]
    if type(artifact) is not dict or type(runtime) is not dict:
        raise RuntimeError("Naive report artifact/runtime shape changed")
    return _summary(
        str(artifact["artifact_sha256"]),
        int(artifact["artifact_bytes"]),
        int(artifact["record_count"]),
        int(artifact["relocation_count"]),
        int(runtime["makespan_cycles"]),
        int(runtime["program_io_initializations"]),
        int(runtime["program_io_probes"]),
    )


def _official_naive_command(
    label: str, args: argparse.Namespace
) -> list[str]:
    lower = label.lower()
    return [
        sys.executable,
        "-B",
        str(
            _ROOT
            / f"llm/test/frontend/integration/run_naive_{lower}.py"
        ),
        "--npusim",
        str(args.npusim),
        "--finalizer",
        str(args.finalizer),
        "--spec",
        str(getattr(args, f"{lower}_spec")),
        "--hardware",
        str(getattr(args, f"{lower}_hardware")),
        "--simulation",
        str(args.simulation),
        "--mapping",
        str(args.mapping),
        "--runtime-root",
        str(args.runtime_root),
    ]


def _validate_official_naive_gate(
    label: str, completed: subprocess.CompletedProcess[str]
) -> str:
    prefix = f"[NAIVE N7 {label}] PASS: "
    lines = tuple(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith(prefix)
    )
    if completed.returncode != 0 or len(lines) != 1:
        raise RuntimeError(
            f"{label} official wrapper gate failed: exit={completed.returncode} "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    marker = lines[0]
    suffix = _OFFICIAL_NAIVE_MARKER_SUFFIX[label]
    if (
        (label == "E1" and not marker.startswith(prefix + "report=naive_run_report_"))
        or not marker.endswith(suffix)
    ):
        raise RuntimeError(
            f"{label} official wrapper static marker changed: {marker!r}"
        )
    if label == "E2" and marker != prefix + suffix:
        raise RuntimeError(
            f"{label} official wrapper static marker changed: {marker!r}"
        )
    return marker


def _naive_input_summary(
    label: str, args: argparse.Namespace, official_marker: str
) -> dict[str, object]:
    lower = label.lower()
    spec = getattr(args, f"{lower}_spec")
    hardware = getattr(args, f"{lower}_hardware")
    return {
        "binary_inputs": _file_summary((args.finalizer, args.npusim)),
        "source_inputs": _file_summary(
            (spec, hardware, args.mapping, args.simulation)
        ),
        "official_wrapper_gate": {
            "path": f"llm/test/frontend/integration/run_naive_{lower}.py",
            "sha256": _sha256(
                _ROOT
                / f"llm/test/frontend/integration/run_naive_{lower}.py"
            ),
            "marker": official_marker,
            "subprocess_exit_code": 0,
        },
        "evidence_shape": (
            "official-wrapper-gate-plus-production-run_naive-raw/v1"
        ),
        "evidence_collector": {
            "callable": "wafer_frontend.runner.run_naive",
            "path": "llm/frontend/wafer_frontend/runner.py",
            "sha256": _sha256(
                _ROOT / "llm/frontend/wafer_frontend/runner.py"
            ),
        },
        "persisted_program_artifact": False,
    }


def _run_naive_regression(
    label: str,
    args: argparse.Namespace,
    temporary: Path,
) -> tuple[dict[str, bytes], NaiveRunReport]:
    case = {"E1": NaiveRunCase.E1, "E2": NaiveRunCase.E2}[label]
    spec = {"E1": args.e1_spec, "E2": args.e2_spec}[label]
    hardware = {"E1": args.e1_hardware, "E2": args.e2_hardware}[label]
    official_command = _official_naive_command(label, args)
    official = _run(
        official_command,
        cwd=_ROOT,
        timeout=360 if label == "E1" else 960,
    )
    official_marker = _validate_official_naive_gate(label, official)
    output = temporary / f"{label.lower()}-result"
    result = run_naive(
        NaiveRunRequest(
            case=case,
            validation=NaiveRunValidation.TIMING,
            spec_path=spec,
            hardware_config_path=hardware,
            simulation_config_path=args.simulation,
            mapping_config_path=args.mapping,
            output_dir=output,
            npusim_path=args.npusim,
            finalizer_path=args.finalizer,
            timeout_seconds=300 if label == "E1" else 900,
        )
    )
    report = load_json_dataclass(
        NaiveRunReport,
        result.report_path,
        path=f"stage2.regression.{label}.report",
    )
    if canonical_digest(report) != canonical_digest(result.report):
        raise RuntimeError(f"{label} persisted/in-memory report changed")
    raw_report = json.loads(result.report_path.read_text(encoding="utf-8"))
    if _naive_summary(raw_report) != _CURRENT_REGRESSION_GOLDENS[label]:
        raise RuntimeError(
            f"{label} current production golden changed: "
            f"{_naive_summary(raw_report)}"
        )
    if (output / "SUCCESS").read_text(encoding="utf-8").strip() != report.id:
        raise RuntimeError(f"{label} SUCCESS does not identify current report")
    files = _read_result_tree(output)
    input_summary = _naive_input_summary(label, args, official_marker)
    files["inputs/freezer_input_summary.json"] = (
        canonical_json(input_summary) + "\n"
    ).encode("utf-8")
    files["official_wrapper.stderr.log"] = official.stderr.encode("utf-8")
    files["official_wrapper.stdout.log"] = official.stdout.encode("utf-8")
    return files, report


def _stage1a_summary(raw: dict[str, object]) -> dict[str, object]:
    artifact = raw["artifact"]
    sidecar = raw["sidecar"]
    if type(artifact) is not dict or type(sidecar) is not dict:
        raise RuntimeError("Stage1a report artifact/sidecar shape changed")
    return _summary(
        str(artifact["program_artifact_sha256"]),
        int(artifact["artifact_size_bytes"]),
        int(artifact["record_count"]),
        int(artifact["relocation_count"]),
        int(raw["makespan_cycles"]),
        int(sidecar["hbm_initialization_count"])
        + int(sidecar["sram_initialization_count"]),
        int(sidecar["hbm_probe_count"])
        + int(sidecar["sram_probe_count"]),
    )


_ISA_REASON = "ISA 1.1 command/wire migration changes encoded identities"
_COST_REASON = "manual-memory compute cost is charged exactly once"
_FULL_FORWARD_REASON = (
    "typed dense full-forward topology changes complete compile/runtime evidence"
)
_K1_FULL_FORWARD_REASON = (
    "K1 now crosses the typed dense full-forward action stream between state IO"
)
_PREFILL_REASON = (
    "pure-prefill KV is write-only, removing unread KV seeds and probes"
)
def _normalize_report(raw: dict[str, object]) -> dict[str, object]:
    if type(raw) is not dict:
        raise RuntimeError("reviewed report must be an object")
    return json.loads(canonical_json(raw))


def _recursive_diff(
    old: object, new: object, path: str = ""
) -> tuple[dict[str, object], ...]:
    if type(old) is dict and type(new) is dict:
        rows = []
        for key in sorted(set(old) | set(new)):
            nested = f"{path}.{key}" if path else key
            if key not in old:
                rows.append(
                    {
                        "kind": "added",
                        "new": new[key],
                        "new_present": True,
                        "old": None,
                        "old_present": False,
                        "path": nested,
                    }
                )
            elif key not in new:
                rows.append(
                    {
                        "kind": "removed",
                        "new": None,
                        "new_present": False,
                        "old": old[key],
                        "old_present": True,
                        "path": nested,
                    }
                )
            else:
                rows.extend(_recursive_diff(old[key], new[key], nested))
        return tuple(rows)
    if type(old) is list and type(new) is list:
        rows = []
        for index in range(max(len(old), len(new))):
            nested = f"{path}[{index}]"
            if index >= len(old):
                rows.append(
                    {
                        "kind": "added",
                        "new": new[index],
                        "new_present": True,
                        "old": None,
                        "old_present": False,
                        "path": nested,
                    }
                )
            elif index >= len(new):
                rows.append(
                    {
                        "kind": "removed",
                        "new": None,
                        "new_present": False,
                        "old": old[index],
                        "old_present": True,
                        "path": nested,
                    }
                )
            else:
                rows.extend(_recursive_diff(old[index], new[index], nested))
        return tuple(rows)
    if old == new and type(old) is type(new):
        return ()
    return (
        {
            "kind": "changed",
            "new": new,
            "new_present": True,
            "old": old,
            "old_present": True,
            "path": path,
        },
    )


_P1_CHANGED_PATHS = (
    "artifact.linked_manifest_digest",
    "artifact.linked_manifest_id",
    "artifact.program_artifact_sha256",
    "id",
    "makespan_cycles",
    "oracle_digest",
    "oracle_id",
    "probes[0].probe_id",
    *(f"repeats[{index}].{field}" for index in range(2) for field in (
        "makespan_cycles", "marker_digest", "probe_digest"
    )),
    "sidecar.contract_digest",
    "sidecar.contract_id",
)
_K1_CHANGED_PATHS = (
    "artifact.artifact_size_bytes",
    "artifact.linked_manifest_digest",
    "artifact.linked_manifest_id",
    "artifact.opcode_counts[0].count",
    "artifact.opcode_counts[1].opcode",
    *(
        f"artifact.opcode_counts[{index}].{field}"
        for index in range(2, 10)
        for field in ("count", "opcode")
    ),
    "artifact.opcode_counts[10]",
    "artifact.opcode_counts[11]",
    "artifact.program_artifact_sha256",
    "artifact.record_count",
    "artifact.relocation_count",
    "id",
    "makespan_cycles",
    "oracle_digest",
    "oracle_id",
    *(
        f"probes[{index}].{field}"
        for index in range(4)
        for field in ("actual_sha256", "expected_sha256", "probe_id")
    ),
    "probes[4].probe_id",
    *(f"probes[5].{field}" for field in (
        "actual_sha256", "expected_sha256", "length_bytes", "probe_id"
    )),
    *(
        f"probes[{index}].{field}"
        for index in (6, 7)
        for field in ("actual_sha256", "expected_sha256", "probe_id")
    ),
    "probes[8].probe_id",
    *(f"repeats[{index}].{field}" for index in range(2) for field in (
        "makespan_cycles", "marker_digest", "probe_digest"
    )),
    "sidecar.contract_digest",
    "sidecar.contract_id",
    "sidecar.sram_initialization_count",
)
_PD1_CHANGED_PATHS = (
    "artifact.artifact_size_bytes",
    "artifact.linked_manifest_digest",
    "artifact.linked_manifest_id",
    "artifact.opcode_counts[0].opcode",
    "artifact.program_artifact_sha256",
    "id",
    "makespan_cycles",
    "oracle_digest",
    "oracle_id",
    "pd_witness.completion_action_ids[0]",
    "pd_witness.completion_action_ids[1]",
    *(
        f"probes[{index}].{field}"
        for index in range(2)
        for field in ("actual_sha256", "expected_sha256", "probe_id")
    ),
    "probes[2].probe_id",
    "probes[3].probe_id",
    *(f"repeats[{index}].{field}" for index in range(2) for field in (
        "makespan_cycles", "marker_digest", "probe_digest"
    )),
    "sidecar.contract_digest",
    "sidecar.contract_id",
)
_E_CHANGED_COMMON = (
    "artifact.artifact_bytes",
    "artifact.artifact_sha256",
    "artifact.finalizer_report_digest",
    "artifact.program_io_digest",
    "artifact.program_io_id",
    "artifact.record_count",
    "artifact.relocation_count",
    "id",
    "inputs.spec_digest",
    "provenance.context_ids[3]",
    "provenance.context_ids[4]",
    "provenance.linked_bundle_id",
    "provenance.linked_manifest_digest",
    "provenance.linked_manifest_id",
    "provenance.linked_profile_id",
    *(
        f"provenance.pass_receipts[{index}].{field}"
        for index in range(5)
        for field in ("input_digest", "output_digest")
    ),
    *(f"provenance.pass_receipts[5].{field}" for field in (
        "context_digest", "input_digest", "output_digest"
    )),
    *(f"provenance.pass_receipts[6].{field}" for field in (
        "context_digest", "input_digest", "output_digest"
    )),
    "provenance.pass_receipts[6].policy_selections[0].id",
    "provenance.pass_receipts[6].policy_selections[0].implementation_schema_version",
    *(
        f"provenance.pass_receipts[{index}].{field}"
        for index in range(7, 10)
        for field in ("input_digest", "output_digest")
    ),
    "provenance.policy_selections[2].id",
    "provenance.policy_selections[2].implementation_schema_version",
    *(f"provenance.stage_digests[{index}]" for index in range(11)),
    "runtime.makespan_cycles",
    "runtime.program_io_initializations",
    "runtime.program_io_probes",
    "static_metrics.action_counts.comp",
    "static_metrics.action_counts.dma_in",
    "static_metrics.fragment_count",
    "static_metrics.op_counts.embedding",
    "static_metrics.op_counts.gemm",
    "static_metrics.op_counts.norm",
    "static_metrics.op_counts.rope",
    "static_metrics.opcode_counts.ATTENTION",
    "static_metrics.opcode_counts.ATTENTION_EXACT",
    "static_metrics.opcode_counts.EMBEDDING_LOOKUP",
    "static_metrics.opcode_counts.LSU_LOAD",
    "static_metrics.opcode_counts.MATMUL",
    "static_metrics.opcode_counts.RMSNORM",
    "static_metrics.opcode_counts.ROPE_QK_EXACT",
    "static_metrics.opcode_counts.SRAM_ALLOC_AT",
    "static_metrics.opcode_counts.SRAM_BIND",
    "static_metrics.opcode_counts.SRAM_FREE",
    "static_metrics.rank_gemm_flops",
    "static_metrics.record_count",
    "static_metrics.scheduled_binding_count",
    "static_metrics.task_counts.comp",
    "static_metrics.task_counts.dma_in",
    "tools.finalizer_sha256",
    "tools.linked_manifest_schema",
    "tools.npusim_sha256",
)
_EXPECTED_CHANGED_PATHS = {
    "P1": _P1_CHANGED_PATHS,
    "K1": _K1_CHANGED_PATHS,
    "PD1": _PD1_CHANGED_PATHS,
    "E1": _E_CHANGED_COMMON + (
        "static_metrics.per_core_sram_max_end.0",
        "static_metrics.per_core_sram_max_end.16",
    ),
    "E2": _E_CHANGED_COMMON + tuple(
        f"static_metrics.per_core_sram_max_end.{core}"
        for core in (0, 4, 8, 12)
    ),
}
_ADDED_PATHS = {
    "K1": {
        "artifact.opcode_counts[10]",
        "artifact.opcode_counts[11]",
    },
    "E1": {
        "static_metrics.op_counts.embedding",
        "static_metrics.op_counts.rope",
        "static_metrics.opcode_counts.ATTENTION_EXACT",
        "static_metrics.opcode_counts.EMBEDDING_LOOKUP",
        "static_metrics.opcode_counts.ROPE_QK_EXACT",
    },
    "E2": {
        "static_metrics.op_counts.embedding",
        "static_metrics.op_counts.rope",
        "static_metrics.opcode_counts.ATTENTION_EXACT",
        "static_metrics.opcode_counts.EMBEDDING_LOOKUP",
        "static_metrics.opcode_counts.ROPE_QK_EXACT",
    },
}
_REMOVED_PATHS = {
    "E1": {"static_metrics.opcode_counts.ATTENTION"},
    "E2": {"static_metrics.opcode_counts.ATTENTION"},
}
_FROZEN_REPORT_DIGESTS = {
    "P1": (
        "023de7fbbaeb12a1267d43bc4a0da22b8856415501c9670ea8ceee3832caabd6",
        "3b175079bd631137d14828649f8f0286dd6f2108d4d79d87ba4727393916dd97",
    ),
    "K1": (
        "5aa71e45444c3532426639c6f2f7458c54adecb66a006eb0df97b02d3ac0c9ae",
        "b9b604162177b512371e2e6ff9233ad490e61480e9d4ac7f2fe836bd27a72c07",
    ),
    "PD1": (
        "e5a00a416609b6db65c4743272d7a529e3705ae08c77a68d813ab26628d64670",
        "cb265ece4d47a236b127816c40a997899e4e58de75585b1eff45737fe7e4b7e4",
    ),
    "E1": (
        "8eab27665470dba1d92fe3a76f24899818520781c3240637c97287fb110b756f",
        "e1444695c16f4236ee6be00493c333d722aab86f3b8bc75121f964c9e7f6d75d",
    ),
    "E2": (
        "bb8c823d05128aa69ee3dc831b88576880a693eb75fce7c1a76870596c770279",
        "03cc511cd1b5581b0d123e44740da8cc251b749e1e0d3a0d9e6ca39e523f1752",
    ),
}


def _frozen_leaf_markers(
    label: str,
) -> tuple[dict[str, str], dict[str, str]]:
    paths = _EXPECTED_CHANGED_PATHS[label]
    added = _ADDED_PATHS.get(label, set())
    removed = _REMOVED_PATHS.get(label, set())
    return (
        {path: "old" for path in paths if path not in added},
        {path: "new" for path in paths if path not in removed},
    )


_FROZEN_NORMALIZED_GOLDENS = {
    label: {
        "old_report_digest": _FROZEN_REPORT_DIGESTS[label][0],
        "current_report_digest": _FROZEN_REPORT_DIGESTS[label][1],
        "old_changed_leaves": _frozen_leaf_markers(label)[0],
        "current_changed_leaves": _frozen_leaf_markers(label)[1],
    }
    for label in ("P1", "K1", "PD1", "E1", "E2")
}
_EXPECTED_CHANGE_CONTRACT = {
    label: {
        str(change["path"]): str(change["kind"])
        for change in _recursive_diff(
            golden["old_changed_leaves"],
            golden["current_changed_leaves"],
        )
    }
    for label, golden in _FROZEN_NORMALIZED_GOLDENS.items()
}


def _exact_reason(label: str, path: str) -> str:
    if path not in _EXPECTED_CHANGE_CONTRACT[label]:
        raise RuntimeError(f"{label} reviewed diff has an unknown changed path: {path}")
    if label == "K1":
        return _K1_FULL_FORWARD_REASON
    if path in {
        "runtime.program_io_initializations",
        "runtime.program_io_probes",
    }:
        return _PREFILL_REASON
    if label in ("E1", "E2") and (
        path == "id" or path.startswith("tools.")
    ):
        return _ISA_REASON
    if label in ("E1", "E2"):
        return _FULL_FORWARD_REASON
    if path == "makespan_cycles" or path.startswith("repeats["):
        return _COST_REASON
    return _ISA_REASON


_EXACT_REASON_MAP = {
    label: {
        path: _exact_reason(label, path)
        for path in _EXPECTED_CHANGE_CONTRACT[label]
    }
    for label in _EXPECTED_CHANGE_CONTRACT
}


def _validate_exact_change_contract(
    label: str, changes: tuple[dict[str, object], ...]
) -> None:
    observed = {
        str(change["path"]): str(change["kind"]) for change in changes
    }
    if observed != _EXPECTED_CHANGE_CONTRACT[label]:
        unknown = tuple(sorted(set(observed) - set(_EXPECTED_CHANGE_CONTRACT[label])))
        missing = tuple(sorted(set(_EXPECTED_CHANGE_CONTRACT[label]) - set(observed)))
        kind_mismatch = tuple(
            sorted(
                path
                for path in set(observed) & set(_EXPECTED_CHANGE_CONTRACT[label])
                if observed[path] != _EXPECTED_CHANGE_CONTRACT[label][path]
            )
        )
        raise RuntimeError(
            f"{label} reviewed diff contract changed: unknown={unknown} "
            f"missing={missing} kind_mismatch={kind_mismatch}"
        )


def _reviewed_diff(
    prior_root: Path,
    current_raw: dict[str, dict[str, object]],
    prior_tree_digest: str,
) -> dict[str, object]:
    rows = []
    for label in ("P1", "K1", "PD1", "E1", "E2"):
        prior_path = prior_root / label.lower() / (
            "runtime_report.json"
            if label in ("P1", "K1", "PD1")
            else "run_report.json"
        )
        if prior_path.is_symlink() or not prior_path.is_file():
            raise RuntimeError(f"prior opaque report is absent: {prior_path}")
        prior_raw = json.loads(prior_path.read_text(encoding="utf-8"))
        if type(prior_raw) is not dict:
            raise RuntimeError(f"prior opaque report is not an object: {prior_path}")
        prior_normalized = _normalize_report(prior_raw)
        current_normalized = _normalize_report(current_raw[label])
        prior_metrics = (
            _stage1a_summary(prior_raw)
            if label in ("P1", "K1", "PD1")
            else _naive_summary(prior_raw)
        )
        current_metrics = (
            _stage1a_summary(current_raw[label])
            if label in ("P1", "K1", "PD1")
            else _naive_summary(current_raw[label])
        )
        if prior_metrics != _PRIOR_REGRESSION_GOLDENS[label]:
            raise RuntimeError(
                f"{label} prior opaque regression golden changed: "
                f"{prior_metrics}"
            )
        if current_metrics != _CURRENT_REGRESSION_GOLDENS[label]:
            raise RuntimeError(
                f"{label} current regression golden changed: "
                f"{current_metrics}"
            )
        raw_changes = _recursive_diff(
            prior_normalized, current_normalized
        )
        _validate_exact_change_contract(label, raw_changes)
        prior_report_digest = canonical_digest(prior_normalized)
        current_report_digest = canonical_digest(current_normalized)
        if (
            prior_report_digest,
            current_report_digest,
        ) != (
            _FROZEN_NORMALIZED_GOLDENS[label]["old_report_digest"],
            _FROZEN_NORMALIZED_GOLDENS[label]["current_report_digest"],
        ):
            raise RuntimeError(
                f"{label} full normalized report golden changed: "
                f"old={prior_report_digest} new={current_report_digest}"
            )
        changes = tuple(
            {
                **change,
                "review_reason": _EXACT_REASON_MAP[label][
                    str(change["path"])
                ],
            }
            for change in raw_changes
        )
        if not changes:
            raise RuntimeError(f"{label} reviewed diff unexpectedly has no changes")
        rows.append(
            {
                "case": label,
                "allowed_reason_map": tuple(
                    {"path": path, "reason": reason}
                    for path, reason in sorted(
                        _EXACT_REASON_MAP[label].items()
                    )
                ),
                "changes": changes,
                "new": {
                    "metrics": current_metrics,
                    "normalized_evidence": current_normalized,
                    "report_digest": current_report_digest,
                },
                "old": {
                    "metrics": prior_metrics,
                    "normalized_evidence": prior_normalized,
                    "report_digest": prior_report_digest,
                },
            }
        )
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_baseline_epoch": _PRIOR_EPOCH,
        "prior_tree_digest": prior_tree_digest,
        "cases": tuple(rows),
    }
    return {
        "schema_version": _DIFF_SCHEMA_VERSION,
        "producer_pass": "freeze_stage2_dense_forward_baseline",
        "id": stable_artifact_id(
            "stage1a_regression_reviewed_diff",
            semantic_key,
            schema_version=_DIFF_SCHEMA_VERSION,
        ),
        **semantic_key,
    }


def _build_capability(
    prior_root: Path,
    stage2: dict[
        int, tuple[Stage2DenseForwardOracle, Stage2DenseForwardRuntimeReport]
    ],
    negative: dict[str, object],
) -> tuple[CaseMatrix, CapabilityManifest]:
    matrix = load_json_dataclass(
        CaseMatrix,
        prior_root / "case_matrix.json",
        path="stage1a.case_matrix",
    )
    manifest = load_json_dataclass(
        CapabilityManifest,
        prior_root / "capability_manifest.json",
        path="stage1a.capability_manifest",
    )
    manifest.validate_against(matrix)
    result = build_stage2_capability_manifest(
        matrix,
        manifest,
        tp1_oracle=stage2[1][0],
        tp1_report=stage2[1][1],
        tp2_oracle=stage2[2][0],
        tp2_report=stage2[2][1],
        tp4_oracle=stage2[4][0],
        tp4_report=stage2[4][1],
        dense_forward_fail_closed_evidence_digest=canonical_digest(negative),
    )
    new_matrix, new_manifest = result
    if (
        new_manifest.coverage_score(CapabilityStage.S1) != Fraction(7, 2)
        or new_manifest.acceptance_score(CapabilityStage.S1) != (1, 3)
        or new_manifest.coverage_score(CapabilityStage.S2) != Fraction(0, 1)
    ):
        raise RuntimeError("Stage2 evidence unexpectedly changed reviewed scores")
    claims = {claim.key: claim for claim in new_manifest.claims}
    if (
        claims["s1.foundation.dense_forward_closure"].status
        is not CapabilityStatus.E2E_TIMING
        or claims["s1.foundation.dense_forward_fail_closed"].status
        is not CapabilityStatus.UNIT_ONLY
        or any(
            claim.status is not CapabilityStatus.UNSUPPORTED
            for claim in new_manifest.claims
            if claim.key.startswith("s2.naive_case.")
        )
    ):
        raise RuntimeError("Stage2 capability manifest overclaims reviewed evidence")
    return result


def _baseline_review(
    *,
    prior_root: Path,
    prior_tree_digest: str,
    diff: dict[str, object],
    reports: dict[int, Stage2DenseForwardRuntimeReport],
    negative: dict[str, object],
) -> dict[str, object]:
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_baseline_epoch": _PRIOR_EPOCH,
        "prior_baseline_path": str(prior_root.relative_to(_ROOT)),
        "prior_baseline_tree_digest": prior_tree_digest,
        "review_kind": "dense-forward-timing-baseline",
        "reason": (
            "Stage2 adds typed dense full-forward TP1/TP2/TP4 timing, exact "
            "traffic/accounting evidence and machine-executed fail-closed evidence."
        ),
        "stage2_cases": tuple(
            {
                "tp_degree": tp,
                "artifact_sha256": reports[tp].artifact.program_artifact_sha256,
                "artifact_size_bytes": reports[tp].artifact.artifact_size_bytes,
                "record_count": reports[tp].artifact.record_count,
                "relocation_count": reports[tp].artifact.relocation_count,
                "makespan_cycles": reports[tp].makespan_cycles,
                "oracle_digest": reports[tp].oracle_digest,
                "report_digest": canonical_digest(reports[tp]),
            }
            for tp in (1, 2, 4)
        ),
        "stage2_negative_evidence_digest": canonical_digest(negative),
        "stage2_negative_evidence_id": negative["id"],
        "stage1a_regression_reviewed_diff_digest": canonical_digest(diff),
        "stage1a_regression_reviewed_diff_id": diff["id"],
        "reviewer": "codex-stage2-dense-forward-development",
        "review_decision": "approved-stage2-dense-forward-timing-baseline",
        "caveats": (
            "Dense-forward evidence proves timing execution and accounting, not compute-functional or model-functional correctness.",
            "GREEDY TP2/TP4 and CE lowering remain fail-closed; training remains unsupported.",
            "Stage2 contributes zero score and leaves S1 at 3.5/6, acceptance 1/3, and S2 at 0.",
            "No ProgramArtifact .npup bytes are persisted in this epoch.",
            "Prior Stage1a runtime/Command/Linked wires are treated as opaque bytes, not loaded by current strict schemas.",
        ),
    }
    return {
        "schema_version": _REVIEW_SCHEMA_VERSION,
        "producer_pass": "freeze_stage2_dense_forward_baseline",
        "id": stable_artifact_id(
            "baseline_review",
            semantic_key,
            schema_version=_REVIEW_SCHEMA_VERSION,
        ),
        **semantic_key,
    }


def _write_files(root: Path, files: dict[str, str | bytes]) -> None:
    _reject_symlink_ancestors(root)
    for relative, value in sorted(files.items()):
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.suffix == ".npup"
        ):
            raise RuntimeError(
                f"invalid checked evidence relative path: {relative}"
            )
        target = root / relative
        _reject_symlink_ancestors(target.parent)
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"duplicate checked evidence path: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if type(value) is bytes:
            with target.open("xb") as stream:
                stream.write(value)
        elif type(value) is str:
            with target.open("x", encoding="utf-8") as stream:
                stream.write(value)
        else:
            raise RuntimeError(f"unsupported checked evidence payload: {target}")


def _checked_summary(
    staging: Path, prior_tree_digest: str
) -> dict[str, object]:
    rows = tuple(
        {
            "path": path,
            "sha256": digest,
            "size_bytes": size,
        }
        for path, digest, size in _tree_rows(staging)
    )
    semantic_key = {
        "baseline_epoch": _BASELINE_EPOCH,
        "prior_tree_digest": prior_tree_digest,
        "files": rows,
        "payload_tree_digest": canonical_digest(rows),
    }
    return {
        "schema_version": _SUMMARY_SCHEMA_VERSION,
        "producer_pass": "freeze_stage2_dense_forward_baseline",
        "id": stable_artifact_id(
            "stage2_checked_rebuild_summary",
            semantic_key,
            schema_version=_SUMMARY_SCHEMA_VERSION,
        ),
        **semantic_key,
    }


def _validate_checked_summary(
    staging: Path, prior_tree_digest: str | None = None
) -> dict[str, object]:
    path = staging / "checked_rebuild_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        type(summary) is not dict
        or set(summary)
        != {
            "baseline_epoch",
            "files",
            "id",
            "payload_tree_digest",
            "prior_tree_digest",
            "producer_pass",
            "schema_version",
        }
        or summary.get("schema_version") != _SUMMARY_SCHEMA_VERSION
        or summary.get("producer_pass")
        != "freeze_stage2_dense_forward_baseline"
        or summary.get("baseline_epoch") != _BASELINE_EPOCH
    ):
        raise RuntimeError("checked rebuild summary identity changed")
    if (
        prior_tree_digest is not None
        and summary.get("prior_tree_digest") != prior_tree_digest
    ):
        raise RuntimeError("checked rebuild prior tree digest changed")
    rows = tuple(
        {
            "path": relative,
            "sha256": digest,
            "size_bytes": size,
        }
        for relative, digest, size in _tree_rows(staging)
        if relative != "checked_rebuild_summary.json"
    )
    if summary.get("files") != list(rows) or summary.get(
        "payload_tree_digest"
    ) != canonical_digest(rows):
        raise RuntimeError("checked rebuild file inventory changed")
    semantic_key = {
        key: summary[key]
        for key in summary
        if key not in ("schema_version", "producer_pass", "id")
    }
    expected_id = stable_artifact_id(
        "stage2_checked_rebuild_summary",
        semantic_key,
        schema_version=_SUMMARY_SCHEMA_VERSION,
    )
    if summary.get("id") != expected_id:
        raise RuntimeError("checked rebuild summary stable id changed")
    return summary


def _require_canonical_file(
    path: Path,
    expected: object,
    *,
    label: str,
    trailing_newline: bool = True,
) -> None:
    suffix = "\n" if trailing_newline else ""
    expected_bytes = (canonical_json(expected) + suffix).encode("utf-8")
    if path.read_bytes() != expected_bytes:
        raise RuntimeError(f"{label} differs from the disk-derived rebuild")


def _validate_stage1a_disk_closure(
    root: Path, report: Stage1aRuntimeReport
) -> None:
    manifest = load_json_dataclass(
        LinkedProgramManifest,
        root / "compile/linked_manifest.json",
        path="stage1a.linked_manifest",
    )
    sidecar = load_json_dataclass(
        ProgramIoContract,
        root / "program/program_io.json",
        path="stage1a.program_io",
    )
    sidecar.validate_against(manifest)
    finalization = json.loads(
        (root / "program/finalization_report.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        report.artifact.linked_manifest_id != manifest.id
        or report.artifact.linked_manifest_digest != canonical_digest(manifest)
        or report.sidecar.contract_id != sidecar.id
        or report.sidecar.contract_digest != canonical_digest(sidecar)
        or sidecar.program_artifact_sha256
        != report.artifact.program_artifact_sha256
        or finalization.get("artifact_sha256")
        != report.artifact.program_artifact_sha256
        or finalization.get("artifact_bytes")
        != report.artifact.artifact_size_bytes
        or finalization.get("record_count") != report.artifact.record_count
        or finalization.get("relocation_count")
        != report.artifact.relocation_count
        or finalization.get("linked_manifest_id") != manifest.id
        or finalization.get("linked_manifest_digest")
        != canonical_digest(manifest)
    ):
        raise RuntimeError("Stage1a persisted artifact closure changed")
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
    _require_canonical_file(
        root / "run/parsed_markers.json",
        parsed,
        label="Stage1a parsed runtime markers",
    )


def _validate_naive_disk_closure(
    root: Path, report: NaiveRunReport
) -> None:
    manifest = load_json_dataclass(
        LinkedProgramManifest,
        root / "compile/linked_manifest.json",
        path="naive.linked_manifest",
    )
    sidecar = load_json_dataclass(
        ProgramIoContract,
        root / "program/program_io.json",
        path="naive.program_io",
    )
    sidecar.validate_against(manifest)
    finalization = json.loads(
        (root / "program/finalization_report.json").read_text(
            encoding="utf-8"
        )
    )
    artifact = report.artifact
    provenance = report.provenance
    if (
        provenance.get("linked_manifest_id") != manifest.id
        or provenance.get("linked_manifest_digest")
        != canonical_digest(manifest)
        or artifact.get("program_io_id") != sidecar.id
        or artifact.get("program_io_digest") != canonical_digest(sidecar)
        or sidecar.program_artifact_sha256 != artifact.get("artifact_sha256")
        or artifact.get("finalizer_report_digest")
        != canonical_digest(finalization)
        or finalization.get("artifact_sha256") != artifact.get("artifact_sha256")
        or finalization.get("artifact_bytes") != artifact.get("artifact_bytes")
        or finalization.get("record_count") != artifact.get("record_count")
        or finalization.get("relocation_count") != artifact.get("relocation_count")
        or finalization.get("linked_manifest_id") != manifest.id
        or finalization.get("linked_manifest_digest")
        != canonical_digest(manifest)
    ):
        raise RuntimeError("E1/E2 persisted artifact closure changed")
    runtime = report.runtime
    parsed = tuple(
        {
            "repeat": index,
            "makespan_cycles": runtime["makespan_cycles"],
            "ack_total": runtime["ack_total"],
            "done_total": runtime["done_total"],
            "ack_by_core": runtime["ack_by_core"],
            "done_by_core": runtime["done_by_core"],
            "observed_transfer_bytes": runtime["observed_transfer_bytes"],
            "d2d_link_packets": runtime["d2d_link_packets"],
        }
        for index in range(2)
    )
    _require_canonical_file(
        root / "run/parsed_markers.json",
        parsed,
        label="E1/E2 parsed runtime markers",
    )


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
    _rename_noreplace(staging, target)


def _rename_noreplace(staging: Path, target: Path) -> None:
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
    if (
        renameat2(
            -100,
            os.fsencode(staging),
            -100,
            os.fsencode(target),
            1,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise RuntimeError(
                f"refusing to overwrite raced publish target: {target}"
            )
        raise OSError(
            error_number,
            f"atomic no-clobber publish failed: {staging} -> {target}",
        )


def _validate_staging(
    staging: Path,
    *,
    args: argparse.Namespace,
    prior_tree_digest: str,
) -> None:
    top_level = {path.name for path in staging.iterdir()}
    expected_top_level = {
        "baseline_review.json",
        "capability_manifest.json",
        "case_matrix.json",
        "checked_rebuild_summary.json",
        "stage1a",
        "stage1a_regression_reviewed_diff.json",
        "stage2",
    }
    if top_level != expected_top_level:
        raise RuntimeError(
            f"checked baseline top-level shape changed: {tuple(sorted(top_level))}"
        )
    rows = _tree_rows(staging)
    if any(path.endswith(".npup") for path, _, _ in rows):
        raise RuntimeError("checked baseline must not persist ProgramArtifact bytes")
    if {path.name for path in (staging / "stage1a").iterdir()} != {
        "e1",
        "e2",
        "k1",
        "negative",
        "p1",
        "pd1",
    }:
        raise RuntimeError("Stage1a immediate child set changed")
    if {path.name for path in (staging / "stage2").iterdir()} != {
        "negative",
        "tp1",
        "tp2",
        "tp4",
    }:
        raise RuntimeError("Stage2 immediate child set changed")

    current_raw: dict[str, dict[str, object]] = {}
    for case in (Stage1aCase.P1, Stage1aCase.K1, Stage1aCase.PD1):
        stem = case.value.lower()
        root = staging / "stage1a" / stem
        expected_files = set(_STAGE1A_CASE_FILES)
        if case is Stage1aCase.PD1:
            expected_files.add("run/dramsys-negative.log")
        _require_exact_subtree(
            root, expected_files, label=f"Stage1a {case.value}"
        )
        oracle = load_json_dataclass(
            Stage1aOracle, root / "oracle.json", path=f"stage1a.{stem}.oracle"
        )
        report = load_json_dataclass(
            Stage1aRuntimeReport,
            root / "runtime_report.json",
            path=f"stage1a.{stem}.runtime",
        )
        report.validate_against(oracle)
        _validate_stage1a_disk_closure(root, report)
        if (root / "SUCCESS").read_text(encoding="utf-8").strip() != report.id:
            raise RuntimeError(f"Stage1a {case.value} SUCCESS changed")
        raw = json.loads((root / "runtime_report.json").read_text(encoding="utf-8"))
        if _stage1a_summary(raw) != _CURRENT_REGRESSION_GOLDENS[case.value]:
            raise RuntimeError(f"Stage1a {case.value} current golden changed")
        current_raw[case.value] = raw
        input_summary_path = root / "inputs/input_summary.json"
        _require_canonical_file(
            input_summary_path,
            _stage1a_input_summary(report, args),
            label=f"Stage1a {case.value} input summary",
        )

    stage1a_negative_root = staging / "stage1a/negative"
    _require_exact_subtree(
        stage1a_negative_root, _NEGATIVE_FILES, label="Stage1a negative"
    )
    stage1a_negative = json.loads(
        (stage1a_negative_root / "negative_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    _validate_stage1a_negative(stage1a_negative, args)

    for label in ("E1", "E2"):
        root = staging / "stage1a" / label.lower()
        expected_files = set(_NAIVE_RESULT_FILES) | {
            "inputs/freezer_input_summary.json",
            "official_wrapper.stderr.log",
            "official_wrapper.stdout.log",
        }
        _require_exact_subtree(
            root, expected_files, label=f"{label} current evidence"
        )
        report = load_json_dataclass(
            NaiveRunReport,
            root / "run_report.json",
            path=f"stage1a.{label}.run_report",
        )
        _validate_naive_disk_closure(root, report)
        if (root / "SUCCESS").read_text(encoding="utf-8").strip() != report.id:
            raise RuntimeError(f"{label} SUCCESS changed")
        raw = json.loads((root / "run_report.json").read_text(encoding="utf-8"))
        if _naive_summary(raw) != _CURRENT_REGRESSION_GOLDENS[label]:
            raise RuntimeError(f"{label} current production golden changed")
        current_raw[label] = raw
        completed = subprocess.CompletedProcess(
            args=("persisted-official-wrapper",),
            returncode=0,
            stdout=(root / "official_wrapper.stdout.log").read_text(
                encoding="utf-8"
            ),
            stderr=(root / "official_wrapper.stderr.log").read_text(
                encoding="utf-8"
            ),
        )
        marker = _validate_official_naive_gate(label, completed)
        _require_canonical_file(
            root / "inputs/freezer_input_summary.json",
            _naive_input_summary(label, args, marker),
            label=f"{label} freezer input summary",
        )

    stage2_evidence: dict[
        int, tuple[Stage2DenseForwardOracle, Stage2DenseForwardRuntimeReport]
    ] = {}
    for tp_degree in (1, 2, 4):
        stem = f"stage2-tp{tp_degree}"
        root = staging / "stage2" / f"tp{tp_degree}"
        raw_root = root / "raw"
        expected_case_files = set(_STAGE2_CASE_FILES) | {
            f"raw/{stem}.{suffix}" for suffix in _STAGE2_RAW_SUFFIXES
        }
        _require_exact_subtree(
            root, expected_case_files, label=f"Stage2 TP{tp_degree}"
        )
        raw_names = tuple(sorted(path.name for path in raw_root.iterdir()))
        expected_raw_names = tuple(
            sorted(f"{stem}.{suffix}" for suffix in _STAGE2_RAW_SUFFIXES)
        )
        if raw_names != expected_raw_names:
            raise RuntimeError(f"Stage2 TP{tp_degree} raw evidence shape changed")
        oracle = load_json_dataclass(
            Stage2DenseForwardOracle,
            raw_root / f"{stem}.oracle.json",
            path=f"stage2.tp{tp_degree}.oracle",
        )
        report = load_json_dataclass(
            Stage2DenseForwardRuntimeReport,
            raw_root / f"{stem}.runtime.json",
            path=f"stage2.tp{tp_degree}.runtime",
        )
        report.validate_against(oracle)
        _validate_stage2_provenance(tp_degree, report, args)
        manifest = load_json_dataclass(
            LinkedProgramManifest,
            raw_root / f"{stem}.linked_manifest.json",
            path=f"stage2.tp{tp_degree}.manifest",
        )
        sidecar = load_json_dataclass(
            ProgramIoContract,
            raw_root / f"{stem}.program_io.json",
            path=f"stage2.tp{tp_degree}.program_io",
        )
        sidecar.validate_against(manifest)
        finalization = json.loads(
            (raw_root / f"{stem}.finalization.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            report.artifact.linked_manifest_id != manifest.id
            or report.artifact.linked_manifest_digest != canonical_digest(manifest)
            or report.sidecar.contract_id != sidecar.id
            or report.sidecar.contract_digest != canonical_digest(sidecar)
            or finalization.get("artifact_sha256")
            != report.artifact.program_artifact_sha256
            or finalization.get("artifact_bytes")
            != report.artifact.artifact_size_bytes
            or finalization.get("record_count") != report.artifact.record_count
            or finalization.get("relocation_count")
            != report.artifact.relocation_count
            or finalization.get("linked_manifest_id") != manifest.id
            or finalization.get("linked_manifest_digest")
            != canonical_digest(manifest)
            or (root / "SUCCESS").read_text(encoding="utf-8").strip()
            != report.id
        ):
            raise RuntimeError(f"Stage2 TP{tp_degree} checked closure changed")
        if not (raw_root / f"{stem}.resolver.log").read_text(
            encoding="utf-8"
        ):
            raise RuntimeError(f"Stage2 TP{tp_degree} resolver log is empty")
        for index in range(2):
            runtime_log = (
                raw_root / f"{stem}.runtime.{index}.log"
            ).read_text(encoding="utf-8")
            if "[SIM_RESULT]" not in runtime_log or "[D2D_TYPE]" not in runtime_log:
                raise RuntimeError(
                    f"Stage2 TP{tp_degree} runtime log {index} lacks markers"
                )
        expected_inputs = {
            "schema_version": "wafer_frontend.stage2_dense_forward_inputs/v1",
            "tools": report.tools,
            "hardware_digest": report.hardware_digest,
            "simulation_digest": report.simulation_digest,
            "mapping_digest": report.mapping_digest,
            "template_digest": report.compile.template_digest,
            "ir1_digest": report.compile.ir1_digest,
            "global_dag_digest": report.compile.global_dag_digest,
            "lowered_digest": report.compile.lowered_digest,
        }
        _require_canonical_file(
            raw_root / f"{stem}.input_digests.json",
            expected_inputs,
            label=f"Stage2 TP{tp_degree} input digests",
            trailing_newline=False,
        )
        _require_canonical_file(
            root / "inputs/input_summary.json",
            _stage2_input_summary(tp_degree, args),
            label=f"Stage2 TP{tp_degree} freezer input summary",
        )
        stage2_evidence[tp_degree] = (oracle, report)

    stage2_negative_root = staging / "stage2/negative"
    _require_exact_subtree(
        stage2_negative_root, _NEGATIVE_FILES, label="Stage2 negative"
    )
    negative = _validate_stage2_negative(
        json.loads(
            (
                stage2_negative_root / "negative_evidence.json"
            ).read_text(encoding="utf-8")
        ),
        args=args,
    )

    matrix = load_json_dataclass(
        CaseMatrix, staging / "case_matrix.json", path="stage2.case_matrix"
    )
    manifest = load_json_dataclass(
        CapabilityManifest,
        staging / "capability_manifest.json",
        path="stage2.capability_manifest",
    )
    manifest.validate_against(matrix)
    if (
        manifest.coverage_score(CapabilityStage.S1) != Fraction(7, 2)
        or manifest.acceptance_score(CapabilityStage.S1) != (1, 3)
        or manifest.coverage_score(CapabilityStage.S2) != Fraction(0, 1)
    ):
        raise RuntimeError("checked capability scores changed")
    rebuilt_matrix, rebuilt_manifest = _build_capability(
        args.prior_root, stage2_evidence, negative
    )
    _require_canonical_file(
        staging / "case_matrix.json",
        rebuilt_matrix,
        label="case matrix",
    )
    _require_canonical_file(
        staging / "capability_manifest.json",
        rebuilt_manifest,
        label="capability manifest",
    )
    rebuilt_diff = _reviewed_diff(
        args.prior_root, current_raw, prior_tree_digest
    )
    _require_canonical_file(
        staging / "stage1a_regression_reviewed_diff.json",
        rebuilt_diff,
        label="Stage1a reviewed diff",
    )
    rebuilt_review = _baseline_review(
        prior_root=args.prior_root,
        prior_tree_digest=prior_tree_digest,
        diff=rebuilt_diff,
        reports={
            tp_degree: pair[1]
            for tp_degree, pair in stage2_evidence.items()
        },
        negative=negative,
    )
    _require_canonical_file(
        staging / "baseline_review.json",
        rebuilt_review,
        label="baseline review",
    )
    _validate_checked_summary(staging, prior_tree_digest)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--npusim", required=True, type=Path)
    parser.add_argument("--finalizer", required=True, type=Path)
    parser.add_argument("--resolver", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--e1-spec", required=True, type=Path)
    parser.add_argument("--e1-hardware", required=True, type=Path)
    parser.add_argument("--e2-spec", required=True, type=Path)
    parser.add_argument("--e2-hardware", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()
    for name in (
        "prior_root",
        "npusim",
        "finalizer",
        "resolver",
        "hardware",
        "simulation",
        "mapping",
        "e1_spec",
        "e1_hardware",
        "e2_spec",
        "e2_hardware",
    ):
        supplied = getattr(args, name)
        if supplied.is_symlink():
            parser.error(
                f"--{name.replace('_', '-')} must be a real path: {supplied}"
            )
        path = supplied.resolve()
        if not path.exists():
            parser.error(f"--{name.replace('_', '-')} must exist: {path}")
        setattr(args, name, path)
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.runtime_root.is_symlink() or not args.runtime_root.is_dir():
        parser.error("--runtime-root must be a real directory")
    args.baseline_root = args.baseline_root.absolute()
    _validate_publish_target(args.baseline_root)
    if args.prior_root.name != _PRIOR_EPOCH or not args.prior_root.is_dir():
        parser.error(f"--prior-root must identify {_PRIOR_EPOCH}")
    return args


def main() -> int:
    args = _parse_args()
    baseline_root = args.baseline_root
    prior_before = _tree_digest(args.prior_root)
    with tempfile.TemporaryDirectory(
        prefix="stage2-freeze-run-", dir=args.runtime_root
    ) as raw:
        temporary = Path(raw)
        stage1a_files = {
            case: _run_stage1a_case(case, args, temporary)
            for case in (Stage1aCase.P1, Stage1aCase.K1, Stage1aCase.PD1)
        }
        stage1a_negative_files = _run_stage1a_negative(args, temporary)
        _validate_stage1a_negative(
            json.loads(stage1a_negative_files["negative_evidence.json"]), args
        )
        regression_files: dict[str, dict[str, bytes]] = {}
        for label in ("E1", "E2"):
            files, _ = _run_naive_regression(label, args, temporary)
            regression_files[label] = files

        stage2_files: dict[int, dict[str, str]] = {}
        stage2_evidence: dict[
            int,
            tuple[Stage2DenseForwardOracle, Stage2DenseForwardRuntimeReport],
        ] = {}
        for tp_degree in (1, 2, 4):
            files, oracle, report = _run_stage2_case(tp_degree, args, temporary)
            stage2_files[tp_degree] = files
            stage2_evidence[tp_degree] = (oracle, report)
        stage2_negative_files, stage2_negative = _run_stage2_negative(
            args, temporary
        )

        current_raw = {
            case.value: json.loads(
                stage1a_files[case]["runtime_report.json"]
            )
            for case in (Stage1aCase.P1, Stage1aCase.K1, Stage1aCase.PD1)
        }
        current_raw.update(
            {
                label: json.loads(
                    regression_files[label]["run_report.json"].decode("utf-8")
                )
                for label in ("E1", "E2")
            }
        )
        diff = _reviewed_diff(args.prior_root, current_raw, prior_before)
        matrix, manifest = _build_capability(
            args.prior_root, stage2_evidence, stage2_negative
        )
        reports = {
            tp_degree: pair[1]
            for tp_degree, pair in stage2_evidence.items()
        }
        review = _baseline_review(
            prior_root=args.prior_root,
            prior_tree_digest=prior_before,
            diff=diff,
            reports=reports,
            negative=stage2_negative,
        )

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{_BASELINE_EPOCH}.", dir=baseline_root.parent
            )
        )
        published = False
        try:
            for case, files in stage1a_files.items():
                _write_files(staging / "stage1a" / case.value.lower(), files)
            _write_files(staging / "stage1a/negative", stage1a_negative_files)
            for label, files in regression_files.items():
                _write_files(staging / "stage1a" / label.lower(), files)
            for tp_degree, files in stage2_files.items():
                _write_files(staging / "stage2" / f"tp{tp_degree}", files)
            _write_files(staging / "stage2/negative", stage2_negative_files)
            _write_new(staging / "case_matrix.json", matrix)
            _write_new(staging / "capability_manifest.json", manifest)
            _write_new(
                staging / "stage1a_regression_reviewed_diff.json", diff
            )
            _write_new(staging / "baseline_review.json", review)
            summary = _checked_summary(staging, prior_before)
            _write_new(staging / "checked_rebuild_summary.json", summary)
            _validate_staging(
                staging,
                args=args,
                prior_tree_digest=prior_before,
            )
            if _tree_digest(args.prior_root) != prior_before:
                raise RuntimeError("prior baseline changed during checked rebuild")
            if _tree_digest(args.prior_root) != prior_before:
                raise RuntimeError("prior baseline changed before atomic publish")
            _atomic_publish_noreplace(staging, baseline_root)
            published = True
        finally:
            if not published and staging.exists() and not staging.is_symlink():
                shutil.rmtree(staging)
    print(
        f"[STAGE2 FREEZE] PASS: baseline={baseline_root} "
        f"prior_digest={prior_before}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
