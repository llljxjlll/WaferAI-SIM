#!/usr/bin/env python3
"""Finalize and resolve the formal Stage 4 PDS TP1 program twice."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from llm.frontend.wafer_frontend.passes import (  # noqa: E402
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (  # noqa: E402
    CommandFragment,
    EmptyCoreAckPolicy,
    ProgramFailurePolicy,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
)
from stage4_pd_cases import (  # noqa: E402
    Stage4PdCaseKind,
    build_stage4_pd_case,
)

_EXPECTED_FRAGMENT_COUNT = 96
_EXPECTED_RECORD_COUNT = 330
_EXPECTED_RELOCATION_COUNT = 602
_EXPECTED_ADDRESS_BINDING_COUNT = 564
_EXPECTED_STATE_BINDING_COUNT = 38
_EXPECTED_INITIALIZATION_COUNT = 120
_EXPECTED_PROBE_COUNT = 2
_EXPECTED_OPCODES = {
    "ATTENTION_EXACT": 4,
    "DTE_RECV": 4,
    "DTE_SEND": 4,
    "DTE_WAIT": 4,
    "EMBEDDING_LOOKUP": 2,
    "LSU_LOAD": 30,
    "LSU_STORE": 8,
    "MATMUL": 18,
    "RESIDUAL": 8,
    "RMSNORM": 10,
    "ROPE_QK_EXACT": 4,
    "SRAM_ALLOC_AT": 90,
    "SRAM_BIND": 50,
    "SRAM_FREE": 90,
    "SWIGLU": 4,
}


def _fail(message: str) -> None:
    raise RuntimeError(f"stage4_pd_tp1_pre_runtime: {message}")


def _executable(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _run(
    command: list[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        _fail(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed


def _leaf_fragments(case: object) -> tuple[CommandFragment, ...]:
    profile = getattr(case, "profile")
    leaves = profile.leaf_fragments
    if any(type(fragment) is not CommandFragment for fragment in leaves):
        _fail("linked carrier contains a non-CommandFragment leaf")
    return leaves


def _validate_static_case(case: object) -> None:
    manifest = getattr(case, "manifest")
    program_io = getattr(case, "program_io")
    leaves = _leaf_fragments(case)
    opcodes = Counter(
        record.opcode.name
        for fragment in leaves
        for stream in fragment.core_streams
        for record in stream.records
    )
    if (
        len(leaves) != _EXPECTED_FRAGMENT_COUNT
        or sum(opcodes.values()) != _EXPECTED_RECORD_COUNT
        or dict(opcodes) != _EXPECTED_OPCODES
        or len(manifest.address_operand_bindings)
        != _EXPECTED_ADDRESS_BINDING_COUNT
        or len(manifest.state_operand_bindings)
        != _EXPECTED_STATE_BINDING_COUNT
        or len(program_io.initializations)
        != _EXPECTED_INITIALIZATION_COUNT
        or len(program_io.output_probes) != _EXPECTED_PROBE_COUNT
    ):
        _fail("formal carrier/record/ProgramIo counts changed")
    if (
        opcodes[RecordOpcode.DTE_SEND.name]
        != opcodes[RecordOpcode.DTE_RECV.name]
        or opcodes[RecordOpcode.DTE_RECV.name]
        != opcodes[RecordOpcode.DTE_WAIT.name]
    ):
        _fail("sliced transfer SEND/RECV/WAIT coverage changed")

    envelope = manifest.envelope
    active = envelope.active_cores
    if (
        len(active) != 2
        or Counter(event.target_core for event in envelope.start_events)
        != Counter({core: 1 for core in active})
        or envelope.terminal_cores != active
        or envelope.expected_ack_cores != active
        or envelope.expected_done_cores != active
        or envelope.empty_core_ack_policy
        is not EmptyCoreAckPolicy.INCLUDE_EMPTY
        or envelope.failure_policy is not ProgramFailurePolicy.ABORT_ALL
        or tuple(stream.logical_core for stream in manifest.core_streams)
        != active
        or any(not stream.records for stream in manifest.core_streams)
    ):
        _fail("ACK/DONE/drain executable control envelope changed")


def _rebuild_sidecar(case: object, artifact_sha256: str):
    profile = getattr(case, "profile")
    placeholder = getattr(case, "program_io")
    manifest = getattr(case, "manifest")
    state_seeds, state_expected = build_deterministic_timing_state_overrides(
        profile
    )
    if (
        tuple(sorted(state_seeds)) != getattr(case, "state_seed_refs")
        or tuple(sorted(state_expected)) != getattr(case, "state_expected_refs")
    ):
        _fail("state override refs changed")
    result = build_timing_program_io(
        profile,
        artifact_sha256,
        state_seed_overrides=state_seeds,
        state_expected_overrides=state_expected,
    )
    result.validate_against(manifest)
    non_sha_fields = (
        "schema_version",
        "producer_pass",
        "mode",
        "source_linked_manifest_id",
        "source_linked_manifest_digest",
        "blobs",
        "initializations",
        "output_probes",
    )
    if tuple(getattr(result, field) for field in non_sha_fields) != tuple(
        getattr(placeholder, field) for field in non_sha_fields
    ):
        _fail("actual-SHA ProgramIo changed non-SHA semantics")
    return result


def _run_case(args: argparse.Namespace) -> None:
    case = build_stage4_pd_case(Stage4PdCaseKind.PDS)
    repeat = build_stage4_pd_case(Stage4PdCaseKind.PDS)
    for field in ("pd_plan", "logical_graph", "graph", "global_carrier", "lowered", "profile", "manifest", "program_io"):
        if canonical_digest(getattr(case, field)) != canonical_digest(
            getattr(repeat, field)
        ):
            _fail(f"builder {field} is not deterministic")
    if case.runtime_hardware_inputs != repeat.runtime_hardware_inputs:
        _fail("runtime hardware inputs are not deterministic")
    _validate_static_case(case)

    with tempfile.TemporaryDirectory(
        prefix="stage4-pds-tp1-", dir=args.runtime_root
    ) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        sidecar_path = directory / "program_io.json"
        artifact_paths = (
            directory / "program.0.npup",
            directory / "program.1.npup",
        )
        report_paths = (
            directory / "finalizer.0.json",
            directory / "finalizer.1.json",
        )
        manifest_path.write_text(canonical_json(case.manifest), encoding="utf-8")

        artifacts: list[bytes] = []
        reports: list[dict[str, object]] = []
        for artifact_path, report_path in zip(
            artifact_paths, report_paths, strict=True
        ):
            _run(
                [
                    str(args.finalizer),
                    "--input",
                    str(manifest_path),
                    "--output",
                    str(artifact_path),
                    "--report",
                    str(report_path),
                ],
                cwd=args.runtime_root,
            )
            artifacts.append(artifact_path.read_bytes())
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        if artifacts[0] != artifacts[1] or reports[0] != reports[1]:
            _fail("production finalizer repeat changed")

        artifact_sha256 = hashlib.sha256(artifacts[0]).hexdigest()
        report = reports[0]
        if (
            report.get("artifact_sha256") != artifact_sha256
            or report.get("artifact_bytes") != len(artifacts[0])
            or report.get("core_count") != 2
            or report.get("record_count") != _EXPECTED_RECORD_COUNT
            or report.get("relocation_count")
            != _EXPECTED_RELOCATION_COUNT
            or report.get("linked_manifest_id") != case.manifest.id
            or report.get("linked_manifest_digest")
            != canonical_digest(case.manifest)
        ):
            _fail(f"production finalizer report closure changed: {report}")

        sidecar = _rebuild_sidecar(case, artifact_sha256)
        sidecar_path.write_text(canonical_json(sidecar), encoding="utf-8")
        resolved = _run(
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(artifact_paths[0]),
                str(sidecar_path),
            ],
            cwd=args.runtime_root,
        )
        witness = (
            f"initializations={_EXPECTED_INITIALIZATION_COUNT} "
            f"probes={_EXPECTED_PROBE_COUNT}"
        )
        if witness not in resolved.stdout:
            _fail(f"resolver lost ProgramIo entry counts: {resolved.stdout}")

    print(
        "[STAGE4 PDS TP1 PRE-RUNTIME] PASS: "
        f"fragments={_EXPECTED_FRAGMENT_COUNT} "
        f"records={_EXPECTED_RECORD_COUNT} "
        f"relocations={_EXPECTED_RELOCATION_COUNT} "
        f"artifact_bytes={len(artifacts[0])} "
        f"sha256={artifact_sha256} repeat=2 "
        "ACK=2 DONE=2 drain_contract=1 runtime_executed=0"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalizer", required=True, type=_executable)
    parser.add_argument("--resolver", required=True, type=_executable)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    _run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
