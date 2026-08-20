#!/usr/bin/env python3
"""Run the formal DP2 x TP2 forward-Train timing program twice."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
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
    RegionManifest,
)
from llm.frontend.wafer_frontend.schema.common import DType  # noqa: E402
from llm.frontend.wafer_frontend.schema.program_io import (  # noqa: E402
    ProgramHbmTarget,
    ProgramIoContract,
    ProgramIoTargetKind,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
)
from llm.frontend.wafer_frontend.schema.train_forward_evidence import (  # noqa: E402
    TRAIN_FORWARD_MARKER_SCHEMA_VERSION,
    TrainForwardArtifactEvidence,
    TrainForwardCeMarkerEvidence,
    TrainForwardControlEvidence,
    TrainForwardCoreCount,
    TrainForwardMemoryEvidence,
    TrainForwardNamedCount,
    TrainForwardOpcodeCount,
    TrainForwardProgramIoEvidence,
    TrainForwardRepeatEvidence,
    TrainForwardRuntimeReport,
    TrainForwardWorkEvidence,
)
from train_forward_cases import (  # noqa: E402
    TrainForwardCase,
    build_train_forward_case,
)

_STATUS = "[PROGRAM_IO] "
_PROBE = "[PROGRAM_IO_PROBE] "
_MEMORY = "[PROGRAM_MEMORY] "
_CE = "[TRAIN_CE] "
_SIM = "[SIM_RESULT] "
_HOST = "[HOSTLANE] "
_HOSTSIG = "[HOSTSIG] "
_P5 = "[P5 P2P DRAIN] "
_P5_TIMING = "[P5 P2P TIMING DRAIN] "
_COLL = "[COLL_DRAIN] "
_DRAIN = "[DRAIN] "
_D2D_TYPE = "[D2D_TYPE] "
_D2D_BEHA = "[D2D_BEHA] "
_D2D_LINK = "[D2D_LINK] "
_PACKET_BYTES = 16
_ACTIVE_CORES = (0, 4, 8, 12)

_EXPECTED_OPCODES = {
    "ATTENTION_EXACT": 8,
    "CROSS_ENTROPY_FORWARD": 4,
    "DTE_ISSUE": 16,
    "DTE_RECV": 32,
    "DTE_SEND": 32,
    "DTE_WAIT": 32,
    "EMBEDDING_LOOKUP": 4,
    "EVENT_SET": 16,
    "EVENT_WAIT": 16,
    "LOCAL_REDUCE": 16,
    "LSU_LOAD": 60,
    "MATMUL": 52,
    "RESIDUAL": 16,
    "RMSNORM": 20,
    "ROPE_QK_EXACT": 8,
    "SRAM_ALLOC_AT": 268,
    "SRAM_BIND": 120,
    "SRAM_FREE": 268,
    "SWIGLU": 8,
}


@dataclass(frozen=True, slots=True)
class _RuntimeObservation:
    makespan_cycles: int
    marker_digest: str
    memory: tuple[TrainForwardMemoryEvidence, ...]
    ce_markers: tuple[TrainForwardCeMarkerEvidence, ...]
    control: TrainForwardControlEvidence
    d2d_links: tuple[tuple[int, int, int, int, int], ...]


def _fail(message: str) -> None:
    raise RuntimeError(f"[TRAIN FORWARD DP2 TP2] FAIL: {message}")


def _executable(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _run(
    command: list[str], *, cwd: Path, timeout: int
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if completed.returncode:
        _fail(
            f"command failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stdout}"
        )
    return completed


def _row(line: str, prefix: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line[len(prefix) :].strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value.rstrip(",.")
    return fields


def _rows(output: str, prefix: str) -> list[dict[str, str]]:
    result = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position >= 0:
            normalized = line[position:].split(" | ", 1)[0].rstrip(". ")
            result.append(_row(normalized, prefix))
    return result


def _number(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key], 10)
    except (KeyError, ValueError) as error:
        _fail(f"missing/non-decimal {key!r} in {row}")
        raise AssertionError from error


def _marker_lines(output: str) -> tuple[str, ...]:
    prefixes = (
        _STATUS,
        _PROBE,
        _MEMORY,
        _CE,
        _SIM,
        _HOST,
        _HOSTSIG,
        _P5,
        _P5_TIMING,
        _COLL,
        _DRAIN,
        _D2D_TYPE,
        _D2D_BEHA,
        _D2D_LINK,
    )
    result = []
    for line in output.splitlines():
        positions = tuple(line.find(prefix) for prefix in prefixes)
        positions = tuple(position for position in positions if position >= 0)
        if positions:
            normalized = line[min(positions) :].split(" | ", 1)[0].rstrip(". ")
            result.append(normalized)
    return tuple(result)


def _leaf_fragments(case: TrainForwardCase) -> tuple[CommandFragment, ...]:
    leaves = tuple(
        item.fragment if isinstance(item, RegionManifest) else item
        for item in case.linked.manifest.fragments
    )
    if any(type(item) is not CommandFragment for item in leaves):
        _fail("linked Train contains a non-command leaf")
    return leaves


def _opcode_counter(case: TrainForwardCase) -> Counter[str]:
    return Counter(
        record.opcode.name
        for leaf in _leaf_fragments(case)
        for stream in leaf.core_streams
        for record in stream.records
    )


def _artifact_evidence(case: TrainForwardCase) -> TrainForwardArtifactEvidence:
    leaves = _leaf_fragments(case)
    records = tuple(
        record
        for leaf in leaves
        for stream in leaf.core_streams
        for record in stream.records
    )
    opcodes = Counter(record.opcode for record in records)
    runtime_relocations = sum(
        len(stream.runtime_relocations)
        for leaf in leaves
        for stream in leaf.core_streams
    )
    address_relocations = sum(
        len(stream.address_relocations)
        for leaf in leaves
        for stream in leaf.core_streams
    )
    manifest = case.linked.manifest
    return TrainForwardArtifactEvidence(
        len({record.source_global_action_id for record in records}),
        len(manifest.fragments),
        len(records),
        runtime_relocations,
        address_relocations,
        runtime_relocations + address_relocations,
        len(manifest.address_operand_bindings),
        len(manifest.state_operand_bindings),
        tuple(
            TrainForwardOpcodeCount(opcode, opcodes[opcode])
            for opcode in sorted(opcodes, key=int)
        ),
    )


def _work_evidence(case: TrainForwardCase) -> TrainForwardWorkEvidence:
    oracle = case.oracle
    leaves = _leaf_fragments(case)
    observed_send_bytes = sum(
        next(
            operand.literal_value
            for operand in record.operands
            if operand.name == "length_bytes"
        )
        for leaf in leaves
        for stream in leaf.core_streams
        for record in stream.records
        if record.opcode is RecordOpcode.DTE_SEND
    )
    return TrainForwardWorkEvidence(
        oracle.parameters.unique_tensor_count,
        oracle.parameters.unique_bytes,
        oracle.parameters.tp_placed_bytes,
        oracle.parameters.dp_replicated_bytes,
        oracle.gemm_flops_per_microbatch,
        oracle.attention_flops_per_microbatch,
        oracle.logical_forward_flops_per_microbatch,
        oracle.rank_forward_flops_per_microbatch,
        oracle.cluster_forward_flops_per_step,
        oracle.collectives.node_count,
        oracle.collectives.node_count
        * oracle.collectives.logical_tensor_bytes_per_node
        * oracle.dp_degree,
        observed_send_bytes,
    )


def _validate_static_case(case: TrainForwardCase) -> None:
    artifact = _artifact_evidence(case)
    manifest = case.linked.manifest
    if (
        artifact.action_count != 308
        or artifact.fragment_count != 172
        or artifact.record_count != 996
        or artifact.runtime_relocation_count != 288
        or artifact.address_relocation_count != 1652
        or artifact.relocation_count != 1940
        or artifact.address_operand_binding_count != 1592
        or artifact.state_operand_binding_count != 60
        or dict(_opcode_counter(case)) != _EXPECTED_OPCODES
        or tuple(item.runtime_core_id for item in manifest.core_bindings)
        != _ACTIVE_CORES
        or _work_evidence(case).collective_observed_send_bytes != 4096
    ):
        _fail("formal Train graph/artifact counts changed")
    envelope = manifest.envelope
    if (
        envelope.active_cores
        != tuple(item.logical_core for item in manifest.core_bindings)
        or envelope.terminal_cores != envelope.active_cores
        or envelope.expected_ack_cores != envelope.active_cores
        or envelope.expected_done_cores != envelope.active_cores
        or len(envelope.start_events) != 4
        or envelope.empty_core_ack_policy
        is not EmptyCoreAckPolicy.INCLUDE_EMPTY
        or envelope.failure_policy is not ProgramFailurePolicy.ABORT_ALL
        or len(manifest.core_streams) != 4
    ):
        _fail("unified four-core control envelope changed")


def _build_actual_sha_program_io(
    case: TrainForwardCase, artifact_sha256: str
) -> ProgramIoContract:
    state_seeds, state_expected = build_deterministic_timing_state_overrides(
        case.linked
    )
    if len(state_seeds) != 30 or state_expected:
        _fail("Train logical state override coverage changed")
    result = build_timing_program_io(
        case.linked,
        artifact_sha256,
        state_seed_overrides=state_seeds,
    )
    result.validate_against(case.linked.manifest)
    if (
        len(result.initializations) != 328
        or len(result.output_probes) != 4
        or sum(
            type(item.target) is ProgramHbmTarget
            for item in result.initializations
        )
        != 60
        or sum(
            type(item.target) is ProgramSramTarget
            for item in result.initializations
        )
        != 268
        or any(type(item.target) is not ProgramSramTarget for item in result.output_probes)
        or any(item.target.dtype is not DType.FP32 for item in result.output_probes)
    ):
        _fail("actual-SHA ProgramIo Train coverage changed")
    return result


def _validate_status(
    output: str, artifact_sha256: str, contract: ProgramIoContract
) -> None:
    rows = _rows(output, _STATUS)
    if tuple(row.get("phase") for row in rows) != (
        "resolved",
        "applied",
        "verify",
    ):
        _fail(f"ProgramIo phase closure changed: {rows}")
    for row in rows:
        if (
            row.get("mode") != "timing"
            or _number(row, "initializations") != 328
            or _number(row, "probes") != 4
            or row.get("pass") != "1"
        ):
            _fail(f"ProgramIo status failed: {row}")
    if any(
        row.get("checksum") != artifact_sha256 for row in rows[:2]
    ):
        _fail("resolved/applied checksum is not the actual artifact SHA")
    if len(contract.initializations) != 328 or len(contract.output_probes) != 4:
        _fail("marker counts disagree with ProgramIo contract")


def _parse_memory(output: str) -> tuple[TrainForwardMemoryEvidence, ...]:
    rows = _rows(output, _MEMORY)
    if tuple(_number(row, "core") for row in rows) != _ACTIVE_CORES:
        _fail(f"PROGRAM_MEMORY core closure changed: {rows}")
    result = tuple(
        TrainForwardMemoryEvidence(
            _number(row, "core"),
            _number(row, "lsu_issued"),
            _number(row, "lsu_completed"),
            _number(row, "lsu_hbm_read_bytes"),
            _number(row, "lsu_hbm_write_bytes"),
            _number(row, "lsu_sram_read_bytes"),
            _number(row, "lsu_sram_write_bytes"),
            _number(row, "lsu_residual"),
            _number(row, "dte_residual"),
        )
        for row in rows
    )
    expected = tuple(
        TrainForwardMemoryEvidence(core, 15, 15, 7328, 0, 0, 7328, 0, 0)
        for core in _ACTIVE_CORES
    )
    if result != expected:
        _fail(f"Train memory accounting changed: {result}")
    return result


def _parse_ce(output: str) -> tuple[TrainForwardCeMarkerEvidence, ...]:
    rows = tuple(
        sorted(
            _rows(output, _CE),
            key=lambda row: _number(row, "core"),
        )
    )
    if tuple(_number(row, "core") for row in rows) != _ACTIVE_CORES:
        _fail(f"TRAIN_CE core closure changed: {rows}")
    result = tuple(
        TrainForwardCeMarkerEvidence(
            _number(row, "core"),
            _number(row, "invocations"),
            _number(row, "rank_rows"),
            _number(row, "label_read_bytes"),
            _number(row, "loss_write_bytes"),
        )
        for row in rows
    )
    expected = tuple(
        TrainForwardCeMarkerEvidence(core, 1, 4, 16, 16)
        for core in _ACTIVE_CORES
    )
    if result != expected:
        _fail(f"CE runtime work changed: {result}")
    return result


def _parse_probes(output: str, contract: ProgramIoContract) -> None:
    rows = _rows(output, _PROBE)
    entries = {item.id: item for item in contract.output_probes}
    blobs = {item.id: item for item in contract.blobs}
    if len(rows) != 4 or len(entries) != 4:
        _fail(f"loss probe multiplicity changed: {rows}")
    seen: set[str] = set()
    for row in rows:
        entry = entries.get(row.get("id", ""))
        if entry is None or type(entry.target) is not ProgramSramTarget:
            _fail(f"unknown/non-SRAM loss probe: {row}")
        expected_sha = blobs[entry.blob_ref].sha256
        if (
            row.get("expected_checksum") != expected_sha
            or row.get("checksum") != expected_sha
            or row.get("valid") != "1"
            or row.get("exact") != "1"
            or row.get("pass") != "1"
            or _number(row, "bytes") != 16
            or _number(row, "core") != entry.target.runtime_core_id
        ):
            _fail(f"loss probe did not close exactly: {row}")
        seen.add(entry.id)
    if seen != set(entries):
        _fail("loss probe IDs do not exactly cover ProgramIo")


def _parse_signature(value: str, arity: int) -> list[tuple[int, ...]]:
    result = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        try:
            parsed = tuple(int(part, 10) for part in item.split(":"))
        except ValueError as error:
            _fail(f"non-decimal HOSTSIG item {item!r}")
            raise AssertionError from error
        if len(parsed) != arity:
            _fail(f"bad HOSTSIG item {item!r}")
        result.append(parsed)
    return sorted(result)


def _parse_control(output: str) -> tuple[int, TrainForwardControlEvidence]:
    simulation = _rows(output, _SIM)
    if len(simulation) != 1:
        _fail("SIM_RESULT must appear exactly once")
    makespan = _number(simulation[0], "makespan_cycles")
    if makespan <= 0 or "[PROTO_WAIT]" in output or "End DONE reception" not in output:
        _fail("simulation/DONE boundary did not close")
    host = _rows(output, _HOST)
    signatures = _rows(output, _HOSTSIG)
    if len(host) != 1 or len(signatures) != 1:
        _fail("HOSTLANE/HOSTSIG must each appear exactly once")
    if (
        _number(host[0], "ack_total") != 8
        or _number(host[0], "done_total") != 4
        or _number(host[0], "mismatch") != 0
    ):
        _fail(f"ACK/DONE totals changed: {host[0]}")
    done = _parse_signature(signatures[0].get("done", ""), 2)
    ack = _parse_signature(signatures[0].get("ack", ""), 3)
    if done != [(core, 1) for core in _ACTIVE_CORES]:
        _fail(f"DONE per-core closure changed: {done}")
    ack_by_core: Counter[int] = Counter()
    for core, _lane, count in ack:
        ack_by_core[core] += count
    if sorted(ack_by_core.items()) != [(core, 2) for core in _ACTIVE_CORES]:
        _fail(f"ACK per-core closure changed: {ack}")
    timing = _rows(output, _P5_TIMING)
    endpoints = _rows(output, _P5)
    if (
        len(timing) != 1
        or _number(timing[0], "residual") != 0
        or tuple(sorted(_number(row, "core") for row in endpoints))
        != _ACTIVE_CORES
        or any(_number(row, "residual") for row in endpoints)
    ):
        _fail("P2P endpoint/timing drain changed")
    collective = _rows(output, _COLL)
    if len(collective) != 1 or any(
        _number(collective[0], key)
        for key in (
            "tree_entries",
            "reduce_nodes",
            "barriers",
            "gather",
            "reduce_rx",
            "endpoints",
            "dte_tokens",
            "event",
        )
    ):
        _fail(f"collective drain changed: {collective}")
    drains = _rows(output, _DRAIN)
    residuals = {
        key: _number(row, key)
        for row in drains
        for key in ("router_residual", "d2d_link_residual")
        if key in row
    }
    if residuals != {"router_residual": 0, "d2d_link_residual": 0}:
        _fail(f"global drain changed: {drains}")
    control = TrainForwardControlEvidence(
        tuple(TrainForwardCoreCount(core, 2) for core in _ACTIVE_CORES),
        tuple(TrainForwardCoreCount(core, 1) for core in _ACTIVE_CORES),
        tuple(
            TrainForwardNamedCount(name, 0)
            for name in ("collective", "global", "p2p", "timing")
        ),
        True,
    )
    return makespan, control


def _parse_d2d(output: str) -> tuple[tuple[int, int, int, int, int], ...]:
    typed = _rows(output, _D2D_TYPE)
    if len(typed) != 1 or _rows(output, _D2D_BEHA):
        _fail("D2D_TYPE exact marker missing or behavioral link present")
    row = typed[0]
    values = {
        key: _number(row, key)
        for key in (
            "request_in",
            "request_out",
            "ack_in",
            "ack_out",
            "data_in",
            "data_out",
        )
    }
    if values != {
        "request_in": 32,
        "request_out": 32,
        "ack_in": 64,
        "ack_out": 64,
        "data_in": 256,
        "data_out": 256,
    }:
        _fail(f"D2D packet counts changed: {values}")
    pattern = re.compile(
        r"\[D2D_LINK\] idx=\d+ die(\d+)->die(\d+) dir=[A-Z?]+ "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)"
    )
    result = tuple(
        sorted(
            (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(4)),
                int(match.group(6)),
                int(match.group(8)),
            )
            for match in pattern.finditer(output)
        )
    )
    expected = (
        (0, 1, 8, 16, 64),
        (1, 0, 8, 16, 64),
        (2, 3, 8, 16, 64),
        (3, 2, 8, 16, 64),
    )
    if result != expected:
        _fail(f"D2D link closure changed: {result}")
    return result


def _observe_runtime(
    output: str, artifact_sha256: str, contract: ProgramIoContract
) -> _RuntimeObservation:
    _validate_status(output, artifact_sha256, contract)
    _parse_probes(output, contract)
    memory = _parse_memory(output)
    ce_markers = _parse_ce(output)
    makespan, control = _parse_control(output)
    d2d_links = _parse_d2d(output)
    lines = _marker_lines(output)
    expected_counts = {
        _STATUS: 3,
        _PROBE: 4,
        _MEMORY: 4,
        _CE: 4,
        _SIM: 1,
        _HOST: 1,
        _HOSTSIG: 1,
        _P5: 4,
        _P5_TIMING: 1,
        _COLL: 1,
        _D2D_TYPE: 1,
        _D2D_LINK: 4,
    }
    for prefix, count in expected_counts.items():
        if sum(line.startswith(prefix.rstrip()) for line in lines) != count:
            _fail(f"marker multiplicity changed for {prefix.strip()}")
    return _RuntimeObservation(
        makespan,
        hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        memory,
        ce_markers,
        control,
        d2d_links,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_report(
    case: TrainForwardCase,
    contract: ProgramIoContract,
    args: argparse.Namespace,
    artifact_size_bytes: int,
    artifact_sha256: str,
    observations: tuple[_RuntimeObservation, _RuntimeObservation],
) -> TrainForwardRuntimeReport:
    artifact = _artifact_evidence(case)
    hbm_initializations = sum(
        item.target.kind is ProgramIoTargetKind.HBM
        for item in contract.initializations
    )
    hbm_probes = sum(
        item.target.kind is ProgramIoTargetKind.HBM
        for item in contract.output_probes
    )
    program_io = TrainForwardProgramIoEvidence(
        contract.mode,
        hbm_initializations,
        len(contract.initializations) - hbm_initializations,
        hbm_probes,
        len(contract.output_probes) - hbm_probes,
        4,
        4,
    )
    first = observations[0]
    repeats = tuple(
        TrainForwardRepeatEvidence(
            index,
            observation.makespan_cycles,
            observation.marker_digest,
            canonical_digest(observation.memory),
            canonical_digest(observation.ce_markers),
            canonical_digest(observation.control),
        )
        for index, observation in enumerate(observations)
    )
    report = TrainForwardRuntimeReport.create(
        spec_digest=canonical_digest(case.spec),
        oracle_id=case.oracle.id,
        oracle_digest=canonical_digest(case.oracle),
        train_linked_id=case.linked.id,
        train_linked_digest=canonical_digest(case.linked),
        linked_manifest_id=case.linked.manifest.id,
        linked_manifest_digest=canonical_digest(case.linked.manifest),
        program_io_id=contract.id,
        program_io_digest=canonical_digest(contract),
        program_artifact_sha256=artifact_sha256,
        artifact_size_bytes=artifact_size_bytes,
        finalizer_sha256=_sha256_file(args.finalizer),
        resolver_sha256=_sha256_file(args.resolver),
        npusim_sha256=_sha256_file(args.npusim),
        hardware_digest=_sha256_file(case.runtime_inputs.source_hardware_path),
        simulation_digest=_sha256_file(args.simulation),
        mapping_digest=_sha256_file(case.runtime_inputs.source_mapping_path),
        artifact=artifact,
        work=_work_evidence(case),
        program_io=program_io,
        memory=first.memory,
        ce_markers=first.ce_markers,
        control=first.control,
        marker_schema_version=TRAIN_FORWARD_MARKER_SCHEMA_VERSION,
        repeat_count=2,
        makespan_cycles=first.makespan_cycles,
        repeats=repeats,
        timing_execution=True,
        train_structure_exact=True,
        analytic_work_exact=True,
        collective_accounting_exact=True,
        program_io_boundary_exact=True,
        compute_functional=False,
        model_functional=False,
    )
    report.validate_against(case.spec, case.oracle, case.linked, contract)
    return report


def _write_report(path: Path, report: TrainForwardRuntimeReport) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"report path must not exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(report))


def _run_case(args: argparse.Namespace) -> None:
    case = build_train_forward_case()
    _validate_static_case(case)
    with tempfile.TemporaryDirectory(
        prefix="train-forward-dp2-tp2-", dir=args.runtime_root
    ) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        sidecar_path = directory / "program_io.json"
        artifacts = (directory / "program.0.npup", directory / "program.1.npup")
        finalizer_reports = (
            directory / "finalizer.0.json",
            directory / "finalizer.1.json",
        )
        manifest_path.write_text(
            canonical_json(case.linked.manifest), encoding="utf-8"
        )
        hardware_path.write_text(case.runtime_inputs.hardware_json, encoding="utf-8")
        mapping_path.write_text(case.runtime_inputs.mapping_text, encoding="utf-8")

        artifact_bytes: list[bytes] = []
        reports: list[dict[str, object]] = []
        for artifact_path, report_path in zip(
            artifacts, finalizer_reports, strict=True
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
                timeout=120,
            )
            artifact_bytes.append(artifact_path.read_bytes())
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        if artifact_bytes[0] != artifact_bytes[1] or reports[0] != reports[1]:
            _fail("finalizer byte/report repeat changed")
        artifact_sha256 = hashlib.sha256(artifact_bytes[0]).hexdigest()
        finalization = reports[0]
        artifact = _artifact_evidence(case)
        if (
            finalization.get("artifact_sha256") != artifact_sha256
            or finalization.get("artifact_bytes") != len(artifact_bytes[0])
            or finalization.get("core_count") != 4
            or finalization.get("record_count") != artifact.record_count
            # The finalizer report counts serialized address relocations;
            # runtime relocations are consumed while assigning runtime symbols.
            or finalization.get("relocation_count")
            != artifact.address_relocation_count
            or finalization.get("linked_manifest_id") != case.linked.manifest.id
            or finalization.get("linked_manifest_digest")
            != canonical_digest(case.linked.manifest)
        ):
            _fail(f"finalizer report closure changed: {finalization}")

        contract = _build_actual_sha_program_io(case, artifact_sha256)
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        resolver = _run(
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(artifacts[0]),
                str(sidecar_path),
            ],
            cwd=args.runtime_root,
            timeout=120,
        )
        if "initializations=328 probes=4" not in resolver.stdout:
            _fail(f"resolver lost ProgramIo counts: {resolver.stdout}")

        observations = []
        for _run_index in range(2):
            execution = _run(
                [
                    str(args.npusim),
                    "--program",
                    str(artifacts[0]),
                    "--linked-manifest",
                    str(manifest_path),
                    "--program-io",
                    str(sidecar_path),
                    "--hardware-config",
                    str(hardware_path),
                    "--simulation-config",
                    str(args.simulation),
                    "--mapping-config",
                    str(mapping_path),
                    "--trace-window",
                    "1000000",
                ],
                cwd=args.runtime_root,
                timeout=args.timeout,
            )
            observations.append(
                _observe_runtime(execution.stdout, artifact_sha256, contract)
            )
        if observations[0] != observations[1]:
            _fail(f"runtime repeat changed: {observations}")
        pair = (observations[0], observations[1])
        report = _build_report(
            case,
            contract,
            args,
            len(artifact_bytes[0]),
            artifact_sha256,
            pair,
        )
        if args.report is not None:
            _write_report(args.report, report)

    print(
        "[TRAIN FORWARD DP2 TP2] PASS: timing_execution=1 "
        "compute_functional=0 model_functional=0 "
        f"artifact={report.artifact_size_bytes}B records=996 "
        f"relocations=1940 init=328 probes=4 repeat=2 "
        f"makespan_cycles={report.makespan_cycles} "
        f"sha256={report.program_artifact_sha256}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", required=True, type=_executable)
    parser.add_argument("--finalizer", required=True, type=_executable)
    parser.add_argument("--resolver", required=True, type=_executable)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    args.simulation = args.simulation.resolve()
    if not args.simulation.is_file():
        parser.error(f"--simulation is not a file: {args.simulation}")
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.report is not None:
        args.report = args.report.absolute()
    _run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
