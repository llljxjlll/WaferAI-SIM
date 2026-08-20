#!/usr/bin/env python3
"""Run exact Stage3 prefill, decode, and mixed timing profiles."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.capability import (  # noqa: E402
    CapabilityStatus,
)
from llm.frontend.wafer_frontend.schema.experiment import (  # noqa: E402
    InferOutput,
)
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
from llm.frontend.wafer_frontend.schema.stage2_dense_forward_evidence import (  # noqa: E402
    Stage2DenseForwardArtifactEvidence,
    Stage2DenseForwardCompileEvidence,
    Stage2DenseForwardControlEvidence,
    Stage2DenseForwardCoreCount,
    Stage2DenseForwardD2DEvidence,
    Stage2DenseForwardMemoryEvidence,
    Stage2DenseForwardNamedCount,
    Stage2DenseForwardOpcodeCount,
    Stage2DenseForwardProbeEvidence,
    Stage2DenseForwardRepeatEvidence,
    Stage2DenseForwardSidecarEvidence,
    Stage2DenseForwardToolEvidence,
)
from llm.frontend.wafer_frontend.schema.stage3_static_profile_evidence import (  # noqa: E402
    STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
    STAGE3_STATIC_PROFILE_MARKER_SCHEMA_VERSION,
    Stage3StaticAttentionEvidence,
    Stage3StaticProfileRuntimeReport,
)
from stage3_decode_cases import (  # noqa: E402
    Stage3StaticCaseKind,
    build_stage3_static_case,
)

_STATUS = "[PROGRAM_IO] "
_PROBE = "[PROGRAM_IO_PROBE] "
_MEMORY = "[PROGRAM_MEMORY] "
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


@dataclass(frozen=True)
class _Golden:
    action_count: int
    leaf_count: int
    record_count: int
    address_binding_count: int
    opcode_counts: tuple[tuple[str, int], ...]
    initialization_count: int
    probe_count: int
    lsu_load_count: int
    lsu_store_count: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    attention_pairs_per_layer: int
    attention_read_bytes_per_layer: int
    attention_write_bytes_per_layer: int


def _opcodes(**values: int) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(values.items()))


_GOLDENS = {
    Stage3StaticCaseKind.PREFILL: _Golden(
        44, 44, 159, 278,
        _opcodes(
            ATTENTION_EXACT=2, EMBEDDING_LOOKUP=1, LSU_LOAD=15,
            LSU_STORE=4, MATMUL=9, RESIDUAL=4, RMSNORM=5,
            ROPE_QK_EXACT=2, SRAM_ALLOC_AT=45, SRAM_BIND=25,
            SRAM_FREE=45, SWIGLU=2,
        ),
        60, 1, 15, 4, 12448, 1024, 36, 0, 512,
    ),
    Stage3StaticCaseKind.DECODE: _Golden(
        104, 104, 275, 422,
        _opcodes(
            ATTENTION_EXACT=2, EMBEDDING_LOOKUP=1, LSU_LOAD=47,
            LSU_STORE=32, MATMUL=9, RESIDUAL=4, RMSNORM=5,
            ROPE_QK_EXACT=2, SRAM_ALLOC_AT=73, SRAM_BIND=25,
            SRAM_FREE=73, SWIGLU=2,
        ),
        120, 33, 47, 32, 30880, 1024, 144, 9216, 512,
    ),
    Stage3StaticCaseKind.MIXED: _Golden(
        80, 80, 235, 374,
        _opcodes(
            ATTENTION_EXACT=2, EMBEDDING_LOOKUP=1, LSU_LOAD=31,
            LSU_STORE=24, MATMUL=9, RESIDUAL=4, RMSNORM=5,
            ROPE_QK_EXACT=2, SRAM_ALLOC_AT=65, SRAM_BIND=25,
            SRAM_FREE=65, SWIGLU=2,
        ),
        96, 17, 31, 24, 17568, 1024, 47, 2560, 512,
    ),
}


@dataclass(frozen=True)
class _ReviewedRuntimeWitness:
    artifact_bytes: int
    record_count: int
    relocation_count: int
    artifact_sha256: str
    makespan_cycles: int
    marker_digest: str


_REVIEWED_RUNTIME_GOLDENS = {
    Stage3StaticCaseKind.PREFILL: _ReviewedRuntimeWitness(
        22258,
        159,
        297,
        "4b1204fa698553e202d97a77dc3fdb4f60f38051b1a82a4dca84ec4981d76777",
        5805,
        "f8fa4a291d691110f634368275564ad99e8f62e606b9e59bfbe8993e4f7ed106",
    ),
    Stage3StaticCaseKind.MIXED: _ReviewedRuntimeWitness(
        32714,
        235,
        429,
        "60af99215ab929d5c107a5f09aaebdfde6a9bed53f1ccdac317ac9ff6a7ad22e",
        8807,
        "8ab5b8d1a1efb23a899278d309d27c3ec9caa63b16f9ba64f528a165dfad060f",
    ),
    Stage3StaticCaseKind.DECODE: _ReviewedRuntimeWitness(
        37818,
        275,
        501,
        "8cd6f01aada63d8091b7ceb9c70726c7287010d63a5b8c2362a905cf51cc5543",
        13081,
        "00d6dde839fcc3a3fd1a4b24b42808ec8b325fba97f743746519f38a2eef2678",
    ),
}


@dataclass(frozen=True)
class _RuntimeObservation:
    makespan_cycles: int
    marker_digest: str
    memory: tuple[Stage2DenseForwardMemoryEvidence, ...]
    probes: tuple[Stage2DenseForwardProbeEvidence, ...]
    control: Stage2DenseForwardControlEvidence
    d2d: Stage2DenseForwardD2DEvidence


def _fail(kind: Stage3StaticCaseKind, message: str) -> None:
    raise RuntimeError(f"[STAGE3 {kind.value.upper()}] FAIL: {message}")


def _validate_reviewed_runtime(
    kind: Stage3StaticCaseKind,
    observed: _ReviewedRuntimeWitness,
) -> None:
    wanted = _REVIEWED_RUNTIME_GOLDENS[kind]
    if observed != wanted:
        _fail(
            kind,
            f"reviewed runtime six-field golden changed: {observed!r} != {wanted!r}",
        )


def _executable(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _run(
    kind: Stage3StaticCaseKind,
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
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
    if completed.returncode != 0:
        _fail(
            kind,
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}",
        )
    return completed


def _row(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line[len(prefix) :].strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value.rstrip(",.")
    return result


def _rows(output: str, prefix: str) -> list[dict[str, str]]:
    result = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position >= 0:
            normalized = line[position:].split(" | ", 1)[0].rstrip(". ")
            result.append(_row(normalized, prefix))
    return result


def _number(
    kind: Stage3StaticCaseKind,
    row: dict[str, str],
    key: str,
) -> int:
    try:
        return int(row[key], 10)
    except (KeyError, ValueError) as error:
        _fail(kind, f"missing/non-decimal {key!r} in {row}")
        raise AssertionError from error


def _marker_lines(output: str) -> tuple[str, ...]:
    prefixes = (
        _STATUS, _PROBE, _MEMORY, _SIM, _HOST, _HOSTSIG, _P5,
        _P5_TIMING, _COLL, _DRAIN, _D2D_TYPE, _D2D_BEHA, _D2D_LINK,
    )
    result = []
    for line in output.splitlines():
        positions = tuple(
            position
            for prefix in prefixes
            if (position := line.find(prefix)) >= 0
        )
        if positions:
            normalized = line[min(positions) :].split(" | ", 1)[0].rstrip(". ")
            result.append(normalized)
    return tuple(result)


def _literal_map(record: object) -> dict[str, object]:
    return {
        operand.name: operand.literal_value
        for operand in record.operands  # type: ignore[attr-defined]
        if operand.literal_value is not None
    }


def _validate_static(kind: Stage3StaticCaseKind, case: object) -> None:
    expected = _GOLDENS[kind]
    case.policy.validate_against(  # type: ignore[attr-defined]
        case.planning_context,  # type: ignore[attr-defined]
        case.scheduling_context,  # type: ignore[attr-defined]
    )
    leaves = case.profile.leaf_fragments  # type: ignore[attr-defined]
    if any(type(item) is not CommandFragment for item in leaves):
        _fail(kind, "manifest contains a non-command leaf")
    opcodes = Counter(
        record.opcode.name
        for leaf in leaves
        for stream in leaf.core_streams
        for record in stream.records
    )
    observed = (
        len(case.global_dag.actions),  # type: ignore[attr-defined]
        len(leaves),
        sum(opcodes.values()),
        len(case.manifest.address_operand_bindings),  # type: ignore[attr-defined]
        tuple(sorted(opcodes.items())),
        len(case.program_io.initializations),  # type: ignore[attr-defined]
        len(case.program_io.output_probes),  # type: ignore[attr-defined]
    )
    wanted = (
        expected.action_count,
        expected.leaf_count,
        expected.record_count,
        expected.address_binding_count,
        expected.opcode_counts,
        expected.initialization_count,
        expected.probe_count,
    )
    if observed != wanted:
        _fail(kind, f"static closure changed: {observed!r} != {wanted!r}")
    attention = tuple(
        record
        for leaf in leaves
        for stream in leaf.core_streams
        for record in stream.records
        if record.opcode is RecordOpcode.ATTENTION_EXACT
    )
    wanted_attention = {
        "mode": 2,
        "query_tokens": 8,
        "context_sum": case.static_profile.key.context_sum,  # type: ignore[attr-defined]
        "context_max": case.static_profile.key.context_max,  # type: ignore[attr-defined]
        "query_key_pairs": expected.attention_pairs_per_layer,
        "rank_kv_read_bytes": expected.attention_read_bytes_per_layer,
        "rank_kv_write_bytes": expected.attention_write_bytes_per_layer,
    }
    if len(attention) != 2 or any(
        any(
            _literal_map(record).get(key) != value
            for key, value in wanted_attention.items()
        )
        for record in attention
    ):
        _fail(kind, f"ATTENTION exact-profile literals changed: {attention}")


def _rebuild_sidecar(
    kind: Stage3StaticCaseKind,
    case: object,
    artifact_sha256: str,
) -> ProgramIoContract:
    seeds, expected = build_deterministic_timing_state_overrides(case.profile)  # type: ignore[attr-defined]
    if (
        tuple(sorted(seeds)) != case.state_seed_refs  # type: ignore[attr-defined]
        or tuple(sorted(expected)) != case.state_expected_refs  # type: ignore[attr-defined]
    ):
        _fail(kind, "state override refs changed")
    result = build_timing_program_io(
        case.profile,  # type: ignore[attr-defined]
        artifact_sha256,
        state_seed_overrides=seeds,
        state_expected_overrides=expected,
    )
    result.validate_against(case.manifest)  # type: ignore[attr-defined]
    placeholder = case.program_io  # type: ignore[attr-defined]
    semantic_fields = (
        "schema_version", "producer_pass", "mode", "source_linked_manifest_id",
        "source_linked_manifest_digest", "blobs", "initializations", "output_probes",
    )
    if tuple(getattr(result, field) for field in semantic_fields) != tuple(
        getattr(placeholder, field) for field in semantic_fields
    ):
        _fail(kind, "actual-SHA ProgramIo semantics changed")
    return result


def _validate_status(
    kind: Stage3StaticCaseKind,
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
) -> None:
    rows = _rows(output, _STATUS)
    if tuple(row.get("phase") for row in rows) != (
        "resolved", "applied", "verify"
    ):
        _fail(kind, f"ProgramIo phase closure changed: {rows}")
    for row in rows:
        if (
            row.get("mode") != "timing"
            or _number(kind, row, "initializations")
            != len(contract.initializations)
            or _number(kind, row, "probes") != len(contract.output_probes)
            or row.get("pass") != "1"
        ):
            _fail(kind, f"ProgramIo status failed: {row}")
    if (
        rows[0].get("checksum") != artifact_sha256
        or rows[1].get("checksum") != artifact_sha256
    ):
        _fail(kind, "resolved/applied checksum is not the artifact SHA")


def _parse_probes(
    kind: Stage3StaticCaseKind,
    output: str,
    contract: ProgramIoContract,
    leaves: tuple[CommandFragment, ...],
) -> tuple[Stage2DenseForwardProbeEvidence, ...]:
    rows = _rows(output, _PROBE)
    expected = _GOLDENS[kind]
    if len(rows) != expected.probe_count:
        _fail(kind, f"probe marker multiplicity changed: {len(rows)}")
    entries = {entry.id: entry for entry in contract.output_probes}
    blobs = {blob.id: blob for blob in contract.blobs}
    state_abis = {
        abi.id: abi
        for fragment in leaves
        for abi in fragment.state_abi
    }
    result: list[Stage2DenseForwardProbeEvidence] = []
    for row in rows:
        entry = entries.get(row.get("id", ""))
        if entry is None:
            _fail(kind, f"unknown probe marker: {row}")
        expected_sha = blobs[entry.blob_ref].sha256
        if (
            row.get("expected_checksum") != expected_sha
            or row.get("checksum") != expected_sha
            or row.get("valid") != "1"
            or row.get("exact") != "1"
            or row.get("pass") != "1"
            or _number(kind, row, "bytes") != entry.length_bytes
        ):
            _fail(kind, f"probe did not close exactly: {row}")
        if type(entry.target) is ProgramSramTarget:
            if _number(kind, row, "core") != entry.target.runtime_core_id:
                _fail(kind, f"SRAM probe core changed: {row}")
            target_kind = ProgramIoTargetKind.SRAM
        elif type(entry.target) is ProgramHbmTarget:
            abi = state_abis.get(entry.target.state_abi_id)
            if abi is None or (
                _number(kind, row, "die") != abi.die_id
                or _number(kind, row, "address")
                != abi.address + entry.offset_bytes
            ):
                _fail(kind, f"HBM probe location changed: {row}")
            target_kind = ProgramIoTargetKind.HBM
        else:
            _fail(kind, f"unsupported probe target: {entry.target}")
        result.append(
            Stage2DenseForwardProbeEvidence(
                probe_id=entry.id,
                target_kind=target_kind,
                length_bytes=entry.length_bytes,
                expected_sha256=expected_sha,
                actual_sha256=row.get("checksum", ""),
                all_bytes_valid=row.get("valid") == "1",
                exact_match=row.get("exact") == "1",
                passed=row.get("pass") == "1",
            )
        )
    if set(entries) != {item.probe_id for item in result}:
        _fail(kind, "runtime omitted a ProgramIo probe")
    return tuple(sorted(result, key=lambda item: item.probe_id))


def _parse_memory(
    kind: Stage3StaticCaseKind,
    output: str,
) -> tuple[Stage2DenseForwardMemoryEvidence, ...]:
    expected = _GOLDENS[kind]
    rows = _rows(output, _MEMORY)
    if len(rows) != 1 or _number(kind, rows[0], "core") != 0:
        _fail(kind, f"PROGRAM_MEMORY core closure changed: {rows}")
    row = rows[0]
    result = tuple(
        _number(kind, row, key)
        for key in (
            "lsu_issued", "lsu_completed", "lsu_hbm_read_bytes",
            "lsu_hbm_write_bytes", "lsu_sram_read_bytes",
            "lsu_sram_write_bytes", "lsu_residual", "dte_residual",
        )
    )
    transfer_count = expected.lsu_load_count + expected.lsu_store_count
    wanted = (
        transfer_count,
        transfer_count,
        expected.hbm_read_bytes,
        expected.hbm_write_bytes,
        expected.hbm_write_bytes,
        expected.hbm_read_bytes,
        0,
        0,
    )
    if result != wanted:
        _fail(kind, f"PROGRAM_MEMORY changed: {result!r} != {wanted!r}")
    return (Stage2DenseForwardMemoryEvidence(0, *result),)


def _validate_control(
    kind: Stage3StaticCaseKind,
    output: str,
) -> tuple[int, Stage2DenseForwardControlEvidence]:
    simulations = _rows(output, _SIM)
    if len(simulations) != 1:
        _fail(kind, "SIM_RESULT must appear exactly once")
    makespan = _number(kind, simulations[0], "makespan_cycles")
    if makespan <= 0 or "[PROTO_WAIT]" in output or "End DONE reception" not in output:
        _fail(kind, "simulation/DONE path did not close")
    host = _rows(output, _HOST)
    signatures = _rows(output, _HOSTSIG)
    if len(host) != 1 or len(signatures) != 1:
        _fail(kind, "HOSTLANE/HOSTSIG must each appear exactly once")
    if (
        _number(kind, host[0], "ack_total") != 2
        or _number(kind, host[0], "done_total") != 1
        or _number(kind, host[0], "mismatch") != 0
        or signatures[0].get("done") != "0:1"
        or signatures[0].get("ack") != "0:0:2"
    ):
        _fail(kind, f"ACK/DONE closure changed: {host} {signatures}")
    timing = _rows(output, _P5_TIMING)
    if len(timing) != 1 or _number(kind, timing[0], "residual") != 0:
        _fail(kind, f"P2P timing drain changed: {timing}")
    if _rows(output, _P5):
        _fail(kind, "TP1 static cases must not emit P2P endpoint drains")
    collective = _rows(output, _COLL)
    if len(collective) != 1 or any(
        _number(kind, collective[0], key)
        for key in (
            "tree_entries", "reduce_nodes", "barriers", "gather",
            "reduce_rx", "endpoints", "dte_tokens", "event",
        )
    ):
        _fail(kind, f"collective drain changed: {collective}")
    drains = _rows(output, _DRAIN)
    residuals = {
        key: _number(kind, row, key)
        for row in drains
        for key in ("router_residual", "d2d_link_residual")
        if key in row
    }
    if residuals != {"router_residual": 0, "d2d_link_residual": 0}:
        _fail(kind, f"global drain changed: {drains}")
    return makespan, Stage2DenseForwardControlEvidence(
        ack_counts=(Stage2DenseForwardCoreCount(0, 2),),
        done_counts=(Stage2DenseForwardCoreCount(0, 1),),
        drain_residuals=tuple(
            Stage2DenseForwardNamedCount(name, 0)
            for name in ("collective", "global", "p2p", "timing")
        ),
        all_done_boundary_reached=True,
    )


def _validate_d2d(
    kind: Stage3StaticCaseKind,
    output: str,
) -> Stage2DenseForwardD2DEvidence:
    rows = _rows(output, _D2D_TYPE)
    if len(rows) != 1:
        _fail(kind, "D2D_TYPE must appear exactly once")
    if any(
        _number(kind, rows[0], key)
        for key in (
            "request_in", "request_out", "ack_in", "ack_out",
            "data_in", "data_out",
        )
    ):
        _fail(kind, f"TP1 D2D counts must be zero: {rows[0]}")
    if _rows(output, _D2D_BEHA):
        _fail(kind, "canonical Stage3 hardware unexpectedly emitted D2D_BEHA")
    for row in _rows(output, _D2D_LINK):
        if any(
            _number(kind, row, key)
            for key in (
                "req_in", "req_out", "ack_in", "ack_out",
                "data_in", "data_out",
            )
        ):
            _fail(kind, f"inactive D2D link is non-zero: {row}")
    return Stage2DenseForwardD2DEvidence(0, 0, 0, 0, 0, 0, 0, ())


def _observe(
    kind: Stage3StaticCaseKind,
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
    leaves: tuple[CommandFragment, ...],
) -> _RuntimeObservation:
    _validate_status(kind, output, artifact_sha256, contract)
    memory = _parse_memory(kind, output)
    probes = _parse_probes(kind, output, contract, leaves)
    makespan, control = _validate_control(kind, output)
    d2d = _validate_d2d(kind, output)
    markers = _marker_lines(output)
    marker_digest = hashlib.sha256(
        ("\n".join(markers) + "\n").encode("utf-8")
    ).hexdigest()
    return _RuntimeObservation(
        makespan, marker_digest, memory, probes, control, d2d
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_runtime_report(
    kind: Stage3StaticCaseKind,
    case: object,
    args: argparse.Namespace,
    finalization: dict[str, object],
    artifact_size_bytes: int,
    artifact_sha256: str,
    contract: ProgramIoContract,
    observations: tuple[_RuntimeObservation, _RuntimeObservation],
) -> Stage3StaticProfileRuntimeReport:
    leaves = case.profile.leaf_fragments  # type: ignore[attr-defined]
    opcode_counter = Counter(
        record.opcode.name
        for leaf in leaves
        for stream in leaf.core_streams
        for record in stream.records
    )
    opcode_counts = tuple(
        Stage2DenseForwardOpcodeCount(RecordOpcode[name], count)
        for name, count in sorted(
            opcode_counter.items(), key=lambda item: int(RecordOpcode[item[0]])
        )
    )
    attention_records = tuple(
        record
        for leaf in leaves
        for stream in leaf.core_streams
        for record in stream.records
        if record.opcode is RecordOpcode.ATTENTION_EXACT
    )
    if not attention_records:
        _fail(kind, "typed report requires ATTENTION_EXACT records")
    literals = tuple(_literal_map(record) for record in attention_records)
    first_literals = literals[0]
    keys = (
        "query_tokens",
        "context_sum",
        "context_max",
        "query_key_pairs",
        "rank_kv_read_bytes",
        "rank_kv_write_bytes",
    )
    if any(tuple(item[key] for key in keys) != tuple(first_literals[key] for key in keys)
           for item in literals):
        _fail(kind, "Attention records disagree within the profile")
    first = observations[0]
    repeats = tuple(
        Stage2DenseForwardRepeatEvidence(
            run_index=index,
            makespan_cycles=observation.makespan_cycles,
            marker_digest=observation.marker_digest,
            memory_digest=canonical_digest(observation.memory),
            probe_digest=canonical_digest(observation.probes),
            control_digest=canonical_digest(observation.control),
            d2d_digest=canonical_digest(observation.d2d),
        )
        for index, observation in enumerate(observations)
    )
    report = Stage3StaticProfileRuntimeReport.create(
        baseline_epoch=STAGE3_STATIC_PROFILE_BASELINE_EPOCH,
        profile_mode=case.static_profile.mode,  # type: ignore[attr-defined]
        tp_degree=1,
        infer_output=InferOutput.LOGITS,
        capability_status=CapabilityStatus.E2E_TIMING,
        static_profile_id=case.static_profile.id,  # type: ignore[attr-defined]
        static_profile_digest=canonical_digest(
            case.static_profile  # type: ignore[attr-defined]
        ),
        oracle_id=case.oracle.id,  # type: ignore[attr-defined]
        oracle_digest=canonical_digest(case.oracle),  # type: ignore[attr-defined]
        policy=case.policy,  # type: ignore[attr-defined]
        compile=Stage2DenseForwardCompileEvidence(
            template_id=case.template.id,  # type: ignore[attr-defined]
            template_digest=canonical_digest(case.template),  # type: ignore[attr-defined]
            ir1_id=case.graph.id,  # type: ignore[attr-defined]
            ir1_digest=canonical_digest(case.graph),  # type: ignore[attr-defined]
            global_dag_id=case.global_dag.id,  # type: ignore[attr-defined]
            global_dag_digest=canonical_digest(case.global_dag),  # type: ignore[attr-defined]
            lowered_id=case.lowered.id,  # type: ignore[attr-defined]
            lowered_digest=canonical_digest(case.lowered),  # type: ignore[attr-defined]
        ),
        tools=Stage2DenseForwardToolEvidence(
            finalizer_sha256=_sha256_file(args.finalizer),
            resolver_sha256=_sha256_file(args.resolver),
            npusim_sha256=_sha256_file(args.npusim),
        ),
        hardware_digest=hashlib.sha256(
            case.runtime_hardware_inputs.hardware_json.encode("utf-8")  # type: ignore[attr-defined]
        ).hexdigest(),
        simulation_digest=_sha256_file(args.simulation),
        mapping_digest=hashlib.sha256(
            case.runtime_hardware_inputs.mapping_text.encode("utf-8")  # type: ignore[attr-defined]
        ).hexdigest(),
        artifact=Stage2DenseForwardArtifactEvidence(
            linked_manifest_id=case.manifest.id,  # type: ignore[attr-defined]
            linked_manifest_digest=canonical_digest(case.manifest),  # type: ignore[attr-defined]
            program_artifact_sha256=artifact_sha256,
            artifact_size_bytes=artifact_size_bytes,
            action_count=len(case.global_dag.actions),  # type: ignore[attr-defined]
            leaf_fragment_count=len(leaves),
            record_count=int(finalization["record_count"]),
            address_binding_count=len(
                case.manifest.address_operand_bindings  # type: ignore[attr-defined]
            ),
            relocation_count=int(finalization["relocation_count"]),
            opcode_counts=opcode_counts,
        ),
        sidecar=Stage2DenseForwardSidecarEvidence(
            contract_id=contract.id,
            contract_digest=canonical_digest(contract),
            mode=contract.mode,
            hbm_initialization_count=sum(
                type(item.target) is ProgramHbmTarget
                for item in contract.initializations
            ),
            sram_initialization_count=sum(
                type(item.target) is ProgramSramTarget
                for item in contract.initializations
            ),
            hbm_probe_count=sum(
                type(item.target) is ProgramHbmTarget
                for item in contract.output_probes
            ),
            sram_probe_count=sum(
                type(item.target) is ProgramSramTarget
                for item in contract.output_probes
            ),
        ),
        attention=Stage3StaticAttentionEvidence(
            record_count=len(attention_records),
            query_tokens_per_record=int(first_literals["query_tokens"]),
            context_sum_per_record=int(first_literals["context_sum"]),
            context_max_per_record=int(first_literals["context_max"]),
            query_key_pairs_per_record=int(first_literals["query_key_pairs"]),
            rank_kv_read_bytes_per_record=int(
                first_literals["rank_kv_read_bytes"]
            ),
            rank_kv_write_bytes_per_record=int(
                first_literals["rank_kv_write_bytes"]
            ),
        ),
        memory=first.memory,
        probes=first.probes,
        control=first.control,
        d2d=first.d2d,
        marker_schema_version=STAGE3_STATIC_PROFILE_MARKER_SCHEMA_VERSION,
        repeat_count=len(observations),
        makespan_cycles=first.makespan_cycles,
        repeats=repeats,
        timing_execution=True,
        dense_forward_structure_exact=True,
        static_request_shape_exact=True,
        analytic_work_exact=True,
        program_io_boundary_exact=True,
        traffic_accounting_exact=True,
        compute_functional=False,
        model_functional=False,
    )
    report.validate_against(case.oracle)  # type: ignore[attr-defined]
    return report


def _expected_report_names(
    kind: Stage3StaticCaseKind,
) -> tuple[str, ...]:
    stem = f"stage3-{kind.value}"
    return tuple(
        sorted(
            (
                f"{stem}.finalization.json",
                f"{stem}.finalizer.0.log",
                f"{stem}.finalizer.1.log",
                f"{stem}.input_digests.json",
                f"{stem}.linked_manifest.json",
                f"{stem}.markers.json",
                f"{stem}.oracle.json",
                f"{stem}.program_io.json",
                f"{stem}.resolver.log",
                f"{stem}.runtime.0.log",
                f"{stem}.runtime.1.log",
                f"{stem}.runtime.json",
            )
        )
    )


def _report_text_entries(
    kind: Stage3StaticCaseKind,
    *,
    case: object,
    runtime_report: Stage3StaticProfileRuntimeReport,
    contract: ProgramIoContract,
    finalization: dict[str, object],
    observations: tuple[_RuntimeObservation, _RuntimeObservation],
    finalizer_logs: tuple[str, str],
    resolver_log: str,
    runtime_logs: tuple[str, str],
) -> tuple[tuple[str, str], ...]:
    if len(finalizer_logs) != 2 or len(runtime_logs) != 2:
        _fail(kind, "report evidence requires exactly two repeat logs")
    if not resolver_log:
        _fail(kind, "resolver evidence log must be non-empty")
    for index, log in enumerate(runtime_logs):
        if _SIM not in log or _D2D_TYPE not in log:
            _fail(kind, f"runtime evidence log {index} lacks markers")
    stem = f"stage3-{kind.value}"
    markers = {
        "schema_version": STAGE3_STATIC_PROFILE_MARKER_SCHEMA_VERSION,
        "runs": tuple(
            {
                "run_index": index,
                "makespan_cycles": observation.makespan_cycles,
                "marker_digest": observation.marker_digest,
                "memory": observation.memory,
                "probes": observation.probes,
                "control": observation.control,
                "d2d": observation.d2d,
            }
            for index, observation in enumerate(observations)
        ),
    }
    inputs = {
        "schema_version": "wafer_frontend.stage3_static_profile_inputs/v1",
        "tools": runtime_report.tools,
        "hardware_digest": runtime_report.hardware_digest,
        "simulation_digest": runtime_report.simulation_digest,
        "mapping_digest": runtime_report.mapping_digest,
        "static_profile_digest": runtime_report.static_profile_digest,
        "policy_selection_digests": tuple(
            canonical_digest(selection)
            for selection in runtime_report.policy.selections
        ),
        "template_digest": runtime_report.compile.template_digest,
        "ir1_digest": runtime_report.compile.ir1_digest,
        "global_dag_digest": runtime_report.compile.global_dag_digest,
        "lowered_digest": runtime_report.compile.lowered_digest,
    }
    entries = (
        (f"{stem}.finalization.json", canonical_json(finalization)),
        (f"{stem}.finalizer.0.log", finalizer_logs[0]),
        (f"{stem}.finalizer.1.log", finalizer_logs[1]),
        (f"{stem}.input_digests.json", canonical_json(inputs)),
        (
            f"{stem}.linked_manifest.json",
            canonical_json(case.manifest),  # type: ignore[attr-defined]
        ),
        (f"{stem}.markers.json", canonical_json(markers)),
        (
            f"{stem}.oracle.json",
            canonical_json(case.oracle),  # type: ignore[attr-defined]
        ),
        (f"{stem}.program_io.json", canonical_json(contract)),
        (f"{stem}.resolver.log", resolver_log),
        (f"{stem}.runtime.0.log", runtime_logs[0]),
        (f"{stem}.runtime.1.log", runtime_logs[1]),
        (f"{stem}.runtime.json", canonical_json(runtime_report)),
    )
    names = tuple(sorted(name for name, _value in entries))
    if (
        names != _expected_report_names(kind)
        or len(names) != len(set(names))
        or any(Path(name).name != name or name.endswith(".npup") for name in names)
        or any(type(value) is not str for _name, value in entries)
    ):
        _fail(kind, "report entries must be unique flat text without NPUP")
    return tuple(sorted(entries))


def _publish_report_texts(
    root: Path,
    kind: Stage3StaticCaseKind,
    entries: tuple[tuple[str, str], ...],
) -> None:
    if root.is_symlink():
        _fail(kind, f"report root must not be a symlink: {root}")
    if root.exists():
        _fail(kind, f"report root must not already exist: {root}")
    if tuple(name for name, _value in entries) != _expected_report_names(kind):
        _fail(kind, "report evidence set is not the exact reviewed set")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.staging-", dir=root.parent
    ) as raw:
        staging = Path(raw)
        for name, value in entries:
            target = staging / name
            if target.exists() or target.is_symlink():
                _fail(kind, f"report staging collision: {target}")
            with target.open("x", encoding="utf-8") as stream:
                stream.write(value)
        actual = tuple(sorted(path.name for path in staging.iterdir()))
        expected = tuple(name for name, _value in entries)
        if actual != expected or any(name.endswith(".npup") for name in actual):
            _fail(kind, "report evidence set is incomplete or contains NPUP")
        if root.exists() or root.is_symlink():
            _fail(kind, "report root appeared before atomic publication")
        staging.rename(root)


def _run_case(args: argparse.Namespace) -> None:
    kind = Stage3StaticCaseKind(args.case)
    if args.report_root is not None:
        if args.report_root.is_symlink():
            _fail(kind, f"report root must not be a symlink: {args.report_root}")
        if args.report_root.exists():
            _fail(kind, f"report root must not already exist: {args.report_root}")
    first = build_stage3_static_case(kind, 1)
    second = build_stage3_static_case(kind, 1)
    for name in (
        "template", "static_profile", "oracle", "policy",
        "planning_context", "scheduling_context", "global_dag", "lowered",
        "manifest", "program_io",
    ):
        if canonical_digest(getattr(first, name)) != canonical_digest(
            getattr(second, name)
        ):
            _fail(kind, f"builder is not deterministic at {name}")
    if first.runtime_hardware_inputs != second.runtime_hardware_inputs:
        _fail(kind, "runtime inputs are not deterministic")
    _validate_static(kind, first)

    with tempfile.TemporaryDirectory(prefix=f"stage3-{kind.value}-") as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        sidecar_path = directory / "program_io.json"
        manifest_path.write_text(canonical_json(first.manifest), encoding="utf-8")
        hardware_path.write_text(
            first.runtime_hardware_inputs.hardware_json, encoding="utf-8"
        )
        mapping_path.write_text(
            first.runtime_hardware_inputs.mapping_text, encoding="utf-8"
        )

        artifacts = []
        reports = []
        finalizer_logs = []
        for index in range(2):
            artifact_path = directory / f"program.{index}.npup"
            report_path = directory / f"finalizer.{index}.json"
            finalized = _run(
                kind,
                [
                    str(args.finalizer), "--input", str(manifest_path),
                    "--output", str(artifact_path), "--report", str(report_path),
                ],
                cwd=args.runtime_root,
                timeout=120,
            )
            finalizer_logs.append(finalized.stdout)
            artifacts.append(artifact_path.read_bytes())
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        if artifacts[0] != artifacts[1] or reports[0] != reports[1]:
            _fail(kind, "finalizer repeat changed")
        report = reports[0]
        artifact_sha256 = hashlib.sha256(artifacts[0]).hexdigest()
        if (
            report.get("artifact_sha256") != artifact_sha256
            or report.get("artifact_bytes") != len(artifacts[0])
            or report.get("record_count") != _GOLDENS[kind].record_count
            or report.get("relocation_count")
            != _REVIEWED_RUNTIME_GOLDENS[kind].relocation_count
            or report.get("linked_manifest_id") != first.manifest.id
            or report.get("linked_manifest_digest")
            != canonical_digest(first.manifest)
        ):
            _fail(kind, f"finalizer report closure changed: {report}")

        contract = _rebuild_sidecar(kind, first, artifact_sha256)
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        resolver = _run(
            kind,
            [
                str(args.resolver), "--resolve", str(manifest_path),
                str(directory / "program.0.npup"), str(sidecar_path),
            ],
            cwd=args.runtime_root,
            timeout=120,
        )
        witness = (
            f"initializations={_GOLDENS[kind].initialization_count} "
            f"probes={_GOLDENS[kind].probe_count}"
        )
        if witness not in resolver.stdout:
            _fail(kind, f"resolver entry counts changed: {resolver.stdout}")

        observations = []
        runtime_logs = []
        leaves = first.profile.leaf_fragments
        for _ in range(2):
            execution = _run(
                kind,
                [
                    str(args.npusim), "--program", str(directory / "program.0.npup"),
                    "--linked-manifest", str(manifest_path),
                    "--program-io", str(sidecar_path),
                    "--hardware-config", str(hardware_path),
                    "--simulation-config", str(args.simulation),
                    "--mapping-config", str(mapping_path),
                    "--trace-window", "1000000",
                ],
                cwd=args.runtime_root,
                timeout=args.timeout,
            )
            observations.append(
                _observe(
                    kind,
                    execution.stdout,
                    artifact_sha256,
                    contract,
                    leaves,
                )
            )
            runtime_logs.append(execution.stdout)
        if observations[0] != observations[1]:
            _fail(kind, f"runtime repeat changed: {observations}")
        _validate_reviewed_runtime(
            kind,
            _ReviewedRuntimeWitness(
                artifact_bytes=len(artifacts[0]),
                record_count=int(report["record_count"]),
                relocation_count=int(report["relocation_count"]),
                artifact_sha256=artifact_sha256,
                makespan_cycles=observations[0].makespan_cycles,
                marker_digest=observations[0].marker_digest,
            ),
        )
        runtime_report = _build_runtime_report(
            kind,
            first,
            args,
            report,
            len(artifacts[0]),
            artifact_sha256,
            contract,
            (observations[0], observations[1]),
        )
        if args.report_root is not None:
            _publish_report_texts(
                args.report_root,
                kind,
                _report_text_entries(
                    kind,
                    case=first,
                    runtime_report=runtime_report,
                    contract=contract,
                    finalization=report,
                    observations=(observations[0], observations[1]),
                    finalizer_logs=(finalizer_logs[0], finalizer_logs[1]),
                    resolver_log=resolver.stdout,
                    runtime_logs=(runtime_logs[0], runtime_logs[1]),
                ),
            )
        print(
            f"[STAGE3 {kind.value.upper()}] PASS: timing_execution=1 "
            f"compute_functional=0 model_functional=0 "
            f"artifact_bytes={len(artifacts[0])} sha256={artifact_sha256} "
            f"records={report['record_count']} relocations={report['relocation_count']} "
            f"makespan_cycles={observations[0].makespan_cycles} "
            f"marker_digest={observations[0].marker_digest}"
            f" report_id={runtime_report.id}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=tuple(item.value for item in Stage3StaticCaseKind),
    )
    parser.add_argument("--finalizer", required=True, type=_executable)
    parser.add_argument("--resolver", required=True, type=_executable)
    parser.add_argument("--npusim", required=True, type=_executable)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.simulation = args.simulation.resolve()
    if args.report_root is not None:
        args.report_root = args.report_root.absolute()
    if not args.runtime_root.is_dir() or not args.simulation.is_file():
        raise RuntimeError("runtime-root/simulation must exist")
    _run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
