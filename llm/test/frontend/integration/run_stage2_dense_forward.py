#!/usr/bin/env python3
"""Run the reviewed Stage2 Dense-forward timing cases on production npusim."""

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
from typing import Any

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
from llm.frontend.wafer_frontend.schema.experiment import InferOutput  # noqa: E402
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
    STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
    STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION,
    Stage2DenseForwardArtifactEvidence,
    Stage2DenseForwardCompileEvidence,
    Stage2DenseForwardControlEvidence,
    Stage2DenseForwardCoreCount,
    Stage2DenseForwardD2DEvidence,
    Stage2DenseForwardD2DLinkEvidence,
    Stage2DenseForwardMemoryEvidence,
    Stage2DenseForwardNamedCount,
    Stage2DenseForwardOpcodeCount,
    Stage2DenseForwardProbeEvidence,
    Stage2DenseForwardRepeatEvidence,
    Stage2DenseForwardRuntimeReport,
    Stage2DenseForwardSidecarEvidence,
    Stage2DenseForwardToolEvidence,
)
from stage2_dense_forward_cases import build_stage2_dense_forward_case  # noqa: E402

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
_PACKET_BYTES = 16


@dataclass(frozen=True)
class _MemoryGolden:
    lsu_issued: int
    lsu_completed: int
    lsu_hbm_read_bytes: int
    lsu_hbm_write_bytes: int
    lsu_sram_read_bytes: int
    lsu_sram_write_bytes: int
    lsu_residual: int = 0
    dte_residual: int = 0


@dataclass(frozen=True)
class _StaticGolden:
    action_count: int
    leaf_count: int
    record_count: int
    address_binding_count: int
    relocation_count: int
    opcode_counts: tuple[tuple[str, int], ...]
    active_cores: tuple[int, ...]
    initialization_count: int
    probe_count: int
    memory: tuple[tuple[int, _MemoryGolden], ...]
    data_flows: int
    logical_data_packets: int
    physical_data_packets: int
    request_packets: int
    ack_packets: int
    link_data_packets: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _StructureWitness:
    action_count: int
    leaf_count: int
    record_count: int
    address_binding_count: int
    relocation_count: int


@dataclass(frozen=True)
class _ReviewedRuntimeWitness:
    artifact_bytes: int
    record_count: int
    relocation_count: int
    artifact_sha256: str
    makespan_cycles: int


def _opcodes(**counts: int) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counts.items()))


_GOLDENS = {
    1: _StaticGolden(
        44, 44, 159, 278, 297,
        _opcodes(
            ATTENTION_EXACT=2, EMBEDDING_LOOKUP=1, LSU_LOAD=15,
            LSU_STORE=4, MATMUL=9, RESIDUAL=4, RMSNORM=5,
            ROPE_QK_EXACT=2, SRAM_ALLOC_AT=45, SRAM_BIND=25,
            SRAM_FREE=45, SWIGLU=2,
        ),
        (0,), 60, 1,
        ((0, _MemoryGolden(19, 19, 12448, 1024, 1024, 12448)),),
        0, 0, 0, 0, 0, (),
    ),
    2: _StaticGolden(
        160, 92, 510, 804, 842,
        _opcodes(
            ATTENTION_EXACT=4, DTE_ISSUE=8, DTE_RECV=16,
            DTE_SEND=16, DTE_WAIT=16, EMBEDDING_LOOKUP=2,
            EVENT_SET=8, EVENT_WAIT=8, LOCAL_REDUCE=8, LSU_LOAD=30,
            LSU_STORE=8, MATMUL=26, RESIDUAL=8, RMSNORM=10,
            ROPE_QK_EXACT=4, SRAM_ALLOC_AT=138, SRAM_BIND=58,
            SRAM_FREE=138, SWIGLU=4,
        ),
        (0, 16), 168, 2,
        (
            (0, _MemoryGolden(19, 19, 7328, 512, 512, 7328)),
            (16, _MemoryGolden(19, 19, 7328, 512, 512, 7328)),
        ),
        16, 128, 128, 16, 32,
        ((0, 1, 64), (1, 0, 64)),
    ),
    4: _StaticGolden(
        544, 180, 1452, 2184, 2260,
        _opcodes(
            ATTENTION_EXACT=8, DTE_ISSUE=16, DTE_RECV=96,
            DTE_SEND=96, DTE_WAIT=64, EMBEDDING_LOOKUP=4,
            EVENT_SET=24, EVENT_WAIT=24, LOCAL_REDUCE=16,
            LSU_LOAD=60, LSU_STORE=16, MATMUL=84, RESIDUAL=16,
            RMSNORM=20, ROPE_QK_EXACT=8, SRAM_ALLOC_AT=372,
            SRAM_BIND=148, SRAM_FREE=372, SWIGLU=8,
        ),
        (0, 4, 8, 12), 432, 4,
        (
            (0, _MemoryGolden(19, 19, 4768, 256, 256, 4768)),
            (4, _MemoryGolden(19, 19, 4768, 256, 256, 4768)),
            (8, _MemoryGolden(19, 19, 4768, 256, 256, 4768)),
            (12, _MemoryGolden(19, 19, 4768, 256, 256, 4768)),
        ),
        96, 384, 512, 128, 256,
        (
            (0, 1, 64), (0, 2, 64), (1, 0, 64), (1, 3, 64),
            (2, 0, 64), (2, 3, 64), (3, 1, 64), (3, 2, 64),
        ),
    ),
}


_REVIEWED_RUNTIME_GOLDENS = {
    1: _ReviewedRuntimeWitness(
        22258, 159, 297,
        "308152e15697e45329660c160a3428a809015e408079019693e8d7c7d4d3fbad",
        5805,
    ),
    2: _ReviewedRuntimeWitness(
        65832, 510, 842,
        "1824ceeea8a23150489df9a0e41f4571ac90edbcf86ff82da0feb786888a830c",
        6628,
    ),
    4: _ReviewedRuntimeWitness(
        179316, 1452, 2260,
        "fad49b3356701b6e5e65139f7e91104b0693c020710536b97af84bd177fae187",
        7503,
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


def _fail(tp_degree: int, message: str) -> None:
    raise RuntimeError(f"[STAGE2 TP{tp_degree}] FAIL: {message}")


def _validate_structure(tp_degree: int, observed: _StructureWitness) -> None:
    expected = _GOLDENS[tp_degree]
    wanted = _StructureWitness(
        action_count=expected.action_count,
        leaf_count=expected.leaf_count,
        record_count=expected.record_count,
        address_binding_count=expected.address_binding_count,
        relocation_count=expected.relocation_count,
    )
    if observed != wanted:
        _fail(
            tp_degree,
            f"five-field structure changed: {observed!r} != {wanted!r}",
        )


def _validate_reviewed_runtime(
    tp_degree: int, observed: _ReviewedRuntimeWitness
) -> None:
    wanted = _REVIEWED_RUNTIME_GOLDENS[tp_degree]
    if observed != wanted:
        _fail(
            tp_degree,
            f"reviewed runtime five-field golden changed: {observed!r} != {wanted!r}",
        )


def _executable(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _run(
    tp_degree: int,
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
            tp_degree,
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}",
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
        if position < 0:
            continue
        normalized = line[position:].split(" | ", 1)[0].rstrip(". ")
        result.append(_row(normalized, prefix))
    return result


def _number(tp_degree: int, row: dict[str, str], key: str) -> int:
    try:
        return int(row[key], 10)
    except (KeyError, ValueError) as error:
        _fail(tp_degree, f"missing/non-decimal {key!r} in {row}")
        raise AssertionError from error


def _marker_lines(output: str) -> tuple[str, ...]:
    prefixes = (
        _STATUS, _PROBE, _MEMORY, _SIM, _HOST, _HOSTSIG, _P5,
        _P5_TIMING, _COLL, _DRAIN, _D2D_TYPE, _D2D_BEHA, _D2D_LINK,
    )
    result = []
    for line in output.splitlines():
        positions = tuple(line.find(prefix) for prefix in prefixes)
        positions = tuple(position for position in positions if position >= 0)
        if positions:
            normalized = line[min(positions) :].split(" | ", 1)[0].rstrip(". ")
            result.append(normalized)
    return tuple(result)


def _leaf_fragments(case: Any) -> tuple[CommandFragment, ...]:
    leaves = case.profile.leaf_fragments
    if any(type(fragment) is not CommandFragment for fragment in leaves):
        _fail(case.spec.placement.tp_degree, "manifest has a non-command leaf")
    return leaves


def _opcode_counter(leaves: tuple[CommandFragment, ...]) -> Counter[str]:
    return Counter(
        record.opcode.name
        for fragment in leaves
        for stream in fragment.core_streams
        for record in stream.records
    )


def _validate_static_case(tp_degree: int, case: Any) -> tuple[CommandFragment, ...]:
    expected = _GOLDENS[tp_degree]
    leaves = _leaf_fragments(case)
    opcodes = _opcode_counter(leaves)
    observed = (
        len(case.global_dag.actions),
        len(leaves),
        sum(opcodes.values()),
        len(case.manifest.address_operand_bindings),
        tuple(sorted(opcodes.items())),
        tuple(sorted(binding.runtime_core_id for binding in case.manifest.core_bindings)),
        len(case.program_io.initializations),
        len(case.program_io.output_probes),
    )
    wanted = (
        expected.action_count,
        expected.leaf_count,
        expected.record_count,
        expected.address_binding_count,
        expected.opcode_counts,
        expected.active_cores,
        expected.initialization_count,
        expected.probe_count,
    )
    if observed != wanted:
        _fail(tp_degree, f"static graph/artifact counts changed: {observed!r} != {wanted!r}")
    if opcodes[RecordOpcode.LSU_LOAD.name] != 15 * tp_degree:
        _fail(tp_degree, "LSU_LOAD static count changed")
    if opcodes[RecordOpcode.LSU_STORE.name] != 4 * tp_degree:
        _fail(tp_degree, "LSU_STORE static count changed")
    if opcodes[RecordOpcode.DTE_SEND.name] != expected.data_flows:
        _fail(tp_degree, "DTE_SEND static flow count changed")
    send_records = tuple(
        record
        for fragment in leaves
        for stream in fragment.core_streams
        for record in stream.records
        if record.opcode is RecordOpcode.DTE_SEND
    )
    send_lengths = tuple(
        next(operand.literal_value for operand in record.operands
             if operand.name == "length_bytes")
        for record in send_records
    )
    if (any(type(length) is not int or length <= 0 or length % _PACKET_BYTES
            for length in send_lengths) or
            sum(send_lengths) // _PACKET_BYTES != expected.logical_data_packets):
        _fail(tp_degree, f"DTE_SEND logical packet bytes changed: {send_lengths}")
    return leaves


def _rebuild_sidecar(
    tp_degree: int, case: Any, artifact_sha256: str
) -> ProgramIoContract:
    placeholder = case.program_io
    state_seeds, state_expected = (
        build_deterministic_timing_state_overrides(case.profile)
    )
    if (
        tuple(sorted(state_seeds)) != case.state_seed_refs
        or tuple(sorted(state_expected)) != case.state_expected_refs
    ):
        _fail(tp_degree, "state override refs changed")
    result = build_timing_program_io(
        case.profile,
        artifact_sha256,
        state_seed_overrides=state_seeds,
        state_expected_overrides=state_expected,
    )
    result.validate_against(case.manifest)
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
    if tuple(getattr(result, name) for name in non_sha_fields) != tuple(
        getattr(placeholder, name) for name in non_sha_fields
    ):
        _fail(
            tp_degree,
            "actual-SHA ProgramIo non-SHA semantics changed",
        )
    return result


def _parse_memory(
    tp_degree: int, output: str
) -> tuple[Stage2DenseForwardMemoryEvidence, ...]:
    expected = dict(_GOLDENS[tp_degree].memory)
    rows = _rows(output, _MEMORY)
    if tuple(_number(tp_degree, row, "core") for row in rows) != tuple(expected):
        _fail(tp_degree, f"PROGRAM_MEMORY core closure changed: {rows}")
    result = []
    for row in rows:
        core = _number(tp_degree, row, "core")
        actual = {
            name: _number(tp_degree, row, name)
            for name in _MemoryGolden.__dataclass_fields__
        }
        wanted = vars(expected[core])
        if actual != wanted:
            _fail(tp_degree, f"PROGRAM_MEMORY core {core} changed: {actual} != {wanted}")
        result.append(Stage2DenseForwardMemoryEvidence(core, **actual))
    return tuple(result)


def _parse_probes(
    tp_degree: int, output: str, contract: ProgramIoContract
) -> tuple[Stage2DenseForwardProbeEvidence, ...]:
    rows = _rows(output, _PROBE)
    if len(rows) != _GOLDENS[tp_degree].probe_count:
        _fail(tp_degree, f"probe marker multiplicity changed: {rows}")
    entries = {entry.id: entry for entry in contract.output_probes}
    blobs = {blob.id: blob for blob in contract.blobs}
    result = []
    for row in rows:
        entry = entries.get(row.get("id", ""))
        if entry is None or type(entry.target) is not ProgramSramTarget:
            _fail(tp_degree, f"unknown/non-SRAM probe marker: {row}")
        expected_sha = blobs[entry.blob_ref].sha256
        if (
            row.get("expected_checksum") != expected_sha
            or row.get("checksum") != expected_sha
            or row.get("valid") != "1"
            or row.get("exact") != "1"
            or row.get("pass") != "1"
            or _number(tp_degree, row, "bytes") != entry.length_bytes
            or _number(tp_degree, row, "core") != entry.target.runtime_core_id
        ):
            _fail(tp_degree, f"probe did not close exactly: {row}")
        result.append(
            Stage2DenseForwardProbeEvidence(
                probe_id=entry.id,
                target_kind=ProgramIoTargetKind.SRAM,
                length_bytes=entry.length_bytes,
                expected_sha256=expected_sha,
                actual_sha256=row["checksum"],
                all_bytes_valid=row["valid"] == "1",
                exact_match=row["exact"] == "1",
                passed=row["pass"] == "1",
            )
        )
    return tuple(sorted(result, key=lambda item: item.probe_id))


def _parse_signature(tp_degree: int, value: str, arity: int) -> list[tuple[int, ...]]:
    result = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        try:
            parsed = tuple(int(part, 10) for part in item.split(":"))
        except ValueError as error:
            _fail(tp_degree, f"non-decimal HOSTSIG item {item!r}")
            raise AssertionError from error
        if len(parsed) != arity:
            _fail(tp_degree, f"bad HOSTSIG item {item!r}")
        result.append(parsed)
    return sorted(result)


def _parse_control(
    tp_degree: int, output: str
) -> tuple[int, Stage2DenseForwardControlEvidence]:
    expected = _GOLDENS[tp_degree]
    simulation = _rows(output, _SIM)
    if len(simulation) != 1:
        _fail(tp_degree, "SIM_RESULT must appear exactly once")
    makespan = _number(tp_degree, simulation[0], "makespan_cycles")
    if makespan <= 0 or "[PROTO_WAIT]" in output or "End DONE reception" not in output:
        _fail(tp_degree, "simulation/DONE path did not close")

    host = _rows(output, _HOST)
    signatures = _rows(output, _HOSTSIG)
    if len(host) != 1 or len(signatures) != 1:
        _fail(tp_degree, "HOSTLANE/HOSTSIG must each appear exactly once")
    if (
        _number(tp_degree, host[0], "ack_total") != 2 * len(expected.active_cores)
        or _number(tp_degree, host[0], "done_total") != len(expected.active_cores)
        or _number(tp_degree, host[0], "mismatch") != 0
    ):
        _fail(tp_degree, f"ACK/DONE totals changed: {host[0]}")
    done = _parse_signature(tp_degree, signatures[0].get("done", ""), 2)
    ack = _parse_signature(tp_degree, signatures[0].get("ack", ""), 3)
    if done != [(core, 1) for core in expected.active_cores]:
        _fail(tp_degree, f"DONE per-core closure changed: {done}")
    ack_by_core: Counter[int] = Counter()
    for core, _lane, count in ack:
        ack_by_core[core] += count
    if sorted(ack_by_core.items()) != [(core, 2) for core in expected.active_cores]:
        _fail(tp_degree, f"ACK per-core closure changed: {ack}")

    timing = _rows(output, _P5_TIMING)
    if len(timing) != 1 or _number(tp_degree, timing[0], "residual") != 0:
        _fail(tp_degree, f"P2P timing drain changed: {timing}")
    endpoints = _rows(output, _P5)
    endpoint_cores = tuple(sorted(_number(tp_degree, row, "core") for row in endpoints))
    expected_endpoint_cores = expected.active_cores if expected.data_flows else ()
    if endpoint_cores != expected_endpoint_cores or any(
        _number(tp_degree, row, "residual") for row in endpoints
    ):
        _fail(tp_degree, f"P2P endpoint drain changed: {endpoints}")

    collective = _rows(output, _COLL)
    if len(collective) != 1 or any(
        _number(tp_degree, collective[0], key)
        for key in (
            "tree_entries", "reduce_nodes", "barriers", "gather",
            "reduce_rx", "endpoints", "dte_tokens", "event",
        )
    ):
        _fail(tp_degree, f"collective drain changed: {collective}")
    drains = _rows(output, _DRAIN)
    residuals = {
        key: _number(tp_degree, row, key)
        for row in drains
        for key in ("router_residual", "d2d_link_residual")
        if key in row
    }
    if residuals != {"router_residual": 0, "d2d_link_residual": 0}:
        _fail(tp_degree, f"global drain changed: {drains}")
    return makespan, Stage2DenseForwardControlEvidence(
        ack_counts=tuple(
            Stage2DenseForwardCoreCount(core, ack_by_core[core])
            for core in expected.active_cores
        ),
        done_counts=tuple(
            Stage2DenseForwardCoreCount(core, count) for core, count in done
        ),
        drain_residuals=tuple(
            Stage2DenseForwardNamedCount(name, 0)
            for name in ("collective", "global", "p2p", "timing")
        ),
        all_done_boundary_reached=True,
    )


def _parse_d2d(tp_degree: int, output: str) -> Stage2DenseForwardD2DEvidence:
    expected = _GOLDENS[tp_degree]
    typed = _rows(output, _D2D_TYPE)
    if len(typed) != 1:
        _fail(tp_degree, "D2D_TYPE must appear exactly once")
    row = typed[0]
    values = {
        key: _number(tp_degree, row, key)
        for key in ("request_in", "request_out", "ack_in", "ack_out", "data_in", "data_out")
    }
    if values != {
        "request_in": expected.request_packets,
        "request_out": expected.request_packets,
        "ack_in": expected.ack_packets,
        "ack_out": expected.ack_packets,
        "data_in": expected.physical_data_packets,
        "data_out": expected.physical_data_packets,
    }:
        _fail(tp_degree, f"D2D_TYPE count changed: {values}")
    if _rows(output, _D2D_BEHA):
        _fail(tp_degree, "canonical Stage2 hardware unexpectedly emitted D2D_BEHA")

    pattern = re.compile(
        r"\[D2D_LINK\] idx=\d+ die(\d+)->die(\d+) dir=[A-Z?]+ "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)"
    )
    links = []
    expected_link_requests = (
        expected.request_packets // len(expected.link_data_packets)
        if expected.link_data_packets
        else 0
    )
    expected_link_acks = (
        expected.ack_packets // len(expected.link_data_packets)
        if expected.link_data_packets
        else 0
    )
    for match in pattern.finditer(output):
        source, destination, req_in, req_out, ack_in, ack_out, data_in, data_out = (
            int(value) for value in match.groups()
        )
        if (req_in != req_out or ack_in != ack_out or data_in != data_out or
                req_out != expected_link_requests or
                ack_out != expected_link_acks):
            _fail(tp_degree, f"D2D link {source}->{destination} is not balanced")
        links.append(
            Stage2DenseForwardD2DLinkEvidence(
                source_die_id=source,
                destination_die_id=destination,
                request_packets=req_out,
                ack_packets=ack_out,
                data_packets=data_out,
            )
        )
    result = tuple(
        sorted(links, key=lambda item: (item.source_die_id, item.destination_die_id))
    )
    observed_link_data = tuple(
        (item.source_die_id, item.destination_die_id, item.data_packets)
        for item in result
    )
    if observed_link_data != expected.link_data_packets:
        _fail(tp_degree, f"D2D_LINK data closure changed: {observed_link_data}")
    if sum(item.data_packets for item in result) != expected.physical_data_packets:
        _fail(tp_degree, "D2D byte-hop packet total changed")
    if expected.logical_data_packets * _PACKET_BYTES not in (0, 2048, 6144):
        _fail(tp_degree, "logical D2D byte oracle is malformed")
    return Stage2DenseForwardD2DEvidence(
        flow_count=expected.data_flows,
        logical_packet_count=expected.logical_data_packets,
        physical_packet_count=values["data_out"],
        request_packet_count=values["request_out"],
        ack_packet_count=values["ack_out"],
        logical_bytes=expected.logical_data_packets * _PACKET_BYTES,
        byte_hop_bytes=values["data_out"] * _PACKET_BYTES,
        links=result,
    )


def _validate_status(
    tp_degree: int,
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
) -> None:
    rows = _rows(output, _STATUS)
    if tuple(row.get("phase") for row in rows) != ("resolved", "applied", "verify"):
        _fail(tp_degree, f"ProgramIo phase closure changed: {rows}")
    for row in rows:
        if (
            row.get("mode") != "timing"
            or _number(tp_degree, row, "initializations")
            != len(contract.initializations)
            or _number(tp_degree, row, "probes") != len(contract.output_probes)
            or row.get("pass") != "1"
        ):
            _fail(tp_degree, f"ProgramIo status failed: {row}")
    if rows[0].get("checksum") != artifact_sha256 or rows[1].get("checksum") != artifact_sha256:
        _fail(tp_degree, "resolved/applied checksum is not actual artifact SHA")


def _observe_runtime(
    tp_degree: int,
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
) -> _RuntimeObservation:
    _validate_status(tp_degree, output, artifact_sha256, contract)
    memory = _parse_memory(tp_degree, output)
    probes = _parse_probes(tp_degree, output, contract)
    makespan, control = _parse_control(tp_degree, output)
    d2d = _parse_d2d(tp_degree, output)
    lines = _marker_lines(output)
    expected_marker_counts = {
        _STATUS: 3,
        _PROBE: _GOLDENS[tp_degree].probe_count,
        _MEMORY: len(_GOLDENS[tp_degree].active_cores),
        _SIM: 1,
        _HOST: 1,
        _HOSTSIG: 1,
        _P5_TIMING: 1,
        _COLL: 1,
        _D2D_TYPE: 1,
    }
    for prefix, count in expected_marker_counts.items():
        if sum(line.startswith(prefix.rstrip()) for line in lines) != count:
            _fail(tp_degree, f"marker multiplicity changed for {prefix.strip()}")
    return _RuntimeObservation(
        makespan_cycles=makespan,
        marker_digest=hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        memory=memory,
        probes=probes,
        control=control,
        d2d=d2d,
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_runtime_report(
    tp_degree: int,
    case: Any,
    args: argparse.Namespace,
    finalization: dict[str, object],
    artifact_size_bytes: int,
    artifact_sha256: str,
    contract: ProgramIoContract,
    observations: tuple[_RuntimeObservation, _RuntimeObservation],
    hardware_path: Path,
    mapping_path: Path,
) -> Stage2DenseForwardRuntimeReport:
    leaves = _leaf_fragments(case)
    opcode_counter = _opcode_counter(leaves)
    opcode_counts = tuple(
        Stage2DenseForwardOpcodeCount(RecordOpcode[name], count)
        for name, count in sorted(
            opcode_counter.items(), key=lambda item: int(RecordOpcode[item[0]])
        )
    )
    hbm_initialization_count = sum(
        type(entry.target) is ProgramHbmTarget
        for entry in contract.initializations
    )
    sram_initialization_count = sum(
        type(entry.target) is ProgramSramTarget
        for entry in contract.initializations
    )
    hbm_probe_count = sum(
        type(entry.target) is ProgramHbmTarget for entry in contract.output_probes
    )
    sram_probe_count = sum(
        type(entry.target) is ProgramSramTarget for entry in contract.output_probes
    )
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
    report = Stage2DenseForwardRuntimeReport.create(
        baseline_epoch=STAGE2_DENSE_FORWARD_BASELINE_EPOCH,
        tp_degree=tp_degree,
        infer_output=InferOutput.LOGITS,
        capability_status=CapabilityStatus.E2E_TIMING,
        oracle_id=case.oracle.id,
        oracle_digest=canonical_digest(case.oracle),
        compile=Stage2DenseForwardCompileEvidence(
            template_id=case.template.id,
            template_digest=canonical_digest(case.template),
            ir1_id=case.graph.id,
            ir1_digest=canonical_digest(case.graph),
            global_dag_id=case.global_dag.id,
            global_dag_digest=canonical_digest(case.global_dag),
            lowered_id=case.lowered.id,
            lowered_digest=canonical_digest(case.lowered),
        ),
        tools=Stage2DenseForwardToolEvidence(
            finalizer_sha256=_sha256_file(args.finalizer),
            resolver_sha256=_sha256_file(args.resolver),
            npusim_sha256=_sha256_file(args.npusim),
        ),
        hardware_digest=_sha256_file(hardware_path),
        simulation_digest=_sha256_file(args.simulation),
        mapping_digest=_sha256_file(mapping_path),
        artifact=Stage2DenseForwardArtifactEvidence(
            linked_manifest_id=case.manifest.id,
            linked_manifest_digest=canonical_digest(case.manifest),
            program_artifact_sha256=artifact_sha256,
            artifact_size_bytes=artifact_size_bytes,
            action_count=len(case.global_dag.actions),
            leaf_fragment_count=len(leaves),
            record_count=int(finalization["record_count"]),
            address_binding_count=len(case.manifest.address_operand_bindings),
            relocation_count=int(finalization["relocation_count"]),
            opcode_counts=opcode_counts,
        ),
        sidecar=Stage2DenseForwardSidecarEvidence(
            contract_id=contract.id,
            contract_digest=canonical_digest(contract),
            mode=contract.mode,
            hbm_initialization_count=hbm_initialization_count,
            sram_initialization_count=sram_initialization_count,
            hbm_probe_count=hbm_probe_count,
            sram_probe_count=sram_probe_count,
        ),
        memory=first.memory,
        probes=first.probes,
        control=first.control,
        d2d=first.d2d,
        marker_schema_version=STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION,
        repeat_count=len(observations),
        makespan_cycles=first.makespan_cycles,
        repeats=repeats,
        timing_execution=True,
        dense_forward_structure_exact=True,
        analytic_work_exact=True,
        program_io_boundary_exact=True,
        traffic_accounting_exact=True,
        compute_functional=False,
        model_functional=False,
    )
    report.validate_against(case.oracle)
    return report


def _report_text_entries(
    tp_degree: int,
    *,
    case: Any,
    runtime_report: Stage2DenseForwardRuntimeReport,
    contract: ProgramIoContract,
    finalization: dict[str, object],
    observations: tuple[_RuntimeObservation, _RuntimeObservation],
    finalizer_logs: tuple[str, str],
    resolver_log: str,
    runtime_logs: tuple[str, str],
) -> tuple[tuple[str, str], ...]:
    if len(finalizer_logs) != 2 or len(runtime_logs) != 2:
        _fail(tp_degree, "report evidence requires exactly two repeat logs")
    if not resolver_log:
        _fail(tp_degree, "resolver evidence log must be non-empty")
    for index, log in enumerate(runtime_logs):
        if _SIM not in log or _D2D_TYPE not in log:
            _fail(tp_degree, f"runtime evidence log {index} lacks markers")
    stem = f"stage2-tp{tp_degree}"
    markers = {
        "schema_version": STAGE2_DENSE_FORWARD_MARKER_SCHEMA_VERSION,
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
        "schema_version": "wafer_frontend.stage2_dense_forward_inputs/v1",
        "tools": runtime_report.tools,
        "hardware_digest": runtime_report.hardware_digest,
        "simulation_digest": runtime_report.simulation_digest,
        "mapping_digest": runtime_report.mapping_digest,
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
        (f"{stem}.linked_manifest.json", canonical_json(case.manifest)),
        (f"{stem}.markers.json", canonical_json(markers)),
        (f"{stem}.oracle.json", canonical_json(case.oracle)),
        (f"{stem}.program_io.json", canonical_json(contract)),
        (f"{stem}.resolver.log", resolver_log),
        (f"{stem}.runtime.0.log", runtime_logs[0]),
        (f"{stem}.runtime.1.log", runtime_logs[1]),
        (f"{stem}.runtime.json", canonical_json(runtime_report)),
    )
    names = tuple(sorted(name for name, _value in entries))
    if (
        names != _expected_report_names(tp_degree)
        or
        len(names) != len(set(names))
        or any(Path(name).name != name or name.endswith(".npup") for name in names)
        or any(type(value) is not str for _name, value in entries)
    ):
        _fail(tp_degree, "report entries must be unique flat text without NPUP")
    return tuple(sorted(entries))


def _expected_report_names(tp_degree: int) -> tuple[str, ...]:
    stem = f"stage2-tp{tp_degree}"
    return tuple(sorted((
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
    )))


def _publish_report_texts(
    root: Path,
    tp_degree: int,
    entries: tuple[tuple[str, str], ...],
) -> None:
    if root.is_symlink():
        _fail(tp_degree, f"report root must not be a symlink: {root}")
    if root.exists():
        _fail(tp_degree, f"report root must not already exist: {root}")
    entry_names = tuple(name for name, _value in entries)
    if entry_names != _expected_report_names(tp_degree):
        _fail(tp_degree, "report evidence set is not the exact reviewed set")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.staging-", dir=root.parent
    ) as raw:
        staging = Path(raw)
        for name, value in entries:
            target = staging / name
            if target.exists() or target.is_symlink():
                _fail(tp_degree, f"report staging collision: {target}")
            with target.open("x", encoding="utf-8") as stream:
                stream.write(value)
        actual = tuple(sorted(path.name for path in staging.iterdir()))
        expected = tuple(name for name, _value in entries)
        if actual != expected or any(name.endswith(".npup") for name in actual):
            _fail(tp_degree, "report evidence set is incomplete or contains NPUP")
        if root.exists() or root.is_symlink():
            _fail(tp_degree, "report root appeared before atomic publication")
        staging.rename(root)


def _run_case(args: argparse.Namespace) -> None:
    tp_degree = args.tp_degree
    if args.report_root is not None:
        if args.report_root.is_symlink():
            _fail(tp_degree, f"report root must not be a symlink: {args.report_root}")
        if args.report_root.exists():
            _fail(tp_degree, f"report root must not already exist: {args.report_root}")
    expected = _GOLDENS[tp_degree]
    case = build_stage2_dense_forward_case(tp_degree)
    repeat_case = build_stage2_dense_forward_case(tp_degree)
    for name in ("template", "oracle", "global_dag", "lowered", "manifest", "program_io"):
        if canonical_digest(getattr(case, name)) != canonical_digest(getattr(repeat_case, name)):
            _fail(tp_degree, f"builder {name} is not deterministic")
    if case.runtime_hardware_inputs != repeat_case.runtime_hardware_inputs:
        _fail(tp_degree, "runtime hardware inputs are not deterministic")
    _validate_static_case(tp_degree, case)

    with tempfile.TemporaryDirectory(prefix=f"stage2-tp{tp_degree}-", dir=args.runtime_root) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        sidecar_path = directory / "program_io.json"
        artifact_paths = (directory / "program.0.npup", directory / "program.1.npup")
        report_paths = (directory / "finalizer.0.json", directory / "finalizer.1.json")
        manifest_path.write_text(canonical_json(case.manifest), encoding="utf-8")
        hardware_path.write_text(case.runtime_hardware_inputs.hardware_json, encoding="utf-8")
        mapping_path.write_text(case.runtime_hardware_inputs.mapping_text, encoding="utf-8")

        reports = []
        artifacts = []
        finalizer_logs = []
        for artifact_path, report_path in zip(artifact_paths, report_paths, strict=True):
            finalized = _run(
                tp_degree,
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
            finalizer_logs.append(finalized.stdout)
            artifacts.append(artifact_path.read_bytes())
            reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        if artifacts[0] != artifacts[1] or reports[0] != reports[1]:
            _fail(tp_degree, "finalizer repeat changed")
        report = reports[0]
        artifact_sha256 = hashlib.sha256(artifacts[0]).hexdigest()
        _validate_structure(
            tp_degree,
            _StructureWitness(
                len(case.global_dag.actions), len(_leaf_fragments(case)),
                int(report["record_count"]),
                len(case.manifest.address_operand_bindings),
                int(report["relocation_count"]),
            ),
        )
        if (
            report.get("artifact_sha256") != artifact_sha256
            or report.get("artifact_bytes") != len(artifacts[0])
            or report.get("record_count") != expected.record_count
            or report.get("relocation_count") != expected.relocation_count
            or report.get("linked_manifest_id") != case.manifest.id
            or report.get("linked_manifest_digest") != canonical_digest(case.manifest)
        ):
            _fail(tp_degree, f"finalizer report closure changed: {report}")

        contract = _rebuild_sidecar(tp_degree, case, artifact_sha256)
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        resolver = _run(
            tp_degree,
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(artifact_paths[0]),
                str(sidecar_path),
            ],
            cwd=args.runtime_root,
            timeout=120,
        )
        witness = f"initializations={expected.initialization_count} probes={expected.probe_count}"
        if witness not in resolver.stdout:
            _fail(tp_degree, f"resolver lost entry counts: {resolver.stdout}")

        observations = []
        runtime_logs = []
        for _run_index in range(2):
            execution = _run(
                tp_degree,
                [
                    str(args.npusim), "--program", str(artifact_paths[0]),
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
            runtime_logs.append(execution.stdout)
            observations.append(
                _observe_runtime(
                    tp_degree, execution.stdout, artifact_sha256, contract
                )
            )
        if observations[0] != observations[1]:
            _fail(tp_degree, f"runtime repeat changed: {observations}")

        observation_pair = (observations[0], observations[1])
        runtime_report = _build_runtime_report(
            tp_degree,
            case,
            args,
            report,
            len(artifacts[0]),
            artifact_sha256,
            contract,
            observation_pair,
            hardware_path,
            mapping_path,
        )
        if args.report_root is not None:
            _publish_report_texts(
                args.report_root,
                tp_degree,
                _report_text_entries(
                    tp_degree,
                    case=case,
                    runtime_report=runtime_report,
                    contract=contract,
                    finalization=report,
                    observations=observation_pair,
                    finalizer_logs=(finalizer_logs[0], finalizer_logs[1]),
                    resolver_log=resolver.stdout,
                    runtime_logs=(runtime_logs[0], runtime_logs[1]),
                ),
            )

    observation = observations[0]
    _validate_reviewed_runtime(
        tp_degree,
        _ReviewedRuntimeWitness(
            artifact_bytes=len(artifacts[0]),
            record_count=int(report["record_count"]),
            relocation_count=int(report["relocation_count"]),
            artifact_sha256=artifact_sha256,
            makespan_cycles=observation.makespan_cycles,
        ),
    )
    print(
        f"[STAGE2 TP{tp_degree}] PASS: timing_execution=1 compute_functional=0 "
        f"model_functional=0 artifact={len(artifacts[0])}B "
        f"records={report['record_count']} relocations={report['relocation_count']} "
        f"LSU_LOAD={15 * tp_degree} LSU_STORE={4 * tp_degree} "
        f"HBM_READ={sum(item.lsu_hbm_read_bytes for _core, item in expected.memory)} "
        f"HBM_WRITE={sum(item.lsu_hbm_write_bytes for _core, item in expected.memory)} "
        f"D2D_LOGICAL_BYTES={expected.logical_data_packets * _PACKET_BYTES} "
        f"D2D_BYTE_HOPS={expected.physical_data_packets * _PACKET_BYTES} "
        f"ACK={2 * len(expected.active_cores)} DONE={len(expected.active_cores)} "
        f"repeat=2 makespan_cycles={observation.makespan_cycles} "
        f"sha256={artifact_sha256}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-degree", required=True, type=int, choices=(1, 2, 4))
    parser.add_argument("--npusim", required=True, type=_executable)
    parser.add_argument("--finalizer", required=True, type=_executable)
    parser.add_argument("--resolver", required=True, type=_executable)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    args.simulation = args.simulation.resolve()
    if not args.simulation.is_file():
        parser.error(f"--simulation is not a file: {args.simulation}")
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.report_root is not None:
        args.report_root = args.report_root.absolute()
    _run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
