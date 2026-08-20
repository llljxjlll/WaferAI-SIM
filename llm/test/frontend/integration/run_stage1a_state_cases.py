#!/usr/bin/env python3
"""Run the three reviewed Stage1a persistent-state foundation cases."""

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
from typing import Any

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from llm.frontend.wafer_frontend.schema import (  # noqa: E402
    STAGE1A_BASELINE_EPOCH,
    Stage1aArtifactEvidence,
    Stage1aCase,
    Stage1aControlEvidence,
    Stage1aCoreCount,
    Stage1aDecodeStartBoundary,
    Stage1aMemoryEvidence,
    Stage1aNamedCount,
    Stage1aOpcodeCount,
    Stage1aOracle,
    Stage1aPdWitness,
    Stage1aProbeEvidence,
    Stage1aRepeatEvidence,
    Stage1aRuntimeReport,
    Stage1aSidecarEvidence,
    Stage1aStatePair,
    Stage1aStatePayload,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (  # noqa: E402
    CommandFragment,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.capability import CapabilityStatus  # noqa: E402
from llm.frontend.wafer_frontend.schema.ir2 import (  # noqa: E402
    SemanticTaskKind,
    StateIoOrigin,
)
from llm.frontend.wafer_frontend.schema.program_io import (  # noqa: E402
    ProgramHbmTarget,
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoTargetKind,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
)
from stage1a_state_cases import (  # noqa: E402
    build_cross_action_kv_foundation_case,
    build_parameter_foundation_case,
    build_pd1_case,
)

_STATUS_PREFIX = "[PROGRAM_IO] "
_PROBE_PREFIX = "[PROGRAM_IO_PROBE] "
_MEMORY_PREFIX = "[PROGRAM_MEMORY] "
_SIM_RESULT_PREFIX = "[SIM_RESULT] "
_HOST_PREFIX = "[HOSTLANE] "
_HOSTSIG_PREFIX = "[HOSTSIG] "
_P5_DRAIN_PREFIX = "[P5 P2P DRAIN] "
_P5_TIMING_PREFIX = "[P5 P2P TIMING DRAIN] "
_COLL_DRAIN_PREFIX = "[COLL_DRAIN] "
_GLOBAL_DRAIN_PREFIX = "[DRAIN] "
_D2D_TYPE_PREFIX = "[D2D_TYPE] "
_D2D_BEHA_PREFIX = "[D2D_BEHA] "
P2P_PAYLOAD_FRAGMENT_BYTES = 16
_DRAINS = ("collective", "global", "p2p", "timing")


@dataclass(frozen=True)
class _D2DEvidence:
    data_flows: int
    logical_data_packets: int
    observed_bytes: int


@dataclass(frozen=True)
class _IntegrationGolden:
    artifact_size_bytes: int
    finalizer_record_count: int
    relocation_count: int
    artifact_sha256: str
    makespan_cycles: int


_INTEGRATION_GOLDENS = {
    Stage1aCase.P1: _IntegrationGolden(
        artifact_size_bytes=1694,
        finalizer_record_count=9,
        relocation_count=16,
        artifact_sha256=(
            "944657529885a9e4a34ade89aeba7bac0ffbbc265456df71da482d54931f9dfe"
        ),
        makespan_cycles=164,
    ),
    Stage1aCase.K1: _IntegrationGolden(
        artifact_size_bytes=21230,
        finalizer_record_count=156,
        relocation_count=287,
        artifact_sha256=(
            "d83e0b498882fd0bdb93d6df56a92c2900620e792c542331c369d34006dc3911"
        ),
        makespan_cycles=1633,
    ),
    Stage1aCase.PD1: _IntegrationGolden(
        artifact_size_bytes=4448,
        finalizer_record_count=30,
        relocation_count=44,
        artifact_sha256=(
            "eb14f21bcb9a20d47e0a54904587007253ef0eceee7d42a4e374b124c852c52f"
        ),
        makespan_cycles=473,
    ),
}


def _fail(case: Stage1aCase, message: str) -> None:
    raise RuntimeError(f"[STAGE1A {case.value}] FAIL: {message}")


def _validate_integration_golden(
    case: Stage1aCase, observed: _IntegrationGolden
) -> None:
    expected = _INTEGRATION_GOLDENS[case]
    if observed != expected:
        _fail(
            case,
            "reviewed integration golden changed: "
            f"observed={observed!r} expected={expected!r}",
        )


def _executable(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _run(
    case: Stage1aCase,
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    expect_success: bool,
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
    if (completed.returncode == 0) != expect_success:
        expected = "success" if expect_success else "failure"
        _fail(
            case,
            f"expected {expected} from {' '.join(command)}; "
            f"exit={completed.returncode}\n{completed.stdout}",
        )
    return completed


def _row(line: str, prefix: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line[len(prefix) :].strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def _rows(output: str, prefix: str) -> list[dict[str, str]]:
    result = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position < 0:
            continue
        normalized = line[position:].split(" | ", 1)[0].rstrip()
        if normalized.endswith("."):
            normalized = normalized[:-1]
        result.append(_row(normalized, prefix))
    return result


def _number(case: Stage1aCase, row: dict[str, str], key: str) -> int:
    try:
        return int(row[key], 10)
    except (KeyError, ValueError) as error:
        _fail(case, f"missing/non-decimal {key!r} in {row}")
        raise AssertionError from error


def _parse_d2d(
    case: Stage1aCase,
    output: str,
    *,
    expected_data_flows: int,
) -> _D2DEvidence:
    typed = _rows(output, _D2D_TYPE_PREFIX)
    behavioral = _rows(output, _D2D_BEHA_PREFIX)
    if len(typed) != 1:
        _fail(case, "D2D_TYPE must appear exactly once")
    if len(behavioral) > 1:
        _fail(case, "D2D_BEHA may appear at most once")
    data_in = _number(case, typed[0], "data_in")
    data_out = _number(case, typed[0], "data_out")
    if data_in != data_out:
        _fail(
            case,
            f"D2D data_in/data_out disagree: {data_in}/{data_out}",
        )
    expected_packets = {
        Stage1aCase.P1: 0,
        Stage1aCase.K1: 0,
        Stage1aCase.PD1: 4,
    }[case]
    if data_out != expected_packets:
        _fail(
            case,
            f"D2D packet count changed: {data_out} != {expected_packets}",
        )
    if behavioral:
        observed_flows = _number(case, behavioral[0], "data_flows")
        logical_packets = _number(
            case, behavioral[0], "logical_data_packets"
        )
        if (
            observed_flows != expected_data_flows
            or logical_packets != data_out
        ):
            _fail(
                case,
                "D2D_BEHA disagrees with static flow/TYPE packet closure: "
                f"{observed_flows}/{logical_packets} != "
                f"{expected_data_flows}/{data_out}",
            )
    return _D2DEvidence(
        data_flows=expected_data_flows,
        logical_data_packets=data_out,
        observed_bytes=data_out * P2P_PAYLOAD_FRAGMENT_BYTES,
    )

def _signature_counts(
    case: Stage1aCase, value: str, arity: int
) -> list[tuple[int, ...]]:
    result = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        try:
            parts = tuple(int(part, 10) for part in item.split(":"))
        except ValueError as error:
            _fail(case, f"non-decimal HOSTSIG item {item!r}")
            raise AssertionError from error
        if len(parts) != arity:
            _fail(case, f"bad HOSTSIG item {item!r}")
        result.append(parts)
    return sorted(result)


def _marker_lines(output: str) -> tuple[str, ...]:
    prefixes = (
        _STATUS_PREFIX,
        _PROBE_PREFIX,
        _MEMORY_PREFIX,
        _SIM_RESULT_PREFIX,
        _HOST_PREFIX,
        _HOSTSIG_PREFIX,
        _P5_DRAIN_PREFIX,
        _P5_TIMING_PREFIX,
        _COLL_DRAIN_PREFIX,
        _GLOBAL_DRAIN_PREFIX,
        _D2D_TYPE_PREFIX,
        _D2D_BEHA_PREFIX,
    )
    result = []
    for line in output.splitlines():
        positions = tuple(line.find(prefix) for prefix in prefixes)
        positions = tuple(position for position in positions if position >= 0)
        if positions:
            normalized = line[min(positions) :].split(" | ", 1)[0].rstrip()
            result.append(normalized[:-1] if normalized.endswith(".") else normalized)
    return tuple(result)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_data(case: Stage1aCase) -> Any:
    return {
        Stage1aCase.P1: build_parameter_foundation_case,
        Stage1aCase.K1: build_cross_action_kv_foundation_case,
        Stage1aCase.PD1: build_pd1_case,
    }[case]()


def _leaf_fragments(data: Any) -> tuple[CommandFragment, ...]:
    leaves = data.linked_profile.leaf_fragments
    if any(type(fragment) is not CommandFragment for fragment in leaves):
        raise RuntimeError("Stage1a runtime accepts command-fragment leaves only")
    return leaves


def _payloads(case: Stage1aCase, data: Any) -> tuple[tuple[str, bytes], ...]:
    if case is Stage1aCase.P1:
        values = ((data.state_ref, data.seed),)
    elif case is Stage1aCase.K1:
        values = data.state_payloads
    else:
        values = (*data.source_payloads, *data.destination_expected)
    if len(dict(values)) != len(values):
        _fail(case, "state payload refs are not unique")
    return tuple(sorted(values))


def _state_pairs(
    case: Stage1aCase,
    data: Any,
    payloads: tuple[tuple[str, bytes], ...],
) -> tuple[Stage1aStatePair, ...]:
    if case is not Stage1aCase.PD1:
        return ()
    payload_by_ref = dict(payloads)
    access_index = {access.id: access for access in data.graph.state_accesses}
    result = tuple(
        Stage1aStatePair(
            source_state_ref=access_index[
                contract.source_state_access_ref
            ].state_ref,
            destination_state_ref=access_index[
                contract.destination_state_access_ref
            ].state_ref,
            payload_sha256=hashlib.sha256(
                payload_by_ref[
                    access_index[contract.source_state_access_ref].state_ref
                ]
            ).hexdigest(),
        )
        for contract in data.contracts
    )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.source_state_ref,
                item.destination_state_ref,
            ),
        )
    )


def _opcode_counts(
    case: Stage1aCase,
    leaves: tuple[CommandFragment, ...],
) -> tuple[Stage1aOpcodeCount, ...]:
    counts = Counter(
        record.opcode
        for fragment in leaves
        for stream in fragment.core_streams
        for record in stream.records
    )
    result = tuple(
        Stage1aOpcodeCount(opcode, count)
        for opcode, count in sorted(counts.items(), key=lambda item: int(item[0]))
    )
    if not result or any(item.count == 0 for item in result):
        _fail(case, f"artifact opcode counter is empty/invalid: {result}")
    return result


def _target_counts(contract: ProgramIoContract) -> tuple[int, int, int, int]:
    return (
        sum(type(x.target) is ProgramHbmTarget for x in contract.initializations),
        sum(type(x.target) is ProgramSramTarget for x in contract.initializations),
        sum(type(x.target) is ProgramHbmTarget for x in contract.output_probes),
        sum(type(x.target) is ProgramSramTarget for x in contract.output_probes),
    )


def _probe_name(entry: Any) -> str:
    prefix = (
        "probe.hbm."
        if type(entry.target) is ProgramHbmTarget
        else "probe.sram."
    )
    return prefix + entry.id


def _probe_payloads(
    contract: ProgramIoContract,
) -> tuple[Stage1aStatePayload, ...]:
    blobs = {blob.id: blob for blob in contract.blobs}
    result = tuple(
        Stage1aStatePayload(
            state_ref=_probe_name(entry),
            size_bytes=entry.length_bytes,
            sha256=blobs[entry.blob_ref].sha256,
        )
        for entry in contract.output_probes
    )
    return tuple(sorted(result, key=lambda item: item.state_ref))


def _oracle(
    case: Stage1aCase,
    data: Any,
    contract: ProgramIoContract,
    opcode_counts: tuple[Stage1aOpcodeCount, ...],
) -> Stage1aOracle:
    raw_payloads = _payloads(case, data)
    payloads = tuple(
        Stage1aStatePayload(
            state_ref=state_ref,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        for state_ref, payload in raw_payloads
    )
    pairs = _state_pairs(case, data, raw_payloads)
    numeric = {
        Stage1aCase.P1: (1, 0, 128, 0, 0, 8, 0, 0, 128),
        Stage1aCase.K1: (4, 4, 128, 128, 0, 8, 8, 0, 0),
        Stage1aCase.PD1: (2, 2, 64, 64, 64, 4, 4, 4, 0),
    }[case]
    runtime_cores = tuple(
        sorted(binding.runtime_core_id for binding in data.manifest.core_bindings)
    )
    if runtime_cores != {
        Stage1aCase.P1: (0,),
        Stage1aCase.K1: (0,),
        Stage1aCase.PD1: (0, 16),
    }[case]:
        _fail(case, f"wrong active runtime cores: {runtime_cores}")
    counts = _target_counts(contract)
    result = Stage1aOracle.create(
        case=case,
        capability_status=CapabilityStatus.E2E_TIMING,
        state_count=len(payloads),
        state_payloads=payloads,
        expected_probe_payloads=_probe_payloads(contract),
        state_pairs=pairs,
        expected_dma_loads=numeric[0],
        expected_dma_stores=numeric[1],
        expected_opcode_counts=opcode_counts,
        expected_hbm_read_bytes=numeric[2],
        expected_hbm_write_bytes=numeric[3],
        expected_d2d_bytes=numeric[4],
        hbm_read_capacity_floor_cycles=numeric[5],
        hbm_write_capacity_floor_cycles=numeric[6],
        d2d_capacity_floor_cycles=numeric[7],
        gemm_flops=numeric[8],
        expected_sidecar_mode=ProgramIoMode.TIMING,
        expected_hbm_initialization_count=counts[0],
        expected_sram_initialization_count=counts[1],
        expected_hbm_probe_count=counts[2],
        expected_sram_probe_count=counts[3],
        expected_ack_counts=tuple(Stage1aCoreCount(core, 2) for core in runtime_cores),
        expected_done_counts=tuple(Stage1aCoreCount(core, 1) for core in runtime_cores),
        expected_drain_names=_DRAINS,
        timing_execution=True,
        state_transport_exact=True,
        compute_functional=False,
        model_functional=False,
        synthetic_pd=case is Stage1aCase.PD1,
        decode_start_boundary=(
            Stage1aDecodeStartBoundary.HOST_AFTER_ALL_DONE
            if case is Stage1aCase.PD1
            else None
        ),
        notes=(
            "production ProgramIo entries are counted in full",
            "state transport is byte-exact",
            "timing-only model execution",
        ),
    )
    result.validate()
    return result


def _runtime_hardware(
    case: Stage1aCase,
    data: Any,
    source: Path,
    destination: Path,
) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    profiles = data.graph.fabric.sram_profiles
    if len(profiles) != 1:
        _fail(case, f"runtime requires one shared SRAM profile: {profiles}")
    profile = profiles[0]
    memory = raw["memory"]
    sram = memory["sram"]
    memory["sram_size"] = profile.capacity_bytes
    sram.update(
        capacity_bytes=profile.capacity_bytes,
        allocation_alignment_bytes=profile.allocation_alignment_bytes,
        bank_count=profile.bank_count,
        bank_interleave_bytes=profile.bank_interleave_bytes,
        real_data_path=profile.real_data_path,
        manual_regions=profile.manual_regions,
        manual_memory_schedule=profile.manual_memory_schedule,
    )
    sram["regions"] = [
        {
            "name": region.name,
            "base_bytes": region.base_bytes,
            "size_bytes": region.size_bytes,
            "allocator": region.allocator.value,
            "spillable": region.spillable,
            "access": [initiator.value for initiator in region.access],
        }
        for region in profile.regions
    ]
    destination.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _dramsys_hardware(
    case: Stage1aCase, source: Path, destination: Path
) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    stacks = raw["memory_system"]["hbm_stacks"]
    if len(stacks) != 2:
        _fail(case, "DRAMSys negative requires two HBM stacks")
    for stack in stacks:
        stack["backend"] = "dramsys"
    destination.write_text(
        json.dumps(raw, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _rebuild_sidecar(data: Any, artifact_sha256: str) -> ProgramIoContract:
    placeholder = data.program_io
    result = ProgramIoContract.create(
        producer_pass=placeholder.producer_pass,
        mode=placeholder.mode,
        source_manifest=data.manifest,
        program_artifact_sha256=artifact_sha256,
        blobs=placeholder.blobs,
        initializations=placeholder.initializations,
        output_probes=placeholder.output_probes,
    )
    result.validate_against(data.manifest)
    return result


def _memory_oracle(
    case: Stage1aCase,
    data: Any,
    leaves: tuple[CommandFragment, ...],
) -> dict[int, dict[str, int]]:
    fields = (
        "lsu_issued",
        "lsu_completed",
        "lsu_hbm_read_bytes",
        "lsu_hbm_write_bytes",
        "lsu_sram_read_bytes",
        "lsu_sram_write_bytes",
        "lsu_residual",
        "dte_residual",
    )
    runtime_by_logical = {
        binding.logical_core: binding.runtime_core_id
        for binding in data.manifest.core_bindings
    }
    result = {
        core: {field: 0 for field in fields}
        for core in sorted(set(runtime_by_logical.values()))
    }
    fragment_index = {fragment.id: fragment for fragment in leaves}
    state_abis: dict[str, Any] = {}
    for fragment in leaves:
        for abi in fragment.state_abi:
            previous = state_abis.setdefault(abi.id, abi)
            if previous != abi:
                _fail(case, "conflicting StateABI definitions")
    for binding in data.manifest.state_operand_bindings:
        fragment = fragment_index[binding.fragment_id]
        streams = tuple(
            stream
            for stream in fragment.core_streams
            if stream.logical_core == binding.logical_core
        )
        if len(streams) != 1:
            _fail(case, "state binding did not resolve to one core stream")
        record = streams[0].records[binding.fragment_record_index]
        abi = state_abis[binding.state_abi_id]
        size_bytes = next(
            operand.literal_value
            for operand in record.operands
            if operand.name == "size_bytes"
        )
        if size_bytes != abi.size_bytes:
            _fail(case, "LSU record and StateABI byte counts disagree")
        row = result[runtime_by_logical[binding.logical_core]]
        row["lsu_issued"] += 1
        row["lsu_completed"] += 1
        if record.opcode is RecordOpcode.LSU_LOAD:
            row["lsu_hbm_read_bytes"] += size_bytes
            row["lsu_sram_write_bytes"] += size_bytes
        elif record.opcode is RecordOpcode.LSU_STORE:
            row["lsu_hbm_write_bytes"] += size_bytes
            row["lsu_sram_read_bytes"] += size_bytes
        else:
            _fail(case, f"state binding references {record.opcode}")
    return result


def _parse_memory(
    case: Stage1aCase,
    output: str,
    expected: dict[int, dict[str, int]],
) -> tuple[Stage1aMemoryEvidence, ...]:
    rows = _rows(output, _MEMORY_PREFIX)
    cores = tuple(_number(case, row, "core") for row in rows)
    if cores != tuple(expected):
        _fail(case, f"PROGRAM_MEMORY core closure changed: {cores}")
    result = []
    for row in rows:
        core = _number(case, row, "core")
        actual = {key: _number(case, row, key) for key in expected[core]}
        if actual != expected[core]:
            _fail(
                case,
                f"PROGRAM_MEMORY core {core} differs: "
                f"{actual} != {expected[core]}",
            )
        result.append(Stage1aMemoryEvidence(core, **actual))
    return tuple(result)


def _parse_probes(
    case: Stage1aCase,
    output: str,
    contract: ProgramIoContract,
    leaves: tuple[CommandFragment, ...],
) -> tuple[Stage1aProbeEvidence, ...]:
    rows = _rows(output, _PROBE_PREFIX)
    if len(rows) != len(contract.output_probes):
        _fail(case, f"wrong probe marker count: {rows}")
    entries = {entry.id: entry for entry in contract.output_probes}
    blobs = {blob.id: blob for blob in contract.blobs}
    state_abis = {
        abi.id: abi for fragment in leaves for abi in fragment.state_abi
    }
    result = []
    for row in rows:
        entry = entries.get(row.get("id", ""))
        if entry is None:
            _fail(case, f"unknown probe marker: {row}")
        blob = blobs[entry.blob_ref]
        expected_sha = blob.sha256
        actual_sha = row.get("checksum", "")
        valid = row.get("valid") == "1"
        exact = row.get("exact") == "1"
        passed = row.get("pass") == "1"
        if (
            row.get("expected_checksum") != expected_sha
            or actual_sha != expected_sha
            or not valid
            or not exact
            or not passed
            or _number(case, row, "bytes") != entry.length_bytes
        ):
            _fail(case, f"probe did not close exactly: {row}")
        if type(entry.target) is ProgramHbmTarget:
            abi = state_abis[entry.target.state_abi_id]
            if (
                _number(case, row, "die") != abi.die_id
                or _number(case, row, "address")
                != abi.address + entry.offset_bytes
            ):
                _fail(case, f"HBM probe location changed: {row}")
            target_kind = ProgramIoTargetKind.HBM
        elif type(entry.target) is ProgramSramTarget:
            if _number(case, row, "core") != entry.target.runtime_core_id:
                _fail(case, f"SRAM probe core changed: {row}")
            target_kind = ProgramIoTargetKind.SRAM
        else:
            _fail(case, f"unknown probe target: {entry.target!r}")
        result.append(
            Stage1aProbeEvidence(
                probe_id=_probe_name(entry),
                target_kind=target_kind,
                length_bytes=entry.length_bytes,
                expected_sha256=expected_sha,
                actual_sha256=actual_sha,
                all_bytes_valid=valid,
                exact_match=exact,
                passed=passed,
            )
        )
    return tuple(sorted(result, key=lambda item: item.probe_id))


def _parse_control(
    case: Stage1aCase,
    output: str,
    oracle: Stage1aOracle,
) -> tuple[int, Stage1aControlEvidence]:
    simulation = _rows(output, _SIM_RESULT_PREFIX)
    if len(simulation) != 1:
        _fail(case, f"SIM_RESULT must appear exactly once: {simulation}")
    makespan = _number(case, simulation[0], "makespan_cycles")
    if makespan <= 0:
        _fail(case, "makespan must be positive")
    if "[PROTO_WAIT]" in output or "End DONE reception" not in output:
        _fail(case, "DONE path did not close")

    host = _rows(output, _HOST_PREFIX)
    signatures = _rows(output, _HOSTSIG_PREFIX)
    if len(host) != 1 or len(signatures) != 1:
        _fail(case, "HOSTLANE/HOSTSIG must each appear once")
    expected_ack = sum(item.count for item in oracle.expected_ack_counts)
    expected_done = sum(item.count for item in oracle.expected_done_counts)
    if (
        _number(case, host[0], "ack_total"),
        _number(case, host[0], "done_total"),
        _number(case, host[0], "mismatch"),
    ) != (expected_ack, expected_done, 0):
        _fail(case, f"ACK/DONE totals changed: {host[0]}")
    done_rows = _signature_counts(case, signatures[0].get("done", ""), 2)
    ack_rows = _signature_counts(case, signatures[0].get("ack", ""), 3)
    done = tuple(Stage1aCoreCount(core, count) for core, count in done_rows)
    ack_by_core: Counter[int] = Counter()
    for core, _lane, count in ack_rows:
        ack_by_core[core] += count
    ack = tuple(
        Stage1aCoreCount(core, ack_by_core[core])
        for core in sorted(ack_by_core)
    )
    if ack != oracle.expected_ack_counts or done != oracle.expected_done_counts:
        _fail(case, f"ACK/DONE per-core closure changed: {ack}, {done}")

    timing = _rows(output, _P5_TIMING_PREFIX)
    if len(timing) != 1 or _number(case, timing[0], "residual") != 0:
        _fail(case, f"P2P timing drain changed: {timing}")
    endpoints = _rows(output, _P5_DRAIN_PREFIX)
    if oracle.expected_d2d_bytes:
        endpoint_cores = tuple(
            sorted(_number(case, row, "core") for row in endpoints)
        )
        done_cores = tuple(
            item.runtime_core_id for item in oracle.expected_done_counts
        )
        if endpoint_cores != done_cores or any(
            _number(case, row, "residual") for row in endpoints
        ):
            _fail(case, f"P2P endpoint drain changed: {endpoints}")
    elif endpoints and any(
        _number(case, row, "residual") for row in endpoints
    ):
        _fail(case, f"unexpected P2P endpoint residual: {endpoints}")

    collective = _rows(output, _COLL_DRAIN_PREFIX)
    if len(collective) != 1:
        _fail(case, "collective drain must appear exactly once")
    if any(
        _number(case, collective[0], key)
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
        _fail(case, f"collective residual changed: {collective[0]}")
    global_rows = _rows(output, _GLOBAL_DRAIN_PREFIX)
    residuals = {
        key: _number(case, row, key)
        for row in global_rows
        for key in ("router_residual", "d2d_link_residual")
        if key in row
    }
    if residuals != {"router_residual": 0, "d2d_link_residual": 0}:
        _fail(case, f"global drain changed: {global_rows}")
    control = Stage1aControlEvidence(
        ack_counts=ack,
        done_counts=done,
        drain_residuals=tuple(
            Stage1aNamedCount(name, 0) for name in _DRAINS
        ),
        all_done_boundary_reached=True,
    )
    control.validate()
    return makespan, control


def _validate_status(
    case: Stage1aCase,
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
) -> None:
    statuses = _rows(output, _STATUS_PREFIX)
    if tuple(row.get("phase") for row in statuses) != (
        "resolved",
        "applied",
        "verify",
    ):
        _fail(case, f"ProgramIo phase closure changed: {statuses}")
    for row in statuses:
        if (
            row.get("mode") != "timing"
            or _number(case, row, "initializations")
            != len(contract.initializations)
            or _number(case, row, "probes")
            != len(contract.output_probes)
            or row.get("pass") != "1"
        ):
            _fail(case, f"ProgramIo status failed: {row}")
    if (
        statuses[0].get("checksum") != artifact_sha256
        or statuses[1].get("checksum") != artifact_sha256
    ):
        _fail(case, "resolved/applied checksum is not actual artifact SHA")


def _pd_witness(
    case: Stage1aCase,
    data: Any,
    oracle: Stage1aOracle,
) -> Stage1aPdWitness | None:
    if case is not Stage1aCase.PD1:
        return None
    destination_refs = set(data.destination_access_ids)
    completion_ids = tuple(
        sorted(
            action.id
            for action in data.global_dag.actions
            if action.task_kind is SemanticTaskKind.DMA_OUT
            and isinstance(action.origin_ref, StateIoOrigin)
            and action.origin_ref.state_access_ref in destination_refs
        )
    )
    return Stage1aPdWitness(
        state_pairs=oracle.state_pairs,
        completion_action_ids=completion_ids,
        done_runtime_core_ids=tuple(
            item.runtime_core_id for item in oracle.expected_done_counts
        ),
        decode_start_boundary=Stage1aDecodeStartBoundary.HOST_AFTER_ALL_DONE,
        all_completion_actions_before_boundary=True,
    )


def _report_text_entries(
    case: Stage1aCase,
    *,
    oracle_text: str,
    runtime_report_text: str,
    finalizer_logs: tuple[str, str],
    resolver_log: str,
    runtime_logs: tuple[str, str],
    dramsys_negative_log: str | None,
) -> tuple[tuple[str, str], ...]:
    if len(finalizer_logs) != 2 or len(runtime_logs) != 2:
        _fail(case, "report evidence requires exactly two finalizer/runtime logs")
    if (case is Stage1aCase.PD1) != (dramsys_negative_log is not None):
        _fail(case, "DRAMSys negative log must identify PD1 exactly")
    for index, log in enumerate(runtime_logs):
        if _SIM_RESULT_PREFIX not in log or _D2D_TYPE_PREFIX not in log:
            _fail(
                case,
                f"runtime repeat {index} lacks SIM_RESULT/D2D_TYPE markers",
            )
    stem = case.value.lower()
    entries = [
        (f"{stem}.oracle.json", oracle_text),
        (f"{stem}.runtime.json", runtime_report_text),
        (f"{stem}.finalizer.0.log", finalizer_logs[0]),
        (f"{stem}.finalizer.1.log", finalizer_logs[1]),
        (f"{stem}.resolver.log", resolver_log),
        (f"{stem}.runtime.0.log", runtime_logs[0]),
        (f"{stem}.runtime.1.log", runtime_logs[1]),
    ]
    if dramsys_negative_log is not None:
        entries.append(
            (f"{stem}.dramsys-negative.log", dramsys_negative_log)
        )
    if any(type(value) is not str for _name, value in entries):
        _fail(case, "report evidence text entries must be strings")
    names = tuple(name for name, _value in entries)
    if (
        len(set(names)) != len(names)
        or any(
            Path(name).name != name or name.endswith(".npup")
            for name in names
        )
    ):
        _fail(case, "report evidence names must be unique flat non-NPUP files")
    return tuple(sorted(entries))


def _publish_report_texts(
    root: Path,
    case: Stage1aCase,
    entries: tuple[tuple[str, str], ...],
) -> None:
    if root.is_symlink():
        _fail(case, f"report root must not be a symlink: {root}")
    if root.exists() and not root.is_dir():
        _fail(case, f"report root must be a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    targets = tuple((root / name, value) for name, value in entries)
    collisions = tuple(
        path for path, _value in targets if path.is_symlink() or path.exists()
    )
    if collisions:
        _fail(
            case,
            "refusing to overwrite/symlink report evidence: "
            + ", ".join(str(path) for path in collisions),
        )
    for target, value in targets:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(value)


def _write_report(
    root: Path | None,
    case: Stage1aCase,
    oracle: Stage1aOracle,
    report: Stage1aRuntimeReport,
    *,
    finalizer_logs: tuple[str, str],
    resolver_log: str,
    runtime_logs: tuple[str, str],
    dramsys_negative_log: str | None,
) -> None:
    if root is None:
        return
    entries = _report_text_entries(
        case,
        oracle_text=canonical_json(oracle),
        runtime_report_text=canonical_json(report),
        finalizer_logs=finalizer_logs,
        resolver_log=resolver_log,
        runtime_logs=runtime_logs,
        dramsys_negative_log=dramsys_negative_log,
    )
    _publish_report_texts(root, case, entries)


def _run_case(
    args: argparse.Namespace,
) -> tuple[Stage1aOracle, Stage1aRuntimeReport]:
    case = Stage1aCase(args.case.upper())
    data = _case_data(case)
    repeat_data = _case_data(case)
    if canonical_json(data.manifest) != canonical_json(repeat_data.manifest):
        _fail(case, "formal case builder is not deterministic")
    leaves = _leaf_fragments(data)
    manifest_text = canonical_json(data.manifest)
    memory_expected = _memory_oracle(case, data, leaves)

    with tempfile.TemporaryDirectory(
        prefix=f"stage1a-{case.value.lower()}-",
        dir=args.runtime_root,
    ) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        artifact_paths = (
            directory / "program.0.npup",
            directory / "program.1.npup",
        )
        report_paths = (
            directory / "finalizer.0.json",
            directory / "finalizer.1.json",
        )
        sidecar_path = directory / "program_io.json"
        runtime_hardware_path = directory / "hardware.json"
        manifest_path.write_text(manifest_text, encoding="utf-8")
        _runtime_hardware(case, data, args.hardware, runtime_hardware_path)

        finalizer_reports = []
        finalizer_logs = []
        for artifact_path, report_path in zip(
            artifact_paths, report_paths, strict=True
        ):
            finalizer_execution = _run(
                case,
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
                timeout=60,
                expect_success=True,
            )
            finalizer_logs.append(finalizer_execution.stdout)
            finalizer_reports.append(
                json.loads(report_path.read_text(encoding="utf-8"))
            )
        artifacts = tuple(path.read_bytes() for path in artifact_paths)
        if artifacts[0] != artifacts[1]:
            _fail(case, "production finalizer bytes are not deterministic")
        if finalizer_reports[0] != finalizer_reports[1]:
            _fail(case, "production finalizer report is not deterministic")
        finalizer_report = finalizer_reports[0]
        artifact_sha256 = hashlib.sha256(artifacts[0]).hexdigest()
        if (
            finalizer_report.get("artifact_sha256") != artifact_sha256
            or finalizer_report.get("artifact_bytes") != len(artifacts[0])
            or finalizer_report.get("linked_manifest_id") != data.manifest.id
            or finalizer_report.get("linked_manifest_digest")
            != canonical_digest(data.manifest)
        ):
            _fail(
                case,
                f"finalizer report input/byte closure changed: "
                f"{finalizer_report}",
            )

        contract = _rebuild_sidecar(data, artifact_sha256)
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        resolver = _run(
            case,
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(artifact_paths[0]),
                str(sidecar_path),
            ],
            cwd=args.runtime_root,
            timeout=60,
            expect_success=True,
        )
        resolver_witness = (
            f"initializations={len(contract.initializations)} "
            f"probes={len(contract.output_probes)}"
        )
        if resolver_witness not in resolver.stdout:
            _fail(case, f"resolver lost entry counts: {resolver.stdout}")

        opcode_counts = _opcode_counts(case, leaves)
        opcode_count_by_kind = {
            item.opcode: item.count for item in opcode_counts
        }
        expected_data_flows = opcode_count_by_kind.get(
            RecordOpcode.DTE_SEND, 0
        )
        frozen_data_flows = {
            Stage1aCase.P1: 0,
            Stage1aCase.K1: 0,
            Stage1aCase.PD1: 2,
        }[case]
        if expected_data_flows != frozen_data_flows:
            _fail(
                case,
                "static DTE_SEND flow count changed: "
                f"{expected_data_flows} != {frozen_data_flows}",
            )
        oracle = _oracle(case, data, contract, opcode_counts)
        memories: tuple[Stage1aMemoryEvidence, ...] | None = None
        probes: tuple[Stage1aProbeEvidence, ...] | None = None
        control: Stage1aControlEvidence | None = None
        d2d: _D2DEvidence | None = None
        makespan = 0
        repeats = []
        runtime_logs = []
        for run_index in range(2):
            execution = _run(
                case,
                [
                    str(args.npusim),
                    "--program",
                    str(artifact_paths[0]),
                    "--linked-manifest",
                    str(manifest_path),
                    "--program-io",
                    str(sidecar_path),
                    "--hardware-config",
                    str(runtime_hardware_path),
                    "--simulation-config",
                    str(args.simulation),
                    "--mapping-config",
                    str(args.mapping),
                    "--trace-window",
                    "1000000",
                ],
                cwd=args.runtime_root,
                timeout=180,
                expect_success=True,
            )
            runtime_logs.append(execution.stdout)
            _validate_status(case, execution.stdout, artifact_sha256, contract)
            current_memory = _parse_memory(
                case, execution.stdout, memory_expected
            )
            current_probes = _parse_probes(
                case, execution.stdout, contract, leaves
            )
            current_makespan, current_control = _parse_control(
                case, execution.stdout, oracle
            )
            current_d2d = _parse_d2d(
                case,
                execution.stdout,
                expected_data_flows=expected_data_flows,
            )
            if memories is None:
                memories = current_memory
                probes = current_probes
                control = current_control
                d2d = current_d2d
                makespan = current_makespan
            elif (
                current_memory != memories
                or current_probes != probes
                or current_control != control
                or current_d2d != d2d
                or current_makespan != makespan
            ):
                _fail(case, "behavioral repeat evidence changed")
            repeats.append(
                Stage1aRepeatEvidence(
                    run_index=run_index,
                    makespan_cycles=current_makespan,
                    marker_digest=canonical_digest(
                        _marker_lines(execution.stdout)
                    ),
                    memory_digest=canonical_digest(current_memory),
                    probe_digest=canonical_digest(current_probes),
                    control_digest=canonical_digest(current_control),
                )
            )
        assert memories is not None
        assert probes is not None
        assert control is not None
        assert d2d is not None

        _validate_integration_golden(
            case,
            _IntegrationGolden(
                artifact_size_bytes=len(artifacts[0]),
                finalizer_record_count=int(finalizer_report["record_count"]),
                relocation_count=int(finalizer_report["relocation_count"]),
                artifact_sha256=artifact_sha256,
                makespan_cycles=makespan,
            ),
        )

        dramsys_negative_log = None
        if case is Stage1aCase.PD1:
            dramsys_path = directory / "hardware.dramsys.json"
            _dramsys_hardware(case, runtime_hardware_path, dramsys_path)
            negative = _run(
                case,
                [
                    str(args.npusim),
                    "--program",
                    str(artifact_paths[0]),
                    "--linked-manifest",
                    str(manifest_path),
                    "--program-io",
                    str(sidecar_path),
                    "--hardware-config",
                    str(dramsys_path),
                    "--simulation-config",
                    str(args.simulation),
                    "--mapping-config",
                    str(args.mapping),
                    "--trace-window",
                    "1000000",
                ],
                cwd=args.runtime_root,
                timeout=60,
                expect_success=False,
            )
            dramsys_negative_log = negative.stdout
            status = _rows(negative.stdout, _STATUS_PREFIX)
            if (
                not status
                or status[-1].get("pass") != "0"
                or "HBM backend does not support debug peeking"
                not in negative.stdout
                or _SIM_RESULT_PREFIX in negative.stdout
                or _MEMORY_PREFIX in negative.stdout
            ):
                _fail(
                    case,
                    "DRAMSys negative failed late or ambiguously: "
                    f"{negative.stdout}",
                )

        full_opcode_record_count = sum(item.count for item in opcode_counts)
        finalizer_record_count = int(finalizer_report["record_count"])
        if full_opcode_record_count != finalizer_record_count:
            _fail(
                case,
                "full leaf opcode count disagrees with finalizer record count: "
                f"{full_opcode_record_count} != {finalizer_record_count}",
            )
        artifact = Stage1aArtifactEvidence(
            linked_manifest_id=data.manifest.id,
            linked_manifest_digest=canonical_digest(data.manifest),
            program_artifact_sha256=artifact_sha256,
            artifact_size_bytes=len(artifacts[0]),
            record_count=finalizer_record_count,
            relocation_count=int(finalizer_report["relocation_count"]),
            opcode_counts=opcode_counts,
        )
        target_counts = _target_counts(contract)
        sidecar = Stage1aSidecarEvidence(
            contract_id=contract.id,
            contract_digest=canonical_digest(contract),
            mode=contract.mode,
            hbm_initialization_count=target_counts[0],
            sram_initialization_count=target_counts[1],
            hbm_probe_count=target_counts[2],
            sram_probe_count=target_counts[3],
        )
        report = Stage1aRuntimeReport.create(
            baseline_epoch=STAGE1A_BASELINE_EPOCH,
            case=case,
            capability_status=CapabilityStatus.E2E_TIMING,
            oracle_id=oracle.id,
            oracle_digest=canonical_digest(oracle),
            hardware_digest=_file_digest(runtime_hardware_path),
            simulation_digest=_file_digest(args.simulation),
            mapping_digest=_file_digest(args.mapping),
            artifact=artifact,
            sidecar=sidecar,
            memory=memories,
            probes=probes,
            control=control,
            observed_hbm_read_bytes=sum(
                item.lsu_hbm_read_bytes for item in memories
            ),
            observed_hbm_write_bytes=sum(
                item.lsu_hbm_write_bytes for item in memories
            ),
            observed_d2d_bytes=d2d.observed_bytes,
            repeat_count=2,
            makespan_cycles=makespan,
            repeats=tuple(repeats),
            timing_execution=True,
            state_transport_exact=True,
            compute_functional=False,
            model_functional=False,
            synthetic_pd=case is Stage1aCase.PD1,
            pd_witness=_pd_witness(case, data, oracle),
        )
        report.validate_against(oracle)
        _write_report(
            args.report_root,
            case,
            oracle,
            report,
            finalizer_logs=tuple(finalizer_logs),
            resolver_log=resolver.stdout,
            runtime_logs=tuple(runtime_logs),
            dramsys_negative_log=dramsys_negative_log,
        )

    print(
        f"[STAGE1A {case.value}] PASS: "
        f"artifact={report.artifact.artifact_size_bytes}B "
        f"finalizer_records={finalizer_report['record_count']} "
        f"finalizer_relocations={finalizer_report['relocation_count']} "
        f"LSU_LOAD={oracle.expected_dma_loads} "
        f"LSU_STORE={oracle.expected_dma_stores} "
        f"HBM_READ={report.observed_hbm_read_bytes} "
        f"HBM_WRITE={report.observed_hbm_write_bytes} "
        f"D2D={report.observed_d2d_bytes} "
        f"initializations={len(contract.initializations)} "
        f"probes={len(contract.output_probes)} "
        f"ACK={sum(item.count for item in report.control.ack_counts)} "
        f"DONE={sum(item.count for item in report.control.done_counts)} "
        f"repeat=2 makespan_cycles={report.makespan_cycles} "
        f"sha256={report.artifact.program_artifact_sha256}"
    )
    return oracle, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", required=True, choices=("p1", "k1", "pd1")
    )
    parser.add_argument("--npusim", required=True, type=_executable)
    parser.add_argument("--finalizer", required=True, type=_executable)
    parser.add_argument("--resolver", required=True, type=_executable)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--report-root", type=Path)
    args = parser.parse_args()
    for name in ("hardware", "simulation", "mapping"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"--{name} is not a file: {path}")
        setattr(args, name, path)
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.report_root is not None:
        args.report_root = args.report_root.absolute()
    _run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
