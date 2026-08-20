#!/usr/bin/env python3
"""Run formal Stage 4 PD-F/PDS/PDR timing cases twice on production npusim."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
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
    build_stage4_pd_oracle,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (  # noqa: E402
    CommandFragment,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.capability import (  # noqa: E402
    CapabilityStatus,
)
from llm.frontend.wafer_frontend.schema.ir2 import (  # noqa: E402
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
)
from llm.frontend.wafer_frontend.schema.program_io import (  # noqa: E402
    ProgramHbmTarget,
    ProgramIoContract,
    ProgramIoMode,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
)
from llm.frontend.wafer_frontend.schema.stage4_pd_evidence import (  # noqa: E402
    STAGE4_PD_BASELINE_EPOCH,
    Stage4PdArtifactEvidence,
    Stage4PdControlEvidence,
    Stage4PdCoreCount,
    Stage4PdD2DEvidence,
    Stage4PdD2DLinkEvidence,
    Stage4PdEndpointRouteEvidence,
    Stage4PdMemoryEvidence,
    Stage4PdNamedDigest,
    Stage4PdNamedCount,
    Stage4PdProgramIoEvidence,
    Stage4PdRepeatEvidence,
    Stage4PdRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.state_transfer import (  # noqa: E402
    SegmentedKvStateTransferContract,
    SlicedKvStateTransferContract,
)
from stage4_pd_cases import (  # noqa: E402
    Stage4PdCase,
    Stage4PdCaseKind,
    build_stage4_pd_case,
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
_PACKET_BYTES = 16
_DRAINS = ("collective", "global", "p2p", "timing")


@dataclass(frozen=True, slots=True)
class _StaticWitness:
    global_action_count: int
    artifact_action_count: int
    fragment_count: int
    record_count: int
    runtime_relocation_count: int
    address_relocation_count: int
    address_operand_binding_count: int
    state_operand_binding_count: int
    runtime_cores: tuple[int, ...]
    initialization_count: int
    probe_count: int
    state_transfer_count: int
    state_transfer_bytes: int
    endpoint_route_count: int


@dataclass(frozen=True, slots=True)
class _RuntimeObservation:
    makespan_cycles: int
    marker_digest: str
    memory: tuple[Stage4PdMemoryEvidence, ...]
    program_io: Stage4PdProgramIoEvidence
    control: Stage4PdControlEvidence
    d2d: Stage4PdD2DEvidence


_STATIC_GOLDENS = {
    Stage4PdCaseKind.FUSED: _StaticWitness(
        92, 92, 92, 322, 0, 602, 560, 42, (0,), 105, 2, 0, 0, 0
    ),
    Stage4PdCaseKind.PDS: _StaticWitness(
        100, 100, 96, 330, 24, 602, 564, 38, (0, 16),
        120, 2, 4, 1024, 1
    ),
    Stage4PdCaseKind.PDR: _StaticWitness(
        428, 396, 152, 971, 858, 1267, 1210, 57, (0, 16, 32),
        228, 3, 8, 1024, 2
    ),
}


def _case_label(kind: Stage4PdCaseKind) -> str:
    return (
        "PDR TP2_TO_TP1"
        if kind is Stage4PdCaseKind.PDR
        else f"{kind.value.upper()} TP1"
    )


def _fail(kind: Stage4PdCaseKind, message: str) -> None:
    raise RuntimeError(f"[STAGE4 {_case_label(kind)}] FAIL: {message}")


def _select_case(value: str) -> Stage4PdCaseKind:
    return Stage4PdCaseKind(value)


def _transfer_units(
    contract: SlicedKvStateTransferContract | SegmentedKvStateTransferContract,
) -> tuple[tuple[int | None, int], ...]:
    if type(contract) is SlicedKvStateTransferContract:
        return ((None, contract.bytes),)
    if type(contract) is SegmentedKvStateTransferContract:
        return tuple(
            (segment_index, segment.bytes)
            for segment_index, segment in enumerate(contract.segments)
        )
    raise TypeError("state transfer must be sliced or segmented")


def _executable(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _run(
    kind: Stage4PdCaseKind,
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


_REPORT_INPUT_SCHEMA_VERSION = "wafer_frontend.stage4_pd_report_inputs/v2"


def _expected_report_names(kind: Stage4PdCaseKind) -> tuple[str, ...]:
    del kind
    return tuple(
        sorted(
            (
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
            )
        )
    )


def _reject_symlink_ancestors(path: Path, kind: Stage4PdCaseKind) -> None:
    candidate = path.absolute()
    for ancestor in (candidate, *candidate.parents):
        if ancestor.is_symlink():
            _fail(kind, f"report path has symlink ancestor: {ancestor}")


def _publish_report_texts(
    root: Path,
    kind: Stage4PdCaseKind,
    entries: tuple[tuple[str, str], ...],
) -> None:
    _reject_symlink_ancestors(root, kind)
    if root.exists() or root.is_symlink():
        _fail(kind, f"report root must not already exist: {root}")
    names = tuple(name for name, _value in entries)
    if (
        names != _expected_report_names(kind)
        or len(names) != len(set(names))
        or any(Path(name).name != name or name.endswith(".npup") for name in names)
        or any(type(value) is not str for _name, value in entries)
    ):
        _fail(kind, "report evidence set is not the exact reviewed flat text set")
    _reject_symlink_ancestors(root.parent, kind)
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
        if actual != names or any(path.suffix == ".npup" for path in staging.iterdir()):
            _fail(kind, "report evidence set is incomplete or contains NPUP")
        if root.exists() or root.is_symlink():
            _fail(kind, "report root appeared before atomic publication")
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            _fail(kind, "atomic no-clobber report publication requires Linux renameat2")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(staging), -100, os.fsencode(root), 1) != 0:
            number = ctypes.get_errno()
            if number == errno.EEXIST:
                _fail(kind, f"report root appeared before atomic publication: {root}")
            _fail(kind, f"atomic report publication failed with errno {number}")


def _report_text_entries(
    kind: Stage4PdCaseKind,
    *,
    case: Stage4PdCase,
    oracle: object,
    runtime_report: Stage4PdRuntimeReport,
    contract: ProgramIoContract,
    finalization: dict[str, object],
    finalizer_logs: tuple[str, str],
    resolver_log: str,
    runtime_logs: tuple[str, str],
) -> tuple[tuple[str, str], ...]:
    if len(finalizer_logs) != 2 or len(runtime_logs) != 2:
        _fail(kind, "report evidence requires exactly two repeat logs")
    if not resolver_log:
        _fail(kind, "resolver evidence log must be non-empty")
    for index, log in enumerate(runtime_logs):
        if _SIM not in log or _MEMORY not in log or _STATUS not in log:
            _fail(kind, f"runtime evidence log {index} lacks exact markers")
    inputs = {
        "schema_version": _REPORT_INPUT_SCHEMA_VERSION,
        "case_kind": kind.value,
        "report_id": runtime_report.id,
        "report_digest": canonical_digest(runtime_report),
        "tool_digests": runtime_report.tool_digests,
        "input_digests": runtime_report.input_digests,
    }
    entries = (
        ("actual_sha_program_io.json", canonical_json(contract)),
        ("finalization.json", canonical_json(finalization)),
        ("finalizer.0.log", finalizer_logs[0]),
        ("finalizer.1.log", finalizer_logs[1]),
        ("input_digests.json", canonical_json(inputs)),
        ("linked_manifest.json", canonical_json(case.manifest)),
        ("mapping.spec", case.runtime_hardware_inputs.mapping_text),
        ("model_spec.json", canonical_json(case.spec)),
        ("oracle.json", canonical_json(oracle)),
        ("plan.json", canonical_json(case.pd_plan)),
        (
            "resolved_hardware.json",
            case.runtime_hardware_inputs.hardware_json,
        ),
        ("resolver.log", resolver_log),
        ("runtime.0.log", runtime_logs[0]),
        ("runtime.1.log", runtime_logs[1]),
        ("runtime_report.json", canonical_json(runtime_report)),
    )
    names = tuple(sorted(name for name, _value in entries))
    if names != _expected_report_names(kind):
        _fail(kind, "report entry builder lost its exact reviewed file set")
    return tuple(sorted(entries))


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
        if position < 0:
            continue
        normalized = line[position:].split(" | ", 1)[0].rstrip(". ")
        result.append(_row(normalized, prefix))
    return result


def _number(
    kind: Stage4PdCaseKind,
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
        _STATUS,
        _PROBE,
        _MEMORY,
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
            normalized = line[min(positions) :].split(" | ", 1)[0]
            result.append(normalized.rstrip(". "))
    return tuple(result)


def _leaf_fragments(case: Any) -> tuple[CommandFragment, ...]:
    leaves = case.profile.leaf_fragments
    if any(type(fragment) is not CommandFragment for fragment in leaves):
        _fail(case.kind, "linked carrier contains a non-CommandFragment leaf")
    return leaves


def _memory_expected(
    case: Any,
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
        for binding in case.manifest.core_bindings
    }
    result = {
        runtime_core: {field: 0 for field in fields}
        for runtime_core in sorted(set(runtime_by_logical.values()))
    }
    fragments = {fragment.id: fragment for fragment in leaves}
    state_abis: dict[str, Any] = {}
    for fragment in leaves:
        for abi in fragment.state_abi:
            previous = state_abis.setdefault(abi.id, abi)
            if previous != abi:
                _fail(case.kind, "conflicting StateABI definitions")
    for binding in case.manifest.state_operand_bindings:
        fragment = fragments[binding.fragment_id]
        streams = tuple(
            stream
            for stream in fragment.core_streams
            if stream.logical_core == binding.logical_core
        )
        if len(streams) != 1:
            _fail(case.kind, "state binding must resolve to one core stream")
        record = streams[0].records[binding.fragment_record_index]
        abi = state_abis[binding.state_abi_id]
        size_bytes = next(
            operand.literal_value
            for operand in record.operands
            if operand.name == "size_bytes"
        )
        if size_bytes <= 0 or size_bytes > abi.size_bytes:
            _fail(case.kind, "LSU record size exceeds its StateABI")
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
            _fail(case.kind, f"state binding references {record.opcode}")
    return result


def _validate_wait_ancestry(case: Any) -> None:
    actions = case.global_carrier.global_dag.actions
    transfers = case.profile.lowering_context.projection.state_transfers
    if case.kind is Stage4PdCaseKind.FUSED:
        if any(
            isinstance(action.origin_ref, StateTransferOrigin)
            for action in actions
        ):
            _fail(case.kind, "fused case contains a state-transfer action")
        return
    action_by_task = {
        (action.source.dag_id, action.source.task_id): action
        for action in actions
    }
    routes = {route.id: route for route in case.graph.cross_routes}
    for contract in transfers:
        route = routes[contract.cross_group_route_ref]
        destination_dma = next(
            action
            for action in actions
            if isinstance(action.origin_ref, StateIoOrigin)
            and action.origin_ref.state_access_ref
            == contract.destination_state_access_ref
        )
        assert destination_dma.dma is not None
        for segment_index, _unit_bytes in _transfer_units(contract):
            matches = tuple(
                action
                for action in actions
                if isinstance(action.origin_ref, StateTransferOrigin)
                and action.origin_ref.state_transfer_ref == contract.id
                and action.origin_ref.segment_index == segment_index
            )
            expected_counts = Counter(
                {
                    SemanticTaskKind.SEND: 1,
                    SemanticTaskKind.RECV: 1,
                    SemanticTaskKind.WAIT: 1,
                }
            )
            if len(route.die_path) > 2:
                expected_counts[SemanticTaskKind.TRANSIT] = (
                    len(route.die_path) - 2
                )
            counts = Counter(action.task_kind for action in matches)
            if counts != expected_counts:
                _fail(
                    case.kind,
                    f"transfer quotient changed for "
                    f"{contract.id}/segment={segment_index}",
                )
            wait = next(
                action
                for action in matches
                if action.task_kind is SemanticTaskKind.WAIT
            )
            for task_id in destination_dma.dma.access_task_refs:
                target = action_by_task[
                    (destination_dma.source.dag_id, task_id)
                ]
                if wait.id not in target.deps:
                    _fail(
                        case.kind,
                        f"decode consumer {target.id} is not gated by "
                        f"{wait.id}",
                    )


def _validate_static_case(case: Any, oracle: Any) -> _StaticWitness:
    leaves = _leaf_fragments(case)
    runtime_relocations = sum(
        len(stream.runtime_relocations)
        for fragment in leaves
        for stream in fragment.core_streams
    )
    address_relocations = sum(
        len(stream.address_relocations)
        for fragment in leaves
        for stream in fragment.core_streams
    )
    transfers = case.profile.lowering_context.projection.state_transfers
    observed = _StaticWitness(
        global_action_count=len(case.global_carrier.global_dag.actions),
        artifact_action_count=len(
            {
                record.source_global_action_id
                for stream in case.manifest.core_streams
                for record in stream.records
            }
        ),
        fragment_count=len(leaves),
        record_count=sum(
            len(stream.records)
            for fragment in leaves
            for stream in fragment.core_streams
        ),
        runtime_relocation_count=runtime_relocations,
        address_relocation_count=address_relocations,
        address_operand_binding_count=len(
            case.manifest.address_operand_bindings
        ),
        state_operand_binding_count=len(case.manifest.state_operand_bindings),
        runtime_cores=tuple(
            sorted(
                binding.runtime_core_id
                for binding in case.manifest.core_bindings
            )
        ),
        initialization_count=len(case.program_io.initializations),
        probe_count=len(case.program_io.output_probes),
        state_transfer_count=len(transfers),
        state_transfer_bytes=sum(contract.bytes for contract in transfers),
        endpoint_route_count=len(case.graph.cross_routes),
    )
    if observed != _STATIC_GOLDENS[case.kind]:
        _fail(
            case.kind,
            f"formal carrier/artifact structure changed: {observed!r}",
        )
    if (
        observed.state_transfer_count != oracle.state_transfer_count
        or observed.state_transfer_bytes != oracle.delivered_bytes
        or observed.endpoint_route_count != oracle.unique_endpoint_route_count
    ):
        _fail(case.kind, "formal transfer structure disagrees with PD oracle")
    _validate_wait_ancestry(case)
    return observed


def _rebuild_sidecar(case: Any, artifact_sha256: str) -> ProgramIoContract:
    state_seeds, state_expected = build_deterministic_timing_state_overrides(
        case.profile
    )
    if (
        tuple(sorted(state_seeds)) != case.state_seed_refs
        or tuple(sorted(state_expected)) != case.state_expected_refs
    ):
        _fail(case.kind, "deterministic state override refs changed")
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
        getattr(case.program_io, name) for name in non_sha_fields
    ):
        _fail(case.kind, "actual-SHA ProgramIo changed non-SHA semantics")
    return result


def _parse_memory(
    case: Any,
    output: str,
    expected: dict[int, dict[str, int]],
) -> tuple[Stage4PdMemoryEvidence, ...]:
    rows = _rows(output, _MEMORY)
    cores = tuple(_number(case.kind, row, "core") for row in rows)
    if cores != tuple(expected):
        _fail(case.kind, f"PROGRAM_MEMORY core closure changed: {cores}")
    result = []
    for row in rows:
        core = _number(case.kind, row, "core")
        actual = {
            field: _number(case.kind, row, field)
            for field in expected[core]
        }
        if actual != expected[core]:
            _fail(
                case.kind,
                f"PROGRAM_MEMORY core {core} changed: "
                f"{actual} != {expected[core]}",
            )
        result.append(Stage4PdMemoryEvidence(core, **actual))
    return tuple(result)


def _parse_program_io(
    case: Any,
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
) -> Stage4PdProgramIoEvidence:
    statuses = _rows(output, _STATUS)
    if tuple(row.get("phase") for row in statuses) != (
        "resolved",
        "applied",
        "verify",
    ):
        _fail(case.kind, f"ProgramIo phase closure changed: {statuses}")
    for row in statuses:
        if (
            row.get("mode") != "timing"
            or _number(case.kind, row, "initializations")
            != len(contract.initializations)
            or _number(case.kind, row, "probes")
            != len(contract.output_probes)
            or row.get("pass") != "1"
        ):
            _fail(case.kind, f"ProgramIo status failed: {row}")
    if (
        statuses[0].get("checksum") != artifact_sha256
        or statuses[1].get("checksum") != artifact_sha256
    ):
        _fail(case.kind, "resolved/applied checksum is not artifact SHA")

    rows = _rows(output, _PROBE)
    if len(rows) != len(contract.output_probes):
        _fail(case.kind, f"probe marker multiplicity changed: {rows}")
    entries = {entry.id: entry for entry in contract.output_probes}
    blobs = {blob.id: blob for blob in contract.blobs}
    state_abis = {
        abi.id: abi
        for fragment in _leaf_fragments(case)
        for abi in fragment.state_abi
    }
    seen = set()
    for row in rows:
        entry = entries.get(row.get("id", ""))
        if entry is None or entry.id in seen:
            _fail(case.kind, f"unknown/duplicate probe marker: {row}")
        seen.add(entry.id)
        expected_sha = blobs[entry.blob_ref].sha256
        if (
            row.get("expected_checksum") != expected_sha
            or row.get("checksum") != expected_sha
            or row.get("valid") != "1"
            or row.get("exact") != "1"
            or row.get("pass") != "1"
            or _number(case.kind, row, "bytes") != entry.length_bytes
        ):
            _fail(case.kind, f"probe did not close exactly: {row}")
        if type(entry.target) is ProgramSramTarget:
            if (
                _number(case.kind, row, "core")
                != entry.target.runtime_core_id
            ):
                _fail(case.kind, f"SRAM probe core changed: {row}")
        elif type(entry.target) is ProgramHbmTarget:
            abi = state_abis[entry.target.state_abi_id]
            if (
                _number(case.kind, row, "die") != abi.die_id
                or _number(case.kind, row, "address")
                != abi.address + entry.offset_bytes
            ):
                _fail(case.kind, f"HBM probe location changed: {row}")
        else:
            _fail(case.kind, f"unsupported ProgramIo target: {entry.target!r}")

    evidence = Stage4PdProgramIoEvidence(
        contract_id=contract.id,
        contract_digest=canonical_digest(contract),
        mode=ProgramIoMode.TIMING,
        hbm_initialization_count=sum(
            type(entry.target) is ProgramHbmTarget
            for entry in contract.initializations
        ),
        sram_initialization_count=sum(
            type(entry.target) is ProgramSramTarget
            for entry in contract.initializations
        ),
        hbm_probe_count=sum(
            type(entry.target) is ProgramHbmTarget
            for entry in contract.output_probes
        ),
        sram_probe_count=sum(
            type(entry.target) is ProgramSramTarget
            for entry in contract.output_probes
        ),
        all_probes_passed=True,
    )
    evidence.validate()
    return evidence


def _parse_signature(
    kind: Stage4PdCaseKind,
    value: str,
    arity: int,
) -> list[tuple[int, ...]]:
    result = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        try:
            parsed = tuple(int(part, 10) for part in item.split(":"))
        except ValueError as error:
            _fail(kind, f"non-decimal HOSTSIG item {item!r}")
            raise AssertionError from error
        if len(parsed) != arity:
            _fail(kind, f"bad HOSTSIG item {item!r}")
        result.append(parsed)
    return sorted(result)


def _parse_control(
    case: Any,
    output: str,
) -> tuple[int, Stage4PdControlEvidence]:
    simulation = _rows(output, _SIM)
    if len(simulation) != 1:
        _fail(case.kind, "SIM_RESULT must appear exactly once")
    makespan = _number(case.kind, simulation[0], "makespan_cycles")
    if (
        makespan <= 0
        or "[PROTO_WAIT]" in output
        or "End DONE reception" not in output
    ):
        _fail(case.kind, "simulation/DONE path did not close")
    runtime_by_logical = {
        binding.logical_core: binding.runtime_core_id
        for binding in case.manifest.core_bindings
    }
    ack_cores = tuple(
        sorted(
            runtime_by_logical[core]
            for core in case.manifest.envelope.expected_ack_cores
        )
    )
    done_cores = tuple(
        sorted(
            runtime_by_logical[core]
            for core in case.manifest.envelope.expected_done_cores
        )
    )
    host = _rows(output, _HOST)
    signatures = _rows(output, _HOSTSIG)
    if len(host) != 1 or len(signatures) != 1:
        _fail(case.kind, "HOSTLANE/HOSTSIG must each appear exactly once")
    if (
        _number(case.kind, host[0], "ack_total") != 2 * len(ack_cores)
        or _number(case.kind, host[0], "done_total") != len(done_cores)
        or _number(case.kind, host[0], "mismatch") != 0
    ):
        _fail(case.kind, f"ACK/DONE totals changed: {host[0]}")
    done = _parse_signature(
        case.kind, signatures[0].get("done", ""), 2
    )
    ack = _parse_signature(
        case.kind, signatures[0].get("ack", ""), 3
    )
    ack_by_core: Counter[int] = Counter()
    for core, _lane, count in ack:
        ack_by_core[core] += count
    if (
        done != [(core, 1) for core in done_cores]
        or sorted(ack_by_core.items())
        != [(core, 2) for core in ack_cores]
    ):
        _fail(case.kind, f"ACK/DONE per-core closure changed: {ack}, {done}")

    timing = _rows(output, _P5_TIMING)
    if (
        len(timing) != 1
        or _number(case.kind, timing[0], "residual") != 0
    ):
        _fail(case.kind, f"P2P timing drain changed: {timing}")
    endpoints = _rows(output, _P5)
    expected_endpoints = (
        () if case.kind is Stage4PdCaseKind.FUSED else done_cores
    )
    endpoint_cores = tuple(
        sorted(_number(case.kind, row, "core") for row in endpoints)
    )
    if endpoint_cores != expected_endpoints or any(
        _number(case.kind, row, "residual") for row in endpoints
    ):
        _fail(case.kind, f"P2P endpoint drain changed: {endpoints}")
    collective = _rows(output, _COLL)
    if len(collective) != 1 or any(
        _number(case.kind, collective[0], field)
        for field in (
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
        _fail(case.kind, f"collective drain changed: {collective}")
    drains = _rows(output, _DRAIN)
    residuals = {
        key: _number(case.kind, row, key)
        for row in drains
        for key in ("router_residual", "d2d_link_residual")
        if key in row
    }
    if residuals != {
        "router_residual": 0,
        "d2d_link_residual": 0,
    }:
        _fail(case.kind, f"global drain changed: {drains}")
    control = Stage4PdControlEvidence(
        ack_counts=tuple(Stage4PdCoreCount(core, 2) for core in ack_cores),
        done_counts=tuple(Stage4PdCoreCount(core, 1) for core in done_cores),
        drain_residuals=tuple(
            Stage4PdNamedCount(name, 0) for name in _DRAINS
        ),
        all_done_boundary_reached=True,
    )
    control.validate()
    return makespan, control


def _unique_semantic_flows(case: Any) -> tuple[Any, ...]:
    flows: dict[str, Any] = {}
    def transport_key(flow: Any) -> tuple[Any, ...]:
        return (
            flow.logical_channel,
            flow.pair_route_ref,
            flow.source_rank,
            flow.destination_rank,
            flow.source_die,
            flow.destination_die,
            flow.die_path,
            flow.tensor_slice,
            flow.bytes,
            flow.dtype,
        )
    for dag in case.profile.lowering_context.projection.dags:
        for flow in dag.flows:
            previous = flows.setdefault(flow.id, flow)
            if transport_key(previous) != transport_key(flow):
                _fail(case.kind, f"semantic flow {flow.id!r} is not exact")
    return tuple(flows[key] for key in sorted(flows))


def _expected_link_traffic(case: Any) -> dict[tuple[int, int], tuple[int, int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for flow in _unique_semantic_flows(case):
        if flow.bytes % _PACKET_BYTES:
            _fail(case.kind, f"semantic flow {flow.id!r} is not packet aligned")
        for source, destination in zip(flow.die_path, flow.die_path[1:]):
            row = result.setdefault((source, destination), [0, 0])
            row[0] += 1
            row[1] += flow.bytes // _PACKET_BYTES
    return {key: (value[0], value[1]) for key, value in result.items()}


def _parse_d2d(case: Any, oracle: Any, output: str) -> Stage4PdD2DEvidence:
    expected_links = _expected_link_traffic(case)
    unique_flows = _unique_semantic_flows(case)
    logical_packets = sum(flow.bytes for flow in unique_flows) // _PACKET_BYTES
    expected_physical = sum(value[1] for value in expected_links.values())
    expected_requests = sum(value[0] for value in expected_links.values())
    expected_flow_count = len(unique_flows)
    typed = _rows(output, _D2D_TYPE)
    if len(typed) != 1:
        _fail(case.kind, "D2D_TYPE must appear exactly once")
    values = {
        key: _number(case.kind, typed[0], key)
        for key in (
            "request_in",
            "request_out",
            "ack_in",
            "ack_out",
            "data_in",
            "data_out",
        )
    }
    expected = {
        "request_in": expected_requests,
        "request_out": expected_requests,
        "ack_in": 2 * expected_requests,
        "ack_out": 2 * expected_requests,
        "data_in": expected_physical,
        "data_out": expected_physical,
    }
    if values != expected:
        _fail(case.kind, f"D2D_TYPE count changed: {values} != {expected}")
    if _rows(output, _D2D_BEHA):
        _fail(case.kind, "matching Stage4 hardware emitted D2D_BEHA")

    pattern = re.compile(
        r"\[D2D_LINK\] idx=\d+ die(\d+)->die(\d+) dir=[A-Z?]+ "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)"
    )
    raw_links: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {}
    for match in pattern.finditer(output):
        source, destination, req_in, req_out, ack_in, ack_out, data_in, data_out = (
            int(value) for value in match.groups()
        )
        if (
            req_in != req_out
            or ack_in != ack_out
            or data_in != data_out
            or (source, destination) in raw_links
        ):
            _fail(
                case.kind,
                f"D2D raw link {source}->{destination} is unbalanced/duplicate",
            )
        raw_links[(source, destination)] = (
            req_in,
            req_out,
            ack_in,
            ack_out,
            data_in,
            data_out,
        )
    links = []
    expected_raw_keys = set(expected_links)
    expected_raw_keys.update(
        (destination, source) for source, destination in expected_links
    )
    for source, destination in sorted(expected_raw_keys):
        request_count, data_count = expected_links.get(
            (source, destination), (0, 0)
        )
        reverse_requests = expected_links.get(
            (destination, source), (0, 0)
        )[0]
        wanted = (
            request_count,
            request_count,
            2 * reverse_requests,
            2 * reverse_requests,
            data_count,
            data_count,
        )
        if raw_links.get((source, destination)) != wanted:
            _fail(
                case.kind,
                f"D2D directional link {source}->{destination} changed: "
                f"observed={raw_links.get((source, destination))} "
                f"wanted={wanted}",
            )
        links.append(
            Stage4PdD2DLinkEvidence(
                source_die_id=source,
                destination_die_id=destination,
                request_in_packets=request_count,
                request_out_packets=request_count,
                ack_in_packets=2 * reverse_requests,
                ack_out_packets=2 * reverse_requests,
                data_in_packets=data_count,
                data_out_packets=data_count,
            )
        )
    if set(raw_links) != expected_raw_keys:
        _fail(case.kind, f"D2D raw link closure changed: {raw_links}")
    links = sorted(
        links, key=lambda item: (item.source_die_id, item.destination_die_id)
    )
    if tuple(
        (item.source_die_id, item.destination_die_id)
        for item in links
    ) != tuple(sorted(expected_raw_keys)):
        _fail(case.kind, f"D2D link closure changed: {links}")
    endpoint_routes = tuple(
        Stage4PdEndpointRouteEvidence(
            source_rank=item.source_rank,
            destination_rank=item.destination_rank,
            logical_unique_bytes=item.logical_unique_bytes,
            delivered_bytes=item.delivered_bytes,
            state_transfer_count=item.state_transfer_count,
        )
        for item in oracle.endpoint_pair_metrics
    )
    result = Stage4PdD2DEvidence(
        flow_count=expected_flow_count,
        logical_packet_count=logical_packets,
        physical_packet_count=expected_physical,
        request_packet_count=expected_requests,
        ack_packet_count=2 * expected_requests,
        logical_bytes=logical_packets * _PACKET_BYTES,
        byte_hop_bytes=expected_physical * _PACKET_BYTES,
        state_transport_logical_bytes=oracle.delivered_bytes,
        state_transfer_count=oracle.state_transfer_count,
        unique_endpoint_route_count=oracle.unique_endpoint_route_count,
        endpoint_routes=endpoint_routes,
        links=tuple(links),
    )
    result.validate()
    return result


def _observe_runtime(
    case: Any,
    oracle: Any,
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
    memory_expected: dict[int, dict[str, int]],
) -> _RuntimeObservation:
    program_io = _parse_program_io(
        case, output, artifact_sha256, contract
    )
    memory = _parse_memory(case, output, memory_expected)
    makespan, control = _parse_control(case, output)
    d2d = _parse_d2d(case, oracle, output)
    lines = _marker_lines(output)
    expected_counts = {
        _STATUS: 3,
        _PROBE: len(contract.output_probes),
        _MEMORY: len(memory_expected),
        _SIM: 1,
        _HOST: 1,
        _HOSTSIG: 1,
        _P5_TIMING: 1,
        _COLL: 1,
        _D2D_TYPE: 1,
        _D2D_LINK: len(d2d.links),
    }
    for prefix, count in expected_counts.items():
        if sum(line.startswith(prefix.rstrip()) for line in lines) != count:
            _fail(case.kind, f"marker multiplicity changed for {prefix.strip()}")
    return _RuntimeObservation(
        makespan_cycles=makespan,
        marker_digest=hashlib.sha256(
            "\n".join(lines).encode("utf-8")
        ).hexdigest(),
        memory=memory,
        program_io=program_io,
        control=control,
        d2d=d2d,
    )


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_report(
    case: Any,
    oracle: Any,
    static: _StaticWitness,
    finalization: dict[str, object],
    artifact: bytes,
    contract: ProgramIoContract,
    observations: tuple[_RuntimeObservation, _RuntimeObservation],
    *,
    tool_digests: tuple[Stage4PdNamedDigest, ...],
    hardware_digest: str,
    simulation_digest: str,
    mapping_digest: str,
) -> Stage4PdRuntimeReport:
    first = observations[0]
    if observations[1] != first:
        _fail(case.kind, "runtime memory/control/D2D/makespan repeat changed")
    if (
        int(finalization["record_count"]) != static.record_count
        or int(finalization["relocation_count"])
        != static.address_relocation_count
    ):
        _fail(case.kind, f"finalizer structure changed: {finalization}")
    repeats = tuple(
        Stage4PdRepeatEvidence(
            run_index=index,
            makespan_cycles=observation.makespan_cycles,
            marker_digest=observation.marker_digest,
            memory_digest=canonical_digest(observation.memory),
            program_io_digest=canonical_digest(observation.program_io),
            control_digest=canonical_digest(observation.control),
            d2d_digest=canonical_digest(observation.d2d),
        )
        for index, observation in enumerate(observations)
    )
    report = Stage4PdRuntimeReport.create(
        baseline_epoch=STAGE4_PD_BASELINE_EPOCH,
        case_id={
            Stage4PdCaseKind.FUSED: "case.stage4.pd_f.tp1",
            Stage4PdCaseKind.PDS: "case.stage4.pds.tp1",
            Stage4PdCaseKind.PDR: "case.stage4.pdr.tp2_to_tp1",
        }[case.kind],
        mode=oracle.mode,
        reshard=oracle.reshard,
        capability_status=CapabilityStatus.E2E_TIMING,
        plan_id=case.pd_plan.id,
        plan_digest=canonical_digest(case.pd_plan),
        oracle_id=oracle.id,
        oracle_digest=canonical_digest(oracle),
        policy=case.policy,
        tool_digests=tool_digests,
        input_digests=(
            Stage4PdNamedDigest("hardware", hardware_digest),
            Stage4PdNamedDigest("manifest", canonical_digest(case.manifest)),
            Stage4PdNamedDigest("mapping", mapping_digest),
            Stage4PdNamedDigest("oracle", canonical_digest(oracle)),
            Stage4PdNamedDigest("plan", canonical_digest(case.pd_plan)),
            Stage4PdNamedDigest("policy", canonical_digest(case.policy)),
            Stage4PdNamedDigest("program_io", canonical_digest(contract)),
            Stage4PdNamedDigest("simulation", simulation_digest),
            Stage4PdNamedDigest("spec", canonical_digest(case.spec)),
        ),
        hardware_digest=hardware_digest,
        simulation_digest=simulation_digest,
        mapping_digest=mapping_digest,
        artifact=Stage4PdArtifactEvidence(
            linked_manifest_id=case.manifest.id,
            linked_manifest_digest=canonical_digest(case.manifest),
            program_artifact_sha256=hashlib.sha256(artifact).hexdigest(),
            artifact_size_bytes=len(artifact),
            action_count=static.artifact_action_count,
            fragment_count=static.fragment_count,
            record_count=static.record_count,
            runtime_relocation_count=static.runtime_relocation_count,
            address_relocation_count=static.address_relocation_count,
            relocation_count=(
                static.runtime_relocation_count
                + static.address_relocation_count
            ),
            address_operand_binding_count=(
                static.address_operand_binding_count
            ),
            state_operand_binding_count=static.state_operand_binding_count,
        ),
        program_io=first.program_io,
        memory=first.memory,
        control=first.control,
        d2d=first.d2d,
        repeat_count=2,
        makespan_cycles=first.makespan_cycles,
        repeats=repeats,
        timing_execution=True,
        state_transport_exact=True,
        program_io_boundary_exact=True,
        model_functional=False,
    )
    report.validate_against(
        case.pd_plan,
        oracle,
        case.manifest,
        case.planning_context,
        case.scheduling_context,
    )
    return report


def _run_case(
    args: argparse.Namespace,
) -> Stage4PdRuntimeReport:
    kind = _select_case(args.case)
    if args.report_root is not None:
        _reject_symlink_ancestors(args.report_root, kind)
        if args.report_root.exists() or args.report_root.is_symlink():
            _fail(
                kind,
                f"report root must not already exist: {args.report_root}",
            )
    case = build_stage4_pd_case(kind)
    oracle = build_stage4_pd_oracle(case.pd_plan)
    oracle.validate_against(case.pd_plan)
    static = _validate_static_case(case, oracle)
    leaves = _leaf_fragments(case)
    memory_expected = _memory_expected(case, leaves)
    manifest_text = canonical_json(case.manifest)

    with tempfile.TemporaryDirectory(
        prefix=f"stage4-{kind.value}-tp1-",
        dir=args.runtime_root,
    ) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        sidecar_path = directory / "program_io.json"
        artifact_paths = (
            directory / "program.0.npup",
            directory / "program.1.npup",
        )
        finalization_paths = (
            directory / "finalizer.0.json",
            directory / "finalizer.1.json",
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")
        hardware_path.write_text(
            case.runtime_hardware_inputs.hardware_json,
            encoding="utf-8",
        )
        mapping_path.write_text(
            case.runtime_hardware_inputs.mapping_text,
            encoding="utf-8",
        )
        finalizations = []
        finalizer_logs = []
        for artifact_path, finalization_path in zip(
            artifact_paths, finalization_paths, strict=True
        ):
            finalizer_run = _run(
                kind,
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
                timeout=args.timeout,
            )
            finalizer_logs.append(finalizer_run.stdout)
            finalizations.append(
                json.loads(
                    finalization_path.read_text(encoding="utf-8")
                )
            )
        artifacts = tuple(path.read_bytes() for path in artifact_paths)
        if artifacts[0] != artifacts[1] or finalizations[0] != finalizations[1]:
            _fail(kind, "finalizer artifact/report repeat changed")
        artifact_sha256 = hashlib.sha256(artifacts[0]).hexdigest()
        finalization = finalizations[0]
        if (
            finalization.get("artifact_sha256") != artifact_sha256
            or finalization.get("artifact_bytes") != len(artifacts[0])
            or finalization.get("linked_manifest_id") != case.manifest.id
            or finalization.get("linked_manifest_digest")
            != canonical_digest(case.manifest)
        ):
            _fail(kind, f"finalizer provenance changed: {finalization}")

        contract = _rebuild_sidecar(case, artifact_sha256)
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        resolver = _run(
            kind,
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(artifact_paths[0]),
                str(sidecar_path),
            ],
            cwd=args.runtime_root,
            timeout=args.timeout,
        )
        witness = (
            f"initializations={len(contract.initializations)} "
            f"probes={len(contract.output_probes)}"
        )
        if witness not in resolver.stdout:
            _fail(kind, f"resolver lost ProgramIo counts: {resolver.stdout}")

        observations = []
        runtime_logs = []
        for _run_index in range(2):
            execution = _run(
                kind,
                [
                    str(args.npusim),
                    "--program",
                    str(artifact_paths[0]),
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
                _observe_runtime(
                    case,
                    oracle,
                    execution.stdout,
                    artifact_sha256,
                    contract,
                    memory_expected,
                )
            )
            runtime_logs.append(execution.stdout)
        report = _build_report(
            case,
            oracle,
            static,
            finalization,
            artifacts[0],
            contract,
            (observations[0], observations[1]),
            tool_digests=tuple(
                Stage4PdNamedDigest(name, _file_digest(path))
                for name, path in (
                    ("finalizer", args.finalizer),
                    ("npusim", args.npusim),
                    ("resolver", args.resolver),
                    ("runner", Path(__file__).resolve()),
                )
            ),
            hardware_digest=_file_digest(hardware_path),
            simulation_digest=_file_digest(args.simulation),
            mapping_digest=_file_digest(mapping_path),
        )
        if args.report_root is not None:
            _publish_report_texts(
                args.report_root,
                kind,
                _report_text_entries(
                    kind,
                    case=case,
                    oracle=oracle,
                    runtime_report=report,
                    contract=contract,
                    finalization=finalization,
                    finalizer_logs=(finalizer_logs[0], finalizer_logs[1]),
                    resolver_log=resolver.stdout,
                    runtime_logs=(runtime_logs[0], runtime_logs[1]),
                ),
            )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(canonical_json(report), encoding="utf-8")
    print(
        f"[STAGE4 {_case_label(kind)}] PASS: "
        f"artifact={report.artifact.artifact_size_bytes}B "
        f"records={report.artifact.record_count} "
        f"relocations={report.artifact.relocation_count} "
        f"actions={report.artifact.action_count} "
        f"state_transfers={report.d2d.state_transfer_count} "
        f"state_D2D={report.d2d.state_transport_logical_bytes}B "
        f"endpoint_routes={report.d2d.unique_endpoint_route_count} "
        f"ACK={sum(item.count for item in report.control.ack_counts)} "
        f"DONE={sum(item.count for item in report.control.done_counts)} "
        f"repeat=2 makespan_cycles={report.makespan_cycles} "
        f"model_functional={int(report.model_functional)} "
        f"sha256={report.artifact.program_artifact_sha256}"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=tuple(kind.value for kind in Stage4PdCaseKind),
    )
    parser.add_argument("--npusim", required=True, type=_executable)
    parser.add_argument("--finalizer", required=True, type=_executable)
    parser.add_argument("--resolver", required=True, type=_executable)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    args.simulation = args.simulation.resolve()
    if not args.simulation.is_file():
        parser.error(f"--simulation is not a file: {args.simulation}")
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.report is not None:
        args.report = args.report.resolve()
    if args.report_root is not None:
        args.report_root = args.report_root.absolute()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    _run_case(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
