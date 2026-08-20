#!/usr/bin/env python3
"""Strict pre-runtime/runtime runner for the S3-Lite MoE backward preview."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from typing import Callable

from lite_moe_backward_cases import (
    LiteMoeBackwardCase,
    build_lite_moe_backward_case,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramIoContract,
    ProgramIoTargetKind,
)
from llm.frontend.wafer_frontend.schema.lite_moe_backward import (
    S3_LITE_MOE_BACKWARD_CASE_ID,
)


_PROGRAM_IO = "[PROGRAM_IO] "
_PROBE = "[PROGRAM_IO_PROBE] "
_MEMORY = "[PROGRAM_MEMORY] "
_SGD = "[TRAIN_SGD] "
_SIM = "[SIM_RESULT] "
_HOST = "[HOSTLANE] "
_HOSTSIG = "[HOSTSIG] "
_P5 = "[P5 P2P DRAIN] "
_P5_TIMING = "[P5 P2P TIMING DRAIN] "
_COLL = "[COLL_DRAIN] "
_DRAIN = "[DRAIN] "
_D2D_TYPE = "[D2D_TYPE] "
_D2D_LINK = "[D2D_LINK] "
_DONE = "End DONE reception"
_ACTIVE_CORES = (0, 16)


def _fail(message: str) -> None:
    raise RuntimeError(f"[S3-LITE MOE BACKWARD] FAIL: {message}")


def build_production_program_io(
    case: LiteMoeBackwardCase,
    artifact_sha256: str,
) -> ProgramIoContract:
    from llm.frontend.wafer_frontend.passes.program_io import (
        build_deterministic_timing_state_overrides,
        build_timing_program_io,
    )

    case.validate()
    seeds, expected = build_deterministic_timing_state_overrides(case.linked)
    if len(seeds) != 4 or expected or sum(map(len, seeds.values())) != 4096:
        _fail("backward trainable-state timing override quotient changed")
    result = build_timing_program_io(
        case.linked,
        artifact_sha256,
        state_seed_overrides=seeds,
        state_expected_overrides=expected,
    )
    result.validate_against(case.linked.manifest)
    if (
        len(result.blobs),
        len(result.initializations),
        len(result.output_probes),
        sum(item.length_bytes for item in result.initializations),
        sum(item.length_bytes for item in result.output_probes),
        sum(
            item.target.kind is ProgramIoTargetKind.SRAM
            for item in result.initializations
        ),
        sum(
            item.target.kind is ProgramIoTargetKind.HBM
            for item in result.initializations
        ),
    ) != (8, 32, 4, 25472, 4096, 28, 4):
        _fail("production ProgramIo count/byte/target quotient changed")
    if any(
        probe.target.kind is not ProgramIoTargetKind.HBM
        for probe in result.output_probes
    ):
        _fail("backward output probes must all target trainable HBM state")
    return result


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardMemory:
    core: int
    lsu_issued: int
    lsu_completed: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    sram_read_bytes: int
    sram_write_bytes: int
    lsu_residual: int = 0
    dte_residual: int = 0


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardSgd:
    core: int
    invocations: int
    element_count: int
    learning_rate_f64_bits: int
    sram_read_bytes: int
    sram_write_bytes: int


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardRuntimeExpectation:
    case: LiteMoeBackwardCase
    program_io: ProgramIoContract
    memory: tuple[LiteMoeBackwardMemory, ...]
    sgd: tuple[LiteMoeBackwardSgd, ...]
    request_packets: int = 4
    ack_packets: int = 8
    data_packets: int = 8
    timing_execution: bool = True
    functional_execution: bool = False


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardRuntimeObservation:
    makespan_cycles: int
    marker_digest: str
    memory: tuple[LiteMoeBackwardMemory, ...]
    sgd: tuple[LiteMoeBackwardSgd, ...]
    d2d_type: tuple[int, ...]
    ack_total: int
    done_total: int
    wgrad_static_count: int
    reduce_static_count: int
    timing_execution: bool
    functional_execution: bool


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardRepeatEvidence:
    first_marker_digest: str
    second_marker_digest: str
    repeat_count: int = 2


def build_mock_expectation(
    artifact_sha256: str,
) -> LiteMoeBackwardRuntimeExpectation:
    case = build_lite_moe_backward_case()
    program_io = build_production_program_io(case, artifact_sha256)
    return _runtime_expectation(case, program_io)


def _runtime_expectation(
    case: LiteMoeBackwardCase,
    program_io: ProgramIoContract,
) -> LiteMoeBackwardRuntimeExpectation:
    learning_bits = struct.unpack("<Q", struct.pack("<d", 0.001))[0]
    return LiteMoeBackwardRuntimeExpectation(
        case,
        program_io,
        tuple(
            LiteMoeBackwardMemory(core, 4, 4, 2048, 2048, 2048, 2048)
            for core in _ACTIVE_CORES
        ),
        tuple(
            LiteMoeBackwardSgd(
                0 if expert < 2 else 16,
                1,
                512,
                learning_bits,
                3072,
                1024,
            )
            for expert in range(4)
        ),
    )


def _row(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line[len(prefix):].strip().split():
        if "=" not in token:
            _fail(f"malformed token {token!r} in {prefix.strip()}")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            _fail(f"duplicate/empty field in {prefix.strip()}")
        result[key] = value.rstrip(",.")
    return result


def _rows(output: str, prefix: str) -> tuple[dict[str, str], ...]:
    result = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position >= 0:
            normalized = line[position:].split(" | ", 1)[0].rstrip(". ")
            result.append(_row(normalized, prefix))
    return tuple(result)


def _number(row: dict[str, str], key: str) -> int:
    try:
        result = int(row[key], 10)
    except (KeyError, ValueError) as error:
        _fail(f"missing/non-decimal {key!r} in {row}")
        raise AssertionError from error
    if result < 0:
        _fail(f"negative {key!r} in {row}")
    return result


def _exact_row(
    output: str, prefix: str, fields: tuple[str, ...],
) -> dict[str, str]:
    rows = _rows(output, prefix)
    if len(rows) != 1 or set(rows[0]) != set(fields):
        _fail(f"{prefix.strip()} exact row changed")
    return rows[0]


def _parse_signature(
    value: str, arity: int,
) -> tuple[tuple[int, ...], ...]:
    result = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        try:
            parts = tuple(int(part, 10) for part in item.split(":"))
        except ValueError as error:
            _fail(f"non-decimal HOSTSIG item {item!r}")
            raise AssertionError from error
        if len(parts) != arity or any(part < 0 for part in parts):
            _fail(f"bad HOSTSIG item {item!r}")
        result.append(parts)
    return tuple(sorted(result))


def observe_lite_moe_backward_runtime(
    output: str,
    expectation: LiteMoeBackwardRuntimeExpectation,
) -> LiteMoeBackwardRuntimeObservation:
    expectation.program_io.validate_against(expectation.case.linked.manifest)
    if "[PROTO_WAIT]" in output or "[D2D_BEHA]" in output:
        _fail("completed backward preview forbids PROTO_WAIT/behavioral D2D")
    if output.count(_DONE) != len(_ACTIVE_CORES):
        _fail("requires one DONE boundary per active core")
    io_rows = _rows(output, _PROGRAM_IO)
    if tuple(row.get("phase") for row in io_rows) != (
        "resolved", "applied", "verify",
    ):
        _fail("ProgramIo phase closure changed")
    for row in io_rows:
        if (
            set(row) != {
                "phase", "mode", "checksum", "initializations", "probes", "pass",
            }
            or row.get("mode") != "timing"
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("checksum", ""))
            or _number(row, "initializations")
            != len(expectation.program_io.initializations)
            or _number(row, "probes")
            != len(expectation.program_io.output_probes)
            or _number(row, "pass") != 1
        ):
            _fail("ProgramIo status changed")
    if any(
        row["checksum"] != expectation.program_io.program_artifact_sha256
        for row in io_rows[:2]
    ):
        _fail("ProgramIo resolved/applied checksum is not the actual artifact SHA")
    probes = _rows(output, _PROBE)
    state_abis = {}
    for linked in expectation.case.linked.manifest.fragments:
        fragment = getattr(linked, "fragment", linked)
        for abi in fragment.state_abi:
            previous = state_abis.setdefault(abi.id, abi)
            if previous != abi:
                _fail("manifest contains conflicting StateABI witnesses")
    expected_probes = {}
    for item in expectation.program_io.output_probes:
        if item.target.kind is not ProgramIoTargetKind.HBM:
            _fail("backward runtime probe must target HBM state")
        abi = state_abis.get(item.target.state_abi_id)
        if abi is None:
            _fail("backward runtime probe lacks a StateABI witness")
        expected_probes[item.id] = (item, abi)
    if len(probes) != len(expected_probes):
        _fail("requires four updated-down-weight probes")
    seen = set()
    for row in probes:
        if set(row) != {
            "id", "die", "address", "bytes", "expected_checksum", "checksum",
            "valid", "exact", "pass",
        }:
            _fail("probe fields changed")
        expected = expected_probes.get(row.get("id", ""))
        if (
            expected is None
            or _number(row, "die") != expected[1].die_id
            or _number(row, "address")
            != expected[1].address + expected[0].offset_bytes
            or _number(row, "bytes") != expected[0].length_bytes
            or row.get("expected_checksum") != row.get("checksum")
            or any(_number(row, key) != 1 for key in ("valid", "exact", "pass"))
        ):
            _fail("updated-down-weight probe changed")
        seen.add(expected[0].id)
    if seen != set(expected_probes):
        _fail("probe identity closure changed")
    memory_fields = (
        "core", "lsu_issued", "lsu_completed", "lsu_hbm_read_bytes",
        "lsu_hbm_write_bytes", "lsu_sram_read_bytes", "lsu_sram_write_bytes",
        "lsu_residual", "dte_residual",
    )
    memory_rows = _rows(output, _MEMORY)
    if len(memory_rows) != 2 or any(set(row) != set(memory_fields) for row in memory_rows):
        _fail("requires two exact PROGRAM_MEMORY rows")
    memory = tuple(sorted((
        LiteMoeBackwardMemory(
            _number(row, "core"), _number(row, "lsu_issued"),
            _number(row, "lsu_completed"), _number(row, "lsu_hbm_read_bytes"),
            _number(row, "lsu_hbm_write_bytes"), _number(row, "lsu_sram_read_bytes"),
            _number(row, "lsu_sram_write_bytes"), _number(row, "lsu_residual"),
            _number(row, "dte_residual"),
        )
        for row in memory_rows
    ), key=lambda item: item.core))
    if memory != expectation.memory or sum(item.hbm_write_bytes for item in memory) != 4096:
        _fail("HBM updated-weight accounting changed")
    sgd_fields = tuple(LiteMoeBackwardSgd.__dataclass_fields__)
    sgd_rows = _rows(output, _SGD)
    if len(sgd_rows) != 4 or any(set(row) != set(sgd_fields) for row in sgd_rows):
        _fail("requires four exact TRAIN_SGD rows")
    sgd = tuple(sorted((
        LiteMoeBackwardSgd(*(_number(row, field) for field in sgd_fields))
        for row in sgd_rows
    ), key=lambda item: (item.core, item.learning_rate_f64_bits)))
    if sgd != expectation.sgd:
        _fail("four expert SGD work contracts changed")
    d2d = _exact_row(
        output, _D2D_TYPE,
        ("request_in", "request_out", "ack_in", "ack_out", "data_in", "data_out"),
    )
    d2d_type = tuple(_number(d2d, key) for key in (
        "request_in", "request_out", "ack_in", "ack_out", "data_in", "data_out",
    ))
    if d2d_type != (
        expectation.request_packets, expectation.request_packets,
        expectation.ack_packets, expectation.ack_packets,
        expectation.data_packets, expectation.data_packets,
    ):
        _fail("128B/eight-packet remote gradient quotient changed")
    pattern = re.compile(
        r"\[D2D_LINK\] idx=(\d+) die(\d+)->die(\d+) dir=[A-Z]+ "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)\."
    )
    links = tuple(sorted(
        tuple(int(value, 10) for value in match.groups())
        for match in pattern.finditer(output)
    ))
    if links != (
        (0, 0, 1, 2, 2, 4, 4, 4, 4),
        (1, 1, 0, 2, 2, 4, 4, 4, 4),
    ):
        _fail("remote gradient directional link accounting changed")
    host = _exact_row(
        output, _HOST, ("done_total", "ack_total", "mismatch", "per_lane_done"),
    )
    done_total, ack_total = _number(host, "done_total"), _number(host, "ack_total")
    if (
        (done_total, ack_total, _number(host, "mismatch")) != (2, 4, 0)
        or host.get("per_lane_done") != "1,0,0,0,1,0,0,0"
    ):
        _fail("host ACK/DONE closure changed")
    signature = _exact_row(output, _HOSTSIG, ("done", "ack"))
    if _parse_signature(signature["done"], 2) != ((0, 1), (16, 1)):
        _fail("per-core DONE closure changed")
    ack_by_core: Counter[int] = Counter()
    for core, _lane, count in _parse_signature(signature["ack"], 3):
        ack_by_core[core] += count
    if tuple(sorted(ack_by_core.items())) != ((0, 2), (16, 2)):
        _fail("per-core ACK closure changed")
    endpoints = _rows(output, _P5)
    if (
        len(endpoints) != 2
        or tuple(sorted(_number(row, "core") for row in endpoints)) != _ACTIVE_CORES
        or any(set(row) != {"core", "residual"} or _number(row, "residual") for row in endpoints)
    ):
        _fail("P2P endpoint drain changed")
    timing = _exact_row(output, _P5_TIMING, ("residual",))
    collective = _exact_row(
        output, _COLL,
        ("tree_entries", "reduce_nodes", "barriers", "gather", "reduce_rx", "endpoints", "dte_tokens", "event"),
    )
    drains = _rows(output, _DRAIN)
    merged = {key: value for row in drains for key, value in row.items()}
    if (
        _number(timing, "residual")
        or any(_number(collective, key) for key in collective)
        or set(merged) != {"router_residual", "d2d_link_residual"}
        or any(_number(merged, key) for key in merged)
    ):
        _fail("runtime engines did not drain exactly")
    simulation = _exact_row(output, _SIM, ("makespan_cycles",))
    makespan = _number(simulation, "makespan_cycles")
    if makespan == 0:
        _fail("makespan must be positive")
    marker_prefixes = (
        _PROGRAM_IO, _PROBE, _MEMORY, _SGD, _SIM, _HOST, _HOSTSIG,
        _P5, _P5_TIMING, _COLL, _DRAIN, _D2D_TYPE, _D2D_LINK,
    )
    marker_lines = []
    for line in output.splitlines():
        positions = tuple(
            position for prefix in marker_prefixes
            if (position := line.find(prefix)) >= 0
        )
        if positions:
            marker_lines.append(line[min(positions):].split(" | ", 1)[0].rstrip(". "))
        elif _DONE in line:
            marker_lines.append(_DONE)
    return LiteMoeBackwardRuntimeObservation(
        makespan,
        hashlib.sha256("\n".join(marker_lines).encode("utf-8")).hexdigest(),
        memory,
        sgd,
        d2d_type,
        ack_total,
        done_total,
        8,
        4,
        True,
        False,
    )


def validate_runtime_repeat(
    first: LiteMoeBackwardRuntimeObservation,
    second: LiteMoeBackwardRuntimeObservation,
) -> LiteMoeBackwardRepeatEvidence:
    if first != second:
        _fail("runtime observation repeat changed")
    return LiteMoeBackwardRepeatEvidence(
        first.marker_digest, second.marker_digest,
    )


@dataclass(frozen=True, slots=True)
class LiteMoeBackwardProductionChain:
    case: LiteMoeBackwardCase
    linked: object
    program_io: ProgramIoContract | None
    expectation: LiteMoeBackwardRuntimeExpectation | None


def build_lite_moe_backward_production_chain(
    artifact_sha256: str | None,
) -> LiteMoeBackwardProductionChain:
    case = build_lite_moe_backward_case()
    if artifact_sha256 is None:
        return LiteMoeBackwardProductionChain(
            case, case.linked, None, None,
        )
    program_io = build_production_program_io(case, artifact_sha256)
    return LiteMoeBackwardProductionChain(
        case,
        case.linked,
        program_io,
        _runtime_expectation(case, program_io),
    )


ProductionBuilder = Callable[[str | None], LiteMoeBackwardProductionChain]
CommandRunner = Callable[[tuple[str, ...], Path, int], str]


def _run(command: tuple[str, ...], cwd: Path, timeout: int) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=False, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, timeout=timeout,
    )
    if completed.returncode:
        _fail(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def run_official_lite_moe_backward(
    args: argparse.Namespace,
    *,
    production_builder: ProductionBuilder = build_lite_moe_backward_production_chain,
    command_runner: CommandRunner = _run,
) -> tuple[LiteMoeBackwardRuntimeObservation, LiteMoeBackwardRepeatEvidence]:
    from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

    pre = production_builder(None)
    pre.case.validate()
    manifest = pre.linked.manifest
    with tempfile.TemporaryDirectory(prefix="s3-lite-moe-backward-", dir=args.runtime_root) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
        hardware_path.write_text(pre.case.forward.source.hardware_json, encoding="utf-8")
        mapping_path.write_text(pre.case.forward.source.mapping_text, encoding="utf-8")
        artifacts = (directory / "program.0.npup", directory / "program.1.npup")
        reports = (directory / "finalizer.0.json", directory / "finalizer.1.json")
        for artifact, report in zip(artifacts, reports, strict=True):
            command_runner((
                str(args.finalizer), "--input", str(manifest_path),
                "--output", str(artifact), "--report", str(report),
            ), args.runtime_root, 120)
        artifact_bytes = tuple(path.read_bytes() for path in artifacts)
        report_values = tuple(json.loads(path.read_text(encoding="utf-8")) for path in reports)
        if artifact_bytes[0] != artifact_bytes[1] or report_values[0] != report_values[1]:
            _fail("finalizer byte/report repeat changed")
        artifact_sha = hashlib.sha256(artifact_bytes[0]).hexdigest()
        leaf_streams = tuple(
            stream for fragment in manifest.fragments for stream in fragment.core_streams
        )
        expected_report = {
            "artifact_sha256": artifact_sha,
            "artifact_bytes": len(artifact_bytes[0]),
            "core_count": 2,
            "record_count": sum(len(stream.records) for stream in leaf_streams),
            "relocation_count": sum(len(stream.address_relocations) for stream in leaf_streams),
            "linked_manifest_id": manifest.id,
            "linked_manifest_digest": canonical_digest(manifest),
        }
        if any(report_values[0].get(key) != value for key, value in expected_report.items()):
            _fail("finalizer report exact closure changed")
        actual = production_builder(artifact_sha)
        if (
            actual.linked != pre.linked
            or actual.program_io is None
            or actual.expectation is None
        ):
            _fail("actual-SHA production rebuild changed")
        sidecar = directory / "program_io.json"
        sidecar.write_text(canonical_json(actual.program_io), encoding="utf-8")
        resolver = command_runner((
            str(args.resolver), "--resolve", str(manifest_path),
            str(artifacts[0]), str(sidecar),
        ), args.runtime_root, 120)
        if (
            f"initializations={len(actual.program_io.initializations)}" not in resolver
            or f"probes={len(actual.program_io.output_probes)}" not in resolver
        ):
            _fail("resolver lost ProgramIo exact counts")
        observations = []
        for _index in range(2):
            output = command_runner((
                str(args.npusim), "--program", str(artifacts[0]),
                "--linked-manifest", str(manifest_path),
                "--program-io", str(sidecar),
                "--hardware-config", str(hardware_path),
                "--simulation-config", str(args.simulation),
                "--mapping-config", str(mapping_path),
                "--trace-window", "1000000",
            ), args.runtime_root, args.timeout)
            observations.append(observe_lite_moe_backward_runtime(
                output, actual.expectation,
            ))
        return observations[0], validate_runtime_repeat(
            observations[0], observations[1],
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the exact S3-Lite MoE backward timing preview twice",
    )
    parser.add_argument(
        "--finalizer",
        type=Path,
        default=Path("build/npusim_program_finalizer"),
    )
    parser.add_argument(
        "--resolver",
        type=Path,
        default=Path("build/npusim_program_io_selftest"),
    )
    parser.add_argument("--npusim", type=Path, default=Path("build/npusim"))
    parser.add_argument(
        "--simulation",
        type=Path,
        default=Path("llm/test/sram/simulation.json"),
    )
    parser.add_argument("--runtime-root", type=Path, default=Path("."))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    observation, repeat = run_official_lite_moe_backward(args)
    print(json.dumps({
        "case_id": S3_LITE_MOE_BACKWARD_CASE_ID,
        "marker_digest": observation.marker_digest,
        "makespan_cycles": observation.makespan_cycles,
        "repeat_count": repeat.repeat_count,
        "timing_execution": observation.timing_execution,
        "functional_execution": observation.functional_execution,
    }, sort_keys=True))
    print("[S3-LITE MOE BACKWARD] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LiteMoeBackwardMemory",
    "LiteMoeBackwardProductionChain",
    "LiteMoeBackwardRepeatEvidence",
    "LiteMoeBackwardRuntimeExpectation",
    "LiteMoeBackwardRuntimeObservation",
    "LiteMoeBackwardSgd",
    "build_mock_expectation",
    "build_lite_moe_backward_production_chain",
    "observe_lite_moe_backward_runtime",
    "build_production_program_io",
    "run_official_lite_moe_backward",
    "validate_runtime_repeat",
]
