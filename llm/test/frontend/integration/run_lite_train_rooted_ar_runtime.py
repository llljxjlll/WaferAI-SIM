#!/usr/bin/env python3
"""Official S2-Lite DP2 rooted-all-reduce runtime driver.

The production path is deliberately injectable for focused tests.  The
default runner invokes the real finalizer twice, rebuilds ProgramIo from the
actual artifact SHA, resolves once, then executes the same artifact twice.
"""

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


_STATUS = "[PROGRAM_IO] "
_PROBE = "[PROGRAM_IO_PROBE] "
_MEMORY = "[PROGRAM_MEMORY] "
_CE_FORWARD = "[TRAIN_CE] "
_CE_BACKWARD = "[TRAIN_CE_BACKWARD] "
_SGD = "[TRAIN_SGD] "
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
_DONE = "End DONE reception"
_ACTIVE_CORES = (0, 16)


def _fail(message: str) -> None:
    raise RuntimeError(f"[S2-LITE DP2 ROOTED-AR] FAIL: {message}")


def _value(value: object) -> object:
    return getattr(value, "value", value)


@dataclass(frozen=True, slots=True)
class RootedArMemory:
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
class RootedArCeForward:
    core: int
    invocations: int
    rank_rows: int
    label_read_bytes: int
    loss_write_bytes: int


@dataclass(frozen=True, slots=True)
class RootedArCeBackward:
    core: int
    invocations: int
    rank_rows: int
    upstream_elements: int
    logits_read_bytes: int
    label_read_bytes: int
    upstream_read_bytes: int
    logits_grad_write_bytes: int


@dataclass(frozen=True, slots=True)
class RootedArSgd:
    core: int
    invocations: int
    element_count: int
    learning_rate_f64_bits: int
    sram_read_bytes: int
    sram_write_bytes: int


@dataclass(frozen=True, slots=True)
class RootedArRuntimeExpectation:
    memory: tuple[RootedArMemory, ...]
    ce_forward: tuple[RootedArCeForward, ...]
    ce_backward: tuple[RootedArCeBackward, ...]
    sgd: tuple[RootedArSgd, ...]
    d2d_packets: tuple[int, int, int]
    ack_total: int = 4
    done_total: int = 2


@dataclass(frozen=True, slots=True)
class RootedArRuntimeObservation:
    makespan_cycles: int
    marker_digest: str
    memory: tuple[RootedArMemory, ...]
    ce_forward: tuple[RootedArCeForward, ...]
    ce_backward: tuple[RootedArCeBackward, ...]
    sgd: tuple[RootedArSgd, ...]
    d2d_packets: tuple[int, int, int]
    ack_total: int
    done_total: int
    drained: bool


@dataclass(frozen=True, slots=True)
class RootedArRepeatEvidence:
    first_marker_digest: str
    second_marker_digest: str
    repeat_count: int = 2


@dataclass(frozen=True, slots=True)
class RootedArPreRuntime:
    case: object
    lowered: object
    linked: object
    program_io: object | None
    expectation: RootedArRuntimeExpectation


def _row(line: str, prefix: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line[len(prefix):].strip().split():
        if "=" not in token:
            _fail(f"malformed token {token!r} in {prefix.strip()}")
        key, value = token.split("=", 1)
        if not key or not value or key in fields:
            _fail(f"duplicate/empty field in {prefix.strip()}")
        fields[key] = value.rstrip(",.")
    return fields


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


def _parse_signature(value: str, arity: int) -> tuple[tuple[int, ...], ...]:
    result = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        try:
            fields = tuple(int(part, 10) for part in item.split(":"))
        except ValueError as error:
            _fail(f"non-decimal HOSTSIG item {item!r}")
            raise AssertionError from error
        if len(fields) != arity or any(field < 0 for field in fields):
            _fail(f"bad HOSTSIG item {item!r}")
        result.append(fields)
    return tuple(sorted(result))


def _marker_tuple(output: str, prefix: str, cls: type[object]) -> tuple[object, ...]:
    fields = tuple(cls.__dataclass_fields__)
    rows = _rows(output, prefix)
    if len(rows) != 2 or any(set(row) != set(fields) for row in rows):
        _fail(f"{prefix.strip()} must have two exact per-core rows")
    result = tuple(
        sorted(
            (cls(*(_number(row, name) for name in fields)) for row in rows),
            key=lambda item: item.core,
        )
    )
    if tuple(item.core for item in result) != _ACTIVE_CORES:
        _fail(f"{prefix.strip()} core closure changed")
    return result


def _expectation(case: object, linked: object) -> RootedArRuntimeExpectation:
    case.validate()
    runtime_by_core = {
        (item.logical_core.die_id, item.logical_core.local_core_id):
        item.runtime_core_id
        for item in linked.manifest.core_bindings
    }
    if tuple(sorted(runtime_by_core.values())) != _ACTIVE_CORES:
        _fail("rooted case must bind exact runtime cores 0 and 16")
    memories = []
    forwards = []
    backwards = []
    sgds = []
    for context in case.n6_intent.lowering_contexts:
        actions = context.global_dag.actions
        dma_in = tuple(
            action for action in actions
            if _value(action.task_kind) == "dma_in"
        )
        dma_out = tuple(
            action for action in actions
            if _value(action.task_kind) == "dma_out"
        )
        computes = tuple(action for action in actions if action.compute is not None)
        forward = tuple(
            action for action in computes
            if _value(action.op_kind) == "ce_forward"
        )
        backward = tuple(
            action for action in computes
            if _value(action.op_kind) == "ce_backward"
        )
        sgd = tuple(
            action for action in computes
            if _value(action.op_kind) == "optimizer_update"
        )
        if (
            len(dma_in) != 16 or len(dma_out) != 1
            or len(forward) != 1 or len(backward) != 1 or len(sgd) != 1
        ):
            _fail("each replica requires 16 LOAD, one STORE, CE/BWD/SGD")
        core_ref = forward[0].logical_core
        core = runtime_by_core[(core_ref.die_id, core_ref.local_core_id)]
        fw = forward[0].compute.workload
        bw = backward[0].compute.workload
        update = sgd[0].compute.workload
        rows = fw.rank_label_shape[0]
        backward_rows = bw.rank_label_shape[0]
        vocab = bw.rank_logits_shape[1]
        upstream = bw.rank_loss_gradient_shape[0]
        elements = update.element_count
        memories.append(RootedArMemory(
            core, 17, 17,
            sum(action.bytes for action in dma_in),
            sum(action.bytes for action in dma_out),
            sum(action.bytes for action in dma_out),
            sum(action.bytes for action in dma_in),
        ))
        forwards.append(RootedArCeForward(core, 1, rows, 4 * rows, 4 * rows))
        backwards.append(RootedArCeBackward(
            core, 1, backward_rows, upstream,
            2 * backward_rows * vocab, 4 * backward_rows,
            4 * upstream, 2 * backward_rows * vocab,
        ))
        sgds.append(RootedArSgd(
            core, 1, elements,
            struct.unpack("<Q", struct.pack("<d", update.learning_rate))[0],
            6 * elements, 2 * elements,
        ))
    unit_kinds = tuple(unit.kind.value for unit in case.n6_intent.units)
    if unit_kinds != (
        "local_copy", "upload_send", "upload_recv", "upload_wait",
        "root_reduce", "download_send", "download_recv", "download_wait",
    ):
        _fail("rooted executable unit quotient changed")
    for replica_index, stream in enumerate(linked.manifest.core_streams):
        dag = case.global_action.local_dags[replica_index]
        wgrad = next(
            action.id for action in dag.actions
            if ".lm_head_wgrad" in getattr(action.source, "task_id", "")
        )
        sgd_id = next(
            action.id for action in dag.actions
            if ".sgd_update" in getattr(action.source, "task_id", "")
        )
        overlay = {
            unit.id for unit in case.n6_intent.units
            if unit.logical_core == stream.logical_core
        }
        action_ids = tuple(ref.source_global_action_id for ref in stream.records)
        if not (
            max(i for i, action_id in enumerate(action_ids) if action_id == wgrad)
            < min(i for i, action_id in enumerate(action_ids) if action_id in overlay)
            <= max(i for i, action_id in enumerate(action_ids) if action_id in overlay)
            < min(i for i, action_id in enumerate(action_ids) if action_id == sgd_id)
        ):
            _fail("both SGD streams must be gated by the rooted overlay")
    return RootedArRuntimeExpectation(
        tuple(sorted(memories, key=lambda item: item.core)),
        tuple(sorted(forwards, key=lambda item: item.core)),
        tuple(sorted(backwards, key=lambda item: item.core)),
        tuple(sorted(sgds, key=lambda item: item.core)),
        (2, 4, 256),
    )


def build_production_pre_runtime(
    artifact_sha256: str | None = None,
) -> RootedArPreRuntime:
    from llm.test.frontend.integration.lite_train_rooted_ar_cases import (
        build_s2_lite_dp2_rooted_ar_case,
    )
    from llm.frontend.wafer_frontend.passes.lite_train_rooted_ar_lower_program import (
        lower_s2_lite_rooted_ar,
    )
    from llm.frontend.wafer_frontend.passes.lite_train_rooted_ar_link_program import (
        link_s2_lite_rooted_ar,
    )

    case = build_s2_lite_dp2_rooted_ar_case()
    lowered = lower_s2_lite_rooted_ar(case.n6_intent)
    linked = link_s2_lite_rooted_ar(lowered)
    if (
        len(lowered.local_fragments) != 92
        or len(lowered.overlay_fragments) != 6
        or len(linked.manifest.fragments) != 98
        or sum(
            len(stream.records)
            for fragment in linked.manifest.fragments
            for stream in fragment.core_streams
        ) != 351
        or sum(
            len(stream.address_relocations)
            for fragment in linked.manifest.fragments
            for stream in fragment.core_streams
        ) != 662
    ):
        _fail("rooted lower/link exact quotient changed")
    program_io = None
    if artifact_sha256 is not None:
        from llm.frontend.wafer_frontend.passes.program_io import (
            build_deterministic_timing_state_overrides,
            build_timing_program_io,
        )
        seeds, expected = build_deterministic_timing_state_overrides(linked)
        program_io = build_timing_program_io(
            linked,
            artifact_sha256,
            state_seed_overrides=seeds,
            state_expected_overrides=expected,
        )
        program_io.validate_against(linked.manifest)
        if (
            len(seeds), len(expected), len(program_io.blobs),
            len(program_io.initializations), len(program_io.output_probes),
        ) != (15, 0, 24, 126, 2):
            _fail("actual-SHA ProgramIo exact quotient changed")
    return RootedArPreRuntime(
        case, lowered, linked, program_io, _expectation(case, linked)
    )


def _validate_program_io_markers(
    output: str, artifact_sha256: str, contract: object,
) -> None:
    status = _rows(output, _STATUS)
    if tuple(row.get("phase") for row in status) != (
        "resolved", "applied", "verify",
    ):
        _fail("ProgramIo phase closure changed")
    for row in status:
        if (
            row.get("mode") != "timing"
            or _number(row, "initializations") != 126
            or _number(row, "probes") != 2
            or row.get("pass") != "1"
        ):
            _fail("ProgramIo status changed")
    if any(row.get("checksum") != artifact_sha256 for row in status[:2]):
        _fail("ProgramIo did not use the actual artifact SHA")
    probes = _rows(output, _PROBE)
    entries = {item.id: item for item in contract.output_probes}
    blobs = {item.id: item for item in contract.blobs}
    if len(probes) != 2 or len(entries) != 2:
        _fail("ProgramIo probe multiplicity changed")
    seen = set()
    for row in probes:
        entry = entries.get(row.get("id", ""))
        if entry is None:
            _fail("runtime returned an unknown ProgramIo probe")
        blob = blobs[entry.blob_ref]
        checksum = blob.sha256
        if (
            row.get("expected_checksum") != checksum
            or row.get("checksum") != checksum
            or row.get("valid") != "1"
            or row.get("exact") != "1"
            or row.get("pass") != "1"
            or entry.length_bytes != blob.length_bytes
            or _number(row, "bytes") != entry.length_bytes
        ):
            _fail("ProgramIo loss probe changed")
        seen.add(entry.id)
    if seen != set(entries):
        _fail("ProgramIo probes do not exactly cover the contract")


def observe_rooted_ar_runtime(
    output: str,
    artifact_sha256: str,
    contract: object,
    expectation: RootedArRuntimeExpectation,
) -> RootedArRuntimeObservation:
    if (
        "[PROTO_WAIT]" in output
        or output.count(_DONE) != len(_ACTIVE_CORES)
    ):
        _fail("PROTO_WAIT/DONE completion boundary changed")
    _validate_program_io_markers(output, artifact_sha256, contract)
    memory_rows = _rows(output, _MEMORY)
    memory = tuple(sorted((
        RootedArMemory(
            _number(row, "core"), _number(row, "lsu_issued"),
            _number(row, "lsu_completed"), _number(row, "lsu_hbm_read_bytes"),
            _number(row, "lsu_hbm_write_bytes"), _number(row, "lsu_sram_read_bytes"),
            _number(row, "lsu_sram_write_bytes"), _number(row, "lsu_residual"),
            _number(row, "dte_residual"),
        )
        for row in memory_rows
    ), key=lambda item: item.core))
    forward = _marker_tuple(output, _CE_FORWARD, RootedArCeForward)
    backward = _marker_tuple(output, _CE_BACKWARD, RootedArCeBackward)
    sgd = _marker_tuple(output, _SGD, RootedArSgd)
    if (
        memory != expectation.memory
        or forward != expectation.ce_forward
        or backward != expectation.ce_backward
        or sgd != expectation.sgd
    ):
        _fail("per-replica HBM/CE/BWD/SGD work changed")
    simulation = _rows(output, _SIM)
    if len(simulation) != 1 or set(simulation[0]) != {"makespan_cycles"}:
        _fail("SIM_RESULT must appear exactly once")
    makespan = _number(simulation[0], "makespan_cycles")
    if makespan == 0:
        _fail("makespan must be positive")
    host = _rows(output, _HOST)
    signatures = _rows(output, _HOSTSIG)
    if len(host) != 1 or len(signatures) != 1:
        _fail("HOSTLANE/HOSTSIG must each appear exactly once")
    ack_total = _number(host[0], "ack_total")
    done_total = _number(host[0], "done_total")
    if (
        ack_total != expectation.ack_total
        or done_total != expectation.done_total
        or _number(host[0], "mismatch") != 0
    ):
        _fail("ACK/DONE totals changed")
    if _parse_signature(signatures[0].get("done", ""), 2) != (
        (0, 1), (16, 1),
    ):
        _fail("DONE per-core closure changed")
    ack = _parse_signature(signatures[0].get("ack", ""), 3)
    by_core: Counter[int] = Counter()
    for core, _lane, count in ack:
        by_core[core] += count
    if tuple(sorted(by_core.items())) != ((0, 2), (16, 2)):
        _fail("ACK per-core closure changed")
    typed = _rows(output, _D2D_TYPE)
    if len(typed) != 1 or _rows(output, _D2D_BEHA):
        _fail("D2D_TYPE exact marker missing or behavioral link present")
    values = {
        key: _number(typed[0], key)
        for key in (
            "request_in", "request_out", "ack_in", "ack_out",
            "data_in", "data_out",
        )
    }
    requests, acknowledgements, data = expectation.d2d_packets
    if values != {
        "request_in": requests, "request_out": requests,
        "ack_in": acknowledgements, "ack_out": acknowledgements,
        "data_in": data, "data_out": data,
    }:
        _fail("rooted upload/download D2D packet quotient changed")
    pattern = re.compile(
        r"\[D2D_LINK\] idx=\d+ die(\d+)->die(\d+) dir=[A-Z?]+ "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)"
    )
    links = tuple(sorted(tuple(int(match.group(i)) for i in range(1, 9))
                         for match in pattern.finditer(output)))
    if links != (
        (0, 1, 1, 1, 2, 2, 128, 128),
        (1, 0, 1, 1, 2, 2, 128, 128),
    ):
        _fail("rooted directional D2D link quotient changed")
    endpoints = _rows(output, _P5)
    timing = _rows(output, _P5_TIMING)
    if (
        tuple(sorted(_number(row, "core") for row in endpoints)) != _ACTIVE_CORES
        or any(_number(row, "residual") for row in endpoints)
        or len(timing) != 1 or _number(timing[0], "residual") != 0
    ):
        _fail("P2P endpoint/timing drain changed")
    collective = _rows(output, _COLL)
    collective_fields = (
        "tree_entries", "reduce_nodes", "barriers", "gather",
        "reduce_rx", "endpoints", "dte_tokens", "event",
    )
    if (
        len(collective) != 1
        or any(_number(collective[0], key) for key in collective_fields)
    ):
        _fail("collective drain changed")
    drains = _rows(output, _DRAIN)
    residuals = {
        key: _number(row, key)
        for row in drains
        for key in ("router_residual", "d2d_link_residual")
        if key in row
    }
    if residuals != {"router_residual": 0, "d2d_link_residual": 0}:
        _fail("router/link drain changed")
    prefixes = (
        _STATUS, _PROBE, _MEMORY, _CE_FORWARD, _CE_BACKWARD, _SGD,
        _SIM, _HOST, _HOSTSIG, _P5, _P5_TIMING, _COLL, _DRAIN,
        _D2D_TYPE, _D2D_LINK,
    )
    marker_lines = []
    for line in output.splitlines():
        positions = tuple(line.find(prefix) for prefix in prefixes)
        positions = tuple(position for position in positions if position >= 0)
        if positions:
            marker_lines.append(line[min(positions):].split(" | ", 1)[0].rstrip(". "))
        elif _DONE in line:
            marker_lines.append(_DONE)
    return RootedArRuntimeObservation(
        makespan,
        hashlib.sha256("\n".join(marker_lines).encode("utf-8")).hexdigest(),
        memory, forward, backward, sgd, expectation.d2d_packets,
        ack_total, done_total, True,
    )


def validate_runtime_repeat(
    first: RootedArRuntimeObservation,
    second: RootedArRuntimeObservation,
) -> RootedArRepeatEvidence:
    if first != second:
        _fail("runtime observation repeat changed")
    return RootedArRepeatEvidence(first.marker_digest, second.marker_digest)


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


def run_official_rooted_ar(
    args: argparse.Namespace,
    *,
    command_runner: CommandRunner = _run,
) -> tuple[RootedArRuntimeObservation, RootedArRepeatEvidence]:
    from llm.frontend.wafer_frontend.schema.serde import (
        canonical_digest, canonical_json,
    )

    pre = build_production_pre_runtime()
    with tempfile.TemporaryDirectory(
        prefix="s2-lite-rooted-ar-", dir=args.runtime_root,
    ) as raw:
        work = Path(raw)
        manifest_path = work / "linked.json"
        hardware_path = work / "hardware.json"
        mapping_path = work / "mapping.spec"
        manifest_path.write_text(canonical_json(pre.linked.manifest), encoding="utf-8")
        hardware_path.write_text(pre.case.source.runtime_inputs.hardware_json, encoding="utf-8")
        mapping_path.write_text(pre.case.source.runtime_inputs.mapping_text, encoding="utf-8")
        artifacts = (work / "program.0.npup", work / "program.1.npup")
        reports = (work / "finalizer.0.json", work / "finalizer.1.json")
        finalizer_logs = []
        for artifact, report in zip(artifacts, reports, strict=True):
            finalizer_logs.append(command_runner((
                str(args.finalizer), "--input", str(manifest_path),
                "--output", str(artifact), "--report", str(report),
            ), args.runtime_root, 120))
        artifact_bytes = tuple(path.read_bytes() for path in artifacts)
        report_values = tuple(json.loads(path.read_text(encoding="utf-8")) for path in reports)
        if artifact_bytes[0] != artifact_bytes[1] or report_values[0] != report_values[1]:
            _fail("finalizer byte/report repeat changed")
        artifact_sha = hashlib.sha256(artifact_bytes[0]).hexdigest()
        expected_report = {
            "artifact_sha256": artifact_sha,
            "artifact_bytes": len(artifact_bytes[0]),
            "core_count": 2,
            "record_count": 351,
            "relocation_count": 662,
            "linked_manifest_id": pre.linked.manifest.id,
            "linked_manifest_digest": canonical_digest(pre.linked.manifest),
        }
        if any(report_values[0].get(key) != value for key, value in expected_report.items()):
            _fail("finalizer report exact quotient changed")
        actual = build_production_pre_runtime(artifact_sha)
        if actual.linked != pre.linked or actual.program_io is None:
            _fail("actual-SHA rebuild changed the rooted production chain")
        sidecar = work / "program_io.json"
        sidecar.write_text(canonical_json(actual.program_io), encoding="utf-8")
        resolver_output = command_runner((
            str(args.resolver), "--resolve", str(manifest_path),
            str(artifacts[0]), str(sidecar),
        ), args.runtime_root, 120)
        if "initializations=126 probes=2" not in resolver_output:
            _fail("resolver lost exact rooted ProgramIo counts")
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
            observations.append(observe_rooted_ar_runtime(
                output, artifact_sha, actual.program_io, actual.expectation,
            ))
        repeat = validate_runtime_repeat(observations[0], observations[1])
        return observations[0], repeat


def _file(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", required=True, type=_file)
    parser.add_argument("--finalizer", required=True, type=_file)
    parser.add_argument("--resolver", required=True, type=_file)
    parser.add_argument("--simulation", required=True, type=_file)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    observation, _repeat = run_official_rooted_ar(args)
    print(
        "[S2-LITE DP2 ROOTED-AR] PASS: timing_execution=1 "
        "functional_execution=0 repeat=2 "
        f"makespan_cycles={observation.makespan_cycles} "
        f"marker_digest={observation.marker_digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RootedArCeBackward",
    "RootedArCeForward",
    "RootedArMemory",
    "RootedArPreRuntime",
    "RootedArRepeatEvidence",
    "RootedArRuntimeExpectation",
    "RootedArRuntimeObservation",
    "RootedArSgd",
    "build_production_pre_runtime",
    "observe_rooted_ar_runtime",
    "run_official_rooted_ar",
    "validate_runtime_repeat",
]
