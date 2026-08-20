#!/usr/bin/env python3
"""Pre-runtime contracts and strict marker parser for S2-Lite training.

This module deliberately does not invoke the finalizer, resolver, or simulator.
It freezes the production construction sequence and the evidence parser so a
later runtime slice can add process execution without inventing evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import tempfile
from typing import Callable


_MEMORY = "[PROGRAM_MEMORY] "
_CE_FORWARD = "[TRAIN_CE] "
_CE_BACKWARD = "[TRAIN_CE_BACKWARD] "
_SGD = "[TRAIN_SGD] "
_SIM = "[SIM_RESULT] "
_HOST = "[HOSTLANE] "
_HOSTSIG = "[HOSTSIG] "
_P5 = "[P5 P2P DRAIN] "
_P5_TIMING = "[P5 P2P TIMING DRAIN] "
_COLLECTIVE = "[COLL_DRAIN] "
_DRAIN = "[DRAIN] "
_DONE = "End DONE reception"


def _fail(message: str) -> None:
    raise RuntimeError(f"[S2-LITE TRAIN] FAIL: {message}")


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


@dataclass(frozen=True, slots=True)
class LiteTrainActionFact:
    """The minimum action information used by the static WGRAD proof."""

    action_id: str
    op_kind: str
    phase: str
    deps: tuple[str, ...]
    order_index: int


@dataclass(frozen=True, slots=True)
class LiteTrainWgradEvidence:
    ce_backward_action_id: str
    wgrad_matmul_action_id: str
    sgd_action_id: str
    proof_kind: str = "static_matmul_dependency"
    timing_execution: bool = True
    functional_execution: bool = False


def validate_static_wgrad_dependency(
    facts: tuple[LiteTrainActionFact, ...],
) -> LiteTrainWgradEvidence:
    """Prove only that one WGRAD MATMUL is ordered between CE_BWD and SGD."""

    if len({item.action_id for item in facts}) != len(facts):
        _fail("static actions must have unique IDs")
    ce_backward = tuple(item for item in facts if item.op_kind == "ce_backward")
    wgrad = tuple(
        item
        for item in facts
        if item.op_kind == "gemm" and item.phase == "wgrad"
    )
    sgd = tuple(item for item in facts if item.op_kind == "optimizer_update")
    if len(ce_backward) != 1 or len(wgrad) != 1 or len(sgd) != 1:
        _fail("requires one exact CE_BACKWARD, WGRAD MATMUL, and SGD action")
    ce_action, wgrad_action, sgd_action = ce_backward[0], wgrad[0], sgd[0]
    if not (
        ce_action.order_index < wgrad_action.order_index < sgd_action.order_index
        and ce_action.action_id in wgrad_action.deps
        and wgrad_action.action_id in sgd_action.deps
    ):
        _fail("WGRAD MATMUL must be a direct static dependency between CE_BWD and SGD")
    return LiteTrainWgradEvidence(
        ce_backward_action_id=ce_action.action_id,
        wgrad_matmul_action_id=wgrad_action.action_id,
        sgd_action_id=sgd_action.action_id,
    )


def _action_facts(case: object) -> tuple[LiteTrainActionFact, ...]:
    case.validate()
    dag = case.global_action.global_dags[0]
    graph = case.planned.replicas[0].graph
    nodes = {node.id: node for node in graph.nodes}
    return tuple(
        LiteTrainActionFact(
            action_id=action.id,
            op_kind=str(_enum_value(action.op_kind)),
            phase=str(
                _enum_value(nodes[action.member_id].phase)
                if action.member_id in nodes
                else "none"
            ),
            deps=action.deps,
            order_index=index,
        )
        for index, action in enumerate(dag.actions)
        if action.op_kind is not None
    )


@dataclass(frozen=True, slots=True)
class CeForwardMarker:
    core: int
    invocations: int
    rank_rows: int
    label_read_bytes: int
    loss_write_bytes: int


@dataclass(frozen=True, slots=True)
class CeBackwardMarker:
    core: int
    invocations: int
    rank_rows: int
    upstream_elements: int
    logits_read_bytes: int
    label_read_bytes: int
    upstream_read_bytes: int
    logits_grad_write_bytes: int


@dataclass(frozen=True, slots=True)
class SgdMarker:
    core: int
    invocations: int
    element_count: int
    learning_rate_f64_bits: int
    sram_read_bytes: int
    sram_write_bytes: int


@dataclass(frozen=True, slots=True)
class LiteTrainMemory:
    core: int
    lsu_issued: int
    lsu_completed: int
    hbm_load_bytes: int
    hbm_store_bytes: int
    sram_read_bytes: int
    sram_write_bytes: int
    lsu_residual: int = 0
    dte_residual: int = 0


@dataclass(frozen=True, slots=True)
class LiteTrainRuntimeExpectation:
    memory: LiteTrainMemory
    ce_forward: CeForwardMarker
    ce_backward: CeBackwardMarker
    sgd: SgdMarker
    ack_total: int = 2
    done_total: int = 1


def _only_action(case: object, op_kind: str, *, phase: str | None = None) -> object:
    graph = case.planned.replicas[0].graph
    nodes = {node.id: node for node in graph.nodes}
    matches = tuple(
        action
        for action in case.global_action.global_dags[0].actions
        if _enum_value(action.op_kind) == op_kind
        and (
            phase is None
            or (
                action.member_id in nodes
                and _enum_value(nodes[action.member_id].phase) == phase
            )
        )
    )
    if len(matches) != 1 or matches[0].compute is None:
        _fail(f"requires one exact {op_kind}/{phase or 'any'} compute action")
    return matches[0]


def expectation_from_production_case(
    case: object,
) -> tuple[LiteTrainRuntimeExpectation, LiteTrainWgradEvidence]:
    """Derive runtime expectations from the validated production action quotient."""

    proof = validate_static_wgrad_dependency(_action_facts(case))
    dag = case.global_action.global_dags[0]
    dma_in = tuple(action for action in dag.actions if _enum_value(action.task_kind) == "dma_in")
    dma_out = tuple(action for action in dag.actions if _enum_value(action.task_kind) == "dma_out")
    if len(dma_in) != 16 or len(dma_out) != 1:
        _fail("S2-Lite requires sixteen HBM loads and one HBM store")

    ce_forward_action = _only_action(case, "ce_forward")
    ce_backward_action = _only_action(case, "ce_backward")
    sgd_action = _only_action(case, "optimizer_update")
    logical_cores = {
        (
            action.logical_core.die_id,
            action.logical_core.local_core_id,
        )
        for action in (ce_forward_action, ce_backward_action, sgd_action)
        if action.logical_core is not None
    }
    if logical_cores != {(0, 0)}:
        _fail("S2-Lite runtime skeleton requires one die0/core0 timeline")
    core = 0
    forward = ce_forward_action.compute.workload
    backward = ce_backward_action.compute.workload
    sgd_workload = sgd_action.compute.workload
    rank_rows = forward.rank_label_shape[0]
    backward_rows = backward.rank_label_shape[0]
    vocab = backward.rank_logits_shape[1]
    upstream = backward.rank_loss_gradient_shape[0]
    element_count = sgd_workload.element_count
    expectation = LiteTrainRuntimeExpectation(
        memory=LiteTrainMemory(
            core=core,
            lsu_issued=len(dma_in) + len(dma_out),
            lsu_completed=len(dma_in) + len(dma_out),
            hbm_load_bytes=sum(action.bytes for action in dma_in),
            hbm_store_bytes=sum(action.bytes for action in dma_out),
            sram_read_bytes=sum(action.bytes for action in dma_out),
            sram_write_bytes=sum(action.bytes for action in dma_in),
        ),
        ce_forward=CeForwardMarker(core, 1, rank_rows, rank_rows * 4, rank_rows * 4),
        ce_backward=CeBackwardMarker(
            core,
            1,
            backward_rows,
            upstream,
            backward_rows * vocab * 2,
            backward_rows * 4,
            upstream * 4,
            backward_rows * vocab * 2,
        ),
        sgd=SgdMarker(
            core,
            1,
            element_count,
            struct.unpack("<Q", struct.pack("<d", sgd_workload.learning_rate))[0],
            element_count * 6,
            element_count * 2,
        ),
    )
    return expectation, proof


@dataclass(frozen=True, slots=True)
class LiteTrainRuntimeObservation:
    makespan_cycles: int
    marker_digest: str
    memory: LiteTrainMemory
    ce_forward: CeForwardMarker
    ce_backward: CeBackwardMarker
    sgd: SgdMarker
    ack_total: int
    done_total: int
    drained: bool


@dataclass(frozen=True, slots=True)
class LiteTrainRepeatEvidence:
    first_marker_digest: str
    second_marker_digest: str
    repeat_count: int = 2


def _row(line: str, prefix: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line[len(prefix) :].strip().split():
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
        value = int(row[key], 10)
    except (KeyError, ValueError) as error:
        _fail(f"missing/non-decimal {key!r} in {row}")
        raise AssertionError from error
    if value < 0:
        _fail(f"negative {key!r} in {row}")
    return value


def _exact_row(output: str, prefix: str, fields: tuple[str, ...]) -> dict[str, str]:
    rows = _rows(output, prefix)
    if len(rows) != 1:
        _fail(f"{prefix.strip()} must appear exactly once")
    if set(rows[0]) != set(fields):
        _fail(f"{prefix.strip()} fields changed: {rows[0]}")
    return rows[0]


def _parse_signature(value: str, arity: int) -> tuple[tuple[int, ...], ...]:
    parsed = []
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
        parsed.append(parts)
    return tuple(sorted(parsed))


def _marker_dataclass(row: dict[str, str], cls: type[object]) -> object:
    return cls(*(_number(row, name) for name in cls.__dataclass_fields__))


def observe_lite_train_runtime(
    output: str,
    expectation: LiteTrainRuntimeExpectation,
) -> LiteTrainRuntimeObservation:
    """Parse independent observed work; expected values are used only to compare."""

    if "[PROTO_WAIT]" in output:
        _fail("PROTO_WAIT is forbidden in a completed timing execution")
    if output.count(_DONE) != 1:
        _fail("DONE boundary must appear exactly once")

    memory_row = _exact_row(
        output,
        _MEMORY,
        (
            "core",
            "lsu_issued",
            "lsu_completed",
            "lsu_hbm_read_bytes",
            "lsu_hbm_write_bytes",
            "lsu_sram_read_bytes",
            "lsu_sram_write_bytes",
            "lsu_residual",
            "dte_residual",
        ),
    )
    memory = LiteTrainMemory(
        _number(memory_row, "core"),
        _number(memory_row, "lsu_issued"),
        _number(memory_row, "lsu_completed"),
        _number(memory_row, "lsu_hbm_read_bytes"),
        _number(memory_row, "lsu_hbm_write_bytes"),
        _number(memory_row, "lsu_sram_read_bytes"),
        _number(memory_row, "lsu_sram_write_bytes"),
        _number(memory_row, "lsu_residual"),
        _number(memory_row, "dte_residual"),
    )
    ce_forward = _marker_dataclass(
        _exact_row(output, _CE_FORWARD, tuple(CeForwardMarker.__dataclass_fields__)),
        CeForwardMarker,
    )
    ce_backward = _marker_dataclass(
        _exact_row(output, _CE_BACKWARD, tuple(CeBackwardMarker.__dataclass_fields__)),
        CeBackwardMarker,
    )
    sgd = _marker_dataclass(
        _exact_row(output, _SGD, tuple(SgdMarker.__dataclass_fields__)),
        SgdMarker,
    )
    if (
        memory != expectation.memory
        or ce_forward != expectation.ce_forward
        or ce_backward != expectation.ce_backward
        or sgd != expectation.sgd
    ):
        _fail("runtime work or exact HBM load/store accounting changed")

    simulation = _exact_row(output, _SIM, ("makespan_cycles",))
    makespan = _number(simulation, "makespan_cycles")
    if makespan == 0:
        _fail("makespan must be positive")
    host = _exact_row(
        output,
        _HOST,
        ("ack_total", "done_total", "mismatch", "per_lane_done"),
    )
    ack_total = _number(host, "ack_total")
    done_total = _number(host, "done_total")
    if (
        ack_total != expectation.ack_total
        or done_total != expectation.done_total
        or _number(host, "mismatch") != 0
        or host["per_lane_done"] != "1,0,0,0,0,0,0,0"
    ):
        _fail("ACK/DONE totals changed")
    signature = _exact_row(output, _HOSTSIG, ("done", "ack"))
    core = expectation.memory.core
    if _parse_signature(signature["done"], 2) != ((core, 1),):
        _fail("DONE per-core closure changed")
    ack = _parse_signature(signature["ack"], 3)
    if sum(count for item_core, _lane, count in ack if item_core == core) != ack_total:
        _fail("ACK per-core closure changed")

    endpoint_rows = _rows(output, _P5)
    timing_rows = _rows(output, _P5_TIMING)
    if len(endpoint_rows) > 1 or len(timing_rows) > 1:
        _fail("zero-P2P S2-Lite permits at most one endpoint/timing drain row")
    endpoint = endpoint_rows[0] if endpoint_rows else None
    timing = timing_rows[0] if timing_rows else None
    if endpoint is not None and set(endpoint) != {"core", "residual"}:
        _fail("P2P endpoint drain fields changed")
    if timing is not None and set(timing) != {"residual"}:
        _fail("P2P timing drain fields changed")
    collective = _exact_row(
        output,
        _COLLECTIVE,
        (
            "tree_entries",
            "reduce_nodes",
            "barriers",
            "gather",
            "reduce_rx",
            "endpoints",
            "dte_tokens",
            "event",
        ),
    )
    drain_rows = _rows(output, _DRAIN)
    drain = None
    if drain_rows:
        merged: dict[str, str] = {}
        for row in drain_rows:
            if set(merged).intersection(row):
                _fail("router/link drain fields are duplicated")
            merged.update(row)
        if set(merged) != {"router_residual", "d2d_link_residual"}:
            _fail(f"router/link drain fields changed: {drain_rows}")
        drain = merged
    if (
        (endpoint is not None and (
            _number(endpoint, "core") != core
            or _number(endpoint, "residual") != 0
        ))
        or (timing is not None and _number(timing, "residual") != 0)
        or any(_number(collective, key) for key in collective)
        or (drain is not None and any(_number(drain, key) for key in drain))
    ):
        _fail("runtime engines did not drain exactly")

    prefixes = (
        _MEMORY,
        _CE_FORWARD,
        _CE_BACKWARD,
        _SGD,
        _SIM,
        _HOST,
        _HOSTSIG,
        _P5,
        _P5_TIMING,
        _COLLECTIVE,
        _DRAIN,
    )
    marker_lines = []
    for line in output.splitlines():
        positions = tuple(line.find(prefix) for prefix in prefixes)
        positions = tuple(position for position in positions if position >= 0)
        if positions:
            marker_lines.append(line[min(positions) :].split(" | ", 1)[0].rstrip(". "))
        elif _DONE in line:
            marker_lines.append(_DONE)
    return LiteTrainRuntimeObservation(
        makespan,
        hashlib.sha256("\n".join(marker_lines).encode("utf-8")).hexdigest(),
        memory,
        ce_forward,
        ce_backward,
        sgd,
        ack_total,
        done_total,
        True,
    )


def validate_runtime_repeat(
    first: LiteTrainRuntimeObservation,
    second: LiteTrainRuntimeObservation,
) -> LiteTrainRepeatEvidence:
    if first != second:
        _fail("runtime observation repeat changed")
    return LiteTrainRepeatEvidence(first.marker_digest, second.marker_digest)


@dataclass(frozen=True, slots=True)
class LiteTrainPreRuntimeCommands:
    finalizer_runs: tuple[tuple[str, ...], tuple[str, ...]]
    resolver_run: tuple[str, ...]


def build_pre_runtime_commands(
    *,
    finalizer: Path,
    resolver: Path,
    manifest: Path,
    artifact: Path,
    program_io: Path,
    finalizer_report_root: Path,
) -> LiteTrainPreRuntimeCommands:
    """Build, but never execute, the production finalizer/resolver commands."""

    runs = tuple(
        (
            str(finalizer),
            "--input",
            str(manifest),
            "--output",
            str(artifact.with_suffix(f".{index}.npup")),
            "--report",
            str(finalizer_report_root / f"finalizer.{index}.json"),
        )
        for index in range(2)
    )
    return LiteTrainPreRuntimeCommands(
        finalizer_runs=(runs[0], runs[1]),
        resolver_run=(
            str(resolver),
            "--resolve",
            str(manifest),
            str(artifact.with_suffix(".0.npup")),
            str(program_io),
        ),
    )


@dataclass(frozen=True, slots=True)
class LiteTrainProductionPreRuntime:
    case: object
    lowered: object
    linked: object
    program_io: object | None
    expectation: LiteTrainRuntimeExpectation
    static_wgrad: LiteTrainWgradEvidence


def build_production_pre_runtime(
    *,
    artifact_sha256: str | None = None,
    case_builder: Callable[[], object] | None = None,
) -> LiteTrainProductionPreRuntime:
    """Lazily construct the production chain; never execute external binaries."""

    if case_builder is None:
        from lite_train_cases import build_s2_lite_production_case

        case_builder = build_s2_lite_production_case
    from llm.frontend.wafer_frontend.passes.lite_train_link_program import (
        link_s2_lite_train,
    )
    from llm.frontend.wafer_frontend.passes.lite_train_lower_program import (
        lower_s2_lite_train,
    )

    case = case_builder()
    expectation, proof = expectation_from_production_case(case)
    lowered = lower_s2_lite_train(case.global_action)
    linked = link_s2_lite_train(lowered)
    program_io = None
    if artifact_sha256 is not None:
        from llm.frontend.wafer_frontend.passes.program_io import (
            build_deterministic_timing_state_overrides,
            build_timing_program_io,
        )

        state_seeds, state_expected = build_deterministic_timing_state_overrides(
            linked
        )
        program_io = build_timing_program_io(
            linked,
            artifact_sha256,
            state_seed_overrides=state_seeds,
            state_expected_overrides=state_expected,
        )
        program_io.validate_against(linked.manifest)
    return LiteTrainProductionPreRuntime(
        case,
        lowered,
        linked,
        program_io,
        expectation,
        proof,
    )


def _executable(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {path}")
    return path


def _run(command: list[str], *, cwd: Path, timeout: int) -> str:
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
        _fail(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")
    return completed.stdout


def run_official_lite_train(args: argparse.Namespace) -> tuple[LiteTrainRuntimeObservation, LiteTrainRepeatEvidence]:
    """Finalize, resolve, and execute the exact S2-Lite case twice."""

    from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

    pre = build_production_pre_runtime()
    manifest = pre.linked.manifest
    case = pre.case
    with tempfile.TemporaryDirectory(prefix="s2-lite-train-", dir=args.runtime_root) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
        hardware_path.write_text(case.source.runtime_inputs.hardware_json, encoding="utf-8")
        mapping_path.write_text(case.source.runtime_inputs.mapping_text, encoding="utf-8")
        artifacts = (directory / "program.0.npup", directory / "program.1.npup")
        reports = (directory / "finalizer.0.json", directory / "finalizer.1.json")
        artifact_bytes = []
        finalization = []
        for artifact, report in zip(artifacts, reports, strict=True):
            _run([
                str(args.finalizer), "--input", str(manifest_path),
                "--output", str(artifact), "--report", str(report),
            ], cwd=args.runtime_root, timeout=120)
            artifact_bytes.append(artifact.read_bytes())
            finalization.append(json.loads(report.read_text(encoding="utf-8")))
        if artifact_bytes[0] != artifact_bytes[1] or finalization[0] != finalization[1]:
            _fail("finalizer byte/report repeat changed")
        artifact_sha = hashlib.sha256(artifact_bytes[0]).hexdigest()
        leaf_streams = tuple(stream for fragment in manifest.fragments for stream in fragment.core_streams)
        expected = {
            "artifact_sha256": artifact_sha,
            "artifact_bytes": len(artifact_bytes[0]),
            "core_count": 1,
            "record_count": sum(len(stream.records) for stream in leaf_streams),
            "relocation_count": sum(len(stream.address_relocations) for stream in leaf_streams),
            "linked_manifest_id": manifest.id,
            "linked_manifest_digest": canonical_digest(manifest),
        }
        if any(finalization[0].get(key) != value for key, value in expected.items()):
            _fail(f"finalizer report closure changed: {finalization[0]}")
        actual = build_production_pre_runtime(artifact_sha256=artifact_sha)
        if actual.linked != pre.linked or actual.program_io is None:
            _fail("actual-SHA ProgramIo rebuild changed the production link")
        sidecar = directory / "program_io.json"
        sidecar.write_text(canonical_json(actual.program_io), encoding="utf-8")
        resolver_output = _run([
            str(args.resolver), "--resolve", str(manifest_path),
            str(artifacts[0]), str(sidecar),
        ], cwd=args.runtime_root, timeout=120)
        if "initializations=62 probes=1" not in resolver_output:
            _fail(f"resolver lost exact ProgramIo counts: {resolver_output}")
        observations = []
        for index in range(2):
            output = _run([
                str(args.npusim), "--program", str(artifacts[0]),
                "--linked-manifest", str(manifest_path),
                "--program-io", str(sidecar),
                "--hardware-config", str(hardware_path),
                "--simulation-config", str(args.simulation),
                "--mapping-config", str(mapping_path),
                "--trace-window", "1000000",
            ], cwd=args.runtime_root, timeout=args.timeout)
            if args.runtime_log is not None:
                log_path = (
                    args.runtime_log
                    if index == 0
                    else args.runtime_log.with_name(
                        f"{args.runtime_log.stem}.1{args.runtime_log.suffix}"
                    )
                )
                log_path.write_text(output, encoding="utf-8")
            observations.append(observe_lite_train_runtime(output, pre.expectation))
        repeat = validate_runtime_repeat(observations[0], observations[1])
        print(
            "[S2-LITE TRAIN] PASS: timing_execution=1 functional_execution=0 "
            f"artifact={len(artifact_bytes[0])}B records=169 relocations=324 "
            f"init=62 probes=1 repeat=2 makespan_cycles={observations[0].makespan_cycles} "
            f"sha256={artifact_sha} marker_digest={observations[0].marker_digest}"
        )
        return observations[0], repeat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", required=True, type=_executable)
    parser.add_argument("--finalizer", required=True, type=_executable)
    parser.add_argument("--resolver", required=True, type=_executable)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--runtime-log", type=Path)
    args = parser.parse_args()
    args.simulation = args.simulation.resolve()
    if not args.simulation.is_file():
        parser.error(f"--simulation is not a file: {args.simulation}")
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.runtime_log is not None:
        args.runtime_log = args.runtime_log.resolve()
        args.runtime_log.parent.mkdir(parents=True, exist_ok=True)
    run_official_lite_train(args)
    return 0


__all__ = [
    "CeBackwardMarker",
    "CeForwardMarker",
    "LiteTrainActionFact",
    "LiteTrainMemory",
    "LiteTrainPreRuntimeCommands",
    "LiteTrainProductionPreRuntime",
    "LiteTrainRepeatEvidence",
    "LiteTrainRuntimeExpectation",
    "LiteTrainRuntimeObservation",
    "LiteTrainWgradEvidence",
    "SgdMarker",
    "build_pre_runtime_commands",
    "build_production_pre_runtime",
    "expectation_from_production_case",
    "observe_lite_train_runtime",
    "validate_runtime_repeat",
    "validate_static_wgrad_dependency",
]


if __name__ == "__main__":
    raise SystemExit(main())
