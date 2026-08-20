#!/usr/bin/env python3
"""Injectable pre-runtime/runtime runner for fixed DP4×TP1 tree AllReduce."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from typing import Callable

from llm.frontend.wafer_frontend.schema.lite_train_dp4 import (
    S2_LITE_DP4_TREE_AR_CASE_ID,
)

_CASE_ID = S2_LITE_DP4_TREE_AR_CASE_ID
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
_D2D_LINK = "[D2D_LINK] "
_DONE = "End DONE reception"


def _fail(message: str) -> None:
    raise RuntimeError(f"[S2-LITE DP4 TREE-AR] FAIL: {message}")


@dataclass(frozen=True, slots=True)
class Dp4FlowExpectation:
    flow_id: str
    source_die: int
    destination_die: int
    bytes: int
    die_path: tuple[int, ...]

    @property
    def data_packets(self) -> int:
        return self.bytes // 16


@dataclass(frozen=True, slots=True)
class Dp4LinkTraffic:
    source_die: int
    destination_die: int
    request_packets: int
    ack_packets: int
    data_packets: int


@dataclass(frozen=True, slots=True)
class Dp4ProbeExpectation:
    probe_id: str
    die_id: int
    address: int
    bytes: int
    checksum: str


@dataclass(frozen=True, slots=True)
class Dp4Memory:
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
class Dp4CeForward:
    core: int
    invocations: int
    rank_rows: int
    label_read_bytes: int
    loss_write_bytes: int


@dataclass(frozen=True, slots=True)
class Dp4CeBackward:
    core: int
    invocations: int
    rank_rows: int
    upstream_elements: int
    logits_read_bytes: int
    label_read_bytes: int
    upstream_read_bytes: int
    logits_grad_write_bytes: int


@dataclass(frozen=True, slots=True)
class Dp4Sgd:
    core: int
    invocations: int
    element_count: int
    learning_rate_f64_bits: int
    sram_read_bytes: int
    sram_write_bytes: int


@dataclass(frozen=True, slots=True)
class Dp4RuntimeExpectation:
    artifact_sha256: str
    active_cores: tuple[int, ...]
    flows: tuple[Dp4FlowExpectation, ...]
    links: tuple[Dp4LinkTraffic, ...]
    memory: tuple[Dp4Memory, ...]
    ce_forward: tuple[Dp4CeForward, ...]
    ce_backward: tuple[Dp4CeBackward, ...]
    sgd: tuple[Dp4Sgd, ...]
    initialization_count: int
    probes: tuple[Dp4ProbeExpectation, ...]
    ack_total: int = 8
    done_total: int = 4

    def validate(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.artifact_sha256):
            _fail("expectation requires an actual canonical artifact SHA")
        if len(self.active_cores) != 4 or self.active_cores != tuple(sorted(set(self.active_cores))):
            _fail("typed manifest must identify four unique active cores")
        if (
            len(self.flows) != 6
            or len({item.flow_id for item in self.flows}) != 6
            or any(item.bytes != 2048 or item.data_packets != 128 for item in self.flows)
            or sum(item.bytes for item in self.flows) != 12288
            or sum(item.data_packets for item in self.flows) != 768
        ):
            _fail("typed tree must contain six exact 2048B/128-packet flows")
        expected_links: dict[tuple[int, int], list[int]] = {}
        for flow in self.flows:
            if (
                len(flow.die_path) < 2
                or flow.die_path[0] != flow.source_die
                or flow.die_path[-1] != flow.destination_die
            ):
                _fail("flow die path does not close its typed endpoints")
            for source, destination in zip(flow.die_path, flow.die_path[1:]):
                counts = expected_links.setdefault((source, destination), [0, 0, 0])
                counts[0] += 1
                counts[1] += 2
                counts[2] += flow.data_packets
        actual_links = {
            (item.source_die, item.destination_die):
            [item.request_packets, item.ack_packets, item.data_packets]
            for item in self.links
        }
        if actual_links != expected_links:
            _fail("directed-link traffic is not independently derived from PairRoute paths")
        for values in (self.memory, self.ce_forward, self.ce_backward, self.sgd):
            if len(values) != 4 or tuple(item.core for item in values) != self.active_cores:
                _fail("per-core runtime expectations must exactly cover four active cores")
        if (
            sum(item.hbm_write_bytes for item in self.memory) != 4096
            or any(item.invocations != 1 for item in self.ce_forward)
            or any(item.invocations != 1 for item in self.ce_backward)
            or any(item.invocations != 1 or item.element_count != 512 for item in self.sgd)
            or len(self.probes) != 4
            or len({item.probe_id for item in self.probes}) != 4
            or sum(item.bytes for item in self.probes) != 4096
            or self.initialization_count <= 0
            or (self.ack_total, self.done_total) != (8, 4)
        ):
            _fail("DP4 work/ProgramIo/control quotient changed")


@dataclass(frozen=True, slots=True)
class Dp4RuntimeObservation:
    makespan_cycles: int
    marker_digest: str
    memory: tuple[Dp4Memory, ...]
    ce_forward: tuple[Dp4CeForward, ...]
    ce_backward: tuple[Dp4CeBackward, ...]
    sgd: tuple[Dp4Sgd, ...]
    request_packets: int
    ack_packets: int
    data_packets: int
    done_total: int
    drained: bool
    timing_execution: bool = True
    functional_execution: bool = False


@dataclass(frozen=True, slots=True)
class Dp4RepeatEvidence:
    first_marker_digest: str
    second_marker_digest: str
    repeat_count: int = 2


@dataclass(frozen=True, slots=True)
class Dp4ProductionChain:
    case: object
    linked: object
    program_io: object | None
    expectation: Dp4RuntimeExpectation | None


ProductionBuilder = Callable[[str | None], Dp4ProductionChain]
CommandRunner = Callable[[tuple[str, ...], Path, int], str]


def _value(value: object) -> object:
    return getattr(value, "value", value)


def _xy_die_path(fabric: object, source_die: int, destination_die: int) -> tuple[int, ...]:
    by_id = {die.id: die for die in fabric.dies}
    if source_die not in by_id or destination_die not in by_id:
        _fail("tree flow references a die outside the typed fabric")
    x, y = by_id[source_die].coord
    destination_x, destination_y = by_id[destination_die].coord
    result = [source_die]
    while x != destination_x:
        x += 1 if destination_x > x else -1
        result.append(y * fabric.die_grid[0] + x)
    while y != destination_y:
        y += 1 if destination_y > y else -1
        result.append(y * fabric.die_grid[0] + x)
    return tuple(result)


def _production_expectation(
    case: object,
    program_io: object,
    artifact_sha256: str,
) -> Dp4RuntimeExpectation:
    runtime_by_core = {
        (item.logical_core.die_id, item.logical_core.local_core_id): item.runtime_core_id
        for item in case.linked.manifest.core_bindings
    }
    memories = []
    forwards = []
    backwards = []
    sgds = []
    for context in case.n6_intent.lowering_contexts:
        actions = context.global_dag.actions
        dma_in = tuple(action for action in actions if _value(action.task_kind) == "dma_in")
        dma_out = tuple(action for action in actions if _value(action.task_kind) == "dma_out")
        computes = tuple(action for action in actions if action.compute is not None)
        forward = tuple(action for action in computes if _value(action.op_kind) == "ce_forward")
        backward = tuple(action for action in computes if _value(action.op_kind) == "ce_backward")
        sgd = tuple(action for action in computes if _value(action.op_kind) == "optimizer_update")
        if (
            len(dma_in) != 16 or len(dma_out) != 1
            or len(forward) != 1 or len(backward) != 1 or len(sgd) != 1
        ):
            _fail("each DP4 replica requires 16 LOAD, one STORE, CE/BWD/SGD")
        core_ref = forward[0].logical_core
        core = runtime_by_core[(core_ref.die_id, core_ref.local_core_id)]
        fw = forward[0].compute.workload
        bw = backward[0].compute.workload
        update = sgd[0].compute.workload
        rank_rows = fw.rank_label_shape[0]
        backward_rows = bw.rank_label_shape[0]
        vocabulary = bw.rank_logits_shape[1]
        upstream = bw.rank_loss_gradient_shape[0]
        elements = update.element_count
        memories.append(Dp4Memory(
            core, 17, 17,
            sum(action.bytes for action in dma_in),
            sum(action.bytes for action in dma_out),
            sum(action.bytes for action in dma_out),
            sum(action.bytes for action in dma_in),
        ))
        forwards.append(Dp4CeForward(core, 1, rank_rows, 4 * rank_rows, 4 * rank_rows))
        backwards.append(Dp4CeBackward(
            core, 1, backward_rows, upstream,
            2 * backward_rows * vocabulary, 4 * backward_rows,
            4 * upstream, 2 * backward_rows * vocabulary,
        ))
        sgds.append(Dp4Sgd(
            core, 1, elements,
            struct.unpack("<Q", struct.pack("<d", update.learning_rate))[0],
            6 * elements, 2 * elements,
        ))
    active_cores = tuple(sorted(runtime_by_core.values()))
    flows = tuple(
        Dp4FlowExpectation(
            flow.id,
            flow.source_die_id,
            flow.destination_die_id,
            flow.gradient_bytes,
            _xy_die_path(
                case.placement_context.fabric,
                flow.source_die_id,
                flow.destination_die_id,
            ),
        )
        for flow in case.global_action.tree_flows
    )
    link_counts: dict[tuple[int, int], list[int]] = {}
    for flow in flows:
        for source, destination in zip(flow.die_path, flow.die_path[1:]):
            counts = link_counts.setdefault((source, destination), [0, 0, 0])
            counts[0] += 1
            counts[1] += 2
            counts[2] += flow.data_packets
    links = tuple(
        Dp4LinkTraffic(source, destination, *counts)
        for (source, destination), counts in sorted(link_counts.items())
    )
    blobs = {blob.id: blob for blob in program_io.blobs}
    symbols = {
        definition.symbol.id: definition
        for definition in case.linked.manifest.program_symbol_definitions
    }
    buffer_abis = {
        abi.id: abi
        for linked in case.linked.manifest.fragments
        for abi in (
            linked.fragment.buffer_abi
            if hasattr(linked, "fragment")
            else linked.buffer_abi
        )
    }
    probes = []
    for probe in program_io.output_probes:
        definition = symbols.get(probe.target.program_symbol_ref)
        blob = blobs.get(probe.blob_ref)
        if definition is None or blob is None:
            _fail("updated-weight probe lacks exact HBM symbol/blob witnesses")
        if _value(probe.target.kind) == "sram":
            abi = buffer_abis.get(probe.target.buffer_abi_id)
            regions = tuple(
                item
                for item in case.linked.manifest.program_symbol_definitions
                if item.symbol.kind.name == "SRAM_REGION"
                and abi is not None
                and item.symbol.source_ref == abi.region_ref
                and abi.logical_core in item.logical_cores
            )
            if abi is None or len(regions) != 1:
                _fail("loss probe lacks one exact BufferABI/SRAM region witness")
            die_id = abi.logical_core.die_id
            address = regions[0].value + abi.region_offset_bytes + probe.offset_bytes
        elif _value(probe.target.kind) == "hbm":
            die_id = definition.logical_cores[0].die_id
            address = definition.value + probe.offset_bytes
        else:
            _fail("probe target kind is not SRAM/HBM")
        probes.append(Dp4ProbeExpectation(
            probe.id,
            die_id,
            address,
            probe.length_bytes,
            blob.sha256,
        ))
    result = Dp4RuntimeExpectation(
        artifact_sha256,
        active_cores,
        flows,
        links,
        tuple(sorted(memories, key=lambda item: item.core)),
        tuple(sorted(forwards, key=lambda item: item.core)),
        tuple(sorted(backwards, key=lambda item: item.core)),
        tuple(sorted(sgds, key=lambda item: item.core)),
        len(program_io.initializations),
        tuple(sorted(probes, key=lambda item: item.probe_id)),
    )
    result.validate()
    return result


def build_production_chain(
    artifact_sha256: str | None = None,
) -> Dp4ProductionChain:
    from llm.test.frontend.integration.lite_train_dp4_cases import (
        build_s2_lite_dp4_tree_ar_case,
    )
    from llm.frontend.wafer_frontend.passes.program_io import (
        build_deterministic_timing_state_overrides,
        build_timing_program_io,
    )

    case = build_s2_lite_dp4_tree_ar_case()
    program_io = None
    expectation = None
    if artifact_sha256 is not None:
        seeds, expected = build_deterministic_timing_state_overrides(case.linked)
        program_io = build_timing_program_io(
            case.linked,
            artifact_sha256,
            state_seed_overrides=seeds,
            state_expected_overrides=expected,
        )
        program_io.validate_against(case.linked.manifest)
        if (
            len(seeds), len(expected), len(program_io.blobs),
            len(program_io.initializations), len(program_io.output_probes),
        ) != (15, 0, 26, 252, 4):
            _fail("DP4 actual-SHA ProgramIo exact quotient changed")
        expectation = _production_expectation(case, program_io, artifact_sha256)
    return Dp4ProductionChain(case, case.linked, program_io, expectation)


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
        value = int(row[key], 10)
    except (KeyError, ValueError) as error:
        _fail(f"missing/non-decimal {key!r} in {row}")
        raise AssertionError from error
    if value < 0:
        _fail(f"negative {key!r} in {row}")
    return value


def _exact_row(output: str, prefix: str, fields: tuple[str, ...]) -> dict[str, str]:
    rows = _rows(output, prefix)
    if len(rows) != 1 or set(rows[0]) != set(fields):
        _fail(f"{prefix.strip()} exact row changed")
    return rows[0]


def _markers(output: str, prefix: str, cls, cores: tuple[int, ...]):
    fields = tuple(cls.__dataclass_fields__)
    rows = _rows(output, prefix)
    if len(rows) != 4 or any(set(row) != set(fields) for row in rows):
        _fail(f"{prefix.strip()} must contain four exact rows")
    values = tuple(sorted(
        (cls(*(_number(row, field) for field in fields)) for row in rows),
        key=lambda item: item.core,
    ))
    if tuple(item.core for item in values) != cores:
        _fail(f"{prefix.strip()} active-core coverage changed")
    return values


def _memory_markers(output: str, cores: tuple[int, ...]) -> tuple[Dp4Memory, ...]:
    fields = (
        "core", "lsu_issued", "lsu_completed", "lsu_hbm_read_bytes",
        "lsu_hbm_write_bytes", "lsu_sram_read_bytes", "lsu_sram_write_bytes",
        "lsu_residual", "dte_residual",
    )
    rows = _rows(output, _MEMORY)
    if len(rows) != 4 or any(set(row) != set(fields) for row in rows):
        _fail(f"{_MEMORY.strip()} must contain four exact rows")
    values = tuple(sorted(
        (
            Dp4Memory(
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
        ),
        key=lambda item: item.core,
    ))
    if tuple(item.core for item in values) != cores:
        _fail(f"{_MEMORY.strip()} active-core coverage changed")
    return values


def _signature(value: str, arity: int) -> tuple[tuple[int, ...], ...]:
    result = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        try:
            fields = tuple(int(part, 10) for part in item.split(":"))
        except ValueError as error:
            _fail("HOSTSIG contains non-decimal fields")
            raise AssertionError from error
        if len(fields) != arity or any(field < 0 for field in fields):
            _fail("HOSTSIG arity/value changed")
        result.append(fields)
    return tuple(sorted(result))


def observe_dp4_tree_ar_runtime(
    output: str,
    expectation: Dp4RuntimeExpectation,
) -> Dp4RuntimeObservation:
    expectation.validate()
    cores = expectation.active_cores
    if "[PROTO_WAIT]" in output or output.count(_DONE) != len(cores):
        _fail("PROTO_WAIT/DONE completion boundary changed")
    status = _rows(output, _STATUS)
    if tuple(row.get("phase") for row in status) != ("resolved", "applied", "verify"):
        _fail("ProgramIo phase closure changed")
    for row in status:
        if (
            set(row) != {"phase", "mode", "checksum", "initializations", "probes", "pass"}
            or row.get("mode") != "timing"
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("checksum", ""))
            or _number(row, "initializations") != expectation.initialization_count
            or _number(row, "probes") != 4
            or _number(row, "pass") != 1
        ):
            _fail("ProgramIo status changed")
    if any(row["checksum"] != expectation.artifact_sha256 for row in status[:2]):
        _fail("ProgramIo did not use actual artifact SHA")
    probes = _rows(output, _PROBE)
    expected_probes = {item.probe_id: item for item in expectation.probes}
    if len(probes) != 4:
        _fail("ProgramIo must return four updated-weight probes")
    seen = set()
    for row in probes:
        if set(row) != {"id", "die", "address", "bytes", "expected_checksum", "checksum", "valid", "exact", "pass"}:
            _fail("ProgramIo probe fields changed")
        expected = expected_probes.get(row.get("id", ""))
        if (
            expected is None
            or _number(row, "die") != expected.die_id
            or _number(row, "address") != expected.address
            or _number(row, "bytes") != expected.bytes
            or row.get("expected_checksum") != expected.checksum
            or row.get("checksum") != expected.checksum
            or any(_number(row, key) != 1 for key in ("valid", "exact", "pass"))
        ):
            _fail("updated-weight probe closure changed")
        seen.add(expected.probe_id)
    if seen != set(expected_probes):
        _fail("updated-weight probe identity closure changed")

    memory = _memory_markers(output, cores)
    forward = _markers(output, _CE_FORWARD, Dp4CeForward, cores)
    backward = _markers(output, _CE_BACKWARD, Dp4CeBackward, cores)
    sgd = _markers(output, _SGD, Dp4Sgd, cores)
    if (memory, forward, backward, sgd) != (
        expectation.memory, expectation.ce_forward,
        expectation.ce_backward, expectation.sgd,
    ):
        _fail("per-core memory/CE/SGD work changed")

    d2d = _exact_row(output, _D2D_TYPE, ("request_in", "request_out", "ack_in", "ack_out", "data_in", "data_out"))
    requests = sum(1 for _ in expectation.flows)
    acks = 2 * requests
    data = sum(item.data_packets for item in expectation.flows)
    if tuple(_number(d2d, key) for key in ("request_in", "request_out", "ack_in", "ack_out", "data_in", "data_out")) != (requests, requests, acks, acks, data, data):
        _fail("aggregate D2D packet quotient changed")
    pattern = re.compile(
        r"\[D2D_LINK\] idx=(\d+) die(\d+)->die(\d+) dir=[A-Z]+ "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) data_in=(\d+) data_out=(\d+)\."
    )
    links = tuple(sorted(
        (int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(6)), int(match.group(8)))
        for match in pattern.finditer(output)
    ))
    expected_links = tuple(sorted(
        (item.source_die, item.destination_die, item.request_packets, item.ack_packets, item.data_packets)
        for item in expectation.links
    ))
    if links != expected_links:
        _fail("directed D2D link packet closure changed")

    host = _exact_row(output, _HOST, ("done_total", "ack_total", "mismatch", "per_lane_done"))
    if (
        _number(host, "done_total") != expectation.done_total
        or _number(host, "ack_total") != expectation.ack_total
        or _number(host, "mismatch") != 0
    ):
        _fail("host ACK/DONE totals changed")
    lane_values = tuple(int(item, 10) for item in host["per_lane_done"].split(","))
    if any(item not in (0, 1) for item in lane_values) or sum(lane_values) != len(cores):
        _fail("per-lane DONE closure changed")
    signature = _exact_row(output, _HOSTSIG, ("done", "ack"))
    if _signature(signature["done"], 2) != tuple((core, 1) for core in cores):
        _fail("per-core DONE signature changed")
    ack_by_core = {core: 0 for core in cores}
    for core, _lane, count in _signature(signature["ack"], 3):
        if core not in ack_by_core:
            _fail("ACK signature references an inactive core")
        ack_by_core[core] += count
    if tuple(ack_by_core.values()) != (2, 2, 2, 2):
        _fail("per-core ACK signature changed")

    endpoints = _rows(output, _P5)
    if (
        len(endpoints) != 4
        or tuple(sorted(_number(row, "core") for row in endpoints)) != cores
        or any(set(row) != {"core", "residual"} or _number(row, "residual") for row in endpoints)
    ):
        _fail("P2P endpoint drain changed")
    timing = _exact_row(output, _P5_TIMING, ("residual",))
    collective = _exact_row(output, _COLL, ("tree_entries", "reduce_nodes", "barriers", "gather", "reduce_rx", "endpoints", "dte_tokens", "event"))
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
    if not makespan:
        _fail("makespan must be positive")
    prefixes = (_STATUS, _PROBE, _MEMORY, _CE_FORWARD, _CE_BACKWARD, _SGD, _SIM, _HOST, _HOSTSIG, _P5, _P5_TIMING, _COLL, _DRAIN, _D2D_TYPE, _D2D_LINK)
    marker_lines = []
    for line in output.splitlines():
        positions = tuple(position for prefix in prefixes if (position := line.find(prefix)) >= 0)
        if positions:
            marker_lines.append(line[min(positions):].split(" | ", 1)[0].rstrip(". "))
        elif _DONE in line:
            marker_lines.append(_DONE)
    digest = hashlib.sha256("\n".join(marker_lines).encode("utf-8")).hexdigest()
    return Dp4RuntimeObservation(
        makespan, digest, memory, forward, backward, sgd,
        requests, acks, data, expectation.done_total, True,
    )


def validate_dp4_repeat(first: Dp4RuntimeObservation, second: Dp4RuntimeObservation) -> Dp4RepeatEvidence:
    if first != second:
        _fail("runtime observation repeat changed")
    return Dp4RepeatEvidence(first.marker_digest, second.marker_digest)


def _run(command: tuple[str, ...], cwd: Path, timeout: int) -> str:
    completed = subprocess.run(command, cwd=cwd, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if completed.returncode:
        _fail(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout}")
    return completed.stdout


def run_official_dp4_tree_ar(
    args: argparse.Namespace,
    *,
    production_builder: ProductionBuilder,
    command_runner: CommandRunner = _run,
) -> tuple[Dp4RuntimeObservation, Dp4RepeatEvidence]:
    from llm.frontend.wafer_frontend.schema.serde import canonical_json

    pre = production_builder(None)
    with tempfile.TemporaryDirectory(prefix="s2-lite-dp4-tree-ar-", dir=args.runtime_root) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        manifest_path.write_text(canonical_json(pre.linked.manifest), encoding="utf-8")
        artifacts = (directory / "program.0.npup", directory / "program.1.npup")
        reports = (directory / "finalizer.0.json", directory / "finalizer.1.json")
        for artifact, report in zip(artifacts, reports, strict=True):
            command_runner((str(args.finalizer), "--input", str(manifest_path), "--output", str(artifact), "--report", str(report)), args.runtime_root, 120)
        artifact_bytes = tuple(item.read_bytes() for item in artifacts)
        report_values = tuple(json.loads(item.read_text(encoding="utf-8")) for item in reports)
        if artifact_bytes[0] != artifact_bytes[1] or report_values[0] != report_values[1]:
            _fail("finalizer byte/report repeat changed")
        artifact_sha = hashlib.sha256(artifact_bytes[0]).hexdigest()
        actual = production_builder(artifact_sha)
        if actual.linked != pre.linked or actual.program_io is None or actual.expectation is None:
            _fail("actual-SHA production rebuild changed")
        sidecar = directory / "program_io.json"
        sidecar.write_text(canonical_json(actual.program_io), encoding="utf-8")
        resolver = command_runner((str(args.resolver), "--resolve", str(manifest_path), str(artifacts[0]), str(sidecar)), args.runtime_root, 120)
        if f"initializations={len(actual.program_io.initializations)}" not in resolver or f"probes={len(actual.program_io.output_probes)}" not in resolver:
            _fail("resolver lost ProgramIo exact counts")
        hardware_path = Path(actual.case.hardware_path)
        mapping_path = Path(actual.case.mapping_path)
        outputs = tuple(command_runner((str(args.npusim), "--program", str(artifacts[0]), "--linked-manifest", str(manifest_path), "--program-io", str(sidecar), "--hardware-config", str(hardware_path), "--simulation-config", str(args.simulation), "--mapping-config", str(mapping_path), "--trace-window", "1000000"), args.runtime_root, args.timeout) for _ in range(2))
        observations = tuple(observe_dp4_tree_ar_runtime(output, actual.expectation) for output in outputs)
        return observations[0], validate_dp4_repeat(*observations)


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the official DP4 tree-AllReduce timing case")
    parser.add_argument("--finalizer", type=Path, default=Path("build/npusim_program_finalizer"))
    parser.add_argument("--resolver", type=Path, default=Path("build/npusim_program_io_selftest"))
    parser.add_argument("--npusim", type=Path, default=Path("build/npusim"))
    parser.add_argument("--simulation", type=Path, default=Path("llm/test/sram/simulation.json"))
    parser.add_argument("--runtime-root", type=Path, default=Path("."))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        _fail("timeout must be positive")
    observation, _repeat = run_official_dp4_tree_ar(
        args,
        production_builder=build_production_chain,
    )
    print(
        "[S2-LITE DP4 TREE-AR] PASS: timing_execution=1 functional_execution=0 "
        f"repeat=2 makespan_cycles={observation.makespan_cycles} "
        f"marker_digest={observation.marker_digest}"
    )
    return 0


__all__ = [
    "Dp4CeBackward", "Dp4CeForward", "Dp4FlowExpectation",
    "Dp4LinkTraffic", "Dp4Memory", "Dp4ProbeExpectation",
    "Dp4ProductionChain", "Dp4RepeatEvidence", "Dp4RuntimeExpectation",
    "Dp4RuntimeObservation", "Dp4Sgd", "observe_dp4_tree_ar_runtime",
    "build_production_chain", "main", "run_official_dp4_tree_ar",
    "validate_dp4_repeat",
]


if __name__ == "__main__":
    raise SystemExit(main())
