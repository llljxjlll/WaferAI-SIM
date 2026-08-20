#!/usr/bin/env python3
"""Injectable strict parser/runner boundary for the S3-Lite DP4 matrix."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Callable

from lite_moe_dp4_cases import LiteMoeDp4Mode
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    ProgramSymbolKind,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.lite_moe_dp4 import (
    S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID,
    S3_LITE_MOE_DP4_INFER_CASE_ID,
    S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID,
)


_STATUS = "[PROGRAM_IO] "
_PROBE = "[PROGRAM_IO_PROBE] "
_MEMORY = "[PROGRAM_MEMORY] "
_D2D = "[D2D_TYPE] "
_LINK = "[D2D_LINK] "
_HOST = "[HOSTLANE] "
_HOSTSIG = "[HOSTSIG] "
_P5 = "[P5 P2P DRAIN] "
_P5_TIMING = "[P5 P2P TIMING DRAIN] "
_COLL = "[COLL_DRAIN] "
_DRAIN = "[DRAIN] "
_SIM = "[SIM_RESULT] "
_DONE = "End DONE reception"


def _fail(message: str) -> None:
    raise RuntimeError(f"[S3-LITE MOE DP4] FAIL: {message}")


@dataclass(frozen=True, slots=True)
class Dp4MoeFlow:
    id: str
    source_die: int
    destination_die: int
    bytes: int
    die_path: tuple[int, ...]

    @property
    def packets(self) -> int:
        return self.bytes // 16


@dataclass(frozen=True, slots=True)
class Dp4MoeProbe:
    id: str
    target_kind: str
    endpoint: int
    address: int
    bytes: int
    checksum: str
    role: str


@dataclass(frozen=True, slots=True)
class Dp4MoeMemory:
    core: int
    lsu_issued: int
    lsu_completed: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    sram_read_bytes: int
    sram_write_bytes: int


@dataclass(frozen=True, slots=True)
class Dp4MoeExpectation:
    mode: LiteMoeDp4Mode
    case_id: str
    artifact_sha256: str
    active_cores: tuple[int, ...]
    flows: tuple[Dp4MoeFlow, ...]
    probes: tuple[Dp4MoeProbe, ...]
    memory: tuple[Dp4MoeMemory, ...]
    initialization_count: int
    gemm_count: int = 0
    swiglu_count: int = 0
    tape_copy_count: int = 0
    wgrad_count: int = 0
    reduce_count: int = 0
    sgd_count: int = 0

    def validate(self) -> None:
        expected_ids = {
            LiteMoeDp4Mode.INFER:
                "case.s3_lite.dp4_ep4.static_moe_infer",
            LiteMoeDp4Mode.TRAIN_FORWARD:
                "case.s3_lite.dp4_ep4.static_moe_train_forward",
            LiteMoeDp4Mode.DOWN_WGRAD:
                "case.s3_lite.dp4_ep4.static_moe_down_wgrad",
        }
        if self.case_id != expected_ids.get(self.mode):
            _fail("case identity/mode changed")
        if not re.fullmatch(r"[0-9a-f]{64}", self.artifact_sha256):
            _fail("artifact SHA is not canonical")
        if len(self.active_cores) != 4 or self.active_cores != tuple(sorted(set(self.active_cores))):
            _fail("requires four unique active cores")
        expected_flow_count = 6 if self.mode is LiteMoeDp4Mode.DOWN_WGRAD else 12
        expected_bytes = 192 if self.mode is LiteMoeDp4Mode.DOWN_WGRAD else 384
        expected_packets = 12 if self.mode is LiteMoeDp4Mode.DOWN_WGRAD else 24
        if (
            len(self.flows) != expected_flow_count
            or len({flow.id for flow in self.flows}) != expected_flow_count
            or any(flow.bytes != 32 or flow.packets != 2 for flow in self.flows)
            or sum(flow.bytes for flow in self.flows) != expected_bytes
            or sum(flow.packets for flow in self.flows) != expected_packets
        ):
            _fail("mode-specific flow byte/packet quotient changed")
        for flow in self.flows:
            if (
                len(flow.die_path) < 2
                or flow.die_path[0] != flow.source_die
                or flow.die_path[-1] != flow.destination_die
                or flow.source_die == flow.destination_die
            ):
                _fail("flow path does not close remote endpoints")
        roles = tuple(probe.role for probe in self.probes)
        if any(
            probe.endpoint < 0
            or probe.address < 0
            or probe.bytes <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", probe.checksum)
            for probe in self.probes
        ):
            _fail("probe endpoint/address/bytes/checksum is not canonical")
        if self.mode is LiteMoeDp4Mode.INFER:
            valid_probes = (
                len(self.probes) == 8
                and roles.count("combined") == 8
                and all(probe.target_kind == "sram" and probe.bytes == 32 for probe in self.probes)
                and (self.gemm_count, self.swiglu_count, self.tape_copy_count) == (24, 8, 0)
            )
        elif self.mode is LiteMoeDp4Mode.TRAIN_FORWARD:
            valid_probes = (
                len(self.probes) == 16
                and roles.count("combined") == 8
                and roles.count("tape") == 8
                and all(
                    probe.bytes == (64 if probe.role == "tape" else 32)
                    for probe in self.probes
                )
                and all(probe.target_kind == "sram" for probe in self.probes)
                and (self.gemm_count, self.swiglu_count, self.tape_copy_count) == (24, 8, 8)
            )
        else:
            valid_probes = (
                len(self.probes) == 4
                and roles == ("updated_weight",) * 4
                and all(probe.target_kind == "hbm" and probe.bytes == 1024 for probe in self.probes)
                and (self.wgrad_count, self.reduce_count, self.sgd_count) == (8, 4, 4)
            )
        if not valid_probes or len({probe.id for probe in self.probes}) != len(self.probes):
            _fail("mode-specific output probe quotient changed")
        if len(self.memory) != 4 or tuple(item.core for item in self.memory) != self.active_cores:
            _fail("memory expectations must exactly cover active cores")
        if self.mode is LiteMoeDp4Mode.DOWN_WGRAD and sum(item.hbm_write_bytes for item in self.memory) != 4096:
            _fail("backward updated-weight HBM write must total 4096B")


@dataclass(frozen=True, slots=True)
class Dp4MoeObservation:
    case_id: str
    makespan_cycles: int
    marker_digest: str
    data_packets: int
    data_packet_hops: int
    ack_total: int
    done_total: int
    timing_execution: bool = True
    functional_execution: bool = False


def _row(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line[len(prefix):].strip().split():
        if "=" not in token:
            _fail(f"malformed token in {prefix.strip()}")
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
            result.append(_row(line[position:].split(" | ", 1)[0].rstrip(". "), prefix))
    return tuple(result)


def _number(row: dict[str, str], key: str) -> int:
    try:
        result = int(row[key], 10)
    except (KeyError, ValueError) as error:
        _fail(f"missing/non-decimal {key!r}")
        raise AssertionError from error
    if result < 0:
        _fail(f"negative {key!r}")
    return result


def _exact(output: str, prefix: str, fields: tuple[str, ...]) -> dict[str, str]:
    rows = _rows(output, prefix)
    if len(rows) != 1 or set(rows[0]) != set(fields):
        _fail(f"{prefix.strip()} exact row changed")
    return rows[0]


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


def observe_lite_moe_dp4_runtime(
    output: str,
    expectation: Dp4MoeExpectation,
) -> Dp4MoeObservation:
    expectation.validate()
    if "[PROTO_WAIT]" in output or "[D2D_BEHA]" in output or output.count(_DONE) != 4:
        _fail("completion boundary changed")
    status = _rows(output, _STATUS)
    if tuple(row.get("phase") for row in status) != ("resolved", "applied", "verify"):
        _fail("ProgramIo phases changed")
    for row in status:
        if (
            set(row) != {"phase", "mode", "checksum", "initializations", "probes", "pass"}
            or row.get("mode") != "timing"
            or _number(row, "initializations") != expectation.initialization_count
            or _number(row, "probes") != len(expectation.probes)
            or _number(row, "pass") != 1
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("checksum", ""))
        ):
            _fail("ProgramIo status changed")
    if any(row["checksum"] != expectation.artifact_sha256 for row in status[:2]):
        _fail("ProgramIo did not use actual artifact SHA")
    expected_probes = {probe.id: probe for probe in expectation.probes}
    observed = _rows(output, _PROBE)
    if len(observed) != len(expected_probes):
        _fail("probe multiplicity changed")
    for row in observed:
        probe = expected_probes.get(row.get("id", ""))
        if probe is None:
            _fail("unknown probe id")
        endpoint_field = "die" if probe.target_kind == "hbm" else "core"
        if set(row) != {
            "id", endpoint_field, "address", "bytes", "expected_checksum",
            "checksum", "valid", "exact", "pass",
        }:
            _fail("probe field set changed")
        if (
            _number(row, endpoint_field) != probe.endpoint
            or _number(row, "address") != probe.address
            or _number(row, "bytes") != probe.bytes
            or row.get("expected_checksum") != probe.checksum
            or row.get("checksum") != probe.checksum
            or any(_number(row, key) != 1 for key in ("valid", "exact", "pass"))
        ):
            _fail("probe exact closure changed")
    memory_fields = (
        "core", "lsu_issued", "lsu_completed", "lsu_hbm_read_bytes",
        "lsu_hbm_write_bytes", "lsu_sram_read_bytes", "lsu_sram_write_bytes",
        "lsu_residual", "dte_residual",
    )
    memory = tuple(sorted((
        Dp4MoeMemory(
            _number(row, "core"), _number(row, "lsu_issued"),
            _number(row, "lsu_completed"), _number(row, "lsu_hbm_read_bytes"),
            _number(row, "lsu_hbm_write_bytes"), _number(row, "lsu_sram_read_bytes"),
            _number(row, "lsu_sram_write_bytes"),
        )
        for row in _rows(output, _MEMORY)
        if set(row) == set(memory_fields)
        and not _number(row, "lsu_residual")
        and not _number(row, "dte_residual")
    ), key=lambda item: item.core))
    if memory != expectation.memory:
        _fail("memory work/residual changed")
    logical_data_packets = sum(flow.packets for flow in expectation.flows)
    requests = sum(len(flow.die_path) - 1 for flow in expectation.flows)
    data_packet_hops = sum(
        flow.packets * (len(flow.die_path) - 1)
        for flow in expectation.flows
    )
    d2d = _exact(output, _D2D, ("request_in", "request_out", "ack_in", "ack_out", "data_in", "data_out"))
    if tuple(_number(d2d, key) for key in (
        "request_in", "request_out", "ack_in", "ack_out", "data_in", "data_out"
    )) != (
        requests, requests, 2 * requests, 2 * requests,
        data_packet_hops, data_packet_hops,
    ):
        _fail("aggregate D2D quotient changed")
    link_expected: dict[tuple[int, int], list[int]] = {}
    for flow in expectation.flows:
        for source, destination in zip(flow.die_path, flow.die_path[1:]):
            counts = link_expected.setdefault((source, destination), [0, 0, 0])
            counts[0] += 1
            counts[2] += flow.packets
    requests_by_link = {
        edge: counts[0] for edge, counts in link_expected.items()
    }
    if expectation.mode is LiteMoeDp4Mode.DOWN_WGRAD:
        for source, destination in tuple(requests_by_link):
            link_expected.setdefault((destination, source), [0, 0, 0])
        for (source, destination), counts in link_expected.items():
            counts[1] = (
                requests_by_link.get((source, destination), 0)
                + requests_by_link.get((destination, source), 0)
            )
    else:
        for counts in link_expected.values():
            counts[1] = 2 * counts[0]
    if sum(counts[2] for counts in link_expected.values()) != data_packet_hops:
        _fail("directed-link DATA sum does not equal aggregate packet-hop")
    pattern = re.compile(
        r"\[D2D_LINK\] idx=\d+ die(\d+)->die(\d+) dir=[A-Z]+ "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)\."
    )
    matches = tuple(pattern.finditer(output))
    if any(
        match.group(3) != match.group(4)
        or match.group(5) != match.group(6)
        or match.group(7) != match.group(8)
        for match in matches
    ):
        _fail("directed-link in/out counters disagree")
    actual_links = {
        (int(match.group(1)), int(match.group(2))):
            [int(match.group(3)), int(match.group(5)), int(match.group(7))]
        for match in matches
    }
    if actual_links != link_expected:
        _fail(
            f"directed-link byte-hop quotient changed: expected={link_expected!r} actual={actual_links!r}"
        )
    host = _exact(output, _HOST, ("done_total", "ack_total", "mismatch", "per_lane_done"))
    if (_number(host, "done_total"), _number(host, "ack_total"), _number(host, "mismatch")) != (4, 8, 0):
        _fail("ACK/DONE totals changed")
    signature = _exact(output, _HOSTSIG, ("done", "ack"))
    if _signature(signature["done"], 2) != tuple(
        (core, 1) for core in expectation.active_cores
    ):
        _fail("per-core DONE signature changed")
    ack_by_core = {core: 0 for core in expectation.active_cores}
    for core, _lane, count in _signature(signature["ack"], 3):
        if core not in ack_by_core:
            _fail("ACK signature references inactive core")
        ack_by_core[core] += count
    if tuple(ack_by_core.values()) != (2, 2, 2, 2):
        _fail("per-core ACK signature changed")
    endpoints = _rows(output, _P5)
    if len(endpoints) != 4 or tuple(sorted(_number(row, "core") for row in endpoints)) != expectation.active_cores or any(_number(row, "residual") for row in endpoints):
        _fail("P5 endpoint drain changed")
    if _number(_exact(output, _P5_TIMING, ("residual",)), "residual"):
        _fail("P5 timing did not drain")
    collective = _exact(output, _COLL, ("tree_entries", "reduce_nodes", "barriers", "gather", "reduce_rx", "endpoints", "dte_tokens", "event"))
    if any(_number(collective, key) for key in collective):
        _fail("collective/event did not drain")
    drain_rows = _rows(output, _DRAIN)
    drain: dict[str, str] = {}
    for row in drain_rows:
        if set(drain).intersection(row):
            _fail("duplicate drain field")
        drain.update(row)
    if (
        set(drain) != {"router_residual", "d2d_link_residual"}
        or any(_number(drain, key) for key in drain)
    ):
        _fail("router/link did not drain")
    makespan = _number(_exact(output, _SIM, ("makespan_cycles",)), "makespan_cycles")
    if not makespan:
        _fail("makespan must be positive")
    marker_prefixes = (
        _STATUS, _PROBE, _MEMORY, _D2D, _LINK, _HOST, _HOSTSIG,
        _P5, _P5_TIMING, _COLL, _DRAIN, _SIM,
    )
    marker_lines = []
    for line in output.splitlines():
        positions = tuple(
            position for prefix in marker_prefixes
            if (position := line.find(prefix)) >= 0
        )
        if positions:
            marker_lines.append(
                line[min(positions):].split(" | ", 1)[0].rstrip(". ")
            )
        elif _DONE in line:
            marker_lines.append(_DONE)
    return Dp4MoeObservation(
        expectation.case_id,
        makespan,
        hashlib.sha256("\n".join(marker_lines).encode()).hexdigest(),
        logical_data_packets,
        data_packet_hops,
        8,
        4,
    )


def validate_lite_moe_dp4_repeat(
    first: Dp4MoeObservation,
    second: Dp4MoeObservation,
) -> None:
    if first != second:
        _fail(f"runtime repeat changed: first={first!r} second={second!r}")


@dataclass(frozen=True, slots=True)
class Dp4MoeProduction:
    mode: LiteMoeDp4Mode
    chain: object
    expectation: Dp4MoeExpectation

    def validate(self) -> None:
        validate = getattr(self.chain, "validate", None)
        if not callable(validate):
            _fail("production chain lacks typed validation")
        validate()
        self.expectation.validate()
        if (
            self.expectation.mode is not self.mode
            or self.expectation.case_id != self.chain.case.case_id
            or self.expectation.artifact_sha256
            != self.chain.program_io.program_artifact_sha256
        ):
            _fail("production chain/expectation provenance changed")


ProductionBuilder = Callable[[LiteMoeDp4Mode, str], Dp4MoeProduction]
CommandRunner = Callable[[tuple[str, ...], Path, int], str]


def _value(value: object) -> object:
    return getattr(value, "value", value)


def _leaf(fragment: object) -> object:
    return fragment.fragment if hasattr(fragment, "fragment") else fragment


def _production_expectation(
    mode: LiteMoeDp4Mode,
    chain: object,
) -> Dp4MoeExpectation:
    case = chain.case
    linked = chain.linked
    program_io = chain.program_io
    runtime_by_logical = {
        item.logical_core: item.runtime_core_id
        for item in linked.manifest.core_bindings
    }
    fragments = {
        _leaf(item).id: _leaf(item)
        for item in linked.manifest.fragments
    }
    memory_by_core: dict[int, list[int]] = {
        core: [0, 0, 0, 0, 0, 0]
        for core in sorted(runtime_by_logical.values())
    }
    for stream in linked.manifest.core_streams:
        runtime_core = stream.runtime_core_id
        for record_ref in stream.records:
            fragment = fragments[record_ref.fragment_id]
            fragment_stream = next(
                item for item in fragment.core_streams
                if item.logical_core == stream.logical_core
            )
            record = fragment_stream.records[record_ref.fragment_record_index]
            opcode = record.opcode
            if opcode not in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE):
                continue
            operands = {item.name: item for item in record.operands}
            size = operands["size_bytes"].literal_value
            if type(size) is not int or size <= 0:
                _fail("LSU record lacks a positive literal size")
            counters = memory_by_core[runtime_core]
            counters[0] += 1
            counters[1] += 1
            if opcode is RecordOpcode.LSU_LOAD:
                counters[2] += size
                counters[5] += size
            else:
                counters[3] += size
                counters[4] += size
    memory = tuple(
        Dp4MoeMemory(core, *counters)
        for core, counters in sorted(memory_by_core.items())
    )

    routes = {
        route.id: route
        for group in case.forward.n4.graph.groups
        for route in group.embedding.routes
    }
    if mode is LiteMoeDp4Mode.DOWN_WGRAD:
        assert case.backward is not None
        flows = tuple(
            Dp4MoeFlow(
                item.id,
                item.source_die_id,
                item.destination_die_id,
                item.bytes,
                routes[item.reverse_pair_route_ref].die_path,
            )
            for item in case.backward.remote_gradients
        )
    else:
        flows = tuple(
            Dp4MoeFlow(
                item.id,
                item.source_die_id,
                item.destination_die_id,
                item.bytes,
                routes[item.pair_route_ref].die_path,
            )
            for item in case.forward.projection.flows
        )

    blobs = {item.id: item for item in program_io.blobs}
    definitions = {
        item.symbol.id: item
        for item in linked.manifest.program_symbol_definitions
    }
    buffer_abis = {
        abi.id: abi
        for item in linked.manifest.fragments
        for abi in _leaf(item).buffer_abi
    }
    state_abis = {
        abi.id: abi
        for item in linked.manifest.fragments
        for abi in _leaf(item).state_abi
    }
    combined = set(case.forward.global_dag.combined_output_refs)
    tapes = (
        {item.value_ref for item in case.train_forward.tape_buffers}
        if case.train_forward is not None
        else set()
    )
    probes = []
    for probe in program_io.output_probes:
        blob = blobs.get(probe.blob_ref)
        if blob is None:
            _fail("probe lacks its exact ProgramBlob")
        target = probe.target
        if _value(target.kind) == "sram":
            abi = buffer_abis.get(target.buffer_abi_id)
            regions = tuple(
                item for item in definitions.values()
                if item.symbol.kind is ProgramSymbolKind.SRAM_REGION
                and abi is not None
                and item.symbol.source_ref == abi.region_ref
                and abi.logical_core in item.logical_cores
            )
            if abi is None or len(regions) != 1:
                _fail("SRAM probe lacks one BufferABI/region witness")
            role = (
                "tape" if abi.value_id in tapes
                else "combined" if abi.value_id in combined
                else ""
            )
            endpoint = target.runtime_core_id
            address = regions[0].value + abi.region_offset_bytes + probe.offset_bytes
            target_kind = "sram"
        elif _value(target.kind) == "hbm":
            abi = state_abis.get(target.state_abi_id)
            if abi is None:
                _fail("HBM probe lacks one StateABI witness")
            role = "updated_weight"
            endpoint = abi.die_id
            address = abi.address + probe.offset_bytes
            target_kind = "hbm"
        else:
            _fail("probe target kind is not SRAM/HBM")
        probes.append(Dp4MoeProbe(
            probe.id,
            target_kind,
            endpoint,
            address,
            probe.length_bytes,
            blob.sha256,
            role,
        ))
    counts = (
        (24, 8, 8 if mode is LiteMoeDp4Mode.TRAIN_FORWARD else 0, 0, 0, 0)
        if mode is not LiteMoeDp4Mode.DOWN_WGRAD
        else (0, 0, 0, 8, 4, 4)
    )
    result = Dp4MoeExpectation(
        mode,
        case.case_id,
        program_io.program_artifact_sha256,
        tuple(sorted(runtime_by_logical.values())),
        flows,
        tuple(sorted(probes, key=lambda item: item.id)),
        memory,
        len(program_io.initializations),
        *counts,
    )
    result.validate()
    return result


def build_lite_moe_dp4_production(
    mode: LiteMoeDp4Mode,
    artifact_sha256: str,
) -> Dp4MoeProduction:
    """Build one typed lower/link/actual-SHA ProgramIo runtime input."""

    from lite_moe_dp4_cases import (
        build_lite_moe_dp4_backward_program_io_case,
        build_lite_moe_dp4_infer_program_io_case,
        build_lite_moe_dp4_train_forward_program_io_case,
    )

    builders = {
        LiteMoeDp4Mode.INFER: build_lite_moe_dp4_infer_program_io_case,
        LiteMoeDp4Mode.TRAIN_FORWARD:
            build_lite_moe_dp4_train_forward_program_io_case,
        LiteMoeDp4Mode.DOWN_WGRAD:
            build_lite_moe_dp4_backward_program_io_case,
    }
    if type(mode) is not LiteMoeDp4Mode:
        _fail("production mode must be typed")
    chain = builders[mode](artifact_sha256)
    result = Dp4MoeProduction(
        mode,
        chain,
        _production_expectation(mode, chain),
    )
    result.validate()
    return result


def _run(command: tuple[str, ...], cwd: Path, timeout: int) -> str:
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
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


def run_lite_moe_dp4_runtime_matrix(
    args: argparse.Namespace,
    *,
    production_builder: ProductionBuilder = build_lite_moe_dp4_production,
    command_runner: CommandRunner = _run,
) -> tuple[Dp4MoeObservation, ...]:
    """Run the three production cases twice from independently finalized bytes."""

    from llm.frontend.wafer_frontend.schema.serde import (
        canonical_digest,
        canonical_json,
    )

    observations = []
    for mode in LiteMoeDp4Mode:
        pre = production_builder(mode, "0" * 64)
        with tempfile.TemporaryDirectory(
            prefix=f"s3-lite-dp4-{mode.value}-",
            dir=args.runtime_root,
        ) as raw:
            directory = Path(raw)
            manifest = pre.chain.linked.manifest
            manifest_path = directory / "linked.json"
            manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
            artifacts = (directory / "program.0.npup", directory / "program.1.npup")
            reports = (directory / "finalizer.0.json", directory / "finalizer.1.json")
            for artifact, report in zip(artifacts, reports, strict=True):
                command_runner((
                    str(args.finalizer), "--input", str(manifest_path),
                    "--output", str(artifact), "--report", str(report),
                ), args.runtime_root, 120)
            artifact_bytes = tuple(path.read_bytes() for path in artifacts)
            report_values = tuple(
                json.loads(path.read_text(encoding="utf-8")) for path in reports
            )
            if artifact_bytes[0] != artifact_bytes[1] or report_values[0] != report_values[1]:
                _fail(f"{mode.value} finalizer byte/report repeat changed")
            artifact_sha = hashlib.sha256(artifact_bytes[0]).hexdigest()
            leaf_streams = tuple(
                stream
                for item in manifest.fragments
                for stream in _leaf(item).core_streams
            )
            expected_report = {
                "artifact_sha256": artifact_sha,
                "artifact_bytes": len(artifact_bytes[0]),
                "core_count": 4,
                "record_count": sum(len(stream.records) for stream in leaf_streams),
                "relocation_count": sum(
                    len(stream.address_relocations) for stream in leaf_streams
                ),
                "linked_manifest_id": manifest.id,
                "linked_manifest_digest": canonical_digest(manifest),
            }
            if any(
                report_values[0].get(key) != value
                for key, value in expected_report.items()
            ):
                _fail(f"{mode.value} finalizer report exact closure changed")
            actual = production_builder(mode, artifact_sha)
            if actual.chain.linked != pre.chain.linked:
                _fail(f"{mode.value} actual-SHA rebuild changed linked program")
            sidecar = directory / "program_io.json"
            sidecar.write_text(canonical_json(actual.chain.program_io), encoding="utf-8")
            resolver = command_runner((
                str(args.resolver), "--resolve", str(manifest_path),
                str(artifacts[0]), str(sidecar),
            ), args.runtime_root, 120)
            if (
                f"initializations={len(actual.chain.program_io.initializations)}"
                not in resolver
                or f"probes={len(actual.chain.program_io.output_probes)}" not in resolver
            ):
                _fail(f"{mode.value} resolver lost ProgramIo exact counts")
            outputs = tuple(command_runner((
                str(args.npusim), "--program", str(artifacts[0]),
                "--linked-manifest", str(manifest_path),
                "--program-io", str(sidecar),
                "--hardware-config", str(actual.chain.case.hardware_path),
                "--simulation-config", str(args.simulation),
                "--mapping-config", str(actual.chain.case.mapping_path),
                "--trace-window", "1000000",
            ), args.runtime_root, args.timeout) for _ in range(2))
            repeated = tuple(
                observe_lite_moe_dp4_runtime(output, actual.expectation)
                for output in outputs
            )
            validate_lite_moe_dp4_repeat(*repeated)
            observations.append(repeated[0])
    return tuple(observations)


MatrixRunner = Callable[[argparse.Namespace], tuple[Dp4MoeObservation, ...]]


def main(
    argv: tuple[str, ...] | None = None,
    *,
    matrix_runner: MatrixRunner = run_lite_moe_dp4_runtime_matrix,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", type=Path, default=Path("build/npusim"))
    parser.add_argument(
        "--finalizer", type=Path,
        default=Path("build/npusim_program_finalizer"),
    )
    parser.add_argument(
        "--resolver", type=Path,
        default=Path("build/npusim_program_io_selftest"),
    )
    parser.add_argument(
        "--simulation", type=Path,
        default=Path("llm/test/sram/simulation.json"),
    )
    parser.add_argument("--runtime-root", type=Path, default=Path("."))
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        _fail("timeout must be positive")
    observations = matrix_runner(args)
    expected_case_ids = (
        S3_LITE_MOE_DP4_INFER_CASE_ID,
        S3_LITE_MOE_DP4_TRAIN_FORWARD_CASE_ID,
        S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID,
    )
    expected_packets = (24, 24, 12)
    if len(observations) != 3 or any(
        type(item) is not Dp4MoeObservation for item in observations
    ):
        _fail("matrix runner did not return three typed observations")
    if (
        tuple(item.case_id for item in observations) != expected_case_ids
        or tuple(item.data_packets for item in observations) != expected_packets
        or any(
            item.makespan_cycles <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", item.marker_digest)
            or item.ack_total != 8
            or item.done_total != 4
            or not item.timing_execution
            or item.functional_execution
            for item in observations
        )
    ):
        _fail("matrix runner did not return exact I/TF/TB order")
    for observation in observations:
        print(
            f"[S3-LITE DP4 {observation.case_id}] PASS: "
            "timing_execution=1 functional_execution=0 repeat=2 "
            f"makespan_cycles={observation.makespan_cycles} "
            f"marker_digest={observation.marker_digest}"
        )
    print("[S3-LITE DP4 MATRIX] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Dp4MoeExpectation", "Dp4MoeFlow", "Dp4MoeMemory", "Dp4MoeObservation",
    "Dp4MoeProduction",
    "Dp4MoeProbe", "LiteMoeDp4Mode", "main", "observe_lite_moe_dp4_runtime",
    "build_lite_moe_dp4_production",
    "run_lite_moe_dp4_runtime_matrix",
    "validate_lite_moe_dp4_repeat",
]
