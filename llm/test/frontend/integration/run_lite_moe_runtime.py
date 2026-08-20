#!/usr/bin/env python3
"""Official S3-Lite MoE finalization, runtime, and strict evidence driver."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def _fail(message: str) -> None:
    raise RuntimeError(f"[S3-LITE MOE] FAIL: {message}")


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
        _fail(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}"
        )
    return completed.stdout


_PROGRAM_IO = "[PROGRAM_IO] "
_PROGRAM_IO_PROBE = "[PROGRAM_IO_PROBE] "
_MEMORY = "[PROGRAM_MEMORY] "
_SIM = "[SIM_RESULT] "
_HOST = "[HOSTLANE] "
_HOSTSIG = "[HOSTSIG] "
_P5 = "[P5 P2P DRAIN] "
_P5_TIMING = "[P5 P2P TIMING DRAIN] "
_COLLECTIVE = "[COLL_DRAIN] "
_DRAIN = "[DRAIN] "
_D2D_TYPE = "[D2D_TYPE] "
_D2D_LINK = "[D2D_LINK] "
_DONE = "End DONE reception."


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
    if len(rows) != 1 or set(rows[0]) != set(fields):
        _fail(f"{prefix.strip()} exact row changed: {rows}")
    return rows[0]


def _parse_signature(value: str, arity: int) -> tuple[tuple[int, ...], ...]:
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


@dataclass(frozen=True, slots=True)
class LiteMoeRuntimeExpectation:
    cores: tuple[int, int]
    hbm_read_bytes_per_core: int
    dte_transfers_per_direction: int
    transfer_bytes: int
    artifact_sha256: str
    probe_ids_and_bytes: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class LiteMoeRuntimeObservation:
    makespan_cycles: int
    marker_digest: str
    memory: tuple[tuple[int, ...], ...]
    d2d_type: tuple[int, ...]
    d2d_links: tuple[tuple[int, ...], ...]
    ack_total: int
    done_total: int


def _d2d_link_rows(output: str) -> tuple[tuple[int, ...], ...]:
    pattern = re.compile(
        r"\[D2D_LINK\] idx=(\d+) die(\d+)->die(\d+) dir=[A-Z]+ "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)\."
    )
    result = []
    for line in output.splitlines():
        match = pattern.search(line)
        if match is not None:
            result.append(tuple(int(item, 10) for item in match.groups()))
    return tuple(sorted(result))


def observe_lite_moe_runtime(
    output: str,
    expectation: LiteMoeRuntimeExpectation,
) -> LiteMoeRuntimeObservation:
    """Cross-check independent runtime markers against the typed S3 quotient."""

    if "[PROTO_WAIT]" in output or "[D2D_BEHA]" in output:
        _fail("completed S3-Lite must not contain PROTO_WAIT or behavioral D2D")
    if output.count(_DONE) != 2:
        _fail("requires one DONE boundary per active core")

    io_rows = _rows(output, _PROGRAM_IO)
    if len(io_rows) != 3 or tuple(row.get("phase") for row in io_rows) != (
        "resolved", "applied", "verify"
    ):
        _fail("ProgramIo phase closure changed")
    for row in io_rows:
        if (
            set(row) != {"phase", "mode", "initializations", "probes", "checksum", "pass"}
            or row["mode"] != "timing"
            or _number(row, "initializations") != 76
            or _number(row, "probes") != 8
            or _number(row, "pass") != 1
            or not re.fullmatch(r"[0-9a-f]{64}", row["checksum"])
        ):
            _fail(f"ProgramIo marker changed: {row}")
    if io_rows[0]["checksum"] != expectation.artifact_sha256 or io_rows[1]["checksum"] != expectation.artifact_sha256:
        _fail("ProgramIo resolved/applied checksum is not the actual artifact SHA")

    probes = _rows(output, _PROGRAM_IO_PROBE)
    if len(probes) != 8:
        _fail("requires exactly eight output probes")
    observed_probe_ids = []
    for row in probes:
        if set(row) != {
            "id", "core", "address", "bytes", "expected_checksum", "checksum",
            "valid", "exact", "pass",
        }:
            _fail(f"probe fields changed: {row}")
        if (
            row["expected_checksum"] != row["checksum"]
            or any(_number(row, key) != 1 for key in ("valid", "exact", "pass"))
            or _number(row, "core") not in expectation.cores
            or _number(row, "address") % 64
        ):
            _fail(f"probe failed exact timing verification: {row}")
        observed_probe_ids.append((row["id"], _number(row, "bytes")))
    if tuple(sorted(observed_probe_ids)) != expectation.probe_ids_and_bytes:
        _fail("runtime probe identities/byte spans changed")

    memory_fields = (
        "core", "lsu_issued", "lsu_completed", "lsu_hbm_read_bytes",
        "lsu_hbm_write_bytes", "lsu_sram_read_bytes", "lsu_sram_write_bytes",
        "lsu_residual", "dte_residual",
    )
    memory_rows = _rows(output, _MEMORY)
    if len(memory_rows) != 2 or any(set(row) != set(memory_fields) for row in memory_rows):
        _fail("requires two exact PROGRAM_MEMORY rows")
    memory = tuple(
        tuple(_number(row, key) for key in memory_fields)
        for row in sorted(memory_rows, key=lambda item: _number(item, "core"))
    )
    expected_memory = tuple(
        (
            core, 12, 12, expectation.hbm_read_bytes_per_core, 0, 0,
            expectation.hbm_read_bytes_per_core, 0, 0,
        )
        for core in expectation.cores
    )
    if memory != expected_memory:
        _fail(f"HBM/SRAM timing work changed: {memory}")

    d2d = _exact_row(
        output, _D2D_TYPE,
        ("request_in", "request_out", "ack_in", "ack_out", "data_in", "data_out"),
    )
    d2d_type = tuple(_number(d2d, key) for key in (
        "request_in", "request_out", "ack_in", "ack_out", "data_in", "data_out"
    ))
    transfers = expectation.dte_transfers_per_direction
    packets = transfers * expectation.transfer_bytes // 16
    if d2d_type != (transfers * 2, transfers * 2, transfers * 4, transfers * 4, packets * 2, packets * 2):
        _fail(f"aggregate D2D accounting changed: {d2d_type}")
    links = _d2d_link_rows(output)
    if links != (
        (0, 0, 1, transfers, transfers, transfers * 2, transfers * 2, packets, packets),
        (1, 1, 0, transfers, transfers, transfers * 2, transfers * 2, packets, packets),
    ):
        _fail(f"directed D2D link accounting changed: {links}")

    host = _exact_row(output, _HOST, ("done_total", "ack_total", "mismatch", "per_lane_done"))
    done_total, ack_total = _number(host, "done_total"), _number(host, "ack_total")
    if (
        (done_total, ack_total, _number(host, "mismatch")) != (2, 4, 0)
        or host["per_lane_done"] != "1,0,0,0,1,0,0,0"
    ):
        _fail("host ACK/DONE closure changed")
    signature = _exact_row(output, _HOSTSIG, ("done", "ack"))
    if _parse_signature(signature["done"], 2) != tuple((core, 1) for core in expectation.cores):
        _fail("per-core DONE closure changed")
    ack_signature = _parse_signature(signature["ack"], 3)
    if sum(item[2] for item in ack_signature) != 4 or any(item[0] not in expectation.cores for item in ack_signature):
        _fail("per-core ACK closure changed")

    endpoints = _rows(output, _P5)
    if (
        len(endpoints) != 2
        or tuple(sorted(_number(row, "core") for row in endpoints)) != expectation.cores
        or any(set(row) != {"core", "residual"} or _number(row, "residual") for row in endpoints)
    ):
        _fail("P2P endpoint drain changed")
    timing = _exact_row(output, _P5_TIMING, ("residual",))
    collective = _exact_row(
        output, _COLLECTIVE,
        ("tree_entries", "reduce_nodes", "barriers", "gather", "reduce_rx", "endpoints", "dte_tokens", "event"),
    )
    drain_rows = _rows(output, _DRAIN)
    merged: dict[str, str] = {}
    for row in drain_rows:
        if set(merged).intersection(row):
            _fail("duplicate drain field")
        merged.update(row)
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
    prefixes = (
        _PROGRAM_IO, _PROGRAM_IO_PROBE, _MEMORY, _SIM, _HOST, _HOSTSIG,
        _P5, _P5_TIMING, _COLLECTIVE, _DRAIN, _D2D_TYPE, _D2D_LINK,
    )
    marker_lines = []
    for line in output.splitlines():
        positions = tuple(position for prefix in prefixes if (position := line.find(prefix)) >= 0)
        if positions:
            marker_lines.append(line[min(positions) :].split(" | ", 1)[0].rstrip(". "))
        elif _DONE in line:
            marker_lines.append(_DONE)
    return LiteMoeRuntimeObservation(
        makespan,
        hashlib.sha256("\n".join(marker_lines).encode("utf-8")).hexdigest(),
        memory,
        d2d_type,
        links,
        ack_total,
        done_total,
    )


def _build_linked() -> tuple[object, object]:
    from lite_moe_cases import build_lite_moe_execution_case
    from llm.frontend.wafer_frontend.passes import (
        link_lite_moe_n6,
        lower_lite_moe_n6,
    )

    case = build_lite_moe_execution_case()
    lowered = lower_lite_moe_n6(
        case.n6_intent,
        case.global_dag,
        case.schedule,
        case.projection,
        case.n4,
    )
    linked = link_lite_moe_n6(lowered)
    return case, linked


def run_official_lite_moe(args: argparse.Namespace) -> tuple[LiteMoeRuntimeObservation, LiteMoeRuntimeObservation]:
    """Finalize twice and run the exact S3-Lite case twice."""

    from llm.frontend.wafer_frontend.passes.program_io import (
        build_deterministic_timing_state_overrides,
        build_timing_program_io,
    )
    from llm.frontend.wafer_frontend.schema.serde import (
        canonical_digest,
        canonical_json,
    )

    case, linked = _build_linked()
    manifest = linked.manifest
    with tempfile.TemporaryDirectory(prefix="s3-lite-moe-", dir=args.runtime_root) as raw:
        directory = Path(raw)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
        hardware_path.write_text(case.source.hardware_json, encoding="utf-8")
        mapping_path.write_text(case.source.mapping_text, encoding="utf-8")

        artifacts = (directory / "program.0.npup", directory / "program.1.npup")
        reports = (directory / "finalizer.0.json", directory / "finalizer.1.json")
        artifact_bytes: list[bytes] = []
        finalization: list[dict[str, object]] = []
        for artifact, report in zip(artifacts, reports, strict=True):
            _run(
                [
                    str(args.finalizer),
                    "--input",
                    str(manifest_path),
                    "--output",
                    str(artifact),
                    "--report",
                    str(report),
                ],
                cwd=args.runtime_root,
                timeout=120,
            )
            artifact_bytes.append(artifact.read_bytes())
            finalization.append(json.loads(report.read_text(encoding="utf-8")))
        if artifact_bytes[0] != artifact_bytes[1] or finalization[0] != finalization[1]:
            _fail("finalizer byte/report repeat changed")
        artifact_sha = hashlib.sha256(artifact_bytes[0]).hexdigest()
        expected = {
            "artifact_sha256": artifact_sha,
            "artifact_bytes": len(artifact_bytes[0]),
            "core_count": 2,
            "record_count": 240,
            "relocation_count": 408,
            "linked_manifest_id": manifest.id,
            "linked_manifest_digest": canonical_digest(manifest),
        }
        if any(finalization[0].get(key) != value for key, value in expected.items()):
            _fail(f"finalizer report closure changed: {finalization[0]}")

        seeds, expected_state = build_deterministic_timing_state_overrides(linked)
        if len(seeds) != 12 or expected_state:
            _fail("S3-Lite requires twelve read-only expert state seeds")
        program_io = build_timing_program_io(
            linked,
            artifact_sha,
            state_seed_overrides=seeds,
            state_expected_overrides=expected_state,
        )
        program_io.validate_against(manifest)
        if (
            len(program_io.blobs),
            len(program_io.initializations),
            len(program_io.output_probes),
        ) != (16, 76, 8):
            _fail("ProgramIo counts changed")
        sidecar = directory / "program_io.json"
        sidecar.write_text(canonical_json(program_io), encoding="utf-8")
        resolver = _run(
            [
                str(args.resolver),
                "--resolve",
                str(manifest_path),
                str(artifacts[0]),
                str(sidecar),
            ],
            cwd=args.runtime_root,
            timeout=120,
        )
        if "initializations=76 probes=8" not in resolver:
            _fail(f"resolver lost exact ProgramIo counts: {resolver}")

        hardware = json.loads(case.source.hardware_json)
        die_pitch = int(hardware["x"]) ** 2
        action_by_id = {action.id: action for action in case.global_dag.actions}
        cores = tuple(sorted({
            action_by_id[unit.action_ref].die_id * die_pitch
            for unit in case.n6_intent.state_loads
        }))
        hbm_by_core = {
            core: sum(
                unit.bytes
                for unit in case.n6_intent.state_loads
                if action_by_id[unit.action_ref].die_id * die_pitch == core
            )
            for core in cores
        }
        direction_counts = {
            direction: sum(
                1
                for unit in case.n6_intent.dte_units
                if (unit.source_die_id, unit.destination_die_id) == direction
            )
            for direction in ((0, 1), (1, 0))
        }
        if (
            cores != (0, 16)
            or set(hbm_by_core.values()) != {12288}
            or set(direction_counts.values()) != {4}
            or {unit.bytes for unit in case.n6_intent.dte_units} != {32}
        ):
            _fail("typed S3 runtime quotient changed")
        expectation = LiteMoeRuntimeExpectation(
            cores=(0, 16),
            hbm_read_bytes_per_core=12288,
            dte_transfers_per_direction=4,
            transfer_bytes=32,
            artifact_sha256=artifact_sha,
            probe_ids_and_bytes=tuple(sorted(
                (probe.id, probe.length_bytes) for probe in program_io.output_probes
            )),
        )
        observations = []
        for index in range(2):
            output = _run(
                [
                    str(args.npusim),
                    "--program",
                    str(artifacts[0]),
                    "--linked-manifest",
                    str(manifest_path),
                    "--program-io",
                    str(sidecar),
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
            log_path = (
                args.runtime_log
                if index == 0
                else args.runtime_log.with_name(
                    f"{args.runtime_log.stem}.1{args.runtime_log.suffix}"
                )
            )
            log_path.write_text(output, encoding="utf-8")
            observations.append(observe_lite_moe_runtime(output, expectation))
        if observations[0] != observations[1]:
            _fail("runtime observation repeat changed")
        print(
            "[S3-LITE MOE] PASS: timing_execution=1 "
            "compute_functional=0 routing_functional=0 "
            f"artifact={len(artifact_bytes[0])}B records=240 relocations=408 "
            f"init=76 probes=8 repeat=2 makespan_cycles={observations[0].makespan_cycles} "
            f"sha256={artifact_sha}"
        )
        return observations[0], observations[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", required=True, type=_executable)
    parser.add_argument("--finalizer", required=True, type=_executable)
    parser.add_argument("--resolver", required=True, type=_executable)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--runtime-log", "--probe-log", dest="runtime_log", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    args.simulation = args.simulation.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_log = args.runtime_log.resolve()
    if not args.simulation.is_file():
        parser.error(f"--simulation is not a file: {args.simulation}")
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    args.runtime_log.parent.mkdir(parents=True, exist_ok=True)
    run_official_lite_moe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
