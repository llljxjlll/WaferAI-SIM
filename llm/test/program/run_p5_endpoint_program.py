#!/usr/bin/env python3
"""P5 endpoint Program fixture, loader, trace, and drain regression oracle."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Any

MANIFEST_PREFIX = "P5_ENDPOINT "
LARGE_SCENARIO = "cross_die_sram_async_32k"
INTEGER_MANIFEST_FIELDS = {
    "source_core",
    "destination_core",
    "fsm_id",
    "send_token",
    "recv_token",
    "send_length_bytes",
    "recv_length_bytes",
    "source_offset_bytes",
    "destination_offset_bytes",
    "include_recv",
}
FIXTURE_ONLY_FIELDS = {
    "option",
    "wire_fragments",
    "loader_error",
    "payload_pattern_seed",
    "payload_pattern_multiplier",
    "destination_region_size_bytes",
    "destination_sentinel_byte",
}
DRAIN = re.compile(r"\[P5 P2P DRAIN\] core=(\d+) residual=(\d+)")
TIMING_DRAIN = re.compile(r"\[P5 P2P TIMING DRAIN\] residual=(\d+)")
P5_STATS_PREFIX = "[P5 P2P STATS] "
P5_PROBE_PREFIX = "[P5 MEMORY PROBE] "
P5_STATS_FIELDS = {
    "core",
    "source_read_bytes",
    "sram_source_read_bytes",
    "hbm_source_read_bytes",
    "wire_bytes",
    "wire_fragments",
    "noc_rx_write_bytes",
    "tx_local_completions",
    "rx_local_completions",
    "admission_requests_sent",
    "admission_requests_received",
    "admission_acks_sent",
    "admission_acks_received",
    "duplicate_requests_suppressed",
    "request_conflicts_rejected",
    "request_aborts",
    "completion_acks_sent",
    "completion_acks_received",
    "source_checksum",
    "wire_checksum",
    "destination_checksum",
}
P5_PROBE_FIELDS = {
    "scenario",
    "source_initialized",
    "payload_bytes",
    "expected_checksum",
    "destination_checksum",
    "payload_match",
    "sentinels_intact",
}


def fail(message: str) -> None:
    raise RuntimeError(f"[P5 ENDPOINT PROGRAM] FAIL: {message}")


def parse_production_stats(stdout: str) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    for line in stdout.splitlines():
        marker = line.find(P5_STATS_PREFIX)
        if marker < 0:
            continue
        payload = line[marker + len(P5_STATS_PREFIX):]
        fields: dict[str, int] = {}
        for key, value in re.findall(r"([a-z_]+)=(\d+)", payload):
            if key in fields:
                fail(f"duplicate P5 STATS field: {key}")
            fields[key] = int(value)
        if set(fields) != P5_STATS_FIELDS:
            fail(f"P5 STATS fields changed: {set(fields)}")
        core = fields["core"]
        if core in result:
            fail(f"duplicate P5 STATS for core {core}")
        result[core] = fields
    return result


def parse_memory_probe(stdout: str) -> dict[str, str]:
    lines = [line for line in stdout.splitlines()
             if line.startswith(P5_PROBE_PREFIX)]
    if len(lines) != 1:
        fail(f"production emitted {len(lines)} P5 memory-probe markers")
    fields: dict[str, str] = {}
    for token in lines[0][len(P5_PROBE_PREFIX):].split():
        if "=" not in token:
            fail(f"malformed P5 memory-probe token: {token}")
        key, value = token.split("=", 1)
        fields[key] = value
    if set(fields) != P5_PROBE_FIELDS:
        fail(f"P5 memory-probe fields changed: {set(fields)}")
    return fields


def payload_pattern(expected: dict[str, Any]) -> bytes:
    length = int(expected["send_length_bytes"])
    seed = int(expected["payload_pattern_seed"])
    multiplier = int(expected["payload_pattern_multiplier"])
    payload = bytes(
        (seed + multiplier * index + (index >> 2)) & 0xff
        for index in range(length)
    )
    if not payload or not any(payload):
        fail("memory-probe payload pattern must contain non-zero data")
    return payload


def crc32c(payload: bytes) -> int:
    checksum = 0xffffffff
    for value in payload:
        checksum ^= value
        for _ in range(8):
            checksum = ((checksum >> 1) ^
                        (0x82f63b78 if checksum & 1 else 0))
    return (~checksum) & 0xffffffff


def write_memory_probe(case_dir: Path, name: str,
                       expected: dict[str, Any]) -> tuple[Path, int]:
    length = int(expected["send_length_bytes"])
    source_space = str(expected["source_space"])
    source_offset = int(expected["source_offset_bytes"])
    destination_offset = int(expected["destination_offset_bytes"])
    region_size = int(expected["destination_region_size_bytes"])
    if destination_offset + length > region_size:
        fail(f"{name} destination payload exceeds its probe region")
    expected_checksum = crc32c(payload_pattern(expected))
    if expected_checksum == 0:
        fail(f"{name} deterministic non-zero payload has zero CRC32C")

    source: dict[str, Any] = {
        "space": source_space,
        "core": int(expected["source_core"]),
        "offset_bytes": source_offset,
        "length_bytes": length,
        "pattern": {
            "kind": "affine_u8_v1",
            "seed": int(expected["payload_pattern_seed"]),
            "multiplier": int(expected["payload_pattern_multiplier"]),
            "quarter_step": 1,
        },
    }
    if source_space == "SRAM":
        source["region"] = "input"
    elif source_space == "HBM":
        source["absolute_address_bytes"] = source_offset
    else:
        fail(f"{name} has unsupported probe source_space={source_space!r}")

    request = {
        "version": 1,
        "scenario": name,
        "source": source,
        "destination": {
            "core": int(expected["destination_core"]),
            "region": "comm",
            "region_size_bytes": region_size,
            "payload_offset_bytes": destination_offset,
            "payload_length_bytes": length,
            "prefill_byte": int(expected["destination_sentinel_byte"]),
            "verify_all_bytes_outside_payload": True,
        },
        "expected_payload_checksum": expected_checksum,
    }
    path = case_dir / "p5_memory_probe.json"
    path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    return path, expected_checksum


def parse_manifest(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines()
             if line.startswith(MANIFEST_PREFIX)]
    if len(lines) != 1:
        fail(f"fixture emitted {len(lines)} endpoint manifests")
    fields: dict[str, Any] = {}
    for token in lines[0][len(MANIFEST_PREFIX):].split():
        if "=" not in token:
            fail(f"malformed fixture manifest token: {token}")
        key, value = token.split("=", 1)
        fields[key] = int(value) if key in INTEGER_MANIFEST_FIELDS else value
    return fields


def verify_manifest(name: str, expected: dict[str, Any],
                    manifest: dict[str, Any]) -> None:
    for key, value in expected.items():
        if key in FIXTURE_ONLY_FIELDS:
            continue
        actual = manifest.get(key)
        if actual != value:
            fail(f"{name} manifest {key}={actual!r}, expected {value!r}")
    if manifest.get("scenario") != name:
        fail(f"{name} fixture reported scenario={manifest.get('scenario')!r}")


def generate_fixture(fixture: Path, case_dir: Path, name: str,
                     expected: dict[str, Any]) -> Path:
    artifact = case_dir / f"{name}.npup"
    proc = subprocess.run(
        [str(fixture), str(artifact), str(expected["option"])],
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        fail(f"{name} fixture generator returned {proc.returncode}")
    if not artifact.is_file() or artifact.stat().st_size == 0:
        fail(f"{name} fixture was not written")
    verify_manifest(name, expected, parse_manifest(proc.stdout))
    return artifact


def simulator_command(
    args: argparse.Namespace,
    artifact: Path,
    hardware: Path | None = None,
    simulation: Path | None = None,
    memory_probe: Path | None = None,
) -> list[str]:
    selected_hardware = args.hardware if hardware is None else hardware
    selected_simulation = args.simulation if simulation is None else simulation
    command = [
        str(args.npusim),
        "--program",
        str(artifact),
        "--hardware-config",
        str(selected_hardware),
        "--simulation-config",
        str(selected_simulation),
        "--mapping-config",
        str(args.mapping),
        "--trace-window",
        "1000000",
    ]
    if memory_probe is not None:
        command.extend(["--p5-memory-probe", str(memory_probe)])
    return command


def run_loader_negative(args: argparse.Namespace, case_dir: Path, name: str,
                        expected: dict[str, Any], artifact: Path) -> None:
    proc = subprocess.run(
        simulator_command(args, artifact),
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    diagnostic = str(expected["loader_error"])
    if proc.returncode == 0:
        print(proc.stdout)
        fail(f"{name} loader negative was accepted")
    if diagnostic not in proc.stdout:
        print(proc.stdout)
        fail(f"{name} missing loader diagnostic {diagnostic!r}")
    if "Loaded Program Format" in proc.stdout:
        print(proc.stdout)
        fail(f"{name} mutated runtime state before loader rejection")

def trace_category(core: int) -> str:
    return f"Core {core:03d}"



def matching_events(events: list[dict[str, Any]], core: int, stage: str,
                    phase: str) -> list[dict[str, Any]]:
    category = trace_category(core)
    return [
        event for event in events
        if event.get("cat") == category
        and event.get("name") == stage
        and event.get("ph") == phase
    ]


def require_event(events: list[dict[str, Any]], core: int, stage: str,
                  phase: str, expected_args: dict[str, int]) -> dict[str, Any]:
    matched = matching_events(events, core, stage, phase)
    if len(matched) != 1:
        fail(f"Core {core} {stage}/{phase} count={len(matched)}, expected 1")
    args = matched[0].get("args", {})
    for key, value in expected_args.items():
        if int(args.get(key, -1)) != value:
            fail(
                f"Core {core} {stage}/{phase} {key}={args.get(key)!r}, "
                f"expected {value}"
            )
    return matched[0]


def verify_runtime_trace(
    name: str, expected: dict[str, Any], stdout: str,
    events: list[dict[str, Any]], require_streaming_timing: bool,
    expected_payload_checksum: int,
) -> dict[str, int]:
    source = int(expected["source_core"])
    destination = int(expected["destination_core"])
    fsm_id = int(expected["fsm_id"])
    length = int(expected["send_length_bytes"])
    fragments = int(expected["wire_fragments"])
    source_stage = (
        "P2P_source_read_HBM"
        if expected["source_space"] == "HBM"
        else "P2P_source_read_SRAM"
    )

    require_event(
        events, source, "P2P_admission_request", "i",
        {"fsm_id": fsm_id},
    )
    require_event(
        events, destination, "P2P_admission_accept", "i",
        {"fsm_id": fsm_id},
    )
    require_event(
        events, destination, "P2P_admission_ack", "i",
        {"fsm_id": fsm_id},
    )
    require_event(
        events, source, "P2P_admission_ack", "i",
        {"fsm_id": fsm_id},
    )
    source_event = require_event(
        events, source, source_stage, "E",
        {"fsm_id": fsm_id, "bytes": length, "fragments": 0},
    )
    wire_event = require_event(
        events, source, "P2P_wire", "E",
        {"fsm_id": fsm_id, "bytes": length, "fragments": fragments},
    )
    destination_event = require_event(
        events, destination, "P2P_noc_rx_write", "E",
        {"fsm_id": fsm_id, "bytes": length, "fragments": 0},
    )
    require_event(
        events, source, "P2P_tx_local_complete", "i",
        {"fsm_id": fsm_id, "bytes": length, "fragments": fragments},
    )
    require_event(
        events, destination, "P2P_rx_local_complete", "i",
        {"fsm_id": fsm_id, "bytes": length, "fragments": 0},
    )
    require_event(
        events, destination, "P2P_completion_ack", "i",
        {"fsm_id": fsm_id, "bytes": length},
    )
    require_event(
        events, source, "P2P_completion_ack", "i",
        {"fsm_id": fsm_id, "bytes": 0},
    )

    checksums = {
        int(source_event.get("args", {}).get("checksum", -1)),
        int(wire_event.get("args", {}).get("checksum", -1)),
        int(destination_event.get("args", {}).get("checksum", -1)),
    }
    if len(checksums) != 1 or -1 in checksums:
        fail(f"{name} source/wire/kNocRx checksums differ: {checksums}")

    checksum = next(iter(checksums))
    if checksum != expected_payload_checksum:
        fail(
            f"{name} production checksum={checksum}, deterministic "
            f"payload checksum={expected_payload_checksum}"
        )
    probe = parse_memory_probe(stdout)
    expected_probe = {
        "scenario": name,
        "source_initialized": "1",
        "payload_bytes": str(length),
        "expected_checksum": str(expected_payload_checksum),
        "destination_checksum": str(expected_payload_checksum),
        "payload_match": "1",
        "sentinels_intact": "1",
    }
    if probe != expected_probe:
        fail(f"{name} memory-probe result={probe}, expected={expected_probe}")

    stats = parse_production_stats(stdout)
    if set(stats) != {source, destination}:
        fail(
            f"{name} P5 STATS cores={set(stats)}, "
            f"expected={{{source}, {destination}}}"
        )
    source_expected = {
        "source_read_bytes": length,
        "sram_source_read_bytes":
            length if expected["source_space"] == "SRAM" else 0,
        "hbm_source_read_bytes":
            length if expected["source_space"] == "HBM" else 0,
        "wire_bytes": length,
        "wire_fragments": fragments,
        "noc_rx_write_bytes": 0,
        "tx_local_completions": 1,
        "rx_local_completions": 0,
        "admission_requests_sent": 1,
        "admission_requests_received": 0,
        "admission_acks_sent": 0,
        "admission_acks_received": 1,
        "duplicate_requests_suppressed": 0,
        "request_conflicts_rejected": 0,
        "request_aborts": 0,
        "completion_acks_sent": 0,
        "completion_acks_received": 1,
        "source_checksum": checksum,
        "wire_checksum": checksum,
        "destination_checksum": 0,
    }
    destination_expected = {
        "source_read_bytes": 0,
        "sram_source_read_bytes": 0,
        "hbm_source_read_bytes": 0,
        "wire_bytes": 0,
        "wire_fragments": 0,
        "noc_rx_write_bytes": length,
        "tx_local_completions": 0,
        "rx_local_completions": 1,
        "admission_requests_sent": 0,
        "admission_requests_received": 1,
        "admission_acks_sent": 1,
        "admission_acks_received": 0,
        "duplicate_requests_suppressed": 0,
        "request_conflicts_rejected": 0,
        "request_aborts": 0,
        "completion_acks_sent": 1,
        "completion_acks_received": 0,
        "source_checksum": 0,
        "wire_checksum": 0,
        "destination_checksum": checksum,
    }
    for core, expected_stats in (
        (source, source_expected),
        (destination, destination_expected),
    ):
        for field, value in expected_stats.items():
            if stats[core][field] != value:
                fail(
                    f"{name} core={core} STATS {field}="
                    f"{stats[core][field]}, expected={value}"
                )

    if require_streaming_timing:
        timing_event = require_event(
            events, destination, "P2P_stream_timing", "i",
            {"fsm_id": fsm_id},
        )
        timing_args = timing_event.get("args", {})
        required_timing = {
            "source_first_ns",
            "source_done_ns",
            "network_tail_cycles",
            "timing_residual",
        }
        if not required_timing.issubset(timing_args):
            fail(
                f"{name} streaming timing fields are incomplete: "
                f"{timing_args}"
            )
        source_first = int(timing_args["source_first_ns"])
        source_done = int(timing_args["source_done_ns"])
        network_tail = int(timing_args["network_tail_cycles"])
        timing_residual = int(timing_args["timing_residual"])
        if source_first > source_done:
            fail(
                f"{name} streaming source_first_ns={source_first} exceeds "
                f"source_done_ns={source_done}"
            )
        if network_tail < 0:
            fail(f"{name} streaming network tail is negative")
        if timing_residual != 0:
            fail(
                f"{name} shared timing sideband residual="
                f"{timing_residual}"
            )

    residuals = {int(core): int(value) for core, value in DRAIN.findall(stdout)}
    expected_cores = {source, destination}
    if set(residuals) != expected_cores:
        fail(
            f"{name} P5 drain cores={set(residuals)}, "
            f"expected={expected_cores}; C3b must emit one drain marker per core"
        )
    if any(value != 0 for value in residuals.values()):
        fail(f"{name} P5 endpoint residual is non-zero: {residuals}")
    timing_drains = [int(value) for value in TIMING_DRAIN.findall(stdout)]
    if timing_drains != [0]:
        fail(
            f"{name} global P5 timing drain markers={timing_drains}, "
            "expected exactly one residual=0 marker"
        )
    return {
        "checksum": next(iter(checksums)),
        "source_read_bytes": length,
        "wire_bytes": length,
        "wire_fragments": fragments,
        "noc_rx_write_bytes": length,
        "completion_events": 4,
        "residual": sum(residuals.values()),
    }


def run_positive(
    args: argparse.Namespace,
    case_dir: Path,
    name: str,
    expected: dict[str, Any],
    artifact: Path,
    hardware: Path,
    simulation: Path,
    require_streaming_timing: bool,
) -> dict[str, int]:
    trace = case_dir / "events.json"
    trace.unlink(missing_ok=True)
    memory_probe, expected_checksum = write_memory_probe(
        case_dir, name, expected
    )
    proc = subprocess.run(
        simulator_command(
            args, artifact, hardware, simulation, memory_probe
        ),
        cwd=case_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )


    if proc.returncode != 0:
        print(proc.stdout)
        fail(f"{name} npusim returned {proc.returncode}")
    if "[PROTO_WAIT]" in proc.stdout or "End DONE reception" not in proc.stdout:
        print(proc.stdout)
        fail(f"{name} did not close the DONE path")
    if not trace.is_file():
        fail(f"{name} did not produce events.json")
    events = json.loads(trace.read_text(encoding="utf-8"))["traceEvents"]
    return verify_runtime_trace(
        name, expected, proc.stdout, events, require_streaming_timing,
        expected_checksum,
    )


def run_selftest() -> int:
    synthetic = (
        "wrote Program Format 1.0 fixture: 1 bytes\n"
        "P5_ENDPOINT scenario=cross_die_sram_async_4k "
        "source_core=16 destination_core=1 source_space=SRAM "
        "completion=ASYNC fsm_id=65537 send_token=7 recv_token=8 "
        "send_length_bytes=4096 recv_length_bytes=4096 "
        "source=input_region source_offset_bytes=0 "
        "destination=comm_region destination_offset_bytes=0 "
        "include_recv=1\n"
    )
    manifest = parse_manifest(synthetic)
    if manifest["source_core"] != 16 or manifest["fsm_id"] != 65537:
        fail(f"manifest integer parsing changed: {manifest}")
    categories = {
        0: "Core 000",
        16: "Core 016",
        255: "Core 255",
    }
    for core, expected in categories.items():
        if trace_category(core) != expected:
            fail(
                f"trace category core={core} is {trace_category(core)!r}, "
                f"expected {expected!r}"
            )
    logger_stats = (
        "[INFO][SYSTEM] " + P5_STATS_PREFIX
        + " ".join(f"{field}=0" for field in sorted(P5_STATS_FIELDS))
        + ". | 1 ns"
    )
    if set(parse_production_stats(logger_stats)) != {0}:
        fail("logger-prefixed P5 STATS parsing changed")
    probe_case = {
        "source_core": 0,
        "destination_core": 1,
        "source_space": "SRAM",
        "send_length_bytes": 9,
        "source_offset_bytes": 17,
        "destination_offset_bytes": 19,
        "destination_region_size_bytes": 64,
        "destination_sentinel_byte": 0xa5,
        "payload_pattern_seed": 17,
        "payload_pattern_multiplier": 37,
    }
    if crc32c(payload_pattern(probe_case)) != 0x92211d7b:
        fail("memory-probe affine pattern or CRC32C contract changed")
    fragment_cases = {
        1: 1, 15: 1, 16: 1, 17: 2, 127: 8, 128: 8,
        129: 9, 255: 16, 256: 16, 257: 17, 32768: 2048,
    }
    for length, fragments in fragment_cases.items():
        actual = (length + 15) // 16
        if actual != fragments:
            fail(
                f"fragment boundary length={length} yielded {actual}, "
                f"expected {fragments}"
            )
    print("[P5 ENDPOINT PROGRAM] PASS: manifest/category selftest")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return run_selftest()
    parser = argparse.ArgumentParser()
    parser.add_argument("--npusim", required=True, type=Path)
    parser.add_argument("--program-fixture", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--large-hardware", type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--streaming-simulation", type=Path)
    parser.add_argument("--bounded-hardware", type=Path)
    parser.add_argument("--behavioral-hardware", type=Path)
    parser.add_argument("--behavioral-simulation", type=Path)
    parser.add_argument(
        "--variant",
        choices=("base", "streaming", "bounded", "behavioral", "all"),
        default="all",
    )
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--negative-only", action="store_true")
    args = parser.parse_args()

    args.npusim = args.npusim.resolve()
    args.program_fixture = args.program_fixture.resolve()
    args.oracle = args.oracle.resolve()
    args.hardware = args.hardware.resolve()
    if args.large_hardware is not None:
        args.large_hardware = args.large_hardware.resolve()
    args.simulation = args.simulation.resolve()
    if args.streaming_simulation is not None:
        args.streaming_simulation = args.streaming_simulation.resolve()
    if args.bounded_hardware is not None:
        args.bounded_hardware = args.bounded_hardware.resolve()
    if args.behavioral_hardware is not None:
        args.behavioral_hardware = args.behavioral_hardware.resolve()
    if args.behavioral_simulation is not None:
        args.behavioral_simulation = args.behavioral_simulation.resolve()
    args.mapping = args.mapping.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    scenarios = json.loads(args.oracle.read_text(encoding="utf-8"))

    source_root = args.hardware.parents[3]
    with tempfile.TemporaryDirectory(
        prefix="p5-endpoint-program-", dir=args.runtime_root
    ) as temp_name:
        root = Path(temp_name)
        os.symlink(source_root / "font", root / "font",
                   target_is_directory=True)
        os.symlink(source_root / "DRAMSys", root / "DRAMSys",
                   target_is_directory=True)

        if args.negative_only:
            selected = {
                name: expected for name, expected in scenarios.items()
                if "loader_error" in expected
            }
            for name, expected in selected.items():
                case_dir = root / name
                case_dir.mkdir()
                artifact = generate_fixture(
                    args.program_fixture, case_dir, name, expected
                )
                run_loader_negative(
                    args, case_dir, name, expected, artifact
                )
        else:
            positive = {
                name: expected for name, expected in scenarios.items()
                if "loader_error" not in expected
            }
            same_die = "same_die_sram_sync_129"
            cross_die = "cross_die_sram_async_4k"
            cross_die_sync = "cross_die_sram_sync_128"
            sram_names = (same_die, cross_die)
            streaming_names = (same_die, cross_die_sync)

            def require_config(attribute: str, option: str) -> Path:
                value = getattr(args, attribute)
                if value is None:
                    fail(f"{option} is required for variant={args.variant}")
                return value

            def execute(
                name: str,
                variant: str,
                hardware: Path,
                simulation: Path,
                streaming: bool = False,
            ) -> dict[str, int]:
                expected = positive[name]
                case_dir = root / f"{name}_{variant}"
                case_dir.mkdir()
                artifact = generate_fixture(
                    args.program_fixture, case_dir, name, expected
                )
                summary = run_positive(
                    args, case_dir, name, expected, artifact,
                    hardware, simulation, streaming,
                )
                print(
                    f"[P5 ENDPOINT VARIANT] scenario={name} "
                    f"variant={variant} checksum={summary['checksum']} "
                    f"fragments={summary['wire_fragments']} residual=0"
                )
                return summary

            if args.variant in {"base", "all"}:
                baseline_names = tuple(positive)
            elif args.variant == "streaming":
                baseline_names = streaming_names
            elif args.variant == "behavioral":
                baseline_names = sram_names
            else:
                baseline_names = (cross_die,)

            baseline: dict[str, dict[str, int]] = {}
            for name in baseline_names:
                hardware = args.hardware
                if name == LARGE_SCENARIO:
                    hardware = require_config(
                        "large_hardware", "--large-hardware"
                    )
                result = execute(
                    name, "cycle_physical", hardware, args.simulation
                )
                baseline[name] = result

            def compare_variant(
                name: str, variant: str, result: dict[str, int]
            ) -> None:
                if name not in baseline:
                    fail(f"{name}/{variant} has no physical comparison")
                if result != baseline[name]:
                    fail(
                        f"{name}/{variant} differs from physical: "
                        f"variant={result}, physical={baseline[name]}"
                    )

            if args.variant in {"streaming", "all"}:
                streaming_simulation = require_config(
                    "streaming_simulation", "--streaming-simulation"
                )
                for name in streaming_names:
                    compare_variant(
                        name,
                        "streaming",
                        execute(
                            name, "streaming", args.hardware,
                            streaming_simulation, True,
                        ),
                    )

            if args.variant in {"bounded", "all"}:
                bounded_hardware = require_config(
                    "bounded_hardware", "--bounded-hardware"
                )
                compare_variant(
                    cross_die,
                    "bounded_saf",
                    execute(
                        cross_die, "bounded_saf", bounded_hardware,
                        args.simulation,
                    ),
                )

            if args.variant in {"behavioral", "all"}:
                behavioral_hardware = require_config(
                    "behavioral_hardware", "--behavioral-hardware"
                )
                behavioral_simulation = require_config(
                    "behavioral_simulation", "--behavioral-simulation"
                )
                for name in sram_names:
                    compare_variant(
                        name,
                        "behavioral",
                        execute(
                            name, "behavioral", behavioral_hardware,
                            behavioral_simulation,
                        ),
                    )

    mode = "loader negatives" if args.negative_only else "runtime positives"
    print(f"[P5 ENDPOINT PROGRAM] PASS: {mode} matched exact fixture oracles")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError,
            json.JSONDecodeError) as error:
        print(error)
        raise SystemExit(1)
