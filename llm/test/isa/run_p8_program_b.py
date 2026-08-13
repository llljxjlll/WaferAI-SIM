#!/usr/bin/env python3
"""P8 representative program B: P2P, collectives, sync, and profile matrix."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROFILES = ("baseline", "broadcast_only", "reduce_only", "reduce_broadcast")
MANIFEST_PREFIX = "P8_PROGRAM_B "
PROFILE_PREFIX = "[P7 PROFILE] "
SOURCE_PREFIX = "[P6 MEMORY SOURCE] "
PROBE_PREFIX = "[P6 MEMORY PROBE] "
STATS_PREFIX = "[P6 COLLECTIVE STATS] "
P6_DRAIN_PREFIX = "[P6 COLLECTIVE DRAIN] "
P5_STATS_PREFIX = "[P5 P2P STATS] "
P5_DRAIN_PREFIX = "[P5 P2P DRAIN] "
P5_TIMING_DRAIN_PREFIX = "[P5 P2P TIMING DRAIN] "
COLL_DRAIN_PREFIX = "[COLL_DRAIN] "
TRACE_ACTION = "P6_collective_action"
TRACE_WAVE = "P6_collective_wave_complete"


def fail(message: str) -> None:
    raise RuntimeError(f"[P8 PROGRAM B] FAIL: {message}")


def crc32c(payload: bytes) -> int:
    checksum = 0xFFFFFFFF
    for value in payload:
        checksum ^= value
        for _ in range(8):
            checksum = ((checksum >> 1) ^
                        (0x82F63B78 if checksum & 1 else 0))
    return (~checksum) & 0xFFFFFFFF


def fields(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in text.strip().split():
        if "=" not in item:
            fail(f"non key=value marker field {item!r}")
        key, value = item.split("=", 1)
        if not key or not value or key in result:
            fail(f"empty or duplicate marker field {item!r}")
        result[key] = value
    return result


def marker_rows(output: str, prefix: str) -> list[dict[str, str]]:
    result = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position >= 0:
            payload = line[position + len(prefix):]
            # LOG_* appends a presentation-only `` | <simulation time>``
            # suffix. It is outside the marker contract; keep the payload
            # itself strict key=value while ignoring that logger decoration.
            payload = payload.split(" | ", 1)[0].rstrip()
            # LOG_INFO also terminates the rendered message with one period.
            # It is not part of the final marker value.
            if payload.endswith("."):
                payload = payload[:-1]
            result.append(fields(payload))
    return result


def decimal(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key], 10)
    except (KeyError, ValueError):
        fail(f"marker field {key!r} is missing or non-decimal: {row}")


def affine(length: int, seed: int, multiplier: int) -> bytes:
    return bytes((seed + multiplier * index + (index >> 2)) & 0xFF
                 for index in range(length))


def allgather_source(length: int, rank: int) -> bytes:
    return affine(length, 0x21 + rank * 0x31, 3 + rank * 2)


def allreduce_source(length: int, rank: int) -> bytes:
    result = bytearray(allgather_source(length, rank))
    boundary = [0x7FFFFFFF, 1, 0xFFFFFFFF, 0x80000000][rank]
    result[:4] = boundary.to_bytes(4, "little")
    return bytes(result)


def reduce_sum_i32(inputs: list[bytes]) -> bytes:
    if not inputs or any(len(value) != len(inputs[0]) for value in inputs):
        fail("allreduce input shape mismatch")
    if len(inputs[0]) % 4:
        fail("INT32 allreduce byte count is not divisible by four")
    output = bytearray(len(inputs[0]))
    for offset in range(0, len(output), 4):
        total = sum(int.from_bytes(value[offset:offset + 4], "little")
                    for value in inputs) & 0xFFFFFFFF
        output[offset:offset + 4] = total.to_bytes(4, "little")
    return bytes(output)


def load_oracle(path: Path) -> dict[str, Any]:
    oracle = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "version", "scenario", "fixture_option", "cores", "group_id",
        "region_bytes", "payload_offset", "sentinel", "p2p",
        "allgather", "allreduce", "regions", "expected_counts",
        "profiles", "current_rejections", "repeat_default",
    }
    if set(oracle) != required or oracle["version"] != 1:
        fail("oracle top-level schema/version changed")
    if tuple(oracle["cores"]) != (1, 3, 7, 11):
        fail("oracle must use the canonical non-contiguous four-core group")
    if tuple(oracle["profiles"]) != PROFILES:
        fail("oracle profile matrix is incomplete or non-canonical")
    if int(oracle["repeat_default"]) <= 0:
        fail("oracle repeat_default must be positive")
    if int(oracle["allreduce"]["bytes"]) % 4:
        fail("oracle INT32 allreduce length is not aligned")
    bases = [int(region["base"]) for region in oracle["regions"].values()]
    if len(bases) != len(set(bases)):
        fail("oracle regions overlap by base address")
    return oracle


def expected_manifest(oracle: dict[str, Any]) -> dict[str, str]:
    count = oracle["expected_counts"]
    return {
        "scenario": oracle["scenario"],
        "cores": ",".join(str(value) for value in oracle["cores"]),
        "group_id": str(oracle["group_id"]),
        "p2p_bytes": str(oracle["p2p"]["bytes"]),
        "payload_offset": str(oracle["payload_offset"]),
        "allgather_bytes": str(oracle["allgather"]["bytes"]),
        "allreduce_bytes": str(oracle["allreduce"]["bytes"]),
        "collective_children": str(count["collective_children"]),
        "collective_actions": str(count["collective_actions"]),
        "collective_waves": str(count["collective_waves"]),
        "compute_count": str(count["compute"]),
        "bind_count": str(count["bind"]),
        "alloc_count": str(count["alloc"]),
        "rename_count": str(count["rename"]),
        "free_count": str(count["free"]),
        "p2p_send_count": str(count["p2p_send"]),
        "p2p_recv_count": str(count["p2p_recv"]),
        "collective_send_count": str(count["collective_send"]),
        "collective_recv_count": str(count["collective_recv"]),
        "reduce_compute_count": str(count["reduce_compute"]),
        "wait_count": str(count["wait"]),
        "fence_count": str(count["fence"]),
        "group_sync_count": str(count["group_sync"]),
        "group_sync_sequences": str(count["group_sync_sequences"]),
        "profiles": ",".join(PROFILES),
        "data_base": str(oracle["regions"]["p2p_source"]["base"]),
        "compute_base": str(oracle["regions"]["compute"]["base"]),
        "region_bytes": str(oracle["region_bytes"]),
    }


def fixture(executable: Path, artifact: Path,
            oracle: dict[str, Any]) -> tuple[dict[str, str], str]:
    completed = subprocess.run(
        [str(executable), str(artifact), str(oracle["fixture_option"])],
        cwd=artifact.parent, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=30, check=False)
    if completed.returncode != 0:
        fail(f"fixture returned {completed.returncode}:\n{completed.stdout}")
    rows = marker_rows(completed.stdout, MANIFEST_PREFIX)
    if len(rows) != 1 or rows[0] != expected_manifest(oracle):
        fail(f"fixture manifest mismatch: {rows}")
    if not artifact.is_file() or artifact.stat().st_size == 0:
        fail("fixture emitted no artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    return rows[0], digest


def range_entry(core: int, region: dict[str, Any], payload: bytes,
                oracle: dict[str, Any], initialize: bool,
                seed: int = 0, multiplier: int = 0,
                reduction_boundary_patch: bool = False,
                expected_after: bytes | None = None) -> dict[str, Any]:
    common = {
        "core": core,
        "region": region["name"],
        "region_size_bytes": int(oracle["region_bytes"]),
    }
    if initialize:
        common.update({
            "space": "SRAM",
            "offset_bytes": int(oracle["payload_offset"]),
            "length_bytes": len(payload),
            "prefill_byte": int(oracle["sentinel"]),
            "pattern": {
                "kind": "affine_u8_v1",
                "seed": seed,
                "multiplier": multiplier,
                "quarter_step": 1,
                "reduction_boundary_patch": reduction_boundary_patch,
            },
            "bytes_base64": base64.b64encode(payload).decode("ascii"),
            "checksum": crc32c(payload),
        })
        if expected_after is not None:
            common.update({
                "expected_after_bytes_base64":
                    base64.b64encode(expected_after).decode("ascii"),
                "expected_after_checksum": crc32c(expected_after),
            })
    else:
        common.update({
            "payload_offset_bytes": int(oracle["payload_offset"]),
            "payload_length_bytes": len(payload),
            "expected_bytes_base64": base64.b64encode(payload).decode("ascii"),
            "expected_checksum": crc32c(payload),
            "prefill_byte": int(oracle["sentinel"]),
            "verify_all_bytes_outside_payload": True,
        })
    return common


def sidecar_document(oracle: dict[str, Any],
                     profile: str = "baseline") -> dict[str, Any]:
    if profile not in PROFILES:
        fail(f"unknown profile for sidecar: {profile}")
    dca_reduce = profile in ("reduce_only", "reduce_broadcast")
    cores = [int(value) for value in oracle["cores"]]
    regions = oracle["regions"]
    p2p = affine(int(oracle["p2p"]["bytes"]),
                 int(oracle["p2p"]["seed"]),
                 int(oracle["p2p"]["multiplier"]))
    ag = [allgather_source(int(oracle["allgather"]["bytes"]), rank)
          for rank in range(4)]
    ar = [allreduce_source(int(oracle["allreduce"]["bytes"]), rank)
          for rank in range(4)]
    ag_staging = b"".join(ag)
    ar_staging = b"".join(ar)
    ar_result = reduce_sum_i32(ar)
    initializations = [range_entry(
        int(oracle["p2p"]["source_core"]), regions["p2p_source"],
        p2p, oracle, True, int(oracle["p2p"]["seed"]),
        int(oracle["p2p"]["multiplier"]))]
    for rank, core in enumerate(cores):
        initializations.append(range_entry(
            core, regions["allgather_input"], ag[rank], oracle, True,
            0x21 + rank * 0x31, 3 + rank * 2))
        initializations.append(range_entry(
            core, regions["allreduce_input"], ar[rank], oracle, True,
            0x21 + rank * 0x31, 3 + rank * 2, True))
        if dca_reduce:
            initial = affine(len(ar_staging), 0x5d + rank * 0x17,
                             11 + rank * 2)
            expected_after = bytearray(initial)
            rank_offset = rank * int(oracle["allreduce"]["bytes"])
            expected_after[rank_offset:rank_offset + len(ar[rank])] = ar[rank]
            initializations.append(range_entry(
                core, regions["allreduce_staging"], initial,
                oracle, True, 0x5d + rank * 0x17, 11 + rank * 2,
                expected_after=bytes(expected_after)))
    verifications = [range_entry(
        int(oracle["p2p"]["destination_core"]),
        regions["p2p_destination"], p2p, oracle, False)]
    for core in cores:
        verifications.append(range_entry(
            core, regions["allgather_staging"], ag_staging, oracle, False))
        if not dca_reduce:
            verifications.append(range_entry(
                core, regions["allreduce_staging"], ar_staging,
                oracle, False))
        verifications.append(range_entry(
            core, regions["allreduce_result"], ar_result, oracle, False))
    return {
        "version": 1,
        "format": "p6-p5-compatible-multi-region-v1",
        "scenario": oracle["scenario"],
        "initializations": initializations,
        "verifications": verifications,
    }


def write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def profile_simulation(template: Path, profile: str) -> dict[str, Any]:
    document = json.loads(template.read_text(encoding="utf-8"))
    document.setdefault("noc", {})["collective"] = {
        "enabled": True,
        "profile": profile,
        "dca": {
            "vector_bits": 512,
            "slice_bits": 64,
            "slices_per_tile": 8,
            "header_fifo_depth": 8,
            "operand_fifo_depth": 8,
            "result_fifo_depth": 2,
            "arbitration": "round_robin",
            "value_mode": "integer_exact",
            "latency": {
                "uint8": {"sum": 4, "max": 3},
                "int32": {"sum": 5, "max": 4},
                "int64": {"sum": 6, "max": 5},
                "fp32": {"sum": 7, "max": 6},
                "fp16": {"sum": 5, "max": 4},
                "fp8": {"sum": 4, "max": 3},
            },
            "initiation_interval": {
                "uint8": {"sum": 1, "max": 1},
                "int32": {"sum": 1, "max": 1},
                "int64": {"sum": 2, "max": 2},
                "fp32": {"sum": 1, "max": 1},
                "fp16": {"sum": 1, "max": 1},
                "fp8": {"sum": 1, "max": 1},
            },
        },
    }
    return document


def prepare_assets(root: Path, source_root: Path) -> None:
    (root / "DRAMSys").mkdir(parents=True, exist_ok=True)
    os.symlink(source_root / "font", root / "font", target_is_directory=True)
    os.symlink(source_root / "DRAMSys" / "configs",
               root / "DRAMSys" / "configs", target_is_directory=True)


def expected_probe_rows(sidecar: dict[str, Any], key: str) -> dict[tuple[int, str], int]:
    def checksum(item: dict[str, Any]) -> int:
        if key == "initializations":
            return int(item.get("expected_after_checksum", item["checksum"]))
        return int(item["expected_checksum"])

    return {(int(item["core"]), str(item["region"])): checksum(item)
            for item in sidecar[key]}


def validate_memory(output: str, sidecar: dict[str, Any]) -> dict[str, Any]:
    expected_sources = expected_probe_rows(sidecar, "initializations")
    expected_probes = expected_probe_rows(sidecar, "verifications")
    source_rows = marker_rows(output, SOURCE_PREFIX)
    probe_rows = marker_rows(output, PROBE_PREFIX)
    actual_sources: dict[tuple[int, str], int] = {}
    for row in source_rows:
        key = (decimal(row, "core"), row.get("region", ""))
        if key in actual_sources or row.get("source_initialized") != "1" or \
                row.get("payload_match") != "1" or \
                row.get("sentinels_intact") != "1":
            fail(f"source probe failed or duplicated: {row}")
        actual_sources[key] = decimal(row, "checksum")
    actual_probes: dict[tuple[int, str], int] = {}
    for row in probe_rows:
        key = (decimal(row, "core"), row.get("region", ""))
        if key in actual_probes or row.get("payload_match") != "1" or \
                row.get("sentinels_intact") != "1":
            fail(f"destination probe failed or duplicated: {row}")
        actual_probes[key] = decimal(row, "checksum")
    if actual_sources != expected_sources:
        fail(f"source byte oracle mismatch: {actual_sources} != {expected_sources}")
    if actual_probes != expected_probes:
        fail(f"destination byte oracle mismatch: {actual_probes} != {expected_probes}")
    return {
        "sources": sorted((core, region, checksum)
                          for (core, region), checksum in actual_sources.items()),
        "probes": sorted((core, region, checksum)
                         for (core, region), checksum in actual_probes.items()),
    }


def validate_stats_and_drain(output: str, profile: str,
                             oracle: dict[str, Any]) -> dict[str, Any]:
    count = oracle["expected_counts"]
    stats = marker_rows(output, STATS_PREFIX)
    if len(stats) != 1 or stats[0].get("scenario") != oracle["scenario"]:
        fail("missing or duplicate P6 collective stats")
    for key in ("child_count", "action_count", "wave_count"):
        expected = int(count[{"child_count": "collective_children",
                              "action_count": "collective_actions",
                              "wave_count": "collective_waves"}[key]])
        if decimal(stats[0], key) != expected:
            fail(f"collective stats {key} mismatch")

    p6_drain = marker_rows(output, P6_DRAIN_PREFIX)
    if len(p6_drain) != 1 or p6_drain[0].get("scenario") != oracle["scenario"]:
        fail("missing or duplicate P6 drain")
    for key in ("aggregate", "admission", "barrier", "endpoint", "timing"):
        if decimal(p6_drain[0], key) != 0:
            fail(f"P6 drain {key} is non-zero")

    cores = [int(value) for value in oracle["cores"]]
    p2p_bytes = int(oracle["p2p"]["bytes"])
    ag_bytes = int(oracle["allgather"]["bytes"])
    ar_bytes = int(oracle["allreduce"]["bytes"])
    if profile == "baseline":
        collective_bytes = 3 * (ag_bytes + ar_bytes)
    elif profile == "reduce_only":
        collective_bytes = 3 * ag_bytes
    else:
        collective_bytes = 0
    expected_cores = (set(cores) if collective_bytes else set())
    expected_cores.update((int(oracle["p2p"]["source_core"]),
                           int(oracle["p2p"]["destination_core"])))
    p5_stats = marker_rows(output, P5_STATS_PREFIX)
    by_core = {decimal(row, "core"): row for row in p5_stats}
    if set(by_core) != expected_cores or len(p5_stats) != len(expected_cores):
        fail(f"P5 stats core membership mismatch: {sorted(by_core)}")
    source_extra = int(oracle["p2p"]["source_core"])
    destination_extra = int(oracle["p2p"]["destination_core"])
    for core, row in by_core.items():
        source_bytes = collective_bytes + (p2p_bytes if core == source_extra else 0)
        destination_bytes = collective_bytes + (p2p_bytes if core == destination_extra else 0)
        for key, expected in (
                ("source_read_bytes", source_bytes),
                ("sram_source_read_bytes", source_bytes),
                ("hbm_source_read_bytes", 0),
                ("wire_bytes", source_bytes),
                ("noc_rx_write_bytes", destination_bytes)):
            if decimal(row, key) != expected:
                fail(f"core {core} P5 stats {key} mismatch")
        for key in ("duplicate_requests_suppressed",
                    "request_conflicts_rejected", "request_aborts"):
            if decimal(row, key) != 0:
                fail(f"core {core} P5 stats {key} is non-zero")

    p5_drain = marker_rows(output, P5_DRAIN_PREFIX)
    if len(p5_drain) != len(expected_cores) or \
            {decimal(row, "core") for row in p5_drain} != expected_cores or \
            any(decimal(row, "residual") != 0 for row in p5_drain):
        fail("P5 endpoint drain is incomplete or non-zero")
    timing = marker_rows(output, P5_TIMING_DRAIN_PREFIX)
    if len(timing) != 1 or decimal(timing[0], "residual") != 0:
        fail("P5 timing drain is missing or non-zero")
    collective = marker_rows(output, COLL_DRAIN_PREFIX)
    if len(collective) != 1:
        fail("global collective drain marker missing")
    for key in ("tree_entries", "reduce_nodes", "barriers", "gather",
                "reduce_rx", "endpoints", "dte_tokens", "event"):
        if decimal(collective[0], key) != 0:
            fail(f"global collective drain {key} is non-zero")
    return {
        "collective_stats": stats[0],
        "p6_drain": p6_drain[0],
        "p5_stats": {str(core): by_core[core] for core in sorted(by_core)},
        "p5_drain": sorted(p5_drain, key=lambda row: decimal(row, "core")),
        "p5_timing_drain": timing[0],
        "collective_drain": collective[0],
    }


def trace_threads(events: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[str, str]]:
    modules: dict[int, str] = {}
    names: dict[tuple[int, int], str] = {}
    for event in events:
        if event.get("ph") != "M":
            continue
        if event.get("name") == "process_name":
            modules[int(event["pid"])] = str(event["args"]["name"])
        elif event.get("name") == "thread_name":
            names[(int(event["pid"]), int(event["tid"]))] = str(
                event["args"]["name"])
    return {key: (modules[key[0]], name) for key, name in names.items()
            if key[0] in modules}


def intervals(events: list[dict[str, Any]], module: str,
              thread: str) -> list[tuple[float, float, str]]:
    mapping = trace_threads(events)
    opened: list[tuple[float, str]] = []
    result = []
    for event in events:
        key = (int(event.get("pid", -1)), int(event.get("tid", -1)))
        if mapping.get(key) != (module, thread):
            continue
        phase = event.get("ph")
        stamp = float(event.get("ts", -1))
        if phase == "B":
            opened.append((stamp, str(event.get("name", ""))))
        elif phase == "E":
            if not opened:
                fail(f"trace {module}/{thread} has E without B")
            begin, name = opened.pop(0)
            if stamp < begin:
                fail(f"trace {module}/{thread} has negative duration")
            result.append((begin, stamp, name))
    if opened:
        fail(f"trace {module}/{thread} has unterminated interval")
    return result


def validate_trace(path: Path, oracle: dict[str, Any]) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    events = document["traceEvents"] if isinstance(document, dict) else document
    if not isinstance(events, list):
        fail("traceEvents is not a list")
    actions = [event for event in events if event.get("name") == TRACE_ACTION]
    waves = [event for event in events if event.get("name") == TRACE_WAVE]
    count = oracle["expected_counts"]
    if len(actions) != int(count["collective_actions"]):
        fail("collective action trace count mismatch")
    indices = sorted(int(event.get("args", {}).get("action_index", -1))
                     for event in actions)
    if indices != list(range(int(count["collective_actions"]))):
        fail("collective action trace indices are not canonical")
    if len(waves) != int(count["collective_waves"]):
        fail("collective wave trace count mismatch")

    threads = trace_threads(events)
    lifecycle: dict[int, list[tuple[float, str, str, int, int, str]]] = {}
    detail = re.compile(
        r"^allocation=(\d+) address=(\d+) bytes=(\d+) label=(.*)$")
    for event in events:
        if event.get("ph") not in {"B", "E"}:
            continue
        key = (int(event.get("pid", -1)), int(event.get("tid", -1)))
        module_thread = threads.get(key)
        if module_thread is None or not module_thread[0].startswith("SRAM_region_"):
            continue
        match = detail.match(str(event.get("name", "")))
        if match is None:
            fail(f"malformed SRAM lifecycle trace: {event}")
        try:
            lifecycle_core = int(module_thread[0].removeprefix("SRAM_region_"))
        except ValueError:
            fail(f"malformed SRAM lifecycle module: {module_thread[0]}")
        _, address, size_bytes, label = match.groups()
        lifecycle.setdefault(lifecycle_core, []).append(
            (float(event.get("ts", -1)), module_thread[1],
             str(event["ph"]), int(address), int(size_bytes), label))

    summary: dict[str, Any] = {
        "action_count": len(actions), "wave_count": len(waves), "cores": {}}
    for core in oracle["cores"]:
        module = f"Core {int(core):03d}"
        compute = intervals(events, module, "Comp_prim")
        sync = intervals(events, module, "Group_sync_prim")
        compute = [item for item in compute if item[2] == "Gelu_f"]
        if len(compute) != 2 or any(end <= begin for begin, end, _ in compute):
            fail(f"core {core} does not have two non-empty GELU windows: {compute}")
        if len(sync) != 4 or any(end < begin for begin, end, _ in sync):
            fail(f"core {core} does not have four GROUP_SYNC windows")
        if not (compute[0][1] <= sync[0][0] <= sync[0][1] <= sync[1][0]
                <= sync[1][1] <= sync[2][0] <= sync[2][1] <= compute[1][0]
                <= compute[1][1] <= sync[3][0]):
            fail(f"core {core} compute/GROUP_SYNC ordering is invalid")
        completed_lifecycle = [item for item in lifecycle.get(int(core), [])
                               if item[2] == "E"]
        expected_lifecycle = [
            ("SRAM_region_alloc", int(oracle["regions"]["compute"]["base"]),
             256, "p8b_precompute_label"),
            ("SRAM_region_rename", int(oracle["regions"]["compute"]["base"]),
             256, "p8b_final_compute_label"),
            ("SRAM_region_free", int(oracle["regions"]["compute"]["base"]),
             256, "p8b_final_compute_label"),
        ]
        actual_lifecycle = [(item[1], item[3], item[4], item[5])
                            for item in completed_lifecycle]
        if actual_lifecycle != expected_lifecycle:
            fail(f"core {core} lifecycle mismatch/residual: {actual_lifecycle}")
        if not (compute[0][1] <= completed_lifecycle[1][0] <= compute[1][0]
                <= compute[1][1] <= completed_lifecycle[2][0]):
            fail(f"core {core} rename/rebind/free ordering is invalid")
        summary["cores"][str(core)] = {
            "compute_count": len(compute), "sync_count": len(sync),
            "lifecycle": [item[1] for item in completed_lifecycle],
            "dynamic_label_residual": 0, "ordered": True,
        }
    return summary


def validate_profile(output: str, profile: str,
                     oracle: dict[str, Any]) -> list[dict[str, str]]:
    rows = marker_rows(output, PROFILE_PREFIX)
    if len(rows) != 2:
        fail(f"profile {profile} emitted {len(rows)} decisions, expected two")
    by_op = {row.get("op", "").lower(): row for row in rows}
    if set(by_op) != {"allgather", "allreduce"}:
        fail(f"profile decision op membership mismatch: {set(by_op)}")
    expected = oracle["profiles"][profile]
    for op, row in by_op.items():
        broadcast = expected[f"{op}_broadcast"]
        reduce = ("not_applicable" if op == "allgather"
                  else expected["allreduce_reduce"])
        multicast_trees = int(expected["multicast_trees"])
        dca_trees = (0 if op == "allgather" else int(expected["dca_trees"]))
        endpoint = "true" if op == "allreduce" and reduce == "endpoint" else "false"
        exact = {
            "profile": profile, "group_size": "4", "status": "accepted",
            "broadcast": broadcast, "reduce": reduce,
            "endpoint_reduce_compute": endpoint,
            "multicast_trees": str(multicast_trees),
            "dca_trees": str(dca_trees),
            "requires_multicast": "true" if multicast_trees else "false",
            "requires_dca": "true" if dca_trees else "false",
            "reason": "accepted",
        }
        for key, value in exact.items():
            if row.get(key) != value:
                fail(f"profile {profile}/{op} {key}: {row.get(key)!r} != {value!r}")
        tree_count = max(multicast_trees, dca_trees)
        if decimal(row, "trees") != tree_count:
            fail(f"profile {profile}/{op} immutable tree count mismatch")
        if decimal(row, "batches") < (1 if tree_count else 0) or \
                decimal(row, "conflicts") < 0:
            fail(f"profile {profile}/{op} invalid batch/conflict counts")
    return sorted(rows, key=lambda row: row["op"])


def validate_backend_markers(output: str, profile: str) -> dict[str, int]:
    counts = {
        "multicast_tx": output.count("[P7_MULTICAST_TX]"),
        "multicast_commit": output.count("[P7_MULTICAST_COMMIT]"),
        "tree_batch": output.count("[P7_TREE_BATCH]"),
        "dca_arm": output.count("[COLL_STREAM_ARM]"),
        "dca_tx": output.count("[COLL_STREAM_TX]"),
        "dca_result": output.count("[COLL_STREAM_RESULT]"),
    }
    wants_multicast = profile in {"broadcast_only", "reduce_broadcast"}
    wants_dca = profile in {"reduce_only", "reduce_broadcast"}
    if wants_multicast:
        if counts["multicast_tx"] == 0 or counts["multicast_commit"] == 0 or \
                counts["tree_batch"] == 0:
            fail(f"profile {profile} lacks real multicast/tree-batch markers")
    elif any(counts[key] for key in ("multicast_tx", "multicast_commit")):
        fail(f"profile {profile} unexpectedly used multicast")
    if wants_dca:
        if not all(counts[key] > 0 for key in ("dca_arm", "dca_tx", "dca_result")):
            fail(f"profile {profile} lacks real DCA stream markers")
    elif any(counts[key] for key in ("dca_arm", "dca_tx", "dca_result")):
        fail(f"profile {profile} unexpectedly used DCA")
    return counts


def rejection_reason(output: str) -> str | None:
    matches = re.findall(
        r"ISA-v1 requested collective backend rejected: ([a-z_]+)", output)
    if not matches:
        return None
    if len(set(matches)) != 1:
        fail(f"runtime emitted inconsistent profile rejection reasons: {matches}")
    return matches[0]


def normalized_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_iteration(args: argparse.Namespace, oracle: dict[str, Any],
                  profile: str, iteration: int) -> tuple[str, str]:
    source_root = args.oracle.parents[3]
    with tempfile.TemporaryDirectory(
            prefix=f"p8-program-b-{profile}-{iteration:02d}-",
            dir=args.runtime_root) as temp:
        root = Path(temp)
        prepare_assets(root, source_root)
        case = root / "case"
        case.mkdir()
        artifact = case / "p8_program_b.npup"
        manifest, artifact_hash = fixture(args.program_fixture, artifact, oracle)
        sidecar = sidecar_document(oracle, profile)
        probe = case / "p8_program_b_probe.json"
        simulation = case / "simulation.json"
        write_json(probe, sidecar)
        write_json(simulation, profile_simulation(args.simulation, profile))
        trace = case / "events.json"
        completed = subprocess.run(
            [str(args.npusim), "--program", str(artifact),
             "--p6-memory-probe", str(probe),
             "--hardware-config", str(args.hardware),
             "--simulation-config", str(simulation),
             "--mapping-config", str(args.mapping),
             "--trace-window", "1000000"],
            cwd=case, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
        reject = rejection_reason(completed.stdout)
        if reject is not None:
            expected_reject = oracle["current_rejections"].get(profile)
            if profile == "baseline" or args.require_accelerated:
                fail(f"profile {profile} was rejected: {reject}")
            if completed.returncode == 0 or reject != expected_reject:
                fail(f"profile {profile} rejection contract mismatch: {reject}")
            forbidden = (PROFILE_PREFIX, SOURCE_PREFIX, PROBE_PREFIX,
                         STATS_PREFIX, "End DONE reception",
                         "[P7_MULTICAST_", "[COLL_STREAM_")
            if any(marker in completed.stdout for marker in forbidden):
                fail(f"profile {profile} rejection leaked runtime/fallback markers")
            if trace.exists():
                fail(f"profile {profile} rejection unexpectedly emitted a trace")
            normalized = {
                "status": "rejected", "profile": profile,
                "reason": reject, "artifact_sha256": artifact_hash,
                "manifest": manifest,
            }
            return "rejected", normalized_hash(normalized)

        if completed.returncode != 0:
            fail(f"profile {profile} returned {completed.returncode}:\n{completed.stdout}")
        if "[PROTO_WAIT]" in completed.stdout or \
                "End DONE reception" not in completed.stdout:
            fail(f"profile {profile} did not close the DONE path")
        if not trace.is_file():
            fail(f"profile {profile} emitted no trace")
        profile_rows = validate_profile(completed.stdout, profile, oracle)
        backend = validate_backend_markers(completed.stdout, profile)
        memory = validate_memory(completed.stdout, sidecar)
        stats = validate_stats_and_drain(completed.stdout, profile, oracle)
        trace_summary = validate_trace(trace, oracle)
        normalized = {
            "status": "passed", "profile": profile,
            "artifact_sha256": artifact_hash, "manifest": manifest,
            "profile_decisions": profile_rows, "backend_markers": backend,
            "memory": memory, "stats_and_drain": stats,
            "trace": trace_summary,
        }
        return "passed", normalized_hash(normalized)


def run_matrix(args: argparse.Namespace, oracle: dict[str, Any]) -> None:
    selected = tuple(item.strip() for item in args.profiles.split(",")
                     if item.strip())
    if not selected or len(set(selected)) != len(selected) or \
            any(item not in PROFILES for item in selected):
        fail("--profiles must be a unique comma-separated subset of the four profiles")
    if args.repeat <= 0:
        fail("--repeat must be positive")
    for profile in selected:
        results = [run_iteration(args, oracle, profile, iteration)
                   for iteration in range(args.repeat)]
        statuses = {status for status, _ in results}
        hashes = {digest for _, digest in results}
        if len(statuses) != 1 or len(hashes) != 1:
            fail(f"profile {profile} is unstable across {args.repeat} isolated runs: {results}")
        status = results[0][0]
        digest = results[0][1]
        print(f"[P8 PROGRAM B STABILITY] profile={profile} "
              f"repeat={args.repeat} status={status} sha256={digest}")
    print(f"[P8 PROGRAM B] PASS: profiles={','.join(selected)} "
          f"repeat={args.repeat} isolated_processes={len(selected) * args.repeat}")


def runner_selftest(oracle: dict[str, Any]) -> None:
    manifest = expected_manifest(oracle)
    if fields(" ".join(f"{key}={value}" for key, value in manifest.items())) != manifest:
        fail("manifest parser selftest failed")
    sidecar = sidecar_document(oracle, "baseline")
    if len(sidecar["initializations"]) != 9 or \
            len(sidecar["verifications"]) != 13:
        fail("sidecar must have nine real sources and thirteen byte oracles")
    dca_sidecar = sidecar_document(oracle, "reduce_only")
    if len(dca_sidecar["initializations"]) != 13 or \
            len(dca_sidecar["verifications"]) != 9:
        fail("DCA sidecar must prove four untouched staging regions")
    staging_name = oracle["regions"]["allreduce_staging"]["name"]
    if sum(item["region"] == staging_name
           for item in dca_sidecar["initializations"]) != 4 or \
            any(item["region"] == staging_name
                for item in dca_sidecar["verifications"]):
        fail("DCA staging must be an unchanged source, not a write target")
    for document in (sidecar, dca_sidecar):
        for key in ("initializations", "verifications"):
            for item in document[key]:
                encoded_key = ("bytes_base64" if key == "initializations"
                               else "expected_bytes_base64")
                checksum_key = ("checksum" if key == "initializations"
                                else "expected_checksum")
                payload = base64.b64decode(item[encoded_key], validate=True)
                if not payload or crc32c(payload) != int(item[checksum_key]):
                    fail(f"sidecar payload selftest failed for {item['region']}")
                if key == "initializations" and \
                        item["region"] == staging_name:
                    pattern = item["pattern"]
                    initial = affine(len(payload), int(pattern["seed"]),
                                     int(pattern["multiplier"]))
                    if payload != initial:
                        fail("DCA staging pattern metadata differs from bytes")
                    rank = [int(value) for value in oracle["cores"]].index(
                        int(item["core"]))
                    expected = bytearray(initial)
                    rank_bytes = int(oracle["allreduce"]["bytes"])
                    rank_payload = allreduce_source(rank_bytes, rank)
                    expected[rank * rank_bytes:(rank + 1) * rank_bytes] = \
                        rank_payload
                    mixed = base64.b64decode(
                        item["expected_after_bytes_base64"], validate=True)
                    frozen_crc = [2476804089, 2758363913,
                                  903833748, 3828659505][rank]
                    if mixed != bytes(expected) or \
                            int(item["expected_after_checksum"]) != frozen_crc or \
                            crc32c(mixed) != frozen_crc:
                        fail("DCA staging mixed expected-after bytes/CRC changed")
    stable = {"artifact": "a" * 64, "manifest": manifest,
              "sidecar": sidecar, "trace": {"ordered": True}}
    first = normalized_hash(stable)
    second = normalized_hash(json.loads(json.dumps(stable)))
    if first != second:
        fail("normalized stability digest is non-deterministic")
    stable["trace"]["ordered"] = False
    if normalized_hash(stable) == first:
        fail("normalized stability digest ignored an oracle mutation")
    if reduce_sum_i32([allreduce_source(64, rank) for rank in range(4)])[:4] != \
            (0xFFFFFFFF).to_bytes(4, "little"):
        fail("INT32 wraparound boundary oracle changed")
    print("[P8 PROGRAM B] PASS: artifact/byte/profile/repeat runner selftest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--npusim", type=Path)
    parser.add_argument("--program-fixture", type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=Path("."))
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--require-accelerated", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    args.oracle = args.oracle.resolve()
    oracle = load_oracle(args.oracle)
    if args.repeat is None:
        args.repeat = int(oracle["repeat_default"])
    if args.selftest:
        runner_selftest(oracle)
        return
    required = (args.npusim, args.program_fixture, args.hardware,
                args.simulation, args.mapping)
    if any(value is None for value in required):
        parser.error("runtime mode requires npusim, fixture, hardware, simulation, and mapping")
    for name in ("npusim", "program_fixture", "hardware", "simulation",
                 "mapping", "runtime_root"):
        setattr(args, name, getattr(args, name).resolve())
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    run_matrix(args, oracle)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
