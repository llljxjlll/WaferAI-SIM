#!/usr/bin/env python3
"""N0 backend gate: deterministic two-die TP=2 reduce-scatter.

The fixture's MATMUL records are timing-only.  Functional partial GEMM
outputs are deliberately pre-seeded through the P6 memory-probe sidecar; the
gate then checks raw-byte P2P staging and standalone LOCAL_REDUCE numerics.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any


CORES = (0, 16)
LANES = 16
CHUNK_BYTES = LANES * 2
RANKS = 2
FIXTURE_OPTION = "--frontend-n0-tp2-rs"
MANIFEST_PREFIXES = ("FRONTEND_N0_TP2_RS ",)
SOURCE_PREFIX = "[P6 MEMORY SOURCE] "
PROBE_PREFIX = "[P6 MEMORY PROBE] "
P6_DRAIN_PREFIX = "[P6 COLLECTIVE DRAIN] "
P5_DRAIN_PREFIX = "[P5 P2P DRAIN] "
P5_TIMING_DRAIN_PREFIX = "[P5 P2P TIMING DRAIN] "
COLL_DRAIN_PREFIX = "[COLL_DRAIN] "
GLOBAL_DRAIN_PREFIX = "[DRAIN] "
HOSTLANE_PREFIX = "[HOSTLANE] "
HOSTSIG_PREFIX = "[HOSTSIG] "


def fail(message: str) -> None:
    raise RuntimeError(f"[N0 BACKEND GATE] FAIL: {message}")


def crc32c(payload: bytes) -> int:
    value = 0xFFFFFFFF
    for byte in payload:
        value ^= byte
        for _ in range(8):
            value = ((value >> 1) ^
                     (0x82F63B78 if value & 1 else 0))
    return (~value) & 0xFFFFFFFF


def marker_fields(payload: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in payload.strip().split():
        if "=" not in token:
            fail(f"non key=value marker token {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            fail(f"empty or duplicate marker token {token!r}")
        result[key] = value
    return result


def marker_rows(output: str, prefix: str) -> list[dict[str, str]]:
    rows = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position < 0:
            continue
        payload = line[position + len(prefix):].split(" | ", 1)[0].rstrip()
        if payload.endswith("."):
            payload = payload[:-1]
        rows.append(marker_fields(payload))
    return rows


def decimal(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key], 10)
    except (KeyError, ValueError):
        fail(f"marker field {key!r} is missing or non-decimal: {row}")


def parse_manifest(output: str) -> tuple[str, dict[str, str]]:
    found: list[tuple[str, dict[str, str]]] = []
    for line in output.splitlines():
        for prefix in MANIFEST_PREFIXES:
            if line.startswith(prefix):
                found.append((prefix, marker_fields(line[len(prefix):])))
    if len(found) != 1:
        fail("fixture must emit exactly one canonical N0 manifest; accepted "
             f"prefixes are {MANIFEST_PREFIXES}, observed {len(found)}")
    return found[0]


def half_value(bits: int) -> tuple[str, int, Fraction | None]:
    sign = -1 if bits & 0x8000 else 1
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    if exponent == 0x1F:
        return ("inf" if fraction == 0 else "nan", sign, None)
    if exponent == 0:
        if fraction == 0:
            return "zero", sign, Fraction(0)
        return "finite", sign, sign * Fraction(fraction, 1 << 24)
    significand = 1024 + fraction
    shift = exponent - 25
    magnitude = (Fraction(significand << shift) if shift >= 0 else
                 Fraction(significand, 1 << -shift))
    return "finite", sign, sign * magnitude


def positive_half_value(bits: int) -> Fraction:
    kind, _, value = half_value(bits)
    if kind not in {"zero", "finite"} or value is None:
        fail("internal half search reached a non-finite code")
    return value


def encode_half_rne(value: Fraction, negative: bool) -> int:
    if value < 0:
        fail("internal half encoder requires a non-negative magnitude")
    sign = 0x8000 if negative else 0
    if value == 0:
        return sign
    # Halfway from max finite (65504) to the first overflowing value (65536).
    if value >= 65520:
        return sign | 0x7C00
    low, high = 0, 0x7BFF
    while low < high:
        middle = (low + high + 1) // 2
        if positive_half_value(middle) <= value:
            low = middle
        else:
            high = middle - 1
    lower = low
    lower_value = positive_half_value(lower)
    if lower_value == value:
        return sign | lower
    upper = lower + 1
    # 0x7c00 is infinity, but the RNE overflow decision uses the virtual
    # finite successor 65536; its midpoint with 65504 is 65520.
    upper_value = (Fraction(65536) if upper == 0x7C00 else
                   positive_half_value(upper))
    lower_distance = value - lower_value
    upper_distance = upper_value - value
    if upper_distance < lower_distance or (
            upper_distance == lower_distance and (lower & 1)):
        return sign | upper
    return sign | lower


def fp16_sum_rank_major(inputs: tuple[int, ...]) -> int:
    if not inputs:
        fail("FP16 rank-major oracle needs at least one input")
    decoded = [half_value(value) for value in inputs]
    if any(kind == "nan" for kind, _, _ in decoded):
        return 0x7E00
    infinities = [sign for kind, sign, _ in decoded if kind == "inf"]
    if infinities:
        if 1 in infinities and -1 in infinities:
            return 0x7E00
        return 0x7C00 if infinities[0] > 0 else 0xFC00
    total = sum((value for _, _, value in decoded if value is not None),
                Fraction(0))
    if total == 0:
        # Frozen AddBinary32Rne preserves -0 only for (-0)+(-0); every
        # non-zero cancellation is +0.
        all_zero = all(kind == "zero" for kind, _, _ in decoded)
        negative = all_zero and all(sign < 0 for _, sign, _ in decoded)
        return 0x8000 if negative else 0
    return encode_half_rne(abs(total), total < 0)


def fp16_bytes(values: tuple[int, ...]) -> bytes:
    return b"".join(value.to_bytes(2, "little") for value in values)


def numerical_vectors() -> tuple[tuple[tuple[bytes, bytes], ...],
                                 tuple[bytes, bytes]]:
    # Values are bit patterns, not host floats.  Together they cover normal
    # arithmetic, even/odd half-ULP ties, subnormal/normal boundaries, both
    # signed zeros, cancellation, overflow, infinities, and quiet/signalling
    # NaNs.  Each inner pair is rank 0 then rank 1 for one output chunk.
    chunk0 = (
        (0x3C00, 0x3C00, 0x3C01, 0x0001,
         0x03FF, 0x0000, 0x8000, 0x7C00,
         0x7C00, 0x7E11, 0x7BFF, 0xFBFF,
         0x3555, 0xB555, 0x0400, 0x8400),
        (0x4000, 0x1000, 0x1000, 0x0001,
         0x0001, 0x8000, 0x8000, 0x3C00,
         0xFC00, 0x3C00, 0x7BFF, 0xFBFF,
         0x3555, 0x3555, 0x8001, 0x0001),
    )
    chunk1 = (
        (0xC000, 0x4000, 0x0002, 0x8001,
         0x7C00, 0xFC00, 0x7C00, 0x7D00,
         0x7BFF, 0x7BFF, 0xFBFF, 0x0400,
         0x3555, 0x3C00, 0x3C00, 0xBC00),
        (0x3C00, 0xC000, 0x0001, 0x0001,
         0x7C00, 0xFC00, 0xFC00, 0x0000,
         0x3C00, 0x7BFF, 0x3C00, 0x03FF,
         0x3556, 0xBC00, 0x8000, 0x8000),
    )
    raw_chunks = tuple((fp16_bytes(pair[0]), fp16_bytes(pair[1]))
                       for pair in (chunk0, chunk1))
    reduced = tuple(fp16_bytes(tuple(
        fp16_sum_rank_major((pair[0][lane], pair[1][lane]))
        for lane in range(LANES))) for pair in (chunk0, chunk1))
    return raw_chunks, reduced  # type: ignore[return-value]


def hardware_regions(path: Path) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        regions = document["memory"]["sram"]["regions"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        fail(f"cannot read hardware SRAM regions: {error}")
    result = {entry.get("name"): entry for entry in regions
              if isinstance(entry, dict) and isinstance(entry.get("name"), str)}
    for name in ("double_a", "double_b", "input", "intermediate", "comm"):
        if name not in result:
            fail(f"hardware is missing SRAM region {name!r}")
        try:
            base = int(result[name]["base_bytes"])
            size = int(result[name]["size_bytes"])
        except (KeyError, TypeError, ValueError):
            fail(f"hardware SRAM region {name!r} has invalid base/size")
        if base < 0 or size <= 0:
            fail(f"hardware SRAM region {name!r} has non-positive layout")
    return result


def validate_manifest(manifest: dict[str, str],
                      regions: dict[str, dict[str, Any]]) -> tuple[int, int]:
    required = {
        "scenario", "cores", "region_bytes", "double_buffer_bytes",
        "payload_offset", "sentinel", "double_a_base", "double_b_base",
        "input_base", "intermediate_base", "comm_base", "chunk_bytes",
        "chunk_count", "rank_count", "MATMUL", "P2P_SEND", "P2P_RECV",
        "WAIT", "LOCAL_REDUCE", "core0_owner_chunk",
        "core16_owner_chunk", "core0_staging", "core16_staging",
        "transport_dtype", "reduce_input", "accumulator", "reduce_output",
        "rounding", "order",
    }
    if set(manifest) != required:
        fail("fixture manifest schema changed: "
             f"missing={sorted(required - set(manifest))} "
             f"extra={sorted(set(manifest) - required)}")
    if manifest["scenario"] != "gemm_tp2_dual_owner_rs" or \
            manifest["cores"] != "0,16" or \
            decimal(manifest, "rank_count") != RANKS:
        fail("fixture must encode the canonical rank/core order 0,16")
    expected_numbers = {
        "chunk_bytes": CHUNK_BYTES, "chunk_count": 2,
        "MATMUL": 2, "P2P_SEND": 2, "P2P_RECV": 2,
        "WAIT": 4, "LOCAL_REDUCE": 2,
    }
    for key, expected in expected_numbers.items():
        if decimal(manifest, key) != expected:
            fail(f"fixture manifest {key} is not {expected}")
    offset = decimal(manifest, "payload_offset")
    sentinel = decimal(manifest, "sentinel")
    if offset < 0 or sentinel < 0 or sentinel > 255:
        fail("fixture payload offset/sentinel is outside its canonical range")
    region_bytes = decimal(manifest, "region_bytes")
    double_buffer_bytes = decimal(manifest, "double_buffer_bytes")
    if any(int(regions[name]["size_bytes"]) != region_bytes
           for name in ("input", "intermediate", "comm")) or \
            any(int(regions[name]["size_bytes"]) != double_buffer_bytes
                for name in ("double_a", "double_b")):
        fail("fixture/hardware region-size contract disagrees")
    semantic = {
        "core0_owner_chunk": "0", "core16_owner_chunk": "1",
        "core0_staging": "rank0,rank1", "core16_staging": "rank0,rank1",
        "transport_dtype": "UINT8", "reduce_input": "FP16",
        "accumulator": "FP32", "reduce_output": "FP16",
        "rounding": "RNE", "order": "RANK_MAJOR",
    }
    for key, expected in semantic.items():
        if manifest[key] != expected:
            fail(f"fixture manifest {key} is not {expected}")
    for name in ("double_a", "double_b", "comm"):
        size = int(regions[name]["size_bytes"])
        if offset + RANKS * CHUNK_BYTES > size:
            fail(f"two chunks do not fit hardware region {name!r}")
    for name in ("double_a", "double_b", "input", "intermediate", "comm"):
        base_key = f"{name}_base"
        if decimal(manifest, base_key) != int(
                regions[name]["base_bytes"]):
            fail(f"fixture/hardware {name} base disagreement")
    return offset, sentinel


def initialization(core: int, region: str, region_size: int, offset: int,
                   before: bytes, sentinel: int, seed: int,
                   expected_after: bytes) -> dict[str, Any]:
    return {
        "space": "SRAM", "core": core, "region": region,
        "region_size_bytes": region_size,
        "offset_bytes": offset, "length_bytes": len(before),
        "prefill_byte": sentinel,
        "pattern": {
            "kind": "affine_u8_v1", "seed": seed,
            "multiplier": 3 + 2 * seed % 251, "quarter_step": 1,
            "reduction_boundary_patch": False,
        },
        "bytes_base64": base64.b64encode(before).decode("ascii"),
        "checksum": crc32c(before),
        "expected_after_bytes_base64":
            base64.b64encode(expected_after).decode("ascii"),
        "expected_after_checksum": crc32c(expected_after),
    }


def verification(core: int, region: str, region_size: int, offset: int,
                 expected: bytes, sentinel: int) -> dict[str, Any]:
    return {
        "core": core, "region": region,
        "region_size_bytes": region_size,
        "payload_offset_bytes": offset,
        "payload_length_bytes": len(expected),
        "expected_bytes_base64": base64.b64encode(expected).decode("ascii"),
        "expected_checksum": crc32c(expected),
        "prefill_byte": sentinel,
        "verify_all_bytes_outside_payload": True,
    }


def sidecar_document(manifest: dict[str, str],
                     regions: dict[str, dict[str, Any]],
                     offset: int, sentinel: int) -> dict[str, Any]:
    chunks, reduced = numerical_vectors()
    region_size = {name: int(regions[name]["size_bytes"])
                   for name in ("double_a", "double_b", "comm")}

    # Only the peer-owned chunk is functional data in double_b.  The local
    # owner contribution is pre-positioned in its rank-major comm slot.
    outbound = (
        initialization(0, "double_b", region_size["double_b"],
                       offset + CHUNK_BYTES, chunks[1][0], sentinel, 11,
                       chunks[1][0]),
        initialization(16, "double_b", region_size["double_b"],
                       offset, chunks[0][1], sentinel, 29, chunks[0][1]),
    )
    initial_comm = (
        chunks[0][0] + bytes([sentinel]) * CHUNK_BYTES,
        bytes([sentinel]) * CHUNK_BYTES + chunks[1][1],
    )
    final_comm = (
        chunks[0][0] + chunks[0][1],
        chunks[1][0] + chunks[1][1],
    )
    staging = tuple(initialization(
        core, "comm", region_size["comm"], offset,
        initial_comm[rank], sentinel, 47 + rank * 18, final_comm[rank])
        for rank, core in enumerate(CORES))
    outputs = tuple(verification(
        core, "double_a", region_size["double_a"],
        offset + rank * CHUNK_BYTES, reduced[rank], sentinel)
        for rank, core in enumerate(CORES))
    return {
        "version": 1,
        "format": "p6-p5-compatible-multi-region-v1",
        "scenario": manifest["scenario"],
        "initializations": list(outbound + staging),
        "verifications": list(outputs),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def prepare_runtime_assets(root: Path, hardware: Path) -> None:
    source_root = None
    for candidate in (hardware.parent, *hardware.parents):
        if (candidate / "font").exists() and (candidate / "DRAMSys").exists():
            source_root = candidate
            break
    if source_root is None:
        fail("cannot locate source-root font/ and DRAMSys/ from --hardware")
    os.symlink(source_root / "font", root / "font", target_is_directory=True)
    os.symlink(source_root / "DRAMSys", root / "DRAMSys",
               target_is_directory=True)


def generate_fixture(executable: Path, artifact: Path) -> tuple[str, dict[str, str]]:
    completed = subprocess.run(
        [str(executable), str(artifact), FIXTURE_OPTION], cwd=artifact.parent,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=30, check=False)
    if completed.returncode != 0:
        fail(f"fixture returned {completed.returncode}:\n{completed.stdout}")
    if not artifact.is_file() or artifact.stat().st_size == 0:
        fail("fixture did not emit a non-empty artifact")
    _, manifest = parse_manifest(completed.stdout)
    return completed.stdout, manifest


def expected_probe_rows(sidecar: dict[str, Any], key: str) -> dict[tuple[int, str], int]:
    def checksum(entry: dict[str, Any]) -> int:
        if key == "initializations":
            return int(entry.get("expected_after_checksum", entry["checksum"]))
        return int(entry["expected_checksum"])
    return {(int(entry["core"]), str(entry["region"])): checksum(entry)
            for entry in sidecar[key]}


def validate_memory(output: str, sidecar: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, prefix, initialized in (
            ("initializations", SOURCE_PREFIX, True),
            ("verifications", PROBE_PREFIX, False)):
        expected = expected_probe_rows(sidecar, key)
        actual: dict[tuple[int, str], int] = {}
        for row in marker_rows(output, prefix):
            identity = (decimal(row, "core"), row.get("region", ""))
            if identity in actual or row.get("payload_match") != "1" or \
                    row.get("sentinels_intact") != "1" or \
                    (initialized and row.get("source_initialized") != "1"):
                fail(f"failed or duplicate memory marker: {row}")
            actual[identity] = decimal(row, "checksum")
        if actual != expected:
            fail(f"{key} byte oracle mismatch: {actual} != {expected}")
        summary[key] = sorted((core, region, checksum)
                              for (core, region), checksum in actual.items())
    return summary


def signature_counts(value: str, fields_per_item: int) -> list[tuple[int, ...]]:
    result = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        try:
            values = tuple(int(part, 10) for part in item.split(":"))
        except ValueError:
            fail(f"malformed HOSTSIG item {item!r}")
        if len(values) != fields_per_item:
            fail(f"malformed HOSTSIG arity in {item!r}")
        result.append(values)
    return sorted(result)


def validate_control_and_drains(output: str,
                                scenario: str) -> dict[str, Any]:
    if "[PROTO_WAIT]" in output or "End DONE reception" not in output:
        fail("runtime did not close the DONE path")
    host = marker_rows(output, HOSTLANE_PREFIX)
    signatures = marker_rows(output, HOSTSIG_PREFIX)
    if len(host) != 1 or len(signatures) != 1:
        fail("HOSTLANE/HOSTSIG marker is missing or duplicated")
    if (decimal(host[0], "done_total"), decimal(host[0], "ack_total"),
            decimal(host[0], "mismatch")) != (2, 4, 0):
        fail(f"ACK/DONE totals or lane routing are wrong: {host[0]}")
    done = signature_counts(signatures[0].get("done", ""), 2)
    ack = signature_counts(signatures[0].get("ack", ""), 3)
    ack_by_core = collections.Counter()
    for source, _, count in ack:
        ack_by_core[source] += count
    if done != [(0, 1), (16, 1)] or \
            dict(ack_by_core) != {0: 2, 16: 2}:
        fail(f"ACK/DONE core multiset is wrong: done={done} ack={ack}")

    p6 = marker_rows(output, P6_DRAIN_PREFIX)
    if len(p6) != 1 or p6[0].get("scenario") != scenario:
        fail("P6 worker drain marker is missing or has the wrong scenario")
    for key in ("aggregate", "admission", "barrier", "endpoint", "timing"):
        if decimal(p6[0], key) != 0:
            fail(f"P6 worker drain {key} is non-zero")

    p5 = marker_rows(output, P5_DRAIN_PREFIX)
    if len(p5) != 2 or {decimal(row, "core") for row in p5} != set(CORES) or \
            any(decimal(row, "residual") != 0 for row in p5):
        fail(f"P2P endpoint drains are incomplete or non-zero: {p5}")
    timing = marker_rows(output, P5_TIMING_DRAIN_PREFIX)
    if len(timing) != 1 or decimal(timing[0], "residual") != 0:
        fail("P2P timing sideband did not drain")

    collective = marker_rows(output, COLL_DRAIN_PREFIX)
    if len(collective) != 1:
        fail("global collective drain marker is missing or duplicated")
    for key in ("tree_entries", "reduce_nodes", "barriers", "gather",
                "reduce_rx", "endpoints", "dte_tokens", "event"):
        if decimal(collective[0], key) != 0:
            fail(f"global collective/token drain {key} is non-zero")

    global_rows = marker_rows(output, GLOBAL_DRAIN_PREFIX)
    residuals = {key: decimal(row, key) for row in global_rows
                 for key in ("router_residual", "d2d_link_residual")
                 if key in row}
    if residuals != {"router_residual": 0, "d2d_link_residual": 0}:
        fail(f"router/D2D drains are incomplete or non-zero: {residuals}")
    return {
        "hostlane": host[0], "done": done, "ack": ack,
        "p6": p6[0], "p5": sorted(p5, key=lambda row: decimal(row, "core")),
        "p5_timing": timing[0], "collective": collective[0],
        "global": residuals,
    }


def trace_summary(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        fail(f"cannot read runtime trace: {error}")
    events = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(events, list):
        fail("runtime trace has no traceEvents array")
    modules: dict[int, str] = {}
    threads: dict[tuple[int, int], str] = {}
    for event in events:
        if event.get("ph") != "M":
            continue
        if event.get("name") == "process_name":
            modules[int(event["pid"])] = str(event["args"]["name"])
        elif event.get("name") == "thread_name":
            threads[(int(event["pid"]), int(event["tid"]))] = str(
                event["args"]["name"])

    core_by_pid: dict[int, int] = {}
    for pid, module in modules.items():
        if not module.startswith("Core "):
            continue
        try:
            core = int(module.split()[-1], 16)
        except ValueError:
            continue
        if core in CORES:
            core_by_pid[pid] = core
    if set(core_by_pid.values()) != set(CORES):
        fail(f"trace is missing one of the two active cores: {core_by_pid}")

    relevant = []
    observed: collections.Counter[tuple[int, str, str, str]] = \
        collections.Counter()
    intervals: dict[tuple[int, str], list[float]] = collections.defaultdict(list)
    completed: collections.Counter[tuple[int, str]] = collections.Counter()
    for event in events:
        pid = int(event.get("pid", -1))
        core = core_by_pid.get(pid)
        if core is None or event.get("ph") not in {"B", "E"}:
            continue
        tid = int(event.get("tid", -1))
        thread = threads.get((pid, tid), "")
        name = str(event.get("name", ""))
        stamp = float(event.get("ts", -1))
        relevant.append((core, thread, name, str(event["ph"]), stamp,
                         event.get("args", {})))
        observed[(core, thread, name, str(event["ph"]))] += 1
        expected_thread = {
            "Matmul_f": "Comp_prim",
            "Collective_data_v1_prim": "Comm_prim",
        }.get(name)
        if core == 0 and thread == expected_thread:
            key = (core, name)
            if event["ph"] == "B":
                intervals[key].append(stamp)
            else:
                if not intervals[key]:
                    fail(f"trace {key} ended without beginning")
                begin = intervals[key].pop(0)
                if stamp <= begin:
                    fail(f"trace {key} has a non-positive duration")
                completed[key] += 1
    expected = {
        (0, "Matmul_f"): 1,
        (0, "Collective_data_v1_prim"): 1,
    }
    if dict(completed) != expected or any(intervals.values()):
        fail(f"trace must contain representative core0 MATMUL and "
             f"LOCAL_REDUCE intervals: "
             f"completed={dict(completed)} open={dict(intervals)} "
             f"observed={sorted(observed.items())}")
    encoded = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
    return {
        "relevant_event_count": len(relevant),
        "sequence_sha256": hashlib.sha256(encoded).hexdigest(),
        "coverage": "representative-core",
        "metadata_cores": list(CORES),
        "matmul_timing_only": True,
        "local_reduce_functional": True,
    }


def normalized_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_once(args: argparse.Namespace, root: Path, artifact: Path,
             artifact_digest: str, manifest: dict[str, str],
             sidecar: dict[str, Any], iteration: int) -> str:
    case = root / f"runtime_{iteration}"
    case.mkdir()
    probe = case / "n0_probe.json"
    write_json(probe, sidecar)
    completed = subprocess.run(
        [str(args.npusim), "--program", str(artifact),
         "--p6-memory-probe", str(probe),
         "--hardware-config", str(args.hardware),
         "--simulation-config", str(args.simulation),
         "--mapping-config", str(args.mapping),
         "--trace-window", "1000000"],
        cwd=case, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
    if completed.returncode != 0:
        fail(f"runtime {iteration} returned {completed.returncode}:\n"
             f"{completed.stdout}")
    memory = validate_memory(completed.stdout, sidecar)
    control = validate_control_and_drains(completed.stdout,
                                          manifest["scenario"])
    trace = trace_summary(case / "events.json")
    summary = {
        "artifact_sha256": artifact_digest,
        "manifest": manifest,
        "memory": memory,
        "control_and_drains": control,
        "trace": trace,
    }
    return normalized_hash(summary)


def runner_selftest() -> None:
    chunks, reduced = numerical_vectors()
    if any(len(chunk) != CHUNK_BYTES for pair in chunks for chunk in pair) or \
            any(len(chunk) != CHUNK_BYTES for chunk in reduced):
        fail("numerical vector shape selftest failed")
    checks = {
        (0, 1): 0x3C00,   # exactly half ULP, even destination
        (0, 2): 0x3C02,   # exactly half ULP, odd destination
        (0, 3): 0x0002,   # subnormal addition
        (0, 4): 0x0400,   # subnormal to normal boundary
        (0, 5): 0x0000,   # +0 + -0
        (0, 6): 0x8000,   # -0 + -0
        (0, 8): 0x7E00,   # +Inf + -Inf
        (0, 9): 0x7E00,   # canonicalized NaN
        (0, 10): 0x7C00,  # finite overflow
        (1, 6): 0x7E00,   # opposite infinities
        (1, 7): 0x7E00,   # signalling NaN canonicalization
        (1, 8): 0x7BFF,   # max finite + 1 rounds back to max finite
    }
    for (chunk, lane), expected in checks.items():
        actual = int.from_bytes(reduced[chunk][2 * lane:2 * lane + 2],
                                "little")
        if actual != expected:
            fail(f"FP16 oracle selftest chunk={chunk} lane={lane}: "
                 f"0x{actual:04x} != 0x{expected:04x}")
    if fp16_sum_rank_major((0x7BFF, 0x4800)) != 0x7BFF or \
            fp16_sum_rank_major((0x7BFF, 0x4C00)) != 0x7C00:
        fail("FP16 overflow threshold selftest failed")
    print("[N0 BACKEND GATE SELFTEST] PASS: bit-exact FP16 vectors, "
          "RNE ties, special values, and overflow threshold")


def existing_file(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"not a file: {value}")
    return path


def existing_directory(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"not a directory: {value}")
    return path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npusim", type=existing_file)
    parser.add_argument("--program-fixture", type=existing_file)
    parser.add_argument("--hardware", type=existing_file)
    parser.add_argument("--simulation", type=existing_file)
    parser.add_argument("--mapping", type=existing_file)
    parser.add_argument("--runtime-root", type=existing_directory)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if not args.selftest:
        missing = [name for name in ("npusim", "program_fixture", "hardware",
                                     "simulation", "mapping", "runtime_root")
                   if getattr(args, name) is None]
        if missing:
            parser.error("runtime mode requires " +
                         " ".join("--" + name.replace("_", "-")
                                  for name in missing))
        if args.timeout <= 0:
            parser.error("--timeout must be positive")
    return args


def run(args: argparse.Namespace) -> None:
    regions = hardware_regions(args.hardware)
    with tempfile.TemporaryDirectory(prefix="n0-backend-gate-",
                                     dir=args.runtime_root) as temp:
        root = Path(temp)
        prepare_runtime_assets(root, args.hardware)
        fixture_dirs = (root / "fixture_0", root / "fixture_1")
        for directory in fixture_dirs:
            directory.mkdir()
        artifact0 = fixture_dirs[0] / "n0_tp2_rs.npup"
        artifact1 = fixture_dirs[1] / "n0_tp2_rs.npup"
        stdout0, manifest0 = generate_fixture(args.program_fixture, artifact0)
        stdout1, manifest1 = generate_fixture(args.program_fixture, artifact1)
        bytes0, bytes1 = artifact0.read_bytes(), artifact1.read_bytes()
        digest0 = hashlib.sha256(bytes0).hexdigest()
        digest1 = hashlib.sha256(bytes1).hexdigest()
        if bytes0 != bytes1 or digest0 != digest1 or manifest0 != manifest1 or \
                parse_manifest(stdout0) != parse_manifest(stdout1):
            fail("two isolated fixture generations are not byte/manifest deterministic")
        offset, sentinel = validate_manifest(manifest0, regions)
        sidecar = sidecar_document(manifest0, regions, offset, sentinel)
        hashes = [run_once(args, root, artifact0, digest0, manifest0,
                           sidecar, iteration) for iteration in range(2)]
        if hashes[0] != hashes[1]:
            fail(f"two isolated runtime summaries differ: {hashes}")
        print("[N0 BACKEND NOTE] matmul_data=timing_only "
              "functional_partials=p6_probe_preseed")
        print(f"[N0 BACKEND STABILITY] artifact_bytes={len(bytes0)} "
              f"artifact_sha256={digest0} "
              f"runtime_sha256={hashes[0]} fixture_repeat=2 runtime_repeat=2")
        print("[N0 BACKEND TRACE] coverage=representative-core core=0 "
              "active_core_metadata=0,16")
        print("[FRONTEND N0 BACKEND] PASS: 2-die TP2 dual-owner RS")


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        runner_selftest()
        if not args.selftest:
            run(args)
        return 0
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
