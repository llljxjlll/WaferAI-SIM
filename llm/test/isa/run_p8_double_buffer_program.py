#!/usr/bin/env python3
"""P8 representative program A: blocking LSU plus DTE double buffering."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MANIFEST_PREFIX = "P8_DOUBLE_BUFFER "
PROBE_PREFIX = "[P8 DOUBLE BUFFER PROBE] "
STATS_PREFIX = "[P8 DOUBLE BUFFER STATS] "
INTEGER_MANIFEST_FIELDS = {
    "core", "payload_bytes", "payload_offset", "region_bytes",
    "batch0_hbm", "batch1_hbm", "batch2_hbm", "output_hbm",
    "token1", "token2", "lsu_load_count", "lsu_store_count",
    "dte_issue_count", "dte_wait_count", "bind_count", "matmul_count",
    "fence_count",
}
STATS_FIELDS = {
    "core", "lsu_issued", "lsu_completed", "lsu_hbm_read_bytes",
    "lsu_hbm_write_bytes", "lsu_sram_read_bytes", "lsu_sram_write_bytes",
    "lsu_residual", "dte_issued", "dte_completed", "dte_hbm_read_bytes",
    "dte_hbm_write_bytes", "dte_sram_read_bytes", "dte_sram_write_bytes",
    "dte_residual", "dte_tracker_residual",
}
PROBE_FIELDS = {
    "scenario", "space", "core", "region", "absolute_address_bytes",
    "payload_bytes", "expected_checksum", "checksum", "payload_match",
    "sentinels_intact",
}


def fail(message: str) -> None:
    raise RuntimeError(f"[P8 DOUBLE BUFFER] FAIL: {message}")


def crc32c(payload: bytes) -> int:
    checksum = 0xFFFFFFFF
    for value in payload:
        checksum ^= value
        for _ in range(8):
            checksum = ((checksum >> 1) ^
                        (0x82F63B78 if checksum & 1 else 0))
    return (~checksum) & 0xFFFFFFFF


def pattern(entry: dict[str, Any], payload_bytes: int) -> bytes:
    seed = int(entry["seed"])
    multiplier = int(entry["multiplier"])
    quarter_step = int(entry["quarter_step"])
    result = bytes(
        (seed + multiplier * index + quarter_step * (index >> 2)) & 0xFF
        for index in range(payload_bytes)
    )
    if not result or not any(result):
        fail(f"{entry['name']} pattern is empty or all-zero")
    if crc32c(result) != int(entry["checksum"]):
        fail(f"{entry['name']} independent CRC32C golden changed")
    return result


def load_oracle(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "scenario", "fixture_option", "core", "payload_bytes",
        "payload_offset", "region_bytes", "batch0_hbm", "batch1_hbm",
        "batch2_hbm", "output_hbm", "token1", "token2",
        "initializations", "verifications", "expected_stats",
    }
    if set(value) != required:
        fail(f"oracle fields changed: {set(value)}")
    if set(value["expected_stats"]) != STATS_FIELDS:
        fail("oracle stats schema changed")
    payload_bytes = int(value["payload_bytes"])
    for collection in ("initializations", "verifications"):
        if not value[collection]:
            fail(f"oracle {collection} is empty")
        for entry in value[collection]:
            pattern(entry, payload_bytes)
    return value


def parse_fields(line: str, prefix: str) -> dict[str, str]:
    marker = line.find(prefix)
    if marker < 0:
        fail(f"missing marker {prefix!r}")
    fields: dict[str, str] = {}
    for token in line[marker + len(prefix):].split():
        if "=" not in token:
            fail(f"malformed marker token {token!r}")
        key, value = token.split("=", 1)
        if key in fields:
            fail(f"duplicate marker field {key}")
        fields[key] = value
    return fields


def parse_manifest(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines()
             if line.startswith(MANIFEST_PREFIX)]
    if len(lines) != 1:
        fail(f"fixture emitted {len(lines)} P8 manifests")
    raw = parse_fields(lines[0], MANIFEST_PREFIX)
    return {key: int(value) if key in INTEGER_MANIFEST_FIELDS else value
            for key, value in raw.items()}


def verify_manifest(oracle: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = {
        "scenario": oracle["scenario"], "core": oracle["core"],
        "payload_bytes": oracle["payload_bytes"],
        "payload_offset": oracle["payload_offset"],
        "region_bytes": oracle["region_bytes"],
        "batch0_hbm": oracle["batch0_hbm"],
        "batch1_hbm": oracle["batch1_hbm"],
        "batch2_hbm": oracle["batch2_hbm"],
        "output_hbm": oracle["output_hbm"],
        "token1": oracle["token1"], "token2": oracle["token2"],
        "lsu_load_count": 1, "lsu_store_count": 1,
        "dte_issue_count": 2, "dte_wait_count": 2,
        "bind_count": 2, "matmul_count": 2, "fence_count": 1,
    }
    if manifest != expected:
        fail(f"fixture manifest mismatch: {manifest} != {expected}")


def sidecar_range(oracle: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "space": entry["space"],
        "core": int(oracle["core"]),
        "region": entry["region"],
        "region_size_bytes": int(oracle["region_bytes"]),
        "absolute_address_bytes": int(entry["base"]),
        "payload_offset_bytes": int(oracle["payload_offset"]),
        "payload_length_bytes": int(oracle["payload_bytes"]),
        "prefill_byte": int(entry["sentinel"]),
        "verify_all_bytes_outside_payload": True,
        "pattern": {
            "kind": "affine_u8_v1",
            "seed": int(entry["seed"]),
            "multiplier": int(entry["multiplier"]),
            "quarter_step": int(entry["quarter_step"]),
        },
        "expected_checksum": int(entry["checksum"]),
    }


def write_sidecar(path: Path, oracle: dict[str, Any]) -> None:
    document = {
        "version": 1,
        "format": "p8_double_buffer_probe_v1",
        "scenario": oracle["scenario"],
        "initializations": [sidecar_range(oracle, entry)
                            for entry in oracle["initializations"]],
        "verifications": [sidecar_range(oracle, entry)
                          for entry in oracle["verifications"]],
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def fixture(program_fixture: Path, output: Path,
            oracle: dict[str, Any]) -> None:
    proc = subprocess.run(
        [str(program_fixture), str(output), str(oracle["fixture_option"])],
        cwd=output.parent, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=30,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        fail(f"fixture returned {proc.returncode}")
    if not output.is_file() or output.stat().st_size == 0:
        fail("fixture did not create a non-empty artifact")
    verify_manifest(oracle, parse_manifest(proc.stdout))


def trace_threads(events: list[dict[str, Any]]) -> dict[tuple[int, int], tuple[str, str]]:
    modules: dict[int, str] = {}
    thread_names: dict[tuple[int, int], str] = {}
    for event in events:
        if event.get("ph") != "M":
            continue
        if event.get("name") == "process_name":
            modules[int(event["pid"])] = str(event["args"]["name"])
        elif event.get("name") == "thread_name":
            thread_names[(int(event["pid"]), int(event["tid"]))] = str(
                event["args"]["name"])
    result: dict[tuple[int, int], tuple[str, str]] = {}
    for key, thread in thread_names.items():
        if key[0] in modules:
            result[key] = (modules[key[0]], thread)
    return result


def intervals(events: list[dict[str, Any]], module: str, thread: str,
              name_pattern: re.Pattern[str]) -> list[tuple[float, float, str]]:
    threads = trace_threads(events)
    opened: list[tuple[float, str]] = []
    result: list[tuple[float, float, str]] = []
    for event in events:
        key = (int(event.get("pid", -1)), int(event.get("tid", -1)))
        if threads.get(key) != (module, thread):
            continue
        name = str(event.get("name", ""))
        if not name_pattern.search(name):
            continue
        phase = event.get("ph")
        timestamp = float(event.get("ts", -1))
        if phase == "B":
            opened.append((timestamp, name))
        elif phase == "E":
            if not opened:
                fail(f"trace {module}/{thread} has E without B")
            begin, begin_name = opened.pop(0)
            if timestamp < begin:
                fail(f"trace {module}/{thread} has negative duration")
            result.append((begin, timestamp, begin_name))
    if opened:
        fail(f"trace {module}/{thread} has unterminated B")
    return result


def overlap(lhs: tuple[float, float, str],
            rhs: tuple[float, float, str]) -> float:
    return max(0.0, min(lhs[1], rhs[1]) - max(lhs[0], rhs[0]))


def one_token_interval(events: list[dict[str, Any]], thread: str,
                       token: int) -> tuple[float, float, str]:
    found = intervals(events, f"DTE_async_0_{token}", thread,
                      re.compile(rf"(?:^| )token={token}(?: |$)"))
    if len(found) != 1:
        fail(f"token {token} has {len(found)} {thread} intervals")
    return found[0]


def verify_trace(events: list[dict[str, Any]], oracle: dict[str, Any]) -> None:
    computes = intervals(events, "Core 000", "Comp_prim",
                         re.compile(r"^Matmul_f$"))
    if len(computes) != 2 or any(end <= begin for begin, end, _ in computes):
        fail(f"expected two non-empty MATMUL windows, got {computes}")

    # MATMUL legitimately emits four internal one-byte weight/bias LSU loads.
    # The two public Lsu_mem primitive windows are the program's explicit
    # blocking LSU_LOAD/LSU_STORE operations and therefore remain exact.
    lsu_ops = intervals(events, "Core 000", "Mem_prim",
                        re.compile(r"^Lsu_mem$"))
    if len(lsu_ops) != 2:
        fail(f"expected two explicit blocking LSU operations, got {lsu_ops}")
    if lsu_ops[0][1] > computes[0][0]:
        fail("LSU_LOAD overlaps the following MATMUL instead of blocking")
    if computes[1][1] > lsu_ops[1][0]:
        fail("LSU_STORE began before the second MATMUL completed")

    for index, token_key in enumerate(("token1", "token2")):
        token = int(oracle[token_key])
        one_token_interval(events, "DTE_async_issue", token)
        wait = one_token_interval(events, "DTE_async_wait", token)
        hbm = intervals(events, "DTE_mem_0", "DTE_mem_hbm",
                        re.compile(rf"(?:^| )token={token}(?: |$)"))
        spm = intervals(events, "DTE_mem_0", "DTE_mem_spm",
                        re.compile(rf"(?:^| )token={token}(?: |$)"))
        if len(hbm) != 1 or len(spm) != 1:
            fail(f"token {token} lacks one real HBM and one SPM interval")
        threads = trace_threads(events)
        commit_count = sum(
            1 for event in events
            if event.get("ph") == "E"
            and threads.get((int(event.get("pid", -1)),
                             int(event.get("tid", -1)))) ==
                ("DTE_mem_0", "DTE_mem_commit")
            and re.search(rf"\btoken={token}\b",
                          str(event.get("name", "")))
        )
        if commit_count != 1:
            fail(f"token {token} has {commit_count} memory commits")
        transfer = (min(hbm[0][0], spm[0][0]),
                    max(hbm[0][1], spm[0][1]), f"token={token}")
        if overlap(transfer, computes[index]) <= 0:
            fail(f"token {token} transfer did not overlap MATMUL {index}")
        if max(overlap(hbm[0], computes[index]),
               overlap(spm[0], computes[index])) <= 0:
            fail(f"token {token} has no physical memory stage overlap")
        if wait[1] < transfer[1]:
            fail(f"token {token} WAIT retired before memory completion")


def parse_stats(stdout: str) -> dict[str, int]:
    lines = [line for line in stdout.splitlines() if STATS_PREFIX in line]
    if len(lines) != 1:
        fail(f"runtime emitted {len(lines)} P8 stats markers")
    raw = parse_fields(lines[0], STATS_PREFIX)
    if set(raw) != STATS_FIELDS:
        fail(f"P8 stats fields changed: {set(raw)}")
    return {key: int(value) for key, value in raw.items()}


def parse_probes(stdout: str) -> list[dict[str, str]]:
    result = []
    for line in stdout.splitlines():
        if PROBE_PREFIX not in line:
            continue
        fields = parse_fields(line, PROBE_PREFIX)
        if set(fields) != PROBE_FIELDS:
            fail(f"P8 probe fields changed: {set(fields)}")
        result.append(fields)
    return result


def verify_probes(stdout: str, oracle: dict[str, Any]) -> None:
    actual = parse_probes(stdout)
    expected = oracle["verifications"]
    if len(actual) != len(expected):
        fail(f"runtime emitted {len(actual)} probes, expected {len(expected)}")
    keys = set()
    for fields in actual:
        key = (fields["space"], fields["region"],
               int(fields["absolute_address_bytes"]))
        if key in keys:
            fail(f"duplicate P8 probe range {key}")
        keys.add(key)
        match = [entry for entry in expected
                 if (entry["space"], entry["region"], int(entry["base"])) == key]
        if len(match) != 1:
            fail(f"unexpected P8 probe range {key}")
        entry = match[0]
        if (fields["scenario"] != oracle["scenario"] or
                int(fields["core"]) != int(oracle["core"]) or
                int(fields["payload_bytes"]) != int(oracle["payload_bytes"]) or
                int(fields["expected_checksum"]) != int(entry["checksum"]) or
                int(fields["checksum"]) != int(entry["checksum"]) or
                fields["payload_match"] != "1" or
                fields["sentinels_intact"] != "1"):
            fail(f"P8 probe mismatch for {key}: {fields}")


def prepare_runtime(runtime_root: Path, source_root: Path) -> None:
    font = runtime_root / "font"
    dramsys_configs = runtime_root / "DRAMSys" / "configs"
    font.parent.mkdir(parents=True, exist_ok=True)
    dramsys_configs.parent.mkdir(parents=True, exist_ok=True)
    if not font.exists():
        font.symlink_to(source_root / "font", target_is_directory=True)
    if not dramsys_configs.exists():
        dramsys_configs.symlink_to(source_root / "DRAMSys" / "configs",
                                   target_is_directory=True)


def run_selftest(oracle_path: Path) -> int:
    oracle = load_oracle(oracle_path)
    manifest = "P8_DOUBLE_BUFFER " + " ".join(
        f"{key}={value}" for key, value in {
            "scenario": oracle["scenario"], "core": oracle["core"],
            "payload_bytes": oracle["payload_bytes"],
            "payload_offset": oracle["payload_offset"],
            "region_bytes": oracle["region_bytes"],
            "batch0_hbm": oracle["batch0_hbm"],
            "batch1_hbm": oracle["batch1_hbm"],
            "batch2_hbm": oracle["batch2_hbm"],
            "output_hbm": oracle["output_hbm"],
            "token1": oracle["token1"], "token2": oracle["token2"],
            "lsu_load_count": 1, "lsu_store_count": 1,
            "dte_issue_count": 2, "dte_wait_count": 2,
            "bind_count": 2, "matmul_count": 2, "fence_count": 1,
        }.items())
    verify_manifest(oracle, parse_manifest(manifest))
    with tempfile.TemporaryDirectory(prefix="p8-runner-selftest-") as tmp:
        sidecar = Path(tmp) / "probe.json"
        write_sidecar(sidecar, oracle)
        document = json.loads(sidecar.read_text(encoding="utf-8"))
        if len(document["initializations"]) != 6 or len(document["verifications"]) != 6:
            fail("sidecar range count changed")
    print("[P8 DOUBLE BUFFER] PASS: runner/oracle selftest")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--npusim", type=Path)
    parser.add_argument("--program-fixture", type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return run_selftest(args.oracle)
    required = (args.npusim, args.program_fixture, args.hardware,
                args.simulation, args.mapping, args.runtime_root)
    if any(value is None for value in required):
        fail("runtime mode requires npusim, fixture, platform, and runtime-root")

    oracle = load_oracle(args.oracle)
    runtime_root = args.runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    source_root = args.oracle.resolve().parents[3]
    prepare_runtime(runtime_root, source_root)
    with tempfile.TemporaryDirectory(prefix="p8-double-buffer-",
                                     dir=runtime_root) as temp:
        case_dir = Path(temp)
        artifact = case_dir / "p8_double_buffer.npup"
        probe = case_dir / "p8_double_buffer_probe.json"
        fixture(args.program_fixture.resolve(), artifact, oracle)
        write_sidecar(probe, oracle)
        trace = case_dir / "events.json"
        command = [
            str(args.npusim.resolve()), "--program", str(artifact),
            "--p8-double-buffer-probe", str(probe),
            "--hardware-config", str(args.hardware.resolve()),
            "--simulation-config", str(args.simulation.resolve()),
            "--mapping-config", str(args.mapping.resolve()),
            "--trace-window", "1000000",
        ]
        proc = subprocess.run(command, cwd=case_dir, text=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=120)
        if proc.returncode != 0:
            print(proc.stdout)
            fail(f"npusim returned {proc.returncode}")
        if "[PROTO_WAIT]" in proc.stdout or "End DONE reception" not in proc.stdout:
            print(proc.stdout)
            fail("program did not close the DONE path")
        if not trace.is_file():
            fail("runtime did not emit events.json")
        events = json.loads(trace.read_text(encoding="utf-8"))["traceEvents"]
        verify_trace(events, oracle)
        verify_probes(proc.stdout, oracle)
        stats = parse_stats(proc.stdout)
        if stats != {key: int(value)
                     for key, value in oracle["expected_stats"].items()}:
            fail(f"production stats mismatch: {stats}")

    print("[P8 DOUBLE BUFFER] PASS: exact HBM/SRAM payload and sentinels")
    print("[P8 DOUBLE BUFFER] PASS: blocking LSU and two DTE/compute overlaps")
    print("[P8 DOUBLE BUFFER] PASS: exact token lifecycle and full drain")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(error)
        sys.exit(1)
