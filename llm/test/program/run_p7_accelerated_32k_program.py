#!/usr/bin/env python3
"""P7 production acceleration gate for the two canonical 32 KiB P6 cases."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[3]
PROGRAM_DIR = SOURCE_ROOT / "llm" / "test" / "program"
ISA_DIR = SOURCE_ROOT / "llm" / "test" / "isa"
PROFILES = ("baseline", "broadcast_only", "reduce_only", "reduce_broadcast")
PROFILE = "[P7 PROFILE] "
TREE_BATCH = "[P7_TREE_BATCH] "
TREE_DRAIN = "[P7_TREE_DRAIN] "
MULTICAST_TX = "[P7_MULTICAST_TX] "
MULTICAST_COMMIT = "[P7_MULTICAST_COMMIT] "
DCA_ARM = "[COLL_STREAM_ARM] "
DCA_TX = "[COLL_STREAM_TX] "
DCA_RESULT = "[COLL_STREAM_RESULT] "


def import_file(name: str, path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p6 = import_file("p7_p6", PROGRAM_DIR / "run_p6_collective_program.py")
p8 = import_file("p7_p8", ISA_DIR / "run_p8_program_b.py")


def fail(message: str) -> None:
    raise RuntimeError(f"[P7 ACCELERATED 32K] FAIL: {message}")


def rows(output: str, prefix: str) -> list[dict[str, str]]:
    return p8.marker_rows(output, prefix)


def integer(row: dict[str, str], key: str) -> int:
    return p8.decimal(row, key)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def load_oracle(path: pathlib.Path) -> dict[str, Any]:
    oracle = json.loads(path.read_text(encoding="utf-8"))
    if set(oracle) != {"version", "scenario", "cores", "profiles",
                      "max_trees_per_batch", "cases"} or oracle["version"] != 1:
        fail("oracle top-level schema/version changed")
    if tuple(oracle["cores"]) != tuple(p6.CORES) or \
            tuple(oracle["profiles"]) != PROFILES or \
            oracle["max_trees_per_batch"] != 1:
        fail("oracle core/profile/K=1 contract changed")
    expected = {
        "--p6-broadcast-gather-n4-l32768-u8-none": "ALLGATHER",
        "--p6-broadcast-reduce-n4-l32768-i32-sum": "ALLREDUCE",
    }
    actual = {case.get("option"): case.get("op") for case in oracle["cases"]}
    if actual != expected or len(oracle["cases"]) != 2:
        fail("oracle must contain exactly the canonical 32 KiB pair")
    return oracle


def resolve_specs(oracle: dict[str, Any], p6_oracle: pathlib.Path) \
        -> list[tuple[dict[str, Any], dict[str, Any]]]:
    # The legacy P6 oracle is an intentionally sampled 37-case matrix, not
    # the complete fixture option domain.  Validate it, then describe the two
    # explicitly requested supported options exactly.  The emitted manifest
    # is still checked by the shared P6 validator before every runtime.
    p6.load_oracle(p6_oracle)
    exact = {
        "ALLGATHER": {"tx": "broadcast", "rx": "gather",
                      "dtype": "UINT8", "reduce_op": "NONE"},
        "ALLREDUCE": {"tx": "broadcast", "rx": "reduce",
                      "dtype": "INT32", "reduce_op": "SUM"},
    }
    result = []
    for case in oracle["cases"]:
        spec = {**exact[case["op"]], "op": case["op"], "n": 4,
                "length_bytes": 32768, "option": case["option"]}
        result.append((case, spec))
    return result


def emit_fixture(executable: pathlib.Path, case: dict[str, Any],
                 spec: dict[str, Any], artifact: pathlib.Path) \
        -> tuple[dict[str, str], str]:
    manifest, sources, expected = p6.run_fixture(
        executable, case["option"], artifact)
    p6.validate_manifest(spec, manifest, sources, expected)
    return manifest, hashlib.sha256(artifact.read_bytes()).hexdigest()


def dca_probe(document: dict[str, Any], spec: dict[str, Any],
              manifest: dict[str, str]) -> dict[str, Any]:
    """Change reduction staging into four canonical local-copy sources."""
    result = copy.deepcopy(document)
    result["verifications"] = [item for item in result["verifications"]
                               if item["region"] != "p6_staging"]
    length = int(spec["length_bytes"])
    total = length * int(spec["n"])
    for rank, core in enumerate(p6.CORES):
        seed = 0x5D + rank * 0x17
        multiplier = 11 + rank * 2
        initial = p8.affine(total, seed, multiplier)
        expected_after = bytearray(initial)
        begin = rank * length
        expected_after[begin:begin + length] = p6.source_segment(
            spec, rank, rank)
        result["initializations"].append({
            "space": "SRAM", "core": core, "region": "p6_staging",
            "region_size_bytes": p6.parse_int(manifest, "staging_size"),
            "offset_bytes": p6.parse_int(manifest, "payload_offset"),
            "length_bytes": total,
            "prefill_byte": p6.parse_int(manifest, "sentinel"),
            "pattern": {"kind": "affine_u8_v1", "seed": seed,
                        "multiplier": multiplier, "quarter_step": 1,
                        "reduction_boundary_patch": False},
            "bytes_base64": base64.b64encode(initial).decode("ascii"),
            "checksum": p6.crc32c(initial),
            "expected_after_bytes_base64":
                base64.b64encode(expected_after).decode("ascii"),
            "expected_after_checksum": p6.crc32c(expected_after),
        })
    return result


def simulation(template: pathlib.Path, profile: str,
               max_trees_per_batch: int) -> dict[str, Any]:
    document = p8.profile_simulation(template, profile)
    document["noc"]["collective"]["max_trees_per_batch"] = \
        max_trees_per_batch
    return document


def validate_memory(output: str, probe: dict[str, Any]) -> dict[str, Any]:
    expected_sources = {
        (int(item["core"]), str(item["region"])):
        int(item.get("expected_after_checksum", item["checksum"]))
        for item in probe["initializations"]}
    actual_sources: dict[tuple[int, str], int] = {}
    for row in rows(output, p8.SOURCE_PREFIX):
        key = (integer(row, "core"), row.get("region", ""))
        if key in actual_sources or row.get("source_initialized") != "1" or \
                row.get("payload_match") != "1" or \
                row.get("sentinels_intact") != "1":
            fail(f"source/expected-after probe failed: {row}")
        actual_sources[key] = integer(row, "checksum")
    if actual_sources != expected_sources:
        fail(f"source byte oracle mismatch: {actual_sources} != {expected_sources}")
    expected_targets = {
        (int(item["core"]), str(item["region"])):
        int(item["expected_checksum"]) for item in probe["verifications"]}
    actual_targets: dict[tuple[int, str], int] = {}
    for row in rows(output, p8.PROBE_PREFIX):
        key = (integer(row, "core"), row.get("region", ""))
        if key in actual_targets or row.get("payload_match") != "1" or \
                row.get("sentinels_intact") != "1":
            fail(f"destination byte probe failed: {row}")
        actual_targets[key] = integer(row, "checksum")
    if actual_targets != expected_targets:
        fail("destination byte oracle membership/checksum mismatch")
    return {"sources": sorted((core, region, crc) for (core, region), crc
                              in actual_sources.items()),
            "targets": sorted((core, region, crc) for (core, region), crc
                              in actual_targets.items())}


def validate_profile(output: str, profile: str,
                     case: dict[str, Any]) -> dict[str, str]:
    markers = rows(output, PROFILE)
    if len(markers) != 1:
        fail(f"{profile}/{case['op']} emitted {len(markers)} profile decisions")
    row = markers[0]
    multicast = profile in {"broadcast_only", "reduce_broadcast"}
    dca = case["op"] == "ALLREDUCE" and profile in {
        "reduce_only", "reduce_broadcast"}
    reduce_backend = (case["reduce"][profile]
                      if isinstance(case["reduce"], dict)
                      else case["reduce"])
    expected = {
        "op": case["op"].lower(), "profile": profile, "group_size": "4",
        "status": "accepted", "broadcast": case["broadcast"][profile],
        "reduce": reduce_backend,
        "endpoint_reduce_compute":
            "true" if case["op"] == "ALLREDUCE" and not dca else "false",
        "multicast_trees": str(case["multicast_trees"] if multicast else 0),
        "dca_trees": str(case["dca_trees"] if dca else 0),
        "requires_multicast": "true" if multicast else "false",
        "requires_dca": "true" if dca else "false", "reason": "accepted",
    }
    for key, value in expected.items():
        if row.get(key) != value:
            fail(f"profile {key}: {row.get(key)!r} != {value!r}")
    tree_count = max(int(expected["multicast_trees"]),
                     int(expected["dca_trees"]))
    if integer(row, "trees") != tree_count or \
            integer(row, "batches") != tree_count or \
            integer(row, "conflicts") < 0:
        fail(f"profile K=1 schedule mismatch: {row}")
    return row


def validate_tree_lifecycle(output: str, manifest: dict[str, str],
                            tree_count: int) -> dict[str, Any]:
    batches = rows(output, TREE_BATCH)
    drains = rows(output, TREE_DRAIN)
    if tree_count == 0:
        if batches or drains:
            fail("non-accelerated profile emitted tree lifecycle markers")
        return {"batches": [], "drain": None}
    if len(batches) != 2 * tree_count or len(drains) != 1:
        fail("K=1 tree begin/end or final drain count mismatch")
    key = f"{manifest['group_id']}:{manifest['collective_id']}:{manifest['epoch']}"
    pairs: dict[tuple[int, str, int], dict[str, dict[str, str]]] = {}
    begun_trees = []
    for row in batches:
        event = row.get("event")
        if event not in {"begin", "end"} or row.get("key") != key:
            fail(f"malformed/tree-key marker: {row}")
        identity = (integer(row, "plan"), row["key"], integer(row, "batch"))
        if event in pairs.setdefault(identity, {}):
            fail(f"duplicate tree batch event: {row}")
        pairs[identity][event] = row
        tree_ids = [int(value) for value in row["tree_ids"].split(",")]
        if tree_ids != sorted(tree_ids) or len(tree_ids) != 1 or \
                integer(row, "trees") != 1:
            fail(f"K=1/ascending tree_ids violated: {row}")
        if integer(row, "peak_entries") <= 0 or \
                integer(row, "peak_entries") > integer(row, "capacity") or \
                integer(row, "conflicts") < 0:
            fail(f"tree capacity/conflict evidence invalid: {row}")
        if event == "begin":
            begun_trees.extend(tree_ids)
            if integer(row, "programmed") != 1 or integer(row, "erased") != 0 or \
                    integer(row, "occupancy_after") <= 0 or \
                    row.get("release_reason") != "none":
                fail(f"invalid tree batch begin: {row}")
        elif integer(row, "programmed") != 0 or integer(row, "erased") != 1 or \
                integer(row, "occupancy_after") != 0 or \
                row.get("release_reason") not in {"batch_complete", "schedule_complete"}:
            fail(f"invalid tree batch end: {row}")
    if len(pairs) != tree_count or len(set(begun_trees)) != tree_count or \
            any(set(pair) != {"begin", "end"} for pair in pairs.values()):
        fail("tree batch pairing/provenance mismatch")
    for pair in pairs.values():
        for field in ("plan", "key", "batch", "tree_ids", "trees",
                      "conflicts", "peak_entries", "capacity"):
            if pair["begin"][field] != pair["end"][field]:
                fail(f"tree begin/end field {field} changed")
    drain = drains[0]
    if drain.get("key") != key or any(integer(drain, field) != 0 for field in
                                      ("tree_entries", "reduce_nodes",
                                       "schedule_entries", "residual")):
        fail(f"tree final drain non-zero: {drain}")
    return {"batches": batches, "drain": drain}


def validate_backend(output: str, profile: str, case: dict[str, Any],
                     manifest: dict[str, str]) -> dict[str, Any]:
    multicast = profile in {"broadcast_only", "reduce_broadcast"}
    dca = case["op"] == "ALLREDUCE" and profile in {
        "reduce_only", "reduce_broadcast"}
    tx = rows(output, MULTICAST_TX)
    commits = rows(output, MULTICAST_COMMIT)
    arms = rows(output, DCA_ARM)
    stream_tx = rows(output, DCA_TX)
    results = rows(output, DCA_RESULT)
    if multicast:
        if len(tx) != 4 or len(commits) != 12 or \
                any(integer(row, "bytes") != 32768 for row in tx + commits):
            fail("multicast lacks four 32KiB TX and twelve real commits")
        trees = {integer(row, "tree") for row in tx}
        if len(trees) != 4 or {integer(row, "tree") for row in commits} != trees or \
                any(integer(row, "session") <= 0 for row in tx):
            fail("multicast tree/session provenance mismatch")
    elif tx or commits:
        fail("profile unexpectedly emitted multicast evidence")
    if dca:
        if len(arms) != 4 or len(stream_tx) != 16 or len(results) != 4:
            fail("DCA marker cardinality is not 4 ARM/16 TX/4 RESULT")
        trees = {integer(row, "tree") for row in arms}
        if len(trees) != 4 or {integer(row, "tree") for row in stream_tx} != trees or \
                {integer(row, "tree") for row in results} != trees or \
                any(integer(row, "bytes") != 32768 or row.get("source") != "SRAM"
                    for row in stream_tx) or \
                any(integer(row, "elements") != 8192 or
                    row.get("value") != "committed-to-SRAM" for row in results):
            fail("DCA real SRAM byte/result provenance mismatch")
    elif arms or stream_tx or results:
        fail("profile unexpectedly emitted DCA evidence")
    lifecycle = validate_tree_lifecycle(
        output, manifest, 4 if multicast or dca else 0)
    return {"multicast_tx": len(tx), "multicast_commit": len(commits),
            "dca_arm": len(arms), "dca_tx": len(stream_tx),
            "dca_result": len(results), "tree_lifecycle": lifecycle}


def validate_stats_and_drains(output: str, profile: str,
                              case: dict[str, Any], spec: dict[str, Any],
                              manifest: dict[str, str]) -> dict[str, Any]:
    stats, drains, _ = p6.parse_runtime_output(output)
    counts = p6.expected_counts(spec)
    if len(stats) != 1 or stats[0].get("scenario") != manifest["scenario"]:
        fail("P6 stats missing/duplicated")
    for key in ("child_count", "action_count", "wave_count"):
        if p6.parse_int(stats[0], key) != counts[key]:
            fail(f"P6 stats {key} mismatch")
    if len(drains) != 1 or any(int(value) != 0 for key, value in drains[0].items()
                               if key != "scenario"):
        fail("P6 drain missing/non-zero")
    multicast = profile in {"broadcast_only", "reduce_broadcast"}
    dca = case["op"] == "ALLREDUCE" and profile in {
        "reduce_only", "reduce_broadcast"}
    endpoint_bytes = 0 if multicast or dca else 3 * int(spec["length_bytes"])
    p5_stats = rows(output, p8.P5_STATS_PREFIX)
    if endpoint_bytes == 0:
        if p5_stats:
            fail("accelerated case leaked endpoint P5 transfers/fallback")
    else:
        by_core = {integer(row, "core"): row for row in p5_stats}
        if set(by_core) != set(p6.CORES) or len(p5_stats) != 4:
            fail("P5 stats core membership mismatch")
        for core, row in by_core.items():
            for key in ("source_read_bytes", "sram_source_read_bytes",
                        "wire_bytes", "noc_rx_write_bytes"):
                if integer(row, key) != endpoint_bytes:
                    fail(f"core {core} P5 {key} byte mismatch")
            for key in ("hbm_source_read_bytes", "duplicate_requests_suppressed",
                        "request_conflicts_rejected", "request_aborts"):
                if integer(row, key) != 0:
                    fail(f"core {core} P5 {key} non-zero")
    p5_drain = rows(output, p8.P5_DRAIN_PREFIX)
    if len(p5_drain) != (4 if endpoint_bytes else 0) or \
            any(integer(row, "residual") != 0 for row in p5_drain):
        fail("P5 endpoint drain count/residual mismatch")
    timing = rows(output, p8.P5_TIMING_DRAIN_PREFIX)
    if len(timing) != 1 or integer(timing[0], "residual") != 0:
        fail("P5 timing drain missing/non-zero")
    global_drain = rows(output, p8.COLL_DRAIN_PREFIX)
    global_fields = ("tree_entries", "reduce_nodes", "barriers", "gather",
                     "reduce_rx", "endpoints", "dte_tokens", "event")
    if len(global_drain) != 1 or \
            any(integer(global_drain[0], key) != 0 for key in global_fields):
        fail("global collective drain missing/non-zero")
    return {"p6_stats": stats[0], "p6_drain": drains[0],
            "p5_stats": p5_stats, "p5_drain": p5_drain,
            "timing": timing[0], "global": global_drain[0]}


def validate_trace(path: pathlib.Path,
                   spec: dict[str, Any]) -> dict[str, int]:
    events = p6.load_trace(path)
    counts = p6.expected_counts(spec)
    actions = [event for event in events if event["name"] == p6.TRACE_ACTION]
    waves = [event for event in events if event["name"] == p6.TRACE_WAVE]
    indices = sorted(int(event.get("args", {}).get("action_index", -1))
                     for event in actions)
    if len(actions) != counts["action_count"] or \
            indices != list(range(counts["action_count"])) or \
            len(waves) != counts["wave_count"]:
        fail("canonical action/wave trace mismatch")
    return {"actions": len(actions), "waves": len(waves)}


def run_case(args: argparse.Namespace, oracle: dict[str, Any],
             case: dict[str, Any], spec: dict[str, Any], profile: str,
             frozen_hash: str) -> str:
    with tempfile.TemporaryDirectory(
            prefix=f"p7-32k-{case['op'].lower()}-{profile}-",
            dir=args.runtime_root) as temp:
        root = pathlib.Path(temp)
        p6.prepare_runtime_assets(root, args.hardware)
        work = root / "case"
        work.mkdir()
        artifact = work / "program.npup"
        manifest, digest = emit_fixture(args.program_fixture, case, spec, artifact)
        if digest != frozen_hash:
            fail("fixture bytes changed between stability and runtime processes")
        probe = p6.memory_probe(spec, manifest)
        if case["op"] == "ALLREDUCE" and profile in {
                "reduce_only", "reduce_broadcast"}:
            probe = dca_probe(probe, spec, manifest)
        probe_path = work / "probe.json"
        hardware_path = work / "hardware.json"
        simulation_path = work / "simulation.json"
        write_json(probe_path, probe)
        p6.write_case_hardware(args.hardware, manifest, hardware_path)
        write_json(simulation_path, simulation(
            args.simulation, profile, oracle["max_trees_per_batch"]))
        completed = subprocess.run(
            [str(args.npusim), "--program", str(artifact),
             "--p6-memory-probe", str(probe_path),
             "--hardware-config", str(hardware_path),
             "--simulation-config", str(simulation_path),
             "--mapping-config", str(args.mapping),
             "--trace-window", "1000000"], cwd=work, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=args.timeout, check=False)
        output = completed.stdout
        if completed.returncode != 0:
            fail(f"{case['op']}/{profile} returned {completed.returncode}:\n{output}")
        forbidden = ("ISA-v1 requested collective backend rejected",
                     "fallback", "[PROTO_WAIT]")
        if any(value in output for value in forbidden) or \
                "End DONE reception" not in output:
            fail(f"{case['op']}/{profile} rejected/fell back/did not drain DONE")
        trace = work / "events.json"
        if not trace.is_file():
            fail("runtime emitted no events.json")
        normalized = {
            "artifact": digest,
            "profile": validate_profile(output, profile, case),
            "backend": validate_backend(output, profile, case, manifest),
            "memory": validate_memory(output, probe),
            "stats": validate_stats_and_drains(
                output, profile, case, spec, manifest),
            "trace": validate_trace(trace, spec),
        }
        return p8.normalized_hash(normalized)


def selftest(specs: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    _, reduce_spec = specs[1]
    manifest = {"payload_offset": "16", "sentinel": "165",
                "staging_size": str(16 + 4 * 32768 + 16)}
    base = {"initializations": [], "verifications": [
        {"region": "p6_staging"}, {"region": "p6_result"}]}
    probe = dca_probe(base, reduce_spec, manifest)
    staging = probe["initializations"]
    if len(staging) != 4 or any(item["region"] == "p6_staging"
                                for item in probe["verifications"]):
        fail("DCA expected-after staging membership selftest failed")
    for rank, item in enumerate(staging):
        initial = base64.b64decode(item["bytes_base64"], validate=True)
        after = base64.b64decode(item["expected_after_bytes_base64"], validate=True)
        begin = rank * 32768
        end = begin + 32768
        if after[:begin] != initial[:begin] or after[end:] != initial[end:] or \
                after[begin:end] != p6.source_segment(reduce_spec, rank, rank) or \
                p6.crc32c(after) != item["expected_after_checksum"]:
            fail("DCA canonical local-copy expected-after bytes changed")
    print("[P7 ACCELERATED 32K] PASS: runner/oracle/expected-after selftest")


def run_matrix(args: argparse.Namespace, oracle: dict[str, Any],
               specs: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    frozen: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="p7-32k-fixture-stability-",
                                     dir=args.runtime_root) as temp:
        root = pathlib.Path(temp)
        for case, spec in specs:
            first = root / f"{case['op'].lower()}-a.npup"
            second = root / f"{case['op'].lower()}-b.npup"
            manifest_a, digest_a = emit_fixture(
                args.program_fixture, case, spec, first)
            manifest_b, digest_b = emit_fixture(
                args.program_fixture, case, spec, second)
            if digest_a != digest_b or manifest_a != manifest_b or \
                    first.read_bytes() != second.read_bytes():
                fail(f"fixture {case['option']} is not byte-stable")
            frozen[case["option"]] = digest_a
    hashes = []
    for case, spec in specs:
        for profile in PROFILES:
            digest = run_case(args, oracle, case, spec, profile,
                              frozen[case["option"]])
            hashes.append(digest)
            print(f"[P7 ACCELERATED 32K CASE] op={case['op']} "
                  f"profile={profile} sha256={digest}")
    if len(hashes) != 8:
        fail("runtime matrix did not execute eight isolated processes")
    print("[P7 ACCELERATED 32K] PASS: cases=2 profiles=4 "
          "isolated_processes=8 K=1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--oracle", type=pathlib.Path, required=True)
    parser.add_argument("--p6-oracle", type=pathlib.Path,
                        default=PROGRAM_DIR / "p6_collective_oracle.json")
    parser.add_argument("--npusim", type=pathlib.Path)
    parser.add_argument("--program-fixture", type=pathlib.Path)
    parser.add_argument("--hardware", type=pathlib.Path)
    parser.add_argument("--simulation", type=pathlib.Path)
    parser.add_argument("--mapping", type=pathlib.Path)
    parser.add_argument("--runtime-root", type=pathlib.Path,
                        default=pathlib.Path("."))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    oracle = load_oracle(args.oracle.resolve())
    specs = resolve_specs(oracle, args.p6_oracle.resolve())
    if args.selftest:
        selftest(specs)
        return
    if any(value is None for value in (args.npusim, args.program_fixture,
                                       args.hardware, args.simulation,
                                       args.mapping)):
        parser.error("runtime mode requires npusim, fixture, hardware, simulation, and mapping")
    for name in ("npusim", "program_fixture", "hardware", "simulation",
                 "mapping", "runtime_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    run_matrix(args, oracle, specs)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
