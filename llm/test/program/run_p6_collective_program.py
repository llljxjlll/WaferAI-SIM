#!/usr/bin/env python3
"""P6 collective Program Format fixture, byte-oracle, and runtime-output runner."""

import argparse
import base64
import json
import os
import pathlib
import subprocess
import sys
import tempfile

MANIFEST_PREFIX = "P6_COLLECTIVE "
SOURCE_PREFIX = "P6_SOURCE "
EXPECT_PREFIX = "P6_EXPECT "
STATS_PREFIX = "[P6 COLLECTIVE STATS] "
DRAIN_PREFIX = "[P6 COLLECTIVE DRAIN] "
PROBE_PREFIX = "[P6 MEMORY PROBE] "
TRACE_ACTION = "P6_collective_action"
TRACE_WAVE = "P6_collective_wave_complete"
CORES = [1, 3, 7, 11]
ROOT_OPS_TX = {"SCATTER", "BROADCAST"}
ROOT_OPS_RX = {"GATHER", "REDUCE"}
REDUCE_OPS = {"REDUCE", "REDUCESCATTER", "ALLREDUCE"}
DTYPE_WIDTH = {"UINT8": 1, "INT32": 4, "INT64": 8}
DTYPE_OPTION = {"UINT8": "u8", "INT32": "i32", "INT64": "i64"}


def fail(message):
    raise RuntimeError(message)


def crc32c(payload):
    checksum = 0xFFFFFFFF
    for value in payload:
        checksum ^= value
        for _ in range(8):
            checksum = ((checksum >> 1) ^
                        (0x82F63B78 if checksum & 1 else 0))
    return (~checksum) & 0xFFFFFFFF


def key_values(text):
    result = {}
    for field in text.strip().split():
        if "=" not in field:
            fail(f"non key=value field: {field!r}")
        key, value = field.split("=", 1)
        if not key or key in result or not value:
            fail(f"non-canonical/duplicate field: {field!r}")
        result[key] = value
    return result


def parse_manifest(stdout):
    manifests = []
    sources = []
    expects = []
    for raw in stdout.splitlines():
        if raw.startswith(MANIFEST_PREFIX):
            manifests.append(key_values(raw[len(MANIFEST_PREFIX):]))
        elif raw.startswith(SOURCE_PREFIX):
            sources.append(key_values(raw[len(SOURCE_PREFIX):]))
        elif raw.startswith(EXPECT_PREFIX):
            expects.append(key_values(raw[len(EXPECT_PREFIX):]))
    if len(manifests) != 1:
        fail(f"expected exactly one P6 manifest, got {len(manifests)}")
    return manifests[0], sources, expects


def operation(tx, rx):
    table = {
        ("unicast", "unicast"): "P2P",
        ("scatter", "unicast"): "SCATTER",
        ("broadcast", "unicast"): "BROADCAST",
        ("unicast", "gather"): "GATHER",
        ("scatter", "gather"): "ALLTOALL",
        ("broadcast", "gather"): "ALLGATHER",
        ("unicast", "reduce"): "REDUCE",
        ("scatter", "reduce"): "REDUCESCATTER",
        ("broadcast", "reduce"): "ALLREDUCE",
    }
    try:
        return table[(tx, rx)]
    except KeyError:
        fail(f"unknown 3x3 cell {tx}/{rx}")


def positive_option(tx, rx, n, length, dtype="UINT8", reduce_op="NONE"):
    return (f"--p6-{tx}-{rx}-n{n}-l{length}-"
            f"{DTYPE_OPTION[dtype]}-{reduce_op.lower()}")


def load_oracle(path):
    with open(path, "r", encoding="utf-8") as stream:
        oracle = json.load(stream)
    if oracle.get("version") != 1:
        fail("P6 oracle version must be 1")
    grid = oracle.get("grid", {})
    if grid.get("group_sizes") != [1, 2, 4] or len(grid.get("cells", [])) != 9:
        fail("P6 oracle must contain the canonical 3x3 N=1/2/4 grid")
    seen = set()
    for cell in grid["cells"]:
        key = (cell.get("tx"), cell.get("rx"))
        if operation(*key) != cell.get("op") or key in seen:
            fail("P6 oracle 3x3 cell is duplicate or misnamed")
        seen.add(key)
    categories = {entry.get("category")
                  for entry in oracle.get("negative_cases", [])}
    required = {"group", "rank", "root", "crossdie", "count",
                "stride", "schedule"}
    if categories != required:
        fail("P6 oracle negative matrix is incomplete")
    return oracle


def expand_positive_cases(oracle):
    result = []
    grid = oracle["grid"]
    for n in grid["group_sizes"]:
        for cell in grid["cells"]:
            reduction = cell["rx"] == "reduce"
            dtype = grid["reduce_dtype"] if reduction else "UINT8"
            reduce_op = grid["reduce_op"] if reduction else "NONE"
            result.append({
                "tx": cell["tx"], "rx": cell["rx"], "op": cell["op"],
                "n": n, "length_bytes": grid["length_bytes"],
                "dtype": dtype, "reduce_op": reduce_op,
                "option": positive_option(cell["tx"], cell["rx"], n,
                                          grid["length_bytes"], dtype,
                                          reduce_op),
            })
    for entry in oracle["length_cases"]:
        item = dict(entry)
        item["op"] = operation(item["tx"], item["rx"])
        item["dtype"] = "UINT8"
        item["reduce_op"] = "NONE"
        item["option"] = positive_option(
            item["tx"], item["rx"], item["n"], item["length_bytes"])
        result.append(item)
    for entry in oracle["reduction_cases"]:
        item = dict(entry)
        reduce_op = item.pop("op")
        item["op"] = operation(item["tx"], item["rx"])
        item["reduce_op"] = reduce_op
        item["option"] = positive_option(
            item["tx"], item["rx"], item["n"], item["length_bytes"],
            item["dtype"], item["reduce_op"])
        result.append(item)
    unique = {}
    for item in result:
        unique[item["option"]] = item
    return [unique[key] for key in sorted(unique)]


def root_rank(spec):
    return 0 if spec["n"] == 1 else (
        1 if spec["op"] in ROOT_OPS_TX | ROOT_OPS_RX else 0)


def want_send(spec, rank):
    if spec["op"] == "P2P":
        return rank == 0
    if spec["op"] in ROOT_OPS_TX:
        return rank == root_rank(spec)
    return True


def want_receive(spec, rank):
    if spec["op"] == "P2P":
        return rank == spec["n"] - 1
    if spec["op"] in ROOT_OPS_RX:
        return rank == root_rank(spec)
    return True


def want_compute(spec, rank):
    if spec["op"] == "REDUCE":
        return rank == root_rank(spec)
    return spec["op"] in {"REDUCESCATTER", "ALLREDUCE"}


def expected_wave_count(spec):
    n = spec["n"]
    op = spec["op"]
    root = root_rank(spec)
    if op == "P2P":
        pairs = [(0, n - 1)]
    elif op in ROOT_OPS_TX:
        pairs = [(root, destination) for destination in range(n)]
    elif op in ROOT_OPS_RX:
        pairs = [(source, root) for source in range(n)]
    else:
        pairs = [(source, destination)
                 for source in range(n) for destination in range(n)]
    flows = [(source, destination) for source, destination in pairs
             if source != destination]
    if not flows:
        return 1

    waves = 1
    sessions = [0] * n
    receive_bytes = [0] * n
    for source, destination in flows:
        def fits():
            return (sessions[source] < 3 and
                    sessions[destination] < 3 and
                    receive_bytes[destination] + spec["length_bytes"] <=
                    32768)
        if not fits():
            waves += 1
            sessions = [0] * n
            receive_bytes = [0] * n
        if not fits():
            fail("P6 child cannot fit an empty production wave")
        sessions[source] += 1
        sessions[destination] += 1
        receive_bytes[destination] += spec["length_bytes"]
    return waves


def expected_counts(spec):
    n = spec["n"]
    op = spec["op"]
    if op == "P2P":
        local = 1 if n == 1 else 0
        child = 0 if n == 1 else 1
    elif op in ROOT_OPS_TX | ROOT_OPS_RX:
        local = 1
        child = n - 1
    else:
        local = n
        child = n * (n - 1)
    reduce_count = (
        (1 if op == "REDUCE" else n)
        if op in REDUCE_OPS and n > 1 else 0)
    waves = expected_wave_count(spec)
    actions = local + 5 * child + 2 * n * waves + reduce_count
    issues = sum(
        int(want_send(spec, rank)) +
        int(want_receive(spec, rank)) +
        int(want_compute(spec, rank))
        for rank in range(n))
    return {
        "child_count": child,
        "local_copy_count": local,
        "reduce_count": reduce_count,
        "wave_count": waves,
        "action_count": actions,
        "issue_count": issues,
    }


def boundary_value(spec, rank):
    lane = rank & 3
    if spec["dtype"] == "UINT8":
        values = ([250, 10, 255, 1] if spec["reduce_op"] == "SUM"
                  else [0, 255, 127, 128])
    elif spec["dtype"] == "INT32":
        values = ([0x7FFFFFFF, 1, 0xFFFFFFFF, 0x80000000]
                  if spec["reduce_op"] == "SUM"
                  else [0x80000000, 0xFFFFFFFF, 0x7FFFFFFF, 0])
    else:
        values = ([0x7FFFFFFFFFFFFFFF, 1, 0xFFFFFFFFFFFFFFFF,
                   0x8000000000000000]
                  if spec["reduce_op"] == "SUM"
                  else [0x8000000000000000, 0xFFFFFFFFFFFFFFFF,
                        0x7FFFFFFFFFFFFFFF, 0])
    return values[lane]


def source_bytes(spec, rank):
    length = spec["length_bytes"]
    byte_count = spec["n"] * length if spec["tx"] == "scatter" else length
    seed = 0x21 + rank * 0x31
    multiplier = 3 + rank * 2
    result = bytearray(
        (seed + multiplier * index + (index >> 2)) & 0xFF
        for index in range(byte_count))
    if spec["rx"] == "reduce":
        width = DTYPE_WIDTH[spec["dtype"]]
        value = boundary_value(spec, rank)
        for offset in range(0, byte_count, length):
            result[offset:offset + width] = value.to_bytes(
                width, "little", signed=False)
    return bytes(result)


def source_segment(spec, source_rank, destination_rank):
    payload = source_bytes(spec, source_rank)
    begin = (destination_rank * spec["length_bytes"]
             if spec["tx"] == "scatter" else 0)
    return payload[begin:begin + spec["length_bytes"]]


def staging_bytes(spec, destination_rank):
    if spec["op"] == "P2P":
        return (source_segment(spec, 0, destination_rank)
                if destination_rank == spec["n"] - 1 else b"")
    if spec["op"] in ROOT_OPS_TX:
        return source_segment(spec, root_rank(spec), destination_rank)
    if spec["op"] in ROOT_OPS_RX:
        if destination_rank != root_rank(spec):
            return b""
        return b"".join(source_segment(spec, source, destination_rank)
                        for source in range(spec["n"]))
    return b"".join(source_segment(spec, source, destination_rank)
                    for source in range(spec["n"]))


def signed_less(left, right, width):
    if width == 1:
        return left < right
    sign = 1 << (width * 8 - 1)
    if bool(left & sign) != bool(right & sign):
        return bool(left & sign)
    return left < right


def result_bytes(spec, destination_rank):
    if spec["op"] not in REDUCE_OPS:
        return staging_bytes(spec, destination_rank)
    width = DTYPE_WIDTH[spec["dtype"]]
    mask = (1 << (width * 8)) - 1
    inputs = [source_segment(spec, source, destination_rank)
              for source in range(spec["n"])]
    output = bytearray(spec["length_bytes"])
    for offset in range(0, spec["length_bytes"], width):
        values = [int.from_bytes(value[offset:offset + width], "little")
                  for value in inputs]
        aggregate = values[0]
        for value in values[1:]:
            if spec["reduce_op"] == "SUM":
                aggregate = (aggregate + value) & mask
            elif signed_less(aggregate, value, width):
                aggregate = value
        output[offset:offset + width] = aggregate.to_bytes(width, "little")
    return bytes(output)


def expected_outputs(spec):
    outputs = []
    for rank in range(spec["n"]):
        if not want_receive(spec, rank):
            continue
        staging = staging_bytes(spec, rank)
        if spec["op"] not in REDUCE_OPS:
            outputs.append((rank, "p6_staging", staging))
        elif spec["n"] > 1:
            outputs.append((rank, "p6_staging", staging))
    if spec["op"] in REDUCE_OPS:
        for rank in range(spec["n"]):
            if want_compute(spec, rank):
                outputs.append((rank, "p6_result",
                                result_bytes(spec, rank)))
    return outputs


def parse_int(fields, name):
    try:
        return int(fields[name], 10)
    except (KeyError, ValueError):
        fail(f"manifest field {name} is missing or non-decimal")


def case_hardware_document(document, manifest):
    try:
        sram = document["memory"]["sram"]
        regions = sram["regions"]
    except (KeyError, TypeError):
        fail("P6 hardware template has no memory.sram.regions")
    if not isinstance(regions, list):
        fail("P6 hardware template regions must be an array")
    workspace_indices = [
        index for index, region in enumerate(regions)
        if isinstance(region, dict) and region.get("name") == "p6_workspace"
    ]
    if len(workspace_indices) != 1:
        fail("P6 hardware template must contain exactly one p6_workspace")
    workspace_index = workspace_indices[0]
    workspace = regions[workspace_index]
    try:
        workspace_base = int(workspace["base_bytes"])
        workspace_size = int(workspace["size_bytes"])
    except (KeyError, TypeError, ValueError):
        fail("P6 hardware workspace base/size is invalid")
    if workspace_base < 0 or workspace_size <= 0:
        fail("P6 hardware workspace base/size must be positive")
    workspace_end = workspace_base + workspace_size

    layouts = []
    for name, prefix in (("p6_input", "input"),
                         ("p6_staging", "staging"),
                         ("p6_result", "result")):
        base = parse_int(manifest, f"{prefix}_base")
        size = parse_int(manifest, f"{prefix}_size")
        end = base + size
        if size <= 0 or base < workspace_base or end > workspace_end:
            fail(f"P6 manifest {name} is outside p6_workspace")
        if layouts and base < layouts[-1][3]:
            fail(f"P6 manifest {name} overlaps its predecessor")
        layouts.append((name, base, size, end))

    replacements = []
    for name, base, size, _ in layouts:
        region = dict(workspace)
        region.update({"name": name, "base_bytes": base,
                       "size_bytes": size})
        replacements.append(region)
    regions[workspace_index:workspace_index + 1] = replacements
    return document


def write_case_hardware(template_path, manifest, output_path):
    with open(template_path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    document = case_hardware_document(document, manifest)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(document, stream, sort_keys=True, indent=2)
        stream.write("\n")


def prepare_runtime_assets(root, hardware_template):
    source_root = hardware_template.parents[3]
    os.symlink(source_root / "font", root / "font",
               target_is_directory=True)
    os.symlink(source_root / "DRAMSys", root / "DRAMSys",
               target_is_directory=True)


def validate_manifest(spec, manifest, sources, expects):
    expected_header = {
        "tx": spec["tx"], "rx": spec["rx"], "op": spec["op"],
        "n": str(spec["n"]),
        "length_bytes": str(spec["length_bytes"]),
        "dtype": spec["dtype"], "reduce_op": spec["reduce_op"],
        "negative": "none",
        "cores": ",".join(str(core) for core in CORES[:spec["n"]]),
    }
    for key, value in expected_header.items():
        if manifest.get(key) != value:
            fail(f"{spec['option']} manifest {key}: "
                 f"{manifest.get(key)!r} != {value!r}")
    for key, value in expected_counts(spec).items():
        if parse_int(manifest, key) != value:
            fail(f"{spec['option']} count {key} mismatch")
    if parse_int(manifest, "payload_offset") != 16 or \
            parse_int(manifest, "sentinel") != 0xA5:
        fail(f"{spec['option']} SRAM sentinel contract mismatch")
    if len(sources) != spec["n"]:
        fail(f"{spec['option']} source manifest count mismatch")
    for fields in sources:
        rank = parse_int(fields, "rank")
        payload = source_bytes(spec, rank)
        if parse_int(fields, "core") != CORES[rank] or \
                parse_int(fields, "bytes") != len(payload) or \
                parse_int(fields, "checksum") != crc32c(payload):
            fail(f"{spec['option']} source rank {rank} oracle mismatch")
    expected = {(rank, region): payload
                for rank, region, payload in expected_outputs(spec)
                if region != "p6_staging" or spec["op"] not in REDUCE_OPS}
    actual = {}
    for fields in expects:
        rank = parse_int(fields, "rank")
        region = fields["region"]
        actual[(rank, region)] = (
            parse_int(fields, "bytes"), parse_int(fields, "checksum"))
    if set(actual) != set(expected):
        fail(f"{spec['option']} expected-output membership mismatch")
    for key, payload in expected.items():
        if actual[key] != (len(payload), crc32c(payload)):
            fail(f"{spec['option']} output {key} checksum mismatch")


def memory_probe(spec, manifest):
    offset = parse_int(manifest, "payload_offset")
    sentinel = parse_int(manifest, "sentinel")
    initializations = []
    for rank in range(spec["n"]):
        payload = source_bytes(spec, rank)
        initializations.append({
            "space": "SRAM", "core": CORES[rank], "region": "p6_input",
            "region_size_bytes": parse_int(manifest, "input_size"),
            "offset_bytes": offset, "length_bytes": len(payload),
            "prefill_byte": sentinel,
            "pattern": {
                "kind": "affine_u8_v1",
                "seed": 0x21 + rank * 0x31,
                "multiplier": 3 + rank * 2,
                "quarter_step": 1,
                "reduction_boundary_patch": spec["op"] in REDUCE_OPS,
            },
            "bytes_base64": base64.b64encode(payload).decode("ascii"),
            "checksum": crc32c(payload),
        })
    verifications = []
    for rank, region, payload in expected_outputs(spec):
        size_name = "staging_size" if region == "p6_staging" else "result_size"
        verifications.append({
            "core": CORES[rank], "region": region,
            "region_size_bytes": parse_int(manifest, size_name),
            "payload_offset_bytes": offset,
            "payload_length_bytes": len(payload),
            "expected_bytes_base64":
                base64.b64encode(payload).decode("ascii"),
            "expected_checksum": crc32c(payload),
            "prefill_byte": sentinel,
            "verify_all_bytes_outside_payload": True,
        })
    return {
        "version": 1,
        "format": "p6-p5-compatible-multi-region-v1",
        "scenario": manifest["scenario"],
        "initializations": initializations,
        "verifications": verifications,
    }


def run_fixture(executable, option, artifact_path):
    completed = subprocess.run(
        [str(executable), str(artifact_path), option],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False)
    if completed.returncode != 0:
        fail(f"{option} fixture failed: {completed.stderr.strip()}")
    if not artifact_path.is_file() or artifact_path.stat().st_size == 0:
        fail(f"{option} did not emit a non-empty artifact")
    return parse_manifest(completed.stdout)


def run_loader_negative(args, artifact, manifest, option):
    with tempfile.TemporaryDirectory(
            prefix="p6-negative-loader-", dir=args.runtime_root) as temp:
        root = pathlib.Path(temp)
        prepare_runtime_assets(root, args.hardware)
        case_dir = root / "case"
        case_dir.mkdir()
        hardware = case_dir / "hardware.json"
        write_case_hardware(args.hardware, manifest, hardware)
        completed = subprocess.run(
            [str(args.npusim), "--program", str(artifact),
             "--hardware-config", str(hardware),
             "--simulation-config", str(args.simulation),
             "--mapping-config", str(args.mapping),
             "--trace-window", "1000000"],
            cwd=case_dir, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=60)
        if completed.returncode == 0:
            fail(f"{option} loader negative returned success")
        if "Loaded Program Format" in completed.stdout:
            fail(f"{option} reached runtime instead of loader rejection")


def fixture_selftest(args, oracle):
    positives = expand_positive_cases(oracle)
    executable = args.program_fixture
    output_root = args.runtime_root / "p6_collective_fixtures"
    output_root.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(positives):
        artifact = output_root / f"positive_{index:02d}.npup"
        manifest, sources, expects = run_fixture(
            executable, spec["option"], artifact)
        validate_manifest(spec, manifest, sources, expects)
        probe_path = artifact.with_suffix(".p5-memory.json")
        with open(probe_path, "w", encoding="utf-8") as stream:
            json.dump(memory_probe(spec, manifest), stream,
                      sort_keys=True, separators=(",", ":"))
            stream.write("\n")
    for index, negative in enumerate(oracle["negative_cases"]):
        artifact = output_root / f"negative_{index:02d}.npup"
        if negative.get("fixture_reject", False):
            completed = subprocess.run(
                [str(executable), str(artifact), negative["option"]],
                text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False)
            if completed.returncode == 0 or artifact.exists() or \
                    "unknown core group" not in completed.stderr:
                fail(f"{negative['option']} was not rejected atomically "
                     "by Program Format validation")
            continue
        manifest, _, _ = run_fixture(
            executable, negative["option"], artifact)
        if manifest.get("negative") != negative["category"]:
            fail(f"{negative['option']} negative category mismatch")
        run_loader_negative(
            args, artifact, manifest, negative["option"])
    emitted_negatives = sum(
        1 for item in oracle["negative_cases"]
        if not item.get("fixture_reject", False))
    print("[P6 COLLECTIVE PROGRAM] PASS: "
          f"{len(positives)} positive + "
          f"{emitted_negatives} negative artifacts + "
          f"{len(oracle['negative_cases']) - emitted_negatives} "
          "fixture rejection + "
          f"{emitted_negatives} loader rejections")


def parse_runtime_output(text):
    stats = []
    drains = []
    probes = []
    for raw in text.splitlines():
        if raw.startswith(STATS_PREFIX):
            stats.append(key_values(raw[len(STATS_PREFIX):]))
        elif raw.startswith(DRAIN_PREFIX):
            drains.append(key_values(raw[len(DRAIN_PREFIX):]))
        elif raw.startswith(PROBE_PREFIX):
            probes.append(key_values(raw[len(PROBE_PREFIX):]))
    return stats, drains, probes


def load_trace(path):
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    events = document["traceEvents"] if isinstance(document, dict) else document
    if not isinstance(events, list):
        fail("P6 trace must contain a traceEvents list")
    return [event for event in events
            if event.get("name") in {TRACE_ACTION, TRACE_WAVE}]


def validate_runtime(spec, manifest, output, trace_path):
    stats, drains, probes = parse_runtime_output(output)
    if len(stats) != 1 or stats[0].get("scenario") != manifest["scenario"]:
        fail("P6 runtime stats missing or mismatched")
    counts = expected_counts(spec)
    for key in ("child_count", "action_count", "wave_count"):
        if parse_int(stats[0], key) != counts[key]:
            fail(f"P6 runtime stats {key} mismatch")
    if len(drains) != 1:
        fail("P6 runtime drain line missing")
    for key, value in drains[0].items():
        if key != "scenario" and int(value) != 0:
            fail(f"P6 residual {key} is non-zero")
    expected = {(CORES[rank], region, crc32c(payload))
                for rank, region, payload in expected_outputs(spec)}
    actual = {(parse_int(item, "core"), item["region"],
               parse_int(item, "checksum"))
              for item in probes
              if item.get("payload_match") == "1" and
              item.get("sentinels_intact") == "1"}
    if actual != expected:
        fail("P6 memory-probe result membership/checksum mismatch")
    events = load_trace(trace_path)
    actions = [event for event in events if event["name"] == TRACE_ACTION]
    waves = [event for event in events if event["name"] == TRACE_WAVE]
    if len(actions) != counts["action_count"]:
        fail("P6 action trace count mismatch")
    indices = sorted(int(event.get("args", {}).get("action_index", -1))
                     for event in actions)
    if indices != list(range(counts["action_count"])):
        fail("P6 action trace indices are not canonical")
    if len(waves) != counts["wave_count"]:
        fail("P6 wave trace count mismatch")


def runner_selftest():
    sum_expected = {
        "UINT8": 4,
        "INT32": 0xFFFFFFFF,
        "INT64": 0xFFFFFFFFFFFFFFFF,
    }
    max_expected = {
        "UINT8": 255,
        "INT32": 0x7FFFFFFF,
        "INT64": 0x7FFFFFFFFFFFFFFF,
    }
    for dtype, width in DTYPE_WIDTH.items():
        for reduce_op, expected in (
                ("SUM", sum_expected[dtype]),
                ("MAX", max_expected[dtype])):
            boundary = {
                "tx": "broadcast", "rx": "reduce",
                "op": "ALLREDUCE", "n": 4,
                "length_bytes": width * 4, "dtype": dtype,
                "reduce_op": reduce_op,
            }
            first = int.from_bytes(
                result_bytes(boundary, 0)[:width], "little")
            if first != expected:
                fail(f"{dtype}/{reduce_op} boundary oracle mismatch")
    spec = {
        "tx": "broadcast", "rx": "reduce", "op": "ALLREDUCE",
        "n": 4, "length_bytes": 1024, "dtype": "INT32",
        "reduce_op": "MAX", "option": "selftest",
    }
    counts = expected_counts(spec)
    manifest = {"scenario": "p6_selftest"}
    stats = (f"{STATS_PREFIX}scenario=p6_selftest "
             f"child_count={counts['child_count']} "
             f"action_count={counts['action_count']} "
             f"wave_count={counts['wave_count']}\n")
    drains = (f"{DRAIN_PREFIX}scenario=p6_selftest aggregate=0 "
              "admission=0 barrier=0 endpoint=0 timing=0\n")
    probes = ""
    for rank, region, payload in expected_outputs(spec):
        probes += (f"{PROBE_PREFIX}scenario=p6_selftest "
                   f"core={CORES[rank]} region={region} "
                   f"checksum={crc32c(payload)} payload_match=1 "
                   "sentinels_intact=1\n")
    events = []
    for index in range(counts["action_count"]):
        events.append({"name": TRACE_ACTION,
                       "args": {"action_index": index}})
    for wave in range(counts["wave_count"]):
        events.append({"name": TRACE_WAVE, "args": {"wave": wave}})
    with tempfile.TemporaryDirectory(prefix="p6-runner-selftest-") as temp:
        trace = pathlib.Path(temp) / "trace.json"
        with open(trace, "w", encoding="utf-8") as stream:
            json.dump({"traceEvents": events}, stream)
        validate_runtime(spec, manifest, stats + drains + probes, trace)
        broken = stats + drains.replace("barrier=0", "barrier=1") + probes
        try:
            validate_runtime(spec, manifest, broken, trace)
        except RuntimeError:
            pass
        else:
            fail("P6 runner selftest accepted non-zero residual")
    print("[P6 COLLECTIVE PROGRAM] PASS: trace/stats/drain/oracle selftest")


def runtime_case(args, oracle):
    positives = expand_positive_cases(oracle)
    selected = [item for item in positives if item["option"] == args.option]
    if len(selected) != 1:
        fail("--option must name exactly one positive oracle fixture")
    spec = selected[0]
    with tempfile.TemporaryDirectory(
            prefix="p6-runtime-", dir=args.runtime_root) as temp:
        root = pathlib.Path(temp)
        prepare_runtime_assets(root, args.hardware)
        case_dir = root / "case"
        case_dir.mkdir()
        artifact = case_dir / "program.npup"
        manifest, sources, expects = run_fixture(
            args.program_fixture, spec["option"], artifact)
        validate_manifest(spec, manifest, sources, expects)
        probe = case_dir / "p6_memory_probe.json"
        with open(probe, "w", encoding="utf-8") as stream:
            json.dump(memory_probe(spec, manifest), stream, sort_keys=True)
        hardware = case_dir / "hardware.json"
        write_case_hardware(args.hardware, manifest, hardware)
        trace = case_dir / "events.json"
        command = [
            str(args.npusim), "--program", str(artifact),
            "--p6-memory-probe", str(probe),
            "--hardware-config", str(hardware),
            "--simulation-config", str(args.simulation),
            "--mapping-config", str(args.mapping),
            "--trace-window", "1000000",
        ]
        completed = subprocess.run(
            command, cwd=case_dir, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            fail("P6 runtime failed (blocked-p6-worker):\n" + completed.stdout)
        if not trace.is_file():
            fail("P6 runtime did not emit events.json")
        validate_runtime(spec, manifest, completed.stdout, trace)
    print(f"[P6 COLLECTIVE CASE] PASS: {spec['option']}")


def runtime_matrix(args, oracle):
    positives = expand_positive_cases(oracle)
    for spec in positives:
        args.option = spec["option"]
        runtime_case(args, oracle)
    print("[P6 COLLECTIVE PROGRAM] PASS: "
          f"runtime matrix {len(positives)} fixtures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--fixture-selftest", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--runtime-all", action="store_true")
    parser.add_argument("--program-fixture", type=pathlib.Path)
    parser.add_argument("--oracle", type=pathlib.Path)
    parser.add_argument("--runtime-root", type=pathlib.Path,
                        default=pathlib.Path("."))
    parser.add_argument("--npusim", type=pathlib.Path)
    parser.add_argument("--hardware", type=pathlib.Path)
    parser.add_argument("--simulation", type=pathlib.Path)
    parser.add_argument("--mapping", type=pathlib.Path)
    parser.add_argument("--option")
    args = parser.parse_args()
    modes = sum((args.selftest, args.fixture_selftest,
                 args.runtime, args.runtime_all))
    if modes != 1:
        parser.error("select exactly one of --selftest, --fixture-selftest, --runtime")
    if args.selftest:
        runner_selftest()
        return
    if args.oracle is None:
        parser.error("--oracle is required")
    oracle = load_oracle(args.oracle)
    if args.fixture_selftest:
        required = [args.program_fixture, args.npusim, args.hardware,
                    args.simulation, args.mapping]
        if any(value is None for value in required):
            parser.error("fixture selftest requires fixture/npusim/configs")
        for name in ("program_fixture", "npusim", "hardware",
                     "simulation", "mapping", "runtime_root"):
            value = getattr(args, name)
            setattr(args, name, value.resolve())
        args.runtime_root.mkdir(parents=True, exist_ok=True)
        fixture_selftest(args, oracle)
        return
    required = [args.program_fixture, args.npusim, args.hardware,
                args.simulation, args.mapping]
    if args.runtime:
        required.append(args.option)
    if any(value is None for value in required):
        parser.error("runtime mode requires fixture/npusim/configs"
                     " and --option for a single case")
    for name in ("program_fixture", "npusim", "hardware", "simulation",
                 "mapping", "oracle", "runtime_root"):
        value = getattr(args, name)
        setattr(args, name, value.resolve())
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.runtime_all:
        runtime_matrix(args, oracle)
    else:
        runtime_case(args, oracle)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"[P6 COLLECTIVE PROGRAM] FAIL: {error}", file=sys.stderr)
        sys.exit(1)
