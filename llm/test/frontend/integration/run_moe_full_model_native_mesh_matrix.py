"""Two-fresh, artifact-reopened MoE full-model native mesh matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[4]
M1_SHAPES = ("1x1", "1x4", "4x1", "2x2", "2x3", "3x2", "3x3", "10x10")
RELEASE_SHAPES = tuple(
    f"{rows}x{columns}"
    for rows in range(1, 11)
    for columns in range(1, 11)
)
_MESH = re.compile(r"([1-9][0-9]*)x([1-9][0-9]*)\Z")
_KV = re.compile(
    r"\[DENSE_SEQUENCE_KV\] index=(\d+) bytes=(\d+) "
    r"digest=([0-9a-f]{64}) pass=(\d+)"
)
_MEMORY = re.compile(
    r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+) "
    r"lsu_completed=(\d+) [^\n]*lsu_residual=(\d+) dte_residual=(\d+)"
)
_LINK = re.compile(
    r"\[D2D_LINK\] idx=(\d+) die(\d+)->die(\d+) dir=([EWNS]) "
    r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
    r"data_in=(\d+) data_out=(\d+)\."
)
_D2D = re.compile(r"\[D2D_DATA\] in_pkts=(\d+) out_pkts=(\d+)")
_D2D_TYPE = re.compile(
    r"\[D2D_TYPE\] request_in=(\d+) request_out=(\d+) "
    r"ack_in=(\d+) ack_out=(\d+) data_in=(\d+) data_out=(\d+)"
)
_MAKESPAN = re.compile(r"\[SIM_RESULT\] makespan_cycles=(\d+)")
_RESOLVER = re.compile(
    r"ProgramIo resolved id=([a-z0-9_]+) "
    r"initializations=(\d+) probes=(\d+)"
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PACKET_PAYLOAD_BYTES = 16


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _shape(value: str) -> tuple[int, int]:
    match = _MESH.fullmatch(value)
    if match is None:
        raise ValueError(f"noncanonical mesh: {value}")
    rows, columns = (int(part) for part in match.groups())
    if not 1 <= rows <= 10 or not 1 <= columns <= 10:
        raise ValueError(f"outside 1..10 release envelope: {value}")
    return rows, columns


def _recompute_flow_evidence(
    value: object, *, rows: int, columns: int,
) -> tuple[list[dict[str, int | str]], list[int]]:
    rank_count = rows * columns
    if type(value) is not list:
        raise ValueError("expected remote flow binding is missing")
    links: dict[tuple[int, int, str], list[int]] = {}
    endpoint_ranks: set[int] = set()
    for flow in value:
        if type(flow) is not dict or set(flow) != {
            "id", "source_rank", "destination_rank", "logical_bytes",
        }:
            raise ValueError("expected remote flow row is malformed")
        flow_id = flow["id"]
        source = flow["source_rank"]
        destination = flow["destination_rank"]
        logical_bytes = flow["logical_bytes"]
        if (type(flow_id) is not str or not flow_id
                or type(source) is not int
                or type(destination) is not int
                or type(logical_bytes) is not int
                or not 0 <= source < rank_count
                or not 0 <= destination < rank_count
                or source == destination or logical_bytes <= 0):
            raise ValueError("expected remote flow row is invalid")
        endpoint_ranks.update((source, destination))
        packets = (
            logical_bytes + _PACKET_PAYLOAD_BYTES - 1
        ) // _PACKET_PAYLOAD_BYTES
        current = source
        source_x, source_y = source % columns, source // columns
        destination_x, destination_y = (
            destination % columns, destination // columns
        )
        while source_x != destination_x:
            step = 1 if source_x < destination_x else -1
            next_rank = current + step
            direction = "E" if step > 0 else "W"
            counts = links.setdefault(
                (current, next_rank, direction), [0, 0]
            )
            counts[0] += 1
            counts[1] += packets
            current = next_rank
            source_x += step
        while source_y != destination_y:
            step = 1 if source_y < destination_y else -1
            next_rank = current + step * columns
            direction = "N" if step > 0 else "S"
            counts = links.setdefault(
                (current, next_rank, direction), [0, 0]
            )
            counts[0] += 1
            counts[1] += packets
            current = next_rank
            source_y += step
        if current != destination:
            raise ValueError("expected remote flow failed X-first routing")
    rows_out = [
        {
            "source_die": source,
            "destination_die": destination,
            "direction": direction,
            "request_hops": counts[0],
            "packet_hops": counts[1],
        }
        for (source, destination, direction), counts in sorted(links.items())
    ]
    return rows_out, sorted(endpoint_ranks)


def select_shapes(
    shapes: tuple[str, ...], *, shard_index: int, shard_count: int,
) -> tuple[str, ...]:
    """Return a deterministic, disjoint canonical shard."""
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if len(shapes) != len(set(shapes)):
        raise ValueError("shapes must be unique")
    for shape in shapes:
        _shape(shape)
    return tuple(
        shape for index, shape in enumerate(shapes)
        if index % shard_count == shard_index
    )


def matrix_binding(
    args: argparse.Namespace, shapes: tuple[str, ...],
) -> dict[str, object]:
    """Bind a shard to exact driver, runner, tools, config, and order."""
    paths = {
        name: getattr(args, name).resolve()
        for name in ("finalizer", "resolver", "npusim", "simulation")
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("all native tools and simulation must be files")
    runner = (
        Path(__file__).resolve().parent
        / "run_moe_full_model_sequence_runtime_canary.py"
    )
    return {
        "schema_version": "moe-full-model-native-matrix-binding-v1",
        "driver_sha256": _sha(Path(__file__).resolve()),
        "runner_sha256": _sha(runner),
        "shapes": list(shapes),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "tool_sha256": {
            name: _sha(path) for name, path in sorted(paths.items())
        },
    }


def _require_matching_matrix_binding(
    path: Path, expected: dict[str, object],
) -> None:
    if json.loads(json.dumps(expected)) != expected:
        raise ValueError("computed matrix binding is not JSON-native")
    if _json(path) != expected:
        raise ValueError("matrix resume binding drifted")


def _expected_artifact_names() -> set[str]:
    result = {"hardware.json", "mapping.spec"}
    for index in range(3):
        result.update({
            f"segment_{index}.linked.json",
            f"segment_{index}.npup",
            f"segment_{index}.finalizer.json",
            f"segment_{index}.program_io.json",
            f"segment_{index}.resolver.stdout.txt",
        })
    return result


def _validate_digest_map(value: object, path: str) -> dict[str, str]:
    if type(value) is not dict or any(
        type(name) is not str
        or type(digest) is not str
        or _DIGEST.fullmatch(digest) is None
        for name, digest in value.items()
    ):
        raise ValueError(f"{path} is not a SHA-256 map")
    return value


def _audit_current_sources(
    digests: dict[str, str], path: str,
) -> None:
    """Fail closed when a bound repository source no longer matches disk."""
    for name, digest in digests.items():
        source = (_ROOT / name).resolve()
        try:
            relative = source.relative_to(_ROOT)
        except ValueError as error:
            raise ValueError(f"{path} escapes the repository: {name}") from error
        if relative.as_posix() != name or not source.is_file():
            raise ValueError(f"{path} is not a current repository file: {name}")
        if _sha(source) != digest:
            raise ValueError(f"{path} drifted from the current worktree: {name}")


def audit_fresh(directory: Path, shape: str) -> dict[str, object]:
    """Re-open all bound artifacts and independently audit native evidence."""
    rows, columns = _shape(shape)
    rank_count = rows * columns
    receipt_path = directory / "compiled_receipt.json"
    binding_path = directory / "source_tool_binding.json"
    receipt = _json(receipt_path)
    binding = _json(binding_path)
    runtime_path = directory / "npusim.stdout.txt"
    if not runtime_path.is_file():
        raise ValueError("native runtime log is missing")
    runtime = runtime_path.read_text(encoding="utf-8")
    if (receipt.get("schema_version") != "moe-full-model-runtime-receipt-v1"
            or binding.get("schema_version")
            != "moe-full-model-source-tool-binding-v1"
            or receipt.get("runtime_status") != "verified"
            or binding.get("runtime_status") != "verified"):
        raise ValueError("MoE runtime receipt or binding is not verified")
    if (receipt.get("mesh") != shape
            or receipt.get("sequence_digest") != binding.get("sequence_digest")
            or receipt.get("workload_case_id")
            != binding.get("workload_case_id")
            or receipt.get("source_request_sha256")
            != binding.get("source_request_sha256")):
        raise ValueError("mesh, request, or sequence binding drifted")
    if receipt.get("source_tool_binding_sha256") != _sha(binding_path):
        raise ValueError("receipt does not bind source_tool_binding.json")
    if receipt.get("runtime_log_sha256") != _sha(runtime_path):
        raise ValueError("receipt does not bind the native runtime log")

    active = receipt.get("active_die_ids")
    if active != list(range(rank_count)):
        raise ValueError("active Die IDs do not exactly cover the physical mesh")
    streams = receipt.get("compiled_core_die_ids")
    if streams != [active, active, active]:
        raise ValueError("three executable segments do not cover every rank")
    grid = receipt.get("frontend_core_grid")
    if (grid != receipt.get("native_core_grid")
            or not isinstance(grid, list) or len(grid) != 2
            or any(type(side) is not int or side <= 0 for side in grid)):
        raise ValueError("frontend/native core grids disagree")
    stride = grid[0] * grid[1]
    if (receipt.get("frontend_cores_per_die") != stride
            or receipt.get("native_cores_per_die") != stride):
        raise ValueError("frontend/native core stride disagrees")

    hardware = _json(directory / "hardware.json")
    if ((hardware.get("x"), hardware.get("y")) != tuple(grid)
            or hardware.get("die") != {"x": columns, "y": rows}):
        raise ValueError("native hardware grid differs from the receipt")
    system = hardware.get("memory_system", {})
    stacks = system.get("hbm_stacks", [])
    ranges = system.get("address_policy", {}).get("home_ranges", [])
    if ([item.get("compute_die_id") for item in stacks]
            != list(range(rank_count))
            or [item.get("die_id") for item in ranges]
            != list(range(rank_count))):
        raise ValueError("native HBM ownership does not cover every rank")

    core_rows = binding.get("executable_core_bindings")
    expected_core_rows = [
        {"rank": rank, "runtime_core_id": rank * stride}
        for rank in range(rank_count)
    ]
    if core_rows != expected_core_rows:
        raise ValueError("executable core bindings disagree with native stride")
    expected_cores = {item["runtime_core_id"] for item in expected_core_rows}
    recomputed_links, endpoint_ranks = _recompute_flow_evidence(
        binding.get("expected_remote_flows"), rows=rows, columns=columns
    )
    rank_to_core = {
        item["rank"]: item["runtime_core_id"] for item in core_rows
    }
    recomputed_p2p_cores = sorted(
        rank_to_core[rank] for rank in endpoint_ranks
    )
    bound_p2p_cores = binding.get("expected_p2p_core_ids")
    if (type(bound_p2p_cores) is not list
            or any(type(core) is not int for core in bound_p2p_cores)
            or bound_p2p_cores != recomputed_p2p_cores):
        raise ValueError(
            "bound P2P endpoint cores differ from signed remote flows"
        )
    memory = tuple(
        tuple(int(part) for part in match.groups())
        for match in _MEMORY.finditer(runtime)
    )
    if (len(memory) != rank_count
            or {row[0] for row in memory} != expected_cores
            or any(issued <= 0 or completed != issued or lsu or dte
                   for _, issued, completed, lsu, dte in memory)):
        raise ValueError("PROGRAM_MEMORY does not close every executable core")
    p2p = tuple(
        (int(core), int(residual))
        for core, residual in re.findall(
            r"\[P5 P2P DRAIN\] core=(\d+) residual=(\d+)", runtime
        )
    )
    expected_p2p_cores = set(recomputed_p2p_cores)
    if (len(p2p) != len(expected_p2p_cores)
            or {core for core, _ in p2p} != expected_p2p_cores
            or any(residual for _, residual in p2p)):
        raise ValueError("P2P drain does not close every transport core")

    artifacts = _validate_digest_map(
        binding.get("artifact_files_sha256"),
        "artifact_files_sha256",
    )
    if set(artifacts) != _expected_artifact_names():
        raise ValueError("bound artifact file set is incomplete or has extras")
    for name, digest in artifacts.items():
        path = directory / name
        if not path.is_file() or _sha(path) != digest:
            raise ValueError(f"bound artifact bytes drifted: {name}")

    resolver_evidence = []
    for index in range(3):
        manifest = _json(directory / f"segment_{index}.linked.json")
        report = _json(directory / f"segment_{index}.finalizer.json")
        sidecar = _json(directory / f"segment_{index}.program_io.json")
        artifact_sha = _sha(directory / f"segment_{index}.npup")
        manifest_sha = _sha(directory / f"segment_{index}.linked.json")
        if (report.get("artifact_sha256") != artifact_sha
                or report.get("linked_manifest_id") != manifest.get("id")
                or report.get("linked_manifest_digest") != manifest_sha
                or sidecar.get("program_artifact_sha256") != artifact_sha
                or sidecar.get("source_linked_manifest_id") != manifest.get("id")
                or sidecar.get("source_linked_manifest_digest") != manifest_sha):
            raise ValueError(f"segment {index} artifact identity closure failed")
        resolved = (
            directory / f"segment_{index}.resolver.stdout.txt"
        ).read_text(encoding="utf-8")
        matches = _RESOLVER.findall(resolved)
        expected_io = (
            sidecar.get("id"),
            str(len(sidecar.get("initializations", []))),
            str(len(sidecar.get("output_probes", []))),
        )
        if matches != [expected_io] or int(matches[0][1]) <= 0:
            raise ValueError(f"segment {index} resolver evidence is incomplete")
        resolver_evidence.append(matches[0])

    segments = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)",
        runtime,
    )
    probes = re.findall(
        r"\[DENSE_SEQUENCE_PROGRAM_IO\] index=(\d+) probes=(\d+) pass=(\d+)",
        runtime,
    )
    drains = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",
        runtime,
    )
    kv = tuple(
        (int(index), int(size), digest, int(passed))
        for index, size, digest, passed in _KV.findall(runtime)
    )
    if (segments != [("0", "0"), ("1", "0"), ("2", "1")]
            or probes != [("0", "1", "1"), ("1", "1", "1"),
                          ("2", "1", "1")]
            or drains != [("3", "1")]
            or len(kv) != 3
            or tuple(item[0] for item in kv) != (0, 1, 2)
            or any(item[3] != 1 for item in kv)):
        raise ValueError("Prefill/Decode/Decode runtime markers are incomplete")
    kv_bytes = tuple(item[1] for item in kv)
    if list(kv_bytes) != binding.get("kv_boundaries_bytes"):
        raise ValueError("native KV boundaries differ from their binding")
    makespans = tuple(int(value) for value in _MAKESPAN.findall(runtime))
    if len(makespans) != 1 or makespans[0] <= 0:
        raise ValueError("native makespan is missing or duplicated")

    expected_links = binding.get("expected_d2d_links")
    if expected_links != recomputed_links:
        raise ValueError(
            "expected D2D links differ from signed remote flows"
        )
    actual_links = []
    for match in _LINK.finditer(runtime):
        (index, source, destination, direction,
         req_in, req_out, ack_in, ack_out, data_in, data_out) = match.groups()
        source, destination = int(source), int(destination)
        sr, sc = divmod(source, columns)
        tr, tc = divmod(destination, columns)
        expected_direction = (
            "E" if sr == tr and tc == sc + 1 else
            "W" if sr == tr and tc == sc - 1 else
            "N" if tr == sr + 1 and tc == sc else
            "S" if tr == sr - 1 and tc == sc else None
        )
        counts = tuple(map(
            int, (req_in, req_out, ack_in, ack_out, data_in, data_out)
        ))
        if direction != expected_direction:
            raise ValueError("native D2D link is nonadjacent or misdirected")
        actual_links.append({
            "source_die": source,
            "destination_die": destination,
            "direction": direction,
            "request_hops": counts[0],
            "packet_hops": counts[4],
        })
        requests, packets = counts[0], counts[4]
        if counts != (
            requests, requests, 2 * requests, 2 * requests,
            packets, packets,
        ):
            raise ValueError("native D2D link counters are not closed")
    actual_links.sort(
        key=lambda item: (
            item["source_die"], item["destination_die"], item["direction"]
        )
    )
    expected_links = sorted(
        expected_links,
        key=lambda item: (
            item["source_die"], item["destination_die"], item["direction"]
        ),
    )
    if actual_links != expected_links:
        raise ValueError("native D2D links differ from all expected flow hops")
    request_hops = sum(item["request_hops"] for item in expected_links)
    packet_hops = sum(item["packet_hops"] for item in expected_links)
    typed = _D2D_TYPE.findall(runtime)
    data = _D2D.findall(runtime)
    if (typed != [tuple(map(str, (
                request_hops, request_hops, 2 * request_hops,
                2 * request_hops, packet_hops, packet_hops,
            )))]
            or data != [(str(packet_hops), str(packet_hops))]):
        raise ValueError("aggregate D2D counters differ from expected flow hops")

    source_tool = binding.get("source_tool_at_entry")
    if type(source_tool) is not dict:
        raise ValueError("source/tool entry snapshot is absent")
    imported = _validate_digest_map(
        source_tool.get("imported_python_sha256"), "imported Python"
    )
    tools = _validate_digest_map(source_tool.get("tool_sha256"), "native tools")
    if set(tools) != {"finalizer", "resolver", "npusim", "simulation"}:
        raise ValueError("native tool binding is incomplete")
    execution = binding.get("npusim_execution")
    if (type(execution) is not dict
            or set(execution) != {"executable", "cwd"}
            or any(type(execution.get(name)) is not str
                   for name in ("executable", "cwd"))):
        raise ValueError("NpuSim execution binding is incomplete")
    executable = Path(execution["executable"])
    cwd = Path(execution["cwd"])
    if (not executable.is_absolute() or not cwd.is_absolute()
            or executable.resolve() != executable
            or cwd.resolve() != cwd
            or executable.parent != cwd
            or not executable.is_file()
            or _sha(executable) != tools["npusim"]):
        raise ValueError("NpuSim executable or cwd drifted")
    additional = _validate_digest_map(
        binding.get("additional_imported_python_sha256"),
        "additional imported Python",
    )
    runner_key = (
        "llm/test/frontend/integration/"
        "run_moe_full_model_sequence_runtime_canary.py"
    )
    if runner_key not in imported:
        raise ValueError("runner source is absent from the source binding")
    overlap = set(imported) & set(additional)
    if any(imported[name] != additional[name] for name in overlap):
        raise ValueError("entry and additional source bindings disagree")
    _audit_current_sources(imported, "imported Python")
    _audit_current_sources(additional, "additional imported Python")

    return {
        "shape": shape,
        "sequence_digest": receipt["sequence_digest"],
        "source_request_sha256": receipt["source_request_sha256"],
        "workload_case_id": receipt["workload_case_id"],
        "core_grid": tuple(grid),
        "core_bindings": tuple(
            (item["rank"], item["runtime_core_id"]) for item in core_rows
        ),
        "memory": memory,
        "p2p_core_ids": tuple(sorted(expected_p2p_cores)),
        "kv_bytes": kv_bytes,
        "kv_digests": tuple(item[2] for item in kv),
        "makespan_cycles": makespans[0],
        "d2d_links": tuple(
            (
                item["source_die"], item["destination_die"],
                item["direction"], item["request_hops"], item["packet_hops"],
            )
            for item in actual_links
        ),
        "resolver_evidence": tuple(resolver_evidence),
        "artifact_files_sha256": artifacts,
        "source_tool_at_entry": source_tool,
        "npusim_execution": execution,
        "additional_imported_python_sha256": additional,
        "source_tool_binding_sha256": _sha(binding_path),
        "runtime_log_sha256": _sha(runtime_path),
        "phase_wall_seconds": receipt.get("phase_wall_seconds"),
        "total_wall_seconds": receipt.get("total_wall_seconds"),
        "python_peak_rss_kib": receipt.get("python_peak_rss_kib"),
        "children_max_rss_kib": receipt.get("children_max_rss_kib"),
        "segment_metrics": receipt.get("segment_metrics"),
    }


def compare_fresh(first: dict[str, object], second: dict[str, object]) -> None:
    varying = {
        "runtime_log_sha256",
        "phase_wall_seconds",
        "total_wall_seconds",
        "python_peak_rss_kib",
        "children_max_rss_kib",
        "segment_metrics",
    }
    stable_first = {
        key: value for key, value in first.items() if key not in varying
    }
    stable_second = {
        key: value for key, value in second.items() if key not in varying
    }
    if stable_first != stable_second:
        raise ValueError(
            "independent MoE executions or source/tool/artifact bindings differ"
        )


def audit_cached_case(case_root: Path, shape: str) -> dict[str, object]:
    """Re-open both fresh directories and the persisted evidence before resume."""
    evidence_path = case_root / "case_evidence.json"
    if not evidence_path.is_file():
        raise ValueError(f"cached {shape} has no case_evidence.json")
    evidence = _json(evidence_path)
    observed = tuple(
        audit_fresh(case_root / f"fresh{fresh}", shape)
        for fresh in (0, 1)
    )
    compare_fresh(*observed)
    expected = json.loads(json.dumps({
        "shape": shape,
        "fresh": list(observed),
    }))
    if evidence != expected:
        raise ValueError(f"cached {shape} evidence bytes or semantics drifted")
    return evidence


def run(args: argparse.Namespace) -> None:
    requested = (
        RELEASE_SHAPES if args.all_release_shapes else tuple(args.shapes)
    )
    shapes = select_shapes(
        requested,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if not shapes:
        raise ValueError("selected matrix shard is empty")
    root = args.output_root.resolve()
    binding = matrix_binding(args, shapes)
    binding_path = root / "matrix_binding.json"
    if root.exists():
        if not args.resume or not binding_path.is_file():
            raise ValueError(
                "matrix output root already exists; use --resume with its binding"
            )
        _require_matching_matrix_binding(binding_path, binding)
    else:
        root.mkdir(parents=True)
        binding_path.write_text(
            json.dumps(binding, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    environment = os.environ.copy()
    completed_shapes: list[str] = []
    for shape in shapes:
        case_root = root / shape
        if case_root.exists():
            audit_cached_case(case_root, shape)
            completed_shapes.append(shape)
            print(f"MoE full-model matrix RESUME {shape} verified", flush=True)
            continue
        case_root.mkdir()
        observations = []
        for fresh in (0, 1):
            directory = case_root / f"fresh{fresh}"
            command = (
                sys.executable,
                "-m",
                "llm.test.frontend.integration."
                "run_moe_full_model_sequence_runtime_canary",
                "--mesh-size", shape,
                "--output", str(directory),
                "--finalizer", str(args.finalizer.resolve()),
                "--resolver", str(args.resolver.resolve()),
                "--npusim", str(args.npusim.resolve()),
                "--simulation", str(args.simulation.resolve()),
                "--timeout", str(args.native_timeout),
            )
            completed = subprocess.run(
                command,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.process_timeout,
                check=False,
            )
            (case_root / f"fresh{fresh}.runner.stdout.txt").write_text(
                completed.stdout, encoding="utf-8"
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{shape} fresh{fresh} failed with exit "
                    f"{completed.returncode}; see {case_root}"
                )
            observations.append(audit_fresh(directory, shape))
        compare_fresh(*observations)
        evidence = {"shape": shape, "fresh": observations}
        (case_root / "case_evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        audit_cached_case(case_root, shape)
        completed_shapes.append(shape)
        print(f"MoE full-model matrix PASS {shape} two fresh", flush=True)

    matrix_receipt = {
        "schema_version": "moe-full-model-native-matrix-receipt-v1",
        "binding_sha256": _sha(binding_path),
        "shapes": shapes,
        "completed_shapes": completed_shapes,
        "status": (
            "verified" if tuple(completed_shapes) == shapes else "incomplete"
        ),
        "cases": [
            shape + "/case_evidence.json" for shape in completed_shapes
        ],
    }
    (root / "matrix_receipt.json").write_text(
        json.dumps(matrix_receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--shapes", nargs="+", default=M1_SHAPES)
    selection.add_argument("--all-release-shapes", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--native-timeout", type=int, default=1200)
    parser.add_argument("--process-timeout", type=int, default=1800)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
