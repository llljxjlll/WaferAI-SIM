"""Two-fresh, physical-Die audited full Dense inference mesh matrix.

The workload and native execution are owned by run_dense_sequence_runtime_canary.
This driver only schedules canonical shapes and refuses a release case when the
observable physical Die placement or the independent-run evidence disagrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


M1_SHAPES = ("1x1", "1x4", "4x1", "2x2", "2x3", "3x2", "3x3", "10x10")
RELEASE_SHAPES = tuple(f"{rows}x{columns}" for rows in range(1, 11) for columns in range(1, 11))
_MESH = re.compile(r"([1-9][0-9]*)x([1-9][0-9]*)\Z")
_KV = re.compile(r"\[DENSE_SEQUENCE_KV\] index=(\d+) bytes=(\d+) digest=([0-9a-f]{64}) pass=(\d+)")
_MEMORY = re.compile(r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+)")
_LINK = re.compile(r"\[D2D_LINK\][^\n]*?die(\d+)->die(\d+) dir=([EWNS])[^\n]*?data_in=(\d+) data_out=(\d+)")
_D2D = re.compile(r"\[D2D_DATA\] in_pkts=(\d+) out_pkts=(\d+)")
_MAKESPAN = re.compile(r"\[SIM_RESULT\] makespan_cycles=(\d+)")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shape(value: str) -> tuple[int, int]:
    match = _MESH.fullmatch(value)
    if match is None:
        raise ValueError(f"noncanonical mesh: {value}")
    rows, columns = (int(part) for part in match.groups())
    if not 1 <= rows <= 10 or not 1 <= columns <= 10:
        raise ValueError(f"outside 1..10 release envelope: {value}")
    return rows, columns


def select_shapes(
    shapes: tuple[str, ...], *, shard_index: int, shard_count: int,
) -> tuple[str, ...]:
    """Return a deterministic, disjoint canonical shard without reordering."""
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    if len(shapes) != len(set(shapes)):
        raise ValueError("shapes must be unique")
    for shape in shapes:
        _shape(shape)
    return tuple(shape for index, shape in enumerate(shapes) if index % shard_count == shard_index)


def matrix_binding(args: argparse.Namespace, shapes: tuple[str, ...]) -> dict[str, object]:
    """Bind a matrix shard to exact driver, native tools, simulation and order."""
    paths = {
        name: getattr(args, name).resolve()
        for name in ("finalizer", "resolver", "npusim", "simulation")
    }
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("all native tools and simulation must be files")
    dram_config = Path(__file__).resolve().parents[4] / "DRAMSys/configs/hbm2-example.json"
    if not dram_config.is_file():
        raise ValueError("bound DRAMSys HBM profile is missing")
    all_dies_scaled = bool(getattr(args, "all_dies_scaled", False))
    all_dies_compact = bool(getattr(args, "all_dies_compact", False))
    if all_dies_scaled and all_dies_compact:
        raise ValueError("select one all-Die profile")
    return {
        "schema_version": ("dense-native-mesh-matrix-binding-v5"
                           if all_dies_compact else
                           "dense-native-mesh-matrix-binding-v4"
                           if all_dies_scaled else
                           "dense-native-mesh-matrix-binding-v3"),
        **({"profile": "all_dies_compact"} if all_dies_compact else
           {"profile": "all_dies_scaled"} if all_dies_scaled else {}),
        "dram_config_sha256": _sha(dram_config),
        "driver_sha256": _sha(Path(__file__).resolve()),
        "runner_sha256": _sha(Path(__file__).resolve().parent / "run_dense_sequence_runtime_canary.py"),
        "shapes": list(shapes),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "tool_sha256": {name: _sha(path) for name, path in sorted(paths.items())},
    }


def audit_cached_case(
    case_root: Path, shape: str, *, all_dies_scaled: bool = False,
) -> dict[str, object]:
    """Re-open every byte needed by a cached two-fresh case before resume."""
    evidence_path = case_root / "case_evidence.json"
    if not evidence_path.is_file():
        raise ValueError(f"cached {shape} has no case_evidence.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    observed = tuple(audit_fresh(case_root / f"fresh{fresh}", shape,
                                 all_dies_scaled=all_dies_scaled)
                     for fresh in (0, 1))
    compare_fresh(*observed)
    expected = json.loads(json.dumps({"shape": shape, "fresh": list(observed)}))
    if evidence != expected:
        raise ValueError(f"cached {shape} evidence bytes or semantics drifted")
    return evidence


def bind_case_dram_config(case_root: Path) -> None:
    """Resolve the simulator's ../DRAMSys config against the frozen source tree."""
    expected = Path(__file__).resolve().parents[4] / "DRAMSys"
    path = case_root / "DRAMSys"
    if path.is_symlink():
        if path.resolve() != expected.resolve():
            raise ValueError("DRAMSys case resource points outside bound source")
    elif path.exists():
        raise ValueError("DRAMSys case resource is not a bound source link")
    else:
        path.symlink_to(expected, target_is_directory=True)


def audit_partial_case(
    case_root: Path, shape: str, *, all_dies_scaled: bool = False,
) -> list[dict[str, object]]:
    """Resume only a contiguous prefix of complete, reopened native fresh runs."""
    if (case_root / "case_evidence.json").exists():
        audit_cached_case(case_root, shape, all_dies_scaled=all_dies_scaled)
        return [audit_fresh(case_root / f"fresh{index}", shape,
                            all_dies_scaled=all_dies_scaled)
                for index in (0, 1)]
    if (case_root / "fresh1").exists() and not (case_root / "fresh0").exists():
        raise ValueError(f"partial {shape} has fresh1 without fresh0")
    observations = []
    for index in (0, 1):
        directory = case_root / f"fresh{index}"
        if not directory.exists():
            break
        observations.append(audit_fresh(
            directory, shape, all_dies_scaled=all_dies_scaled))
    if len(observations) == 2:
        compare_fresh(*observations)
    return observations


def _mode(
    rows: int, columns: int, *, all_dies_scaled: bool = False,
    all_dies_compact: bool = False,
) -> tuple[str, ...]:
    if all_dies_compact:
        return ("--compact-scaled-all-dies",)
    if all_dies_scaled:
        return ("--scaled-all-dies",)
    count = rows * columns
    if count in (2, 3, 5):
        return ("--scaled-all-dies",)
    if count >= 6 and (rows, columns) not in ((2, 3), (3, 2), (3, 3)):
        return ("--fixed-global-tp6",)
    return ()


def audit_fresh(
    directory: Path, shape: str, *, all_dies_scaled: bool = False,
) -> dict[str, object]:
    """Audit native execution, full-sequence KV, and true physical placement."""
    rows, columns = _shape(shape)
    receipt = json.loads((directory / "compiled_receipt.json").read_text(encoding="utf-8"))
    binding = json.loads((directory / "source_tool_binding.json").read_text(encoding="utf-8"))
    text = (directory / "npusim.stdout.txt").read_text(encoding="utf-8")
    if receipt.get("runtime_status") != "verified" or binding.get("runtime_status") != "verified":
        raise ValueError("native runtime status is not verified")
    if receipt.get("mesh") != shape or receipt.get("sequence_digest") != binding.get("sequence_digest"):
        raise ValueError("mesh or sequence binding drifted")
    active = receipt.get("active_die_ids")
    if not isinstance(active, list) or not active or any(type(die) is not int or die < 0 or die >= rows * columns for die in active):
        raise ValueError("active Die IDs are invalid")
    if len(set(active)) != len(active):
        raise ValueError("active Die IDs repeat")
    if all_dies_scaled and active != list(range(rows * columns)):
        raise ValueError("scaled all-Die profile did not activate every physical Die")
    streams = receipt.get("compiled_core_die_ids")
    if not isinstance(streams, list) or len(streams) != 3 or any(sorted(part) != sorted(active) for part in streams):
        raise ValueError("three native segments do not cover every active Die")
    grid = receipt.get("frontend_core_grid")
    if grid != receipt.get("native_core_grid") or not isinstance(grid, list) or len(grid) != 2 or any(type(side) is not int or side <= 0 for side in grid):
        raise ValueError("frontend/native per-Die core grids disagree")
    stride = grid[0] * grid[1]
    hardware = json.loads((directory / "hardware.json").read_text(encoding="utf-8"))
    die_hardware = hardware.get("die", {})
    if (hardware.get("x"), hardware.get("y")) != tuple(grid) or (die_hardware.get("x"), die_hardware.get("y")) != (columns, rows):
        raise ValueError("native hardware core or physical Die mesh differs from bound receipt")
    if receipt.get("frontend_cores_per_die") != stride or receipt.get("native_cores_per_die") != stride:
        raise ValueError("frontend/native core stride disagrees with hardware")
    used_cores = tuple(sorted({int(core) for core, issued in _MEMORY.findall(text) if int(issued) > 0}))
    physical_dies = tuple(sorted({core // stride for core in used_cores}))
    if physical_dies != tuple(sorted(active)):
        raise ValueError(f"actual NpuSim physical Dies {physical_dies} differ from logical {active}")
    kv = tuple((int(index), int(size), digest, int(passed)) for index, size, digest, passed in _KV.findall(text))
    if len(kv) != 3 or tuple(index for index, *_ in kv) != (0, 1, 2) or any(passed != 1 for *_, passed in kv):
        raise ValueError("Prefill/Decode/Decode KV markers are incomplete")
    boundaries = tuple(size for _, size, _, _ in kv)
    if boundaries != tuple(binding.get("kv_boundaries_bytes", ()) or ()) or not (0 < boundaries[0] < boundaries[1] < boundaries[2]):
        raise ValueError("actual KV bytes do not match bound increasing state")
    makespans = tuple(int(value) for value in _MAKESPAN.findall(text))
    if len(makespans) != 1 or makespans[0] <= 0:
        raise ValueError("native makespan is missing or duplicated")
    native_d2d = tuple(int(value) for pair in _D2D.findall(text) for value in pair)
    if len(native_d2d) != 2 or native_d2d[0] != native_d2d[1]:
        raise ValueError("native D2D packet accounting is invalid")
    links = tuple((int(source), int(target), direction, int(data_in), int(data_out)) for source, target, direction, data_in, data_out in _LINK.findall(text))
    for source, target, direction, data_in, data_out in links:
        if source >= rows * columns or target >= rows * columns:
            raise ValueError("native D2D link lies outside physical mesh")
        sr, sc = divmod(source, columns)
        tr, tc = divmod(target, columns)
        expected = "E" if sr == tr and tc == sc + 1 else "W" if sr == tr and tc == sc - 1 else "N" if tr == sr + 1 and tc == sc else "S" if tr == sr - 1 and tc == sc else None
        if direction != expected or data_in != data_out:
            raise ValueError("native D2D link is nonadjacent or traffic disagrees")
    if len(active) > 1 and (not links or native_d2d[0] == 0 or not set(active).issubset({die for source, target, _, data_in, _ in links if data_in > 0 for die in (source, target)})):
        raise ValueError("active Dies did not participate in native D2D traffic")
    if len(active) == 1 and native_d2d[0] != 0:
        raise ValueError("single Die unexpectedly has D2D traffic")
    for key, file_prefix in (("linked_manifest_sha256", "linked.json"), ("npup_sha256", "npup"), ("program_io_sha256", "program_io.json")):
        digests = binding.get(key)
        if not isinstance(digests, list) or len(digests) != 3:
            raise ValueError(f"{key} binding is incomplete")
        for index, digest in enumerate(digests):
            file = directory / (f"segment_{index}.{file_prefix}")
            if not file.is_file() or _sha(file) != digest:
                raise ValueError(f"{key} bytes drifted for segment {index}")
    if _sha(directory / "hardware.json") != binding.get("hardware_sha256"):
        raise ValueError("hardware bytes drifted")
    return {
        "shape": shape,
        "active_die_ids": active,
        "physical_die_ids": physical_dies,
        "core_grid": grid,
        "used_cores": used_cores,
        "kv_bytes": boundaries,
        "kv_digests": tuple(item[2] for item in kv),
        "makespan_cycles": makespans[0],
        "d2d_packets": native_d2d,
        "d2d_links": links,
        "source_tool_binding_sha256": _sha(directory / "source_tool_binding.json"),
        "source_request_sha256": receipt.get("source_request_sha256"),
        "workload_case_id": receipt.get("workload_case_id"),
        "sequence_digest": receipt["sequence_digest"],
        "linked_manifest_sha256": binding["linked_manifest_sha256"],
        "npup_sha256": binding["npup_sha256"],
        "program_io_sha256": binding["program_io_sha256"],
        "hardware_sha256": binding["hardware_sha256"],
        "phase_wall_seconds": receipt.get("phase_wall_seconds"),
        "total_wall_seconds": receipt.get("total_wall_seconds"),
    }


def compare_fresh(first: dict[str, object], second: dict[str, object]) -> None:
    varying = {"phase_wall_seconds", "total_wall_seconds"}
    if {key: value for key, value in first.items() if key not in varying} != {key: value for key, value in second.items() if key not in varying}:
        raise ValueError("independent native executions or source/tool/artifact bindings differ")


def run(args: argparse.Namespace) -> None:
    requested = RELEASE_SHAPES if args.all_release_shapes else tuple(args.shapes)
    shapes = select_shapes(
        requested, shard_index=args.shard_index, shard_count=args.shard_count,
    )
    if not shapes:
        raise ValueError("selected matrix shard is empty")
    root = args.output_root.resolve()
    all_dies_compact = bool(getattr(args, "all_dies_compact", False))
    all_dies_scaled = bool(getattr(args, "all_dies_scaled", False)
                           or all_dies_compact)
    binding = matrix_binding(args, shapes)
    binding_path = root / "matrix_binding.json"
    if root.exists():
        if not args.resume or not binding_path.is_file():
            raise ValueError("matrix output root already exists; use --resume with its binding")
        cached_binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if cached_binding != binding:
            raise ValueError("matrix resume binding drifted")
    else:
        root.mkdir(parents=True)
        binding_path.write_text(
            json.dumps(binding, indent=2, sort_keys=True), encoding="utf-8",
        )
    environment = os.environ.copy()
    completed_shapes: list[str] = []
    for shape in shapes:
        rows, columns = _shape(shape)
        case_root = root / shape
        if case_root.exists():
            if (case_root / "case_evidence.json").exists():
                audit_cached_case(
                    case_root, shape, all_dies_scaled=all_dies_scaled)
                completed_shapes.append(shape)
                print(f"Dense full-sequence native mesh RESUME {shape} verified", flush=True)
                continue
            observations = audit_partial_case(
                case_root, shape, all_dies_scaled=all_dies_scaled)
        else:
            case_root.mkdir()
            observations = []
        bind_case_dram_config(case_root)
        for fresh in range(len(observations), 2):
            directory = case_root / f"fresh{fresh}"
            command = (
                sys.executable, "-m", "llm.test.frontend.integration.run_dense_sequence_runtime_canary",
                "--mesh-size", shape, "--output", str(directory),
                "--finalizer", str(args.finalizer.resolve()),
                "--resolver", str(args.resolver.resolve()),
                "--npusim", str(args.npusim.resolve()),
                "--simulation", str(args.simulation.resolve()),
                "--timeout", str(args.native_timeout),
                "--compile-timeout", str(args.compile_timeout),
                *_mode(rows, columns, all_dies_scaled=bool(args.all_dies_scaled),
                       all_dies_compact=all_dies_compact),
            )
            completed = subprocess.run(
                command, cwd=Path(__file__).resolve().parents[4],
                env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
                timeout=args.process_timeout, check=False,
            )
            (case_root / f"fresh{fresh}.runner.stdout.txt").write_text(
                completed.stdout, encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{shape} fresh{fresh} failed with exit {completed.returncode}; "
                    f"see {case_root}"
                )
            observations.append(audit_fresh(
            directory, shape, all_dies_scaled=all_dies_scaled))
        compare_fresh(*observations)
        evidence = {"shape": shape, "fresh": observations}
        (case_root / "case_evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8",
        )
        audit_cached_case(
            case_root, shape, all_dies_scaled=all_dies_scaled)
        completed_shapes.append(shape)
        print(f"Dense full-sequence native mesh PASS {shape} two fresh", flush=True)
    (root / "matrix_receipt.json").write_text(json.dumps({
        "binding_sha256": _sha(binding_path),
        "shapes": shapes,
        "completed_shapes": completed_shapes,
        "status": "verified" if tuple(completed_shapes) == shapes else "incomplete",
        "cases": [shape + "/case_evidence.json" for shape in completed_shapes],
    }, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--shapes", nargs="+", default=M1_SHAPES)
    selection.add_argument("--all-release-shapes", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument("--all-dies-scaled", action="store_true",
                         help="run the shape-scaled model on every physical Die")
    profile.add_argument("--all-dies-compact", action="store_true",
                         help="run TP=all Dies with one request per shape")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--native-timeout", type=int, default=1200)
    parser.add_argument("--compile-timeout", type=int, default=2400)
    parser.add_argument("--process-timeout", type=int, default=3600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
