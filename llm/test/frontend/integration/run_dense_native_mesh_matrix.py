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


def _mode(rows: int, columns: int) -> tuple[str, ...]:
    count = rows * columns
    if count in (2, 3, 5):
        return ("--scaled-all-dies",)
    if count >= 6 and (rows, columns) not in ((2, 3), (3, 2), (3, 3)):
        return ("--fixed-global-tp6",)
    return ()


def audit_fresh(directory: Path, shape: str) -> dict[str, object]:
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
    streams = receipt.get("compiled_core_die_ids")
    if not isinstance(streams, list) or len(streams) != 3 or any(sorted(part) != sorted(active) for part in streams):
        raise ValueError("three native segments do not cover every active Die")
    grid = receipt.get("frontend_core_grid")
    if grid != receipt.get("native_core_grid") or not isinstance(grid, list) or len(grid) != 2 or any(type(side) is not int or side <= 0 for side in grid):
        raise ValueError("frontend/native per-Die core grids disagree")
    stride = grid[0] * grid[1]
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
    shapes = tuple(args.shapes)
    if len(shapes) != len(set(shapes)) or any(_shape(shape) is None for shape in shapes):
        raise ValueError("shapes must be unique canonical release meshes")
    root = args.output_root.resolve()
    if root.exists():
        raise ValueError("matrix output root must be an empty new path")
    root.mkdir(parents=True)
    environment = os.environ.copy()
    for shape in shapes:
        rows, columns = _shape(shape)
        observations = []
        case_root = root / shape
        case_root.mkdir()
        for fresh in (0, 1):
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
                *_mode(rows, columns),
            )
            completed = subprocess.run(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.process_timeout, check=False)
            (case_root / f"fresh{fresh}.runner.stdout.txt").write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"{shape} fresh{fresh} failed with exit {completed.returncode}; see {case_root}")
            observations.append(audit_fresh(directory, shape))
        compare_fresh(*observations)
        (case_root / "case_evidence.json").write_text(json.dumps({"shape": shape, "fresh": observations}, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Dense full-sequence native mesh PASS {shape} two fresh", flush=True)
    (root / "matrix_receipt.json").write_text(json.dumps({"shapes": shapes, "status": "verified", "cases": [shape + "/case_evidence.json" for shape in shapes]}, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=M1_SHAPES)
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
