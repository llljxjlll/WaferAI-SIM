"""Two independent full TP6 Dense inference low-HBM materializations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

_MEMORY = re.compile(r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+)")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active_for_mesh(mesh_size: str) -> tuple[int, int, list[int]]:
    try:
        rows, columns = map(int, mesh_size.split("x"))
    except (TypeError, ValueError):
        raise ValueError("TP6 mesh must be canonical rowsxcolumns") from None
    if (mesh_size != f"{rows}x{columns}" or not 1 <= rows <= 10
            or not 1 <= columns <= 10 or rows * columns < 6):
        raise ValueError("TP6 mesh requires 6..100 physical Dies in 1..10 rectangle")
    count = rows * columns
    active = ([0, 1, 2, 6, 7, 8] if (rows, columns) == (3, 3)
              else [index * (count - 1) // 5 for index in range(6)])
    return rows, columns, active


def audit_full_fresh(directory: Path, mesh_size: str) -> dict[str, object]:
    rows, columns, active = _active_for_mesh(mesh_size)
    report = json.loads((directory / "dense-inference-tp6-paged-runtime-evidence.json").read_text(encoding="utf-8"))
    hardware = json.loads((directory / "hardware.json").read_text(encoding="utf-8"))
    if report.get("resident_rejection_code") != "memory_capacity_exceeded":
        raise ValueError("same-model resident-only 18KiB negative is missing")
    if (report.get("logical_hbm_capacity_bytes_per_die") != 18432 or
            report.get("native_aligned_window_bytes_per_die") != 18432 or
            report.get("native_stack_interleave_bytes") != 2048 or
            report.get("native_hbm_home_bases") !=
            [die * 1073741824 for die in range(rows * columns)] or
            report.get("observed_paged_peak_end_bytes_per_die") != 17088):
        raise ValueError("logical/native HBM boundary or paged peak disagrees")
    if (report.get("mesh_rows") != rows or report.get("mesh_columns") != columns or
            report.get("frontend_core_grid") != [2, 2] or
            report.get("native_core_grid") != [2, 2]):
        raise ValueError("frontend/native TP6 core geometry disagrees")
    if ((hardware.get("x"), hardware.get("y")) != (2, 2) or
            hardware.get("die") != {"x": columns, "y": rows} or
            _sha(directory / "hardware.json") != report.get("hardware_sha256")):
        raise ValueError("physical native hardware bytes or geometry disagree")
    if (not report.get("model_digest") or
            report.get("resident_logical_graph_digest") !=
            report.get("offload_logical_graph_digest")):
        raise ValueError("resident/offload useful full-model graph differs")
    if (report.get("kv_page_bytes") != [0, 13824, 16128, 18432] or
            report.get("kv_versions") != [0, 1, 2, 3] or
            report.get("external_dma_events") != 390 or
            report.get("external_kv_probes") != 72 or
            report.get("external_read_bytes") != 762048 or
            report.get("external_write_bytes") != 48384 or
            report.get("functional") is not False):
        raise ValueError("native external TP6 DMA/KV traffic or timing mode disagrees")
    if (report.get("used_runtime_cores") != [4 * die for die in active] or
            report.get("physical_die_ids") != active):
        raise ValueError("native program did not use six active physical Dies")
    packets = report.get("d2d_packets")
    if (not isinstance(packets, list) or len(packets) != 2 or
            packets[0] <= 0 or packets[0] != packets[1]):
        raise ValueError("physical D2D packet evidence missing")
    if (report.get("paired_fresh_runs") != 2 or
            not isinstance(report.get("source_tool_binding_sha256"), str) or
            report.get("source_tool_at_exit", {}).get("tool_sha256", {}).get("npusim") !=
            report.get("npusim_sha256")):
        raise ValueError("native replay or source/tool closure is missing")
    for index in range(3):
        paths = [directory / f"segment_{index}.{suffix}" for suffix in
                 ("linked.json", "npup", "program_io.json", "resolver.stdout.txt")]
        if not all(path.is_file() and path.stat().st_size for path in paths):
            raise ValueError(f"segment {index} finalizer/ProgramIO closure missing")
    for native in (0, 1):
        stdout = (directory / f"fresh_{native}" / "npusim.stdout.txt").read_text(encoding="utf-8")
        cores = {int(core) for core, count in _MEMORY.findall(stdout) if int(count) > 0}
        if cores != {4 * die for die in active}:
            raise ValueError("physical TP6 core0 streams are incomplete")
        if "[DENSE_INFERENCE_PAGED_DMA_DRAIN] events=390" not in stdout:
            raise ValueError("native TP6 external DMA did not drain")
    return {
        "report": report,
        "hardware_sha256": _sha(directory / "hardware.json"),
        "sidecar_sha256": _sha(directory / "dense_inference_paged_runtime.json"),
        "linked_sha256": [_sha(directory / f"segment_{i}.linked.json") for i in range(3)],
        "npup_sha256": [_sha(directory / f"segment_{i}.npup") for i in range(3)],
        "program_io_sha256": [_sha(directory / f"segment_{i}.program_io.json") for i in range(3)],
    }


def run(args: argparse.Namespace) -> None:
    _active_for_mesh(args.mesh_size)
    root = args.output_root.resolve()
    if root.exists():
        raise ValueError("full-fresh output root must be a new path")
    root.mkdir(parents=True)
    evidence = []
    for index in (0, 1):
        directory = root / f"full_fresh_{index}"
        command = (
            sys.executable, "-m",
            "llm.test.frontend.integration.run_dense_inference_tp6_paged_offload_runtime_canary",
            "--output", str(directory), "--mesh-size", args.mesh_size,
            "--npusim", str(args.npusim.resolve()),
            "--finalizer", str(args.finalizer.resolve()),
            "--resolver", str(args.resolver.resolve()),
            "--simulation", str(args.simulation.resolve()),
            "--timeout", str(args.native_timeout),
        )
        completed = subprocess.run(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   timeout=args.process_timeout, check=False)
        (root / f"full_fresh_{index}.runner.stdout.txt").write_text(
            completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"independent TP6 full materialization {index} failed with exit {completed.returncode}")
        evidence.append(audit_full_fresh(directory, args.mesh_size))
    if evidence[0] != evidence[1]:
        raise ValueError("two independently compiled source-bound TP6 offload materializations disagree")
    (root / "full_fresh_evidence.json").write_text(json.dumps({
        "status": "verified", "mesh_size": args.mesh_size,
        "independent_full_materializations": 2, "case": evidence[0],
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Dense TP6 {args.mesh_size} inference 18KiB external offload PASS two independent full materializations", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mesh-size", default="2x3",
                        help="canonical 1..10 rectangle with at least six Dies")
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--native-timeout", type=int, default=900)
    parser.add_argument("--process-timeout", type=int, default=3600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
