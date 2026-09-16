"""Two independent full materializations of the low-HBM TP4 2x2 Dense inference pair."""

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


def audit_full_fresh(directory: Path) -> dict[str, object]:
    report = json.loads((directory / "dense-inference-rect-paged-runtime-evidence.json").read_text(encoding="utf-8"))
    hardware = json.loads((directory / "hardware.json").read_text(encoding="utf-8"))
    if report.get("resident_rejection_code") != "memory_capacity_exceeded":
        raise ValueError("same-model resident-only low-HBM negative is missing")
    if (report.get("logical_hbm_capacity_bytes_per_die") != 12288 or
            report.get("native_aligned_window_bytes_per_die") != 12288 or
            report.get("native_stack_interleave_bytes") != 4096 or
            report.get("native_hbm_home_bases") !=
            [0, 1073741824, 2147483648, 3221225472] or
            report.get("observed_paged_peak_end_bytes_per_die") != 11328):
        raise ValueError("logical/native HBM boundary or observed peak disagrees")
    if report.get("frontend_core_grid") != [2, 2] or report.get("native_core_grid") != [2, 2]:
        raise ValueError("frontend/native low-HBM inference core geometry disagrees")
    if (hardware.get("x"), hardware.get("y")) != (2, 2) or hardware.get("die") != {"x": 2, "y": 2}:
        raise ValueError("actual native hardware core or physical Die geometry disagrees")
    if _sha(directory / "hardware.json") != report.get("hardware_sha256"):
        raise ValueError("native hardware bytes drifted from full-fresh report")
    if report.get("model_digest") is None or report.get("resident_logical_graph_digest") != report.get("offload_logical_graph_digest"):
        raise ValueError("resident/offload useful full-model graph differs")
    if report.get("kv_page_bytes") != [0, 4096, 5120, 6144] or report.get("kv_versions") != [0, 1, 2, 3]:
        raise ValueError("external KV versions or real bytes are incomplete")
    if (report.get("external_dma_events") != 260 or
            report.get("external_kv_probes") != 48 or
            report.get("external_read_bytes") != 334592 or
            report.get("external_write_bytes") != 15360 or
            report.get("functional") is not False):
        raise ValueError("real external transfer/probe byte contract or timing mode disagrees")
    if (report.get("used_runtime_cores") != [0, 4, 8, 12] or
            report.get("physical_die_ids") != [0, 1, 2, 3]):
        raise ValueError("actual runtime cores do not cover four physical Dies")
    d2d = report.get("d2d_packets")
    if (not isinstance(d2d, list) or len(d2d) != 2 or
            d2d[0] <= 0 or d2d[0] != d2d[1]):
        raise ValueError("native D2D packet evidence is missing or unbalanced")
    if report.get("paired_fresh_runs") != 2 or not isinstance(report.get("source_tool_binding_sha256"), str):
        raise ValueError("native replay or source/tool closure missing")
    if report.get("source_tool_at_exit", {}).get("tool_sha256", {}).get("npusim") != report.get("npusim_sha256"):
        raise ValueError("actual native tool bytes drifted from full-fresh report")
    for index in range(3):
        manifest = directory / f"segment_{index}.linked.json"
        artifact = directory / f"segment_{index}.npup"
        contract = directory / f"segment_{index}.program_io.json"
        resolver = directory / f"segment_{index}.resolver.stdout.txt"
        if not all(path.is_file() and path.stat().st_size for path in (manifest, artifact, contract, resolver)):
            raise ValueError(f"segment {index} production finalizer/resolver/ProgramIO closure missing")
    for native in (0, 1):
        path = directory / f"fresh_{native}" / "npusim.stdout.txt"
        stdout = path.read_text(encoding="utf-8")
        active_cores = {int(core) for core, count in _MEMORY.findall(stdout) if int(count) > 0}
        if active_cores != {0, 4, 8, 12}:
            raise ValueError("actual NpuSim low-HBM inference ran outside the four physical Die-local core0 streams")
        if "[DENSE_INFERENCE_PAGED_DMA_DRAIN] events=260" not in stdout:
            raise ValueError("native external DMA did not drain all useful full-model transfers")
    return {
        "report": report,
        "hardware_sha256": _sha(directory / "hardware.json"),
        "sidecar_sha256": _sha(directory / "dense_inference_paged_runtime.json"),
        "linked_sha256": [_sha(directory / f"segment_{i}.linked.json") for i in range(3)],
        "npup_sha256": [_sha(directory / f"segment_{i}.npup") for i in range(3)],
        "program_io_sha256": [_sha(directory / f"segment_{i}.program_io.json") for i in range(3)],
    }


def compare_full_fresh(first: dict[str, object], second: dict[str, object]) -> None:
    if first != second:
        raise ValueError("two independent full-model resident-negative/offload materializations disagree")


def run(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    if root.exists():
        raise ValueError("full-fresh output root must be a new empty path")
    root.mkdir(parents=True)
    evidence = []
    for index in (0, 1):
        directory = root / f"full_fresh_{index}"
        command = (
            sys.executable, "-m", "llm.test.frontend.integration.run_dense_inference_rect_paged_offload_runtime_canary",
            "--output", str(directory),
            "--npusim", str(args.npusim.resolve()),
            "--finalizer", str(args.finalizer.resolve()),
            "--resolver", str(args.resolver.resolve()),
            "--simulation", str(args.simulation.resolve()),
            "--timeout", str(args.native_timeout),
        )
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=args.process_timeout, check=False)
        (root / f"full_fresh_{index}.runner.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"independent full materialization {index} failed with exit {completed.returncode}")
        evidence.append(audit_full_fresh(directory))
    compare_full_fresh(*evidence)
    (root / "full_fresh_evidence.json").write_text(json.dumps({"status": "verified", "independent_full_materializations": 2, "case": evidence[0]}, indent=2, sort_keys=True), encoding="utf-8")
    print("Dense TP4 2x2 full inference low-HBM external offload PASS two independent full materializations", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--native-timeout", type=int, default=900)
    parser.add_argument("--process-timeout", type=int, default=3600)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
