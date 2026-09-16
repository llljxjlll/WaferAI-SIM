"""Two independent full MoE 1x2 low-HBM compile/finalize/native materializations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[4]
_SOURCE_FILES = (
    "llm/test/frontend/integration/run_moe_inference_paged_offload_runtime_canary.py",
    "llm/test/frontend/integration/run_moe_full_model_sequence_runtime_canary.py",
    "llm/test/frontend/integration/run_dense_sequence_runtime_canary.py",
    "llm/frontend/wafer_frontend/passes/moe_inference_paged_compile_sequence.py",
    "llm/frontend/wafer_frontend/passes/moe_inference_paged_runtime.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(args: argparse.Namespace) -> dict:
    return {
        "source_sha256": {name: _sha(_ROOT / name) for name in _SOURCE_FILES},
        "driver_sha256": _sha(Path(__file__).resolve()),
        "tool_sha256": {name: _sha(getattr(args, name).resolve()) for name in
                        ("npusim", "finalizer", "simulation")},
    }


def audit(directory: Path, binding: dict) -> dict:
    report = json.loads((directory / "moe-inference-paged-runtime-evidence.json").read_text())
    hardware_path = directory / "hardware.json"
    hardware = json.loads(hardware_path.read_text())
    if (report.get("resident_rejection_code") != "memory_capacity_exceeded" or
        report.get("mesh") != "1x2" or report.get("ep") != 2 or
        report.get("hbm_capacity_bytes_per_die") != 1024 or
        report.get("paired_fresh_runs") != 2 or
        report.get("actual_dma_events") != 89 or
        report.get("physical_external_read_bytes") != 4600 or
        report.get("physical_external_write_bytes") != 2880 or
        report.get("d2d_packets") != 12 or
        report.get("frontend_core_grid") != [2, 2] or
        report.get("native_core_grid") != [2, 2] or
        report.get("npusim_sha256") != binding["tool_sha256"]["npusim"] or
        report.get("finalizer_sha256") != binding["tool_sha256"]["finalizer"] or
        report.get("simulation_sha256") != binding["tool_sha256"]["simulation"] or
        report.get("hardware_sha256") != _sha(hardware_path) or
        report.get("functional") is not False):
        raise ValueError("full-model low-HBM rejection/DMA/physical tool evidence drifted")
    if (hardware.get("x"), hardware.get("y"), hardware.get("die")) != (2, 2, {"x": 2, "y": 1}):
        raise ValueError("frontend/native physical core and Die grids differ")
    artifact_hashes = {}
    for step in (0, 1, 2):
        for kind in ("source.linked.json", "source.npup", "linked.json",
                     "npup", "program_io.json"):
            path = directory / f"segment_{step}.{kind}"
            artifact_hashes[path.name] = _sha(path)
    for fresh in (0, 1):
        text = (directory / f"fresh_{fresh}" / "npusim.stdout.txt").read_text()
        if (re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", text) != ["9545"] or
            re.findall(r"\[D2D_DATA\] in_pkts=(\d+) out_pkts=(\d+)", text) != [("12", "12")] or
            {int(core): int(count) for core, count in
             re.findall(r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+)", text)} != {0: 71, 4: 18} or
            len(re.findall(r"\[MOE_INFERENCE_PAGED_DMA_EVENT\]", text)) != 89 or
            "[MOE_INFERENCE_PAGED_DMA_DRAIN] events=89" not in text or
            "[P5 P2P DRAIN] core=4 residual=0" not in text or
            len(re.findall(r"\[MOE_INFERENCE_PAGED_EXTERNAL_PROGRAM_IO\] index=\d+", text)) != 3):
            raise ValueError(f"actual physical MoE native Fresh{fresh} drifted")
    return {"report": report, "hardware_sha256": _sha(hardware_path),
            "paged_sidecar_sha256": _sha(directory / "moe_inference_paged_runtime.json"),
            "artifacts_sha256": artifact_hashes}


def run(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    if root.exists():
        raise ValueError("full Fresh root must be a new empty path")
    binding = _binding(args)
    root.mkdir(parents=True)
    evidence = []
    for index in (0, 1):
        directory = root / f"full_fresh_{index}"
        command = (
            sys.executable, "-m", "llm.test.frontend.integration.run_moe_inference_paged_offload_runtime_canary",
            "--output", str(directory), "--npusim", str(args.npusim.resolve()),
            "--finalizer", str(args.finalizer.resolve()),
            "--simulation", str(args.simulation.resolve()),
            "--expected-npusim-sha256", binding["tool_sha256"]["npusim"],
            "--expected-finalizer-sha256", binding["tool_sha256"]["finalizer"],
            "--timeout", str(args.native_timeout),
        )
        completed = subprocess.run(command, cwd=_ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, check=False,
                                   timeout=args.process_timeout)
        (root / f"full_fresh_{index}.runner.stdout.txt").write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"MoE complete materialization {index} failed: {completed.returncode}")
        if _binding(args) != binding:
            raise RuntimeError("source or native tools changed during full MoE materialization")
        evidence.append(audit(directory, binding))
    if evidence[0] != evidence[1]:
        raise ValueError("two independent full MoE materializations disagree")
    (root / "full_fresh_evidence.json").write_text(json.dumps({
        "status": "verified", "independent_full_materializations": 2,
        "independent_native_executions": 4, "binding": binding,
        "case": evidence[0],
    }, indent=2, sort_keys=True) + "\n")
    print("MoE 1x2 full low-HBM offload PASS two independent full materializations", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--native-timeout", type=int, default=900)
    parser.add_argument("--process-timeout", type=int, default=2400)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
