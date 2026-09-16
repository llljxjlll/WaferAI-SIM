"""Compile, relink, finalize and build ProgramIO for fixed 1x4 MoE offload.

This stage intentionally stops before the versioned native pager/sidecar exists.
It cannot be counted as a low-HBM offload native execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source_root: Path, finalizer: Path, output: Path) -> dict:
    source_root = source_root.resolve()
    finalizer = finalizer.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("EP4 compile canary needs a new output root")
    if not finalizer.is_file():
        raise ValueError("production finalizer is missing")
    module_path = Path(__file__).resolve().parents[4] / (
        "llm/frontend/wafer_frontend/passes/"
        "moe_inference_paged_compile_sequence_ep4.py"
    )
    module_name = (
        "llm.frontend.wafer_frontend.passes."
        "moe_inference_paged_compile_sequence_ep4"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ValueError("EP4 relinker cannot be loaded")
    relinker = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = relinker
    spec.loader.exec_module(relinker)

    from llm.frontend.wafer_frontend.passes.dense_inference_paged_program_io import (
        retarget_dense_inference_paged_sram_program_io,
    )
    from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
    from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import (
        compile_moe_full_model_inference_sequence,
    )
    from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
    from llm.frontend.wafer_frontend.schema.workload_run import WorkloadFamily
    from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
    from llm.test.frontend.integration.run_moe_full_model_sequence_runtime_canary import (
        build_full_model_program_io, prove_full_model_dataflow,
    )
    from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
    from llm.test.frontend.unit.test_moe_compile_sequence import _manifest
    from llm.test.frontend.unit.test_moe_full_model_compile_sequence import _legacy_template

    frozen_module = Path(sys.modules[_manifest.__module__].__file__).resolve()
    if not frozen_module.is_relative_to(source_root):
        raise ValueError("production MoE source was not loaded from frozen root")
    fabric = physical_fabric_from_data(minimal_hardware(4, 1, sram_bytes=65536))
    sequence = compile_moe_full_model_inference_sequence(
        _manifest(WorkloadFamily.MOE_INFERENCE, rows=1, columns=4),
        _legacy_template(), fabric,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    output.mkdir(parents=True)
    finalizer_sha = _sha(finalizer)
    driver_sha = _sha(Path(__file__).resolve())
    relinker_sha = _sha(module_path)
    units = {item.id: item for item in sequence.moe_blocks.units}
    segments = []
    for step, segment in enumerate(sequence.segments):
        prove_full_model_dataflow(
            segment, tuple(units[ref] for ref in segment.moe_unit_refs)
        )
        source = segment.executable_manifest
        paged = relinker.relink_moe_inference_paged_segment_ep4(source, step)
        artifacts = {}
        for role, manifest in (("source", source), ("paged", paged)):
            manifest_path = output / f"segment_{step}.{role}.linked.json"
            artifact_path = output / f"segment_{step}.{role}.npup"
            report_path = output / f"segment_{step}.{role}.finalizer.json"
            manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
            completed = subprocess.run(
                [str(finalizer), "--input", str(manifest_path),
                 "--output", str(artifact_path), "--report", str(report_path)],
                cwd=output, capture_output=True, text=True, timeout=120,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"production {role} finalizer failed step={step}: "
                    f"{completed.stdout}\n{completed.stderr}"
                )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (_sha(finalizer) != finalizer_sha
                    or report["artifact_sha256"] != _sha(artifact_path)
                    or report["linked_manifest_id"] != manifest.id
                    or report["linked_manifest_digest"] != canonical_digest(manifest)):
                raise ValueError("production finalizer artifact closure drifted")
            artifacts[role] = {
                "manifest_id": manifest.id,
                "manifest_sha256": _sha(manifest_path),
                "artifact_sha256": _sha(artifact_path),
                "report_sha256": _sha(report_path),
            }
        original_io = build_full_model_program_io(
            segment, artifacts["source"]["artifact_sha256"]
        )
        original_io.validate_against(source)
        paged_io = retarget_dense_inference_paged_sram_program_io(
            original_io, paged, artifacts["paged"]["artifact_sha256"]
        )
        paged_io.validate_against(paged)
        program_io_path = output / f"segment_{step}.program_io.json"
        program_io_path.write_text(canonical_json(paged_io), encoding="utf-8")
        segments.append({
            "step": step,
            "source": artifacts["source"],
            "paged": artifacts["paged"],
            "program_io_sha256": _sha(program_io_path),
            "program_io_initializations": len(paged_io.initializations),
            "program_io_probes": len(paged_io.output_probes),
            "runtime_core_ids": [item.runtime_core_id for item in paged.core_streams],
        })
        print(f"MoE EP4 compile/finalizer/ProgramIO PASS step={step}", flush=True)
    result = {
        "schema_version": "moe-ep4-low-hbm-compile-canary-v1",
        "status": "compile_finalizer_program_io_only",
        "mesh": "1x4",
        "ep": 4,
        "source_root": str(source_root),
        "driver_sha256": driver_sha,
        "relinker_sha256": relinker_sha,
        "finalizer_sha256": finalizer_sha,
        "segments": segments,
    }
    (output / "compile_canary_receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_root, args.finalizer, args.output)


if __name__ == "__main__":
    main()
