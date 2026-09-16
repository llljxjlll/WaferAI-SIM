"""Read-only reopen of the EP6 low-HBM source/finalizer/ProgramIO gap receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from llm.frontend.wafer_frontend.schema.artifact_manifest import LinkedProgramManifest
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, load_json_dataclass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, source_root: Path, finalizer: Path) -> None:
    root, source_root, finalizer = root.resolve(), source_root.resolve(), finalizer.resolve()
    receipt = json.loads((root / "ep6_gap_receipt.json").read_text(encoding="utf-8"))
    if (receipt["schema_version"] != "moe-ep6-low-hbm-source-blocker-v1"
            or receipt["status"] != "source_compile_finalizer_program_io_only"
            or [item["mesh"] for item in receipt["cases"]] != ["2x3", "3x2"]):
        raise ValueError("EP6 source gap receipt status/shape changed")
    binding = receipt["binding"]
    probe = Path(__file__).with_name("probe_moe_inference_ep6_offload_gap.py")
    relinker = Path(__file__).resolve().parents[4] / (
        "llm/frontend/wafer_frontend/passes/moe_inference_paged_compile_sequence_ep4.py"
    )
    if (binding["source_root"] != str(source_root)
            or binding["driver_sha256"] != _sha(probe)
            or binding["ep4_relinker_sha256"] != _sha(relinker)
            or binding["finalizer_sha256"] != _sha(finalizer)):
        raise ValueError("EP6 frozen source/tool/probe binding drifted")
    for relative, digest in binding["frozen_imported_source_sha256"].items():
        if _sha(source_root / relative) != digest:
            raise ValueError(f"EP6 frozen imported source drifted: {relative}")
    models = set()
    for case in receipt["cases"]:
        mesh = case["mesh"]
        models.add(case["model_sha256"])
        if (not case["resident_rejection"].startswith("memory_capacity_exceeded")
                or case["status"] != "source_compile_finalizer_program_io_only"
                or case["first_existing_pager_blocker"] != {
                    "code": "schema_error",
                    "message": "schema_error at source: two-layer EP4 full-model linked source shape changed",
                }):
            raise ValueError(f"EP6 {mesh} capacity or pager blocker changed")
        for step, segment in enumerate(case["source_segment_receipts"]):
            if segment["step"] != step:
                raise ValueError(f"EP6 {mesh} segment order changed")
            directory = root / mesh
            paths = {
                "manifest_sha256": directory / f"segment_{step}.source.linked.json",
                "artifact_sha256": directory / f"segment_{step}.source.npup",
                "finalizer_report_sha256": directory / f"segment_{step}.source.finalizer.json",
                "program_io_sha256": directory / f"segment_{step}.source.program_io.json",
            }
            for key, path in paths.items():
                if _sha(path) != segment[key]:
                    raise ValueError(f"EP6 {mesh} step{step} {key} drifted")
            manifest = load_json_dataclass(LinkedProgramManifest, paths["manifest_sha256"])
            io = load_json_dataclass(ProgramIoContract, paths["program_io_sha256"])
            io.validate_against(manifest)
            report = json.loads(paths["finalizer_report_sha256"].read_text())
            if (manifest.id != segment["manifest_id"]
                    or report["linked_manifest_id"] != manifest.id
                    or report["linked_manifest_digest"] != canonical_digest(manifest)
                    or report["artifact_sha256"] != segment["artifact_sha256"]
                    or len(io.initializations) != segment["program_io_initializations"]
                    or len(io.output_probes) != segment["program_io_probes"]):
                raise ValueError(f"EP6 {mesh} typed source/finalizer/ProgramIO closure changed")
    if len(models) != 1:
        raise ValueError("EP6 rectangles use different models")
    print("EP6 2x3/3x2 source gap RESUME verified; native offload remains blocked")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.source_root, args.finalizer)


if __name__ == "__main__":
    main()
