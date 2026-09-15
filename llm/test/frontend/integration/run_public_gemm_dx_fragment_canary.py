"""Source-bound one-core HBM→LSU_LOAD→named GEMM FP32 dX timing canary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.public_gemm_dx_fragment_program_io import (
    build_public_gemm_dx_fragment_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import LinkedProgramManifest
from llm.frontend.wafer_frontend.schema.serde import canonical_json, from_data
from llm.test.frontend.integration.flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)

_ROOT = Path(__file__).resolve().parents[4]
_SIM = _ROOT / "llm/test/program/p5_behavioral_simulation.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_SOURCES = (
    _ROOT / "llm/frontend/wafer_frontend/schema/artifact_manifest.py",
    _ROOT / "llm/frontend/wafer_frontend/schema/action.py",
    _ROOT / "llm/frontend/wafer_frontend/schema/ir0.py",
    _ROOT / "llm/frontend/wafer_frontend/schema/ir1.py",
    _ROOT / "llm/frontend/wafer_frontend/schema/program_io.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/validate_ir0.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/program_io.py",
    _ROOT / "llm/frontend/wafer_frontend/lowering/coarse.py",
    _ROOT / "llm/frontend/wafer_frontend/lowering/linker.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/public_gemm_dx_fragment_program_io.py",
    Path(__file__).resolve(),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, emitter: Path, finalizer: Path, npusim: Path,
        output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    source_sha = {str(path.relative_to(_ROOT)): _sha(path) for path in _SOURCES}
    tool_sha = {name: _sha(path) for name, path in (
        ("emitter", emitter), ("finalizer", finalizer), ("npusim", npusim))}
    emitted = subprocess.run(
        [str(emitter), "--emit-gemm-dx-state-manifest"],
        cwd=_ROOT / "llm", capture_output=True, text=True, check=True)
    manifest = from_data(LinkedProgramManifest, json.loads(emitted.stdout))
    manifest.validate("public_gemm_dx_emitted_source")
    linked = output / "gemm_dx.linked.json"
    linked.write_text(canonical_json(manifest), encoding="utf-8")
    artifact = output / "gemm_dx.npup"
    report = output / "gemm_dx.finalizer.json"
    finalized = subprocess.run([
        str(finalizer), "--input", str(linked), "--output", str(artifact),
        "--report", str(report)], cwd=_ROOT / "llm",
        capture_output=True, text=True, check=False)
    (output / "gemm_dx.finalizer.stdout.txt").write_text(
        finalized.stdout + finalized.stderr, encoding="utf-8")
    if finalized.returncode:
        raise RuntimeError(f"native 0x26 finalizer failed: {finalized.stderr}")
    digest = _sha(artifact)
    io = build_public_gemm_dx_fragment_program_io(manifest, digest)
    negative = {}
    for name, keyword in (("zero_weight", "weight_payload_override"),
                          ("zero_upstream", "upstream_payload_override")):
        try:
            build_public_gemm_dx_fragment_program_io(
                manifest, digest, **{keyword: bytes(256 if name == "zero_weight"
                                                    else 128)})
        except SchemaError:
            negative[name] = True
        else:
            negative[name] = False
    if negative != {"zero_weight": True, "zero_upstream": True}:
        raise RuntimeError("zeroed source operands escaped signed producer gate")
    sidecar = output / "gemm_dx.program_io.json"
    sidecar.write_text(canonical_json(io), encoding="utf-8")
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 65536
    hardware["memory"]["sram"]["capacity_bytes"] = 65536
    hardware["memory"]["sram"]["regions"] = [{
        "name": "sram_main", "base_bytes": 0, "size_bytes": 4096,
        "allocator": "block", "spillable": False,
        "access": ["compute", "dte", "lsu", "legacy", "noc_rx"],
    }]
    hw = output / "hardware.json"
    hw.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")),
                  encoding="utf-8")
    runtime = subprocess.run([
        str(npusim), "--program", str(artifact),
        "--linked-manifest", str(linked), "--program-io", str(sidecar),
        "--hardware-config", str(hw), "--simulation-config", str(_SIM),
        "--mapping-config", str(_MAPPING), "--trace-window", "1000000",
    ], cwd=_ROOT / "llm", stdout=subprocess.PIPE,
       stderr=subprocess.STDOUT, text=True, check=False, timeout=600)
    transcript = output / "gemm_dx.npusim.stdout.txt"
    transcript.write_text(runtime.stdout, encoding="utf-8")
    if source_sha != {str(path.relative_to(_ROOT)): _sha(path)
                      for path in _SOURCES} or tool_sha != {
            name: _sha(path) for name, path in (
                ("emitter", emitter), ("finalizer", finalizer),
                ("npusim", npusim))}:
        raise RuntimeError("0x26 source/tool SHA drifted during physical run")
    return {
        "source_sha256": source_sha, "tool_sha256": tool_sha,
        "manifest_id": manifest.id, "manifest_sha256": _sha(linked),
        "artifact_sha256": digest, "program_io_id": io.id,
        "program_io_sha256": _sha(sidecar),
        "initializations": len(io.initializations),
        "hbm_initializations": sum(item.target.kind.value == "hbm"
                                   for item in io.initializations),
        "probes": len(io.output_probes), "negative": negative,
        "finalizer_exit": finalized.returncode,
        "npusim_exit": runtime.returncode,
        "npusim_stdout_sha256": _sha(transcript),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emitter", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(emitter=args.emitter.resolve(),
                 finalizer=args.finalizer.resolve(),
                 npusim=args.npusim.resolve(), output=args.output.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["npusim_exit"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
