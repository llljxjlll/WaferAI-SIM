"""Scoped 0x25 FP16 X/dY→FP32 parameter-gradient physical timing canary."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.public_gemm_wgrad_fragment_program_io import (
    build_public_gemm_wgrad_fragment_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment, LinkedProgramManifest, ManifestInputKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest, canonical_json, from_data,
)
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
    _ROOT / "llm/frontend/wafer_frontend/lowering/coarse.py",
    _ROOT / "llm/frontend/wafer_frontend/lowering/linker.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/public_gemm_wgrad_fragment_program_io.py",
    Path(__file__).resolve(),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_core_persistent_fp32(source: LinkedProgramManifest) -> LinkedProgramManifest:
    source.validate("gemm_wgrad_fixture")
    core = source.core_bindings[0].logical_core
    old = source.fragments[0]
    original = old.core_streams[0]
    if len(original.records) != 8 or int(original.records[4].opcode) != 0x25:
        raise SchemaError("exact 0x25 three-operand carrier is absent",
                          path="gemm_wgrad_fixture")
    # The fixture's final FREE targets only FP32 output.  Finalizer checks the
    # sole owned FP32 root and promotes its TASK allocation to PERSISTENT.
    final = len(original.records) - 1
    stream = replace(original, records=original.records[:-1],
        address_relocations=tuple(relocation for relocation
            in original.address_relocations if relocation.record_index != final))
    frag_args = old._semantic_key()
    frag_args.update(claimed_action_ids=("a0",), core_streams=(stream,),
        buffer_abi=tuple(abi for abi in old.buffer_abi
                         if abi.logical_core == core))
    fragment = CommandFragment.create(
        producer_pass="public_gemm_wgrad_allocated_fragment", **frag_args)
    fragment.validate("gemm_wgrad_fragment")
    linked_stream = replace(source.core_streams[0], records=tuple(
        replace(ref, fragment_id=fragment.id)
        for ref in source.core_streams[0].records
        if ref.fragment_record_index != final))
    bindings = tuple(sorted((
        replace(binding, fragment_id=fragment.id)
        for binding in source.address_operand_bindings
        if binding.logical_core == core and
           binding.fragment_record_index != final
    ), key=lambda binding: (
        binding.logical_core.die_id, binding.logical_core.local_core_id,
        binding.fragment_id, binding.fragment_record_index,
        int(binding.operand_id))))
    envelope = replace(source.envelope, active_cores=(core,),
        start_events=tuple(event for event in source.envelope.start_events
                           if event.target_core == core),
        terminal_cores=(core,), expected_ack_cores=(core,),
        expected_done_cores=(core,))
    args = source._semantic_key()
    args.update(
        fragments=(fragment,),
        fragment_interfaces=(replace(source.fragment_interfaces[0],
                                     fragment_id=fragment.id),),
        core_bindings=(source.core_bindings[0],),
        core_streams=(linked_stream,),
        runtime_symbol_definitions=tuple(definition
            for definition in source.runtime_symbol_definitions
            if core in definition.logical_cores),
        program_symbol_definitions=tuple(replace(definition,
            logical_cores=(core,)) for definition
            in source.program_symbol_definitions),
        address_operand_bindings=bindings,
        input_digests=tuple(replace(digest, artifact_id=fragment.id,
            digest=canonical_digest(fragment))
            if digest.kind is ManifestInputKind.COMMAND_FRAGMENT else digest
            for digest in source.input_digests),
        envelope=envelope,
    )
    result = LinkedProgramManifest.create(
        producer_pass="public_gemm_wgrad_allocated_fragment", **args)
    result.validate("public_gemm_wgrad_allocated_fragment")
    return result


def run(*, emitter: Path, finalizer: Path, npusim: Path,
        output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    source_sha = {str(path.relative_to(_ROOT)): _sha(path)
                  for path in _SOURCES}
    tool_sha = {name: _sha(path) for name, path in (
        ("emitter", emitter), ("finalizer", finalizer), ("npusim", npusim))}
    emitted = subprocess.run([str(emitter), "--emit-gemm-wgrad-manifest"],
                             check=True, capture_output=True, text=True)
    source = from_data(LinkedProgramManifest, json.loads(emitted.stdout))
    manifest = _one_core_persistent_fp32(source)
    linked = output / "gemm_wgrad.linked.json"
    linked.write_text(canonical_json(manifest), encoding="utf-8")
    artifact = output / "gemm_wgrad.npup"
    report = output / "gemm_wgrad.finalizer.json"
    finalized = subprocess.run([
        str(finalizer), "--input", str(linked), "--output", str(artifact),
        "--report", str(report)], cwd=_ROOT / "llm", capture_output=True,
        text=True, check=False)
    (output / "gemm_wgrad.finalizer.stdout.txt").write_text(
        finalized.stdout + finalized.stderr)
    if finalized.returncode:
        raise RuntimeError(f"0x25 finalizer failed: {finalized.stderr}")
    digest = _sha(artifact)
    io = build_public_gemm_wgrad_fragment_program_io(manifest, digest)
    try:
        build_public_gemm_wgrad_fragment_program_io(
            manifest, digest, activation_payload_override=bytes(64))
    except SchemaError as error:
        negative_zero_activation = "runtime FP16 activation blob disagrees" in str(error)
    else:
        negative_zero_activation = False
    if not negative_zero_activation:
        raise RuntimeError("zero FP16 runtime operand bypassed signed 0x25 source")
    sidecar = output / "gemm_wgrad.program_io.json"
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
    hw.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")))
    runtime = subprocess.run([
        str(npusim), "--program", str(artifact),
        "--linked-manifest", str(linked), "--program-io", str(sidecar),
        "--hardware-config", str(hw), "--simulation-config", str(_SIM),
        "--mapping-config", str(_MAPPING), "--trace-window", "1000000",
    ], cwd=_ROOT / "llm", stdout=subprocess.PIPE,
       stderr=subprocess.STDOUT, text=True, check=False, timeout=600)
    transcript = output / "gemm_wgrad.npusim.stdout.txt"
    transcript.write_text(runtime.stdout)
    if (source_sha != {str(path.relative_to(_ROOT)): _sha(path)
                      for path in _SOURCES} or tool_sha != {
            name: _sha(path) for name, path in (
                ("emitter", emitter), ("finalizer", finalizer), ("npusim", npusim))}):
        raise RuntimeError("0x25 source/tool SHA drifted during physical run")
    return {
        "source_sha256": source_sha, "tool_sha256": tool_sha,
        "manifest_id": manifest.id, "manifest_sha256": _sha(linked),
        "artifact_sha256": digest, "program_io_id": io.id,
        "program_io_sha256": _sha(sidecar),
        "initializations": len(io.initializations),
        "probes": len(io.output_probes),
        "negative_zero_activation_rejected": negative_zero_activation,
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
