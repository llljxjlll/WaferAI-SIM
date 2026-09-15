"""Scoped physical 0x23 INT32 scatter WGRAD ProgramIO/NpuSim canary.

The source carrier comes from ISA selftest.  This runner canonically keeps one
core, adds the missing upstream SRAM_ALLOC_AT/FREE, then uses the production
finalizer and NpuSim ProgramIO resolver.  It proves physical timing only.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.public_weight_gradient_fragment_program_io import (
    build_public_embedding_wgrad_fragment_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    AddressOperandBinding, AddressRelocation, CommandFragment,
    LinkedProgramManifest, LinkedRecordRef, ManifestInputKind,
    ProgramSymbolKind, RecordOperand, SemanticOperandId,
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
    _ROOT / "llm/frontend/wafer_frontend/lowering/coarse.py",
    _ROOT / "llm/frontend/wafer_frontend/lowering/linker.py",
    _ROOT / "llm/frontend/wafer_frontend/passes/public_weight_gradient_fragment_program_io.py",
    Path(__file__).resolve(),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_core_with_real_upstream_allocation(source: LinkedProgramManifest) -> LinkedProgramManifest:
    source.validate("wgrad_fixture")
    core = source.core_bindings[0].logical_core
    old = source.fragments[0]
    original = old.core_streams[0]
    aux = next(abi for abi in old.buffer_abi
               if abi.logical_core == core and abi.binding_id == "abs_aux")
    alloc_operands = list(original.records[0].operands)
    alloc_operands[1] = RecordOperand.address(
        "label_symbol", SemanticOperandId.LABEL_SYMBOL, "p_label_aux")
    alloc_operands[2] = RecordOperand.literal("region_offset_bytes", 0x400)
    alloc_operands[3] = RecordOperand.literal("size_bytes", aux.size_bytes)
    aux_alloc = replace(original.records[0], operands=tuple(alloc_operands))
    # Source schema declares TASK; finalizer's producer-scoped physical
    # terminal policy safely promotes the sole typed gradient output.
    persistent_output_alloc = original.records[2]
    aux_free = replace(original.records[-1], operands=(
        RecordOperand.address("symbol", SemanticOperandId.SYMBOL,
                              "p_label_aux"),))
    # Keep the FP32 OWNED output alive for AFTER_PROGRAM physical probe.
    records = (*original.records[:2], persistent_output_alloc, aux_alloc,
               *original.records[3:-1], aux_free)
    relocations = [replace(relocation,
                            record_index=relocation.record_index +
                            (relocation.record_index >= 3))
                   for relocation in original.address_relocations
                   if relocation.record_index != len(original.records) - 1]
    relocations.extend((
        AddressRelocation(3, SemanticOperandId.REGION_NAME,
                          ProgramSymbolKind.SRAM_REGION, "p_region", 0),
        AddressRelocation(3, SemanticOperandId.LABEL_SYMBOL,
                          ProgramSymbolKind.SRAM_LABEL, "p_label_aux", 0),
        AddressRelocation(8, SemanticOperandId.SYMBOL,
                          ProgramSymbolKind.SRAM_LABEL, "p_label_aux", 0),
    ))
    stream = replace(original, records=records,
        address_relocations=tuple(sorted(relocations,
              key=lambda r: (r.record_index, int(r.operand_id)))))
    frag_args = old._semantic_key()
    frag_args.update(claimed_action_ids=("a0",), core_streams=(stream,),
                     buffer_abi=tuple(abi for abi in old.buffer_abi
                                      if abi.logical_core == core))
    fragment = CommandFragment.create(
        producer_pass="public_embedding_wgrad_allocated_fragment", **frag_args)
    fragment.validate("wgrad_allocated_fragment")
    old_refs = source.core_streams[0].records
    refs = [replace(ref, fragment_id=fragment.id,
                    fragment_record_index=ref.fragment_record_index +
                    (ref.fragment_record_index >= 3)) for ref in old_refs
            if ref.fragment_record_index != len(original.records) - 1]
    refs.extend((LinkedRecordRef(fragment.id, 3, "a0"),
                 LinkedRecordRef(fragment.id, 8, "a0")))
    linked_stream = replace(source.core_streams[0],
                            records=tuple(sorted(refs,
                                key=lambda ref: ref.fragment_record_index)))
    shifted = [replace(binding, fragment_id=fragment.id,
                       fragment_record_index=binding.fragment_record_index +
                       (binding.fragment_record_index >= 3))
               for binding in source.address_operand_bindings
               if binding.logical_core == core
               and binding.fragment_record_index != len(original.records) - 1]
    witness = next(binding for binding in shifted
                   if binding.fragment_record_index == 0
                   and binding.operand_id is SemanticOperandId.REGION_NAME)
    shifted.extend((
        replace(witness, fragment_record_index=3,
                operand_id=SemanticOperandId.REGION_NAME,
                buffer_abi_ids=(aux.id,), tensor_slices=(aux.tensor_slice,)),
        replace(witness, fragment_record_index=3,
                operand_id=SemanticOperandId.LABEL_SYMBOL,
                buffer_abi_ids=(aux.id,), tensor_slices=(aux.tensor_slice,)),
        replace(witness, fragment_record_index=8,
                operand_id=SemanticOperandId.SYMBOL,
                buffer_abi_ids=(aux.id,), tensor_slices=(aux.tensor_slice,)),
    ))
    envelope = source.envelope
    envelope = replace(envelope, active_cores=(core,),
        start_events=tuple(event for event in envelope.start_events
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
        address_operand_bindings=tuple(sorted(shifted, key=lambda binding: (
            binding.logical_core.die_id, binding.logical_core.local_core_id,
            binding.fragment_id, binding.fragment_record_index,
            int(binding.operand_id)))),
        input_digests=tuple(replace(digest, artifact_id=fragment.id,
            digest=canonical_digest(fragment))
            if digest.kind is ManifestInputKind.COMMAND_FRAGMENT else digest
            for digest in source.input_digests),
        envelope=envelope,
    )
    result = LinkedProgramManifest.create(
        producer_pass="public_embedding_wgrad_allocated_fragment", **args)
    result.validate("public_embedding_wgrad_allocated_fragment")
    return result


def run(*, emitter: Path, finalizer: Path, npusim: Path,
        output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    source_sha = {str(path.relative_to(_ROOT)): _sha(path)
                  for path in _SOURCES}
    tool_sha = {name: _sha(path) for name, path in (
        ("emitter", emitter), ("finalizer", finalizer), ("npusim", npusim))}
    emitted = subprocess.run([str(emitter), "--emit-wgrad-manifest"],
                             check=True, capture_output=True, text=True)
    source = from_data(LinkedProgramManifest, json.loads(emitted.stdout))
    source.validate("wgrad_emitted_source")
    manifest = _one_core_with_real_upstream_allocation(source)
    linked = output / "embedding_wgrad.linked.json"
    linked.write_text(canonical_json(manifest), encoding="utf-8")
    artifact = output / "embedding_wgrad.npup"
    report = output / "embedding_wgrad.finalizer.json"
    finalized = subprocess.run([
        str(finalizer), "--input", str(linked), "--output", str(artifact),
        "--report", str(report)], cwd=_ROOT / "llm", capture_output=True,
        text=True, check=False)
    (output / "embedding_wgrad.finalizer.stdout.txt").write_text(
        finalized.stdout + finalized.stderr)
    if finalized.returncode:
        raise RuntimeError(f"finalizer failed ({finalized.returncode}): {finalized.stderr}")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    io = build_public_embedding_wgrad_fragment_program_io(manifest, digest)
    # Producer-layer negative: the runtime blob must come from this record's
    # INT32 source trace, not a separately synchronized zero-token oracle.
    try:
        build_public_embedding_wgrad_fragment_program_io(
            manifest, digest, index_payload_override=b"\x00" * 16)
    except SchemaError as error:
        negative_index_blob = "runtime INT32 token blob disagrees" in str(error)
    else:
        negative_index_blob = False
    if not negative_index_blob:
        raise RuntimeError("zero token runtime blob bypassed 0x23 source trace")
    sidecar = output / "embedding_wgrad.program_io.json"
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
    transcript = output / "embedding_wgrad.npusim.stdout.txt"
    transcript.write_text(runtime.stdout)
    if source_sha != {str(path.relative_to(_ROOT)): _sha(path)
                      for path in _SOURCES} or tool_sha != {
            name: _sha(path) for name, path in (
                ("emitter", emitter), ("finalizer", finalizer), ("npusim", npusim))}:
        raise RuntimeError("public WGRAD source/tool SHA drifted during physical run")
    return {
        "source_sha256": source_sha,
        "tool_sha256": tool_sha,
        "manifest_id": manifest.id, "manifest_sha256": hashlib.sha256(
            linked.read_bytes()).hexdigest(), "artifact_sha256": digest,
        "program_io_id": io.id,
        "program_io_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "index_blob_sha256": hashlib.sha256(
            next(blob.payload() for blob in io.blobs
                 if blob.length_bytes == 16)).hexdigest(),
        "initializations": len(io.initializations),
        "probes": len(io.output_probes),
        "negative_zero_index_blob_rejected": negative_index_blob,
        "finalizer_exit": finalized.returncode,
        "npusim_exit": runtime.returncode,
        "npusim_stdout_sha256": hashlib.sha256(transcript.read_bytes()).hexdigest(),
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
