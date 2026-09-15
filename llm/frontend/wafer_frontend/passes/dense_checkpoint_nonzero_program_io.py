"""Nonzero signed activation backing/probe for bounded L2 timing checkpoint runs.

Only the actual source BufferABI selected by a typed activation tape is
initialized.  The native HBM probe checks the bytes after real LSU_STORE;
this does not claim numerically computed logits or gradients in timing mode.
"""
from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LinkedProgramManifest, ProgramSymbolKind, StateABI,
)
from ..schema.program_io import (
    ProgramBlob, ProgramHbmTarget, ProgramIoContract,
    ProgramIoTargetKind, ProgramOutputCapture, ProgramOutputComparison,
    ProgramOutputProbe, ProgramSramInitialization, _entry_order,
)
from ..schema.train_n6 import TrainLinkedProgram
from ..lowering.full_training_ce_tape_graft import CeGraftedPhysicalForward
from .dense_checkpoint_physical_source import (
    DenseCheckpointActivationTape, DenseCheckpointPhysicalCut,
)
from .full_training_ce_seed_program_io import (
    build_bounded_seeded_ce_program_io,
)


def _activation_seed(size: int) -> bytes:
    return bytes((index % 251) + 1 for index in range(size))


def build_dense_checkpoint_nonzero_program_io(
    profile: TrainLinkedProgram,
    manifest: LinkedProgramManifest,
    graft: CeGraftedPhysicalForward,
    cut: DenseCheckpointPhysicalCut,
    tape: DenseCheckpointActivationTape,
    activation: StateABI,
    artifact_sha256: str,
) -> ProgramIoContract:
    """Seed actual nonzero SRAM activation; probe signed HBM StateABI after run."""
    manifest.validate("checkpoint_program_io_source")
    tape.validate(cut)
    if (activation.kind.value != "activation" or
            activation.hbm_binding_ref != tape.hbm_binding_ref or
            activation.address != tape.activation_hbm_address or
            activation.size_bytes != tape.size_bytes or
            activation.die_id != tape.die_id):
        raise SchemaError("ProgramIO activation StateABI differs from typed source tape",
                          path="checkpoint_program_io_activation")
    base = build_bounded_seeded_ce_program_io(
        profile, manifest, graft, artifact_sha256)
    chosen_abi = tape.saved_buffer_abi_id
    chosen = [entry for entry in base.initializations
              if getattr(entry.target, "buffer_abi_id", None) == chosen_abi]
    weight = [entry for entry in base.initializations
              if getattr(entry.target, "state_abi_id", None) ==
                 cut.replay_parameter_state.id]
    if len(weight) != 1 or weight[0].length_bytes != cut.replay_weight.buffer.size_bytes:
        raise SchemaError("replay W lacks one exact source StateABI HBM seed",
                          path="checkpoint_program_io_weight")
    if len(chosen) != 1 or chosen[0].length_bytes != activation.size_bytes:
        raise SchemaError("actual saved activation has no single full-size host seed",
                          path="checkpoint_program_io_initializations")
    payload = _activation_seed(activation.size_bytes)
    if not any(payload):
        raise AssertionError("activation seed must be nonzero")
    blob = ProgramBlob.create(payload)
    weight_blob = ProgramBlob.create(bytes(((index * 7) % 251) + 1
                                           for index in range(weight[0].length_bytes)))
    entries = []
    for old in base.initializations:
        entries.append(ProgramSramInitialization.create(
            target=old.target, offset_bytes=old.offset_bytes,
            length_bytes=old.length_bytes,
            blob_ref=blob.id if old is chosen[0] else
                     weight_blob.id if old is weight[0] else old.blob_ref,
            purpose=old.purpose))
    definitions = {definition.symbol.id: (index, definition)
                   for index, definition in
                   enumerate(manifest.program_symbol_definitions)}
    hbm_symbols = [(index, definition)
                   for index, definition in
                   enumerate(manifest.program_symbol_definitions)
                   if definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
                   and definition.symbol.source_ref == activation.hbm_binding_ref]
    if len(hbm_symbols) != 1 or hbm_symbols[0][1].value != activation.address:
        raise SchemaError("actual HBM ProgramSymbolDefinition lost activation home",
                          path="checkpoint_program_io_hbm_symbol")
    index, definition = hbm_symbols[0]
    hbm_target = ProgramHbmTarget(
        ProgramIoTargetKind.HBM, definition.symbol.id, index,
        definition.name, activation.id, activation.state_ref,
        activation.hbm_binding_ref)
    probe = ProgramOutputProbe.create(
        target=hbm_target, offset_bytes=0,
        length_bytes=activation.size_bytes, blob_ref=blob.id,
        comparison=ProgramOutputComparison.EXACT_BYTES,
        capture=ProgramOutputCapture.AFTER_PROGRAM)
    probes = tuple(sorted((*base.output_probes, probe), key=_entry_order))
    used = {entry.blob_ref for entry in entries}
    used.update(entry.blob_ref for entry in probes)
    blobs = {old.id: old for old in base.blobs}
    blobs[blob.id] = blob
    blobs[weight_blob.id] = weight_blob
    result = ProgramIoContract.create(
        producer_pass="dense_checkpoint_nonzero_program_io",
        mode=base.mode, source_manifest=manifest,
        program_artifact_sha256=artifact_sha256,
        blobs=tuple(blobs[item] for item in sorted(used)),
        initializations=tuple(sorted(entries, key=_entry_order)),
        output_probes=probes)
    result.validate_against(manifest)
    return result


__all__ = ["build_dense_checkpoint_nonzero_program_io"]
