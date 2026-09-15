"""Exact physical ProgramIO SRAM initialization for independently seeded CE.

This is a timing-only *bounded* Dense-forward plus CE-backward sidecar.  It
retains the existing production Dense input/state seeds and probes, rebinds
the physically extended logits/labels ABIs, then adds one nonzero per-row
FP32 BORROWED dLoss initialization.  No functional logits-gradient result
is claimed: the native CE primitive currently publishes timing work.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..lowering.full_training_ce_tape_graft import CeGraftedPhysicalForward
from ..schema.artifact_manifest import LinkedProgramManifest, ProgramSymbolKind
from ..schema.program_io import (
    ProgramBlob, ProgramHbmTarget, ProgramIoContract, ProgramIoPurpose,
    ProgramIoTargetKind, ProgramSramInitialization, ProgramSramTarget,
    ProgramOutputProbe,
)
from ..schema.train_n6 import TrainLinkedProgram
from .program_io import build_timing_program_io


def build_bounded_seeded_ce_program_io(
    profile: TrainLinkedProgram,
    manifest: LinkedProgramManifest,
    graft: CeGraftedPhysicalForward,
    artifact_sha256: str,
) -> ProgramIoContract:
    """Host-init the actual nonzero FP32 dLoss physical buffer and root ABI."""
    profile.validate("seeded_ce_program_io_forward_profile")
    manifest.validate("seeded_ce_program_io_manifest")
    if (graft.source_forward_manifest_id != profile.manifest.id
            or graft.loss_gradient_abi_id is None
            or graft.loss_gradient_seed is None
            or not any(graft.loss_gradient_seed)):
        raise SchemaError("CE dLoss must use independent nonzero physical seed",
                          path="graft.loss_gradient_seed")
    declarations = {abi.id: abi for fragment in manifest.fragments
                    for abi in fragment.buffer_abi}
    seed = declarations.get(graft.loss_gradient_abi_id)
    if (seed is None or seed.ownership.value != "borrowed"
            or seed.dtype.value != "fp32"
            or len(graft.loss_gradient_seed) != seed.size_bytes
            or graft.loss_gradient_seed != b"\x00\x00\x80\x3f" *
               (seed.size_bytes // 4)):
        raise SchemaError("dLoss initializer must be FP32 1.0 per declared row",
                          path="graft.loss_gradient_seed")
    state_seeds = {abi.state_ref: bytes(abi.size_bytes)
                   for fragment in profile.manifest.fragments
                   for abi in fragment.state_abi}
    base = build_timing_program_io(
        profile, artifact_sha256, state_seed_overrides=state_seeds,
    )
    remapped = dict(graft.old_to_extended_tape_abi_ids)
    definitions = {entry.symbol.id: (index, entry)
                   for index, entry in enumerate(manifest.program_symbol_definitions)}
    blobs = {blob.id: blob for blob in base.blobs}
    entries = []
    for entry in base.initializations:
        target = entry.target
        new_id = (remapped.get(target.buffer_abi_id, target.buffer_abi_id)
                  if isinstance(target, ProgramSramTarget)
                  else target.state_abi_id)
        abi = (declarations.get(new_id)
               if isinstance(target, ProgramSramTarget) else
               next((state for fragment in manifest.fragments
                     for state in fragment.state_abi if state.id == new_id), None))
        if abi is None:
            raise SchemaError("real Dense ProgramIO input disappeared from graft",
                              path="base.initializations")
        definition = definitions.get(target.program_symbol_ref)
        if definition is None:
            raise SchemaError("actual input label/HBM home lacks final symbol",
                              path="base.initializations")
        index, resolved = definition
        if isinstance(target, ProgramSramTarget):
            target = replace(target, buffer_abi_id=new_id,
                             finalized_symbol_index=index,
                             expected_symbol_name=resolved.name)
        else:
            target = replace(target, state_abi_id=new_id,
                             finalized_symbol_index=index,
                             expected_symbol_name=resolved.name)
        entries.append(ProgramSramInitialization.create(
            target=target, offset_bytes=entry.offset_bytes,
            length_bytes=entry.length_bytes, blob_ref=entry.blob_ref,
            purpose=entry.purpose,
        ))
    probes = []
    for entry in base.output_probes:
        target = entry.target
        index, definition = definitions[target.program_symbol_ref]
        target = replace(target, finalized_symbol_index=index,
                         expected_symbol_name=definition.name)
        probes.append(ProgramOutputProbe.create(
            target=target, offset_bytes=entry.offset_bytes,
            length_bytes=entry.length_bytes, blob_ref=entry.blob_ref,
            comparison=entry.comparison, capture=entry.capture,
        ))
    label = next((definition for definition in
                  manifest.program_symbol_definitions
                  if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
                  and definition.symbol.source_ref == seed.storage_id), None)
    if label is None:
        raise SchemaError("seed BufferABI lacks real SRAM_ALLOC_AT label",
                          path="graft.loss_gradient_abi_id")
    index, _ = definitions[label.symbol.id]
    core = next(binding.runtime_core_id for binding in manifest.core_bindings
                if binding.logical_core == seed.logical_core)
    target = ProgramSramTarget(
        ProgramIoTargetKind.SRAM, core, label.symbol.id, index, label.name,
        seed.id, seed.storage_id, seed.value_id, seed.tensor_slice,
        seed.dtype, seed.layout,
    )
    blob = ProgramBlob.create(graft.loss_gradient_seed)
    blobs[blob.id] = blob
    entries.append(ProgramSramInitialization.create(
        target=target, offset_bytes=0, length_bytes=seed.size_bytes,
        blob_ref=blob.id, purpose=ProgramIoPurpose.ACTIVATION,
    ))
    result = ProgramIoContract.create(
        producer_pass="bounded_dense_seeded_ce_program_io", mode=base.mode,
        source_manifest=manifest, program_artifact_sha256=artifact_sha256,
        blobs=tuple(blobs.values()), initializations=tuple(entries),
        output_probes=tuple(probes),
    )
    result.validate_against(manifest)
    return result


__all__ = ["build_bounded_seeded_ce_program_io"]
