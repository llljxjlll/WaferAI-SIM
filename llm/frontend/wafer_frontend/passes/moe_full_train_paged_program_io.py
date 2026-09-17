"""Retarget true MoE training ProgramIO for 19 external parameter pages."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest, StateKind
from ..schema.program_io import (
    ProgramHbmTarget, ProgramIoContract, ProgramSramInitialization,
    ProgramSramTarget,
)


def retarget_moe_full_train_paged_program_io(
    original: ProgramIoContract,
    source: LinkedProgramManifest,
    paged: LinkedProgramManifest,
    paged_artifact_sha256: str,
) -> ProgramIoContract:
    """Retain SRAM input/loss and pinned route bytes; external pager owns weights."""
    original.validate_against(source)
    paged.validate("moe_train_paged.linked")
    original_states = {abi.state_ref: abi for fragment in source.fragments
                       for abi in fragment.state_abi}
    paged_states = {abi.state_ref: abi for fragment in paged.fragments
                    for abi in fragment.state_abi}
    if (len(original_states) != 21 or len(paged_states) != 21
            or set(original_states) != set(paged_states)):
        raise SchemaError("MoE training route/parameter state inventory changed",
                          path="source")
    retained = []
    external = []
    for item in original.initializations:
        if type(item.target) is ProgramSramTarget:
            retained.append(item)
            continue
        if type(item.target) is not ProgramHbmTarget:
            raise SchemaError("unsupported ProgramIO target", path="original")
        state = original_states[item.target.state_ref]
        if state.kind is StateKind.TRAINABLE_PARAMETER:
            external.append(item)
            continue
        if state.kind is not StateKind.MOE_STATIC_ROUTE:
            raise SchemaError("unexpected resident HBM state", path="original")
        relocated = paged_states[state.state_ref]
        retained.append(ProgramSramInitialization.create(
            target=replace(item.target, state_abi_id=relocated.id),
            offset_bytes=item.offset_bytes, length_bytes=item.length_bytes,
            blob_ref=item.blob_ref, purpose=item.purpose,
        ))
    route_refs = {item.target.state_ref for item in retained
                  if type(item.target) is ProgramHbmTarget}
    if (len(external) != 19 or len(route_refs) != 2
            or len(retained) != 152
            or len(original.output_probes) != 1
            or type(original.output_probes[0].target) is not ProgramSramTarget):
        raise SchemaError("19 external weights and two pinned routes not closed",
                          path="original.initializations")
    blob_refs = {item.blob_ref for item in (*retained, *original.output_probes)}
    blobs = tuple(item for item in original.blobs if item.id in blob_refs)
    if {item.id for item in blobs} != blob_refs:
        raise SchemaError("paged ProgramIO blob closure changed", path="original.blobs")
    result = ProgramIoContract.create(
        producer_pass=original.producer_pass, mode=original.mode,
        source_manifest=paged, program_artifact_sha256=paged_artifact_sha256,
        blobs=blobs, initializations=tuple(retained),
        output_probes=original.output_probes,
    )
    result.validate_against(paged)
    return result


__all__ = ["retarget_moe_full_train_paged_program_io"]
