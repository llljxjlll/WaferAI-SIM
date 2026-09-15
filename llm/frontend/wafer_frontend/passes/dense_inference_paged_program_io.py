"""Retarget production Dense timing ProgramIO to bounded external authority.

The original producer proves SRAM ownership/first-read requirements. This
retarget retains those exact SRAM transactions and terminal probes, while the
separate source-signed pager owns all HBM parameter/KV initialization and
versioned KV probes. No aliased weight HBM bytes are initialized at startup.
"""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest
from ..schema.program_io import (
    ProgramHbmTarget,
    ProgramIoContract,
    ProgramSramTarget,
)


def retarget_dense_inference_paged_sram_program_io(
    source: ProgramIoContract,
    paged_manifest: LinkedProgramManifest,
    paged_artifact_sha256: str,
) -> ProgramIoContract:
    """Keep production SRAM timing seeds/probes; defer HBM to actual DMA."""

    paged_manifest.validate("dense_paged.linked")
    sram_initializations = tuple(
        item for item in source.initializations
        if type(item.target) is ProgramSramTarget
    )
    sram_probes = tuple(
        item for item in source.output_probes
        if type(item.target) is ProgramSramTarget
    )
    if (
        not sram_initializations or not sram_probes
        or any(type(item.target) not in (ProgramSramTarget, ProgramHbmTarget)
               for item in (*source.initializations, *source.output_probes))
    ):
        raise SchemaError("production SRAM timing IO closure missing", path="source")
    used_blobs = {
        item.blob_ref for item in (*sram_initializations, *sram_probes)
    }
    blobs = tuple(
        item for item in source.blobs if item.id in used_blobs
    )
    if {item.id for item in blobs} != used_blobs:
        raise SchemaError("SRAM ProgramIO blob closure changed", path="source.blobs")
    paged = ProgramIoContract.create(
        producer_pass=source.producer_pass,
        mode=source.mode,
        source_manifest=paged_manifest,
        program_artifact_sha256=paged_artifact_sha256,
        blobs=blobs,
        initializations=sram_initializations,
        output_probes=sram_probes,
    )
    paged.validate_against(paged_manifest)
    return paged


__all__ = ["retarget_dense_inference_paged_sram_program_io"]
