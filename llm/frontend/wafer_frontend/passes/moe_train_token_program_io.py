"""Replace only true borrowed forward token seed from a source-bound input.

Native Embedding TABLE WGRAD later reads this produced ProgramIO contract,
independently decoding its bytes.  Caller cannot supply a parallel hand-built
index_trace to self-certify a duplicated/nonzero vocabulary sequence.
"""

from __future__ import annotations

import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest
from ..schema.common import DType
from ..schema.moe_train_token_input_case import MoeTrainTokenInputCase
from ..schema.program_io import (
    ProgramBlob, ProgramIoContract, ProgramSramInitialization,
    ProgramSramTarget,
)
from ..schema.workload_materialization import WorkloadMaterializationManifest


def build_source_bound_moe_train_token_program_io(
    manifest: LinkedProgramManifest,
    base: ProgramIoContract,
    source: WorkloadMaterializationManifest,
    token_input: MoeTrainTokenInputCase,
    *,
    step: int,
    token_value_id: str,
) -> ProgramIoContract:
    """Write the sole rank0 T0.token_ids SRAM blob from TRAIN source input."""
    token_input.validate_against(source)
    base.validate_against(manifest, "moe_token_program_io.base")
    if type(step) is not int or step not in (0, 1) or not token_value_id:
        raise SchemaError("source step and token value are required",
                          path="moe_token_program_io")
    matches = [(index, entry) for index, entry in
               enumerate(base.initializations)
               if type(entry.target) is ProgramSramTarget
               and entry.target.value_id == token_value_id]
    if len(matches) != 1:
        raise SchemaError("actual ProgramIO must borrow precisely one source INT32 token tensor",
                          path="moe_token_program_io.initializations")
    index, entry = matches[0]
    abis = {abi.id: abi for fragment in manifest.fragments
            for abi in fragment.buffer_abi}
    abi = abis.get(entry.target.buffer_abi_id)
    token_ids = token_input.tokens_by_step[step]
    if (abi is None or abi.logical_core.die_id != 0
            or entry.target.value_id != abi.value_id
            or abi.dtype is not DType.INT32
            or abi.tensor_slice.shape != (len(token_ids),)
            or abi.size_bytes != 4 * len(token_ids)
            or entry.offset_bytes != 0
            or entry.length_bytes != abi.size_bytes):
        raise SchemaError("source input cannot write an unrelated physical token ABI",
                          path="moe_token_program_io.token_abi")
    payload = struct.pack(f"<{len(token_ids)}i", *token_ids)
    new_blob = ProgramBlob.create(payload)
    originals = {blob.id: blob for blob in base.blobs}
    originals[new_blob.id] = new_blob
    entries = list(base.initializations)
    entries[index] = ProgramSramInitialization.create(
        target=entry.target, offset_bytes=entry.offset_bytes,
        length_bytes=entry.length_bytes, blob_ref=new_blob.id,
        purpose=entry.purpose,
    )
    used = {item.blob_ref for item in (*entries, *base.output_probes)}
    result = ProgramIoContract.create(
        producer_pass="source_bound_moe_train_token_program_io",
        mode=base.mode, source_manifest=manifest,
        program_artifact_sha256=base.program_artifact_sha256,
        blobs=tuple(sorted((blob for blob in originals.values()
                            if blob.id in used), key=lambda item: item.id)),
        initializations=tuple(sorted(entries, key=lambda item: item.id)),
        output_probes=base.output_probes,
    )
    result.validate_against(manifest, "moe_token_program_io.result")
    return result


__all__ = ["build_source_bound_moe_train_token_program_io"]
