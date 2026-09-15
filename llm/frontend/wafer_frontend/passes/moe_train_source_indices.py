"""Derive native Embedding WGRAD index_trace from the *real* forward IO seed.

This producer is source-bound to a validated linked forward and finalized
ProgramIO contract.  It cannot choose a convenient fake token-ID sequence.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest
from ..schema.common import DType
from ..schema.program_io import ProgramIoContract, ProgramSramTarget
from ..schema.serde import canonical_digest
from ..schema.moe_training_ir0_workloads import EmbeddingTableWgradWorkload
from ..schema.moe_train_token_input_case import MoeTrainTokenInputCase
from ..schema.workload_materialization import WorkloadMaterializationManifest


@dataclass(frozen=True, slots=True)
class SourceTokenIndexTrace:
    source_linked_manifest_id: str
    source_linked_manifest_digest: str
    source_program_io_id: str
    source_program_io_digest: str
    token_buffer_abi_id: str
    token_seed_blob_id: str
    token_seed_blob_sha256: str
    token_value_id: str
    rank_rows: int
    active_ids: tuple[int, ...]
    index_trace: tuple[int, ...]


def derive_forward_embedding_index_trace(
    manifest: LinkedProgramManifest,
    contract: ProgramIoContract,
    *,
    token_value_id: str,
    rank_rows: int,
    vocab_size: int,
    vocab_start: int,
    vocab_rows: int,
    hidden_size: int,
) -> SourceTokenIndexTrace:
    """Bind 16-slot native indices to one exact INT32 source SRAM init blob."""
    if (type(contract) is not ProgramIoContract or not token_value_id):
        raise SchemaError("real validated ProgramIO sidecar/source token is required",
                          path="moe_train_source_indices")
    contract.validate_against(manifest, "moe_train_source_indices.contract")
    matches = [(entry, entry.target)
               for entry in contract.initializations
               if type(entry.target) is ProgramSramTarget
               and entry.target.value_id == token_value_id]
    if len(matches) != 1:
        raise SchemaError("one real borrowed INT32 forward token seed is required",
                          path="moe_train_source_indices.initializations")
    entry, target = matches[0]
    abis = {abi.id: abi for fragment in manifest.fragments
            for abi in fragment.buffer_abi}
    abi = abis.get(target.buffer_abi_id)
    if (abi is None or abi.logical_core.die_id != 0
            or abi.value_id != token_value_id or abi.dtype is not DType.INT32
            or abi.tensor_slice.shape != (rank_rows,)
            or abi.size_bytes != 4 * rank_rows
            or target.tensor_slice.shape != (rank_rows,)
            or entry.offset_bytes != 0 or entry.length_bytes != abi.size_bytes):
        raise SchemaError("token seed is not the complete exact source index BufferABI",
                          path="moe_train_source_indices.token_buffer")
    blobs = {blob.id: blob for blob in contract.blobs}
    blob = blobs[entry.blob_ref]
    payload = base64.b64decode(blob.bytes_base64, validate=True)
    if len(payload) != rank_rows * 4:
        raise SchemaError("real INT32 token payload byte extent differs",
                          path="moe_train_source_indices.blob")
    active = struct.unpack(f"<{rank_rows}i", payload)
    trace = (*active, *((0,) * (16 - rank_rows)))
    declared = EmbeddingTableWgradWorkload(
        logical_rows=rank_rows, rank_rows=rank_rows, tp_degree=1,
        vocab_size=vocab_size, vocab_start=vocab_start,
        vocab_rows=vocab_rows, hidden_size=hidden_size,
        index_trace=trace,
    )
    declared.validate("moe_train_source_indices.native_contract")
    return SourceTokenIndexTrace(
        manifest.id, canonical_digest(manifest), contract.id,
        canonical_digest(contract), abi.id, blob.id, blob.sha256,
        token_value_id, rank_rows, active, trace,
    )


def require_embedding_trace_matches_source_input(
    trace: SourceTokenIndexTrace,
    source: WorkloadMaterializationManifest,
    case: MoeTrainTokenInputCase,
    *,
    step: int,
) -> None:
    """Compare independent signed case input with actual IO-decoded payload."""
    case.validate_against(source)
    if (type(step) is not int or step not in (0, 1)
            or trace.rank_rows != len(case.tokens_by_step[step])
            or trace.active_ids != case.tokens_by_step[step]
            or trace.index_trace != (*trace.active_ids,
                                     *((0,) * (16 - trace.rank_rows)))
            or trace.token_seed_blob_sha256 != hashlib.sha256(
                struct.pack(f"<{trace.rank_rows}i",
                            *case.tokens_by_step[step])).hexdigest()):
        raise SchemaError("native token trace/actual IO blob differs from source-signed case",
                          path=f"moe_train_source_indices.step{step}")


__all__ = ["SourceTokenIndexTrace", "derive_forward_embedding_index_trace",
           "require_embedding_trace_matches_source_input"]
