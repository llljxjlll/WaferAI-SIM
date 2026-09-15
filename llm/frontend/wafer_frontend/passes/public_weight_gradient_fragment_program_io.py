"""Strict host seed for one public 0x23 physical timing fragment.

The INT32 bytes come from the source record's verified index trace.  This is
scoped ProgramIO evidence for a physical fragment; it does not certify full
TRAIN IR1, gradient numerics, or cross-action backward dominance.
"""

from __future__ import annotations

import hashlib
import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LinkedProgramManifest, OperandKind, ProgramSymbolKind, RecordOpcode,
)
from ..schema.common import DType
from ..schema.program_io import (
    ProgramBlob, ProgramIoContract, ProgramIoMode, ProgramIoPurpose,
    ProgramIoTargetKind, ProgramSramInitialization, ProgramSramTarget,
    ProgramOutputProbe, ProgramOutputComparison, ProgramOutputCapture,
)
from ..schema.ir2 import BufferOwnership


def build_public_embedding_wgrad_fragment_program_io(
    manifest: LinkedProgramManifest,
    artifact_sha256: str,
    *,
    index_payload_override: bytes | None = None,
) -> ProgramIoContract:
    manifest.validate("public_embedding_wgrad_source")
    definitions = {definition.symbol.id: (index, definition)
                   for index, definition in enumerate(
                       manifest.program_symbol_definitions)}
    core_ids = {binding.logical_core: binding.runtime_core_id
                for binding in manifest.core_bindings}
    blobs = {}
    entries = []
    probes = []
    source_count = 0
    declarations = {abi.id: abi for fragment in manifest.fragments
                    for abi in fragment.buffer_abi}
    for fragment in manifest.fragments:
        for stream in fragment.core_streams:
            records = [record for record in stream.records
                       if record.opcode is RecordOpcode.EMBEDDING_TABLE_WGRAD_TIMING]
            if len(records) != 1:
                raise SchemaError("one physical 0x23 source record per core required",
                                  path="public_wgrad.core_streams")
            record = records[0]
            literals = {operand.name: operand.literal_value
                        for operand in record.operands
                        if operand.kind is OperandKind.LITERAL}
            rows = literals["rank_rows"]
            trace = tuple(literals[f"index{i:02d}"] for i in range(16))
            if (rows != 4 or trace[:rows] != (3, 3, 5, 7)
                    or any(trace[rows:]) or literals["vocab_size"] != 16
                    or literals["vocab_rows"] != 8 or literals["hidden_size"] != 8):
                raise SchemaError("fragment source trace/profile differs from signed nonzero canary",
                                  path="public_wgrad.source_trace")
            index_bytes = struct.pack("<4i", *trace[:rows])
            if index_payload_override is not None:
                if type(index_payload_override) is not bytes or index_payload_override != index_bytes:
                    raise SchemaError("runtime INT32 token blob disagrees with source index trace",
                                      path="public_wgrad.index_payload")
                index_bytes = index_payload_override
            seed_by_binding = {
                "abs_input": (index_bytes, DType.INT32, ProgramIoPurpose.ACTIVATION),
                "abs_data": (struct.pack("<e", 0.5) * 64, DType.FP16,
                             ProgramIoPurpose.WEIGHT),
                "abs_aux": (struct.pack("<e", 1.0) * 32, DType.FP16,
                            ProgramIoPurpose.ACTIVATION),
            }
            for binding_id, (payload, dtype, purpose) in seed_by_binding.items():
                roots = [abi for abi in fragment.buffer_abi
                         if abi.logical_core == stream.logical_core
                         and abi.binding_id == binding_id]
                if len(roots) != 1:
                    raise SchemaError("physical source BufferABI root missing",
                                      path=f"public_wgrad.{binding_id}")
                abi = roots[0]
                if (abi.id not in declarations or abi.dtype is not dtype
                        or abi.ownership is not BufferOwnership.BORROWED
                        or len(payload) != abi.size_bytes or not any(payload)):
                    raise SchemaError("physical source seed violates typed BORROWED BufferABI",
                                      path=f"public_wgrad.{binding_id}")
                matches = [(symbol_id, index, definition)
                           for symbol_id, (index, definition) in definitions.items()
                           if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
                           and definition.symbol.source_ref == abi.storage_id]
                if len(matches) != 1:
                    raise SchemaError("physical source has no unique SRAM_ALLOC_AT label",
                                      path=f"public_wgrad.{binding_id}")
                symbol_id, symbol_index, definition = matches[0]
                target = ProgramSramTarget(
                    ProgramIoTargetKind.SRAM, core_ids[stream.logical_core],
                    symbol_id, symbol_index, definition.name,
                    abi.id, abi.storage_id, abi.value_id, abi.tensor_slice,
                    abi.dtype, abi.layout,
                )
                blob = ProgramBlob.create(payload)
                blobs[blob.id] = blob
                entries.append(ProgramSramInitialization.create(
                    target=target, offset_bytes=0, length_bytes=abi.size_bytes,
                    blob_ref=blob.id, purpose=purpose,
                ))
            outputs = [abi for abi in fragment.buffer_abi
                       if abi.logical_core == stream.logical_core
                       and abi.binding_id == "abs_output"]
            if (len(outputs) != 1 or outputs[0].dtype is not DType.FP32
                    or outputs[0].ownership is not BufferOwnership.OWNED
                    or outputs[0].size_bytes != literals["vocab_rows"] *
                       literals["hidden_size"] * 4):
                raise SchemaError("0x23 needs independent FP32 table-gradient BufferABI",
                                  path="public_wgrad.gradient")
            abi = outputs[0]
            matches = [(symbol_id, index, definition)
                       for symbol_id, (index, definition) in definitions.items()
                       if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
                       and definition.symbol.source_ref == abi.storage_id]
            if len(matches) != 1:
                raise SchemaError("FP32 output has no unique physical SRAM label",
                                  path="public_wgrad.gradient")
            symbol_id, symbol_index, definition = matches[0]
            target = ProgramSramTarget(
                ProgramIoTargetKind.SRAM, core_ids[stream.logical_core],
                symbol_id, symbol_index, definition.name,
                abi.id, abi.storage_id, abi.value_id, abi.tensor_slice,
                abi.dtype, abi.layout,
            )
            # The timing primitive charges physical work but performs no FP32
            # scatter arithmetic.  A zero capture is an execution probe only.
            probe_blob = ProgramBlob.create(bytes(abi.size_bytes))
            blobs[probe_blob.id] = probe_blob
            entries.append(ProgramSramInitialization.create(
                target=target, offset_bytes=0, length_bytes=abi.size_bytes,
                blob_ref=probe_blob.id,
                purpose=ProgramIoPurpose.TIMING_PARTIAL,
            ))
            probes.append(ProgramOutputProbe.create(
                target=target, offset_bytes=0, length_bytes=abi.size_bytes,
                blob_ref=probe_blob.id,
                comparison=ProgramOutputComparison.EXACT_BYTES,
                capture=ProgramOutputCapture.AFTER_PROGRAM,
            ))
            source_count += 1
    if source_count == 0 or len(entries) != 4 * source_count:
        raise SchemaError("physical 0x23 input source inventory incomplete",
                          path="public_wgrad.source_count")
    contract = ProgramIoContract.create(
        producer_pass="public_embedding_wgrad_fragment_program_io",
        mode=ProgramIoMode.TIMING, source_manifest=manifest,
        program_artifact_sha256=artifact_sha256,
        blobs=tuple(blobs.values()), initializations=tuple(entries),
        output_probes=tuple(probes),
    )
    contract.validate_against(manifest)
    return contract


__all__ = ["build_public_embedding_wgrad_fragment_program_io"]
