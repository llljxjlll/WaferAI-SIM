"""Typed nonzero host seeds for a scoped public GEMM FP32 WGRAD timing fragment.

This proves that the exact physical FP16 operands and FP32 output BufferABI are
materialized.  The timing primitive does not compute a numerical gradient.
"""
from __future__ import annotations

import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LinkedProgramManifest, OperandKind, ProgramSymbolKind, RecordOpcode,
)
from ..schema.common import DType
from ..schema.ir2 import BufferOwnership
from ..schema.program_io import (
    ProgramBlob, ProgramIoContract, ProgramIoMode, ProgramIoPurpose,
    ProgramIoTargetKind, ProgramSramInitialization, ProgramSramTarget,
    ProgramOutputProbe, ProgramOutputComparison, ProgramOutputCapture,
)


def build_public_gemm_wgrad_fragment_program_io(
    manifest: LinkedProgramManifest, artifact_sha256: str, *,
    activation_payload_override: bytes | None = None,
) -> ProgramIoContract:
    manifest.validate("public_gemm_wgrad_source")
    definitions = {definition.symbol.id: (index, definition)
                   for index, definition in enumerate(
                       manifest.program_symbol_definitions)}
    core_ids = {binding.logical_core: binding.runtime_core_id
                for binding in manifest.core_bindings}
    blobs: dict[str, ProgramBlob] = {}
    entries = []
    probes = []
    source_count = 0
    declarations = {abi.id: abi for fragment in manifest.fragments
                    for abi in fragment.buffer_abi}
    for fragment in manifest.fragments:
        for stream in fragment.core_streams:
            records = [record for record in stream.records
                       if record.opcode is RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING]
            if len(records) != 1:
                raise SchemaError("one exact physical 0x25 source record per core required",
                                  path="public_gemm_wgrad.core_streams")
            literals = {operand.name: operand.literal_value
                        for operand in records[0].operands
                        if operand.kind is OperandKind.LITERAL}
            if (literals["m"], literals["n"], literals["k"]) != (8, 16, 4):
                raise SchemaError("physical 0x25 geometry differs from signed source fixture",
                                  path="public_gemm_wgrad.geometry")
            activation = struct.pack("<e", 0.5) * (literals["k"] * literals["m"])
            upstream = struct.pack("<e", 1.0) * (literals["k"] * literals["n"])
            if activation_payload_override is not None:
                if type(activation_payload_override) is not bytes or activation_payload_override != activation:
                    raise SchemaError("runtime FP16 activation blob disagrees with signed source fixture",
                                      path="public_gemm_wgrad.activation_payload")
                activation = activation_payload_override
            roots = {abi.binding_id: abi for abi in fragment.buffer_abi
                     if abi.logical_core == stream.logical_core}
            expected = {
                "abs_input": (activation, DType.FP16,
                              ProgramIoPurpose.ACTIVATION, BufferOwnership.BORROWED),
                "abs_data": (upstream, DType.FP16,
                             ProgramIoPurpose.ACTIVATION, BufferOwnership.BORROWED),
                "abs_output": (bytes(4 * literals["m"] * literals["n"]),
                               DType.FP32, ProgramIoPurpose.TIMING_PARTIAL,
                               BufferOwnership.OWNED),
            }
            for binding_id, (payload, dtype, purpose, ownership) in expected.items():
                abi = roots.get(binding_id)
                if (abi is None or abi.id not in declarations or abi.dtype is not dtype
                        or abi.ownership is not ownership or len(payload) != abi.size_bytes
                        or (binding_id != "abs_output" and not any(payload))):
                    raise SchemaError("physical 0x25 source violates typed BufferABI",
                                      path=f"public_gemm_wgrad.{binding_id}")
                matches = [(symbol_id, index, definition)
                           for symbol_id, (index, definition) in definitions.items()
                           if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
                           and definition.symbol.source_ref == abi.storage_id]
                if len(matches) != 1:
                    raise SchemaError("physical 0x25 source has no unique SRAM_ALLOC_AT label",
                                      path=f"public_gemm_wgrad.{binding_id}")
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
                if binding_id == "abs_output":
                    probes.append(ProgramOutputProbe.create(
                        target=target, offset_bytes=0, length_bytes=abi.size_bytes,
                        blob_ref=blob.id,
                        comparison=ProgramOutputComparison.EXACT_BYTES,
                        capture=ProgramOutputCapture.AFTER_PROGRAM,
                    ))
            source_count += 1
    if source_count != 1 or len(entries) != 3 or len(probes) != 1:
        raise SchemaError("scoped public 0x25 input source inventory incomplete",
                          path="public_gemm_wgrad.source_count")
    result = ProgramIoContract.create(
        producer_pass="public_gemm_wgrad_fragment_program_io",
        mode=ProgramIoMode.TIMING, source_manifest=manifest,
        program_artifact_sha256=artifact_sha256,
        blobs=tuple(blobs.values()), initializations=tuple(entries),
        output_probes=tuple(probes),
    )
    result.validate_against(manifest)
    return result


__all__ = ["build_public_gemm_wgrad_fragment_program_io"]
