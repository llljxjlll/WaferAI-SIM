"""Signed HBM weight and SRAM upstream seeds for a scoped FP16 GEMM dX timing record."""
from __future__ import annotations

import struct

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    FragmentKind, LinkedProgramManifest, OperandKind, ProgramSymbolKind,
    RecordOpcode, SemanticOperandId,
)
from ..schema.common import DType
from ..schema.ir2 import BufferOwnership
from ..schema.persistent_state import PersistentStateAccess, StateKind
from ..schema.program_io import (
    ProgramBlob, ProgramHbmTarget, ProgramIoContract, ProgramIoMode,
    ProgramIoPurpose, ProgramIoTargetKind, ProgramOutputCapture,
    ProgramOutputComparison, ProgramOutputProbe, ProgramSramInitialization,
    ProgramSramTarget,
)


def build_public_gemm_dx_fragment_program_io(
    manifest: LinkedProgramManifest, artifact_sha256: str, *,
    weight_payload_override: bytes | None = None,
    upstream_payload_override: bytes | None = None,
) -> ProgramIoContract:
    manifest.validate("public_gemm_dx_source")
    if len(manifest.fragments) != 2 or len(manifest.core_bindings) != 1:
        raise SchemaError("one compute and one StateIO fragment required",
                          path="public_gemm_dx.fragments")
    compute = next((f for f in manifest.fragments if f.kind is FragmentKind.COARSE), None)
    state_leaf = next((f for f in manifest.fragments if f.kind is FragmentKind.STATE_IO), None)
    if compute is None or state_leaf is None or len(state_leaf.state_abi) != 1:
        raise SchemaError("exact 0x26 StateIO source absent",
                          path="public_gemm_dx.fragments")
    core = manifest.core_bindings[0].logical_core
    if (len(compute.core_streams) != 1 or len(state_leaf.core_streams) != 1
            or compute.core_streams[0].logical_core != core
            or state_leaf.core_streams[0].logical_core != core):
        raise SchemaError("source fragment core differs from physical binding",
                          path="public_gemm_dx.core")
    records = compute.core_streams[0].records
    dx_records = [record for record in records
                  if record.opcode is RecordOpcode.GEMM_DX_TIMING]
    loads = [record for record in state_leaf.core_streams[0].records
             if record.opcode is RecordOpcode.LSU_LOAD]
    if len(dx_records) != 1 or len(loads) != 1:
        raise SchemaError("one named dX and one blocking HBM load required",
                          path="public_gemm_dx.records")
    literals = {op.name: op.literal_value for op in dx_records[0].operands
                if op.kind is OperandKind.LITERAL}
    if (literals["m"], literals["n"], literals["k"]) != (8, 16, 4):
        raise SchemaError("physical dX tile differs from signed source",
                          path="public_gemm_dx.geometry")
    w_address = next(op.symbol_ref for op in dx_records[0].operands
                     if op.name == "weight_address")
    load_destination = next(op.symbol_ref for op in loads[0].operands
                            if op.name == "destination_address")
    if w_address != load_destination:
        raise SchemaError("HBM load does not feed the dX weight operand",
                          path="public_gemm_dx.weight_address")
    linked = manifest.core_streams[0].records
    load_position = next((i for i, ref in enumerate(linked)
                          if ref.fragment_id == state_leaf.id and
                          state_leaf.core_streams[0].records[
                              ref.fragment_record_index].opcode is RecordOpcode.LSU_LOAD), -1)
    dx_position = next((i for i, ref in enumerate(linked)
                        if ref.fragment_id == compute.id and
                        compute.core_streams[0].records[
                            ref.fragment_record_index].opcode is RecordOpcode.GEMM_DX_TIMING), -1)
    if load_position < 0 or dx_position <= load_position:
        raise SchemaError("StateABI load must dominate named dX producer",
                          path="public_gemm_dx.order")
    state = state_leaf.state_abi[0]
    if (state.state_ref != "head_weight_state" or
            state.hbm_binding_ref != "head_weight_hbm" or
            state.kind is not StateKind.PARAMETER or
            state.access is not PersistentStateAccess.READ_ONLY or
            state.dtype is not DType.FP16 or state.shape != (8, 16) or
            state.size_bytes != 256):
        raise SchemaError("source FP16 parameter StateABI mismatch",
                          path="public_gemm_dx.state_abi")
    definition_by_id = {definition.symbol.id: (index, definition)
                        for index, definition in enumerate(
                            manifest.program_symbol_definitions)}
    hbm_matches = [(symbol_id, index, definition)
                   for symbol_id, (index, definition) in definition_by_id.items()
                   if definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
                   and definition.symbol.source_ref == state.hbm_binding_ref]
    if len(hbm_matches) != 1:
        raise SchemaError("StateABI has no unique final HBM symbol",
                          path="public_gemm_dx.hbm")
    hbm_id, hbm_index, hbm_definition = hbm_matches[0]
    weight = struct.pack("<e", 0.25) * (literals["m"] * literals["n"])
    upstream = struct.pack("<e", 1.0) * (literals["k"] * literals["n"])
    if weight_payload_override is not None and weight_payload_override != weight:
        raise SchemaError("runtime FP16 weight blob disagrees with signed StateABI source",
                          path="public_gemm_dx.weight_payload")
    if upstream_payload_override is not None and upstream_payload_override != upstream:
        raise SchemaError("runtime FP16 dY blob disagrees with signed producer",
                          path="public_gemm_dx.upstream_payload")
    roots = {abi.binding_id: abi for abi in compute.buffer_abi
             if abi.logical_core == core}
    for binding_id, dtype, ownership, size in (
        ("abs_input", DType.FP16, BufferOwnership.OWNED, 256),
        ("abs_data", DType.FP16, BufferOwnership.BORROWED, 128),
        ("abs_output", DType.FP16, BufferOwnership.OWNED, 64),
    ):
        abi = roots.get(binding_id)
        if (abi is None or abi.dtype is not dtype or
                abi.ownership is not ownership or abi.size_bytes != size):
            raise SchemaError("typed physical BufferABI extent mismatch",
                              path=f"public_gemm_dx.{binding_id}")
    if state_leaf.buffer_abi != (roots["abs_input"],):
        raise SchemaError("StateIO destination must reuse exact weight BufferABI",
                          path="public_gemm_dx.weight_abi")
    blobs: dict[str, ProgramBlob] = {}
    entries: list[ProgramSramInitialization] = []
    probes: list[ProgramOutputProbe] = []
    hbm_target = ProgramHbmTarget(
        ProgramIoTargetKind.HBM, hbm_id, hbm_index,
        hbm_definition.name, state.id, state.state_ref, state.hbm_binding_ref,
    )
    blob = ProgramBlob.create(weight)
    blobs[blob.id] = blob
    entries.append(ProgramSramInitialization.create(
        target=hbm_target, offset_bytes=0, length_bytes=state.size_bytes,
        blob_ref=blob.id, purpose=ProgramIoPurpose.STATE,
    ))
    for binding_id, payload, purpose in (
        ("abs_data", upstream, ProgramIoPurpose.ACTIVATION),
        ("abs_output", bytes(64), ProgramIoPurpose.TIMING_PARTIAL),
    ):
        abi = roots[binding_id]
        matches = [(symbol_id, index, definition)
                   for symbol_id, (index, definition) in definition_by_id.items()
                   if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
                   and definition.symbol.source_ref == abi.storage_id]
        if len(matches) != 1 or len(payload) != abi.size_bytes:
            raise SchemaError("SRAM source lacks exact label/extent",
                              path=f"public_gemm_dx.{binding_id}")
        symbol_id, symbol_index, definition = matches[0]
        target = ProgramSramTarget(
            ProgramIoTargetKind.SRAM, manifest.core_bindings[0].runtime_core_id,
            symbol_id, symbol_index, definition.name, abi.id,
            abi.storage_id, abi.value_id, abi.tensor_slice, abi.dtype, abi.layout,
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
    result = ProgramIoContract.create(
        producer_pass="public_gemm_dx_fragment_program_io",
        mode=ProgramIoMode.TIMING, source_manifest=manifest,
        program_artifact_sha256=artifact_sha256,
        blobs=tuple(blobs.values()), initializations=tuple(entries),
        output_probes=tuple(probes),
    )
    result.validate_against(manifest)
    return result


__all__ = ["build_public_gemm_dx_fragment_program_io"]
