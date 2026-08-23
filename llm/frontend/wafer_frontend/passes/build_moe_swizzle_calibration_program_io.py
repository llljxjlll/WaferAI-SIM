"""Actual-SHA ProgramIo for one isolated MoE calibration program."""

from __future__ import annotations

import re

from ..errors import SchemaError
from ..schema.artifact_manifest import ProgramSymbolKind
from ..schema.ir2 import BufferOwnership
from ..schema.program_io import (
    ProgramBlob,
    ProgramIoContract,
    ProgramIoMode,
    ProgramIoPurpose,
    ProgramIoTargetKind,
    ProgramOutputCapture,
    ProgramOutputComparison,
    ProgramOutputProbe,
    ProgramSramInitialization,
    ProgramSramTarget,
)
from ..schema.swizzle_moe_calibration_program import (
    MoeSwizzleCalibrationStandardLinkedProgram,
)


_PRODUCER = "build_moe_swizzle_calibration_program_io"


def _target(
    source: MoeSwizzleCalibrationStandardLinkedProgram,
    abi: object,
) -> ProgramSramTarget:
    definitions = source.manifest.program_symbol_definitions
    runtime = next(
        item.runtime_core_id
        for item in source.manifest.core_bindings
        if item.logical_core == abi.logical_core
    )
    if abi.ownership is BufferOwnership.BORROWED:
        candidates = tuple(
            (index, item)
            for index, item in enumerate(definitions)
            if item.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
            and item.symbol.source_ref == abi.binding_id
            and item.logical_cores == (abi.logical_core,)
        )
    else:
        candidates = tuple(
            (index, item)
            for index, item in enumerate(definitions)
            if item.symbol.kind is ProgramSymbolKind.SRAM_LABEL
            and item.symbol.source_ref == abi.storage_id
            and item.logical_cores == (abi.logical_core,)
        )
    if len(candidates) != 1:
        raise SchemaError(
            "calibration ABI does not resolve to one exact SRAM symbol",
            path="moe_swizzle_calibration_program_io.manifest",
        )
    index, definition = candidates[0]
    return ProgramSramTarget(
        kind=ProgramIoTargetKind.SRAM,
        runtime_core_id=runtime,
        program_symbol_ref=definition.symbol.id,
        finalized_symbol_index=index,
        expected_symbol_name=definition.name,
        buffer_abi_id=abi.id,
        storage_id=abi.storage_id,
        value_id=abi.value_id,
        tensor_slice=abi.tensor_slice,
        dtype=abi.dtype,
        layout=abi.layout,
    )


def _build(
    source: MoeSwizzleCalibrationStandardLinkedProgram,
    program_artifact_sha256: str,
) -> ProgramIoContract:
    abis = {item.id: item for item in source.fragment.buffer_abi}
    blobs: dict[str, ProgramBlob] = {}

    def zeros(size: int) -> ProgramBlob:
        blob = ProgramBlob.create(bytes(size))
        blobs.setdefault(blob.id, blob)
        return blob

    initializations = []
    for abi_id in source.source.input_buffer_abi_ids:
        abi = abis[abi_id]
        blob = zeros(abi.size_bytes)
        initializations.append(ProgramSramInitialization.create(
            target=_target(source, abi),
            offset_bytes=0,
            length_bytes=abi.size_bytes,
            blob_ref=blob.id,
            purpose=ProgramIoPurpose.ACTIVATION,
        ))
    probes = []
    for abi_id in source.source.output_buffer_abi_ids:
        abi = abis[abi_id]
        blob = zeros(abi.size_bytes)
        # Calibration programs execute the timing ISA path: compute records
        # account cycles but intentionally do not materialize result bytes in
        # SRAM. Seed the exact persistent terminal range so the boundary probe
        # witnesses both a live allocation and valid bytes. This remains
        # timing-only via TIMING_PARTIAL and never applies to scratch storage.
        initializations.append(ProgramSramInitialization.create(
            target=_target(source, abi),
            offset_bytes=0,
            length_bytes=abi.size_bytes,
            blob_ref=blob.id,
            purpose=ProgramIoPurpose.TIMING_PARTIAL,
        ))
        probes.append(ProgramOutputProbe.create(
            target=_target(source, abi),
            offset_bytes=0,
            length_bytes=abi.size_bytes,
            blob_ref=blob.id,
            comparison=ProgramOutputComparison.EXACT_BYTES,
            capture=ProgramOutputCapture.AFTER_PROGRAM,
        ))
    result = ProgramIoContract.create(
        producer_pass=_PRODUCER,
        mode=ProgramIoMode.TIMING,
        source_manifest=source.manifest,
        program_artifact_sha256=program_artifact_sha256,
        blobs=tuple(blobs.values()),
        initializations=tuple(initializations),
        output_probes=tuple(probes),
    )
    result.validate_against(source.manifest, "moe_swizzle_calibration_program_io")
    return result


def build_moe_swizzle_calibration_program_io(
    source: MoeSwizzleCalibrationStandardLinkedProgram,
    program_artifact_sha256: str,
) -> ProgramIoContract:
    if type(source) is not MoeSwizzleCalibrationStandardLinkedProgram:
        raise SchemaError("requires exact calibration wrapper", path="source")
    if source.program_io is not None:
        raise SchemaError(
            "ProgramIo must be built from a sidecar-free wrapper",
            path="source.program_io",
        )
    if (
        re.fullmatch(r"[0-9a-f]{64}", program_artifact_sha256) is None
        or program_artifact_sha256 == "0" * 64
    ):
        raise SchemaError(
            "calibration ProgramIo requires an actual nonzero artifact SHA",
            path="program_artifact_sha256",
        )
    source.validate("source")
    return _build(source, program_artifact_sha256)


def validate_moe_swizzle_calibration_program_io_against(
    contract: ProgramIoContract,
    source: MoeSwizzleCalibrationStandardLinkedProgram,
    path: str = "moe_swizzle_calibration_program_io",
) -> None:
    contract.validate_against(source.manifest, path)
    if contract != _build(source, contract.program_artifact_sha256):
        raise SchemaError("calibration ProgramIo is not deterministic", path=path)


__all__ = [
    "build_moe_swizzle_calibration_program_io",
    "validate_moe_swizzle_calibration_program_io_against",
]
