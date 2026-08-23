"""Exact ProgramIo sidecar for one whole MoE Swizzle standard program."""

from __future__ import annotations

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    ProgramSymbolKind,
    RecordOpcode,
    SemanticOperandId,
)
from ..schema.ir2 import BufferOwnership
from ..schema.program_io import (
    ProgramBlob,
    ProgramHbmTarget,
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
from ..schema.swizzle_moe_standard import MoeSwizzleStandardLinkedProgram


_PRODUCER = "build_moe_swizzle_program_io"


def _timing_sentinel_ranges(
    source: MoeSwizzleStandardLinkedProgram,
    roots_by_binding: dict[str, object],
) -> tuple[tuple[object, int, int], ...]:
    """Return exact side-effect-free compute ranges observed after compute.

    Manual-memory compute accounts timing but intentionally does not write
    SRAM bytes.  Downstream DTE sources and directly probed terminal outputs
    must therefore be seeded explicitly; otherwise validity depends on stale
    bytes from an unrelated prior occupant.
    """

    symbols = {item.id: item for item in source.fragment.program_symbols}
    abi_by_binding = {
        item.binding_id: item for item in source.fragment.buffer_abi
    }

    def binding_for_operand(operand: object) -> tuple[object, object]:
        symbol = symbols.get(operand.symbol_ref)
        abi = (
            abi_by_binding.get(symbol.source_ref)
            if symbol is not None
            and symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
            else None
        )
        if abi is None:
            raise SchemaError(
                "manual-memory operand lacks one exact BufferABI binding",
                path="moe_swizzle_program_io.fragment.core_streams",
            )
        root = roots_by_binding.get(
            abi.alias_of if abi.alias_of is not None else abi.binding_id
        )
        if root is None or root.storage_id != abi.storage_id:
            raise SchemaError(
                "manual-memory operand lacks one exact physical root",
                path="moe_swizzle_program_io.fragment.buffer_abi",
            )
        return root, abi

    compute_outputs = []
    dte_sources = set()
    for stream in source.fragment.core_streams:
        for record in stream.records:
            if record.opcode in (RecordOpcode.MATMUL, RecordOpcode.SWIGLU):
                output = tuple(
                    operand for operand in record.operands
                    if operand.operand_id
                    is SemanticOperandId.COMPUTE_OUTPUT_ADDRESS
                )
                if len(output) != 1:
                    raise SchemaError(
                        "manual-memory compute lacks one exact output operand",
                        path="moe_swizzle_program_io.fragment.core_streams",
                    )
                compute_outputs.append(binding_for_operand(output[0]))
            elif record.opcode in (RecordOpcode.DTE_SEND, RecordOpcode.DTE_ISSUE):
                inputs = tuple(
                    operand for operand in record.operands
                    if operand.operand_id is SemanticOperandId.SOURCE_ADDRESS
                )
                if len(inputs) != 1:
                    raise SchemaError(
                        "DTE source lacks one exact source operand",
                        path="moe_swizzle_program_io.fragment.core_streams",
                    )
                root, _abi = binding_for_operand(inputs[0])
                dte_sources.add(root.storage_id)

    actual_dte = {
        root.storage_id for root, _abi in compute_outputs
        if root.storage_id in dte_sources
    }
    has_tape = any(
        item.layout == "moe_swizzle_terminal_tape_root/v1"
        for item in roots_by_binding.values()
    )
    expected = {
        item.storage_id for item in roots_by_binding.values()
        if item.layout == "moe_swizzle_combine_output_root/v1"
        or (
            has_tape
            and item.layout == "moe_swizzle_swiglu_output_root/v1"
        )
    }
    if actual_dte != expected:
        raise SchemaError(
            "timing sentinel roots do not exactly match compute-to-DTE storage",
            path="moe_swizzle_program_io.fragment.buffer_abi",
        )
    ranges = set()
    for root, abi in compute_outputs:
        dte_observed = root.storage_id in dte_sources
        if dte_observed:
            ranges.add((
                root,
                abi.region_offset_bytes,
                abi.region_offset_bytes + abi.size_bytes,
            ))
    terminal_values = {item.value_ref for item in source.workload.terminals}
    terminal_aliases = tuple(
        abi for abi in source.fragment.buffer_abi
        if abi.alias_of is not None
        and abi.value_id in terminal_values
        and abi.layout in (
            "moe_swizzle_terminal_combined_subview/v1",
            "moe_swizzle_terminal_tape_subview/v1",
        )
    )
    direct_terminal_outputs = tuple(
        (root, abi) for root, abi in compute_outputs
        if abi.layout in (
            "moe_swizzle_terminal_combined_subview/v1",
            "moe_swizzle_terminal_tape_subview/v1",
        )
        and root.storage_id not in dte_sources
    )
    covered_terminal_values = set()
    for root, abi in direct_terminal_outputs:
        end = abi.region_offset_bytes + abi.size_bytes
        covered = tuple(
            terminal for terminal in terminal_aliases
            if terminal.storage_id == root.storage_id
            and abi.region_offset_bytes <= terminal.region_offset_bytes
            and terminal.region_offset_bytes + terminal.size_bytes <= end
        )
        if not covered:
            raise SchemaError(
                "direct terminal compute output lacks exact probe subviews",
                path="moe_swizzle_program_io.fragment.buffer_abi",
            )
        ranges.add((root, abi.region_offset_bytes, end))
        covered_terminal_values.update(item.value_id for item in covered)
    expected_direct_terminal_values = {
        terminal.value_id for terminal in terminal_aliases
        if any(
            terminal.storage_id == root.storage_id
            and abi.region_offset_bytes <= terminal.region_offset_bytes
            and terminal.region_offset_bytes + terminal.size_bytes
            <= abi.region_offset_bytes + abi.size_bytes
            for root, abi in direct_terminal_outputs
        )
    }
    if covered_terminal_values != expected_direct_terminal_values:
        raise SchemaError(
            "direct terminal compute outputs do not exactly cover probe subviews",
            path="moe_swizzle_program_io.fragment.buffer_abi",
        )
    if not ranges:
        raise SchemaError(
            "whole timing program lacks observed compute output ranges",
            path="moe_swizzle_program_io.fragment.buffer_abi",
        )
    return tuple(sorted(
        ranges,
        key=lambda item: (item[0].id, item[1], item[2]),
    ))


def _sram_target(
    source: MoeSwizzleStandardLinkedProgram,
    abi: object,
    root: object,
) -> ProgramSramTarget:
    if root.ownership is BufferOwnership.BORROWED:
        root_alias_bindings = {
            item.binding_id for item in source.fragment.buffer_abi
            if item.alias_of == root.binding_id
            and item.storage_id == root.storage_id
            and item.region_offset_bytes == root.region_offset_bytes
        }
        definitions = tuple(sorted((
            (index, definition)
            for index, definition in enumerate(
                source.manifest.program_symbol_definitions
            )
            if definition.symbol.kind
            is ProgramSymbolKind.ABSOLUTE_ADDRESS
            and definition.symbol.source_ref in root_alias_bindings
            and root.logical_core in definition.logical_cores
        ), key=lambda item: item[1].symbol.id))
    else:
        definitions = tuple(
            (index, definition)
            for index, definition in enumerate(
                source.manifest.program_symbol_definitions
            )
            if definition.symbol.kind is ProgramSymbolKind.SRAM_LABEL
            and definition.symbol.source_ref == root.storage_id
        )
    if not definitions:
        raise SchemaError(
            "BufferABI root does not resolve to one exact SRAM address symbol",
            path="moe_swizzle_program_io.manifest.program_symbol_definitions",
        )
    index, definition = definitions[0]
    runtime = tuple(
        item.runtime_core_id for item in source.manifest.core_bindings
        if item.logical_core == root.logical_core
    )
    if len(runtime) != 1:
        raise SchemaError(
            "BufferABI root does not resolve to one runtime core",
            path="moe_swizzle_program_io.manifest.core_bindings",
        )
    return ProgramSramTarget(
        kind=ProgramIoTargetKind.SRAM,
        runtime_core_id=runtime[0],
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


def _hbm_target(
    source: MoeSwizzleStandardLinkedProgram,
    abi: object,
) -> ProgramHbmTarget:
    definitions = tuple(
        (index, definition)
        for index, definition in enumerate(
            source.manifest.program_symbol_definitions
        )
        if definition.symbol.kind is ProgramSymbolKind.ABSOLUTE_ADDRESS
        and definition.symbol.source_ref == abi.hbm_binding_ref
    )
    if len(definitions) != 1:
        raise SchemaError(
            "StateABI does not resolve to one exact HBM symbol",
            path="moe_swizzle_program_io.manifest.program_symbol_definitions",
        )
    index, definition = definitions[0]
    return ProgramHbmTarget(
        kind=ProgramIoTargetKind.HBM,
        program_symbol_ref=definition.symbol.id,
        finalized_symbol_index=index,
        expected_symbol_name=definition.name,
        state_abi_id=abi.id,
        state_ref=abi.state_ref,
        hbm_binding_ref=abi.hbm_binding_ref,
    )


def _build_moe_swizzle_program_io(
    source: MoeSwizzleStandardLinkedProgram,
    program_artifact_sha256: str,
) -> ProgramIoContract:
    """Build exact zero-seed timing IO after source validation."""

    roots_by_binding = {
        item.binding_id: item for item in source.fragment.buffer_abi
        if item.alias_of is None
    }
    aliases_by_value: dict[str, list[object]] = {}
    for item in source.fragment.buffer_abi:
        if item.alias_of is not None:
            aliases_by_value.setdefault(item.value_id, []).append(item)

    blobs: dict[str, ProgramBlob] = {}

    def zeros(size_bytes: int) -> ProgramBlob:
        blob = ProgramBlob.create(bytes(size_bytes))
        blobs.setdefault(blob.id, blob)
        return blob

    initializations = []
    for root in sorted(roots_by_binding.values(), key=lambda item: item.id):
        if root.ownership is not BufferOwnership.BORROWED:
            continue
        blob = zeros(root.size_bytes)
        initializations.append(ProgramSramInitialization.create(
            target=_sram_target(source, root, root),
            offset_bytes=0,
            length_bytes=root.size_bytes,
            blob_ref=blob.id,
            purpose=ProgramIoPurpose.ACTIVATION,
        ))

    sentinel_ranges = _timing_sentinel_ranges(source, roots_by_binding)
    borrowed_roots = tuple(
        item for item in roots_by_binding.values()
        if item.ownership is BufferOwnership.BORROWED
    )
    physical_keys = {
        (item.logical_core, item.region_ref)
        for item in (
            *(entry[0] for entry in sentinel_ranges),
            *borrowed_roots,
        )
    }
    for logical_core, region_ref in sorted(
        physical_keys,
        key=lambda item: (
            item[0].die_id, item[0].local_core_id, item[1],
        ),
    ):
        desired = tuple(
            item for item in sentinel_ranges
            if item[0].logical_core == logical_core
            and item[0].region_ref == region_ref
        )
        if not desired:
            continue
        already_seeded = tuple(
            item for item in borrowed_roots
            if item.logical_core == logical_core
            and item.region_ref == region_ref
        )
        boundaries = sorted({
            value
            for start, end in (
                *((item[1], item[2]) for item in desired),
                *((
                    item.region_offset_bytes,
                    item.region_offset_bytes + item.size_bytes,
                ) for item in already_seeded),
            )
            for value in (start, end)
        })
        for start, end in zip(boundaries, boundaries[1:]):
            candidates = tuple(
                item for item in desired
                if item[1] <= start and end <= item[2]
            )
            if not candidates or any(
                item.region_offset_bytes <= start
                and end <= item.region_offset_bytes + item.size_bytes
                for item in already_seeded
            ):
                continue
            root = min(
                (item[0] for item in candidates),
                key=lambda item: item.id,
            )
            blob = zeros(end - start)
            initializations.append(ProgramSramInitialization.create(
                target=_sram_target(source, root, root),
                offset_bytes=start - root.region_offset_bytes,
                length_bytes=end - start,
                blob_ref=blob.id,
                purpose=ProgramIoPurpose.TIMING_PARTIAL,
            ))

    for state in sorted(source.fragment.state_abi, key=lambda item: item.id):
        blob = zeros(state.size_bytes)
        initializations.append(ProgramSramInitialization.create(
            target=_hbm_target(source, state),
            offset_bytes=0,
            length_bytes=state.size_bytes,
            blob_ref=blob.id,
            purpose=ProgramIoPurpose.STATE,
        ))

    probes = []
    for terminal in source.workload.terminals:
        expected_layout = (
            f"moe_swizzle_terminal_{terminal.kind.value}_subview/v1"
        )
        candidates = tuple(
            item for item in aliases_by_value.get(terminal.value_ref, ())
            if item.ownership is BufferOwnership.ALIASED
            and item.layout == expected_layout
            and item.size_bytes == terminal.bytes
            and item.dtype is terminal.dtype
            and item.tensor_slice.shape == terminal.shape
        )
        if len(candidates) != 1:
            raise SchemaError(
                "semantic terminal does not resolve to one exact probe subview",
                path=(
                    "moe_swizzle_program_io.workload.terminals"
                    f"[{terminal.value_ref!r}]"
                ),
            )
        abi = candidates[0]
        root = roots_by_binding.get(abi.alias_of)
        if root is None or root.ownership is not BufferOwnership.OWNED:
            raise SchemaError(
                "terminal probe subview lacks one exact OWNED root",
                path="moe_swizzle_program_io.fragment.buffer_abi",
            )
        offset = abi.region_offset_bytes - root.region_offset_bytes
        blob = zeros(terminal.bytes)
        probes.append(ProgramOutputProbe.create(
            target=_sram_target(source, abi, root),
            offset_bytes=offset,
            length_bytes=terminal.bytes,
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
    result.validate_against(source.manifest, "moe_swizzle_program_io")
    return result


def build_moe_swizzle_program_io(
    source: MoeSwizzleStandardLinkedProgram,
    program_artifact_sha256: str,
) -> ProgramIoContract:
    """Build exact zero-seed timing IO from a sidecar-free whole program."""

    if type(source) is not MoeSwizzleStandardLinkedProgram:
        raise SchemaError(
            "source must be an exact MoE standard linked program",
            path="source",
        )
    if source.program_io is not None:
        raise SchemaError(
            "ProgramIo must be built from the program without a sidecar",
            path="source.program_io",
        )
    source.validate("source")
    return _build_moe_swizzle_program_io(source, program_artifact_sha256)


def validate_moe_swizzle_program_io_against(
    contract: ProgramIoContract,
    source: MoeSwizzleStandardLinkedProgram,
    path: str = "moe_swizzle_program_io",
) -> None:
    """Cross-check one sidecar against typed terminal and storage truth."""

    if type(contract) is not ProgramIoContract:
        raise SchemaError("requires exact ProgramIoContract", path=path)
    contract.validate_against(source.manifest, path)
    expected = _build_moe_swizzle_program_io(
        source, contract.program_artifact_sha256,
    )
    if contract != expected:
        raise SchemaError(
            "ProgramIo is not the deterministic whole-workload rebuild",
            path=path,
        )


__all__ = [
    "build_moe_swizzle_program_io",
    "validate_moe_swizzle_program_io_against",
]
