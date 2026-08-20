"""Exact fixed-offset SRAM lifecycle lowering for leaf command fragments."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    AddressRelocation,
    BufferABI,
    CommandFragment,
    CoreFragmentStream,
    ProgramSymbol,
    ProgramSymbolKind,
    RecordOpcode,
    RecordOperand,
    RelocatableRecord,
    RuntimeOperandField,
    RuntimeRelocation,
    SemanticOperandId,
)
from ..schema.common import stable_artifact_id
from ..schema.global_action import GlobalAction, LogicalCoreRef
from ..schema.ir1 import SramAllocator
from ..schema.ir2 import BufferOwnership
from .context import LoweringContext


_RUNTIME_FIELD_ORDER = {
    field: index for index, field in enumerate(RuntimeOperandField)
}


def _storage_label_symbol(
    schedule_id: str,
    runtime_core_id: int,
    storage_id: str,
) -> ProgramSymbol:
    semantic = {
        "schedule_id": schedule_id,
        "runtime_core_id": runtime_core_id,
        "storage_id": storage_id,
        "kind": int(ProgramSymbolKind.SRAM_LABEL),
    }
    return ProgramSymbol(
        stable_artifact_id(
            "program_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        ProgramSymbolKind.SRAM_LABEL,
        storage_id,
    )


def _region_symbol(region_ref: str) -> ProgramSymbol:
    semantic = {
        "region_ref": region_ref,
        "kind": int(ProgramSymbolKind.SRAM_REGION),
    }
    return ProgramSymbol(
        stable_artifact_id(
            "program_symbol",
            semantic,
            schema_version=COMMAND_FRAGMENT_SCHEMA_VERSION,
        ),
        ProgramSymbolKind.SRAM_REGION,
        region_ref,
    )


def _core_runtime_id(context: LoweringContext, logical_core: LogicalCoreRef) -> int:
    die = next(
        (item for item in context.ir1.fabric.dies if item.id == logical_core.die_id),
        None,
    )
    if die is None:
        raise SchemaError("lifecycle core references an unknown die", path="fragment")
    core = next(
        (
            item
            for item in die.cores
            if item.local_core_id == logical_core.local_core_id
        ),
        None,
    )
    if core is None:
        raise SchemaError("lifecycle core references an unknown core", path="fragment")
    return core.runtime_core_id


def _region(context: LoweringContext, abi: BufferABI):
    die = next(item for item in context.ir1.fabric.dies if item.id == abi.logical_core.die_id)
    core = next(
        item
        for item in die.cores
        if item.local_core_id == abi.logical_core.local_core_id
    )
    profile = next(
        item
        for item in context.ir1.fabric.sram_profiles
        if item.id == core.sram_profile_ref
    )
    region = next((item for item in profile.regions if item.id == abi.region_ref), None)
    if region is None:
        raise SchemaError(
            "BufferABI references an unknown named SRAM region",
            path="fragment.buffer_abi",
        )
    if region.allocator is not SramAllocator.BLOCK:
        raise SchemaError(
            "SRAM_ALLOC_AT requires a BLOCK named region",
            path="fragment.buffer_abi",
        )
    if abi.region_offset_bytes + abi.size_bytes > region.size_bytes:
        raise SchemaError(
            "BufferABI span exceeds the named SRAM region",
            path="fragment.buffer_abi",
        )
    return region


def _order_key(abi: BufferABI) -> tuple[str, int, str, str]:
    return (abi.region_ref, abi.region_offset_bytes, abi.storage_id, abi.id)


def add_fixed_sram_lifecycle(
    fragment: CommandFragment,
    context: LoweringContext,
    *,
    validate: bool = True,
) -> CommandFragment:
    """Insert exact ALLOC_AT/FREE records at scheduled first/last uses.

    Records stay owned by the first/last GlobalAction, so leaf action provenance
    and core-order concatenation remain explicit. No allocator or lifetime
    decision is made here; every field is copied from the N5 BufferABI.
    """

    if type(fragment) is not CommandFragment:
        raise SchemaError("must be a CommandFragment", path="fragment")
    if type(context) is not LoweringContext:
        raise SchemaError("must be a LoweringContext", path="context")
    if type(validate) is not bool:
        raise SchemaError("must be bool", path="validate")
    if validate:
        context.validate()
        fragment.validate_against(context.global_dag)
    if any(
        record.opcode in (RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.SRAM_FREE)
        for stream in fragment.core_streams
        for record in stream.records
    ):
        raise SchemaError(
            "fragment already contains SRAM lifecycle records",
            path="fragment.core_streams",
        )

    actions = {action.id: action for action in context.global_dag.actions}
    abi_by_binding = {
        (abi.schedule_id, abi.binding_id): abi for abi in fragment.buffer_abi
    }
    abis_by_storage: dict[str, list[BufferABI]] = defaultdict(list)
    for abi in fragment.buffer_abi:
        abis_by_storage[abi.storage_id].append(abi)
    root_by_storage: dict[str, BufferABI] = {}
    for storage_id, members in abis_by_storage.items():
        roots = tuple(
            abi
            for abi in members
            if abi.ownership is not BufferOwnership.ALIASED
            and abi.alias_of is None
        )
        if len(roots) != 1:
            raise SchemaError(
                "lifecycle requires exactly one canonical root per storage",
                path="fragment.buffer_abi",
            )
        root = roots[0]
        aliases = tuple(abi for abi in members if abi is not root)
        for alias in aliases:
            if (
                alias.ownership is not BufferOwnership.ALIASED
                or alias.alias_of != root.binding_id
                or (
                    alias.schedule_id,
                    alias.logical_core,
                    alias.region_ref,
                    alias.region_offset_bytes,
                    alias.size_bytes,
                    alias.alignment_bytes,
                    alias.banks,
                    alias.storage_id,
                    alias.tensor_slice.offset,
                    alias.tensor_slice.shape,
                    alias.dtype,
                    alias.layout,
                )
                != (
                    root.schedule_id,
                    root.logical_core,
                    root.region_ref,
                    root.region_offset_bytes,
                    root.size_bytes,
                    root.alignment_bytes,
                    root.banks,
                    root.storage_id,
                    root.tensor_slice.offset,
                    root.tensor_slice.shape,
                    root.dtype,
                    root.layout,
                )
                or root.lifetime_start > alias.lifetime_start
                or root.lifetime_end_exclusive
                < alias.lifetime_end_exclusive
            ):
                raise SchemaError(
                    "lifecycle alias must directly reuse one root geometry and covered lifetime",
                    path="fragment.buffer_abi",
                )
        root_by_storage[storage_id] = root
    symbols = {symbol.id: symbol for symbol in fragment.program_symbols}
    streams: list[CoreFragmentStream] = []

    for stream in fragment.core_streams:
        runtime_core_id = _core_runtime_id(context, stream.logical_core)
        runtime_by_record: dict[int, list[RuntimeRelocation]] = defaultdict(list)
        address_by_record: dict[int, list[AddressRelocation]] = defaultdict(list)
        for relocation in stream.runtime_relocations:
            runtime_by_record[relocation.record_index].append(relocation)
        for relocation in stream.address_relocations:
            address_by_record[relocation.record_index].append(relocation)

        records: list[RelocatableRecord] = []
        runtime_relocations: list[RuntimeRelocation] = []
        address_relocations: list[AddressRelocation] = []
        cursor = 0
        while cursor < len(stream.records):
            action_id = stream.records[cursor].source_global_action_id
            end = cursor + 1
            while (
                end < len(stream.records)
                and stream.records[end].source_global_action_id == action_id
            ):
                end += 1
            action = actions[action_id]
            if action.logical_core != stream.logical_core or action.core_order_index is None:
                raise SchemaError(
                    "fragment action/core order is not executable",
                    path="fragment.core_streams",
                )
            used: dict[str, BufferABI] = {}
            for use in action.buffer_uses:
                abi = abi_by_binding.get((action.source.schedule_id, use.binding_id))
                if abi is None:
                    raise SchemaError(
                        "action buffer use lacks its leaf BufferABI",
                        path="fragment.buffer_abi",
                    )
                root = root_by_storage[abi.storage_id]
                previous = used.setdefault(abi.storage_id, root)
                if previous != root:
                    raise SchemaError(
                        "lifecycle storage must resolve to one canonical root",
                        path="fragment.buffer_abi",
                    )

            starts = tuple(
                sorted(
                    (
                        abi
                        for abi in used.values()
                        if abi.lifetime_start == action.core_order_index
                    ),
                    key=_order_key,
                )
            )
            ends = tuple(
                reversed(
                    sorted(
                        (
                            abi
                            for abi in used.values()
                            if abi.lifetime_end_exclusive
                            == action.core_order_index + 1
                        ),
                        key=_order_key,
                    )
                )
            )

            for abi in starts:
                region = _region(context, abi)
                region_symbol = _region_symbol(abi.region_ref)
                label_symbol = _storage_label_symbol(
                    action.source.schedule_id,
                    runtime_core_id,
                    abi.storage_id,
                )
                symbols[region_symbol.id] = region_symbol
                symbols[label_symbol.id] = label_symbol
                record_index = len(records)
                records.append(
                    RelocatableRecord(
                        action.id,
                        RecordOpcode.SRAM_ALLOC_AT,
                        (
                            RecordOperand.address(
                                "region_name",
                                SemanticOperandId.REGION_NAME,
                                region_symbol.id,
                            ),
                            RecordOperand.address(
                                "label_symbol",
                                SemanticOperandId.LABEL_SYMBOL,
                                label_symbol.id,
                            ),
                            RecordOperand.literal(
                                "region_offset_bytes", abi.region_offset_bytes
                            ),
                            RecordOperand.literal("size_bytes", abi.size_bytes),
                            RecordOperand.literal(
                                "alignment_bytes", abi.alignment_bytes
                            ),
                            RecordOperand.literal("lifetime", 0),
                            RecordOperand.literal("spillable", region.spillable),
                        ),
                    )
                )
                address_relocations.extend(
                    (
                        AddressRelocation(
                            record_index,
                            SemanticOperandId.REGION_NAME,
                            ProgramSymbolKind.SRAM_REGION,
                            region_symbol.id,
                            0,
                        ),
                        AddressRelocation(
                            record_index,
                            SemanticOperandId.LABEL_SYMBOL,
                            ProgramSymbolKind.SRAM_LABEL,
                            label_symbol.id,
                            0,
                        ),
                    )
                )

            for old_index in range(cursor, end):
                new_index = len(records)
                records.append(stream.records[old_index])
                runtime_relocations.extend(
                    RuntimeRelocation(new_index, relocation.field, relocation.symbol_ref)
                    for relocation in runtime_by_record[old_index]
                )
                address_relocations.extend(
                    AddressRelocation(
                        new_index,
                        relocation.operand_id,
                        relocation.symbol_kind,
                        relocation.symbol_ref,
                        relocation.addend,
                    )
                    for relocation in address_by_record[old_index]
                )

            for abi in ends:
                label_symbol = _storage_label_symbol(
                    action.source.schedule_id,
                    runtime_core_id,
                    abi.storage_id,
                )
                symbols[label_symbol.id] = label_symbol
                record_index = len(records)
                records.append(
                    RelocatableRecord(
                        action.id,
                        RecordOpcode.SRAM_FREE,
                        (
                            RecordOperand.address(
                                "symbol",
                                SemanticOperandId.SYMBOL,
                                label_symbol.id,
                            ),
                        ),
                    )
                )
                address_relocations.append(
                    AddressRelocation(
                        record_index,
                        SemanticOperandId.SYMBOL,
                        ProgramSymbolKind.SRAM_LABEL,
                        label_symbol.id,
                        0,
                    )
                )
            cursor = end

        streams.append(
            CoreFragmentStream(
                stream.logical_core,
                tuple(records),
                tuple(
                    sorted(
                        runtime_relocations,
                        key=lambda relocation: (
                            relocation.record_index,
                            _RUNTIME_FIELD_ORDER[relocation.field],
                        ),
                    )
                ),
                tuple(
                    sorted(
                        address_relocations,
                        key=lambda relocation: (
                            relocation.record_index,
                            int(relocation.operand_id),
                        ),
                    )
                ),
            )
        )

    result = CommandFragment.create(
        producer_pass=fragment.producer_pass,
        source_global_dag_id=fragment.source_global_dag_id,
        kind=fragment.kind,
        claimed_action_ids=fragment.claimed_action_ids,
        core_streams=tuple(streams),
        runtime_symbols=fragment.runtime_symbols,
        program_symbols=tuple(sorted(symbols.values(), key=lambda symbol: symbol.id)),
        buffer_abi=fragment.buffer_abi,
        state_abi=fragment.state_abi,
    )
    if validate:
        result.validate_against(context.global_dag)
    return result


__all__ = ["add_fixed_sram_lifecycle"]
