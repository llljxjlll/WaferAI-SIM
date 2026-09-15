"""Relink fixed Dense Prefill+2Decode physical StateABI into timed HBM slots.

This pass changes only HBM StateABI bases. Every source record, SRAM address,
action, fragment boundary, and control envelope comes from the real Dense
production linker. Reused weight pages are safe only with the matching
blocking CoreLsuUnit pager: an LSU Load consumes its page before slot reuse.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    CommandFragment,
    LinkedProgramManifest,
    ManifestInputDigest,
    ManifestInputKind,
    StateABI,
    StateKind,
)
from ..schema.serde import canonical_digest


HBM_CAPACITY_BYTES = 12288
P3_WORKSPACE_END_BYTES = 1600
WEIGHT_SLOT_BASE = 1600
WEIGHT_SLOT_SIZE_BYTES = 8192
KV_SLOT_BASE = 9792
KV_SLOT_SIZE_BYTES = 192
PAGED_HIGHEST_END_BYTES = 10560
PARAMETER_SOURCE_BYTES = 53568
KV_SOURCE_BASE = 53568


def _state_abis(source: LinkedProgramManifest) -> tuple[StateABI, ...]:
    if any(type(fragment) is not CommandFragment for fragment in source.fragments):
        raise SchemaError("requires production CommandFragment Dense linker", path="source.fragments")
    unique: dict[str, StateABI] = {}
    for fragment in source.fragments:
        for abi in fragment.state_abi:
            old = unique.setdefault(abi.id, abi)
            if old != abi:
                raise SchemaError("shared StateABI has conflicting definitions", path="source.fragments")
    return tuple(sorted(unique.values(), key=lambda abi: abi.id))


def _kv_kind(kind: StateKind) -> bool:
    return kind in (StateKind.KV_KEY, StateKind.KV_VALUE)


def relink_dense_inference_paged_segment(
    source: LinkedProgramManifest,
    segment_index: int,
) -> LinkedProgramManifest:
    """Return one source-closed linked segment for 12,288B physical HBM."""

    if segment_index not in (0, 1, 2):
        raise SchemaError("requires actual Prefill+2Decode segment", path="segment_index")
    source.validate("dense_paged.source")
    if (
        source.producer_pass != "manifest_linker"
        or len(source.fragments) != (44 if segment_index == 0 else 48)
        or len(source.core_streams) != 1
        or len(source.core_streams[0].records) != (159 if segment_index == 0 else 163)
        or len(source.state_operand_bindings) != (19 if segment_index == 0 else 23)
    ):
        raise SchemaError("fixed two-layer production source shape changed", path="source")
    abis = _state_abis(source)
    weights = tuple(sorted(
        (abi for abi in abis if abi.kind is StateKind.PARAMETER),
        key=lambda abi: abi.address,
    ))
    kv = tuple(sorted(
        (abi for abi in abis if _kv_kind(abi.kind)),
        key=lambda abi: abi.address,
    ))
    if len(weights) != 15 or len(kv) != 4 or len(abis) != 19:
        raise SchemaError("15 parameter and four KV StateABI are required", path="source.state_abi")
    cursor = 0
    for abi in weights:
        if (
            abi.die_id != 0 or abi.address != cursor or
            abi.size_bytes <= 0 or abi.size_bytes > WEIGHT_SLOT_SIZE_BYTES
        ):
            raise SchemaError("physical parameter ABI does not tightly cover P3 source", path="source.state_abi")
        cursor += abi.size_bytes
    if cursor != PARAMETER_SOURCE_BYTES:
        raise SchemaError("P3 parameter source allocation changed", path="source.state_abi")
    page_bytes = 128 + 32 * segment_index
    for index, abi in enumerate(kv):
        if (
            abi.die_id != 0
            or abi.address != KV_SOURCE_BASE + KV_SLOT_SIZE_BYTES * index
            or abi.size_bytes != page_bytes
        ):
            raise SchemaError("KV source pages must retain four versioned addresses", path="source.state_abi")
    if PAGED_HIGHEST_END_BYTES > HBM_CAPACITY_BYTES or (
        WEIGHT_SLOT_BASE < P3_WORKSPACE_END_BYTES
        or WEIGHT_SLOT_BASE + WEIGHT_SLOT_SIZE_BYTES > KV_SLOT_BASE
    ):
        raise SchemaError("paged slots collide with P3 workspace or physical capacity", path="slots")

    relocated_abi: dict[str, StateABI] = {}
    for abi in abis:
        physical = (
            WEIGHT_SLOT_BASE
            if abi.kind is StateKind.PARAMETER
            else KV_SLOT_BASE + (abi.address - KV_SOURCE_BASE)
        )
        relocated_abi[abi.id] = StateABI.create(
            state_ref=abi.state_ref,
            hbm_binding_ref=abi.hbm_binding_ref,
            kind=abi.kind,
            lifetime=abi.lifetime,
            access=abi.access,
            shape=abi.shape,
            dtype=abi.dtype,
            layout=abi.layout,
            die_id=abi.die_id,
            address=physical,
            size_bytes=abi.size_bytes,
            alignment_bytes=abi.alignment_bytes,
        )

    fragment_ids: dict[str, str] = {}
    fragments: list[CommandFragment] = []
    for fragment in source.fragments:
        key = fragment._semantic_key()
        key["state_abi"] = tuple(sorted(
            (relocated_abi[abi.id] for abi in fragment.state_abi),
            key=lambda abi: abi.id,
        ))
        item = CommandFragment.create(
            producer_pass=fragment.producer_pass, **key,
        )
        item.validate("dense_paged.fragment")
        fragment_ids[fragment.id] = item.id
        fragments.append(item)
    if len(fragment_ids) != len(source.fragments):
        raise SchemaError("source fragment identities repeat", path="source.fragments")
    fragment_by_old = {
        old.id: new for old, new in zip(source.fragments, fragments)
    }
    fragments.sort(key=lambda item: item.id)
    inputs = tuple(sorted((
        ManifestInputDigest(
            ManifestInputKind.COMMAND_FRAGMENT,
            fragment_by_old[item.artifact_id].id,
            fragment_by_old[item.artifact_id].schema_version,
            canonical_digest(fragment_by_old[item.artifact_id]),
        ) if item.kind is ManifestInputKind.COMMAND_FRAGMENT
        else item
        for item in source.input_digests
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    state_by_binding = {
        abi.hbm_binding_ref: abi for abi in relocated_abi.values()
    }
    definitions = tuple(
        replace(item, value=state_by_binding[item.symbol.source_ref].address)
        if item.symbol.source_ref in state_by_binding
        else item
        for item in source.program_symbol_definitions
    )
    key = source._semantic_key()
    key.update(
        input_digests=inputs,
        fragments=tuple(fragments),
        fragment_interfaces=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in source.fragment_interfaces
        ), key=lambda item: item.fragment_id)),
        core_streams=tuple(
            replace(stream, records=tuple(
                replace(record, fragment_id=fragment_ids[record.fragment_id])
                for record in stream.records
            ))
            for stream in source.core_streams
        ),
        program_symbol_definitions=definitions,
        address_operand_bindings=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in source.address_operand_bindings
        ), key=lambda item: (item.logical_core.die_id,
                            item.logical_core.local_core_id,
                            item.fragment_id,
                            item.fragment_record_index,
                            int(item.operand_id)))),
        state_operand_bindings=tuple(sorted((
            replace(item,
                    fragment_id=fragment_ids[item.fragment_id],
                    state_abi_id=relocated_abi[item.state_abi_id].id)
            for item in source.state_operand_bindings
        ), key=lambda item: (item.logical_core.die_id,
                            item.logical_core.local_core_id,
                            item.fragment_id,
                            item.fragment_record_index,
                            int(item.operand_id)))),
    )
    paged = LinkedProgramManifest.create(
        producer_pass=source.producer_pass, **key,
    )
    paged.validate("dense_paged.linked")
    return paged


__all__ = ["relink_dense_inference_paged_segment"]
