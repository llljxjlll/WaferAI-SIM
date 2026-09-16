"""Relink the fixed nine-die full MoE inference source into real low-HBM pages.

Every command, DTE flow, SRAM buffer and logical action remains the production
full-model linker output. Only physical HBM StateABI bases change; the matching
blocking pager must complete each restore before LSU and expert/KV writeback
before reuse of the 192-byte per-die parameter slot.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    CommandFragment, LinkedProgramManifest, ManifestInputDigest,
    ManifestInputKind, StateABI, StateKind,
)
from ..schema.serde import canonical_digest

HBM_HOME_BASE = tuple(die << 30 for die in range(9))
HBM_CAPACITY_BYTES = 1024
P3_WORKSPACE_END_BYTES = 464
WEIGHT_SLOT_OFFSET = 512
WEIGHT_SLOT_BYTES = 192
KV_SLOT_OFFSET = 704
KV_SLOT_BYTES = 64
HIGHEST_PAGED_END_BYTES = 960
KV_SOURCE_BASE = 1344
KV_SOURCE_STRIDE = 64


def unique_state_abis(source: LinkedProgramManifest) -> tuple[StateABI, ...]:
    if any(type(fragment) is not CommandFragment for fragment in source.fragments):
        raise SchemaError("requires true full-model CommandFragment source", path="source.fragments")
    unique: dict[str, StateABI] = {}
    for fragment in source.fragments:
        for abi in fragment.state_abi:
            old = unique.setdefault(abi.id, abi)
            if old != abi:
                raise SchemaError("shared KV StateABI definition changed", path="source.fragments")
    return tuple(sorted(unique.values(), key=lambda abi: abi.id))


def _validate_source(source: LinkedProgramManifest, step: int) -> tuple[StateABI, ...]:
    """Require the actual nine-core full-model source and its physical ABI."""
    if step not in (0, 1, 2):
        raise SchemaError("requires true Prefill+2Decode segment", path="step")
    source.validate("moe_paged_ep9.source")
    expected_records = ((207, 52, 52, 52, 30, 30, 30, 30, 30) if step == 0
                        else (199, 52, 30, 30, 30, 30, 30, 30, 30))
    if (
        source.producer_pass != "moe_full_model_region_linker"
        or len(source.fragments) != (39 if step == 0 else 43)
        or tuple(stream.runtime_core_id for stream in source.core_streams)
            != (0, 4, 8, 12, 16, 20, 24, 28, 32)
        or tuple(len(stream.records) for stream in source.core_streams)
            != expected_records
        or len(source.state_operand_bindings) != (69 if step == 0 else 73)
    ):
        raise SchemaError("two-layer EP9 full-model linked source shape changed", path="source")
    abis = unique_state_abis(source)
    per_die = {die: tuple(abi for abi in abis if abi.die_id == die)
               for die in range(9)}
    shared0 = tuple(abi for abi in per_die[0]
                    if abi.kind is StateKind.PARAMETER)
    kv = tuple(sorted((abi for abi in abis if abi.kind in
                       (StateKind.KV_KEY, StateKind.KV_VALUE)),
                      key=lambda abi: abi.address))
    if (len(abis) != 51 or len(shared0) != 13
            or sum(abi.size_bytes for abi in shared0) != 696
            or len(kv) != 4):
        raise SchemaError("EP9 shared/router/expert/KV StateABI inventory changed",
                          path="source.state_abi")
    for die in range(9):
        experts = tuple(abi for abi in per_die[die]
                        if abi.kind is StateKind.TRAINABLE_PARAMETER)
        gates = tuple(abi for abi in per_die[die]
                      if abi.kind is StateKind.PARAMETER
                      and (die != 0 or abi.address >= 1 << 24))
        if (len(experts) != 2 or any(abi.size_bytes != 192 for abi in experts)
                or len(gates) != 2 or any(abi.size_bytes != 72 for abi in gates)):
            raise SchemaError("EP9 per-Die expert/router pages changed",
                              path="source.state_abi")
        expected_expert = tuple(HBM_HOME_BASE[die] + (1 << 24) + layer * (1 << 20)
                                for layer in (0, 1))
        if (tuple(sorted(abi.address for abi in experts)) != expected_expert
                or tuple(sorted(abi.address for abi in gates))
                != tuple(address + 4096 for address in expected_expert)):
            raise SchemaError("EP9 expert source HBM home/hole anchoring changed",
                              path="source.state_abi")
    page = 32 + 16 * step
    if any(abi.die_id != 0
           or abi.address != KV_SOURCE_BASE + index * KV_SOURCE_STRIDE
           or abi.size_bytes != page for index, abi in enumerate(kv)):
        raise SchemaError("four physical KV pages/version lengths changed",
                          path="source.state_abi")
    if any(abi.size_bytes > WEIGHT_SLOT_BYTES for abi in abis
           if abi.kind in (StateKind.PARAMETER, StateKind.TRAINABLE_PARAMETER)):
        raise SchemaError("one physical parameter page exceeds low-HBM slot",
                          path="source.state_abi")
    if not (P3_WORKSPACE_END_BYTES <= WEIGHT_SLOT_OFFSET
            and WEIGHT_SLOT_OFFSET + WEIGHT_SLOT_BYTES <= KV_SLOT_OFFSET
            and HIGHEST_PAGED_END_BYTES <= HBM_CAPACITY_BYTES):
        raise SchemaError("timed slots collide with P3 workspace or low HBM",
                          path="slots")
    return abis


def relink_moe_inference_paged_segment_ep9(
    source: LinkedProgramManifest, step: int,
) -> LinkedProgramManifest:
    """Retain all nine production EP cores in 1024B per-Die HBM."""
    abis = _validate_source(source, step)
    relocated: dict[str, StateABI] = {}
    for abi in abis:
        physical = HBM_HOME_BASE[abi.die_id] + (
            KV_SLOT_OFFSET + abi.address - KV_SOURCE_BASE
            if abi.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
            else WEIGHT_SLOT_OFFSET
        )
        relocated[abi.id] = StateABI.create(
            state_ref=abi.state_ref, hbm_binding_ref=abi.hbm_binding_ref,
            kind=abi.kind, lifetime=abi.lifetime, access=abi.access,
            shape=abi.shape, dtype=abi.dtype, layout=abi.layout,
            die_id=abi.die_id, address=physical,
            size_bytes=abi.size_bytes,
            # The 4096B expert source alignment spaces whole layers through
            # the original high-address hole. A single 192B DMA page requires
            # only the 64B physical HBM transaction alignment; this dedicated
            # relink keeps the generic StateABI validator strict.
            alignment_bytes=min(abi.alignment_bytes, 64),
        )
    fragment_ids: dict[str, str] = {}
    fragments: list[CommandFragment] = []
    for fragment in source.fragments:
        key = fragment._semantic_key()
        key["state_abi"] = tuple(sorted(
            (relocated[abi.id] for abi in fragment.state_abi), key=lambda abi: abi.id,
        ))
        item = CommandFragment.create(producer_pass=fragment.producer_pass, **key)
        item.validate("moe_paged.fragment")
        fragment_ids[fragment.id] = item.id
        fragments.append(item)
    if len(fragment_ids) != len(source.fragments):
        raise SchemaError("source fragment identity repeated", path="source.fragments")
    new_by_old = {old.id: new for old, new in zip(source.fragments, fragments)}
    inputs = tuple(sorted((
        ManifestInputDigest(
            ManifestInputKind.COMMAND_FRAGMENT,
            new_by_old[item.artifact_id].id,
            new_by_old[item.artifact_id].schema_version,
            canonical_digest(new_by_old[item.artifact_id]),
        ) if item.kind is ManifestInputKind.COMMAND_FRAGMENT else item
        for item in source.input_digests
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    by_binding = {abi.hbm_binding_ref: abi for abi in relocated.values()}
    definitions = tuple(
        replace(item, value=by_binding[item.symbol.source_ref].address)
        if item.symbol.source_ref in by_binding else item
        for item in source.program_symbol_definitions
    )
    key = source._semantic_key()
    key.update(
        input_digests=inputs,
        fragments=tuple(sorted(fragments, key=lambda item: item.id)),
        fragment_interfaces=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in source.fragment_interfaces
        ), key=lambda item: item.fragment_id)),
        core_streams=tuple(replace(stream, records=tuple(
            replace(ref, fragment_id=fragment_ids[ref.fragment_id])
            for ref in stream.records
        )) for stream in source.core_streams),
        program_symbol_definitions=definitions,
        address_operand_bindings=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in source.address_operand_bindings
        ), key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id,
                            item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
        state_operand_bindings=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id],
                    state_abi_id=relocated[item.state_abi_id].id)
            for item in source.state_operand_bindings
        ), key=lambda item: (item.logical_core.die_id, item.logical_core.local_core_id,
                            item.fragment_id, item.fragment_record_index, int(item.operand_id)))),
    )
    paged = LinkedProgramManifest.create(producer_pass=source.producer_pass, **key)
    paged.validate("moe_paged.linked")
    return paged


__all__ = ["relink_moe_inference_paged_segment_ep9", "unique_state_abis"]
