"""Relink the complete two-step Dense AdamW program to a bounded HBM DMA slot.

Every HBM LSU in this TP1/DP1 program is a physical StateABI access.  The
single-core LSU completes each load/store before issuing the next record, so a
blocking external DMA can reuse one HBM slot after each operation.  This pass
only changes StateABI addresses and their linked relocations; the source graph,
all compute records, and their order remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections import Counter

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    CommandFragment, LinkedProgramManifest, ManifestInputDigest,
    ManifestInputKind, RecordOpcode, StateABI,
)
from ..schema.full_training_physical_dag import FullTrainingPhysicalDAG
from ..schema.serde import canonical_digest
from ..schema.train_n6 import TrainLinkedProgram
from .full_dense_adamw_offload_preflight import FullDenseAdamwOffloadWindow


@dataclass(frozen=True, slots=True)
class FullDenseAdamwPagedState:
    state_ref: str
    source_abi_id: str
    paged_abi_id: str
    kind: str
    external_address: int
    hbm_address: int
    size_bytes: int
    seed_hex: str


@dataclass(frozen=True, slots=True)
class FullDenseAdamwPagedEvent:
    index: int
    program_record_index: int
    step: int
    kind: str
    state_ref: str
    external_address: int
    hbm_address: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class FullDenseAdamwPagedProgram:
    source_manifest_id: str
    source_manifest_digest: str
    paged_manifest: LinkedProgramManifest
    physical_dag_digest: str
    source_ir1_id: str
    hbm_capacity_bytes: int
    external_capacity_bytes: int
    slot_address: int
    slot_bytes: int
    state_payload_bytes: int
    states: tuple[FullDenseAdamwPagedState, ...]
    events: tuple[FullDenseAdamwPagedEvent, ...]


def _binding_order(binding):
    core = binding.logical_core
    return (core.die_id, core.local_core_id, binding.fragment_id,
            binding.fragment_record_index, int(binding.operand_id))


def _source_state_inventory(manifest: LinkedProgramManifest) -> tuple[dict, dict]:
    states = {}
    by_abi = {}
    for fragment in manifest.fragments:
        if type(fragment) is not CommandFragment:
            raise SchemaError("full AdamW pager requires native leaf fragments",
                              path="manifest.fragments")
        for abi in fragment.state_abi:
            prior = states.setdefault(abi.state_ref, abi)
            if prior != abi:
                raise SchemaError("one persistent state changed physical ABI",
                                  path=abi.state_ref)
            by_abi[abi.id] = abi
    if len(states) != 75 or Counter(item.kind.value for item in states.values()) != {
            "trainable_parameter": 15, "optimizer_master": 15,
            "optimizer_moment1": 15, "optimizer_moment2": 15,
            "optimizer_step": 15,
    }:
        raise SchemaError("full AdamW pager needs all 75 physical state roles",
                          path="manifest.fragments")
    ordered = sorted((item.address, item.address + item.size_bytes, ref)
                     for ref, item in states.items())
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise SchemaError("source external state ranges overlap",
                          path="manifest.fragments")
    return states, by_abi


def rebase_full_dense_adamw_to_blocking_slot(
    source: TrainLinkedProgram,
    physical: FullTrainingPhysicalDAG,
    window: FullDenseAdamwOffloadWindow,
    state_seeds: dict[str, bytes],
) -> FullDenseAdamwPagedProgram:
    """Build a source-bound low-HBM program and its exact 350 DMA gates.

    State payload is initially authoritative in the finite external tier.
    Every LSU_LOAD restores its state immediately before SRAM consumption;
    every LSU_STORE writes the dirty HBM slot back before another record can
    reuse it.  The slot therefore has no live state between LSU records.
    """
    # The preflight factory has already deep-validated this TrainLinkedProgram.
    # Revalidate the consumed native manifest here without re-linking 520
    # fragments a second time.
    manifest = source.manifest
    manifest.validate("full_adamw_source_manifest")
    physical.validate("full_adamw_physical")
    if (window.linked_manifest_id != manifest.id or
            window.source_ir1_id !=
            source.source.replicas[0].lowering_context.ir1.id or
            not window.blocking_offload_window_necessary or
            not window.resident_only_rejected or
            window.state_count != 75 or
            window.hbm_capacity_bytes >= window.resident_hbm_highwater_bytes):
        raise SchemaError("full AdamW low-HBM source/preflight identity drifted",
                          path="window")
    if len(manifest.fragments) != 520 or len(physical.actions) != 520 or len(
            physical.state_version_edges) != 75 or len(manifest.core_streams) != 1:
        raise SchemaError("full AdamW two-step physical coverage is incomplete",
                          path="source")
    source_states, by_abi = _source_state_inventory(manifest)
    if (set(state_seeds) != set(source_states) or
            any(type(state_seeds[ref]) is not bytes or
                len(state_seeds[ref]) != abi.size_bytes
                for ref, abi in source_states.items())):
        raise SchemaError("external seeds must exactly cover 75 StateABI payloads",
                          path="state_seeds")
    slot_bytes = max(item.size_bytes for item in source_states.values())
    alignment = max(item.alignment_bytes for item in source_states.values())
    slot_address = 0
    if slot_address % alignment or slot_bytes > window.hbm_capacity_bytes:
        raise SchemaError("one physical StateABI cannot fit the HBM DMA slot",
                          path="window.hbm_capacity_bytes")
    new_abis = {
        ref: StateABI.create(
            state_ref=abi.state_ref, hbm_binding_ref=abi.hbm_binding_ref,
            kind=abi.kind, lifetime=abi.lifetime, access=abi.access,
            shape=abi.shape, dtype=abi.dtype, layout=abi.layout,
            die_id=abi.die_id, address=slot_address,
            size_bytes=abi.size_bytes, alignment_bytes=abi.alignment_bytes,
        )
        for ref, abi in source_states.items()
    }
    fragments = {}
    fragment_ids = {}
    abi_ids = {}
    for fragment in manifest.fragments:
        if not fragment.state_abi:
            new_fragment = fragment
        else:
            key = fragment._semantic_key()
            key["state_abi"] = tuple(sorted(
                (new_abis[item.state_ref] for item in fragment.state_abi),
                key=lambda item: item.id))
            new_fragment = CommandFragment.create(
                producer_pass=fragment.producer_pass, **key)
            new_fragment.validate("paged.fragment")
            for item in fragment.state_abi:
                abi_ids[item.id] = new_abis[item.state_ref].id
        fragments[fragment.id] = new_fragment
        fragment_ids[fragment.id] = new_fragment.id
    hbm_addresses = {item.hbm_binding_ref: slot_address
                     for item in source_states.values()}
    key = manifest._semantic_key()
    key.update(
        fragments=tuple(sorted(fragments.values(), key=lambda item: item.id)),
        input_digests=tuple(sorted((
            ManifestInputDigest(
                ManifestInputKind.COMMAND_FRAGMENT,
                fragments[item.artifact_id].id,
                fragments[item.artifact_id].schema_version,
                canonical_digest(fragments[item.artifact_id]),
            ) if item.kind is ManifestInputKind.COMMAND_FRAGMENT else item
            for item in manifest.input_digests
        ), key=lambda item: (item.kind.value, item.artifact_id))),
        fragment_interfaces=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in manifest.fragment_interfaces
        ), key=lambda item: item.fragment_id)),
        core_streams=tuple(replace(stream, records=tuple(
            replace(record, fragment_id=fragment_ids[record.fragment_id])
            for record in stream.records
        )) for stream in manifest.core_streams),
        program_symbol_definitions=tuple(
            replace(item, value=hbm_addresses[item.symbol.source_ref])
            if item.symbol.source_ref in hbm_addresses else item
            for item in manifest.program_symbol_definitions),
        address_operand_bindings=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in manifest.address_operand_bindings
        ), key=_binding_order)),
        state_operand_bindings=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id],
                    state_abi_id=abi_ids[item.state_abi_id])
            for item in manifest.state_operand_bindings
        ), key=_binding_order)),
    )
    paged = LinkedProgramManifest.create(
        producer_pass=manifest.producer_pass, **key)
    paged.validate("full_adamw_paged_manifest")
    if (len(paged.fragments) != 520 or len(paged.state_operand_bindings) != 350 or
            paged.source_ir1_id != manifest.source_ir1_id):
        raise SchemaError("relocated program changed full training coverage",
                          path="paged")
    old_frags = {item.id: item for item in manifest.fragments}
    state_bindings = {
        (item.fragment_id, item.fragment_record_index): item
        for item in manifest.state_operand_bindings
    }
    action_steps = {item.id: item.step for item in physical.actions}
    events = []
    counts = Counter()
    for index, record in enumerate(manifest.core_streams[0].records):
        fragment = old_frags[record.fragment_id]
        native = fragment.core_streams[0].records[record.fragment_record_index]
        if native.opcode not in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE):
            continue
        binding = state_bindings.get(
            (record.fragment_id, record.fragment_record_index))
        if binding is None or record.source_global_action_id not in action_steps:
            raise SchemaError("non-state HBM LSU or unsigned physical action",
                              path=f"source.core_stream.records[{index}]")
        abi = by_abi[binding.state_abi_id]
        step = action_steps[record.source_global_action_id]
        kind = "restore_before_lsu_load" if native.opcode is RecordOpcode.LSU_LOAD \
            else "writeback_after_lsu_store"
        counts[step, kind] += 1
        events.append(FullDenseAdamwPagedEvent(
            len(events), index, step, kind, abi.state_ref,
            abi.address, slot_address, abi.size_bytes))
    if (len(events) != 350 or counts != {
            (0, "restore_before_lsu_load"): 100,
            (0, "writeback_after_lsu_store"): 75,
            (1, "restore_before_lsu_load"): 100,
            (1, "writeback_after_lsu_store"): 75,
    } or [item.step for item in events] != sorted(item.step for item in events)):
        raise SchemaError("full two-step LSU/DMA order differs from signed source",
                          path="source.core_stream")
    for step in (0, 1):
        loads = {item.state_ref for item in events
                 if item.step == step and item.kind == "restore_before_lsu_load"}
        stores = {item.state_ref for item in events
                  if item.step == step and item.kind == "writeback_after_lsu_store"}
        if loads != set(source_states) or stores != set(source_states):
            raise SchemaError("each step must restore and write back all 75 states",
                              path=f"events.step[{step}]")
    states = tuple(FullDenseAdamwPagedState(
        ref, abi.id, new_abis[ref].id, abi.kind.value, abi.address,
        slot_address, abi.size_bytes, state_seeds[ref].hex())
        for ref, abi in sorted(source_states.items()))
    return FullDenseAdamwPagedProgram(
        manifest.id, canonical_digest(manifest), paged,
        canonical_digest(physical), manifest.source_ir1_id,
        window.hbm_capacity_bytes, window.external_capacity_bytes,
        slot_address, slot_bytes, sum(item.size_bytes for item in source_states.values()),
        states, tuple(events))


__all__ = ["FullDenseAdamwPagedEvent", "FullDenseAdamwPagedProgram",
           "FullDenseAdamwPagedState", "rebase_full_dense_adamw_to_blocking_slot"]
