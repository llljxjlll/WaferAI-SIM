"""Relink a real EP1 two-step MoE training program into a bounded HBM page.

The production DAG, fragments and core records are unchanged. A separate
source-bound runtime pager must restore each state before its LSU LOAD and
write back every trainable LSU STORE; this relinker alone is not offload.
"""

from __future__ import annotations

from dataclasses import replace
from collections import Counter

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    CommandFragment, LinkedProgramManifest, ManifestInputDigest,
    ManifestInputKind, RecordOpcode, StateABI, StateKind,
)
from ..schema.serde import canonical_digest


HBM_CAPACITY_BYTES = 2560
P3_WORKSPACE_END_BYTES = 2016
STATE_SLOT_ADDRESS = 2048
STATE_SLOT_BYTES = 128
ROUTE_HOME_ADDRESSES = (2304, 2432)


def relink_moe_full_train_paged_step(
    source: LinkedProgramManifest, step: int,
) -> LinkedProgramManifest:
    """Preserve the 160-leaf graph: page 19 weights, pin two route traces."""
    if step not in (0, 1):
        raise SchemaError("MoE train pager requires step0 or step1", path="step")
    source.validate("moe_train_paged.source")
    if (source.producer_pass != "manifest_linker"
            or len(source.fragments) != 160
            or len(source.core_streams) != 1
            or source.core_streams[0].runtime_core_id != 0
            or len(source.state_operand_bindings) != 67
            or any(type(item) is not CommandFragment
                   for item in source.fragments)):
        raise SchemaError("full EP1 two-layer MoE train linked source changed",
                          path="source")
    expected_records = {
        RecordOpcode.MATMUL: 17, RecordOpcode.SWIGLU: 4,
        RecordOpcode.RESIDUAL: 10, RecordOpcode.RMSNORM: 5,
        RecordOpcode.ROPE_QK_EXACT: 2, RecordOpcode.ATTENTION_EXACT: 2,
        RecordOpcode.EMBEDDING_LOOKUP: 1,
        RecordOpcode.CROSS_ENTROPY_FORWARD: 1,
        RecordOpcode.CROSS_ENTROPY_BACKWARD: 1,
        RecordOpcode.SGD_UPDATE: 19,
        RecordOpcode.SWIGLU_BACKWARD_TIMING: 2,
        RecordOpcode.EMBEDDING_TABLE_WGRAD_TIMING: 1,
        RecordOpcode.NORM_GAMMA_WGRAD_TIMING: 5,
        RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING: 13,
        RecordOpcode.GEMM_DX_TIMING: 13,
        RecordOpcode.MOE_SCORE_WEIGHTED_FORWARD: 2,
        RecordOpcode.MOE_SCORE_WEIGHT_BACKWARD: 2,
        RecordOpcode.RMSNORM_BACKWARD_TIMING: 5,
        RecordOpcode.ATTENTION_BACKWARD_TIMING: 2,
        RecordOpcode.ROPE_BACKWARD_TIMING: 2,
        RecordOpcode.RESIDUAL_BACKWARD_TIMING: 4,
        RecordOpcode.LOCAL_REDUCE: 2,
        RecordOpcode.LSU_LOAD: 48, RecordOpcode.LSU_STORE: 19,
        RecordOpcode.DTE_ISSUE: 4, RecordOpcode.DTE_WAIT: 4,
        RecordOpcode.SRAM_BIND: 113,
        RecordOpcode.SRAM_FREE: 159,
        RecordOpcode.SRAM_ALLOC_AT: 159,
    }
    records = Counter(record.opcode for fragment in source.fragments
                      for stream in fragment.core_streams
                      for record in stream.records)
    if records != expected_records:
        raise SchemaError("MoE train full 621-record source closure changed",
                          path="source.fragments")
    abis: dict[str, StateABI] = {}
    for fragment in source.fragments:
        for abi in fragment.state_abi:
            prior = abis.setdefault(abi.id, abi)
            if prior != abi:
                raise SchemaError("conflicting source StateABI", path="source.state_abi")
    trainable = tuple(abi for abi in abis.values()
                      if abi.kind is StateKind.TRAINABLE_PARAMETER)
    routes = tuple(abi for abi in abis.values()
                   if abi.kind is StateKind.MOE_STATIC_ROUTE)
    if (len(trainable) != 19 or len(routes) != 2
            or sum(abi.size_bytes for abi in trainable) != 952
            or sum(abi.size_bytes for abi in routes) != 160
            or any(abi.die_id != 0 or abi.size_bytes > STATE_SLOT_BYTES
                   for abi in abis.values())
            or STATE_SLOT_ADDRESS < P3_WORKSPACE_END_BYTES
            or STATE_SLOT_ADDRESS + STATE_SLOT_BYTES > ROUTE_HOME_ADDRESSES[0]
            or ROUTE_HOME_ADDRESSES[0] + routes[0].size_bytes > ROUTE_HOME_ADDRESSES[1]
            or ROUTE_HOME_ADDRESSES[1] + routes[1].size_bytes > HBM_CAPACITY_BYTES):
        raise SchemaError("21 actual states cannot fit signed 128B page",
                          path="source.state_abi")
    route_address = {abi.id: ROUTE_HOME_ADDRESSES[index]
                     for index, abi in enumerate(sorted(routes, key=lambda item: item.address))}
    remapped = {
        abi.id: StateABI.create(
            state_ref=abi.state_ref, hbm_binding_ref=abi.hbm_binding_ref,
            kind=abi.kind, lifetime=abi.lifetime, access=abi.access,
            shape=abi.shape, dtype=abi.dtype, layout=abi.layout,
            die_id=abi.die_id,
            address=(route_address[abi.id] if abi.id in route_address
                     else STATE_SLOT_ADDRESS),
            size_bytes=abi.size_bytes,
            alignment_bytes=min(abi.alignment_bytes, 64),
        ) for abi in abis.values()
    }
    fragment_ids: dict[str, str] = {}
    fragments: list[CommandFragment] = []
    for fragment in source.fragments:
        key = fragment._semantic_key()
        key["state_abi"] = tuple(sorted(
            (remapped[abi.id] for abi in fragment.state_abi),
            key=lambda abi: abi.id,
        ))
        relocated = CommandFragment.create(
            producer_pass=fragment.producer_pass, **key,
        )
        relocated.validate("moe_train_paged.fragment")
        fragment_ids[fragment.id] = relocated.id
        fragments.append(relocated)
    by_old_id = {old.id: new for old, new in
                 zip(source.fragments, fragments)}
    inputs = tuple(sorted((
        ManifestInputDigest(
            ManifestInputKind.COMMAND_FRAGMENT,
            by_old_id[item.artifact_id].id,
            by_old_id[item.artifact_id].schema_version,
            canonical_digest(by_old_id[item.artifact_id]),
        ) if item.kind is ManifestInputKind.COMMAND_FRAGMENT else item
        for item in source.input_digests
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    by_binding = {abi.hbm_binding_ref: abi for abi in remapped.values()}
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
        ), key=lambda item: (item.logical_core.die_id,
                            item.logical_core.local_core_id,
                            item.fragment_id, item.fragment_record_index,
                            int(item.operand_id)))),
        state_operand_bindings=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id],
                    state_abi_id=remapped[item.state_abi_id].id)
            for item in source.state_operand_bindings
        ), key=lambda item: (item.logical_core.die_id,
                            item.logical_core.local_core_id,
                            item.fragment_id, item.fragment_record_index,
                            int(item.operand_id)))),
    )
    linked = LinkedProgramManifest.create(
        producer_pass=source.producer_pass, **key,
    )
    linked.validate("moe_train_paged.linked")
    return linked


__all__ = ["relink_moe_full_train_paged_step"]
