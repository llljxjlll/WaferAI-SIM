"""Source-bound actual-LSU external pages for the fixed full 1x4 EP4 MoE model."""

from __future__ import annotations

import hashlib
import json

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LinkedProgramManifest, RecordOpcode, SemanticOperandId, StateKind,
)
from ..schema.external_memory import ExternalMemoryFabric
from ..schema.memory_plan import MemoryObjectKind, MemoryTier
from ..schema.serde import canonical_digest, canonical_json
from .moe_inference_paged_compile_sequence_ep4 import (
    HBM_CAPACITY_BYTES, HBM_HOME_BASE, HIGHEST_PAGED_END_BYTES,
    KV_SLOT_BYTES, KV_SLOT_OFFSET, KV_SOURCE_BASE, KV_SOURCE_STRIDE,
    P3_WORKSPACE_END_BYTES, WEIGHT_SLOT_BYTES, WEIGHT_SLOT_OFFSET,
    unique_state_abis,
)

PARAMETER_EXTERNAL_RESERVED_BYTES = 4032
KV_EXTERNAL_BASE = 4032
EXTERNAL_CAPACITY_BYTES = 8192
PHYSICAL_PARAMETER_BYTES = 2344
KV_PHYSICAL_BYTES = 256


def _record(fragment, core, index):
    stream = next((item for item in fragment.core_streams
                   if item.logical_core == core), None)
    if stream is None or index >= len(stream.records):
        raise SchemaError("linked core record has no physical fragment", path="record")
    return stream.records[index], stream


def _lsu_size(record) -> int:
    values = tuple(item.literal_value for item in record.operands
                   if item.name == "size_bytes")
    if len(values) != 1 or type(values[0]) is not int or values[0] <= 0:
        raise SchemaError("actual LSU size literal absent", path="record")
    return values[0]


def _hbm_addend(stream, index: int) -> int:
    values = tuple(item.addend for item in stream.address_relocations
                   if item.record_index == index and
                   item.operand_id is SemanticOperandId.HBM_ADDRESS)
    if len(values) != 1:
        raise SchemaError("actual LSU HBM relocation absent", path="record")
    return values[0]


def _external_parameter_allocations(offload):
    """Bind each physical page to its true P3 rank/role external allocation."""
    plan = offload.memory_plan
    requests = {item.id: item for item in plan.requests}
    versions = {item.id: item for item in plan.state_versions}
    inventory = {item.id: item for item in offload.state_inventory}
    by_rank_role = {}
    for allocation in plan.allocations:
        request = requests[allocation.request_ref]
        if (request.object_kind is not MemoryObjectKind.PARAMETER
                or request.tier is not MemoryTier.EXTERNAL):
            continue
        state = inventory[versions[request.state_version_ref].state_ref]
        role = "expert" if request.size_bytes == 384 else "shared"
        if (state.logical_rank not in range(4)
                or request.size_bytes not in (384, 616)
                or (state.logical_rank, role) in by_rank_role):
            raise SchemaError("EP4 external rank/role allocation changed",
                              path="offload.memory_plan")
        by_rank_role[state.logical_rank, role] = (
            allocation.address, request.size_bytes, allocation.id
        )
    if set(by_rank_role) != {(die, role) for die in range(4)
                            for role in ("expert", "shared")}:
        raise SchemaError("eight true P3 EP4 external allocations missing",
                          path="offload.memory_plan")
    if max(address + size for address, size, _ in by_rank_role.values()) != KV_EXTERNAL_BASE:
        raise SchemaError("P3 external parameter reservation changed",
                          path="offload.memory_plan")
    return by_rank_role


def _spans(source: LinkedProgramManifest, allocations):
    abis = unique_state_abis(source)
    if len(abis) != 31:
        raise SchemaError("31 true EP4 source StateABI pages required",
                          path="source.state_abi")
    spans: list[dict[str, object]] = []
    for die in range(4):
        experts = sorted((abi for abi in abis if abi.die_id == die and
                          abi.kind is StateKind.TRAINABLE_PARAMETER),
                         key=lambda abi: abi.address)
        shared = sorted((abi for abi in abis if abi.die_id == die and
                         abi.kind is StateKind.PARAMETER),
                        key=lambda abi: abi.address)
        if (len(experts) != 2 or sum(abi.size_bytes for abi in experts) != 384
                or len(shared) != (13 if die == 0 else 2)
                or sum(abi.size_bytes for abi in shared) != (616 if die == 0 else 64)):
            raise SchemaError("actual EP4 source physical weight inventory changed",
                              path="source.state_abi")
        for kind, items, allocation in (
            ("expert_retention", experts, allocations[die, "expert"]),
            ("shared_or_router", shared, allocations[die, "shared"]),
        ):
            cursor, reserved, _ = allocation
            used = 0
            for abi in items:
                spans.append({
                    "die_id": die, "kind": kind,
                    "source_state_ref": abi.state_ref,
                    "source_hbm_address": abi.address,
                    "external_address": cursor,
                    "hbm_address": HBM_HOME_BASE[die] + WEIGHT_SLOT_OFFSET,
                    "size_bytes": abi.size_bytes,
                })
                cursor += abi.size_bytes
                used += abi.size_bytes
            if (kind == "expert_retention" and used != reserved
                    or die == 0 and kind == "shared_or_router" and used != reserved
                    or die != 0 and kind == "shared_or_router" and used > reserved):
                raise SchemaError("actual physical pages exceed P3 external allocation",
                                  path="source.state_abi")
    if (len(spans) != 27
            or sum(item["size_bytes"] for item in spans) != PHYSICAL_PARAMETER_BYTES):
        raise SchemaError("27 true EP4 physical parameter pages missing",
                          path="source.state_abi")
    latest_kv = sorted((abi for abi in abis if abi.kind in
                        (StateKind.KV_KEY, StateKind.KV_VALUE)), key=lambda a: a.address)
    kv: list[dict[str, object]] = []
    if len(latest_kv) != 4:
        raise SchemaError("four true rank0 KV pages missing", path="source.state_abi")
    for index, abi in enumerate(latest_kv):
        if abi.die_id != 0 or abi.address != KV_SOURCE_BASE + index * KV_SOURCE_STRIDE:
            raise SchemaError("KV physical pages are not true rank0 homes",
                              path="source.state_abi")
        kv.append({
            "die_id": 0, "kind": abi.kind.value,
            "source_state_ref": abi.state_ref,
            "source_hbm_address": abi.address,
            "external_address": KV_EXTERNAL_BASE + index * KV_SLOT_BYTES,
            "hbm_address": KV_SLOT_OFFSET + index * KV_SLOT_BYTES,
            "size_bytes": KV_SLOT_BYTES,
        })
    return spans, kv


def build_moe_inference_paged_runtime_ep4(
    *, resident, offload,
    source_manifests: tuple[LinkedProgramManifest, ...],
    paged_manifests: tuple[LinkedProgramManifest, ...],
    fabric: ExternalMemoryFabric,
) -> dict[str, object]:
    """Sign every actual four-core weight, expert retention and KV gate."""
    if len(source_manifests) != 3 or len(paged_manifests) != 3:
        raise SchemaError("three full MoE infer segments required", path="manifests")
    fabric.validate()
    if (
        len(fabric.external_capacities) != 1 or
        fabric.external_capacities[0].capacity_bytes != EXTERNAL_CAPACITY_BYTES or
        len(fabric.hbm_capacities) != 4 or
        {item.location_ref: (item.base_address, item.capacity_bytes)
         for item in fabric.hbm_capacities} !=
            {f"die:{die}": (HBM_HOME_BASE[die], HBM_CAPACITY_BYTES)
             for die in range(4)} or
        len(fabric.links) != 1 or len(fabric.connections) != 4
    ):
        raise SchemaError("official one-link/four-die low-HBM fabric changed", path="fabric")
    if any(getattr(resident.request, field) != getattr(offload.request, field)
           for field in ("model", "steps", "mesh", "parallel", "execution")):
        raise SchemaError("resident/offload useful model identity changed", path="offload.request")
    external_allocations = _external_parameter_allocations(offload)
    requests = {item.id: item for item in offload.memory_plan.requests}
    for die in range(4):
        end = max(
            item.address + requests[item.request_ref].size_bytes - HBM_HOME_BASE[die]
            for item in offload.memory_plan.allocations
            if requests[item.request_ref].tier is MemoryTier.HBM and
            requests[item.request_ref].location_ref == f"die:{die}"
        )
        if end != (464, 464, 464, 456)[die]:
            raise SchemaError("P3 timed workspace/KV reservation changed", path="offload.memory_plan")
    physical_kv_inventory = tuple(item for item in offload.state_inventory
                                  if item.object_kind is MemoryObjectKind.KV and
                                  item.logical_rank == 0)
    if len(physical_kv_inventory) != 1 or physical_kv_inventory[0].size_bytes != KV_PHYSICAL_BYTES:
        raise SchemaError("source rank0 physical KV inventory changed", path="offload.state_inventory")

    parameter_spans, kv_spans = _spans(source_manifests[2], external_allocations)
    by_source = {(item["die_id"], item["source_hbm_address"]): item
                 for item in (*parameter_spans, *kv_spans)}
    if len(by_source) != 31:
        raise SchemaError("physical source pages overlap", path="source.state_abi")
    events: list[dict[str, object]] = []
    per_core_counts: list[dict[str, int]] = []
    for segment, (source, paged) in enumerate(zip(source_manifests, paged_manifests)):
        source.validate("moe_paged.source")
        paged.validate("moe_paged.linked")
        original_fragments = {item.id: item for item in source.fragments}
        relocated_fragments = {item.id: item for item in paged.fragments}
        old_bindings = {(b.logical_core.die_id, b.logical_core.local_core_id,
                         b.fragment_id, b.fragment_record_index): b
                        for b in source.state_operand_bindings}
        new_bindings = {(b.logical_core.die_id, b.logical_core.local_core_id,
                         b.fragment_id, b.fragment_record_index): b
                        for b in paged.state_operand_bindings}
        old_abis = {a.id: a for a in unique_state_abis(source)}
        new_abis = {a.id: a for a in unique_state_abis(paged)}
        if len(source.core_streams) != 4 or len(paged.core_streams) != 4:
            raise SchemaError("four real MoE cores required", path="manifests")
        counts = {"weight_restore_before_load": 0,
                  "expert_writeback_after_store": 0,
                  "kv_restore_before_load": 0,
                  "kv_writeback_after_store": 0}
        for old_core, new_core in zip(source.core_streams, paged.core_streams):
            if (old_core.logical_core != new_core.logical_core or
                old_core.runtime_core_id != new_core.runtime_core_id or
                len(old_core.records) != len(new_core.records)):
                raise SchemaError("relink changed MoE runtime core streams", path="manifests")
            logical_core = new_core.logical_core
            core_id = new_core.runtime_core_id
            for index, (old_ref, new_ref) in enumerate(zip(old_core.records,
                                                            new_core.records)):
                old_record, _ = _record(original_fragments[old_ref.fragment_id],
                                        logical_core, old_ref.fragment_record_index)
                new_record, new_stream = _record(relocated_fragments[new_ref.fragment_id],
                                                 logical_core, new_ref.fragment_record_index)
                old_key = (logical_core.die_id, logical_core.local_core_id,
                           old_ref.fragment_id, old_ref.fragment_record_index)
                new_key = (logical_core.die_id, logical_core.local_core_id,
                           new_ref.fragment_id, new_ref.fragment_record_index)
                if (old_ref.source_global_action_id != new_ref.source_global_action_id
                    or old_record != new_record or
                    (old_key in old_bindings) != (new_key in new_bindings)):
                    raise SchemaError("full MoE command/StateABI closure changed", path="manifests")
                if new_key not in new_bindings:
                    if new_record.opcode in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE):
                        raise SchemaError("actual MoE LSU lacks StateOperandBinding", path="manifests")
                    continue
                old_abi = old_abis[old_bindings[old_key].state_abi_id]
                new_abi = new_abis[new_bindings[new_key].state_abi_id]
                if (old_abi.state_ref != new_abi.state_ref or
                    old_abi.kind is not new_abi.kind or
                    old_abi.size_bytes != new_abi.size_bytes or
                    old_abi.die_id != new_abi.die_id):
                    raise SchemaError("relink changed logical shared/expert/KV state", path="manifests")
                span = by_source.get((old_abi.die_id, old_abi.address))
                if span is None or new_abi.address != span["hbm_address"]:
                    raise SchemaError("actual MoE state has no true physical external page", path="manifests")
                lsu_bytes = _lsu_size(new_record)
                hbm_address = new_abi.address + _hbm_addend(new_stream,
                                                              new_ref.fragment_record_index)
                if new_abi.kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
                    kind = ("kv_restore_before_load" if new_record.opcode is RecordOpcode.LSU_LOAD
                            else "kv_writeback_after_store")
                    if (new_record.opcode not in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE)
                        or (segment == 0 and kind != "kv_writeback_after_store")
                        or lsu_bytes != (new_abi.size_bytes if kind.endswith("before_load")
                                         else 32 if segment == 0 else 16)
                        or hbm_address != new_abi.address +
                            (0 if kind.endswith("before_load") or segment == 0
                             else 32 + 16 * (segment - 1))):
                        raise SchemaError("KV Store must use the true suffix relocation/full-page authority", path="manifests")
                elif new_abi.kind is StateKind.TRAINABLE_PARAMETER:
                    kind = ("weight_restore_before_load" if new_record.opcode is RecordOpcode.LSU_LOAD
                            else "expert_writeback_after_store")
                    if (new_record.opcode not in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE)
                        or lsu_bytes != 192 or hbm_address != new_abi.address):
                        raise SchemaError("true expert192 roundtrip LSU changed", path="manifests")
                elif new_abi.kind is StateKind.PARAMETER:
                    kind = "weight_restore_before_load"
                    if (new_record.opcode is not RecordOpcode.LSU_LOAD or
                        lsu_bytes != new_abi.size_bytes or hbm_address != new_abi.address):
                        raise SchemaError("shared/router true weight LSU changed", path="manifests")
                else:
                    raise SchemaError("unexpected persistent StateABI role", path="manifests")
                events.append({
                    "segment_index": segment, "runtime_core_id": core_id,
                    "linked_record_index": index,
                    "fragment_id": new_ref.fragment_id,
                    "fragment_record_index": new_ref.fragment_record_index,
                    "kind": kind, "state_ref": new_abi.state_ref,
                    "state_abi_id": new_abi.id,
                    "source_hbm_address": old_abi.address,
                    "external_address": span["external_address"],
                    "hbm_address": new_abi.address,
                    "lsu_address": hbm_address,
                    "lsu_size_bytes": lsu_bytes,
                    "dma_size_bytes": new_abi.size_bytes,
                })
                counts[kind] += 1
        if counts != {
            "weight_restore_before_load": 27,
            "expert_writeback_after_store": 8,
            "kv_restore_before_load": 0 if segment == 0 else 4,
            "kv_writeback_after_store": 4,
        }:
            raise SchemaError("MoE 27 weights/8 expert roundtrip/4KV exact linked gates changed", path="manifests")
        per_core_counts.append(counts)
    if len(events) != 125:
        raise SchemaError("all 125 physical MoE LSU gates required", path="events")
    reads = sum(item["dma_size_bytes"] for item in events
                if item["kind"].endswith("before_load"))
    writes = sum(item["dma_size_bytes"] for item in events
                 if item["kind"].endswith("after_store"))
    if (reads, writes) != (7480, 5184):
        raise SchemaError("independent true StateABI transfer byte oracle changed", path="events")
    # Seed exactly the 27 physical parameter pages. Ranks 1..3 each have
    # 552B of P3 shared replica without a matching full-model HBM ABI; those
    # bytes are not invented, seeded, or counted as actual traffic.
    seeds = [{"external_address": item["external_address"],
              "payload_hex": bytes((item["external_address"] + j) % 251 + 1
                                   for j in range(item["size_bytes"])).hex()}
             for item in parameter_spans]
    seeds.extend({"external_address": item["external_address"],
                  "payload_hex": bytes(item["size_bytes"]).hex()}
                 for item in kv_spans)
    key: dict[str, object] = {
        "schema_version": "wafer_frontend.moe_inference_paged_runtime/v2alpha1",
        "producer_pass": "moe_inference_paged_runtime",
        "request_digest": offload.request.digest,
        "model_digest": canonical_digest(resident.request.model),
        "logical_graph_digest": offload.logical_graph_digest,
        "offload_memory_plan_digest": canonical_digest(offload.memory_plan),
        "source_external_parameter_allocation_refs": [
            external_allocations[die, role][2] for die in range(4)
            for role in ("expert", "shared")
        ],
        "source_rank0_kv_inventory_digest": canonical_digest(physical_kv_inventory),
        "linked_manifest_ids": [item.id for item in paged_manifests],
        "linked_manifest_digests": [canonical_digest(item) for item in paged_manifests],
        "hbm_capacity_bytes_per_die": HBM_CAPACITY_BYTES,
        "workspace_end_bytes_per_die": P3_WORKSPACE_END_BYTES,
        "physical_parameter_bytes": PHYSICAL_PARAMETER_BYTES,
        "physical_kv_bytes": KV_PHYSICAL_BYTES,
        "active_die_ids": list(range(4)),
        "expected_dma_events": 125,
        "expected_external_read_bytes": 7480,
        "expected_external_write_bytes": 5184,
        "highest_relative_state_end_bytes": HIGHEST_PAGED_END_BYTES,
        "fabric": json.loads(canonical_json(fabric)),
        "parameter_spans": parameter_spans,
        "kv_spans": kv_spans,
        "events": events,
        "seeds": seeds,
    }
    key["id"] = "moe_inference_paged_runtime_" + hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return key


__all__ = ["build_moe_inference_paged_runtime_ep4"]
