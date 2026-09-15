"""Source-bound real-LSU DMA sidecar for fixed two-layer Dense inference."""

from __future__ import annotations

import hashlib
import json

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    CommandFragment,
    LinkedProgramManifest,
    RecordOpcode,
    SemanticOperandId,
    StateKind,
)
from ..schema.external_memory import ExternalMemoryFabric
from ..schema.memory_plan import MemoryObjectKind, MemoryTier
from ..schema.serde import canonical_digest, canonical_json
from .dense_inference_paged_compile_sequence import (
    HBM_CAPACITY_BYTES,
    KV_SLOT_BASE,
    KV_SLOT_SIZE_BYTES,
    KV_SOURCE_BASE,
    PAGED_HIGHEST_END_BYTES,
    P3_WORKSPACE_END_BYTES,
    PARAMETER_SOURCE_BYTES,
    WEIGHT_SLOT_BASE,
)


def _abi_by_id(manifest: LinkedProgramManifest):
    return {
        abi.id: abi
        for fragment in manifest.fragments
        for abi in fragment.state_abi
    }


def _record_by_ref(manifest: LinkedProgramManifest):
    fragments = {item.id: item for item in manifest.fragments}
    return {
        (item.id, index): record
        for item in fragments.values()
        for stream in item.core_streams
        for index, record in enumerate(stream.records)
    }, {
        (item.id, index): stream
        for item in fragments.values()
        for stream in item.core_streams
        for index, _ in enumerate(stream.records)
    }


def _state_binding_by_ref(manifest: LinkedProgramManifest):
    return {
        (item.fragment_id, item.fragment_record_index): item
        for item in manifest.state_operand_bindings
    }


def _lsu_size(record) -> int:
    found = tuple(
        item.literal_value
        for item in record.operands
        if item.name == "size_bytes"
    )
    if len(found) != 1 or type(found[0]) is not int or found[0] <= 0:
        raise SchemaError("linked LSU has no exact size literal", path="record")
    return found[0]


def _hbm_addend(stream, local_index: int) -> int:
    found = tuple(
        item.addend
        for item in stream.address_relocations
        if item.record_index == local_index
        and item.operand_id is SemanticOperandId.HBM_ADDRESS
    )
    if len(found) != 1:
        raise SchemaError("linked LSU has no exact HBM relocation", path="record")
    return found[0]


def build_dense_inference_paged_runtime(
    *,
    resident,
    offload,
    source_manifests: tuple[LinkedProgramManifest, ...],
    paged_manifests: tuple[LinkedProgramManifest, ...],
    fabric: ExternalMemoryFabric,
) -> dict[str, object]:
    """Sign source-bound pages and their actual, ordered linked LSU windows."""

    if len(source_manifests) != 3 or len(paged_manifests) != 3:
        raise SchemaError("three real Prefill+2Decode manifests required", path="manifests")
    fabric.validate()
    if (
        len(fabric.hbm_capacities) != 1 or
        fabric.hbm_capacities[0].capacity_bytes != HBM_CAPACITY_BYTES or
        len(fabric.external_capacities) != 1 or
        fabric.external_capacities[0].capacity_bytes != 54400
    ):
        raise SchemaError("fixed bounded physical fabric drifted", path="fabric")
    if (
        resident.request.model != offload.request.model
        or resident.request.steps != offload.request.steps
        or resident.request.mesh != offload.request.mesh
        or resident.request.parallel != offload.request.parallel
    ):
        raise SchemaError("resident/offload model identity changed", path="offload.request")
    requests = {item.id: item for item in offload.memory_plan.requests}
    external_parameters = tuple(
        allocation
        for allocation in offload.memory_plan.allocations
        if requests[allocation.request_ref].object_kind is MemoryObjectKind.PARAMETER
        and requests[allocation.request_ref].tier is MemoryTier.EXTERNAL
    )
    if (
        len(external_parameters) != 1
        or external_parameters[0].address != 0
        or requests[external_parameters[0].request_ref].size_bytes !=
            PARAMETER_SOURCE_BYTES
    ):
        raise SchemaError("P3 external parameter allocation is not exact", path="offload.memory_plan")
    hbm_end = max(
        allocation.address + requests[allocation.request_ref].size_bytes
        for allocation in offload.memory_plan.allocations
        if requests[allocation.request_ref].tier is MemoryTier.HBM
    )
    if hbm_end != P3_WORKSPACE_END_BYTES:
        raise SchemaError("P3 workspace/KV reservation changed", path="offload.memory_plan")
    kv_inventory = tuple(
        item for item in offload.state_inventory
        if item.object_kind is MemoryObjectKind.KV
    )
    if sum(item.size_bytes for item in kv_inventory) != 768:
        raise SchemaError("P3 KV final source inventory is not 768B", path="offload.state_inventory")

    source_abis = tuple(_abi_by_id(item) for item in source_manifests)
    paged_abis = tuple(_abi_by_id(item) for item in paged_manifests)
    original_weights = tuple(sorted(
        (abi for abi in source_abis[0].values()
         if abi.kind is StateKind.PARAMETER),
        key=lambda abi: abi.address,
    ))
    latest_kv = tuple(sorted(
        (abi for abi in source_abis[2].values()
         if abi.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)),
        key=lambda abi: abi.address,
    ))
    if len(original_weights) != 15 or len(latest_kv) != 4:
        raise SchemaError("true source parameter/KV physical ABI drifted", path="source_manifests")
    weights = tuple({
        "kind": "parameter",
        "state_ref": abi.state_ref,
        "source_hbm_address": abi.address,
        "external_address": abi.address,
        "hbm_address": WEIGHT_SLOT_BASE,
        "size_bytes": abi.size_bytes,
    } for abi in original_weights)
    kv = tuple({
        "kind": abi.kind.value,
        "state_ref": abi.state_ref,
        "source_hbm_address": abi.address,
        "external_address": abi.address,
        "hbm_address": KV_SLOT_BASE + (abi.address - KV_SOURCE_BASE),
        "size_bytes": KV_SLOT_SIZE_BYTES,
    } for abi in latest_kv)
    if tuple(item["source_hbm_address"] for item in kv) != tuple(
        KV_SOURCE_BASE + KV_SLOT_SIZE_BYTES * index for index in range(4)
    ):
        raise SchemaError("four source KV page homes drifted", path="source_manifests")

    events: list[dict[str, object]] = []
    for segment, (source, paged) in enumerate(zip(source_manifests, paged_manifests)):
        source.validate("dense_paged.source")
        paged.validate("dense_paged.linked")
        old_records, old_streams = _record_by_ref(source)
        new_records, new_streams = _record_by_ref(paged)
        old_bindings = _state_binding_by_ref(source)
        new_bindings = _state_binding_by_ref(paged)
        if len(source.core_streams) != 1 or len(paged.core_streams) != 1:
            raise SchemaError("requires actual die0 core0 linked stream", path="manifests")
        if len(source.core_streams[0].records) != len(paged.core_streams[0].records):
            raise SchemaError("relink changed linked record count", path="manifests")
        local_counts = {"weight_restore_before_load": 0,
                        "kv_restore_before_load": 0,
                        "kv_writeback_after_store": 0}
        for index, (old_ref, new_ref) in enumerate(zip(
            source.core_streams[0].records,
            paged.core_streams[0].records,
        )):
            old_key = (old_ref.fragment_id, old_ref.fragment_record_index)
            new_key = (new_ref.fragment_id, new_ref.fragment_record_index)
            old_record = old_records[old_key]
            new_record = new_records[new_key]
            if (
                old_ref.source_global_action_id != new_ref.source_global_action_id
                or old_record != new_record
                or (old_key in old_bindings) != (new_key in new_bindings)
            ):
                raise SchemaError("relink changed useful records or StateABI closure", path="manifests")
            if new_key not in new_bindings:
                if new_record.opcode in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE):
                    raise SchemaError("unbound real LSU remains", path="manifests")
                continue
            old_abi = source_abis[segment][old_bindings[old_key].state_abi_id]
            new_abi = paged_abis[segment][new_bindings[new_key].state_abi_id]
            if (
                old_abi.state_ref != new_abi.state_ref
                or old_abi.size_bytes != new_abi.size_bytes
                or old_abi.kind is not new_abi.kind
            ):
                raise SchemaError("relink changed logical persistent state", path="manifests")
            size = _lsu_size(new_record)
            addend = _hbm_addend(new_streams[new_key], new_key[1])
            if new_abi.kind is StateKind.PARAMETER:
                kind = "weight_restore_before_load"
                if new_record.opcode is not RecordOpcode.LSU_LOAD:
                    raise SchemaError("weight page must be actual LSU Load", path="manifests")
            elif new_abi.kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
                kind = (
                    "kv_restore_before_load"
                    if new_record.opcode is RecordOpcode.LSU_LOAD
                    else "kv_writeback_after_store"
                )
                if new_record.opcode not in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE):
                    raise SchemaError("KV page is not actual LSU", path="manifests")
            else:
                raise SchemaError("unexpected persistent state role", path="manifests")
            event = {
                "segment_index": segment,
                "linked_record_index": index,
                "fragment_id": new_ref.fragment_id,
                "fragment_record_index": new_ref.fragment_record_index,
                "kind": kind,
                "state_ref": new_abi.state_ref,
                "state_abi_id": new_abi.id,
                "source_hbm_address": old_abi.address,
                "external_address": old_abi.address,
                "hbm_address": new_abi.address,
                "lsu_address": new_abi.address + addend,
                "lsu_size_bytes": size,
                "dma_size_bytes": new_abi.size_bytes,
            }
            events.append(event)
            local_counts[kind] += 1
        if local_counts != {
            "weight_restore_before_load": 15,
            "kv_restore_before_load": 0 if segment == 0 else 4,
            "kv_writeback_after_store": 4,
        }:
            raise SchemaError("actual segment LSU coverage drifted", path="manifests")
    if len(events) != 65:
        raise SchemaError("true linked sequence requires 65 DMA gates", path="events")
    if (
        sum(event["dma_size_bytes"] for event in events
            if event["kind"] != "kv_writeback_after_store") != 162112
        or sum(event["dma_size_bytes"] for event in events
               if event["kind"] == "kv_writeback_after_store") != 1920
    ):
        raise SchemaError("state ABI independent traffic byte oracle changed", path="events")
    parameter_seed = bytes((index % 251) + 1 for index in range(PARAMETER_SOURCE_BYTES))
    kv_seed = bytes(768)
    key: dict[str, object] = {
        "schema_version": "wafer_frontend.dense_inference_paged_runtime/v1alpha1",
        "producer_pass": "dense_inference_paged_runtime",
        "request_digest": offload.request.digest,
        "model_digest": canonical_digest(resident.request.model),
        "logical_graph_digest": offload.logical_graph_digest,
        "offload_memory_plan_digest": canonical_digest(offload.memory_plan),
        "source_parameter_allocation_ref": external_parameters[0].id,
        "source_kv_inventory_digest": canonical_digest(kv_inventory),
        "linked_manifest_ids": [item.id for item in paged_manifests],
        "linked_manifest_digests": [canonical_digest(item) for item in paged_manifests],
        "hbm_capacity_bytes": HBM_CAPACITY_BYTES,
        "workspace_end_bytes": P3_WORKSPACE_END_BYTES,
        "parameter_state_bytes": PARAMETER_SOURCE_BYTES,
        "kv_capacity_bytes": 768,
        "highest_paged_end_bytes": PAGED_HIGHEST_END_BYTES,
        "fabric": json.loads(canonical_json(fabric)),
        "weight_spans": list(weights),
        "kv_spans": list(kv),
        "events": events,
        "parameter_seed_hex": parameter_seed.hex(),
        "kv_seed_hex": kv_seed.hex(),
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    key["id"] = "dense_inference_paged_runtime_" + digest[:20]
    return key


__all__ = ["build_dense_inference_paged_runtime"]
