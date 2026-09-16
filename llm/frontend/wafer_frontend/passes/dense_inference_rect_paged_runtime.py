"""Relink and sign bounded-HBM paging for the true TP4 Dense rectangle."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    CommandFragment,
    LinkedProgramManifest,
    ManifestInputDigest,
    ManifestInputKind,
    RecordOpcode,
    RegionManifest,
    SemanticOperandId,
    StateABI,
    StateKind,
)
from ..schema.external_memory import ExternalMemoryFabric
from ..schema.memory_plan import MemoryTier
from ..schema.serde import canonical_digest, canonical_json
from .dense_inference_rect_physical_source import PhysicalInferenceOffloadSource


HBM_CAPACITY_BYTES_PER_DIE = 12288
WEIGHT_SLOT_BASE = 1600
WEIGHT_SLOT_SIZE_BYTES = 8192
KV_SLOT_BASE = 9792
KV_SLOT_SIZE_BYTES = 384
PAGED_HIGHEST_END_BYTES = 11328
PARAMETER_BYTES_PER_DIE = 26944
KV_BYTES_PER_DIE = 1536


def _fragment(item):
    return item.fragment if isinstance(item, RegionManifest) else item


def _state_abis(manifest: LinkedProgramManifest) -> dict[str, StateABI]:
    result: dict[str, StateABI] = {}
    for linked in manifest.fragments:
        for abi in _fragment(linked).state_abi:
            old = result.setdefault(abi.id, abi)
            if old != abi:
                raise SchemaError("shared StateABI definitions conflict",
                                  path="manifest.fragments.state_abi")
    return result


def _rebuild_fragment(linked, relocated: dict[str, StateABI]):
    source = _fragment(linked)
    key = source._semantic_key()
    key["state_abi"] = tuple(sorted(
        (relocated[item.id] for item in source.state_abi),
        key=lambda item: item.id,
    ))
    fragment = CommandFragment.create(producer_pass=source.producer_pass, **key)
    if isinstance(linked, RegionManifest):
        result = RegionManifest.create(
            producer_pass=linked.producer_pass,
            region_id=linked.region_id,
            fusion_plan_id=linked.fusion_plan_id,
            target_dies=linked.target_dies,
            fragment=fragment,
        )
    else:
        result = fragment
    result.validate("dense_rect_paged.fragment")
    return result


def relink_dense_inference_rect_paged_segment(
    source: LinkedProgramManifest,
    segment_index: int,
) -> LinkedProgramManifest:
    """Time-multiplex each Die's true parameter ABIs into one local slot."""

    if segment_index not in (0, 1, 2):
        raise SchemaError("requires Prefill+2Decode segment", path="segment_index")
    source.validate("dense_rect_paged.source")
    if source.producer_pass != "manifest_linker" or len(source.core_streams) != 4:
        raise SchemaError("requires a true four-stream TP4 linked source",
                          path="source")
    stream_dies = tuple(item.logical_core.die_id for item in source.core_streams)
    if set(stream_dies) != {0, 1, 2, 3} or len(set(stream_dies)) != 4:
        raise SchemaError("linked streams must bijectively cover four Dies",
                          path="source.core_streams")
    abis = _state_abis(source)
    if len(abis) != 76:
        raise SchemaError("requires 76 true TP4 physical StateABIs",
                          path="source.state_abi")

    die_bases: dict[int, int] = {}
    relocated: dict[str, StateABI] = {}
    page_bytes = 256 + 64 * segment_index
    for die in range(4):
        weights = sorted(
            (item for item in abis.values()
             if item.die_id == die and item.kind is StateKind.PARAMETER),
            key=lambda item: item.address,
        )
        kv = sorted(
            (item for item in abis.values()
             if item.die_id == die and item.kind in
             (StateKind.KV_KEY, StateKind.KV_VALUE)),
            key=lambda item: item.address,
        )
        if len(weights) != 15 or len(kv) != 4:
            raise SchemaError("each Die requires 15 parameter and four KV ABIs",
                              path=f"source.die[{die}]")
        base = weights[0].address
        die_bases[die] = base
        cursor = base
        for item in weights:
            if item.address != cursor or item.size_bytes <= 0 or (
                item.size_bytes > WEIGHT_SLOT_SIZE_BYTES
            ):
                raise SchemaError("parameter ABIs are not a tight pageable source",
                                  path=f"source.die[{die}]")
            cursor += item.size_bytes
        if cursor - base != PARAMETER_BYTES_PER_DIE:
            raise SchemaError("physical parameter bytes per Die changed",
                              path=f"source.die[{die}]")
        if tuple(item.address for item in kv) != tuple(
            cursor + KV_SLOT_SIZE_BYTES * index for index in range(4)
        ) or any(item.size_bytes != page_bytes for item in kv):
            raise SchemaError("four versioned KV homes per Die changed",
                              path=f"source.die[{die}]")
        for item in (*weights, *kv):
            local = (
                WEIGHT_SLOT_BASE
                if item.kind is StateKind.PARAMETER
                else KV_SLOT_BASE + item.address - cursor
            )
            relocated[item.id] = StateABI.create(
                state_ref=item.state_ref,
                hbm_binding_ref=item.hbm_binding_ref,
                kind=item.kind,
                lifetime=item.lifetime,
                access=item.access,
                shape=item.shape,
                dtype=item.dtype,
                layout=item.layout,
                die_id=item.die_id,
                address=base + local,
                size_bytes=item.size_bytes,
                alignment_bytes=item.alignment_bytes,
            )
    if PAGED_HIGHEST_END_BYTES > HBM_CAPACITY_BYTES_PER_DIE:
        raise SchemaError("paged slots exceed bounded per-Die HBM", path="slots")

    rebuilt_by_old = {
        old.id: _rebuild_fragment(old, relocated) for old in source.fragments
    }
    if len(rebuilt_by_old) != len(source.fragments):
        raise SchemaError("source fragment identities repeat", path="source.fragments")
    fragment_ids = {old: new.id for old, new in rebuilt_by_old.items()}
    rebuilt_artifacts = dict(rebuilt_by_old)
    for old in source.fragments:
        if isinstance(old, RegionManifest):
            rebuilt = rebuilt_by_old[old.id]
            rebuilt_artifacts[old.fragment.id] = rebuilt.fragment
            fragment_ids[old.fragment.id] = rebuilt.fragment.id
    state_by_binding = {item.hbm_binding_ref: item for item in relocated.values()}
    inputs = tuple(sorted((
        ManifestInputDigest(
            item.kind,
            rebuilt_artifacts[item.artifact_id].id,
            rebuilt_artifacts[item.artifact_id].schema_version,
            canonical_digest(rebuilt_artifacts[item.artifact_id]),
        ) if item.kind in (
            ManifestInputKind.COMMAND_FRAGMENT,
            ManifestInputKind.REGION_MANIFEST,
        ) else item
        for item in source.input_digests
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    key = source._semantic_key()
    key.update(
        input_digests=inputs,
        fragments=tuple(sorted(rebuilt_by_old.values(), key=lambda item: item.id)),
        fragment_interfaces=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in source.fragment_interfaces
        ), key=lambda item: item.fragment_id)),
        core_streams=tuple(
            replace(stream, records=tuple(
                replace(ref, fragment_id=fragment_ids[ref.fragment_id])
                for ref in stream.records
            )) for stream in source.core_streams
        ),
        program_symbol_definitions=tuple(
            replace(item, value=state_by_binding[item.symbol.source_ref].address)
            if item.symbol.source_ref in state_by_binding else item
            for item in source.program_symbol_definitions
        ),
        address_operand_bindings=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in source.address_operand_bindings
        ), key=lambda item: (item.logical_core.die_id,
                            item.logical_core.local_core_id,
                            item.fragment_id, item.fragment_record_index,
                            int(item.operand_id)))),
        state_operand_bindings=tuple(sorted((
            replace(item,
                    fragment_id=fragment_ids[item.fragment_id],
                    state_abi_id=relocated[item.state_abi_id].id)
            for item in source.state_operand_bindings
        ), key=lambda item: (item.logical_core.die_id,
                            item.logical_core.local_core_id,
                            item.fragment_id, item.fragment_record_index,
                            int(item.operand_id)))),
    )
    result = LinkedProgramManifest.create(producer_pass=source.producer_pass, **key)
    result.validate("dense_rect_paged.linked")
    return result


def _insert_record_identity(records, streams, key, record, stream) -> None:
    if key in records and records[key] != record:
        raise SchemaError("outer/inner fragment record identity conflicts",
                          path="manifest.fragments")
    records[key] = record
    streams[key] = stream


def _record_maps(manifest: LinkedProgramManifest):
    records = {}
    streams = {}
    for linked in manifest.fragments:
        fragment = _fragment(linked)
        fragment_ids = (linked.id, fragment.id) if isinstance(
            linked, RegionManifest
        ) else (linked.id,)
        for stream in fragment.core_streams:
            for index, record in enumerate(stream.records):
                for fragment_id in fragment_ids:
                    key = (stream.logical_core.die_id,
                           stream.logical_core.local_core_id,
                           fragment_id, index)
                    _insert_record_identity(records, streams, key,
                                            record, stream)
    return records, streams


def _bindings(manifest: LinkedProgramManifest):
    return {
        (item.logical_core.die_id, item.logical_core.local_core_id,
         item.fragment_id, item.fragment_record_index): item
        for item in manifest.state_operand_bindings
    }


def _lsu_size(record) -> int:
    values = tuple(item.literal_value for item in record.operands
                   if item.name == "size_bytes")
    if len(values) != 1 or type(values[0]) is not int or values[0] <= 0:
        raise SchemaError("linked LSU has no exact size literal", path="record")
    return values[0]


def _hbm_addend(stream, index: int) -> int:
    values = tuple(item.addend for item in stream.address_relocations
                   if item.record_index == index and
                   item.operand_id is SemanticOperandId.HBM_ADDRESS)
    if len(values) != 1:
        raise SchemaError("linked LSU has no exact HBM relocation", path="record")
    return values[0]


def build_dense_inference_rect_paged_runtime(
    *,
    physical_source: PhysicalInferenceOffloadSource,
    source_manifests: tuple[LinkedProgramManifest, ...],
    paged_manifests: tuple[LinkedProgramManifest, ...],
    fabric: ExternalMemoryFabric,
) -> dict[str, object]:
    """Sign all four per-core LSU timelines and their physical external pages."""

    if len(source_manifests) != 3 or len(paged_manifests) != 3:
        raise SchemaError("three Prefill+2Decode manifests required", path="manifests")
    fabric.validate()
    if (len(fabric.external_capacities) != 1 or
            fabric.external_capacities[0].id != physical_source.external_capacity.id or
            len(fabric.hbm_capacities) != 4 or
            any(item.capacity_bytes != HBM_CAPACITY_BYTES_PER_DIE
                for item in fabric.hbm_capacities) or
            len(fabric.links) != 1 or len(fabric.connections) != 4 or
            fabric.links[0].ingress_die_id != 0 or
            fabric.links[0].queue_depth != 4 or
            fabric.links[0].max_outstanding != 4 or
            {item.link_ref for item in fabric.connections} != {fabric.links[0].id}):
        raise SchemaError("requires one external ingress and four routed TP4 HBM homes",
                          path="fabric")
    requests = {item.id: item for item in
                physical_source.external_manifest.memory_plan.requests}
    allocations = {item.id: item for item in
                   physical_source.external_manifest.memory_plan.allocations}
    declaration_by_abi = {
        abi: item for item in physical_source.declarations
        for abi in item.linked_state_abi_ids
    }
    connection_by_die = {item.target_die_id: item.id for item in fabric.connections}
    if len(declaration_by_abi) != 108 or set(connection_by_die) != {0, 1, 2, 3}:
        raise SchemaError("physical declaration or connection closure is incomplete",
                          path="physical_source")

    source_abis = tuple(_state_abis(item) for item in source_manifests)
    paged_abis = tuple(_state_abis(item) for item in paged_manifests)
    runtime_core_by_die = {
        item.logical_core.die_id: item.runtime_core_id
        for item in paged_manifests[0].core_streams
    }
    spans = []
    seed = bytearray(physical_source.external_capacity.capacity_bytes)
    for declaration in physical_source.declarations:
        allocation = allocations[declaration.external_allocation_ref]
        request = requests[allocation.request_ref]
        if request.tier is not MemoryTier.EXTERNAL or (
            request.size_bytes != declaration.physical_bytes
        ):
            raise SchemaError("declaration does not own its exact external allocation",
                              path="physical_source.declarations")
        latest = source_abis[2][declaration.linked_state_abi_ids[2]]
        die_base = min(
            item.address for item in source_abis[0].values()
            if item.die_id == declaration.die_id and
            item.kind is StateKind.PARAMETER
        )
        local = (WEIGHT_SLOT_BASE if latest.kind is StateKind.PARAMETER else
                 KV_SLOT_BASE + latest.address -
                 (die_base + PARAMETER_BYTES_PER_DIE))
        spans.append({
            "kind": declaration.role,
            "state_ref": latest.state_ref,
            "die_id": declaration.die_id,
            "runtime_core_id": runtime_core_by_die[declaration.die_id],
            "connection_ref": connection_by_die[declaration.die_id],
            "source_hbm_address": declaration.source_hbm_address,
            "external_address": allocation.address,
            "hbm_address": local,
            "size_bytes": declaration.physical_bytes,
        })
        if latest.kind is StateKind.PARAMETER:
            for index in range(declaration.physical_bytes):
                seed[allocation.address + index] = (allocation.address + index) % 251 + 1

    span_by_source = {
        (item["die_id"], item["source_hbm_address"]): item for item in spans
    }
    events = []
    event_index = 0
    for segment, (source, paged) in enumerate(zip(source_manifests, paged_manifests)):
        old_records, _ = _record_maps(source)
        new_records, new_fragment_streams = _record_maps(paged)
        old_bindings, new_bindings = _bindings(source), _bindings(paged)
        new_stream_by_die = {item.logical_core.die_id: item
                             for item in paged.core_streams}
        for old_stream, new_stream in zip(source.core_streams, paged.core_streams):
            die = new_stream.logical_core.die_id
            if (old_stream.logical_core != new_stream.logical_core or
                    old_stream.runtime_core_id != new_stream.runtime_core_id or
                    len(old_stream.records) != len(new_stream.records)):
                raise SchemaError("relink changed a physical core stream",
                                  path="manifests")
            counts = {"weight_restore_before_load": 0,
                      "kv_restore_before_load": 0,
                      "kv_writeback_after_store": 0}
            for linked_index, (old_ref, new_ref) in enumerate(zip(
                    old_stream.records, new_stream.records)):
                old_key = (die, old_stream.logical_core.local_core_id,
                           old_ref.fragment_id, old_ref.fragment_record_index)
                new_key = (die, new_stream.logical_core.local_core_id,
                           new_ref.fragment_id, new_ref.fragment_record_index)
                old_record, new_record = old_records[old_key], new_records[new_key]
                if (old_ref.source_global_action_id != new_ref.source_global_action_id or
                        old_record != new_record or
                        (old_key in old_bindings) != (new_key in new_bindings)):
                    raise SchemaError("relink changed useful records or bindings",
                                      path="manifests")
                if new_key not in new_bindings:
                    if new_record.opcode in (RecordOpcode.LSU_LOAD,
                                             RecordOpcode.LSU_STORE):
                        raise SchemaError("unbound real LSU remains", path="manifests")
                    continue
                old_abi = source_abis[segment][old_bindings[old_key].state_abi_id]
                new_abi = paged_abis[segment][new_bindings[new_key].state_abi_id]
                if (old_abi.state_ref != new_abi.state_ref or
                        old_abi.size_bytes != new_abi.size_bytes or
                        old_abi.kind is not new_abi.kind or old_abi.die_id != die):
                    raise SchemaError("relink changed logical persistent state",
                                      path="manifests")
                span = span_by_source[(die, old_abi.address)]
                if old_abi.kind is StateKind.PARAMETER:
                    kind = "weight_restore_before_load"
                    if new_record.opcode is not RecordOpcode.LSU_LOAD:
                        raise SchemaError("parameter page is not an LSU Load",
                                          path="manifests")
                elif old_abi.kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
                    kind = ("kv_restore_before_load"
                            if new_record.opcode is RecordOpcode.LSU_LOAD
                            else "kv_writeback_after_store")
                else:
                    raise SchemaError("unexpected persistent state role",
                                      path="manifests")
                size = _lsu_size(new_record)
                addend = _hbm_addend(new_fragment_streams[new_key], new_key[3])
                events.append({
                    "event_index": event_index,
                    "segment_index": segment,
                    "linked_record_index": linked_index,
                    "fragment_id": new_ref.fragment_id,
                    "fragment_record_index": new_ref.fragment_record_index,
                    "kind": kind,
                    "state_ref": new_abi.state_ref,
                    "state_abi_id": new_abi.id,
                    "die_id": die,
                    "runtime_core_id": new_stream.runtime_core_id,
                    "connection_ref": span["connection_ref"],
                    "source_hbm_address": old_abi.address,
                    "external_address": span["external_address"],
                    "hbm_address": span["hbm_address"],
                    "lsu_address": new_abi.address + addend,
                    "lsu_size_bytes": size,
                    "dma_size_bytes": new_abi.size_bytes,
                })
                counts[kind] += 1
                event_index += 1
            expected = {"weight_restore_before_load": 15,
                        "kv_restore_before_load": 0 if segment == 0 else 4,
                        "kv_writeback_after_store": 4}
            if counts != expected:
                raise SchemaError("per-Die LSU page lifecycle is incomplete",
                                  path=f"manifests[{segment}].die[{die}]")
        if set(new_stream_by_die) != {0, 1, 2, 3}:
            raise SchemaError("paged streams do not cover all Dies", path="manifests")
    if len(events) != 260:
        raise SchemaError("TP4 sequence requires exactly 260 DMA gates", path="events")

    key: dict[str, object] = {
        "schema_version": "wafer_frontend.dense_inference_paged_runtime/v1alpha2",
        "producer_pass": "dense_inference_rect_paged_runtime",
        "physical_source_digest": physical_source.digest,
        "external_materialization_digest": physical_source.external_materialization_digest,
        "linked_manifest_ids": [item.id for item in paged_manifests],
        "linked_manifest_digests": [canonical_digest(item) for item in paged_manifests],
        "mesh_rows": physical_source.external_manifest.request.mesh.rows,
        "mesh_columns": physical_source.external_manifest.request.mesh.columns,
        "hbm_capacity_bytes_per_die": HBM_CAPACITY_BYTES_PER_DIE,
        "weight_slot_base_bytes_per_die": WEIGHT_SLOT_BASE,
        "parameter_state_bytes": 4 * PARAMETER_BYTES_PER_DIE,
        "kv_capacity_bytes": 4 * KV_BYTES_PER_DIE,
        "highest_paged_end_bytes_per_die": PAGED_HIGHEST_END_BYTES,
        "fabric": json.loads(canonical_json(fabric)),
        "spans": spans,
        "events": events,
        "external_seed_hex": bytes(seed).hex(),
    }
    digest = hashlib.sha256(json.dumps(
        key, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    key["id"] = "dense_inference_rect_paged_runtime_" + digest[:20]
    return key


__all__ = [
    "build_dense_inference_rect_paged_runtime",
    "relink_dense_inference_rect_paged_segment",
]
