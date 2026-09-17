"""Source-bound 18KiB/Die external pager for physical TP6 Dense inference."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LinkedProgramManifest, ManifestInputDigest, ManifestInputKind,
    RecordOpcode, SemanticOperandId, StateABI, StateKind,
)
from ..schema.external_memory import ExternalMemoryFabric
from ..schema.memory_plan import MemoryTier
from ..schema.serde import canonical_digest, canonical_json
from .dense_inference_rect_physical_source import PhysicalInferenceOffloadSource
from .dense_inference_rect_paged_runtime import (
    _bindings, _hbm_addend, _lsu_size, _rebuild_fragment, _record_maps,
    _state_abis,
)

HBM_CAPACITY = 18432
WEIGHT_BASE = 1600
WEIGHT_SIZE = 12288
KV_BASE = 14016
KV_SLOT = 768
PAGED_END = KV_BASE + 4 * KV_SLOT
PARAMETER_BYTES_PER_DIE = 40416
KV_BYTES_PER_DIE = 3072


def tp6_route(rows: int, columns: int, die: int) -> tuple[int, ...]:
    """Deterministic shortest Manhattan route from physical ingress Die 0."""
    if not (1 <= rows <= 10 and 1 <= columns <= 10 and 0 <= die < rows * columns):
        raise SchemaError("invalid physical TP6 rectangle or route target", path="mesh")
    target_row, target_column = divmod(die, columns)
    route = [0]
    for column in range(1, target_column + 1):
        route.append(column)
    for row in range(1, target_row + 1):
        route.append(row * columns + target_column)
    return tuple(route)


def relink_dense_inference_tp6_paged_segment(
    source: LinkedProgramManifest, segment_index: int,
) -> LinkedProgramManifest:
    """Retarget complete parameter LSU pages and four stable KV homes per Die."""
    if segment_index not in (0, 1, 2):
        raise SchemaError("requires Prefill+2Decode", path="segment_index")
    source.validate("dense_tp6_paged.source")
    dies = {item.logical_core.die_id for item in source.core_streams}
    if (source.producer_pass != "manifest_linker" or len(source.core_streams) != 6
            or len(dies) != 6 or len(source.state_operand_bindings) !=
            (114 if segment_index == 0 else 138)):
        raise SchemaError("requires six real TP6 linked streams", path="source")
    abis = _state_abis(source)
    if len(abis) != 114:
        raise SchemaError("requires 114 real TP6 StateABIs", path="source.state_abi")
    relocated: dict[str, StateABI] = {}
    for die in sorted(dies):
        weights = sorted((item for item in abis.values()
                          if item.die_id == die and item.kind is StateKind.PARAMETER),
                         key=lambda item: item.address)
        kv = sorted((item for item in abis.values()
                     if item.die_id == die and item.kind in
                     (StateKind.KV_KEY, StateKind.KV_VALUE)),
                    key=lambda item: item.address)
        if (len(weights) != 15 or len(kv) != 4 or
                sum(item.size_bytes for item in weights) != PARAMETER_BYTES_PER_DIE or
                sum(item.size_bytes for item in kv) != 4 * (576 + 96 * segment_index)):
            raise SchemaError("per-Die physical parameter/KV inventory changed",
                              path=f"source.die[{die}]")
        die_base = weights[0].address
        cursor = die_base
        for item in weights:
            if item.address < cursor or item.size_bytes > WEIGHT_SIZE:
                raise SchemaError("parameter source overlaps or exceeds complete LSU slot",
                                  path=f"source.die[{die}]")
            cursor = item.address + item.size_bytes
        kv_source_base = kv[0].address
        if (kv_source_base < cursor or tuple(item.address for item in kv) !=
                tuple(kv_source_base + KV_SLOT * index for index in range(4))):
            raise SchemaError("four stable KV source homes changed",
                              path=f"source.die[{die}]")
        for item in (*weights, *kv):
            address = die_base + (WEIGHT_BASE if item.kind is StateKind.PARAMETER
                                  else KV_BASE + item.address - kv_source_base)
            if address + item.size_bytes > die_base + HBM_CAPACITY:
                raise SchemaError("paged state exceeds bounded physical HBM",
                                  path=f"source.die[{die}]")
            relocated[item.id] = StateABI.create(
                state_ref=item.state_ref, hbm_binding_ref=item.hbm_binding_ref,
                kind=item.kind, lifetime=item.lifetime, access=item.access,
                shape=item.shape, dtype=item.dtype, layout=item.layout,
                die_id=item.die_id, address=address,
                size_bytes=item.size_bytes,
                alignment_bytes=item.alignment_bytes,
            )
    rebuilt = {old.id: _rebuild_fragment(old, relocated)
               for old in source.fragments}
    if len(rebuilt) != len(source.fragments):
        raise SchemaError("duplicate source fragment identity", path="source.fragments")
    fragment_ids = {old: new.id for old, new in rebuilt.items()}
    artifacts = dict(rebuilt)
    for old in source.fragments:
        if hasattr(old, "fragment"):
            artifacts[old.fragment.id] = rebuilt[old.id].fragment
            fragment_ids[old.fragment.id] = rebuilt[old.id].fragment.id
    bindings = {item.hbm_binding_ref: item for item in relocated.values()}
    inputs = tuple(sorted((
        ManifestInputDigest(item.kind, artifacts[item.artifact_id].id,
                            artifacts[item.artifact_id].schema_version,
                            canonical_digest(artifacts[item.artifact_id]))
        if item.kind in (ManifestInputKind.COMMAND_FRAGMENT,
                         ManifestInputKind.REGION_MANIFEST) else item
        for item in source.input_digests
    ), key=lambda item: (item.kind.value, item.artifact_id)))
    key = source._semantic_key()
    key.update(
        input_digests=inputs,
        fragments=tuple(sorted(rebuilt.values(), key=lambda item: item.id)),
        fragment_interfaces=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in source.fragment_interfaces
        ), key=lambda item: item.fragment_id)),
        core_streams=tuple(replace(stream, records=tuple(
            replace(ref, fragment_id=fragment_ids[ref.fragment_id])
            for ref in stream.records)) for stream in source.core_streams),
        program_symbol_definitions=tuple(
            replace(item, value=bindings[item.symbol.source_ref].address)
            if item.symbol.source_ref in bindings else item
            for item in source.program_symbol_definitions),
        address_operand_bindings=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id])
            for item in source.address_operand_bindings
        ), key=lambda item: (item.logical_core.die_id,
                            item.logical_core.local_core_id, item.fragment_id,
                            item.fragment_record_index, int(item.operand_id)))),
        state_operand_bindings=tuple(sorted((
            replace(item, fragment_id=fragment_ids[item.fragment_id],
                    state_abi_id=relocated[item.state_abi_id].id)
            for item in source.state_operand_bindings
        ), key=lambda item: (item.logical_core.die_id,
                            item.logical_core.local_core_id, item.fragment_id,
                            item.fragment_record_index, int(item.operand_id)))),
    )
    result = LinkedProgramManifest.create(producer_pass=source.producer_pass, **key)
    result.validate("dense_tp6_paged.linked")
    return result


def build_dense_inference_tp6_paged_runtime(
    *, physical_source: PhysicalInferenceOffloadSource,
    source_manifests: tuple[LinkedProgramManifest, ...],
    paged_manifests: tuple[LinkedProgramManifest, ...],
    fabric: ExternalMemoryFabric,
) -> dict[str, object]:
    """Sign all six per-core LSU gates and their exact external allocations."""
    if len(source_manifests) != 3 or len(paged_manifests) != 3:
        raise SchemaError("requires Prefill+2Decode manifests", path="manifests")
    fabric.validate()
    active = tuple(physical_source.external_manifest.placement.active_die_ids)
    if (len(active) != 6 or len(fabric.external_capacities) != 1 or
            fabric.external_capacities[0].id != physical_source.external_capacity.id or
            len(fabric.hbm_capacities) != 6 or
            any(item.capacity_bytes != HBM_CAPACITY for item in fabric.hbm_capacities) or
            len(fabric.links) != 1 or len(fabric.connections) != 6 or
            fabric.links[0].ingress_die_id != 0 or
            fabric.links[0].queue_depth != 6 or
            fabric.links[0].max_outstanding != 6):
        raise SchemaError("requires one external ingress and six 18KiB HBM homes",
                          path="fabric")
    rows = physical_source.external_manifest.request.mesh.rows
    columns = physical_source.external_manifest.request.mesh.columns
    conn = {item.target_die_id: item for item in fabric.connections}
    if set(conn) != set(active) or any(
        conn[die].route_die_ids != tp6_route(rows, columns, die) or
        conn[die].route_latency_cycles != len(conn[die].route_die_ids) - 1 or
        conn[die].route_bytes_per_cycle != (None if die == 0 else 256) or
        conn[die].link_ref != fabric.links[0].id or
        conn[die].hbm_capacity_ref != next(
            item.id for item in fabric.hbm_capacities
            if item.location_ref == f"die:{die}")
        for die in active
    ):
        raise SchemaError("external routes do not match physical TP6 mesh",
                          path="fabric.connections")
    requests = {item.id: item for item in
                physical_source.external_manifest.memory_plan.requests}
    allocations = {item.id: item for item in
                   physical_source.external_manifest.memory_plan.allocations}
    declaration_by_abi = {abi: item for item in physical_source.declarations
                          for abi in item.linked_state_abi_ids}
    if len(physical_source.declarations) != 114 or len(declaration_by_abi) != 162:
        raise SchemaError("physical declaration closure incomplete",
                          path="physical_source")
    source_abis = tuple(_state_abis(item) for item in source_manifests)
    paged_abis = tuple(_state_abis(item) for item in paged_manifests)
    core_by_die = {item.logical_core.die_id: item.runtime_core_id
                   for item in paged_manifests[0].core_streams}
    if set(core_by_die) != set(active):
        raise SchemaError("paged streams do not cover active TP6 Dies", path="manifests")
    seed = bytearray(physical_source.external_capacity.capacity_bytes)
    spans = []
    for declaration in physical_source.declarations:
        allocation = allocations[declaration.external_allocation_ref]
        request = requests[allocation.request_ref]
        if request.tier is not MemoryTier.EXTERNAL or request.size_bytes != declaration.physical_bytes:
            raise SchemaError("declaration does not own exact external allocation",
                              path="physical_source.declarations")
        latest = source_abis[2][declaration.linked_state_abi_ids[2]]
        die = declaration.die_id
        old_kv_base = min(item.address for item in source_abis[2].values()
                          if item.die_id == die and item.kind in
                          (StateKind.KV_KEY, StateKind.KV_VALUE))
        local = (WEIGHT_BASE if latest.kind is StateKind.PARAMETER
                 else KV_BASE + latest.address - old_kv_base)
        spans.append({
            "kind": declaration.role,
            "state_ref": latest.state_ref,
            "die_id": die,
            "runtime_core_id": core_by_die[die],
            "connection_ref": conn[die].id,
            "source_hbm_address": declaration.source_hbm_address,
            "external_address": allocation.address,
            "hbm_address": local,
            "size_bytes": declaration.physical_bytes,
        })
        if latest.kind is StateKind.PARAMETER:
            for index in range(declaration.physical_bytes):
                seed[allocation.address + index] = (allocation.address + index) % 251 + 1
    span_by_source = {(item["die_id"], item["source_hbm_address"]): item
                      for item in spans}
    events = []
    for segment, (source, paged) in enumerate(zip(source_manifests, paged_manifests)):
        old_records, _ = _record_maps(source)
        new_records, new_fragment_streams = _record_maps(paged)
        old_bindings, new_bindings = _bindings(source), _bindings(paged)
        for old_stream, new_stream in zip(source.core_streams, paged.core_streams):
            die = new_stream.logical_core.die_id
            if (old_stream.logical_core != new_stream.logical_core or
                    old_stream.runtime_core_id != new_stream.runtime_core_id or
                    len(old_stream.records) != len(new_stream.records)):
                raise SchemaError("relink changed a physical core stream", path="manifests")
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
                    raise SchemaError("relink changed useful record or binding", path="manifests")
                if new_key not in new_bindings:
                    if new_record.opcode in (RecordOpcode.LSU_LOAD, RecordOpcode.LSU_STORE):
                        raise SchemaError("unbound real LSU remains", path="manifests")
                    continue
                old_abi = source_abis[segment][old_bindings[old_key].state_abi_id]
                new_abi = paged_abis[segment][new_bindings[new_key].state_abi_id]
                if (old_abi.state_ref != new_abi.state_ref or
                        old_abi.size_bytes != new_abi.size_bytes or
                        old_abi.kind is not new_abi.kind or old_abi.die_id != die):
                    raise SchemaError("relink changed logical persistent state", path="manifests")
                span = span_by_source[(die, old_abi.address)]
                if old_abi.kind is StateKind.PARAMETER:
                    kind = "weight_restore_before_load"
                    if new_record.opcode is not RecordOpcode.LSU_LOAD:
                        raise SchemaError("parameter page is not an LSU Load", path="manifests")
                elif old_abi.kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
                    if new_record.opcode not in (RecordOpcode.LSU_LOAD,
                                                 RecordOpcode.LSU_STORE):
                        raise SchemaError("KV page is not a real LSU Load/Store",
                                          path="manifests")
                    kind = ("kv_restore_before_load" if new_record.opcode is RecordOpcode.LSU_LOAD
                            else "kv_writeback_after_store")
                else:
                    raise SchemaError("unexpected persistent state role", path="manifests")
                size = _lsu_size(new_record)
                addend = _hbm_addend(new_fragment_streams[new_key], new_key[3])
                events.append({
                    "event_index": len(events),
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
            expected = {"weight_restore_before_load": 15,
                        "kv_restore_before_load": 0 if segment == 0 else 4,
                        "kv_writeback_after_store": 4}
            if counts != expected:
                raise SchemaError("per-Die LSU page lifecycle incomplete",
                                  path=f"manifests[{segment}].die[{die}]")
    if len(spans) != 114 or len(events) != 390:
        raise SchemaError("TP6 sequence requires 114 spans and 390 DMA gates",
                          path="events")
    key: dict[str, object] = {
        "schema_version": "wafer_frontend.dense_inference_paged_runtime/v1alpha3",
        "producer_pass": "dense_inference_tp6_paged_runtime",
        "physical_source_digest": physical_source.digest,
        "external_materialization_digest": physical_source.external_materialization_digest,
        "linked_manifest_ids": [item.id for item in paged_manifests],
        "linked_manifest_digests": [canonical_digest(item) for item in paged_manifests],
        "mesh_rows": rows,
        "mesh_columns": columns,
        "active_die_ids": list(active),
        "hbm_capacity_bytes_per_die": HBM_CAPACITY,
        "weight_slot_base_bytes_per_die": WEIGHT_BASE,
        "parameter_state_bytes": 6 * PARAMETER_BYTES_PER_DIE,
        "kv_capacity_bytes": 6 * KV_BYTES_PER_DIE,
        "highest_paged_end_bytes_per_die": PAGED_END,
        "fabric": json.loads(canonical_json(fabric)),
        "spans": spans,
        "events": events,
        "external_seed_hex": bytes(seed).hex(),
    }
    digest = hashlib.sha256(json.dumps(
        key, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    key["id"] = "dense_inference_tp6_paged_runtime_" + digest[:20]
    return key


__all__ = ["build_dense_inference_tp6_paged_runtime",
           "relink_dense_inference_tp6_paged_segment", "tp6_route"]
