"""Source-signed 19-parameter MoE SGD external DMA gate schedule.

This emits a runtime contract; only a native NpuSim pager may claim execution.
"""

from __future__ import annotations

import hashlib
import json

from ..errors import SchemaError
from ..schema.artifact_manifest import (
    LinkedProgramManifest, RecordOpcode, StateKind,
)
from ..schema.external_memory import ExternalMemoryFabric
from ..schema.memory_plan import MemoryObjectKind, MemoryTier
from ..schema.program_io import ProgramHbmTarget, ProgramIoContract
from ..schema.serde import canonical_digest, canonical_json
from ..schema.workload_materialization import WorkloadMaterializationManifest
from ..schema.workload_run import WorkloadFamily, WorkloadMemoryMode
from .moe_full_train_paged_compile_sequence import (
    HBM_CAPACITY_BYTES, P3_WORKSPACE_END_BYTES, ROUTE_HOME_ADDRESSES,
    STATE_SLOT_ADDRESS, STATE_SLOT_BYTES,
    relink_moe_full_train_paged_step,
)


def _states(manifest: LinkedProgramManifest):
    states = {}
    for fragment in manifest.fragments:
        for state in fragment.state_abi:
            prior = states.setdefault(state.id, state)
            if prior != state:
                raise SchemaError("conflicting physical StateABI", path="manifest")
    return states


def _record(fragment, core, index):
    stream = next((item for item in fragment.core_streams
                   if item.logical_core == core), None)
    if stream is None or index >= len(stream.records):
        raise SchemaError("linked record lacks true source fragment", path="record")
    return stream.records[index], stream


def _lsu_size(record):
    sizes = tuple(item.literal_value for item in record.operands
                  if item.name == "size_bytes")
    if len(sizes) != 1 or type(sizes[0]) is not int or sizes[0] <= 0:
        raise SchemaError("actual LSU size literal absent", path="record")
    return sizes[0]


def _hbm_addend(stream, index):
    from ..schema.artifact_manifest import SemanticOperandId
    addends = tuple(item.addend for item in stream.address_relocations
                    if item.record_index == index and
                    item.operand_id is SemanticOperandId.HBM_ADDRESS)
    if len(addends) != 1:
        raise SchemaError("actual LSU HBM relocation absent", path="record")
    return addends[0]


def _external_groups(offload: WorkloadMaterializationManifest):
    requests = {item.id: item for item in offload.memory_plan.requests}
    versions = {item.id: item for item in offload.memory_plan.state_versions}
    inventory = {item.id: item for item in offload.state_inventory}
    groups = {}
    hbm_end = 0
    for allocation in offload.memory_plan.allocations:
        request = requests[allocation.request_ref]
        if request.tier is MemoryTier.HBM:
            if request.location_ref != "die:0":
                raise SchemaError("unexpected MoE train HBM home", path="memory_plan")
            hbm_end = max(hbm_end, allocation.address + request.size_bytes)
            continue
        state = inventory[versions[request.state_version_ref].state_ref]
        if (request.tier is not MemoryTier.EXTERNAL or
                request.object_kind is not MemoryObjectKind.PARAMETER or
                state.logical_name not in (
                    "parameter.shared.tp0.rank0",
                    "parameter.expert.0.tp0.rank0",
                )):
            raise SchemaError("external allocation is not a true MoE parameter group",
                              path="memory_plan")
        groups[state.logical_name] = (allocation.address, request.size_bytes,
                                      allocation.id)
    if (hbm_end != P3_WORKSPACE_END_BYTES or groups.keys() != {
        "parameter.shared.tp0.rank0", "parameter.expert.0.tp0.rank0"
    } or groups["parameter.shared.tp0.rank0"][:2] != (0, 568)
            or groups["parameter.expert.0.tp0.rank0"][:2] != (576, 384)):
        raise SchemaError("actual P3 MoE HBM/external allocation changed",
                          path="memory_plan")
    return groups


def build_moe_full_train_paged_runtime(
    offload: WorkloadMaterializationManifest,
    fabric: ExternalMemoryFabric,
    source_manifests: tuple[LinkedProgramManifest, LinkedProgramManifest],
    paged_manifests: tuple[LinkedProgramManifest, LinkedProgramManifest],
    source_program_io: tuple[ProgramIoContract, ProgramIoContract],
) -> dict[str, object]:
    """Bind real source LSU records and ProgramIO seeds to P3 external homes."""
    offload.validate("offload")
    fabric.validate("fabric")
    if (offload.request.family is not WorkloadFamily.MOE_TRAINING or
            offload.request.memory.mode is not WorkloadMemoryMode.EXTERNAL_OFFLOAD or
            (offload.request.mesh.rows, offload.request.mesh.columns) != (1, 1) or
            offload.request.parallel.ep != 1 or
            len(source_manifests) != 2 or len(paged_manifests) != 2 or
            len(source_program_io) != 2 or
            len(fabric.external_capacities) != 1 or
            len(fabric.hbm_capacities) != 1 or
            fabric.hbm_capacities[0].capacity_bytes != HBM_CAPACITY_BYTES or
            fabric.external_capacities[0].capacity_bytes != 2048 or
            len(fabric.links) != 1 or len(fabric.connections) != 1 or
            fabric.links[0].external_capacity_ref != fabric.external_capacities[0].id or
            fabric.links[0].ingress_die_id != 0 or
            (fabric.links[0].bytes_per_cycle, fabric.links[0].latency_cycles,
             fabric.links[0].queue_depth, fabric.links[0].max_outstanding)
            != (256, 2, 2, 2) or
            fabric.connections[0].link_ref != fabric.links[0].id or
            fabric.connections[0].hbm_capacity_ref != fabric.hbm_capacities[0].id or
            fabric.connections[0].target_die_id != 0 or
            fabric.connections[0].route_die_ids != (0,) or
            fabric.connections[0].route_latency_cycles != 0):
        raise SchemaError("requires fixed EP1 two-step 2560B HBM external fabric",
                          path="offload")
    groups = _external_groups(offload)
    last_source_states = _states(source_manifests[0])
    first_paged_states = {item.state_ref: item for item in
                          _states(paged_manifests[0]).values()}
    weights = sorted((item for item in last_source_states.values()
                      if item.kind is StateKind.TRAINABLE_PARAMETER),
                     key=lambda item: item.address)
    routes = sorted((item for item in last_source_states.values()
                     if item.kind is StateKind.MOE_STATIC_ROUTE),
                    key=lambda item: item.address)
    if len(weights) != 19 or len(routes) != 2:
        raise SchemaError("physical MoE training state inventory changed",
                          path="source_manifests")
    addresses = {}
    spans = []
    for group, selected in (
        ("parameter.shared.tp0.rank0",
         [item for item in weights if ".expert0." not in item.layout]),
        ("parameter.expert.0.tp0.rank0",
         [item for item in weights if ".expert0." in item.layout]),
    ):
        base, size, allocation = groups[group]
        cursor = base
        for item in selected:
            addresses[item.state_ref] = cursor
            spans.append({
                "state_ref": item.state_ref,
                "state_abi_id": first_paged_states[item.state_ref].id,
                "source_hbm_address": item.address,
                "external_address": cursor, "hbm_address": STATE_SLOT_ADDRESS,
                "size_bytes": item.size_bytes, "group": group,
                "source_allocation_ref": allocation,
            })
            cursor += item.size_bytes
        if cursor != base + size:
            raise SchemaError("P3 parameter group differs from 19 actual ABIs",
                              path=group)
    if len(addresses) != 19:
        raise SchemaError("physical trainable state repeated", path="spans")
    blob_by_id = {item.id: item for item in source_program_io[0].blobs}
    seeds = []
    initial = {}
    for item in source_program_io[0].initializations:
        if type(item.target) is ProgramHbmTarget:
            initial[item.target.state_ref] = blob_by_id[item.blob_ref].payload()
    if set(initial) != {item.state_ref for item in (*weights, *routes)}:
        raise SchemaError("source ProgramIO lacks all 21 exact HBM seeds",
                          path="source_program_io")
    for item in spans:
        payload = initial[item["state_ref"]]
        if len(payload) != item["size_bytes"] or not any(payload):
            raise SchemaError("offloaded parameter has invalid source payload",
                              path=item["state_ref"])
        seeds.append({"external_address": item["external_address"],
                      "payload_hex": payload.hex()})
    events = []
    for step, (source, paged, original_io) in enumerate(zip(
        source_manifests, paged_manifests, source_program_io
    )):
        source.validate("source_manifest")
        paged.validate("paged_manifest")
        original_io.validate_against(source)
        if relink_moe_full_train_paged_step(source, step) != paged:
            raise SchemaError("paged manifest changed from source relinker",
                              path=f"paged_manifests[{step}]")
        source_states = _states(source)
        paged_states = _states(paged)
        if ({item.state_ref: item.size_bytes for item in source_states.values()
             if item.kind is StateKind.TRAINABLE_PARAMETER} !=
                {item.state_ref: item.size_bytes for item in last_source_states.values()
                 if item.kind is StateKind.TRAINABLE_PARAMETER} or
                sorted((item.layout, item.size_bytes) for item in source_states.values()
                       if item.kind is StateKind.MOE_STATIC_ROUTE) !=
                sorted((item.layout, item.size_bytes) for item in last_source_states.values()
                       if item.kind is StateKind.MOE_STATIC_ROUTE)):
            raise SchemaError("two-step trainable state or route layout drifted",
                              path="source_manifests")
        original_fragments = {item.id: item for item in source.fragments}
        paged_fragments = {item.id: item for item in paged.fragments}
        old_bindings = {(item.fragment_id, item.fragment_record_index): item
                        for item in source.state_operand_bindings}
        new_bindings = {(item.fragment_id, item.fragment_record_index): item
                        for item in paged.state_operand_bindings}
        old_core, new_core = source.core_streams[0], paged.core_streams[0]
        if (old_core.logical_core != new_core.logical_core or
                old_core.runtime_core_id != new_core.runtime_core_id or
                len(old_core.records) != len(new_core.records)):
            raise SchemaError("relink altered MoE training core stream",
                              path="core_streams")
        counts = {"restore_before_load": 0, "writeback_after_store": 0,
                  "resident_route_load": 0}
        for index, (old_ref, new_ref) in enumerate(zip(old_core.records,
                                                       new_core.records)):
            old_record, _ = _record(original_fragments[old_ref.fragment_id],
                                    old_core.logical_core,
                                    old_ref.fragment_record_index)
            new_record, new_stream = _record(
                paged_fragments[new_ref.fragment_id], new_core.logical_core,
                new_ref.fragment_record_index)
            old_key = (old_ref.fragment_id, old_ref.fragment_record_index)
            new_key = (new_ref.fragment_id, new_ref.fragment_record_index)
            if (old_ref.source_global_action_id != new_ref.source_global_action_id or
                    old_record != new_record or
                    (old_key in old_bindings) != (new_key in new_bindings)):
                raise SchemaError("paged graph/record/state binding changed",
                                  path="core_streams")
            if old_key not in old_bindings:
                continue
            old_state = source_states[old_bindings[old_key].state_abi_id]
            new_state = paged_states[new_bindings[new_key].state_abi_id]
            if (old_state.state_ref != new_state.state_ref or
                    old_state.kind is not new_state.kind or
                    old_state.size_bytes != new_state.size_bytes or
                    _lsu_size(new_record) != old_state.size_bytes or
                    _hbm_addend(new_stream, new_ref.fragment_record_index) != 0):
                raise SchemaError("physical state LSU payload changed",
                                  path="core_streams")
            if old_state.kind is StateKind.MOE_STATIC_ROUTE:
                if (new_record.opcode is not RecordOpcode.LSU_LOAD or
                        new_state.address not in ROUTE_HOME_ADDRESSES):
                    raise SchemaError("pinned route LSU changed", path="routes")
                counts["resident_route_load"] += 1
                continue
            if (old_state.kind is not StateKind.TRAINABLE_PARAMETER or
                    new_state.address != STATE_SLOT_ADDRESS):
                raise SchemaError("trainable LSU lacks bounded page", path="weights")
            kind = ("restore_before_load" if new_record.opcode is RecordOpcode.LSU_LOAD
                    else "writeback_after_store" if new_record.opcode is RecordOpcode.LSU_STORE
                    else None)
            if kind is None:
                raise SchemaError("trainable state uses unsupported LSU", path="weights")
            events.append({
                "step_index": step, "runtime_core_id": new_core.runtime_core_id,
                "linked_record_index": index, "fragment_id": new_ref.fragment_id,
                "fragment_record_index": new_ref.fragment_record_index,
                "kind": kind, "state_ref": new_state.state_ref,
                "state_abi_id": new_state.id,
                "source_hbm_address": old_state.address,
                "external_address": addresses[new_state.state_ref],
                "hbm_address": new_state.address,
                "size_bytes": new_state.size_bytes,
            })
            counts[kind] += 1
        if counts != {"restore_before_load": 46,
                      "writeback_after_store": 19,
                      "resident_route_load": 2}:
            raise SchemaError("per-step 46/19 DMA gates or two routes missing",
                              path=f"events[{step}]")
    read_bytes = sum(item["size_bytes"] for item in events
                     if item["kind"] == "restore_before_load")
    write_bytes = sum(item["size_bytes"] for item in events
                      if item["kind"] == "writeback_after_store")
    if len(events) != 130 or (read_bytes, write_bytes) != (4864, 1904):
        raise SchemaError("MoE training external traffic oracle changed",
                          path="events")
    key = {
        "schema_version": "wafer_frontend.moe_full_train_paged_runtime/v1alpha1",
        "producer_pass": "moe_full_train_paged_runtime",
        "request_digest": offload.request_digest,
        "logical_graph_digest": offload.logical_graph_digest,
        "offload_memory_plan_digest": canonical_digest(offload.memory_plan),
        "source_external_allocation_refs": [groups[name][2] for name in (
            "parameter.shared.tp0.rank0", "parameter.expert.0.tp0.rank0")],
        "linked_manifest_ids": [item.id for item in paged_manifests],
        "linked_manifest_digests": [canonical_digest(item)
                                    for item in paged_manifests],
        "hbm_capacity_bytes": HBM_CAPACITY_BYTES,
        "workspace_end_bytes": P3_WORKSPACE_END_BYTES,
        "weight_slot_address": STATE_SLOT_ADDRESS,
        "weight_slot_bytes": STATE_SLOT_BYTES,
        "pinned_route_addresses": list(ROUTE_HOME_ADDRESSES),
        "external_capacity_bytes": 2048,
        "expected_events": 130,
        "expected_external_read_bytes": read_bytes,
        "expected_external_write_bytes": write_bytes,
        "fabric": json.loads(canonical_json(fabric)),
        "parameter_spans": spans,
        "events": events,
        "seeds": seeds,
    }
    key["id"] = "moe_full_train_paged_runtime_" + hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return key


__all__ = ["build_moe_full_train_paged_runtime"]
