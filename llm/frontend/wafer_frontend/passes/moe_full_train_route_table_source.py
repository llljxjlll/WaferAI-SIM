"""Exact signed 20-byte-per-token native route-table payload for MoE TRAIN.

The payload is derived only from the P2 assignment trace already bound to the
full-model source.  This is an input artifact for a future ProgramIO loader;
creating it alone is not a native full-training execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

from ..errors import SchemaError
from ..schema.common import DType, stable_artifact_id
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.serde import canonical_digest
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase

_VERSION = "wafer_frontend.moe_full_train_route_table_source/v1"


@dataclass(frozen=True, slots=True)
class MoeFullTrainRouteTableSeed:
    step: int
    layer: int
    source_unit_ref: str
    source_trace_ref: str
    source_trace_digest: str
    route_value_ref: str
    route_rows: int
    route_bytes: int
    payload_sha256: str
    payload_hex: str

    @property
    def payload(self) -> bytes:
        return bytes.fromhex(self.payload_hex)


@dataclass(frozen=True, slots=True)
class MoeFullTrainRouteTableSource:
    source_ir0_ref: str
    source_sequence_ref: str
    seeds: tuple[MoeFullTrainRouteTableSeed, ...]
    id: str

    def validate_against(self, phase: FullMoeForwardIr0Phase,
                         sequence: MoeCompileSequence) -> None:
        if self != build_moe_full_train_route_table_source(phase, sequence):
            raise SchemaError("native route bytes differ from the full-model P2 frozen trace",
                              path="moe_full_train_route_table_source")


def build_moe_full_train_route_table_source(
    phase: FullMoeForwardIr0Phase, sequence: MoeCompileSequence,
) -> MoeFullTrainRouteTableSource:
    phase.validate()
    sequence.validate()
    if phase.source_moe_sequence_ref != sequence.id:
        raise SchemaError("route source sequence differs from full-model IR0",
                          path="moe_full_train_route_table_source.sequence")
    values = {value.id: value for value in phase.graph.values}
    traces = {(trace.step, trace.layer): trace for trace in
              sequence.materialization.logical_graph.route_traces}
    units = tuple(unit for unit in sequence.units if unit.step == phase.step)
    if len(units) != 2 or {unit.layer for unit in units} != {0, 1}:
        raise SchemaError("full-model route source requires two layer units",
                          path="moe_full_train_route_table_source.units")
    prefix = phase.graph.instances[0].id
    seeds = []
    for unit in sorted(units, key=lambda item: item.layer):
        trace = traces.get((unit.step, unit.layer))
        route_ref = f"{prefix}.layer{unit.layer}.moe.route_ids"
        value = values.get(route_ref)
        assignments = tuple(sorted(unit.spec.trace.assignments,
                                   key=lambda item: item.token_index))
        if (trace is None or value is None
                or unit.route_trace_ref != trace.id
                or unit.route_trace_digest != canonical_digest(trace)
                or value.shape != (trace.token_count, 5)
                or value.dtype is not DType.INT32
                or value.producer != f"{prefix}.layer{unit.layer}.moe.route_freeze"
                or len(assignments) != trace.token_count
                or tuple(item.token_index for item in assignments)
                   != tuple(range(trace.token_count))
                or tuple(item.expert_index for item in assignments)
                   != trace.expert_by_token
                or any(item.source_rank != 0
                       or item.expert_home_rank != item.expert_index
                       for item in assignments)):
            raise SchemaError("physical five-field route value differs from frozen P2 assignments",
                              path=f"moe_full_train_route_table_source.layer{unit.layer}")
        slots = {expert: [] for expert in range(len(trace.expert_token_counts))}
        for assignment in assignments:
            slots[assignment.expert_index].append(assignment.slot_index)
        if any(sorted(slots[expert]) != list(range(count))
               for expert, count in enumerate(trace.expert_token_counts)):
            raise SchemaError("signed expert slots must be a complete rank-local permutation",
                              path=f"moe_full_train_route_table_source.layer{unit.layer}")
        payload = b"".join(struct.pack("<IIIII", item.token_index,
                                       item.source_rank, item.expert_index,
                                       item.expert_home_rank, item.slot_index)
                           for item in assignments)
        if len(payload) != 20 * trace.token_count or not any(payload):
            raise SchemaError("native route payload is empty or truncated",
                              path=f"moe_full_train_route_table_source.layer{unit.layer}")
        seeds.append(MoeFullTrainRouteTableSeed(
            unit.step, unit.layer, unit.id, trace.id, canonical_digest(trace),
            route_ref, trace.token_count, len(payload),
            hashlib.sha256(payload).hexdigest(), payload.hex(),
        ))
    semantic = dict(source_ir0_ref=phase.graph.id,
                    source_sequence_ref=sequence.id, seeds=tuple(seeds))
    return MoeFullTrainRouteTableSource(**semantic, id=stable_artifact_id(
        "moe_full_train_route_table_source", semantic, schema_version=_VERSION))


__all__ = ["MoeFullTrainRouteTableSeed", "MoeFullTrainRouteTableSource",
           "build_moe_full_train_route_table_source"]
