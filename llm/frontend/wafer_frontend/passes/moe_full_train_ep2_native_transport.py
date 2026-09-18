"""Bind EP2 source transfers to actual signed P2 DTE program records.

This proves component records exist; the full-model timeline linker must still
integrate them with shared forward/backward and parameter transport.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import RecordOpcode
from ..schema.ir0 import OpKind
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.n6 import _leaf_fragments
from .moe_full_train_ep2_rank_plan import MoeEp2RankPlan
from .moe_full_train_ep_ir1_source import MoeEpSharedReverseIr1Candidate


@dataclass(frozen=True, slots=True)
class MoeEp2NativeRecordRef:
    fragment_ref: str
    core_die: int
    local_core_id: int
    record_index: int
    opcode: RecordOpcode
    action_ref: str


@dataclass(frozen=True, slots=True)
class MoeEp2NativeTransferBinding:
    step: int
    layer: int
    value_ref: str
    source_flow_ref: str
    bytes: int
    send: MoeEp2NativeRecordRef
    recv: MoeEp2NativeRecordRef
    wait: MoeEp2NativeRecordRef


@dataclass(frozen=True, slots=True)
class MoeEp2NativeTransportProof:
    source_ir1_id: str
    source_sequence_id: str
    bindings: tuple[MoeEp2NativeTransferBinding, ...]

    def validate_against(
        self, plan: MoeEp2RankPlan,
        source: MoeEpSharedReverseIr1Candidate,
        sequence: MoeCompileSequence,
    ) -> None:
        if self != _derive(plan, source, sequence):
            raise SchemaError("EP2 native DTE records differ from signed source flow",
                              path="moe_ep2_native_transport")


def _one_record(manifest, action_ref: str, die_id: int,
                opcode: RecordOpcode, expected_bytes: int | None,
                value_ref: str) -> MoeEp2NativeRecordRef:
    matches = tuple(
        (fragment, stream, index, record)
        for fragment in _leaf_fragments(manifest.fragments)
        for stream in fragment.core_streams
        if stream.logical_core.die_id == die_id
        for index, record in enumerate(stream.records)
        if record.source_global_action_id == action_ref
        and record.opcode is opcode
    )
    if len(matches) != 1:
        raise SchemaError("signed EP2 action has no unique native DTE record",
                          path=f"source.values.{value_ref}")
    fragment, stream, index, record = matches[0]
    if expected_bytes is not None:
        lengths = tuple(operand.literal_value for operand in record.operands
                        if operand.name == "length_bytes")
        if lengths != (expected_bytes,):
            raise SchemaError("native DTE byte length differs from source tensor",
                              path=f"source.values.{value_ref}")
    return MoeEp2NativeRecordRef(fragment.id, die_id,
                                 stream.logical_core.local_core_id, index,
                                 opcode, action_ref)


def _derive(
    plan: MoeEp2RankPlan,
    source: MoeEpSharedReverseIr1Candidate,
    sequence: MoeCompileSequence,
) -> MoeEp2NativeTransportProof:
    plan.validate_against(source, sequence)
    graph = source.source_ir0
    steps = {node.workload.step for node in graph.nodes
             if node.kind is OpKind.MOE_COMBINE}
    if len(steps) != 1:
        raise SchemaError("EP2 native transport needs one source step",
                          path="source")
    step = next(iter(steps))
    instance_id = graph.instances[0].id
    units = {(unit.step, unit.layer): unit for unit in sequence.units}
    bindings = []
    for transfer in plan.transfers:
        if transfer.source_flow_ref is None:
            continue  # Router weight needs a separate owner1→router state transport.
        layers = tuple(layer for layer in (0, 1)
                       if transfer.value_ref.startswith(
                           f"{instance_id}.layer{layer}.moe."))
        if len(layers) != 1:
            raise SchemaError("EP2 transfer has no exact source layer",
                              path=transfer.value_ref)
        layer = layers[0]
        manifest = units[step, layer].linked_manifest
        send = _one_record(manifest, transfer.source_send_action_ref,
                           transfer.source_die, RecordOpcode.DTE_SEND,
                           transfer.bytes, transfer.value_ref)
        recv = _one_record(manifest, transfer.destination_recv_action_ref,
                           transfer.destination_die, RecordOpcode.DTE_RECV,
                           transfer.bytes, transfer.value_ref)
        wait = _one_record(manifest, transfer.destination_wait_action_ref,
                           transfer.destination_die, RecordOpcode.DTE_WAIT,
                           None, transfer.value_ref)
        bindings.append(MoeEp2NativeTransferBinding(
            step, layer, transfer.value_ref, transfer.source_flow_ref,
            transfer.bytes, send, recv, wait,
        ))
    if len(bindings) != 4:
        raise SchemaError("two-layer EP2 needs four native dispatch/return flows",
                          path="moe_ep2_native_transport")
    return MoeEp2NativeTransportProof(plan.source_ir1_id,
                                      sequence.id, tuple(bindings))


def build_moe_ep2_native_transport_proof(
    plan: MoeEp2RankPlan,
    source: MoeEpSharedReverseIr1Candidate,
    sequence: MoeCompileSequence,
) -> MoeEp2NativeTransportProof:
    result = _derive(plan, source, sequence)
    result.validate_against(plan, source, sequence)
    return result


__all__ = ["MoeEp2NativeRecordRef", "MoeEp2NativeTransferBinding",
           "MoeEp2NativeTransportProof",
           "build_moe_ep2_native_transport_proof"]
