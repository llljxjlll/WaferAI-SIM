"""Audit P2 DTE records available for the EP2 dExpert1 source handoff.

The records are executable P2 components. Their SEND still has no signed
full-model dExpert producer, so this candidate is not an executed reverse.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from ..schema.artifact_manifest import RecordOpcode
from ..schema.flexible_moe import MoeRectActionKind
from ..schema.ir0 import OpKind
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_ep2_native_transport import (
    MoeEp2NativeRecordRef, _one_record,
)
from .moe_full_train_ep2_reverse_rank_plan import MoeEp2ReverseRankPlan
from .moe_full_train_ep_expert_reverse_ir1_source import (
    MoeEp2ExpertReverseIr1Candidate,
)


@dataclass(frozen=True, slots=True)
class MoeEp2BackwardNativeCandidate:
    source_ir1_id: str
    source_sequence_id: str
    step: int
    layer: int
    value_ref: str
    source_producer_ref: str
    p2_flow_ref: str
    bytes: int
    send: MoeEp2NativeRecordRef
    recv: MoeEp2NativeRecordRef
    wait: MoeEp2NativeRecordRef

    def validate_against(
        self, plan: MoeEp2ReverseRankPlan,
        source: MoeEp2ExpertReverseIr1Candidate,
        sequence: MoeCompileSequence,
    ) -> None:
        if self != _derive(plan, source, sequence):
            raise SchemaError("EP2 backward native candidate differs from source/P2",
                              path="moe_ep2_backward_native_candidate")


def _derive(
    plan: MoeEp2ReverseRankPlan,
    source: MoeEp2ExpertReverseIr1Candidate,
    sequence: MoeCompileSequence,
) -> MoeEp2BackwardNativeCandidate:
    plan.validate_against(source, sequence)
    graph = source.source_ir0
    instance_id = graph.instances[0].id
    value_ref = f"backward::{instance_id}.layer1.moe.combine.dexpert1"
    matching = tuple(item for item in plan.remote_payloads
                     if item.value_ref == value_ref)
    if len(matching) != 1:
        raise SchemaError("missing unique remote dExpert1 payload",
                          path="plan.remote_payloads")
    payload = matching[0]
    if (payload.existing_flow_ref is not None
            or payload.source_state_ref is not None
            or payload.source_rank != 0
            or payload.destination_rank != 1
            or not all((payload.candidate_backward_flow_ref,
                        payload.candidate_backward_send_action_ref,
                        payload.candidate_backward_recv_action_ref,
                        payload.candidate_backward_wait_action_ref))):
        raise SchemaError("dExpert1 candidate must remain unbound to source",
                          path="plan.remote_payloads")
    steps = {node.workload.step for node in graph.nodes
             if node.kind is OpKind.MOE_COMBINE}
    if len(steps) != 1:
        raise SchemaError("candidate requires one source step", path="source")
    step = next(iter(steps))
    unit = next((unit for unit in sequence.units
                 if (unit.step, unit.layer) == (step, 1)), None)
    if unit is None:
        raise SchemaError("source P2 layer unit absent", path="sequence")
    actions = {action.id: action for action in unit.plan.actions}
    send_action = actions[payload.candidate_backward_send_action_ref]
    # Old P2 has physical DTE transport, but no early source-bound score/
    # dExpert producer. A signed recompile and full timeline must replace it.
    if any(actions[ref].kind is
           MoeRectActionKind.SCORE_WEIGHT_BACKWARD_PRE_DISPATCH
           for ref in send_action.deps):
        raise SchemaError("candidate cannot stand in for a source-bound SEND",
                          path="sequence.units")
    manifest = unit.linked_manifest
    send = _one_record(manifest, send_action.id, payload.source_die,
                       RecordOpcode.DTE_SEND, payload.bytes, value_ref)
    recv = _one_record(manifest,
                       payload.candidate_backward_recv_action_ref,
                       payload.destination_die, RecordOpcode.DTE_RECV,
                       payload.bytes, value_ref)
    wait = _one_record(manifest,
                       payload.candidate_backward_wait_action_ref,
                       payload.destination_die, RecordOpcode.DTE_WAIT,
                       None, value_ref)
    return MoeEp2BackwardNativeCandidate(
        plan.source_ir1_id, sequence.id, step, 1, value_ref,
        payload.producer_node_ref, payload.candidate_backward_flow_ref,
        payload.bytes, send, recv, wait,
    )


def build_moe_ep2_backward_native_candidate(
    plan: MoeEp2ReverseRankPlan,
    source: MoeEp2ExpertReverseIr1Candidate,
    sequence: MoeCompileSequence,
) -> MoeEp2BackwardNativeCandidate:
    result = _derive(plan, source, sequence)
    result.validate_against(plan, source, sequence)
    return result


__all__ = ["MoeEp2BackwardNativeCandidate",
           "build_moe_ep2_backward_native_candidate"]
