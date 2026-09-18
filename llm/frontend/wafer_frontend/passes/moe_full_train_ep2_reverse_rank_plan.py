"""EP2 expert reverse ranks and unique physical remote payloads.

The existing P2 dispatch and return are retained for their backward readers.
The P2 backward-gradient flow exists, but is only a physical candidate until
its SEND reads the real dExpert1 producer on a complete model timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.ir0 import OpKind, StateAccessMode
from ..schema.flexible_moe import MoeRectActionKind, MoeRectFlowStage
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_ep2_rank_plan import (
    MoeEp2NodeRank, MoeEp2RankPlan, build_moe_ep2_rank_plan,
)
from .moe_full_train_ep_expert_reverse_ir1_source import (
    MoeEp2ExpertReverseIr1Candidate,
)

_BYTES = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}


@dataclass(frozen=True, slots=True)
class MoeEp2RemotePayload:
    value_ref: str
    producer_node_ref: str | None
    source_state_ref: str | None
    consumer_node_refs: tuple[str, ...]
    source_rank: int
    destination_rank: int
    source_die: int
    destination_die: int
    route_ref: str
    die_path: tuple[int, ...]
    bytes: int
    existing_flow_ref: str | None
    existing_send_action_ref: str | None
    existing_recv_action_ref: str | None
    existing_wait_action_ref: str | None
    candidate_backward_flow_ref: str | None
    candidate_backward_send_action_ref: str | None
    candidate_backward_recv_action_ref: str | None
    candidate_backward_wait_action_ref: str | None

    @property
    def lacks_source_bound_transport(self) -> bool:
        return self.existing_flow_ref is None


@dataclass(frozen=True, slots=True)
class MoeEp2ReverseRankPlan:
    source_ir1_id: str
    base: MoeEp2RankPlan
    node_ranks: tuple[MoeEp2NodeRank, ...]
    remote_payloads: tuple[MoeEp2RemotePayload, ...]

    def validate_against(
        self, source: MoeEp2ExpertReverseIr1Candidate,
        sequence: MoeCompileSequence,
    ) -> None:
        if self != _derive(source, sequence):
            raise SchemaError("EP2 reverse ranks/payloads differ from source and signed P2",
                              path="moe_ep2_reverse_rank_plan")


def _derive(source: MoeEp2ExpertReverseIr1Candidate,
            sequence: MoeCompileSequence) -> MoeEp2ReverseRankPlan:
    if type(source) is not MoeEp2ExpertReverseIr1Candidate:
        raise SchemaError("requires physical EP2 expert reverse source", path="source")
    base = build_moe_ep2_rank_plan(source.shared, sequence)
    graph = source.source_ir0
    ir1 = source.physical_ir1
    graph.validate("moe_ep2_reverse_rank_plan.source_ir0")
    ir1.validate("moe_ep2_reverse_rank_plan.source_ir1")
    if (graph.producer_pass != "moe_full_train_expert_backward_ir0"
            or ir1.source_ir0_id != graph.id
            or ir1.groups != source.shared.physical_ir1.groups
            or len(graph.nodes) != 41
            or tuple(node.id for node in graph.nodes[:38]) !=
               tuple(item.node_ref for item in base.node_ranks)):
        raise SchemaError("requires exact expanded EP2 expert reverse source",
                          path="source")
    group = ir1.groups[0]
    die_by_rank = {place.rank: place.die_id for place in group.placements}
    route_by_pair = {(route.source_rank, route.destination_rank): route
                     for route in group.embedding.routes}
    instance_id = graph.instances[0].id
    expected_added = (
        (f"backward::{instance_id}.layer1.moe.combine",
         OpKind.MOE_COMBINE_BACKWARD, 0),
        (f"backward::{instance_id}.layer1.moe.expert0",
         OpKind.MOE_EXPERT_BACKWARD, 0),
        (f"backward::{instance_id}.layer1.moe.expert1",
         OpKind.MOE_EXPERT_BACKWARD, 1),
    )
    if tuple((node.id, node.kind,
              node.workload.expert if node.kind is OpKind.MOE_EXPERT_BACKWARD
              else 0) for node in graph.nodes[38:]) != expected_added:
        raise SchemaError("EP2 added reverse nodes differ from owner ranks",
                          path="source.nodes")
    node_ranks = (*base.node_ranks,
                  *(MoeEp2NodeRank(ref, rank, die_by_rank[rank])
                    for ref, _, rank in expected_added))
    rank_by_node = {item.node_ref: item.rank for item in node_ranks}
    states = {state.id: state for state in graph.persistent_states}
    bindings = {binding.state_ref: binding
                for binding in ir1.persistent_state_manifest.bindings}
    base_by_key = {(item.value_ref, item.destination_rank): item
                   for item in base.transfers}
    if len(base_by_key) != len(base.transfers):
        raise SchemaError("base EP2 payloads must be unique per destination",
                          path="base.transfers")
    grouped: dict[tuple[str, int], list[str]] = {}
    origins: dict[tuple[str, int], tuple[int, str | None]] = {}
    for value in graph.values:
        for consumer_ref in value.consumers:
            destination_rank = rank_by_node[consumer_ref]
            state_ref = None
            if value.producer is not None:
                source_rank = rank_by_node[value.producer]
            else:
                accesses = tuple(access for access in graph.state_accesses
                                 if access.node_ref == consumer_ref
                                 and states[access.state_ref].identity.tensor_ref
                                    == value.id)
                if not accesses:
                    continue
                if (len(accesses) != 1
                        or accesses[0].mode is not StateAccessMode.READ):
                    raise SchemaError("remote parameter lacks one owner read",
                                      path=f"source.values.{value.id}")
                state_ref = accesses[0].state_ref
                source_rank = accesses[0].rank
                if bindings[state_ref].die_id != die_by_rank[source_rank]:
                    raise SchemaError("remote parameter home differs from owner",
                                      path=f"source.values.{value.id}")
            if source_rank == destination_rank:
                continue
            key = (value.id, destination_rank)
            origin = (source_rank, state_ref)
            if key in origins and origins[key] != origin:
                raise SchemaError("remote payload has inconsistent owner",
                                  path=f"source.values.{value.id}")
            origins[key] = origin
            grouped.setdefault(key, []).append(consumer_ref)
    dexpert_ref = f"backward::{instance_id}.layer1.moe.combine.dexpert1"
    expected_keys = set(base_by_key) | {(dexpert_ref, 1)}
    if set(grouped) != expected_keys or len(grouped) != 7:
        raise SchemaError("EP2 reverse must have seven unique remote payloads",
                          path="source.values")
    values = {value.id: value for value in graph.values}
    payloads = []
    for key in sorted(grouped):
        value_ref, destination_rank = key
        value = values[value_ref]
        source_rank, state_ref = origins[key]
        route = route_by_pair.get((source_rank, destination_rank))
        if route is None or value.dtype not in _BYTES:
            raise SchemaError("remote reverse payload lacks route or dtype",
                              path=f"source.values.{value_ref}")
        bytes_ = prod(value.shape) * _BYTES[value.dtype]
        consumers = tuple(sorted(grouped[key]))
        base_item = base_by_key.get(key)
        if base_item is not None:
            if (base_item.consumer_node_ref not in consumers
                    or (base_item.source_node_ref, base_item.source_state_ref,
                        base_item.source_rank, base_item.source_die,
                        base_item.destination_die, base_item.route_ref,
                        base_item.die_path, base_item.bytes)
                       != (value.producer, state_ref, source_rank,
                           die_by_rank[source_rank],
                           die_by_rank[destination_rank], route.id,
                           route.die_path, bytes_)):
                raise SchemaError("retained P2 payload differs from source",
                                  path=f"source.values.{value_ref}")
            refs = (base_item.source_flow_ref,
                    base_item.source_send_action_ref,
                    base_item.destination_recv_action_ref,
                    base_item.destination_wait_action_ref)
            candidate_refs = (None, None, None, None)
        else:
            if (key != (dexpert_ref, 1)
                    or value.producer != expected_added[0][0]
                    or state_ref is not None
                    or source_rank != 0
                    or consumers != (expected_added[2][0],)):
                raise SchemaError("new EP2 remote use must be exact dExpert1",
                                  path=f"source.values.{value_ref}")
            step = graph.nodes[38].workload.step
            unit = next((unit for unit in sequence.units
                         if (unit.step, unit.layer) == (step, 1)), None)
            if unit is None:
                raise SchemaError("dExpert1 lacks source P2 layer unit",
                                  path=f"source.values.{value_ref}")
            forward = next(node for node in graph.nodes
                           if node.id == f"{instance_id}.layer1.moe.expert1")
            token_refs = tuple(f"assignment.{index}" for index, expert
                               in enumerate(forward.workload.frozen_expert_by_token)
                               if expert == 1)
            flows = tuple(flow for flow in unit.plan.flows
                          if flow.stage is MoeRectFlowStage.BACKWARD_GRADIENT
                          and flow.source_rank == source_rank
                          and flow.destination_rank == destination_rank
                          and flow.logical_bytes == bytes_
                          and flow.assignment_refs == token_refs)
            if len(flows) != 1:
                raise SchemaError("dExpert1 lacks exact existing P2 backward flow",
                                  path=f"source.values.{value_ref}")
            flow = flows[0]
            def one_action(kind, rank):
                found = tuple(action for action in unit.plan.actions
                              if action.kind is kind and action.rank == rank
                              and action.flow_ref == flow.id)
                if len(found) != 1:
                    raise SchemaError("P2 backward flow lacks exact action",
                                      path=f"source.values.{value_ref}")
                return found[0]
            send = one_action(MoeRectActionKind.SEND, source_rank)
            recv = one_action(MoeRectActionKind.RECV, destination_rank)
            wait = one_action(MoeRectActionKind.WAIT, destination_rank)
            if not {send.id, recv.id}.issubset(wait.deps):
                raise SchemaError("P2 backward WAIT lacks SEND/RECV dependency",
                                  path=f"source.values.{value_ref}")
            # The baseline P2 SEND is not fed by this new full-model dExpert.
            # Keep the physical candidate distinct from a source-bound flow.
            refs = (None, None, None, None)
            candidate_refs = (flow.id, send.id, recv.id, wait.id)
        payloads.append(MoeEp2RemotePayload(
            value_ref, value.producer, state_ref, consumers,
            source_rank, destination_rank,
            die_by_rank[source_rank], die_by_rank[destination_rank],
            route.id, route.die_path, bytes_, *refs, *candidate_refs,
        ))
    result = MoeEp2ReverseRankPlan(ir1.id, base, tuple(node_ranks),
                                    tuple(payloads))
    return result


def build_moe_ep2_reverse_rank_plan(
    source: MoeEp2ExpertReverseIr1Candidate,
    sequence: MoeCompileSequence,
) -> MoeEp2ReverseRankPlan:
    result = _derive(source, sequence)
    result.validate_against(source, sequence)
    return result


__all__ = ["MoeEp2RemotePayload", "MoeEp2ReverseRankPlan",
           "build_moe_ep2_reverse_rank_plan"]
