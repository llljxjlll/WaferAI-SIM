"""Source-bound EP2 node ranks and required physical inter-die value transfers.

This is a transport contract, not an executed SEND/RECV program.  The
ordinary IR2 projector must consume it before an EP2 training run is legal.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from ..errors import SchemaError
from ..schema.common import DType, MeshAxisName
from ..schema.ir0 import OpKind, StateAccessMode
from ..schema.flexible_moe import MoeRectActionKind, MoeRectFlowStage
from ..schema.moe_compile_sequence import MoeCompileSequence
from .moe_full_train_ep_ir1_source import MoeEpSharedReverseIr1Candidate


_BYTES = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}


@dataclass(frozen=True, slots=True)
class MoeEp2NodeRank:
    node_ref: str
    rank: int
    die_id: int


@dataclass(frozen=True, slots=True)
class MoeEp2Transfer:
    value_ref: str
    source_node_ref: str | None
    source_state_ref: str | None
    consumer_node_ref: str
    source_rank: int
    destination_rank: int
    source_die: int
    destination_die: int
    route_ref: str
    die_path: tuple[int, ...]
    source_flow_ref: str | None
    source_send_action_ref: str | None
    destination_recv_action_ref: str | None
    destination_wait_action_ref: str | None
    bytes: int


@dataclass(frozen=True, slots=True)
class MoeEp2RankPlan:
    source_ir1_id: str
    node_ranks: tuple[MoeEp2NodeRank, ...]
    transfers: tuple[MoeEp2Transfer, ...]

    def validate_against(
        self, source: MoeEpSharedReverseIr1Candidate,
        sequence: MoeCompileSequence,
    ) -> None:
        expected = _derive(source, sequence)
        if self != expected:
            raise SchemaError(
                "EP2 node rank or physical transfer differs from source/owner route",
                path="moe_ep2_rank_plan",
            )


def _derive(
    source: MoeEpSharedReverseIr1Candidate,
    sequence: MoeCompileSequence,
) -> MoeEp2RankPlan:
    if type(source) is not MoeEpSharedReverseIr1Candidate:
        raise SchemaError("requires source-backed shared reverse candidate", path="source")
    sequence.validate()
    if source.forward.source_moe_sequence_ref != sequence.id:
        raise SchemaError("EP2 rank plan must use the signed source MoE sequence",
                          path="sequence")
    graph = source.source_ir0
    ir1 = source.physical_ir1
    graph.validate("moe_ep2_rank_plan.source_ir0")
    ir1.validate("moe_ep2_rank_plan.source_ir1")
    group = source.forward.physical_ir1.groups[0]
    if (graph.producer_pass != "moe_full_train_shared_reverse_ir0"
            or graph.instances[0].parallel.tp != 1
            or graph.instances[0].parallel.ep != 2
            or len(graph.nodes) != 38
            or ir1.source_ir0_id != graph.id
            or ir1.groups != (group,)
            or group.axis is not MeshAxisName.EP
            or group.logical_shape != (1, 2)
            or tuple(place.rank for place in group.placements) != (0, 1)):
        raise SchemaError("requires exact two-layer TP1×EP2 shared reverse source",
                          path="source")
    instance_id = graph.instances[0].id
    dies = {place.rank: place.die_id for place in group.placements}
    routes = {(route.source_rank, route.destination_rank): route
              for route in group.embedding.routes}
    if (set(routes) != {(0, 1), (1, 0)}
            or any(route.die_path[0] != dies[route.source_rank]
                   or route.die_path[-1] != dies[route.destination_rank]
                   for route in routes.values())):
        raise SchemaError("EP2 requires authenticated routes in both directions",
                          path="source.physical_ir1.groups")
    rank_by_node = {}
    for node in graph.nodes:
        rank = node.workload.expert if node.kind is OpKind.MOE_EXPERT_FORWARD else 0
        if rank not in dies:
            raise SchemaError("expert has no EP owner", path=f"source.nodes.{node.id}")
        rank_by_node[node.id] = rank
    node_ranks = tuple(MoeEp2NodeRank(node.id, rank_by_node[node.id],
                                       dies[rank_by_node[node.id]])
                       for node in graph.nodes)
    steps = {node.workload.step for node in graph.nodes
             if node.kind is OpKind.MOE_COMBINE}
    if len(steps) != 1:
        raise SchemaError("EP2 rank plan requires one signed training step",
                          path="source.nodes")
    step = next(iter(steps))
    units = {(unit.step, unit.layer): unit for unit in sequence.units}
    state_by_id = {state.id: state for state in graph.persistent_states}
    binding_by_ref = {binding.state_ref: binding
                      for binding in ir1.persistent_state_manifest.bindings}
    transfers = []
    for value in graph.values:
        for consumer_ref in value.consumers:
            destination_rank = rank_by_node[consumer_ref]
            state_ref = None
            if value.producer is None:
                accesses = tuple(access for access in graph.state_accesses
                                 if access.node_ref == consumer_ref
                                 and state_by_id[access.state_ref].identity.tensor_ref
                                    == value.id)
                if not accesses:
                    continue
                if len(accesses) != 1 or accesses[0].mode is not StateAccessMode.READ:
                    raise SchemaError("EP2 state-backed value lacks one owner read",
                                      path=f"source.values.{value.id}")
                access = accesses[0]
                source_rank = access.rank
                state_ref = access.state_ref
                if binding_by_ref[state_ref].die_id != dies[source_rank]:
                    raise SchemaError("EP2 parameter owner differs from HBM home",
                                      path=f"source.values.{value.id}")
            else:
                source_rank = rank_by_node[value.producer]
            if source_rank == destination_rank:
                continue
            route = routes.get((source_rank, destination_rank))
            if route is None or value.dtype not in _BYTES:
                raise SchemaError("EP2 cross-die value lacks physical route or dtype",
                                  path=f"source.values.{value.id}")
            logical_bytes = prod(value.shape) * _BYTES[value.dtype]
            source_flow_ref = None
            send_ref = recv_ref = wait_ref = None
            for layer in (0, 1):
                prefix = f"{instance_id}.layer{layer}.moe."
                stage = (
                    MoeRectFlowStage.DISPATCH
                    if value.id == prefix + "dispatch1"
                    else MoeRectFlowStage.COMBINE
                    if value.id == prefix + "expert1.output"
                    else None
                )
                if stage is None:
                    continue
                unit = units.get((step, layer))
                matching = tuple(flow for flow in unit.plan.flows
                                 if flow.stage is stage
                                 and flow.source_rank == source_rank
                                 and flow.destination_rank == destination_rank
                                 and flow.logical_bytes == logical_bytes) if unit else ()
                if len(matching) != 1:
                    raise SchemaError("EP2 remote value lacks one exact signed P2 flow",
                                      path=f"source.values.{value.id}")
                source_flow_ref = matching[0].id
                actions = tuple(unit.plan.actions)
                def one(kind, rank):
                    found = tuple(action for action in actions
                                  if action.kind is kind and action.rank == rank
                                  and action.flow_ref == source_flow_ref)
                    if len(found) != 1:
                        raise SchemaError("EP2 signed flow lacks one native action",
                                          path=f"source.values.{value.id}")
                    return found[0]
                send = one(MoeRectActionKind.SEND, source_rank)
                recv = one(MoeRectActionKind.RECV, destination_rank)
                wait = one(MoeRectActionKind.WAIT, destination_rank)
                if not {send.id, recv.id}.issubset(wait.deps):
                    raise SchemaError("EP2 WAIT does not depend on signed SEND/RECV",
                                      path=f"source.values.{value.id}")
                if stage is MoeRectFlowStage.DISPATCH:
                    producers = tuple(action for action in actions
                                      if action.kind is MoeRectActionKind.PACK
                                      and action.rank == source_rank
                                      and action.id in send.deps)
                    consumers = tuple(action for action in actions
                                      if action.kind is MoeRectActionKind.EXPERT_FORWARD
                                      and action.rank == destination_rank
                                      and wait.id in action.deps)
                else:
                    producers = tuple(action for action in actions
                                      if action.kind is MoeRectActionKind.EXPERT_FORWARD
                                      and action.rank == source_rank
                                      and action.id in send.deps)
                    consumers = tuple(action for action in actions
                                      if action.kind is MoeRectActionKind.WEIGHTED_COMBINE
                                      and action.rank == destination_rank
                                      and wait.id in action.deps)
                if len(producers) != 1 or len(consumers) != 1:
                    raise SchemaError("EP2 P2 action chain does not carry true expert value",
                                      path=f"source.values.{value.id}")
                send_ref, recv_ref, wait_ref = send.id, recv.id, wait.id
                break
            transfers.append(MoeEp2Transfer(
                value.id, value.producer, state_ref, consumer_ref,
                source_rank, destination_rank, dies[source_rank],
                dies[destination_rank], route.id, route.die_path,
                source_flow_ref, send_ref, recv_ref, wait_ref, logical_bytes,
            ))
    transfers = tuple(sorted(transfers, key=lambda item: (item.value_ref,
                                                           item.consumer_node_ref)))
    # A two-layer source has exactly one remote router weight, dispatch and
    # expert return per layer; no shared-spine value may cross implicitly.
    expected = {
        (value_ref, source_rank, destination_rank)
        for layer in (0, 1)
        for value_ref, source_rank, destination_rank in (
            (f"{instance_id}.layer{layer}.moe.router.weight.ep1", 1, 0),
            (f"{instance_id}.layer{layer}.moe.dispatch1", 0, 1),
            (f"{instance_id}.layer{layer}.moe.expert1.output", 1, 0),
        )
    }
    if ({(item.value_ref, item.source_rank, item.destination_rank)
         for item in transfers} != expected
            or len(transfers) != 6
            or any(item.bytes <= 0 for item in transfers)
            or any((item.source_flow_ref is None) !=
                   (item.source_state_ref is not None) for item in transfers)
            or any((item.source_flow_ref is None) !=
                   (item.source_send_action_ref is None)
                   or (item.source_flow_ref is None) !=
                   (item.destination_recv_action_ref is None)
                   or (item.source_flow_ref is None) !=
                   (item.destination_wait_action_ref is None)
                   for item in transfers)):
        raise SchemaError("EP2 needs six exact router/dispatch/return transfers",
                          path="source.values")
    return MoeEp2RankPlan(ir1.id, node_ranks, transfers)


def build_moe_ep2_rank_plan(
    source: MoeEpSharedReverseIr1Candidate,
    sequence: MoeCompileSequence,
) -> MoeEp2RankPlan:
    result = _derive(source, sequence)
    result.validate_against(source, sequence)
    return result


__all__ = ["MoeEp2NodeRank", "MoeEp2Transfer", "MoeEp2RankPlan",
           "build_moe_ep2_rank_plan"]
