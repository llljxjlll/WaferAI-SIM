"""Witness every two-step Dense source action using its real native records."""

from __future__ import annotations

from collections import Counter

from ..errors import SchemaError
from ..schema.artifact_manifest import LinkedProgramManifest, RecordOpcode
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.dense_state_version_fence import dense_dp2_state_version_fences
from ..schema.artifact_manifest import canonical_dense_state_version_record
from ..schema.full_dense_gradient_requirements import DenseFullTrainRequirements
from ..schema.full_training_physical_dag import (
    FullTrainingPhysicalDAG, build_full_training_physical_dag,
)
from ..schema.ir0 import (
    CollectiveKind, OpKind, OpPhase, ResidualWorkload, SwiGluWorkload,
)
from ..schema.n6 import _leaf_fragments
from ..schema.ir2 import (
    FlowRouteRole, SemanticTaskKind, canonical_transit_completion_event,
)
from ..schema.persistent_state import PersistentStateAccess, StateKind
from ..schema.train_n6 import TrainLinkedProgram


def dense_two_step_native_opcode_contract(plan: FlexibleDenseTrainPlan) -> tuple[
    dict[str, RecordOpcode], dict[str, RecordOpcode],
]:
    """Derive derivative opcode families solely from validated forward nodes."""
    forward = {node.id: node for node in plan.forward_graph.nodes}
    backward = {}
    for node in forward.values():
        if node.kind in (OpKind.CE_FORWARD, OpKind.EMBEDDING, OpKind.COLLECTIVE):
            continue
        opcode = {
            OpKind.GEMM: RecordOpcode.GEMM_DX_TIMING,
            OpKind.NORM: RecordOpcode.RMSNORM_BACKWARD_TIMING,
            OpKind.ATTENTION: RecordOpcode.ATTENTION_BACKWARD_TIMING,
            OpKind.ROPE: RecordOpcode.ROPE_BACKWARD_TIMING,
        }.get(node.kind)
        if node.kind is OpKind.ELEMENTWISE:
            opcode = (RecordOpcode.SWIGLU_BACKWARD_TIMING
                      if type(node.workload) is SwiGluWorkload else
                      RecordOpcode.RESIDUAL_BACKWARD_TIMING
                      if type(node.workload) is ResidualWorkload else None)
        if opcode is None:
            raise SchemaError("untyped Dense derivative opcode family", path=node.id)
        backward[f"backward::{node.id}"] = opcode
    wgrad = {}
    for template in plan.parameter_templates:
        kinds = {forward[ref].kind for ref in template.forward_consumer_refs}
        if len(kinds) != 1:
            raise SchemaError("parameter derivative has ambiguous source", path=template.state_ref)
        opcode = {
            OpKind.GEMM: RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
            OpKind.NORM: RecordOpcode.NORM_GAMMA_WGRAD_TIMING,
            OpKind.EMBEDDING: RecordOpcode.EMBEDDING_TABLE_WGRAD_TIMING,
        }.get(next(iter(kinds)))
        if opcode is None:
            raise SchemaError("parameter derivative has no native opcode", path=template.state_ref)
        wgrad[template.wgrad_ref] = opcode
    return backward, wgrad


def require_named_tp_collective_reverse_records(
    program: TrainLinkedProgram,
    plan: FlexibleDenseTrainPlan,
    operation_by_action: dict[str, tuple[str, str, int, int | None, str]],
    records_by_action: dict[str, set[RecordOpcode]],
    transport_edges: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """Require each inverse TP collective's real rank-local DTE/SUM timeline.

    A reverse ReduceScatter has one FP16/FP32 SUM per owner; reverse
    AllGather transports peer shards and ends in one native rank barrier.
    The validated source/N4/N5 GlobalActionDAG fixes all rank/chunk/route
    identities; this check binds every one to its executable native records.
    """
    contexts = tuple(replica.lowering_context for replica in program.source.replicas)
    nodes_by_replica = tuple({node.id: node for node in context.ir1.nodes}
                             for context in contexts)
    native = {action.id: action for context in contexts
              for action in context.global_dag.actions}
    tp = plan.spec.tp_degree
    inverse = {}
    for node in plan.forward_graph.nodes:
        if node.kind is not OpKind.COLLECTIVE:
            continue
        reverse = (
            CollectiveKind.REDUCE_SCATTER
            if node.workload.collective is CollectiveKind.ALL_GATHER
            else CollectiveKind.ALL_GATHER
            if node.workload.collective is CollectiveKind.REDUCE_SCATTER else None
        )
        if reverse is None:
            raise SchemaError("forward TP collective lacks typed inverse", path=node.id)
        inverse[f"backward::{node.id}"] = reverse
    if tp == 1 and inverse:
        raise SchemaError("singleton TP must not claim cross-die inverse collective",
                          path="plan.forward_graph")
    if not inverse:
        return ()
    require_ops = {
        SemanticTaskKind.LOCAL_COPY: {
            RecordOpcode.DTE_ISSUE, RecordOpcode.DTE_WAIT,
        },
        SemanticTaskKind.SEND: {RecordOpcode.DTE_SEND},
        SemanticTaskKind.RECV: {RecordOpcode.DTE_RECV},
        SemanticTaskKind.WAIT: {RecordOpcode.DTE_WAIT},
        SemanticTaskKind.REDUCE: {RecordOpcode.LOCAL_REDUCE},
        SemanticTaskKind.BARRIER: {
            RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT,
        },
    }
    for dp, nodes in enumerate(nodes_by_replica):
      for ref, reverse in sorted(inverse.items()):
        for step in (0, 1):
            source_ref = f"{ref}::step{step}__dp{dp}"
            node = nodes.get(source_ref)
            if (node is None or node.kind is not OpKind.COLLECTIVE
                    or node.phase is not OpPhase.DGRAD
                    or node.workload.collective is not reverse):
                raise SchemaError("inverse collective differs from real source IR1",
                                  path=source_ref)
            for rank in range(dp * tp, (dp + 1) * tp):
                actions = tuple(
                    native[action_id]
                    for action_id, (_, op_ref, action_step, _, phase)
                    in operation_by_action.items()
                    if op_ref == ref and action_step == step
                    and phase == "backward"
                    and native[action_id].logical_core.die_id == rank
                )
                expected = {
                    SemanticTaskKind.LOCAL_COPY: 1,
                    SemanticTaskKind.SEND: tp - 1,
                    SemanticTaskKind.RECV: tp - 1,
                    (SemanticTaskKind.WAIT if reverse is
                     CollectiveKind.REDUCE_SCATTER else
                     SemanticTaskKind.BARRIER):
                        tp - 1 if reverse is CollectiveKind.REDUCE_SCATTER else 1,
                }
                if reverse is CollectiveKind.REDUCE_SCATTER:
                    expected[SemanticTaskKind.REDUCE] = 1
                if Counter(action.task_kind for action in actions) != expected:
                    raise SchemaError(
                        "inverse TP collective omits rank-local COPY/SEND/RECV/WAIT/SUM",
                        path=f"{ref}.step{step}.rank{rank}",
                    )
                for action in actions:
                    # A newly materialized collective buffer can carry native
                    # SRAM lifecycle records on its owner action. The linked
                    # BufferABI validator checks those independently; require
                    # the exact executable DTE/SUM records alongside them.
                    native_ops = records_by_action.get(action.id, set())
                    functional_ops = native_ops - {
                        RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.SRAM_FREE,
                    }
                    if (action.member_id != source_ref
                            or functional_ops != require_ops[action.task_kind]):
                        raise SchemaError(
                            "inverse TP collective action lacks its exact native DTE/SUM opcode",
                            path=action.id,
                        )
            edges = tuple((send, recv) for send, recv in transport_edges
                          if operation_by_action[send][1] == ref
                          and operation_by_action[send][2] == step
                          and operation_by_action[recv][1] == ref
                          and operation_by_action[recv][2] == step
                          and native[send].logical_core.die_id // tp == dp
                          and native[recv].logical_core.die_id // tp == dp)
            if len(edges) != tp * (tp - 1):
                raise SchemaError("inverse TP collective omits a real SEND→RECV flow",
                                  path=f"{ref}.step{step}")
    return tuple(sorted(inverse))


def build_full_dense_two_step_physical_dag(
    program: TrainLinkedProgram,
    plan: FlexibleDenseTrainPlan,
    requirements: DenseFullTrainRequirements,
) -> FullTrainingPhysicalDAG:
    """Record one verifiable physical STORE0 -> LOAD1 edge per StateABI.

    DP1 uses the WGRAD output directly. DP2 must prove both independently
    owned replicas and the native rank-major FP32 SUM route.
    """
    program.validate("full_dense_two_step_program")
    requirements.validate_against(plan)
    if (requirements.steps != 2
            or len(program.source.replicas) != plan.spec.dp_degree
            or plan.spec.dp_degree not in (1, 2)
            or tuple(replica.replica_index for replica in program.source.replicas)
               != tuple(range(plan.spec.dp_degree))):
        raise SchemaError("two-step physical witness requires exact DP1 or DP2 replicas",
                          path="requirements")
    contexts = tuple(replica.lowering_context for replica in program.source.replicas)
    all_actions = tuple(action for context in contexts
                        for action in context.global_dag.actions)
    if len({action.id for action in all_actions}) != len(all_actions):
        raise SchemaError("physical DP replicas alias a global action id", path="source.replicas")
    manifest: LinkedProgramManifest = program.manifest
    leaves = _leaf_fragments(manifest.fragments)
    states = {state.id: state for context in contexts
              for state in context.ir1.persistent_state_manifest.declarations}
    old_states = {state.id: state for state in plan.forward_graph.persistent_states}
    old_by_shard = {(state.identity.tensor_ref, state.identity.shard_index): state.id
                    for state in old_states.values()}
    if len(old_by_shard) != len(old_states):
        raise SchemaError("source parameter TP shards are not unique",
                          path="plan.forward_graph.persistent_states")
    source_state = {}
    for state in states.values():
        if (state.identity.kind is not StateKind.TRAINABLE_PARAMETER
                or state.access is not PersistentStateAccess.READ_WRITE
                or (state.identity.tensor_ref, state.identity.shard_index)
                not in old_by_shard):
            raise SchemaError("physical source has an unknown trainable state",
                              path="context.ir1.persistent_state_manifest")
        old_id = old_by_shard[state.identity.tensor_ref,
                              state.identity.shard_index]
        old = old_states[old_id]
        if (state.identity.instance_ref != old.identity.instance_ref
                or state.identity.mesh_ref != old.identity.mesh_ref
                or state.identity.shard_index != old.identity.shard_index
                or state.identity.generation != old.identity.generation
                or state.shape != old.shape or state.dtype != old.dtype):
            raise SchemaError("trainable state changed source owner/shape/dtype",
                              path="context.ir1.persistent_state_manifest")
        source_state[state.id] = old_id
    if set(source_state.values()) != set(old_states):
        raise SchemaError("every source parameter needs one real trainable StateABI",
                          path="context.ir1.persistent_state_manifest")
    binding_to_old = {binding.id: source_state[binding.state_ref]
                      for context in contexts
                      for binding in context.ir1.persistent_state_manifest.bindings}
    by_replica = tuple({node.id: node for node in context.ir1.nodes}
                       for context in contexts)
    native_source_action_ids = {
        record.source_global_action_id
        for fragment in leaves for stream in fragment.core_streams
        for record in stream.records
    }
    transits = {}
    operation_by_action: dict[str, tuple[str, str, int, int | None, str]] = {}
    stores: dict[str, object] = {}
    loads: dict[str, list[object]] = {}
    for action in all_actions:
        # TRANSIT belongs to the routed fabric and has no executable core,
        # buffer binding or native record.  Actual DTE SEND/RECV endpoints
        # below still bind the exact source flow and executable bytes.
        if action.task_kind is SemanticTaskKind.TRANSIT:
            if (action.logical_core is not None or action.state_uses
                    or action.buffer_uses or action.id in native_source_action_ids
                    or action.flow_id is None or action.flow is None
                    or action.flow_route is None):
                raise SchemaError("fabric transit lacks strict non-executable flow binding",
                                  path=action.id)
            transits.setdefault(action.flow_id, []).append(action)
            continue
        origin = action.origin_ref
        member = action.member_id
        rank = action.logical_core.die_id
        dp = rank // plan.spec.tp_degree
        if dp >= len(by_replica):
            raise SchemaError("native action escapes its physical DP replica", path=action.id)
        by_id = by_replica[dp]
        source_node = (by_id.get(member) if member is not None else
                       by_id.get(getattr(origin, "node_ref", None)))
        if source_node is None or not source_node.id.endswith(f"__dp{dp}"):
            raise SchemaError("native action lacks exact DP-replica IR1 source node", path=action.id)
        node_ref = source_node.id.removesuffix(f"__dp{dp}")
        if "::step0" in node_ref:
            step = 0
        elif "::step1" in node_ref:
            step = 1
        else:
            raise SchemaError("native action lacks exact step namespace", path=action.id)
        canonical = node_ref.replace(f"::step{step}_backward", "_backward")
        canonical = canonical.replace(f"::step{step}", "")
        binding_refs = {use.hbm_binding_ref for use in action.state_uses}
        if len(binding_refs) > 1:
            raise SchemaError("physical state action crosses source parameters", path=action.id)
        state_ref = binding_to_old[next(iter(binding_refs))] if binding_refs else None
        if (state_ref is not None
                and (old_states[state_ref].identity.shard_index
                     != rank % plan.spec.tp_degree
                     or rank not in next(template.owner_ranks
                                         for template in plan.parameter_templates
                                         if template.state_ref == state_ref))):
            raise SchemaError("native HBM state action uses a different TP shard owner",
                              path=action.id)
        if action.task_kind is SemanticTaskKind.DMA_OUT:
            if state_ref is None:
                raise SchemaError("STORE lacks source parameter StateABI", path=action.id)
            operation = f"store::{state_ref}::r{rank}::step{step}"
            phase = "optimizer"
            key = (state_ref, step, rank)
            if key in stores:
                raise SchemaError("physical STORE duplicated one parameter/rank/step", path=action.id)
            stores[key] = action
        elif action.task_kind is SemanticTaskKind.DMA_IN:
            if state_ref is None:
                raise SchemaError("LOAD lacks source parameter StateABI", path=action.id)
            operation = f"load::{state_ref}::r{rank}::step{step}"
            phase = "forward" if source_node.phase.value == "fwd" else "backward"
            loads.setdefault((state_ref, step, rank), []).append(action)
        elif source_node.kind is OpKind.OPTIMIZER_UPDATE:
            weight = source_node.inputs[0].removesuffix(f"__dp{dp}")
            owner_states = tuple(old_id for (tensor, shard), old_id
                                 in old_by_shard.items()
                                 if tensor == weight and shard == rank % plan.spec.tp_degree)
            state_ref = owner_states[0] if len(owner_states) == 1 else None
            if state_ref is None:
                raise SchemaError("SGD lacks exact source weight", path=action.id)
            if f"::tp{rank % plan.spec.tp_degree}::" not in node_ref:
                raise SchemaError("native SGD node shard differs from its physical die",
                                  path=action.id)
            operation = f"sgd::{state_ref}::r{rank}::step{step}"
            phase = "optimizer"
        else:
            if node_ref.startswith("dp_sync::"):
                trainable_ref = node_ref.split("::", 2)[1]
                original_ref = source_state.get(trainable_ref)
                if original_ref is None:
                    raise SchemaError("DP SUM lacks source parameter StateDecl",
                                      path=action.id)
                operation = node_ref.replace(trainable_ref, original_ref, 1)
            else:
                operation = canonical
            phase = ("loss" if source_node.kind in (
                OpKind.CE_FORWARD, OpKind.CE_BACKWARD) else
                "optimizer" if source_node.kind is OpKind.OPTIMIZER_UPDATE else
                "backward" if source_node.phase.value in ("dgrad", "wgrad")
                else "forward")
        tensor_ref = (old_states[state_ref].identity.tensor_ref
                      if state_ref is not None else canonical)
        layer = next((index for index in (0, 1)
                      if f".layer{index}." in tensor_ref), None)
        operation_by_action[action.id] = (action.id, operation, step, layer, phase)
    expected = {path.parameter_state_ref for path in requirements.paths}
    expected_versions = {(path.parameter_state_ref, path.step, path.rank)
                         for path in requirements.paths}
    if set(stores) != expected_versions or set(loads) != expected_versions:
        raise SchemaError(
            "two-step physical state coverage is incomplete: "
            f"source={len(expected)} stores={len(stores)} loads={len(loads)} "
            f"store_parameters={len({key[0] for key in stores})} "
            f"load_parameters={len({key[0] for key in loads})}",
            path="actions",
        )
    state_edges = []
    fences = {fence.store_action_id: [] for context in contexts
              for fence in dense_dp2_state_version_fences(context.global_dag)}
    for context in contexts:
        for fence in dense_dp2_state_version_fences(context.global_dag):
            fences[fence.store_action_id].append(fence)
    native_records = {action_id: [] for action_id in operation_by_action}
    for fragment in leaves:
        for stream in fragment.core_streams:
            for record in stream.records:
                if record.source_global_action_id in native_records:
                    native_records[record.source_global_action_id].append(record)
    owners = {(path.parameter_state_ref, path.rank) for path in requirements.paths}
    for old_ref, rank in sorted(owners):
        store = stores.get((old_ref, 0, rank))
        read = loads.get((old_ref, 1, rank))
        if store is None or not read:
            raise SchemaError("STORE0 needs same-state LOAD1", path=old_ref)
        first_by_core = {}
        for item in read:
            core = item.logical_core
            previous = first_by_core.get(core)
            if previous is None or item.core_order_index < previous.core_order_index:
                first_by_core[core] = item
        for first in first_by_core.values():
            if first.logical_core == store.logical_core:
                if first.core_order_index <= store.core_order_index:
                    raise SchemaError("same-core LOAD1 precedes STORE0", path=old_ref)
            else:
                pair = tuple(fence for fence in fences.get(store.id, ())
                             if fence.load_action_id == first.id)
                if len(pair) != 1:
                    raise SchemaError("cross-core state version lacks exact source event pair",
                                      path=old_ref)
                fence = pair[0]
                required_set = canonical_dense_state_version_record(
                    fence, owner_id=store.id, opcode=RecordOpcode.EVENT_SET)
                required_wait = canonical_dense_state_version_record(
                    fence, owner_id=first.id, opcode=RecordOpcode.EVENT_WAIT)
                if (required_set not in native_records[store.id]
                        or required_wait not in native_records[first.id]):
                    raise SchemaError("cross-core state version lacks native SET/WAIT records",
                                      path=old_ref)
            state_edges.append((store.id, first.id))
    # Bind each cross-die SEND/RECV by its original flow and executable DTE
    # records, including TP backward ReduceScatter's distinct peer routes.
    records_by_action = {}
    for fragment in leaves:
        for stream in fragment.core_streams:
            for record in stream.records:
                records_by_action.setdefault(record.source_global_action_id, set()).add(
                    record.opcode)
    flows = {}
    for action in all_actions:
        if action.task_kind not in (SemanticTaskKind.SEND, SemanticTaskKind.RECV):
            continue
        if action.flow_id is None or action.flow is None or action.flow_route is None:
            raise SchemaError("cross-die DTE action lacks source flow/route", path=action.id)
        flow = flows.setdefault(action.flow_id, {})
        if action.task_kind in flow:
            raise SchemaError("source flow duplicates a physical DTE endpoint", path=action.flow_id)
        flow[action.task_kind] = action
    if not set(transits).issubset(flows):
        raise SchemaError("routed fabric transit lacks a physical DTE flow peer",
                          path="global_dag.actions")
    def shares_route_flow(action, source):
        # SemanticFlow.task_ids is die-local: SEND, each TRANSIT and RECV
        # legitimately have distinct task IDs while every route/payload field
        # must remain identical to the original source flow.
        fields = ("id", "logical_channel", "pair_route_ref", "source_rank",
                  "destination_rank", "source_die", "destination_die",
                  "die_path", "tensor_slice", "bytes", "dtype")
        return (action.flow is not None
                and action.source.task_id in action.flow.task_ids
                and all(getattr(action.flow, key) == getattr(source.flow, key)
                        for key in fields))
    transport_edges = []
    for flow_id, pair in sorted(flows.items()):
        if set(pair) != {SemanticTaskKind.SEND, SemanticTaskKind.RECV}:
            raise SchemaError("source flow lacks one physical DTE peer", path=flow_id)
        send, recv = pair[SemanticTaskKind.SEND], pair[SemanticTaskKind.RECV]
        path = send.flow.die_path
        if (not shares_route_flow(send, send)
                or send.logical_core == recv.logical_core
                or send.logical_core.die_id != path[0]
                or recv.logical_core.die_id != path[-1]
                or send.source_rank != recv.source_rank
                or send.destination_rank != recv.destination_rank
                or not shares_route_flow(recv, send)
                or send.flow_route.flow_id != flow_id
                or recv.flow_route.flow_id != flow_id
                or send.flow_route.pair_route_ref != send.flow.pair_route_ref
                or recv.flow_route.pair_route_ref != send.flow.pair_route_ref
                or send.flow_route.role is not FlowRouteRole.SOURCE
                or recv.flow_route.role is not FlowRouteRole.DESTINATION
                or send.tensor_slice != recv.tensor_slice
                or send.dtype != recv.dtype or send.bytes != recv.bytes
                or RecordOpcode.DTE_SEND not in records_by_action.get(send.id, ())
                or RecordOpcode.DTE_RECV not in records_by_action.get(recv.id, ())):
            raise SchemaError("DTE endpoints differ in source flow/route or executable record",
                              path=flow_id)
        transits_by_die = {}
        for transit in transits.get(flow_id, ()):
            for die in path[1:-1]:
                if transit.source.task_id == f"task.transit.{flow_id}.die.{die}":
                    if die in transits_by_die:
                        raise SchemaError("fabric flow has duplicate transit die", path=flow_id)
                    transits_by_die[die] = transit
                    break
            else:
                raise SchemaError("fabric transit lacks canonical route hop",
                                  path=transit.id)
        if set(transits_by_die) != set(path[1:-1]):
            raise SchemaError("fabric route omits a physical intermediate die",
                              path=flow_id)
        route_steps = [send, *(transits_by_die[die] for die in path[1:-1]), recv]
        for hop, action in enumerate(route_steps):
            route = action.flow_route
            if (not shares_route_flow(action, send) or route.flow_id != flow_id
                    or route.pair_route_ref != send.flow.pair_route_ref
                    or action.source_rank != send.source_rank
                    or action.destination_rank != send.destination_rank
                    or action.tensor_slice != send.tensor_slice
                    or action.dtype != send.dtype or action.bytes != send.bytes):
                raise SchemaError("fabric route hop changes semantic flow payload",
                                  path=action.id)
            if 0 < hop < len(route_steps) - 1:
                if (route.role is not FlowRouteRole.TRANSIT
                        or action.sync is None
                        or action.sync.completion_event !=
                            canonical_transit_completion_event(flow_id, path[hop])):
                    raise SchemaError("fabric transit lacks exact die/hop completion",
                                      path=action.id)
            if hop + 1 < len(route_steps):
                peer_ingress = route_steps[hop + 1].flow_route.ingress
                if (route.egress is None or peer_ingress is None
                        or route.egress.link_ref != peer_ingress.link_ref):
                    raise SchemaError("fabric flow's adjacent D2D route legs differ",
                                      path=flow_id)
        transport_edges.append((send.id, recv.id))
    backward, wgrad = dense_two_step_native_opcode_contract(plan)
    named_collectives = require_named_tp_collective_reverse_records(
        program, plan, operation_by_action, records_by_action,
        tuple(sorted(transport_edges)),
    )
    required = {
        *backward, *wgrad, *named_collectives,
        *(path.named_sync_op_ref for path in requirements.paths
          if len(path.dp_group_ranks) > 1),
        *(path.named_optimizer_op_ref for path in requirements.paths),
        *(path.named_store_op_ref for path in requirements.paths),
    }
    missing = required - {entry[1] for entry in operation_by_action.values()}
    if missing:
        raise SchemaError("physical Dense operation lacks exact source action: "
                          f"{tuple(sorted(missing))!r}", path="required_operation_ids")
    result = build_full_training_physical_dag(
        fragments=leaves, streams=manifest.core_streams,
        source_artifact_ids=tuple(sorted({artifact_id for context in contexts
                                          for artifact_id in (context.ir1.id,
                                                              context.global_dag.id)})),
        operation_by_action=operation_by_action,
        transport_edges=tuple(sorted(transport_edges)),
        state_version_edges=tuple(sorted(state_edges)),
        required_operation_ids=tuple(sorted(required)),
    )
    return result


__all__ = ["build_full_dense_two_step_physical_dag",
           "dense_two_step_native_opcode_contract"]
