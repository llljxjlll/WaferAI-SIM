"""EP1 router activation dX from its real gate state and same-layer 0x28 dScore."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, StateAccess, StateAccessMode,
)


def append_moe_full_train_router_dx_ir0(source: IR0) -> IR0:
    source.validate("moe_router_dx_source")
    if (source.producer_pass != "moe_full_train_expert_backward_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1):
        raise SchemaError("router dX requires physical EP1 expert/dScore source",
                          path="source")
    instance = source.instances[0]
    nodes = {item.id: item for item in source.nodes}
    values = {item.id: item for item in source.values}
    router = nodes.get(f"{instance.id}.layer1.moe.router")
    combine = nodes.get(f"backward::{instance.id}.layer1.moe.combine")
    if (router is None or router.kind is not OpKind.MOE_ROUTER
            or combine is None or combine.kind is not OpKind.MOE_COMBINE_BACKWARD
            or router.workload.expert_count != 1
            or router.workload.step != combine.workload.step
            or router.workload.layer != combine.workload.layer
            or router.workload.source_route_trace_digest
               != combine.workload.source_route_trace_digest
            or router.outputs[0] != combine.inputs[1]
            or combine.outputs[0] not in values):
        raise SchemaError("router dX lacks same-layer forward/0x28 dScore",
                          path="source.nodes")
    activation, weight = (values[ref] for ref in router.inputs)
    upstream = values[combine.outputs[0]]
    m, h, e = (router.workload.token_count,
               router.workload.hidden_size, router.workload.expert_count)
    states = tuple(item for item in source.persistent_states
                   if item.identity.tensor_ref == weight.id)
    if (len(states) != 1 or states[0].shape != (h, e)
            or states[0].identity.ep_owner_rank != 0
            or activation.shape != (m, h)
            or upstream.shape != (m, e)
            or upstream.dtype is not DType.FP16
            or upstream.producer != combine.id):
        raise SchemaError("router dX needs unique owned H×1 StateDecl and true dScore",
                          path="source")
    state = states[0]
    node_id = f"backward::{router.id}.input"
    output_id = f"{node_id}.gradient"
    if node_id in nodes or output_id in values:
        raise SchemaError("router dX already exists", path="source")
    output = TensorValue(
        output_id, activation.shape, DType.FP16,
        "MH_router_input_gradient", activation.sharding,
        node_id, (), None,
    )
    backward = LogicalNode(
        node_id, instance.id, OpKind.GEMM_INPUT_DX,
        OpPhase.DGRAD, router.stage, router.mesh_ref,
        (weight.id, upstream.id), (output.id,),
        GemmInputDxWorkload(
            k=m, m=h, n=e, source_forward_op_ref=router.id,
            source_parameter_state_ref=state.id,
        ),
        router.math, NodeEffects(EffectKind.PURE, None, None),
        "gemm_input_dx_timing",
    )
    all_nodes = (*source.nodes, backward)
    all_values = (*source.values, output)
    consumers = {value.id: [] for value in all_values}
    for node in all_nodes:
        for ref in node.inputs:
            consumers[ref].append(node.id)
    final_values = tuple(replace(value, consumers=tuple(consumers[value.id]))
                         for value in all_values)
    data_edges = tuple(GraphEdge(
        f"{value.id}.edge_to.{consumer}", EdgeKind.DATA,
        value.producer, consumer, value.id,
    ) for value in final_values if value.producer is not None
                  for consumer in value.consumers)
    controls = tuple(edge for edge in source.edges
                     if edge.kind is EdgeKind.CONTROL)
    result = IR0.create(
        producer_pass="moe_full_train_router_dx_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=(*source.state_accesses,
                        StateAccess.create(node_ref=backward.id,
                                           state_ref=state.id,
                                           mode=StateAccessMode.READ, rank=0)),
    )
    result.validate("moe_full_train_router_dx_ir0")
    return result


__all__ = ["append_moe_full_train_router_dx_ir0"]
