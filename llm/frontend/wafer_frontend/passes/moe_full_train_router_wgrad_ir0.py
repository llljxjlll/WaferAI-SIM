"""Source-bound FP32 router gate WGRAD from the real native 0x28 dScore."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase,
)


def append_moe_full_train_router_wgrad_ir0(source: IR0) -> IR0:
    source.validate("moe_router_wgrad_source")
    if (source.producer_pass != "moe_full_train_combine_backward_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.ep != 1
            or source.instances[0].parallel.tp != 1):
        raise SchemaError("requires source-bound EP1 native 0x28 graph",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    router = nodes.get(f"{instance.id}.layer1.moe.router")
    backward = nodes.get(f"backward::{instance.id}.layer1.moe.combine")
    if (router is None or router.kind is not OpKind.MOE_ROUTER
            or backward is None
            or backward.kind is not OpKind.MOE_COMBINE_BACKWARD
            or router.workload.step != backward.workload.step
            or router.workload.layer != backward.workload.layer
            or len(router.inputs) != 2):
        raise SchemaError("actual same-step router and 0x28 dScore required",
                          path="source.nodes")
    gate = values[router.inputs[1]]
    states = tuple(item for item in source.persistent_states
                   if item.identity.tensor_ref == gate.id)
    if len(states) != 1 or states[0].shape != gate.shape:
        raise SchemaError("router gate has no unique real StateDecl",
                          path="source.persistent_states")
    state = states[0]
    m, h, e = (router.workload.token_count,
               router.workload.hidden_size,
               router.workload.expert_count)
    if (e != 1 or gate.shape != (h, e)
            or values[router.inputs[0]].shape != (m, h)
            or values[backward.outputs[0]].shape != (m, e)):
        raise SchemaError("router 0x25 physical H/E/M geometry drifted",
                          path="source.nodes")
    node_id = f"backward::{router.id}::{state.id}"
    output_id = f"{node_id}.weight_gradient"
    if node_id in nodes or output_id in values:
        raise SchemaError("router WGRAD already exists", path="source")
    output = TensorValue(
        output_id, gate.shape, DType.FP32,
        "HE_router_gate_weight_gradient",
        gate.sharding, node_id, (), None,
    )
    wgrad = LogicalNode(
        node_id, instance.id, OpKind.GEMM_WEIGHT_WGRAD,
        OpPhase.WGRAD, router.stage, router.mesh_ref,
        (router.inputs[0], backward.outputs[0]), (output_id,),
        GemmWeightWgradWorkload(
            m=h, n=e, k=m, source_forward_op_ref=router.id,
            source_parameter_state_ref=state.id,
        ),
        router.math, NodeEffects(EffectKind.PURE, None, None),
        "gemm_weight_wgrad_timing",
    )
    all_nodes = (*source.nodes, wgrad)
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
        producer_pass="moe_full_train_router_wgrad_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=source.state_accesses,
    )
    result.validate("moe_full_train_router_wgrad_ir0")
    return result
