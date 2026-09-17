"""Merge real EP1 expert and router input derivatives into layer1 norm2 dX.

The expert tensor uses packed expert-slot order.  This first physical source
admits it as token order only after proving the signed dispatch slots are the
identity permutation; other routes require an explicit inverse dispatch.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, ResidualWorkload,
)


def append_moe_full_train_input_gradient_ir0(source: IR0) -> IR0:
    source.validate("moe_input_gradient_source")
    if (source.producer_pass != "moe_full_train_router_dx_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1):
        raise SchemaError("requires exact EP1 router dX source", path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    prefix = f"{instance.id}.layer1.moe"
    dispatch = nodes.get(f"{prefix}.dispatch")
    expert = nodes.get(f"backward::{prefix}.expert0")
    router = nodes.get(f"backward::{prefix}.router.input")
    norm = nodes.get(f"{instance.id}.layer1.norm2")
    if (dispatch is None or dispatch.kind is not OpKind.MOE_DISPATCH
            or expert is None or expert.kind is not OpKind.MOE_EXPERT_BACKWARD
            or router is None or router.kind is not OpKind.GEMM_INPUT_DX
            or norm is None or norm.kind is not OpKind.NORM
            or dispatch.inputs[0] != norm.outputs[0]
            or expert.inputs[0] != dispatch.outputs[0]
            or router.workload.source_forward_op_ref != f"{prefix}.router"
            or dispatch.workload.expert_count != 1
            or dispatch.workload.frozen_expert_by_token !=
               (0,) * dispatch.workload.token_count
            or dispatch.workload.frozen_slot_by_token !=
               tuple(range(dispatch.workload.token_count))
            or expert.workload.source_route_trace_digest !=
               dispatch.workload.source_route_trace_digest
            or expert.workload.step != dispatch.workload.step
            or expert.workload.layer != dispatch.workload.layer
            or values[expert.outputs[0]].shape != values[norm.outputs[0]].shape
            or values[router.outputs[0]].shape != values[norm.outputs[0]].shape
            or values[expert.outputs[0]].dtype is not DType.FP16
            or values[router.outputs[0]].dtype is not DType.FP16):
        raise SchemaError("expert/router dX lacks signed identity dispatch and same-layer norm2 source",
                          path="source.nodes")
    node_id = f"backward::{prefix}.input_sum"
    output_id = f"{node_id}.norm2_gradient"
    if node_id in nodes or output_id in values:
        raise SchemaError("MoE input gradient sum already exists", path="source")
    input_value = values[norm.outputs[0]]
    output = TensorValue(
        output_id, input_value.shape, DType.FP16,
        "MH_layer1_norm2_gradient", input_value.sharding,
        node_id, (), None,
    )
    merge = LogicalNode(
        node_id, instance.id, OpKind.ELEMENTWISE,
        OpPhase.DGRAD, norm.stage, norm.mesh_ref,
        (expert.outputs[0], router.outputs[0]), (output_id,),
        ResidualWorkload(input_value.shape, input_value.shape, DType.FP16),
        norm.math, NodeEffects(EffectKind.PURE, None, None), "residual",
    )
    all_nodes = (*source.nodes, merge)
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
        producer_pass="moe_full_train_input_gradient_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=source.state_accesses,
    )
    result.validate("moe_full_train_input_gradient_ir0")
    return result


__all__ = ["append_moe_full_train_input_gradient_ir0"]
