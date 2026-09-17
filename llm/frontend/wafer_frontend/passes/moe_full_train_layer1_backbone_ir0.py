"""Propagate source-bound layer1 MoE dX through norm2 into residual1.

This stage stops at the residual1 upstream.  Attention, layer0 and updates
require later producers and must not be inferred from this graph alone.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.dense_backward_workloads import RmsNormBackwardWorkload
from ..schema.moe_training_ir0_workloads import NormGammaWgradWorkload
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, ResidualWorkload,
)


def append_moe_full_train_layer1_backbone_ir0(source: IR0) -> IR0:
    source.validate("moe_layer1_backbone_source")
    if (source.producer_pass != "moe_full_train_input_gradient_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1):
        raise SchemaError("requires exact EP1 MoE norm2 gradient source",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    prefix = f"{instance.id}.layer1"
    norm = nodes.get(f"{prefix}.norm2")
    residual1 = nodes.get(f"{prefix}.residual1")
    residual2 = nodes.get(f"{prefix}.residual2")
    residual2_dx = nodes.get(f"backward::{residual2.id}") if residual2 else None
    moe_sum = nodes.get(f"backward::{prefix}.moe.input_sum")
    if (norm is None or norm.kind is not OpKind.NORM
            or residual1 is None or residual1.kind is not OpKind.ELEMENTWISE
            or residual2 is None or residual2.kind is not OpKind.ELEMENTWISE
            or residual2_dx is None
            or residual2_dx.kind is not OpKind.RESIDUAL_BACKWARD
            or moe_sum is None or moe_sum.kind is not OpKind.ELEMENTWISE
            or norm.inputs[0] != residual1.outputs[0]
            or residual2.inputs[0] != residual1.outputs[0]
            or len(residual2_dx.outputs) != 2
            or residual2_dx.inputs[0] != residual2.outputs[0]
            or values[moe_sum.outputs[0]].producer != moe_sum.id
            or values[residual2_dx.outputs[0]].producer != residual2_dx.id
            or norm.stage != residual1.stage or norm.mesh_ref != residual1.mesh_ref):
        raise SchemaError("layer1 norm2/residual1/skip reverse source drifted",
                          path="source.nodes")
    rows, hidden = norm.workload.rank_activation_shape
    activation = values[norm.inputs[0]]
    upstream = values[moe_sum.outputs[0]]
    skip = values[residual2_dx.outputs[0]]
    if (activation.shape != (rows, hidden)
            or upstream.shape != activation.shape
            or skip.shape != activation.shape
            or upstream.dtype is not DType.FP16
            or skip.dtype is not DType.FP16
            or upstream.sharding != activation.sharding
            or skip.sharding != activation.sharding):
        raise SchemaError("layer1 norm2 upstream/skip tensor extent drifted",
                          path="source.values")
    weights = tuple(state for state in source.persistent_states
                    if state.identity.tensor_ref == norm.inputs[1])
    if len(weights) != 1 or weights[0].shape != (hidden,):
        raise SchemaError("layer1 norm2 gamma needs one owned StateDecl",
                          path="source.persistent_states")
    weight = weights[0]
    gamma_id = f"backward::{norm.id}::{weight.id}"
    dx_id = f"backward::{norm.id}"
    merge_id = f"backward::{residual1.id}.merge"
    gamma_value = f"{gamma_id}.output"
    dx_value = f"{dx_id}.input_gradient"
    merge_value = f"{merge_id}.input_gradient"
    if any(ref in nodes for ref in (gamma_id, dx_id, merge_id)) or any(
            ref in values for ref in (gamma_value, dx_value, merge_value)):
        raise SchemaError("layer1 backbone gradient already exists", path="source")
    pure = NodeEffects(EffectKind.PURE, None, None)
    gamma = LogicalNode(
        gamma_id, instance.id, OpKind.NORM_GAMMA_WGRAD,
        OpPhase.WGRAD, norm.stage, norm.mesh_ref,
        (activation.id, upstream.id), (gamma_value,),
        NormGammaWgradWorkload(rows, rows, 1, hidden, 0),
        norm.math, pure, "norm_gamma_wgrad_timing",
    )
    dx = LogicalNode(
        dx_id, instance.id, OpKind.RMSNORM_BACKWARD,
        OpPhase.DGRAD, norm.stage, norm.mesh_ref,
        (activation.id, upstream.id), (dx_value,),
        RmsNormBackwardWorkload(rows, hidden, 1),
        norm.math, pure, "rmsnorm_backward_timing",
    )
    merge = LogicalNode(
        merge_id, instance.id, OpKind.ELEMENTWISE,
        OpPhase.DGRAD, residual1.stage, residual1.mesh_ref,
        (skip.id, dx_value), (merge_value,),
        ResidualWorkload(activation.shape, activation.shape, DType.FP16),
        residual1.math, pure, "residual",
    )
    new_values = (
        TensorValue(gamma_value, values[norm.inputs[1]].shape, DType.FP32,
                    "H_layer1_norm2_gamma_gradient",
                    values[norm.inputs[1]].sharding, gamma_id, (), None),
        TensorValue(dx_value, activation.shape, DType.FP16,
                    "MH_layer1_norm2_input_gradient", activation.sharding,
                    dx_id, (), None),
        TensorValue(merge_value, activation.shape, DType.FP16,
                    "MH_layer1_residual1_gradient", activation.sharding,
                    merge_id, (), None),
    )
    all_nodes = (*source.nodes, gamma, dx, merge)
    all_values = (*source.values, *new_values)
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
        producer_pass="moe_full_train_layer1_backbone_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=source.state_accesses,
    )
    result.validate("moe_full_train_layer1_backbone_ir0")
    return result


__all__ = ["append_moe_full_train_layer1_backbone_ir0"]
