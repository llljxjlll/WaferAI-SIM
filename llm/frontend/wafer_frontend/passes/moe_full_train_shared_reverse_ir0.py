"""Derive the real shared final norm and layer1 dCombined reverse path.

The returned dCombined is the physical input needed by later MoE score and
expert backward.  No expert gradient or SGD is claimed here.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.dense_backward_workloads import (
    RmsNormBackwardWorkload, ResidualBackwardWorkload,
)
from ..schema.moe_training_ir0_workloads import NormGammaWgradWorkload
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase,
)


def append_moe_full_train_shared_reverse_ir0(source: IR0) -> IR0:
    """Carry head dHidden through final norm and residual2 into dCombined."""
    source.validate("moe_shared_reverse_source")
    if (source.producer_pass != "moe_full_train_head_backward_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1):
        raise SchemaError("requires exact EP1 MoE head backward source",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    head = nodes.get(f"{instance.id}.lm_head")
    norm = nodes.get(f"{instance.id}.final_norm")
    residual = nodes.get(f"{instance.id}.layer1.residual2")
    combine = nodes.get(f"{instance.id}.layer1.moe.combine")
    if (head is None or norm is None or residual is None or combine is None
            or head.kind is not OpKind.GEMM
            or norm.kind is not OpKind.NORM
            or residual.kind is not OpKind.ELEMENTWISE
            or combine.kind is not OpKind.MOE_COMBINE
            or head.inputs[0] != norm.outputs[0]
            or norm.inputs[0] != residual.outputs[0]
            or residual.inputs[1] != combine.outputs[0]):
        raise SchemaError("head→final norm→residual2→MoE combine chain drifted",
                          path="source.nodes")
    head_grad = nodes.get(f"backward::{head.id}")
    if (head_grad is None or head_grad.kind is not OpKind.GEMM_INPUT_DX
            or len(head_grad.outputs) != 1
            or values[head_grad.outputs[0]].shape !=
               values[norm.outputs[0]].shape):
        raise SchemaError("real LM-head dHidden producer is required",
                          path="source.nodes")
    weights = tuple(state for state in source.persistent_states
                    if state.identity.tensor_ref == norm.inputs[1])
    if len(weights) != 1 or weights[0].shape != values[norm.inputs[1]].shape:
        raise SchemaError("final norm gamma needs one real parameter state",
                          path="source.persistent_states")
    weight = weights[0]
    rows, hidden = norm.workload.rank_activation_shape
    if (rows, hidden) != values[norm.inputs[0]].shape:
        raise SchemaError("final norm input geometry drifted", path="source")
    gamma_id = f"backward::{norm.id}::{weight.id}"
    norm_dx_id = f"backward::{norm.id}"
    residual_dx_id = f"backward::{residual.id}"
    ids = (gamma_id, norm_dx_id, residual_dx_id)
    if any(ref in nodes for ref in ids):
        raise SchemaError("shared reverse nodes already exist", path="source")
    gamma_value_id = f"{gamma_id}.output"
    norm_value_id = f"{norm_dx_id}.input_gradient"
    skip_id = f"{residual_dx_id}.left_gradient"
    combined_id = f"{residual_dx_id}.dcombined_gradient"
    if {gamma_value_id, norm_value_id, skip_id, combined_id} & values.keys():
        raise SchemaError("shared reverse values already exist", path="source")
    pure = NodeEffects(EffectKind.PURE, None, None)
    upstream = head_grad.outputs[0]
    gamma = LogicalNode(
        gamma_id, instance.id, OpKind.NORM_GAMMA_WGRAD,
        OpPhase.WGRAD, norm.stage, norm.mesh_ref,
        (norm.inputs[0], upstream), (gamma_value_id,),
        NormGammaWgradWorkload(rows, rows, 1, hidden, 0),
        norm.math, pure, "norm_gamma_wgrad_timing",
    )
    norm_dx = LogicalNode(
        norm_dx_id, instance.id, OpKind.RMSNORM_BACKWARD,
        OpPhase.DGRAD, norm.stage, norm.mesh_ref,
        (norm.inputs[0], upstream), (norm_value_id,),
        RmsNormBackwardWorkload(rows, hidden, 1),
        norm.math, pure, "rmsnorm_backward_timing",
    )
    residual_dx = LogicalNode(
        residual_dx_id, instance.id, OpKind.RESIDUAL_BACKWARD,
        OpPhase.DGRAD, residual.stage, residual.mesh_ref,
        (residual.outputs[0], norm_value_id), (skip_id, combined_id),
        ResidualBackwardWorkload(rows, rows, 1, hidden),
        residual.math, pure, "residual_backward_timing",
    )
    new_values = (
        TensorValue(gamma_value_id, values[norm.inputs[1]].shape, DType.FP32,
                    "H_final_norm_gamma_gradient",
                    values[norm.inputs[1]].sharding, gamma_id, (), None),
        TensorValue(norm_value_id, values[norm.inputs[0]].shape, DType.FP16,
                    "MH_final_norm_input_gradient",
                    values[norm.inputs[0]].sharding, norm_dx_id, (), None),
        TensorValue(skip_id, values[residual.inputs[0]].shape, DType.FP16,
                    "MH_layer1_skip_gradient",
                    values[residual.inputs[0]].sharding, residual_dx_id, (), None),
        TensorValue(combined_id, values[combine.outputs[0]].shape, DType.FP16,
                    "MH_layer1_dcombined_gradient",
                    values[combine.outputs[0]].sharding, residual_dx_id, (), None),
    )
    all_nodes = (*source.nodes, gamma, norm_dx, residual_dx)
    all_values = (*source.values, *new_values)
    consumers = {value.id: [] for value in all_values}
    for node in all_nodes:
        for ref in node.inputs:
            consumers[ref].append(node.id)
    final_values = tuple(replace(value, consumers=tuple(consumers[value.id]))
                         for value in all_values)
    edges = tuple(GraphEdge(
        f"{value.id}.edge_to.{consumer}", EdgeKind.DATA,
        value.producer, consumer, value.id,
    ) for value in final_values if value.producer is not None
              for consumer in value.consumers)
    edges += tuple(edge for edge in source.edges
                   if edge.kind is EdgeKind.CONTROL)
    result = IR0.create(
        producer_pass="moe_full_train_shared_reverse_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=edges,
        fusion_candidates=source.fusion_candidates,
        profile=source.profile, train=source.train,
        persistent_states=source.persistent_states,
        state_accesses=source.state_accesses,
    )
    result.validate("moe_full_train_shared_reverse_ir0")
    if not any(value.id == combined_id
               and value.producer == residual_dx_id
               for value in result.values):
        raise SchemaError("shared dCombined physical source lost", path="result")
    return result
