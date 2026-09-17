"""Connect EP1 layer1 attention dX through RoPE/QKV/norm1 to layer0."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, Sharding, TensorValue
from ..schema.dense_backward_workloads import (
    RmsNormBackwardWorkload, RopeBackwardWorkload,
)
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.moe_training_ir0_workloads import NormGammaWgradWorkload
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, ResidualWorkload, StateAccess, StateAccessMode,
)


def append_moe_full_train_layer1_qkv_ir0(source: IR0) -> IR0:
    source.validate("moe_layer1_qkv_source")
    if (source.producer_pass != "moe_full_train_layer1_attention_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1):
        raise SchemaError("requires exact EP1 layer1 attention reverse source",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    prefix = f"{instance.id}.layer1"
    attention_dx = nodes.get(f"backward::{prefix}.attention")
    rope = nodes.get(f"{prefix}.rope")
    qkv = nodes.get(f"{prefix}.qkv")
    norm = nodes.get(f"{prefix}.norm1")
    residual = nodes.get(f"{prefix}.residual1")
    residual_dx = nodes.get(f"backward::{prefix}.residual1")
    if (attention_dx is None or attention_dx.kind is not OpKind.ATTENTION_BACKWARD
            or rope is None or rope.kind is not OpKind.ROPE
            or qkv is None or qkv.kind is not OpKind.GEMM
            or norm is None or norm.kind is not OpKind.NORM
            or residual is None or residual.kind is not OpKind.ELEMENTWISE
            or residual_dx is None or residual_dx.kind is not OpKind.RESIDUAL_BACKWARD
            or attention_dx.inputs[0] != rope.outputs[0]
            or rope.inputs[0] != qkv.outputs[0]
            or qkv.inputs[0] != norm.outputs[0]
            or norm.inputs[0] != residual.inputs[0]
            or residual_dx.outputs[0] not in values):
        raise SchemaError("layer1 RoPE/QKV/norm1 source drifted", path="source.nodes")
    r = rope.workload
    a = attention_dx.workload
    rows = a.tokens
    hidden = a.rank_heads * a.head_dim
    packed = (a.rank_heads + 2 * a.rank_kv_heads) * a.head_dim
    if (r.rank_input_shape != (rows, packed)
            or r.rank_output_shape != (rows, packed)
            or values[qkv.outputs[0]].shape != (rows, packed)
            or values[norm.inputs[0]].shape != (rows, hidden)
            or values[qkv.inputs[1]].shape != (hidden, packed)
            or qkv.workload.rank_shape != (rows, packed, hidden)
            or r.rotary_dim != r.head_dim
            or r.head_dim != a.head_dim
            or r.rank_num_heads != a.rank_heads
            or r.rank_num_kv_heads != a.rank_kv_heads):
        raise SchemaError("layer1 RoPE/QKV rank extent drifted", path="source.values")
    qkv_states = tuple(state for state in source.persistent_states
                       if state.identity.tensor_ref == qkv.inputs[1])
    norm_states = tuple(state for state in source.persistent_states
                        if state.identity.tensor_ref == norm.inputs[1])
    if (len(qkv_states) != 1 or qkv_states[0].shape != (hidden, packed)
            or len(norm_states) != 1 or norm_states[0].shape != (hidden,)):
        raise SchemaError("layer1 QKV/norm1 weights need unique owned states",
                          path="source.persistent_states")
    qstate, nstate = qkv_states[0], norm_states[0]
    ids = (
        f"backward::{rope.id}",
        f"backward::{qkv.id}::{qstate.id}",
        f"backward::{qkv.id}",
        f"backward::{norm.id}::{nstate.id}",
        f"backward::{norm.id}",
        f"backward::{norm.id}.merge_layer0",
    )
    outputs = (
        f"{ids[0]}.input_gradient",
        f"{ids[1]}.output",
        f"{ids[2]}.input_gradient",
        f"{ids[3]}.output",
        f"{ids[4]}.input_gradient",
        f"{ids[5]}.input_gradient",
    )
    position_id = f"{rope.id}.position_ids"
    if any(ref in nodes for ref in ids) or any(ref in values for ref in (*outputs, position_id)):
        raise SchemaError("layer1 QKV reverse already exists", path="source")
    pure = NodeEffects(EffectKind.PURE, None, None)
    rope_dx = LogicalNode(
        ids[0], instance.id, OpKind.ROPE_BACKWARD, OpPhase.DGRAD,
        rope.stage, rope.mesh_ref,
        (position_id, attention_dx.outputs[0]), (outputs[0],),
        RopeBackwardWorkload(
            r.profile.prefill_tokens, rows, r.num_heads, r.num_kv_heads,
            r.rank_num_heads, r.rank_num_kv_heads, 1, r.head_dim,
            r.rotary_dim, r.max_position_embeddings,
        ), rope.math, pure, "rope_backward_timing",
    )
    wgrad = LogicalNode(
        ids[1], instance.id, OpKind.GEMM_WEIGHT_WGRAD, OpPhase.WGRAD,
        qkv.stage, qkv.mesh_ref,
        (qkv.inputs[0], outputs[0]), (outputs[1],),
        GemmWeightWgradWorkload(hidden, packed, rows, qkv.id, qstate.id),
        qkv.math, pure, "gemm_weight_wgrad_timing",
    )
    qkv_dx = LogicalNode(
        ids[2], instance.id, OpKind.GEMM_INPUT_DX, OpPhase.DGRAD,
        qkv.stage, qkv.mesh_ref,
        (qkv.inputs[1], outputs[0]), (outputs[2],),
        GemmInputDxWorkload(rows, hidden, packed, qkv.id, qstate.id),
        qkv.math, pure, "gemm_input_dx_timing",
    )
    gamma = LogicalNode(
        ids[3], instance.id, OpKind.NORM_GAMMA_WGRAD, OpPhase.WGRAD,
        norm.stage, norm.mesh_ref,
        (norm.inputs[0], outputs[2]), (outputs[3],),
        NormGammaWgradWorkload(rows, rows, 1, hidden, 0),
        norm.math, pure, "norm_gamma_wgrad_timing",
    )
    norm_dx = LogicalNode(
        ids[4], instance.id, OpKind.RMSNORM_BACKWARD, OpPhase.DGRAD,
        norm.stage, norm.mesh_ref,
        (norm.inputs[0], outputs[2]), (outputs[4],),
        RmsNormBackwardWorkload(rows, hidden, 1),
        norm.math, pure, "rmsnorm_backward_timing",
    )
    merge = LogicalNode(
        ids[5], instance.id, OpKind.ELEMENTWISE, OpPhase.DGRAD,
        norm.stage, norm.mesh_ref,
        (residual_dx.outputs[0], outputs[4]), (outputs[5],),
        ResidualWorkload((rows, hidden), (rows, hidden), DType.FP16),
        norm.math, pure, "residual",
    )
    source_values = (values[qkv.inputs[0]], values[norm.inputs[0]],
                     values[qkv.inputs[1]], values[norm.inputs[1]])
    new_values = (
        TensorValue(position_id, (rows,), DType.INT32, "M_layer1_position_ids",
                    Sharding(rope.mesh_ref, (None,), ()), None, (), None),
        TensorValue(outputs[0], (rows, packed), DType.FP16,
                    "MQKV_layer1_rope_gradient", values[qkv.outputs[0]].sharding,
                    ids[0], (), None),
        TensorValue(outputs[1], (hidden, packed), DType.FP32,
                    "HQKV_layer1_weight_gradient", source_values[2].sharding,
                    ids[1], (), None),
        TensorValue(outputs[2], (rows, hidden), DType.FP16,
                    "MH_layer1_norm1_gradient", source_values[0].sharding,
                    ids[2], (), None),
        TensorValue(outputs[3], (hidden,), DType.FP32,
                    "H_layer1_norm1_gamma_gradient", source_values[3].sharding,
                    ids[3], (), None),
        TensorValue(outputs[4], (rows, hidden), DType.FP16,
                    "MH_layer1_norm1_input_gradient", source_values[1].sharding,
                    ids[4], (), None),
        TensorValue(outputs[5], (rows, hidden), DType.FP16,
                    "MH_layer0_output_gradient", source_values[1].sharding,
                    ids[5], (), None),
    )
    all_nodes = (*source.nodes, rope_dx, wgrad, qkv_dx, gamma, norm_dx, merge)
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
        producer_pass="moe_full_train_layer1_qkv_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=(*source.state_accesses,
                        StateAccess.create(node_ref=qkv_dx.id,
                                           state_ref=qstate.id,
                                           mode=StateAccessMode.READ, rank=0)),
    )
    result.validate("moe_full_train_layer1_qkv_ir0")
    return result


__all__ = ["append_moe_full_train_layer1_qkv_ir0"]
