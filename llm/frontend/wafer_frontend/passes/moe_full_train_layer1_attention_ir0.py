"""Continue EP1 MoE layer1 residual1 dX through output projection and attention."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.dense_backward_workloads import (
    AttentionBackwardWorkload, ResidualBackwardWorkload,
)
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, StateAccess, StateAccessMode,
)


def _append_moe_full_train_layer_attention_ir0(
    source: IR0, *, layer: int, expected_source_pass: str, producer_pass: str,
) -> IR0:
    source.validate(f"moe_layer{layer}_attention_source")
    if (source.producer_pass != expected_source_pass
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1):
        raise SchemaError("requires exact EP1 residual1 upstream source",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    prefix = f"{instance.id}.layer{layer}"
    residual = nodes.get(f"{prefix}.residual1")
    merge = nodes.get(f"backward::{residual.id}.merge") if residual else None
    projection = nodes.get(f"{prefix}.o")
    attention = nodes.get(f"{prefix}.attention")
    if (residual is None or residual.kind is not OpKind.ELEMENTWISE
            or merge is None or merge.kind is not OpKind.ELEMENTWISE
            or projection is None or projection.kind is not OpKind.GEMM
            or attention is None or attention.kind is not OpKind.ATTENTION
            or projection.outputs[0] != residual.inputs[1]
            or attention.outputs[0] != projection.inputs[0]
            or merge.outputs[0] not in values
            or values[merge.outputs[0]].producer != merge.id
            or residual.stage != projection.stage
            or projection.stage != attention.stage
            or residual.mesh_ref != projection.mesh_ref
            or projection.mesh_ref != attention.mesh_ref):
        raise SchemaError(f"layer{layer} attention/output/residual source drifted",
                          path="source.nodes")
    upstream = values[merge.outputs[0]]
    skip = values[residual.inputs[0]]
    out = values[projection.outputs[0]]
    attn_out = values[attention.outputs[0]]
    packed = values[attention.inputs[0]]
    weight = values[projection.inputs[1]]
    rows, hidden = skip.shape
    if (upstream.shape != skip.shape or out.shape != skip.shape
            or attn_out.shape != skip.shape
            or upstream.dtype is not DType.FP16
            or packed.dtype is not DType.FP16
            or len(weight.shape) != 2
            or weight.shape != (hidden, hidden)
            or attention.workload.query_tokens != rows
            or attention.workload.hidden_size != hidden
            or attention.workload.rank_num_heads *
               attention.workload.head_dim != hidden):
        raise SchemaError(f"layer{layer} causal attention/output projection geometry drifted",
                          path="source.values")
    declarations = tuple(state for state in source.persistent_states
                         if state.identity.tensor_ref == weight.id)
    if len(declarations) != 1 or declarations[0].shape != weight.shape:
        raise SchemaError(f"layer{layer} output weight needs one owned StateDecl",
                          path="source.persistent_states")
    state = declarations[0]
    residual_id = f"backward::{residual.id}"
    wgrad_id = f"backward::{projection.id}::{state.id}"
    projection_dx_id = f"backward::{projection.id}"
    attention_dx_id = f"backward::{attention.id}"
    ids = (residual_id, wgrad_id, projection_dx_id, attention_dx_id)
    if any(ref in nodes for ref in ids):
        raise SchemaError(f"layer{layer} attention reverse already exists",
                          path="source")
    skip_id = f"{residual_id}.left_gradient"
    output_id = f"{residual_id}.right_gradient"
    wgrad_value_id = f"{wgrad_id}.output"
    projection_dx_value_id = f"{projection_dx_id}.input_gradient"
    attention_dx_value_id = f"{attention_dx_id}.input_gradient"
    if any(ref in values for ref in (skip_id, output_id, wgrad_value_id,
                                     projection_dx_value_id,
                                     attention_dx_value_id)):
        raise SchemaError(f"layer{layer} attention reverse values already exist",
                          path="source")
    pure = NodeEffects(EffectKind.PURE, None, None)
    residual_dx = LogicalNode(
        residual_id, instance.id, OpKind.RESIDUAL_BACKWARD,
        OpPhase.DGRAD, residual.stage, residual.mesh_ref,
        (residual.outputs[0], upstream.id), (skip_id, output_id),
        ResidualBackwardWorkload(rows, rows, 1, hidden),
        residual.math, pure, "residual_backward_timing",
    )
    wgrad = LogicalNode(
        wgrad_id, instance.id, OpKind.GEMM_WEIGHT_WGRAD,
        OpPhase.WGRAD, projection.stage, projection.mesh_ref,
        (attn_out.id, output_id), (wgrad_value_id,),
        GemmWeightWgradWorkload(
            m=hidden, n=hidden, k=rows,
            source_forward_op_ref=projection.id,
            source_parameter_state_ref=state.id,
        ),
        projection.math, pure, "gemm_weight_wgrad_timing",
    )
    projection_dx = LogicalNode(
        projection_dx_id, instance.id, OpKind.GEMM_INPUT_DX,
        OpPhase.DGRAD, projection.stage, projection.mesh_ref,
        (weight.id, output_id), (projection_dx_value_id,),
        GemmInputDxWorkload(
            k=rows, m=hidden, n=hidden,
            source_forward_op_ref=projection.id,
            source_parameter_state_ref=state.id,
        ),
        projection.math, pure, "gemm_input_dx_timing",
    )
    workload = attention.workload
    attention_dx = LogicalNode(
        attention_dx_id, instance.id, OpKind.ATTENTION_BACKWARD,
        OpPhase.DGRAD, attention.stage, attention.mesh_ref,
        (packed.id, projection_dx_value_id), (attention_dx_value_id,),
        AttentionBackwardWorkload(
            rows, workload.rank_num_heads, workload.rank_num_kv_heads,
            workload.head_dim, 1, workload.profile.num_seqs,
            workload.query_key_pairs,
        ),
        attention.math, pure, "attention_backward_timing",
    )
    new_values = (
        TensorValue(skip_id, skip.shape, DType.FP16,
                    f"MH_layer{layer}_residual1_skip_gradient", skip.sharding,
                    residual_id, (), None),
        TensorValue(output_id, out.shape, DType.FP16,
                    f"MH_layer{layer}_attention_output_gradient", out.sharding,
                    residual_id, (), None),
        TensorValue(wgrad_value_id, weight.shape, DType.FP32,
                    f"HH_layer{layer}_attention_output_weight_gradient",
                    weight.sharding, wgrad_id, (), None),
        TensorValue(projection_dx_value_id, attn_out.shape, DType.FP16,
                    f"MH_layer{layer}_attention_gradient", attn_out.sharding,
                    projection_dx_id, (), None),
        TensorValue(attention_dx_value_id, packed.shape, DType.FP16,
                    f"MQKV_layer{layer}_attention_input_gradient", packed.sharding,
                    attention_dx_id, (), None),
    )
    all_nodes = (*source.nodes, residual_dx, wgrad, projection_dx,
                 attention_dx)
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
        producer_pass=producer_pass,
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=(*source.state_accesses,
                        StateAccess.create(node_ref=projection_dx.id,
                                           state_ref=state.id,
                                           mode=StateAccessMode.READ, rank=0)),
    )
    result.validate(producer_pass)
    return result


def append_moe_full_train_layer1_attention_ir0(source: IR0) -> IR0:
    return _append_moe_full_train_layer_attention_ir0(
        source, layer=1,
        expected_source_pass="moe_full_train_layer1_backbone_ir0",
        producer_pass="moe_full_train_layer1_attention_ir0",
    )


def append_moe_full_train_layer0_attention_ir0(source: IR0) -> IR0:
    return _append_moe_full_train_layer_attention_ir0(
        source, layer=0,
        expected_source_pass="moe_full_train_layer0_moe_ir0",
        producer_pass="moe_full_train_layer0_attention_ir0",
    )


__all__ = ["append_moe_full_train_layer1_attention_ir0",
           "append_moe_full_train_layer0_attention_ir0"]
