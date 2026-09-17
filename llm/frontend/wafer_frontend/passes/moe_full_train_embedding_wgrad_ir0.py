"""Bind the final EP1 MoE embedding table WGRAD to layer0 input dX."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, StateAccess, StateAccessMode,
)
from ..schema.moe_training_ir0_workloads import EmbeddingTableWgradWorkload


def append_moe_full_train_embedding_wgrad_ir0(source: IR0) -> IR0:
    source.validate("moe_embedding_wgrad_source")
    if (source.producer_pass != "moe_full_train_layer0_qkv_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1):
        raise SchemaError("requires complete EP1 two-layer input dX source",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    embedding = nodes.get(f"{instance.id}.embedding")
    input_dx = nodes.get(f"backward::{instance.id}.layer0.norm1.merge_layer0")
    if (embedding is None or embedding.kind is not OpKind.EMBEDDING
            or input_dx is None or input_dx.kind is not OpKind.ELEMENTWISE
            or len(embedding.inputs) != 2 or len(input_dx.outputs) != 1
            or embedding.outputs[0] != nodes[f"{instance.id}.layer0.residual1"].inputs[0]):
        raise SchemaError("embedding/layer0 input source chain drifted",
                          path="source.nodes")
    table = values[embedding.inputs[1]]
    upstream = values[input_dx.outputs[0]]
    rows, hidden = upstream.shape
    if (rows < 1 or rows > 16 or table.shape[1] != hidden
            or upstream.dtype is not DType.FP16
            or values[embedding.inputs[0]].dtype is not DType.INT32):
        raise SchemaError("embedding WGRAD token/table geometry drifted",
                          path="source.values")
    states = tuple(state for state in source.persistent_states
                   if state.identity.tensor_ref == table.id)
    if len(states) != 1 or states[0].shape != table.shape:
        raise SchemaError("embedding table has no unique source StateDecl",
                          path="source.persistent_states")
    state = states[0]
    node_id = f"backward::{embedding.id}::{state.id}"
    output_id = f"{node_id}.output"
    if node_id in nodes or output_id in values:
        raise SchemaError("embedding WGRAD already exists", path="source")
    node = LogicalNode(
        node_id, instance.id, OpKind.EMBEDDING_TABLE_WGRAD, OpPhase.WGRAD,
        embedding.stage, embedding.mesh_ref,
        (*embedding.inputs, upstream.id), (output_id,),
        EmbeddingTableWgradWorkload(
            rows, rows, 1, state.shape[0], 0, state.shape[0], hidden,
            tuple(range(rows)) + (0,) * (16 - rows)),
        embedding.math, NodeEffects(EffectKind.PURE, None, None),
        "embedding_table_wgrad_timing",
    )
    output = TensorValue(output_id, table.shape, DType.FP32,
                         "VH_embedding_table_gradient", table.sharding,
                         node_id, (), None)
    all_nodes = (*source.nodes, node)
    all_values = (*source.values, output)
    consumers = {value.id: [] for value in all_values}
    for item in all_nodes:
        for ref in item.inputs:
            consumers[ref].append(item.id)
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
        producer_pass="moe_full_train_embedding_wgrad_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=(*source.state_accesses,
                        StateAccess.create(node_ref=node.id, state_ref=state.id,
                                           mode=StateAccessMode.READ, rank=0)),
    )
    result.validate("moe_full_train_embedding_wgrad_ir0")
    return result


__all__ = ["append_moe_full_train_embedding_wgrad_ir0"]
