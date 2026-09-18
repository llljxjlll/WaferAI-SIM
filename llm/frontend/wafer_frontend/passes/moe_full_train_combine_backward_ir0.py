"""Bind layer1 combine reverse to real shared dCombined and forward tape.

EP1 can lower to native 0x28. EP2 has distinct per-expert gradients in IR0;
its multi-expert native lowering and cross-die return remain separate work.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase,
)
from ..schema.moe_combine_backward_workload import MoeCombineBackwardWorkload


def append_moe_full_train_combine_backward_ir0(source: IR0) -> IR0:
    source.validate("moe_combine_backward_source")
    if (source.producer_pass != "moe_full_train_shared_reverse_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep not in (1, 2)):
        raise SchemaError("requires actual EP1/EP2 shared dCombined graph",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    combine = nodes.get(f"{instance.id}.layer1.moe.combine")
    residual_dx = nodes.get(f"backward::{instance.id}.layer1.residual2")
    if (combine is None or combine.kind is not OpKind.MOE_COMBINE
            or residual_dx is None
            or residual_dx.kind is not OpKind.RESIDUAL_BACKWARD
            or len(residual_dx.outputs) != 2
            or values[residual_dx.outputs[1]].shape !=
               values[combine.outputs[0]].shape):
        raise SchemaError("layer1 combine/dCombined source is absent",
                          path="source.nodes")
    forward = combine.workload
    m, h, e = (forward.token_count, forward.hidden_size,
               forward.expert_count)
    if e not in (1, 2) or instance.parallel.ep != e:
        raise SchemaError("combine reverse requires matching EP1/EP2 source",
                          path="source.nodes")
    backward_id = f"backward::{combine.id}"
    score_id = f"{backward_id}.dscore"
    expert_ids = tuple(
        f"{backward_id}.dexpert{index}" if e > 1
        else f"{backward_id}.dexpert"
        for index in range(e)
    )
    if backward_id in nodes or any(ref in values for ref in (score_id, *expert_ids)):
        raise SchemaError("combine backward source already exists",
                          path="source")
    route_ref, score_ref = combine.inputs[-2:]
    expert_refs = combine.inputs[:-2]
    dcombined_ref = residual_dx.outputs[1]
    score = values[score_ref]
    score_grad = TensorValue(
        score_id, score.shape, DType.FP16, "ME_router_score_gradient",
        score.sharding, backward_id, (), None,
    )
    expert_grads = tuple(TensorValue(
        gradient_id, values[expert_ref].shape, DType.FP16,
        "MH_expert_return_gradient", values[expert_ref].sharding,
        backward_id, (), None,
    ) for gradient_id, expert_ref in zip(expert_ids, expert_refs, strict=True))
    backward = LogicalNode(
        backward_id, instance.id, OpKind.MOE_COMBINE_BACKWARD,
        OpPhase.DGRAD, combine.stage, combine.mesh_ref,
        (route_ref, score_ref, *expert_refs, dcombined_ref),
        (score_id, *expert_ids),
        MoeCombineBackwardWorkload(
            source_forward_op_ref=combine.id,
            source_route_trace_digest=forward.source_route_trace_digest,
            step=forward.step, layer=forward.layer,
            token_count=m, hidden_size=h, expert_count=e,
            route_bytes=20*m,
        ),
        combine.math, NodeEffects(EffectKind.PURE, None, None),
        "moe_combine_backward",
    )
    all_nodes = (*source.nodes, backward)
    all_values = (*source.values, score_grad, *expert_grads)
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
        producer_pass="moe_full_train_combine_backward_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=source.state_accesses,
    )
    result.validate("moe_full_train_combine_backward_ir0")
    return result
