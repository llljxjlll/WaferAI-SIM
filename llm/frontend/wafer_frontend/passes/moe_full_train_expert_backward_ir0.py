"""Bind each expert reverse to its own combine dExpert and forward weights.

The source carries recompute and FP32 gate/up/down gradients. Native lowering
still accepts only EP1 until EP2 BufferABI and cross-die transport are wired.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase,
)
from ..schema.moe_expert_backward_workload import MoeExpertBackwardWorkload


def append_moe_full_train_expert_backward_ir0(source: IR0) -> IR0:
    source.validate("moe_expert_backward_source")
    if (len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or (source.instances[0].parallel.ep,
                source.producer_pass) not in (
                    (1, "moe_full_train_router_wgrad_ir0"),
                    (2, "moe_full_train_combine_backward_ir0"))):
        raise SchemaError("requires real EP1 router WGRAD or EP2 combine reverse",
                          path="source")
    instance = source.instances[0]
    expert_count = instance.parallel.ep
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    combine = nodes.get(f"backward::{instance.id}.layer1.moe.combine")
    if (combine is None or combine.kind is not OpKind.MOE_COMBINE_BACKWARD
            or combine.workload.expert_count != expert_count
            or len(combine.outputs) != expert_count + 1):
        raise SchemaError("same-layer combine backward tape absent",
                          path="source.nodes")
    new_nodes = []
    new_values = []
    for expert in range(expert_count):
        forward = nodes.get(f"{instance.id}.layer1.moe.expert{expert}")
        if (forward is None or forward.kind is not OpKind.MOE_EXPERT_FORWARD
                or forward.outputs[0] != combine.inputs[2 + expert]
                or forward.workload.expert_count != expert_count
                or forward.workload.expert != expert
                or forward.workload.source_route_trace_digest !=
                   combine.workload.source_route_trace_digest):
            raise SchemaError("same-layer expert forward/return tape absent",
                              path="source.nodes")
        m, h, i = (forward.workload.owned_token_count,
                   forward.workload.hidden_size,
                   forward.workload.intermediate_size)
        gradient = values[combine.outputs[1 + expert]]
        if (gradient.shape != (m, h) or gradient.dtype is not DType.FP16
                or gradient.producer != combine.id):
            raise SchemaError("dExpert shape/source differs from expert return",
                              path="source.nodes")
        node_id = f"backward::{forward.id}"
        output_ids = tuple(f"{node_id}.{suffix}" for suffix in (
            "activation_gradient", "gate_weight_gradient",
            "up_weight_gradient", "down_weight_gradient",
        ))
        if node_id in nodes or any(ref in values for ref in output_ids):
            raise SchemaError("expert reverse already exists", path="source")
        output_specs = (
            ((m, h), DType.FP16, "MH_expert_activation_gradient",
             forward.inputs[0]),
            ((h, i), DType.FP32, "HI_expert_gate_gradient",
             forward.inputs[1]),
            ((h, i), DType.FP32, "HI_expert_up_gradient",
             forward.inputs[2]),
            ((i, h), DType.FP32, "IH_expert_down_gradient",
             forward.inputs[3]),
        )
        new_values.extend(TensorValue(
            ref, shape, dtype, layout, values[source_ref].sharding,
            node_id, (), None,
        ) for ref, (shape, dtype, layout, source_ref)
            in zip(output_ids, output_specs, strict=True))
        new_nodes.append(LogicalNode(
            node_id, instance.id, OpKind.MOE_EXPERT_BACKWARD,
            OpPhase.DGRAD, forward.stage, forward.mesh_ref,
            (*forward.inputs, gradient.id), output_ids,
            MoeExpertBackwardWorkload(
                source_forward_op_ref=forward.id,
                source_combine_backward_op_ref=combine.id,
                source_route_trace_digest=
                    forward.workload.source_route_trace_digest,
                step=forward.workload.step, layer=forward.workload.layer,
                expert=expert, token_count=m, hidden_size=h,
                intermediate_size=i, expert_count=expert_count,
            ),
            forward.math, NodeEffects(EffectKind.PURE, None, None),
            "moe_expert_backward_recompute",
        ))
    all_nodes = (*source.nodes, *new_nodes)
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
        producer_pass="moe_full_train_expert_backward_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=source.state_accesses,
    )
    result.validate("moe_full_train_expert_backward_ir0")
    return result


__all__ = ["append_moe_full_train_expert_backward_ir0"]
