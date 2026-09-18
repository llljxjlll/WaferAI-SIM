"""Append a source-bound CE loss gradient to one MoE training forward step.

This establishes the first reverse edge; it does not claim a complete
backward pass or a physical optimizer update.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.ir0 import (
    CrossEntropyBackwardWorkload, CrossEntropyForwardWorkload,
    CrossEntropyReduction, EdgeKind, EffectKind, GraphEdge, IR0, JobKind,
    LogicalNode, NodeEffects, OpKind, OpPhase,
)
from .moe_full_train_forward_ir0 import FullMoeForwardIr0Phase


def append_moe_full_train_ce_backward_ir0(phase: FullMoeForwardIr0Phase) -> IR0:
    """Give each step's forward CE a distinct incoming dLoss and real dLogits."""
    phase.validate()
    forward = phase.graph
    if (forward.producer_pass != "moe_full_train_forward_replacement_ir0"
            or forward.job is not JobKind.TRAIN
            or len(forward.instances) != 1
            or forward.instances[0].parallel.tp != 1
            or forward.instances[0].parallel.ep not in (1, 2)
            or any(node.phase is not OpPhase.FWD for node in forward.nodes)):
        raise SchemaError("requires exact EP1/EP2 MoE forward phase", path="phase")
    ce_nodes = tuple(node for node in forward.nodes
                     if node.kind is OpKind.CE_FORWARD)
    if len(ce_nodes) != 1:
        raise SchemaError("one real forward CE is required", path="phase.graph.nodes")
    ce = ce_nodes[0]
    values = {value.id: value for value in forward.values}
    logits, labels = (values[ref] for ref in ce.inputs)
    loss = values[ce.outputs[0]]
    if (type(ce.workload) is not CrossEntropyForwardWorkload
            or ce.workload.reduction is not CrossEntropyReduction.NONE
            or logits.dtype is not DType.FP16
            or labels.dtype is not DType.INT32
            or loss.dtype is not DType.FP32
            or loss.producer != ce.id
            or logits.shape != ce.workload.logical_logits_shape
            or labels.shape != ce.workload.logical_label_shape
            or loss.shape != ce.workload.logical_loss_shape):
        raise SchemaError("CE forward tape geometry drifted", path="phase.graph.nodes")
    backward_id = f"{ce.id}_backward"
    dloss_id = f"{ce.instance_id}.loss_gradient"
    dlogits_id = f"{ce.instance_id}.logits_gradient"
    if backward_id in {node.id for node in forward.nodes} or (
            {dloss_id, dlogits_id} & values.keys()):
        raise SchemaError("CE backward IDs already exist", path="phase.graph")
    dloss = TensorValue(
        id=dloss_id, shape=loss.shape, dtype=DType.FP32,
        logical_layout="M_loss_gradient", sharding=labels.sharding,
        producer=None, consumers=(backward_id,), alias_set=None,
    )
    dlogits = TensorValue(
        id=dlogits_id, shape=logits.shape, dtype=DType.FP16,
        logical_layout="MV_logits_gradient", sharding=logits.sharding,
        producer=backward_id, consumers=(), alias_set=None,
    )
    backward = LogicalNode(
        id=backward_id, instance_id=ce.instance_id,
        kind=OpKind.CE_BACKWARD, phase=OpPhase.DGRAD,
        stage=ce.stage, mesh_ref=ce.mesh_ref,
        inputs=(logits.id, labels.id, dloss.id), outputs=(dlogits.id,),
        workload=CrossEntropyBackwardWorkload(
            profile=forward.profile, reduction=CrossEntropyReduction.NONE,
            logical_logits_shape=logits.shape,
            rank_logits_shape=ce.workload.rank_logits_shape,
            logical_label_shape=labels.shape,
            rank_label_shape=ce.workload.rank_label_shape,
            logical_loss_gradient_shape=dloss.shape,
            rank_loss_gradient_shape=ce.workload.rank_loss_shape,
            logical_logits_gradient_shape=dlogits.shape,
            rank_logits_gradient_shape=ce.workload.rank_logits_shape,
            logits_dtype=DType.FP16, label_dtype=DType.INT32,
            loss_gradient_dtype=DType.FP32, logits_gradient_dtype=DType.FP16,
        ),
        math=ce.math, effects=NodeEffects(EffectKind.PURE, None, None),
        impl_ref="cross_entropy_backward",
    )
    result = IR0.create(
        producer_pass="moe_full_train_ce_backward_ir0", job=forward.job,
        instances=forward.instances, nodes=(*forward.nodes, backward),
        values=tuple(replace(value, consumers=(*value.consumers, backward_id))
                     if value.id in (logits.id, labels.id) else value
                     for value in forward.values) + (dloss, dlogits),
        edges=(*forward.edges,
               GraphEdge(f"{logits.id}.edge_to.moe_ce_backward", EdgeKind.DATA,
                         logits.producer, backward_id, logits.id),
               GraphEdge(f"{ce.id}.control_to.moe_ce_backward", EdgeKind.CONTROL,
                         ce.id, backward_id, None)),
        fusion_candidates=forward.fusion_candidates, profile=forward.profile,
        train=forward.train, persistent_states=forward.persistent_states,
        state_accesses=forward.state_accesses,
    )
    result.validate("moe_full_train_ce_backward_ir0")
    return result
