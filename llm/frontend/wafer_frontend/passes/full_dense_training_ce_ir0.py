"""Derive the first genuine backward source node from a complete Dense forward IR0.

This is a CE-only source graph.  Full-model backward, IR1/projection/schedule,
and a runtime receipt are separate requirements; this graph cannot satisfy them.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.experiment import ExperimentSpec
from ..schema.ir0 import (
    CrossEntropyBackwardWorkload,
    CrossEntropyForwardWorkload,
    CrossEntropyReduction,
    EdgeKind,
    EffectKind,
    GraphEdge,
    IR0,
    JobKind,
    LogicalNode,
    NodeEffects,
    OpKind,
    OpPhase,
)
from .train_forward import build_train_forward_ir0


def append_dense_training_ce_backward_source(forward: IR0) -> IR0:
    """Keep the forward loss and independent incoming dLoss as distinct values.

    The source supports rank-row-partitioned TP CE with a replicated vocabulary
    and unreduced per-row loss.  Returning an IR0 does not promise that the
    existing S2-Lite training validator, planner,
    or physical lowering accepts a complete multi-layer backward graph.
    """

    if type(forward) is not IR0:
        raise SchemaError("requires source IR0", path="forward")
    forward.validate("forward")
    if (
        forward.job is not JobKind.TRAIN
        or forward.producer_pass != "train_forward_expand"
        or len(forward.instances) != 1
        or any(node.phase is not OpPhase.FWD for node in forward.nodes)
    ):
        raise SchemaError("requires complete forward-only Dense train graph", path="forward")
    ce_nodes = tuple(node for node in forward.nodes if node.kind is OpKind.CE_FORWARD)
    if len(ce_nodes) != 1:
        raise SchemaError("requires exactly one forward CE", path="forward.nodes")
    ce = ce_nodes[0]
    if type(ce.workload) is not CrossEntropyForwardWorkload:
        raise SchemaError("CE forward workload is not exact", path="forward.nodes")
    values_by_id = {value.id: value for value in forward.values}
    logits, labels = (values_by_id[ref] for ref in ce.inputs)
    forward_loss = values_by_id[ce.outputs[0]]
    if (
        logits.dtype is not DType.FP16
        or labels.dtype is not DType.INT32
        or forward_loss.dtype is not DType.FP32
        or forward_loss.producer != ce.id
        or logits.shape != ce.workload.logical_logits_shape
        or labels.shape != ce.workload.logical_label_shape
        or forward_loss.shape != ce.workload.logical_loss_shape
        or ce.workload.reduction is not CrossEntropyReduction.NONE
    ):
        raise SchemaError("forward CE tensor contract drifted", path="forward.nodes")
    backward_id = f"{ce.id}_backward"
    loss_gradient_id = f"{ce.instance_id}.loss_gradient"
    logits_gradient_id = f"{ce.instance_id}.logits_gradient"
    if backward_id in {node.id for node in forward.nodes} or {
        loss_gradient_id, logits_gradient_id
    } & values_by_id.keys():
        raise SchemaError("backward source IDs already exist", path="forward")
    incoming_gradient = TensorValue(
        id=loss_gradient_id,
        shape=forward_loss.shape,
        dtype=DType.FP32,
        logical_layout="M_loss_gradient",
        sharding=labels.sharding,
        producer=None,
        consumers=(backward_id,),
        alias_set=None,
    )
    output_gradient = TensorValue(
        id=logits_gradient_id,
        shape=logits.shape,
        dtype=DType.FP16,
        logical_layout="MV_logits_gradient",
        sharding=logits.sharding,
        producer=backward_id,
        consumers=(),
        alias_set=None,
    )
    backward = LogicalNode(
        id=backward_id,
        instance_id=ce.instance_id,
        kind=OpKind.CE_BACKWARD,
        phase=OpPhase.DGRAD,
        stage=ce.stage,
        mesh_ref=ce.mesh_ref,
        inputs=(logits.id, labels.id, incoming_gradient.id),
        outputs=(output_gradient.id,),
        workload=CrossEntropyBackwardWorkload(
            profile=forward.profile,
            reduction=CrossEntropyReduction.NONE,
            logical_logits_shape=logits.shape,
            rank_logits_shape=ce.workload.rank_logits_shape,
            logical_label_shape=labels.shape,
            rank_label_shape=ce.workload.rank_label_shape,
            logical_loss_gradient_shape=incoming_gradient.shape,
            rank_loss_gradient_shape=ce.workload.rank_loss_shape,
            logical_logits_gradient_shape=output_gradient.shape,
            rank_logits_gradient_shape=ce.workload.rank_logits_shape,
            logits_dtype=DType.FP16,
            label_dtype=DType.INT32,
            loss_gradient_dtype=DType.FP32,
            logits_gradient_dtype=DType.FP16,
        ),
        math=ce.math,
        effects=NodeEffects(EffectKind.PURE, None, None),
        impl_ref="cross_entropy_backward",
    )
    original_values = tuple(
        replace(value, consumers=(*value.consumers, backward_id))
        if value.id in (logits.id, labels.id)
        else value
        for value in forward.values
    )
    edges = (
        *forward.edges,
        GraphEdge(
            id=f"{logits.id}.edge_to.cross_entropy_backward",
            kind=EdgeKind.DATA,
            source_node=logits.producer,
            destination_node=backward_id,
            value_id=logits.id,
        ),
        GraphEdge(
            id=f"{ce.id}.control_to.cross_entropy_backward",
            kind=EdgeKind.CONTROL,
            source_node=ce.id,
            destination_node=backward_id,
            value_id=None,
        ),
    )
    result = IR0.create(
        producer_pass="dense_training_ce_backward_source",
        job=forward.job,
        instances=forward.instances,
        nodes=(*forward.nodes, backward),
        values=(*original_values, incoming_gradient, output_gradient),
        edges=edges,
        fusion_candidates=forward.fusion_candidates,
        profile=forward.profile,
        train=forward.train,
        persistent_states=forward.persistent_states,
        state_accesses=forward.state_accesses,
    )
    result.validate("dense_training_ce_backward_source")
    return result


def build_dense_training_ce_backward_source(spec: ExperimentSpec) -> IR0:
    """Bind CE source to the validated production Dense forward experiment."""

    return append_dense_training_ce_backward_source(build_train_forward_ir0(spec))
