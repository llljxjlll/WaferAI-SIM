"""Connect a real LM-head backward source to a two-layer MoE CE source.

The FP32 weight gradient and FP16 hidden gradient have separate nodes and
separate output tensors.  This source transform does not claim physical
lowering, an optimizer step, or a complete model backward pass.
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.ir0 import (
    EdgeKind,
    EffectKind,
    GemmPartition,
    GemmWorkload,
    GraphEdge,
    IR0,
    JobKind,
    LogicalNode,
    NodeEffects,
    OpKind,
    OpPhase,
    StateAccess,
    StateAccessMode,
)


def append_moe_full_train_head_backward_ir0(source: IR0) -> IR0:
    """Differentiate LM head against CE dLogits and the saved final-norm value.

    For a replicated TP1 GEMM ``hidden[M,H] × weight[H,V]``, the two
    derivatives are ``dWeight[H,V]`` in FP32 and ``dHidden[M,H]`` in FP16.
    Its sole parameter source must be a real persistent StateDecl.
    """

    if type(source) is not IR0:
        raise SchemaError("requires source IR0", path="source")
    source.validate("source")
    if (
        source.producer_pass != "moe_full_train_ce_backward_ir0"
        or source.job is not JobKind.TRAIN
        or len(source.instances) != 1
        or source.instances[0].parallel.tp != 1
        or source.instances[0].parallel.ep != 1
        or len(source.nodes) < 3
    ):
        raise SchemaError("requires exact TP1/EP1 MoE CE backward source", path="source")
    instance = source.instances[0]
    node_by_id = {node.id: node for node in source.nodes}
    value_by_id = {value.id: value for value in source.values}
    head = node_by_id.get(f"{instance.id}.lm_head")
    ce = node_by_id.get(f"{instance.id}.cross_entropy")
    if (
        head is None or head.kind is not OpKind.GEMM
        or head.phase is not OpPhase.FWD
        or ce is None or ce.kind is not OpKind.CE_FORWARD
        or len(head.inputs) != 2 or len(head.outputs) != 1
        or len(ce.outputs) != 1
    ):
        raise SchemaError("complete LM head and CE forward are required", path="source.nodes")
    ce_backward = node_by_id.get(f"{ce.id}_backward")
    if (
        ce_backward is None or ce_backward.kind is not OpKind.CE_BACKWARD
        or ce_backward.phase is not OpPhase.DGRAD
        or ce_backward.inputs[2] == ce.outputs[0]
        or ce_backward.inputs[0] != head.outputs[0]
        or len(ce_backward.outputs) != 1
    ):
        raise SchemaError("independently seeded CE dLogits are required", path="source.nodes")
    hidden, weight = (value_by_id[ref] for ref in head.inputs)
    logits = value_by_id[head.outputs[0]]
    dlogits = value_by_id[ce_backward.outputs[0]]
    if (
        type(head.workload) is not GemmWorkload
        or head.workload.partition is not GemmPartition.REPLICATED
        or head.workload.dtype is not DType.FP16
        or hidden.dtype is not DType.FP16
        or weight.dtype is not DType.FP16
        or logits.dtype is not DType.FP16
        or dlogits.dtype is not DType.FP16
        or dlogits.producer != ce_backward.id
        or hidden.producer is None
        or hidden.shape[0] != source.profile.prefill_tokens
        or len(hidden.shape) != 2 or len(weight.shape) != 2
        or weight.shape[0] != hidden.shape[1]
        or logits.shape != (hidden.shape[0], weight.shape[1])
        or dlogits.shape != logits.shape
        or head.workload.logical_shape != (
            hidden.shape[0], weight.shape[1], hidden.shape[1]
        )
    ):
        raise SchemaError("replicated head, dLogits and saved activation geometry drifted", path="source")
    declarations = tuple(state for state in source.persistent_states
                         if state.identity.tensor_ref == weight.id)
    if len(declarations) != 1 or declarations[0].shape != weight.shape:
        raise SchemaError("head weight requires one real parameter StateDecl", path="source.persistent_states")
    state = declarations[0]
    weight_grad_ref = f"backward::{head.id}::{state.id}"
    hidden_grad_ref = f"backward::{head.id}"
    weight_grad_value_ref = f"{head.id}.weight_gradient"
    hidden_grad_value_ref = f"{head.id}.hidden_gradient"
    if (
        {weight_grad_ref, hidden_grad_ref} & node_by_id.keys()
        or {weight_grad_value_ref, hidden_grad_value_ref} & value_by_id.keys()
    ):
        raise SchemaError("LM-head backward source IDs already exist", path="source")
    hidden_grad = TensorValue(
        id=hidden_grad_value_ref, shape=hidden.shape, dtype=DType.FP16,
        logical_layout="MH_hidden_gradient", sharding=hidden.sharding,
        producer=hidden_grad_ref, consumers=(), alias_set=None,
    )
    weight_grad = TensorValue(
        id=weight_grad_value_ref, shape=weight.shape, dtype=DType.FP32,
        logical_layout="HV_weight_gradient", sharding=weight.sharding,
        producer=weight_grad_ref, consumers=(), alias_set=None,
    )
    m, h = hidden.shape
    v = weight.shape[1]
    pure = NodeEffects(EffectKind.PURE, None, None)
    wgrad = LogicalNode(
        id=weight_grad_ref, instance_id=instance.id,
        kind=OpKind.GEMM_WEIGHT_WGRAD, phase=OpPhase.WGRAD, stage=head.stage, mesh_ref=head.mesh_ref,
        inputs=(hidden.id, dlogits.id), outputs=(weight_grad.id,),
        workload=GemmWeightWgradWorkload(
            m=h, n=v, k=m, source_forward_op_ref=head.id,
            source_parameter_state_ref=state.id,
        ),
        math=head.math, effects=pure,
        impl_ref="gemm_weight_wgrad_timing",
    )
    dgrad = LogicalNode(
        id=hidden_grad_ref, instance_id=instance.id, kind=OpKind.GEMM_INPUT_DX,
        phase=OpPhase.DGRAD, stage=head.stage, mesh_ref=head.mesh_ref,
        inputs=(weight.id, dlogits.id), outputs=(hidden_grad.id,),
        workload=GemmInputDxWorkload(
            k=m, m=h, n=v, source_forward_op_ref=head.id,
            source_parameter_state_ref=state.id,
        ),
        math=head.math, effects=pure, impl_ref="gemm_input_dx_timing",
    )
    changed = {hidden.id: weight_grad_ref, weight.id: hidden_grad_ref}
    values = tuple(
        replace(value, consumers=(*value.consumers, changed[value.id]))
        if value.id in changed else
        replace(value, consumers=(weight_grad_ref, hidden_grad_ref))
        if value.id == dlogits.id else value
        for value in source.values
    )
    result = IR0.create(
        producer_pass="moe_full_train_head_backward_ir0", job=source.job,
        instances=source.instances, nodes=(*source.nodes, wgrad, dgrad),
        values=(*values, weight_grad, hidden_grad),
        edges=(
            *source.edges,
            GraphEdge(f"{hidden.id}.edge_to.lm_head_wgrad", EdgeKind.DATA,
                      hidden.producer, wgrad.id, hidden.id),
            GraphEdge(f"{dlogits.id}.edge_to.lm_head_wgrad", EdgeKind.DATA,
                      ce_backward.id, wgrad.id, dlogits.id),
            GraphEdge(f"{dlogits.id}.edge_to.lm_head_dgrad", EdgeKind.DATA,
                      ce_backward.id, dgrad.id, dlogits.id),
        ),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=(
            *source.state_accesses,
            StateAccess.create(node_ref=dgrad.id, state_ref=state.id,
                               mode=StateAccessMode.READ, rank=0),
        ),
    )
    result.validate("moe_full_train_head_backward_ir0")
    return result


__all__ = ["append_moe_full_train_head_backward_ir0"]
