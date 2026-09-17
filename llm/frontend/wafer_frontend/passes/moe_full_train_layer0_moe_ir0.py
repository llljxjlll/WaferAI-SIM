"""Continue EP1 two-step source-bound layer0 MoE and norm2 reverse."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.dense_backward_workloads import (
    ResidualBackwardWorkload, RmsNormBackwardWorkload,
)
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.moe_combine_backward_workload import MoeCombineBackwardWorkload
from ..schema.moe_expert_backward_workload import MoeExpertBackwardWorkload
from ..schema.moe_training_ir0_workloads import NormGammaWgradWorkload
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, ResidualWorkload, StateAccess, StateAccessMode,
)


def append_moe_full_train_layer0_moe_ir0(source: IR0) -> IR0:
    source.validate("moe_layer0_moe_source")
    if (source.producer_pass != "moe_full_train_layer1_qkv_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1):
        raise SchemaError("requires exact EP1 layer1-to-layer0 gradient source",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    prefix = f"{instance.id}.layer0"
    residual = nodes.get(f"{prefix}.residual2")
    combine = nodes.get(f"{prefix}.moe.combine")
    expert = nodes.get(f"{prefix}.moe.expert0")
    router = nodes.get(f"{prefix}.moe.router")
    dispatch = nodes.get(f"{prefix}.moe.dispatch")
    norm = nodes.get(f"{prefix}.norm2")
    residual1 = nodes.get(f"{prefix}.residual1")
    layer1_merge = nodes.get(f"backward::{instance.id}.layer1.norm1.merge_layer0")
    layer1_residual = nodes.get(f"{instance.id}.layer1.residual1")
    layer1_residual_dx = nodes.get(f"backward::{instance.id}.layer1.residual1")
    if (residual is None or residual.kind is not OpKind.ELEMENTWISE
            or combine is None or combine.kind is not OpKind.MOE_COMBINE
            or expert is None or expert.kind is not OpKind.MOE_EXPERT_FORWARD
            or router is None or router.kind is not OpKind.MOE_ROUTER
            or dispatch is None or dispatch.kind is not OpKind.MOE_DISPATCH
            or norm is None or norm.kind is not OpKind.NORM
            or residual1 is None or residual1.kind is not OpKind.ELEMENTWISE
            or layer1_merge is None or layer1_merge.kind is not OpKind.ELEMENTWISE
            or layer1_residual is None or layer1_residual.kind is not OpKind.ELEMENTWISE
            or layer1_residual_dx is None or layer1_residual_dx.kind is not OpKind.RESIDUAL_BACKWARD
            or layer1_residual.inputs[0] != residual.outputs[0]
            or layer1_merge.inputs[0] != layer1_residual_dx.outputs[0]
            or residual.inputs != (residual1.outputs[0], combine.outputs[0])
            or combine.inputs[0] != expert.outputs[0]
            or combine.inputs[1] != dispatch.inputs[1]
            or combine.inputs[2] != router.outputs[0]
            or expert.inputs[0] != dispatch.outputs[0]
            or dispatch.inputs[0] != norm.outputs[0]
            or router.inputs[0] != norm.outputs[0]
            or norm.inputs[0] != residual1.outputs[0]):
        raise SchemaError("layer0 residual/MoE/norm2 source drifted",
                          path="source.nodes")
    m, h, e = (combine.workload.token_count,
               combine.workload.hidden_size, combine.workload.expert_count)
    i = expert.workload.intermediate_size
    if (e != 1 or expert.workload.expert_count != 1
            or expert.workload.expert != 0
            or expert.workload.owned_token_count != m
            or router.workload.token_count != m
            or router.workload.hidden_size != h
            or router.workload.expert_count != e
            or dispatch.workload.expert_count != 1
            or dispatch.workload.frozen_expert_by_token != (0,) * m
            or dispatch.workload.frozen_slot_by_token != tuple(range(m))
            or len({combine.workload.source_route_trace_digest,
                    expert.workload.source_route_trace_digest,
                    dispatch.workload.source_route_trace_digest}) != 1
            or combine.workload.step != expert.workload.step
            or combine.workload.layer != expert.workload.layer
            or values[layer1_merge.outputs[0]].shape != (m, h)):
        raise SchemaError("layer0 EP1 identity route/tape source drifted",
                          path="source.nodes")
    state_by_tensor = {}
    for state in source.persistent_states:
        state_by_tensor.setdefault(state.identity.tensor_ref, []).append(state)
    def owned_state(ref: str, shape: tuple[int, ...], *, expert_owned: bool = False):
        found = state_by_tensor.get(ref, ())
        if (len(found) != 1 or found[0].shape != shape
                or (expert_owned and found[0].identity.ep_owner_rank != 0)):
            raise SchemaError("layer0 parameter lacks unique EP1 StateDecl",
                              path=ref)
        return found[0]
    gate = owned_state(router.inputs[1], (h, 1), expert_owned=True)
    gamma = owned_state(norm.inputs[1], (h,))
    expert_states = (
        owned_state(expert.inputs[1], (h, i), expert_owned=True),
        owned_state(expert.inputs[2], (h, i), expert_owned=True),
        owned_state(expert.inputs[3], (i, h), expert_owned=True),
    )
    rid = f"backward::{residual.id}"
    cid = f"backward::{combine.id}"
    wid = f"backward::{router.id}::{gate.id}"
    eid = f"backward::{expert.id}"
    did = f"backward::{router.id}.input"
    sid = f"backward::{prefix}.moe.input_sum"
    gid = f"backward::{norm.id}::{gamma.id}"
    nid = f"backward::{norm.id}"
    mid = f"backward::{residual1.id}.merge"
    ids = (rid, cid, wid, eid, did, sid, gid, nid, mid)
    out = (
        f"{rid}.left_gradient", f"{rid}.dcombined_gradient",
        f"{cid}.dscore", f"{cid}.dexpert",
        f"{wid}.weight_gradient",
        f"{eid}.activation_gradient", f"{eid}.gate_weight_gradient",
        f"{eid}.up_weight_gradient", f"{eid}.down_weight_gradient",
        f"{did}.gradient", f"{sid}.norm2_gradient", f"{gid}.output",
        f"{nid}.input_gradient", f"{mid}.input_gradient",
    )
    if any(ref in nodes for ref in ids) or any(ref in values for ref in out):
        raise SchemaError("layer0 MoE reverse already exists", path="source")
    pure = NodeEffects(EffectKind.PURE, None, None)
    reverse_nodes = (
        LogicalNode(rid, instance.id, OpKind.RESIDUAL_BACKWARD, OpPhase.DGRAD,
                    residual.stage, residual.mesh_ref,
                    (residual.outputs[0], layer1_merge.outputs[0]), out[0:2],
                    ResidualBackwardWorkload(m, m, 1, h),
                    residual.math, pure, "residual_backward_timing"),
        LogicalNode(cid, instance.id, OpKind.MOE_COMBINE_BACKWARD, OpPhase.DGRAD,
                    combine.stage, combine.mesh_ref,
                    (*combine.inputs[1:], combine.inputs[0], out[1]), out[2:4],
                    MoeCombineBackwardWorkload(
                        source_forward_op_ref=combine.id,
                        source_route_trace_digest=combine.workload.source_route_trace_digest,
                        step=combine.workload.step, layer=combine.workload.layer,
                        token_count=m, hidden_size=h, expert_count=1,
                        route_bytes=20*m),
                    combine.math, pure, "moe_combine_backward"),
        LogicalNode(wid, instance.id, OpKind.GEMM_WEIGHT_WGRAD, OpPhase.WGRAD,
                    router.stage, router.mesh_ref,
                    (router.inputs[0], out[2]), (out[4],),
                    GemmWeightWgradWorkload(h, 1, m, router.id, gate.id),
                    router.math, pure, "gemm_weight_wgrad_timing"),
        LogicalNode(eid, instance.id, OpKind.MOE_EXPERT_BACKWARD, OpPhase.DGRAD,
                    expert.stage, expert.mesh_ref,
                    (*expert.inputs, out[3]), out[5:9],
                    MoeExpertBackwardWorkload(
                        source_forward_op_ref=expert.id,
                        source_combine_backward_op_ref=cid,
                        source_route_trace_digest=expert.workload.source_route_trace_digest,
                        step=expert.workload.step, layer=expert.workload.layer,
                        expert=0, token_count=m, hidden_size=h,
                        intermediate_size=i, expert_count=1),
                    expert.math, pure, "moe_expert_backward_recompute"),
        LogicalNode(did, instance.id, OpKind.GEMM_INPUT_DX, OpPhase.DGRAD,
                    router.stage, router.mesh_ref,
                    (router.inputs[1], out[2]), (out[9],),
                    GemmInputDxWorkload(m, h, 1, router.id, gate.id),
                    router.math, pure, "gemm_input_dx_timing"),
        LogicalNode(sid, instance.id, OpKind.ELEMENTWISE, OpPhase.DGRAD,
                    norm.stage, norm.mesh_ref,
                    (out[5], out[9]), (out[10],),
                    ResidualWorkload((m,h), (m,h), DType.FP16),
                    norm.math, pure, "residual"),
        LogicalNode(gid, instance.id, OpKind.NORM_GAMMA_WGRAD, OpPhase.WGRAD,
                    norm.stage, norm.mesh_ref,
                    (norm.inputs[0], out[10]), (out[11],),
                    NormGammaWgradWorkload(m,m,1,h,0),
                    norm.math, pure, "norm_gamma_wgrad_timing"),
        LogicalNode(nid, instance.id, OpKind.RMSNORM_BACKWARD, OpPhase.DGRAD,
                    norm.stage, norm.mesh_ref,
                    (norm.inputs[0], out[10]), (out[12],),
                    RmsNormBackwardWorkload(m,h,1),
                    norm.math, pure, "rmsnorm_backward_timing"),
        LogicalNode(mid, instance.id, OpKind.ELEMENTWISE, OpPhase.DGRAD,
                    residual1.stage, residual1.mesh_ref,
                    (out[0], out[12]), (out[13],),
                    ResidualWorkload((m,h), (m,h), DType.FP16),
                    residual1.math, pure, "residual"),
    )
    shapes = ((m,h),(m,h),(m,1),(m,h),(h,1),(m,h),
              (h,i),(h,i),(i,h),(m,h),(m,h),(h,),(m,h),(m,h))
    dtypes = ((DType.FP16,)*4 + (DType.FP32,) + (DType.FP16,)
              + (DType.FP32,)*3 + (DType.FP16,)*2 + (DType.FP32,)
              + (DType.FP16,)*2)
    sharding_refs = (residual.inputs[0], combine.outputs[0],
                     router.outputs[0], expert.outputs[0], router.inputs[1],
                     dispatch.outputs[0], *expert.inputs[1:4],
                     norm.outputs[0], norm.outputs[0], norm.inputs[1],
                     norm.inputs[0], residual1.outputs[0])
    owners = (rid,rid,cid,cid,wid,eid,eid,eid,eid,did,sid,gid,nid,mid)
    new_values = tuple(TensorValue(
        ref, shape, dtype, f"layer0_{ref.rsplit('.',1)[-1]}",
        values[shard_ref].sharding, owner, (), None,
    ) for ref, shape, dtype, shard_ref, owner in zip(
        out, shapes, dtypes, sharding_refs, owners, strict=True))
    all_nodes = (*source.nodes, *reverse_nodes)
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
        producer_pass="moe_full_train_layer0_moe_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=(*source.state_accesses,
                        StateAccess.create(node_ref=did, state_ref=gate.id,
                                           mode=StateAccessMode.READ, rank=0)),
    )
    result.validate("moe_full_train_layer0_moe_ir0")
    return result


__all__ = ["append_moe_full_train_layer0_moe_ir0"]
