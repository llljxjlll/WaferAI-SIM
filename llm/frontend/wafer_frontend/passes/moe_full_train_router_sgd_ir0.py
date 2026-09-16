"""One genuine EP1 router gate SGD state write from native 0x28→0x25.

This partial optimizer step cannot stand for complete MoE training.
"""

from __future__ import annotations

from dataclasses import replace
from math import prod

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, SgdUpdateWorkload, StateAccess, StateAccessMode,
)
from ..schema.moe_compile_sequence import MoeCompileSequence
from ..schema.persistent_state import StateKind
from ..schema.workload_run import WorkloadOptimizerKind


def append_moe_full_train_router_sgd_ir0(
    source: IR0, sequence: MoeCompileSequence,
) -> IR0:
    source.validate("moe_router_sgd_source")
    sequence.validate("moe_router_sgd_sequence")
    if (source.producer_pass != "moe_full_train_router_wgrad_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.ep != 1
            or source.instances[0].parallel.tp != 1
            or sequence.materialization.request.optimizer is None
            or sequence.materialization.request.optimizer.kind
               is not WorkloadOptimizerKind.SGD):
        raise SchemaError("requires real EP1 router FP32 gradient and P2 SGD",
                          path="source")
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    instance = source.instances[0]
    router = nodes[f"{instance.id}.layer1.moe.router"]
    wgrad = source.nodes[-1]
    if (wgrad.kind is not OpKind.GEMM_WEIGHT_WGRAD
            or wgrad.workload.source_forward_op_ref != router.id):
        raise SchemaError("native 0x25 router WGRAD absent", path="source")
    weight = values[router.inputs[1]]
    gradient = values[wgrad.outputs[0]]
    state = next((item for item in source.persistent_states
                  if item.id == wgrad.workload.source_parameter_state_ref), None)
    if (state is None or state.identity.tensor_ref != weight.id
            or state.identity.kind is not StateKind.TRAINABLE_PARAMETER
            or weight.dtype is not DType.FP16
            or gradient.dtype is not DType.FP32
            or weight.shape != gradient.shape):
        raise SchemaError("gate state and FP32 gradient shape differ",
                          path="source")
    unit = next((unit for unit in sequence.units
                 if (unit.step, unit.layer) ==
                    (router.workload.step, router.workload.layer)), None)
    group = next((item for item in unit.parameter_bindings
                  if item.expert is None
                  and item.parameter_refs == ("layer.1.router.weight",)
                  and weight.id.endswith(".router.weight.ep0")
                  and len(item.input_parameter_state_refs) == 1
                  and len(item.output_parameter_state_refs) == 1
                  and len(item.sgd_operation_refs) == 1),
                 None) if unit is not None else None
    if group is None:
        raise SchemaError("same-step P2 gate parameter group absent",
                          path="sequence.units")
    node_id = f"sgd_update::{weight.id}"
    updated_id = f"{node_id}.updated_weight"
    if node_id in nodes or updated_id in values:
        raise SchemaError("router SGD already exists", path="source")
    alias = f"trainable:{weight.id}"
    updated = TensorValue(
        updated_id, weight.shape, DType.FP16, weight.logical_layout,
        weight.sharding, node_id, (), alias,
    )
    optimizer = sequence.materialization.request.optimizer
    update = LogicalNode(
        node_id, instance.id, OpKind.OPTIMIZER_UPDATE, OpPhase.UPDATE,
        router.stage, router.mesh_ref, (weight.id, gradient.id),
        (updated.id,),
        SgdUpdateWorkload(
            weight.shape, state.shape, gradient.shape, state.shape,
            updated.shape, state.shape, prod(state.shape),
            optimizer.learning_rate, 0.0,
            DType.FP16, DType.FP32, DType.FP16,
        ),
        wgrad.math, NodeEffects(EffectKind.INPLACE,
                                f"{node_id}.effect", alias),
        "sgd_update",
    )
    all_nodes = (*source.nodes, update)
    all_values = (*source.values, updated)
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
    accesses = tuple(sorted((
        *source.state_accesses,
        StateAccess.create(node_ref=node_id, state_ref=state.id,
                           mode=StateAccessMode.READ_WRITE, rank=0),
    ), key=lambda item: (item.node_ref, item.state_ref, item.rank, item.id)))
    result = IR0.create(
        producer_pass="moe_full_train_router_sgd_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=accesses,
    )
    result.validate("moe_full_train_router_sgd_ir0")
    return result
