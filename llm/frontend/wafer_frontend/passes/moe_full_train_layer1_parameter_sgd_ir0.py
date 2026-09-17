"""Bind layer1 EP1 router and three expert gradients to their P2 SGD states.

This is a structural state-update source.  It does not claim numeric gradient
production from timing-only WGRAD primitives or a two-step native invocation.
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


def append_moe_full_train_layer1_parameter_sgd_ir0(
    source: IR0, sequence: MoeCompileSequence,
) -> IR0:
    source.validate("moe_layer1_parameter_sgd_source")
    sequence.validate("moe_layer1_parameter_sgd_sequence")
    if (source.producer_pass != "moe_full_train_layer1_backbone_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1
            or sequence.materialization.request.optimizer is None
            or sequence.materialization.request.optimizer.kind
               is not WorkloadOptimizerKind.SGD):
        raise SchemaError("requires source-bound EP1 layer1 FP32 gradients and P2 SGD",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    router = nodes.get(f"{instance.id}.layer1.moe.router")
    expert = nodes.get(f"{instance.id}.layer1.moe.expert0")
    expert_dx = nodes.get(f"backward::{expert.id}") if expert else None
    router_wgrad = next((node for node in source.nodes
                         if node.kind is OpKind.GEMM_WEIGHT_WGRAD
                         and node.workload.source_forward_op_ref == router.id),
                        None) if router else None
    if (router is None or router.kind is not OpKind.MOE_ROUTER
            or expert is None or expert.kind is not OpKind.MOE_EXPERT_FORWARD
            or expert_dx is None
            or expert_dx.kind is not OpKind.MOE_EXPERT_BACKWARD
            or router_wgrad is None
            or expert_dx.workload.source_forward_op_ref != expert.id
            or expert.workload.expert_count != 1
            or expert.workload.expert != 0
            or router.workload.step != expert.workload.step
            or router.workload.layer != expert.workload.layer
            or expert_dx.workload.step != expert.workload.step
            or expert_dx.workload.layer != expert.workload.layer):
        raise SchemaError("four same-layer router/expert gradient producers absent",
                          path="source.nodes")
    unit = next((unit for unit in sequence.units
                 if (unit.step, unit.layer) ==
                    (expert.workload.step, expert.workload.layer)), None)
    if unit is None:
        raise SchemaError("same-step P2 parameter groups absent",
                          path="sequence.units")
    targets = (
        (router.inputs[1], router_wgrad.outputs[0],
         "layer.1.router.weight", None, router, router_wgrad),
        *((expert.inputs[index + 1], expert_dx.outputs[index + 1],
           f"layer.1.expert.0.{name}.weight", 0, expert, expert_dx)
          for index, name in enumerate(("gate", "up", "down"))),
    )
    optimizer = sequence.materialization.request.optimizer
    updates, outputs, accesses = [], [], list(source.state_accesses)
    states = {state.identity.tensor_ref: state
              for state in source.persistent_states
              if state.identity.kind is StateKind.TRAINABLE_PARAMETER}
    for weight_ref, gradient_ref, logical_name, owner, forward, producer in targets:
        weight, gradient = values[weight_ref], values[gradient_ref]
        state = states.get(weight_ref)
        group = next((item for item in unit.parameter_bindings
                      if logical_name in item.parameter_refs), None)
        position = (group.parameter_refs.index(logical_name)
                    if group is not None else -1)
        if (state is None or group is None
                or group.expert != owner
                or state.identity.ep_owner_rank != 0
                or state.shape != weight.shape
                or group.input_parameter_state_refs[position] !=
                   next((item.id for item in sequence.materialization.logical_graph.state_versions
                         if item.logical_name == logical_name
                         and item.version == expert.workload.step), None)
                or len(group.output_parameter_state_refs) <= position
                or len(group.sgd_operation_refs) <= position
                or not group.output_parameter_state_refs[position]
                or not group.sgd_operation_refs[position]
                or weight.dtype is not DType.FP16
                or gradient.dtype is not DType.FP32
                or gradient.shape != weight.shape
                or gradient.producer != producer.id
                or weight.id not in forward.inputs):
            raise SchemaError("P2 owner/state/FP32 gradient source differs",
                              path=f"source.{logical_name}")
        node_id = f"sgd_update::{weight.id}"
        updated_id = f"{node_id}.updated_weight"
        if node_id in nodes or updated_id in values:
            raise SchemaError("layer1 parameter SGD already exists",
                              path=f"source.{logical_name}")
        alias = f"trainable:{weight.id}"
        outputs.append(TensorValue(
            updated_id, weight.shape, DType.FP16, weight.logical_layout,
            weight.sharding, node_id, (), alias,
        ))
        updates.append(LogicalNode(
            node_id, instance.id, OpKind.OPTIMIZER_UPDATE, OpPhase.UPDATE,
            forward.stage, forward.mesh_ref,
            (weight.id, gradient.id), (updated_id,),
            SgdUpdateWorkload(
                weight.shape, state.shape, gradient.shape, state.shape,
                weight.shape, state.shape, prod(state.shape),
                optimizer.learning_rate, 0.0,
                DType.FP16, DType.FP32, DType.FP16,
            ),
            producer.math, NodeEffects(EffectKind.INPLACE,
                                       f"{node_id}.effect", alias),
            "sgd_update",
        ))
        accesses.append(StateAccess.create(
            node_ref=node_id, state_ref=state.id,
            mode=StateAccessMode.READ_WRITE, rank=0,
        ))
    all_nodes = (*source.nodes, *updates)
    all_values = (*source.values, *outputs)
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
        producer_pass="moe_full_train_layer1_parameter_sgd_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=tuple(accesses),
    )
    result.validate("moe_full_train_layer1_parameter_sgd_ir0")
    return result


__all__ = ["append_moe_full_train_layer1_parameter_sgd_ir0"]
