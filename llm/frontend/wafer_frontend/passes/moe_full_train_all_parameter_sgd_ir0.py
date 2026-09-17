"""Bind all 19 EP1 two-layer MoE trainable states to source gradients/SGD."""

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


def _moe_logical_name(weight_ref: str) -> str | None:
    parts = weight_ref.split(".")
    if len(parts) < 5 or parts[0] != "T0" or not parts[1].startswith("layer"):
        return None
    layer = parts[1][5:]
    if layer not in ("0", "1") or parts[2] != "moe":
        return None
    if parts[3] == "router" and parts[4:6] == ["weight", "ep0"]:
        return f"layer.{layer}.router.weight"
    if (len(parts) == 6 and parts[3] == "expert0"
            and parts[4] in ("gate", "up", "down")
            and parts[5] == "weight"):
        return f"layer.{layer}.expert.0.{parts[4]}.weight"
    return None


def append_moe_full_train_all_parameter_sgd_ir0(
    source: IR0, sequence: MoeCompileSequence,
) -> IR0:
    source.validate("moe_all_parameter_sgd_source")
    sequence.validate("moe_all_parameter_sgd_sequence")
    if (source.producer_pass != "moe_full_train_embedding_wgrad_ir0"
            or len(source.instances) != 1
            or source.instances[0].parallel.tp != 1
            or source.instances[0].parallel.ep != 1
            or sequence.materialization.request.optimizer is None
            or sequence.materialization.request.optimizer.kind
               is not WorkloadOptimizerKind.SGD):
        raise SchemaError("requires complete EP1 two-layer gradients and SGD",
                          path="source")
    instance = source.instances[0]
    nodes = {node.id: node for node in source.nodes}
    values = {value.id: value for value in source.values}
    states = tuple(sorted((state for state in source.persistent_states
                           if state.identity.kind is StateKind.TRAINABLE_PARAMETER),
                          key=lambda state: state.identity.tensor_ref or ""))
    if (len(states) != 19
            or len({state.identity.tensor_ref for state in states}) != 19):
        raise SchemaError("full EP1 trainable state set must have 19 unique tensors",
                          path="source.persistent_states")
    gradient_by_state = {}
    for state in states:
        weight_ref = state.identity.tensor_ref
        weight = values.get(weight_ref)
        if weight is None or weight.dtype is not DType.FP16 or weight.shape != state.shape:
            raise SchemaError("SGD weight lacks exact FP16 StateDecl tensor",
                              path=f"source.{weight_ref}")
        matching = tuple(node for node in source.nodes
                         if node.phase is OpPhase.WGRAD
                         and node.id.endswith(f"::{state.id}")
                         and len(node.outputs) == 1)
        if matching:
            if len(matching) != 1 or matching[0].kind not in (
                OpKind.GEMM_WEIGHT_WGRAD,
                OpKind.NORM_GAMMA_WGRAD,
                OpKind.EMBEDDING_TABLE_WGRAD,
            ):
                raise SchemaError("parameter has ambiguous WGRAD producer",
                                  path=f"source.{weight_ref}")
            producer = matching[0]
            gradient_ref = producer.outputs[0]
        else:
            experts = tuple(node for node in source.nodes
                            if node.kind is OpKind.MOE_EXPERT_BACKWARD
                            and weight_ref in node.inputs[1:4])
            if len(experts) != 1:
                raise SchemaError("expert parameter lacks one true backward producer",
                                  path=f"source.{weight_ref}")
            producer = experts[0]
            gradient_ref = producer.outputs[
                producer.inputs[1:4].index(weight_ref) + 1]
        gradient = values[gradient_ref]
        if (gradient.producer != producer.id or gradient.dtype is not DType.FP32
                or gradient.shape != state.shape):
            raise SchemaError("SGD needs producer-owned FP32 full weight gradient",
                              path=f"source.{weight_ref}")
        gradient_by_state[state.id] = (weight, gradient, producer)
    if len({gradient.id for _weight, gradient, _producer
            in gradient_by_state.values()}) != 19:
        raise SchemaError("two trainable states share one gradient value", path="source")
    # EP-owned MoE weights must retain exact P2 step/layer/group version proof.
    for state in states:
        logical = _moe_logical_name(state.identity.tensor_ref or "")
        if logical is None:
            if state.identity.ep_owner_rank is not None:
                raise SchemaError("shared state unexpectedly has EP owner", path=state.id)
            continue
        layer = int(logical.split(".")[1])
        forward_expert = nodes.get(f"{instance.id}.layer{layer}.moe.expert0")
        if forward_expert is None or forward_expert.kind is not OpKind.MOE_EXPERT_FORWARD:
            raise SchemaError("P2 group lacks same-layer forward expert", path=logical)
        unit = next((item for item in sequence.units
                     if (item.step, item.layer) ==
                        (forward_expert.workload.step, layer)), None)
        groups = tuple(group for group in unit.parameter_bindings
                       if logical in group.parameter_refs) if unit else ()
        if (len(groups) != 1 or state.identity.ep_owner_rank != 0
                or groups[0].expert != (0 if ".expert." in logical else None)):
            raise SchemaError("MoE SGD lacks exact P2 owner/group binding",
                              path=f"source.{logical}")
        group = groups[0]
        position = group.parameter_refs.index(logical)
        source_version = next((item.id for item in
            sequence.materialization.logical_graph.state_versions
            if item.logical_name == logical and item.version == unit.step), None)
        if (group.input_parameter_state_refs[position] != source_version
                or not group.output_parameter_state_refs[position]
                or not group.sgd_operation_refs[position]):
            raise SchemaError("MoE SGD P2 version chain differs",
                              path=f"source.{logical}")
    optimizer = sequence.materialization.request.optimizer
    updates, outputs, accesses = [], [], list(source.state_accesses)
    for state in states:
        weight, gradient, producer = gradient_by_state[state.id]
        node_id = f"sgd_update::{weight.id}"
        output_id = f"{node_id}.updated_weight"
        if node_id in nodes or output_id in values:
            raise SchemaError("full parameter SGD already exists", path=weight.id)
        alias = f"trainable:{weight.id}"
        outputs.append(TensorValue(output_id, weight.shape, DType.FP16,
                                   weight.logical_layout, weight.sharding,
                                   node_id, (), alias))
        updates.append(LogicalNode(
            node_id, instance.id, OpKind.OPTIMIZER_UPDATE, OpPhase.UPDATE,
            producer.stage, producer.mesh_ref,
            (weight.id, gradient.id), (output_id,),
            SgdUpdateWorkload(
                weight.shape, state.shape, gradient.shape, state.shape,
                weight.shape, state.shape, prod(state.shape),
                optimizer.learning_rate, 0.0,
                DType.FP16, DType.FP32, DType.FP16,
            ), producer.math,
            NodeEffects(EffectKind.INPLACE, f"{node_id}.effect", alias),
            "sgd_update",
        ))
        accesses.append(StateAccess.create(
            node_ref=node_id, state_ref=state.id,
            mode=StateAccessMode.READ_WRITE, rank=0))
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
        producer_pass="moe_full_train_all_parameter_sgd_ir0",
        job=source.job, instances=source.instances, nodes=all_nodes,
        values=final_values, edges=(*data_edges, *controls),
        fusion_candidates=source.fusion_candidates, profile=source.profile,
        train=source.train, persistent_states=source.persistent_states,
        state_accesses=tuple(accesses),
    )
    result.validate("moe_full_train_all_parameter_sgd_ir0")
    return result


__all__ = ["append_moe_full_train_all_parameter_sgd_ir0"]
