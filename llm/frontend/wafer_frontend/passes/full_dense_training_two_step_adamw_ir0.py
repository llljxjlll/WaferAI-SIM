"""Two authentic Dense AdamW steps sharing persistent parameter and optimizer states."""
from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import TensorValue
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.ir0 import (
    AdamwUpdateWorkload, EdgeKind, GraphEdge, IR0, OpKind, StateAccess,
    StateAccessMode,
)
from ..schema.persistent_state import StateKind
from .full_dense_training_adamw_ir0 import build_full_dense_training_adamw_ir0


def build_full_dense_training_two_step_adamw_ir0(
    plan: FlexibleDenseTrainPlan,
) -> IR0:
    """Return two complete source iterations with exact 75-state version edges.

    Persistent input values remain external in both iterations.  Their real
    LOAD/UPDATE/STORE timing and host residency require later N5/N6 gates.
    """
    one = build_full_dense_training_adamw_ir0(plan)
    shared = {state.identity.tensor_ref for state in one.persistent_states}
    if None in shared or len(shared) != 75:
        raise SchemaError("AdamW needs 75 unique persistent source tensors",
                          path="one.persistent_states")
    trainable = {
        state.id: state.identity.tensor_ref
        for state in one.persistent_states
        if state.identity.kind is StateKind.TRAINABLE_PARAMETER
    }
    if len(trainable) != 15:
        raise SchemaError("AdamW needs fifteen authentic trainable weights",
                          path="one.persistent_states")
    update_by_state = {}
    for update in one.nodes:
        if update.kind is not OpKind.OPTIMIZER_UPDATE:
            continue
        for access in one.state_accesses:
            if access.node_ref == update.id:
                if access.state_ref in update_by_state:
                    raise SchemaError("AdamW persistent state has multiple updates",
                                      path=access.state_ref)
                update_by_state[access.state_ref] = update.id
    if len(update_by_state) != 75:
        raise SchemaError("AdamW source leaves a persistent state without update",
                          path="one.state_accesses")

    def node_ref(ref: str, step: int) -> str:
        if ref.startswith("backward::"):
            return f"backward::{ref.removeprefix('backward::')}::step{step}"
        if ref.startswith("gradient_sum::"):
            return f"gradient_sum::{ref.removeprefix('gradient_sum::')}::step{step}"
        if ref.endswith("_backward"):
            return f"{ref.removesuffix('_backward')}::step{step}_backward"
        return f"{ref}::step{step}"

    def value_ref(ref: str, step: int) -> str:
        return ref if ref in shared else f"{ref}::step{step}"

    nodes = []
    values: dict[str, TensorValue] = {}
    accesses = []
    controls = []
    for step in range(2):
        for value in one.values:
            if value.id in shared and step == 1:
                continue
            values[value_ref(value.id, step)] = replace(
                value, id=value_ref(value.id, step),
                producer=None if value.producer is None
                         else node_ref(value.producer, step),
                consumers=(),
            )
        for node in one.nodes:
            workload = node.workload
            if type(workload) in (GemmInputDxWorkload,
                                  GemmWeightWgradWorkload):
                workload = replace(
                    workload,
                    source_forward_op_ref=node_ref(
                        workload.source_forward_op_ref, step),
                )
            elif type(workload) is AdamwUpdateWorkload:
                workload = replace(workload, step=step + 1)
            nodes.append(replace(
                node, id=node_ref(node.id, step),
                inputs=tuple(value_ref(ref, step) for ref in node.inputs),
                outputs=tuple(value_ref(ref, step) for ref in node.outputs),
                workload=workload,
                effects=replace(node.effects,
                    effect_token=None if node.effects.effect_token is None
                    else f"{node_ref(node.id, step)}.effect"),
            ))
        for access in one.state_accesses:
            accesses.append(StateAccess.create(
                node_ref=node_ref(access.node_ref, step),
                state_ref=access.state_ref, mode=access.mode, rank=access.rank,
                read_offset=access.read_offset, read_shape=access.read_shape,
                write_offset=access.write_offset, write_shape=access.write_shape,
            ))
        for edge in one.edges:
            if edge.kind is EdgeKind.CONTROL:
                controls.append(GraphEdge(
                    f"{edge.id}::step{step}", EdgeKind.CONTROL,
                    node_ref(edge.source_node, step),
                    node_ref(edge.destination_node, step), None,
                ))
    for access in one.state_accesses:
        if access.mode is not StateAccessMode.READ:
            continue
        update = update_by_state[access.state_ref]
        reader = access.node_ref
        controls.append(GraphEdge(
            f"{update}.store0_to.{reader}.load1", EdgeKind.CONTROL,
            node_ref(update, 0), node_ref(reader, 1), None,
        ))
    for update in one.nodes:
        if update.kind is OpKind.OPTIMIZER_UPDATE:
            controls.append(GraphEdge(
                f"{update.id}.store0_to.update1", EdgeKind.CONTROL,
                node_ref(update.id, 0), node_ref(update.id, 1), None,
            ))
    consumers = {ref: [] for ref in values}
    for node in nodes:
        for ref in node.inputs:
            consumers[ref].append(node.id)
    final_values = tuple(replace(value, consumers=tuple(consumers[value.id]))
                         for value in values.values())
    data = tuple(GraphEdge(
        f"{value.id}.edge_to.{consumer}", EdgeKind.DATA,
        value.producer, consumer, value.id,
    ) for value in final_values if value.producer is not None
      for consumer in value.consumers)
    result = IR0.create(
        producer_pass="full_dense_training_two_step_adamw_source",
        job=one.job, instances=one.instances, nodes=tuple(nodes),
        values=final_values, edges=(*data, *controls),
        fusion_candidates=(), profile=one.profile, train=one.train,
        persistent_states=one.persistent_states, state_accesses=tuple(accesses),
    )
    result.validate("full_dense_training_two_step_adamw_source")
    from .validate_ir0 import DenseIR0Validator
    DenseIR0Validator.validate(result, "full_dense_training_two_step_adamw_source")
    return result


__all__ = ["build_full_dense_training_two_step_adamw_ir0"]
