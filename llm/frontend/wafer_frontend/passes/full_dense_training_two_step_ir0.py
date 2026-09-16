"""Expand two physical SGD iterations from the exact Dense source graph."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import TensorValue
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.ir0 import EdgeKind, GraphEdge, IR0, OpKind, StateAccess
from .full_dense_training_sgd_ir0 import build_full_dense_training_sgd_ir0


def build_full_dense_training_two_step_ir0(plan: FlexibleDenseTrainPlan) -> IR0:
    """Return two source-complete iterations sharing one trainable HBM inventory.

    A parameter tensor remains an external value in each iteration: the HBM
    READ for step 1 is ordered after the step 0 SGD WRITE by state control.
    This IR0 does not itself assert that the linked program has replayed those
    state-version edges; that is checked by the physical merger.
    """
    one = build_full_dense_training_sgd_ir0(plan)
    shared = {state.identity.tensor_ref for state in one.persistent_states}
    shards = {(state.identity.tensor_ref, state.identity.shard_index)
              for state in one.persistent_states}
    if None in shared or len(shards) != len(one.persistent_states):
        raise SchemaError("source parameter tensor/TP shard identities must be unique",
                          path="one.persistent_states")
    nodes = []
    values: dict[str, TensorValue] = {}
    accesses = []
    controls = []
    for step in range(2):
        def node_ref(ref: str) -> str:
            if ref.startswith("backward::"):
                return f"backward::{ref.removeprefix('backward::')}::step{step}"
            if ref.startswith("gradient_sum::"):
                return f"gradient_sum::{ref.removeprefix('gradient_sum::')}::step{step}"
            if ref.endswith("_backward"):
                return f"{ref.removesuffix('_backward')}::step{step}_backward"
            return f"{ref}::step{step}"

        def value_ref(ref: str) -> str:
            return ref if ref in shared else f"{ref}::step{step}"

        for value in one.values:
            if value.id in shared and step == 1:
                continue
            values[value_ref(value.id)] = replace(
                value, id=value_ref(value.id),
                producer=None if value.producer is None else node_ref(value.producer),
                consumers=(),
            )
        for node in one.nodes:
            workload = node.workload
            if type(workload) in (GemmInputDxWorkload,
                                  GemmWeightWgradWorkload):
                workload = replace(workload,
                    source_forward_op_ref=node_ref(workload.source_forward_op_ref))
            nodes.append(replace(
                node, id=node_ref(node.id),
                inputs=tuple(value_ref(ref) for ref in node.inputs),
                outputs=tuple(value_ref(ref) for ref in node.outputs),
                workload=workload,
                effects=replace(node.effects,
                    effect_token=None if node.effects.effect_token is None
                    else f"{node_ref(node.id)}.effect"),
            ))
        for access in one.state_accesses:
            accesses.append(StateAccess.create(
                node_ref=node_ref(access.node_ref),
                state_ref=access.state_ref, mode=access.mode, rank=access.rank,
                read_offset=access.read_offset, read_shape=access.read_shape,
                write_offset=access.write_offset, write_shape=access.write_shape,
            ))
        for edge in one.edges:
            if edge.kind is EdgeKind.CONTROL:
                controls.append(GraphEdge(
                    f"{edge.id}::step{step}", EdgeKind.CONTROL,
                    node_ref(edge.source_node), node_ref(edge.destination_node), None,
                ))
    by_state_update = {
        (state.identity.tensor_ref, state.identity.shard_index): next(
            node.id for node in one.nodes
            if node.kind is OpKind.OPTIMIZER_UPDATE
            and node.inputs[0] == state.identity.tensor_ref
            and node.id == (
                f"sgd_update::{state.identity.tensor_ref}"
                f"::tp{state.identity.shard_index}"
            )
        )
        for state in one.persistent_states
    }
    for node in one.nodes:
        if node.kind not in (OpKind.GEMM, OpKind.NORM, OpKind.EMBEDDING):
            continue
        if len(node.inputs) != 2 or node.inputs[1] not in shared:
            continue
        weight = node.inputs[1]
        for rank in range(plan.spec.tp_degree):
            update_ref = by_state_update[(weight, rank)]
            controls.append(GraphEdge(
                f"{update_ref}.store0_to.{node.id}.load1",
                EdgeKind.CONTROL,
                f"{update_ref}::step0", f"{node.id}::step1", None,
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
        producer_pass="full_dense_training_two_step_source",
        job=one.job, instances=one.instances, nodes=tuple(nodes),
        values=final_values, edges=(*data, *controls),
        fusion_candidates=(), profile=one.profile, train=one.train,
        persistent_states=one.persistent_states,
        state_accesses=tuple(accesses),
    )
    result.validate("full_dense_training_two_step_source")
    # Canonical V1 candidates survive only when the intermediate has no
    # backward tape reader.  Discovery runs on the completed two-step graph.
    from .discover_fusion import with_discovered_fusion_candidates
    return with_discovered_fusion_candidates(result)


__all__ = ["build_full_dense_training_two_step_ir0"]
