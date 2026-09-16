"""Attach actual FP32 WGRAD -> FP16 SGD writes to a Dense source step.

The resulting IR0 still needs the independent source and physical gates; it
must never be used as a surrogate for a two-step runtime receipt.
"""

from __future__ import annotations

from dataclasses import replace
from math import prod

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.ir0 import (
    EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode, NodeEffects,
    OpKind, OpPhase, SgdUpdateWorkload, StateAccess, StateAccessMode,
)
from ..schema.persistent_state import (
    PersistentStateAccess, PersistentStateDecl, PersistentStateIdentity,
    PersistentStateLifetime, StateKind,
)
from .full_dense_training_backward_ir0 import build_full_dense_training_backward_ir0


def build_full_dense_training_sgd_ir0(plan: FlexibleDenseTrainPlan) -> IR0:
    """Build all 15 trainable source parameters and independently bound SGD updates.

    The runtime version is intentionally established by the later STORE -> LOAD
    physical dependency, not by reusing a forward output as a parameter input.
    """
    source = build_full_dense_training_backward_ir0(plan)
    old_states = {state.id: state for state in source.persistent_states}
    states: dict[str, PersistentStateDecl] = {}
    for old in old_states.values():
        identity = PersistentStateIdentity.create(
            kind=StateKind.TRAINABLE_PARAMETER,
            instance_ref=old.identity.instance_ref,
            mesh_ref=old.identity.mesh_ref,
            request_ref=None, layer_index=None,
            tensor_ref=old.identity.tensor_ref,
            shard_index=old.identity.shard_index,
            generation=old.identity.generation,
        )
        states[old.id] = PersistentStateDecl.create(
            identity=identity, shape=old.shape, dtype=old.dtype,
            layout=old.layout, lifetime=PersistentStateLifetime.PERSISTENT,
            access=PersistentStateAccess.READ_WRITE,
        )
    values = {value.id: value for value in source.values}
    nodes = []
    for node in source.nodes:
        if (node.kind in (OpKind.GEMM_INPUT_DX, OpKind.GEMM_WEIGHT_WGRAD)
                and type(node.workload) in (GemmInputDxWorkload,
                                             GemmWeightWgradWorkload)):
            old = node.workload.source_parameter_state_ref
            node = replace(node, workload=replace(
                node.workload, source_parameter_state_ref=states[old].id,
            ))
        nodes.append(node)
    accesses = [StateAccess.create(
        node_ref=access.node_ref, state_ref=states[access.state_ref].id,
        mode=access.mode, rank=access.rank,
        read_offset=access.read_offset, read_shape=access.read_shape,
        write_offset=access.write_offset, write_shape=access.write_shape,
    ) for access in source.state_accesses]
    edges = list(source.edges)
    for template in plan.parameter_templates:
        old = old_states[template.state_ref]
        state = states[old.id]
        weight_ref = old.identity.tensor_ref
        assert weight_ref is not None
        weight = values[weight_ref]
        gradient_ref = f"{template.wgrad_ref}.output"
        gradient = values[gradient_ref]
        if (weight.dtype is not DType.FP16
                or gradient.dtype is not DType.FP32
                or weight.shape != gradient.shape
                or weight.producer is not None):
            raise SchemaError("SGD source needs one exact FP16 parameter and FP32 derivative",
                              path=template.state_ref)
        node_ref = f"sgd_update::{weight_ref}::tp0"
        output_ref = f"{node_ref}.updated_weight"
        alias = f"trainable:{weight_ref}"
        updated = TensorValue(
            output_ref, weight.shape, DType.FP16,
            weight.logical_layout, weight.sharding, node_ref, (), alias,
        )
        values[output_ref] = updated
        nodes.append(LogicalNode(
            node_ref, source.instances[0].id,
            OpKind.OPTIMIZER_UPDATE, OpPhase.UPDATE, 0, weight.sharding.mesh_ref,
            (weight_ref, gradient_ref), (output_ref,),
            SgdUpdateWorkload(
                weight.shape, weight.shape, gradient.shape, gradient.shape,
                updated.shape, updated.shape, prod(weight.shape),
                plan.spec.learning_rate, 0.0,
                DType.FP16, DType.FP32, DType.FP16,
            ),
            next(node.math for node in source.nodes if node.id == template.wgrad_ref),
            NodeEffects(EffectKind.INPLACE, f"{node_ref}.effect", alias),
            "sgd_update",
        ))
        accesses.append(StateAccess.create(
            node_ref=node_ref, state_ref=state.id,
            mode=StateAccessMode.READ_WRITE, rank=0,
        ))
        edges.append(GraphEdge(
            f"{gradient_ref}.edge_to.{node_ref}", EdgeKind.DATA,
            template.wgrad_ref, node_ref, gradient_ref,
        ))
        values[weight_ref] = replace(
            weight, consumers=(*weight.consumers, node_ref),
        )
        values[gradient_ref] = replace(
            gradient, consumers=(*gradient.consumers, node_ref),
        )
    result = IR0.create(
        producer_pass="full_dense_training_sgd_source",
        job=source.job, instances=source.instances, nodes=tuple(nodes),
        values=tuple(values.values()), edges=tuple(edges),
        fusion_candidates=(), profile=source.profile, train=source.train,
        persistent_states=tuple(states.values()),
        state_accesses=tuple(accesses),
    )
    result.validate("full_dense_training_sgd_source")
    return result


__all__ = ["build_full_dense_training_sgd_ir0"]
