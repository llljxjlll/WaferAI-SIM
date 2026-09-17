"""Attach device-side AdamW to every genuine Dense L2 source WGRAD.

The initial scope is one TP/DP rank.  Each parameter has four independently
persistent FP32/INT32 optimizer states; the following two-step and physical
state-version passes must prove their LOAD/UPDATE/STORE paths separately.
"""
from __future__ import annotations

from dataclasses import replace
from math import prod

from ..errors import SchemaError
from ..schema.common import DType, Sharding, TensorValue
from ..schema.flexible_dense_train import FlexibleDenseTrainPlan
from ..schema.gemm_input_dx_workload import GemmInputDxWorkload
from ..schema.gemm_weight_wgrad_workload import GemmWeightWgradWorkload
from ..schema.ir0 import (
    AdamwUpdateWorkload, EdgeKind, EffectKind, GraphEdge, IR0, LogicalNode,
    NodeEffects, OpKind, OpPhase, StateAccess, StateAccessMode,
)
from ..schema.persistent_state import (
    PersistentStateAccess, PersistentStateDecl, PersistentStateIdentity,
    PersistentStateLifetime, StateKind,
)
from .full_dense_training_backward_ir0 import build_full_dense_training_backward_ir0


_OPTIMIZER_STATES = (
    ("master", StateKind.OPTIMIZER_MASTER, DType.FP32),
    ("m", StateKind.OPTIMIZER_MOMENT1, DType.FP32),
    ("v", StateKind.OPTIMIZER_MOMENT2, DType.FP32),
    ("step", StateKind.OPTIMIZER_STEP, DType.INT32),
)


def build_full_dense_training_adamw_ir0(
    plan: FlexibleDenseTrainPlan, *, step: int = 1,
    beta1: float = 0.9, beta2: float = 0.999,
    epsilon: float = 1e-8, weight_decay: float = 0.01,
) -> IR0:
    """Build real forward/loss/backward plus 15 FP16/FP32 AdamW updates.

    This one-step source is a production building block, not a runtime or
    offload receipt.  All norm gamma vectors keep their genuine 1D shape.
    """
    if plan.spec.tp_degree != 1 or plan.spec.dp_degree != 1:
        raise SchemaError("AdamW full source currently requires one TP/DP rank",
                          path="plan.spec")
    if step not in (1, 2):
        raise SchemaError("two-step AdamW source requires step 1 or 2",
                          path="step")
    source = build_full_dense_training_backward_ir0(plan)
    old_states = {state.id: state for state in source.persistent_states}
    states: dict[str, PersistentStateDecl] = {}
    trainable_by_old: dict[str, PersistentStateDecl] = {}
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
        state = PersistentStateDecl.create(
            identity=identity, shape=old.shape, dtype=old.dtype,
            layout=old.layout, lifetime=PersistentStateLifetime.PERSISTENT,
            access=PersistentStateAccess.READ_WRITE,
        )
        states[state.id] = state
        trainable_by_old[old.id] = state
    values = {value.id: value for value in source.values}
    nodes = []
    for node in source.nodes:
        if (node.kind in (OpKind.GEMM_INPUT_DX, OpKind.GEMM_WEIGHT_WGRAD)
                and type(node.workload) in (GemmInputDxWorkload,
                                             GemmWeightWgradWorkload)):
            old_ref = node.workload.source_parameter_state_ref
            node = replace(node, workload=replace(
                node.workload,
                source_parameter_state_ref=trainable_by_old[old_ref].id,
            ))
        nodes.append(node)
    accesses = [StateAccess.create(
        node_ref=access.node_ref,
        state_ref=trainable_by_old[access.state_ref].id,
        mode=access.mode, rank=access.rank,
        read_offset=access.read_offset, read_shape=access.read_shape,
        write_offset=access.write_offset, write_shape=access.write_shape,
    ) for access in source.state_accesses]
    edges = list(source.edges)
    for template in plan.parameter_templates:
        old = old_states[template.state_ref]
        trainable = trainable_by_old[old.id]
        weight_ref = old.identity.tensor_ref
        assert weight_ref is not None
        weight = values[weight_ref]
        gradient_ref = f"{template.wgrad_ref}.output"
        gradient = values[gradient_ref]
        if (weight.dtype is not DType.FP16 or gradient.dtype is not DType.FP32
                or weight.shape != gradient.shape or weight.producer is not None
                or gradient.producer != template.wgrad_ref):
            raise SchemaError("AdamW needs an exact source weight and FP32 WGRAD",
                              path=template.state_ref)
        update_ref = f"adamw_update::{weight_ref}::tp0"
        weight_alias = f"trainable:{weight_ref}"
        values[weight_ref] = replace(
            weight, alias_set=weight_alias,
            consumers=(*weight.consumers, update_ref),
        )
        role_inputs = []
        role_outputs = []
        for role, kind, dtype in _OPTIMIZER_STATES:
            input_ref = f"{update_ref}.{role}"
            output_ref = f"{update_ref}.updated_{role}"
            shape = (1,) if role == "step" else weight.shape
            sharding = (Sharding(weight.sharding.mesh_ref, (None,), ())
                        if role == "step" else weight.sharding)
            alias = f"optimizer:{role}:{weight_ref}"
            identity = PersistentStateIdentity.create(
                kind=kind, instance_ref=old.identity.instance_ref,
                mesh_ref=old.identity.mesh_ref,
                request_ref=None, layer_index=None,
                tensor_ref=input_ref, shard_index=0, generation=0,
            )
            state = PersistentStateDecl.create(
                identity=identity, shape=shape, dtype=dtype,
                layout=f"{weight.logical_layout}.{role}",
                lifetime=PersistentStateLifetime.PERSISTENT,
                access=PersistentStateAccess.READ_WRITE,
            )
            states[state.id] = state
            values[input_ref] = TensorValue(
                input_ref, shape, dtype, state.layout, sharding,
                None, (update_ref,), alias,
            )
            values[output_ref] = TensorValue(
                output_ref, shape, dtype, state.layout, sharding,
                update_ref, (), alias,
            )
            role_inputs.append(input_ref)
            role_outputs.append(output_ref)
            accesses.append(StateAccess.create(
                node_ref=update_ref, state_ref=state.id,
                mode=StateAccessMode.READ_WRITE, rank=0,
            ))
        updated_weight_ref = f"{update_ref}.updated_weight"
        values[updated_weight_ref] = TensorValue(
            updated_weight_ref, weight.shape, DType.FP16,
            weight.logical_layout, weight.sharding,
            update_ref, (), weight_alias,
        )
        values[gradient_ref] = replace(
            gradient, consumers=(*gradient.consumers, update_ref),
        )
        nodes.append(LogicalNode(
            update_ref, source.instances[0].id,
            OpKind.OPTIMIZER_UPDATE, OpPhase.UPDATE, 0,
            weight.sharding.mesh_ref,
            (weight_ref, gradient_ref, *role_inputs),
            (updated_weight_ref, *role_outputs),
            AdamwUpdateWorkload(
                logical_weight_shape=weight.shape,
                rank_weight_shape=trainable.shape,
                element_count=prod(trainable.shape), step=step,
                learning_rate=plan.spec.learning_rate,
                beta1=beta1, beta2=beta2, epsilon=epsilon,
                weight_decay=weight_decay,
            ),
            next(node.math for node in source.nodes
                 if node.id == template.wgrad_ref),
            NodeEffects(EffectKind.INPLACE,
                        f"{update_ref}.effect", weight_alias),
            "adamw_update",
        ))
        accesses.append(StateAccess.create(
            node_ref=update_ref, state_ref=trainable.id,
            mode=StateAccessMode.READ_WRITE, rank=0,
        ))
        edges.append(GraphEdge(
            f"{gradient_ref}.edge_to.{update_ref}", EdgeKind.DATA,
            gradient.producer, update_ref, gradient_ref,
        ))
    result = IR0.create(
        producer_pass="full_dense_training_adamw_source",
        job=source.job, instances=source.instances,
        nodes=tuple(nodes), values=tuple(values.values()), edges=tuple(edges),
        fusion_candidates=(), profile=source.profile, train=source.train,
        persistent_states=tuple(states.values()),
        state_accesses=tuple(accesses),
    )
    result.validate("full_dense_training_adamw_source")
    from .validate_ir0 import DenseIR0Validator
    DenseIR0Validator.validate(result, "full_dense_training_adamw_source")
    return result


__all__ = ["build_full_dense_training_adamw_ir0"]
