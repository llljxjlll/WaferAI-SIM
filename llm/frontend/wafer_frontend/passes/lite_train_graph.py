"""Append the exact S2-Lite LM-head backward/update chain to Train forward."""

from __future__ import annotations

from dataclasses import replace

from ..errors import SchemaError
from ..schema.common import DType, TensorValue
from ..schema.experiment import ExperimentSpec
from ..schema.ir0 import (
    CrossEntropyBackwardWorkload,
    CrossEntropyReduction,
    EdgeKind,
    EffectKind,
    GemmPartition,
    GemmWorkload,
    GraphEdge,
    IR0,
    LogicalNode,
    NodeEffects,
    OpKind,
    OpPhase,
    SgdUpdateWorkload,
    StateAccess,
    StateAccessMode,
)
from ..schema.lite_train import S2LiteLmHeadTrainContract, S2LiteLmHeadTrainOracle
from ..schema.lite_train_graph import S2LiteLmHeadTrainIR0
from ..schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    StateKind,
)
from ..schema.serde import canonical_digest
from .train_forward import build_train_forward_ir0


def _value_with_consumer(value: TensorValue, consumer: str) -> TensorValue:
    if consumer in value.consumers:
        raise SchemaError("consumer already exists", path=f"values.{value.id}")
    return replace(value, consumers=(*value.consumers, consumer))


def _append_s2_lite_lm_head_train(
    base_graph: IR0,
    contract: S2LiteLmHeadTrainContract,
    oracle: S2LiteLmHeadTrainOracle,
) -> IR0:
    """Internal pure graph transform; callers validate the three source objects."""

    instance = base_graph.instances[0]
    prefix = instance.id
    node_index = {node.id: node for node in base_graph.nodes}
    value_index = {value.id: value for value in base_graph.values}
    lm_head = node_index[f"{prefix}.lm_head"]
    ce_forward = node_index[f"{prefix}.cross_entropy"]
    logits = value_index[lm_head.outputs[0]]
    labels = value_index[ce_forward.inputs[1]]
    hidden = value_index[lm_head.inputs[0]]
    weight = value_index[lm_head.inputs[1]]
    rows = oracle.logical_rows
    hidden_size = contract.hidden_size
    vocabulary = contract.vocabulary_size
    if (
        logits.shape != (rows, vocabulary)
        or labels.shape != (rows,)
        or hidden.shape != (rows, hidden_size)
        or weight.shape != (hidden_size, vocabulary)
    ):
        raise SchemaError(
            "base LM-head geometry does not match the Lite contract",
            path="base_graph",
        )

    ce_backward_id = f"{prefix}.cross_entropy_backward"
    wgrad_id = f"{prefix}.lm_head_wgrad"
    update_id = f"{prefix}.sgd_update"
    loss_gradient_id = f"{prefix}.loss_gradient"
    logits_gradient_id = f"{prefix}.logits_gradient"
    weight_gradient_id = f"{prefix}.lm_head.weight_gradient"
    updated_weight_id = f"{prefix}.lm_head.weight.updated"
    trainable_alias = f"trainable:{weight.id}"

    changed_values = {
        logits.id: _value_with_consumer(logits, ce_backward_id),
        labels.id: _value_with_consumer(labels, ce_backward_id),
        hidden.id: _value_with_consumer(hidden, wgrad_id),
        weight.id: _value_with_consumer(weight, update_id),
    }
    values = [changed_values.get(value.id, value) for value in base_graph.values]
    values.extend(
        (
            TensorValue(
                id=loss_gradient_id,
                shape=(rows,),
                dtype=DType.FP32,
                logical_layout="M_loss_gradient",
                sharding=labels.sharding,
                producer=None,
                consumers=(ce_backward_id,),
                alias_set=None,
            ),
            TensorValue(
                id=logits_gradient_id,
                shape=(rows, vocabulary),
                dtype=DType.FP16,
                logical_layout="MV_logits_gradient",
                sharding=logits.sharding,
                producer=ce_backward_id,
                consumers=(wgrad_id,),
                alias_set=None,
            ),
            TensorValue(
                id=weight_gradient_id,
                shape=(hidden_size, vocabulary),
                dtype=DType.FP32,
                logical_layout="HV_weight_gradient",
                sharding=weight.sharding,
                producer=wgrad_id,
                consumers=(update_id,),
                alias_set=None,
            ),
            TensorValue(
                id=updated_weight_id,
                shape=(hidden_size, vocabulary),
                dtype=DType.FP16,
                logical_layout=weight.logical_layout,
                sharding=weight.sharding,
                producer=update_id,
                consumers=(),
                alias_set=trainable_alias,
            ),
        )
    )

    math_policy = ce_forward.math
    nodes = [*base_graph.nodes]
    nodes.extend(
        (
            LogicalNode(
                id=ce_backward_id,
                instance_id=instance.id,
                kind=OpKind.CE_BACKWARD,
                phase=OpPhase.DGRAD,
                stage=0,
                mesh_ref=lm_head.mesh_ref,
                inputs=(logits.id, labels.id, loss_gradient_id),
                outputs=(logits_gradient_id,),
                workload=CrossEntropyBackwardWorkload(
                    profile=base_graph.profile,
                    reduction=CrossEntropyReduction.NONE,
                    logical_logits_shape=(rows, vocabulary),
                    rank_logits_shape=(rows, vocabulary),
                    logical_label_shape=(rows,),
                    rank_label_shape=(rows,),
                    logical_loss_gradient_shape=(rows,),
                    rank_loss_gradient_shape=(rows,),
                    logical_logits_gradient_shape=(rows, vocabulary),
                    rank_logits_gradient_shape=(rows, vocabulary),
                    logits_dtype=DType.FP16,
                    label_dtype=DType.INT32,
                    loss_gradient_dtype=DType.FP32,
                    logits_gradient_dtype=DType.FP16,
                ),
                math=math_policy,
                effects=NodeEffects(EffectKind.PURE, None, None),
                impl_ref="cross_entropy_backward",
            ),
            LogicalNode(
                id=wgrad_id,
                instance_id=instance.id,
                kind=OpKind.GEMM,
                phase=OpPhase.WGRAD,
                stage=0,
                mesh_ref=lm_head.mesh_ref,
                inputs=(hidden.id, logits_gradient_id),
                outputs=(weight_gradient_id,),
                workload=GemmWorkload(
                    logical_shape=(hidden_size, vocabulary, rows),
                    rank_shape=(hidden_size, vocabulary, rows),
                    partition=GemmPartition.REPLICATED,
                    dtype=DType.FP16,
                ),
                math=math_policy,
                effects=NodeEffects(EffectKind.PURE, None, None),
                impl_ref="lm_head_wgrad",
            ),
            LogicalNode(
                id=update_id,
                instance_id=instance.id,
                kind=OpKind.OPTIMIZER_UPDATE,
                phase=OpPhase.UPDATE,
                stage=0,
                mesh_ref=lm_head.mesh_ref,
                inputs=(weight.id, weight_gradient_id),
                outputs=(updated_weight_id,),
                workload=SgdUpdateWorkload(
                    logical_weight_shape=(hidden_size, vocabulary),
                    rank_weight_shape=(hidden_size, vocabulary),
                    logical_gradient_shape=(hidden_size, vocabulary),
                    rank_gradient_shape=(hidden_size, vocabulary),
                    logical_updated_weight_shape=(hidden_size, vocabulary),
                    rank_updated_weight_shape=(hidden_size, vocabulary),
                    element_count=hidden_size * vocabulary,
                    learning_rate=contract.learning_rate,
                    momentum=contract.momentum,
                    weight_dtype=DType.FP16,
                    gradient_dtype=DType.FP32,
                    updated_weight_dtype=DType.FP16,
                ),
                math=math_policy,
                effects=NodeEffects(
                    EffectKind.INPLACE,
                    f"{update_id}.effect",
                    trainable_alias,
                ),
                impl_ref="sgd_update",
            ),
        )
    )

    state_id_map: dict[str, PersistentStateDecl] = {}
    states: list[PersistentStateDecl] = []
    for declaration in base_graph.persistent_states:
        if declaration.identity.tensor_ref != weight.id:
            states.append(declaration)
            continue
        identity = PersistentStateIdentity.create(
            kind=StateKind.TRAINABLE_PARAMETER,
            instance_ref=declaration.identity.instance_ref,
            mesh_ref=declaration.identity.mesh_ref,
            request_ref=None,
            layer_index=None,
            tensor_ref=weight.id,
            shard_index=declaration.identity.shard_index,
            generation=declaration.identity.generation,
        )
        trainable = PersistentStateDecl.create(
            identity=identity,
            shape=declaration.shape,
            dtype=declaration.dtype,
            layout=declaration.layout,
            lifetime=PersistentStateLifetime.PERSISTENT,
            access=PersistentStateAccess.READ_WRITE,
        )
        state_id_map[declaration.id] = trainable
        states.append(trainable)
    if len(state_id_map) != 1:
        raise SchemaError(
            "TP1 base graph must contain one LM-head parameter declaration",
            path="base_graph.persistent_states",
        )
    old_state_id, trainable_state = next(iter(state_id_map.items()))
    accesses = [
        replace(access, state_ref=trainable_state.id)
        if access.state_ref == old_state_id
        else access
        for access in base_graph.state_accesses
    ]
    accesses = [
        StateAccess.create(
            node_ref=access.node_ref,
            state_ref=access.state_ref,
            mode=access.mode,
            rank=access.rank,
            read_offset=access.read_offset,
            read_shape=access.read_shape,
            write_offset=access.write_offset,
            write_shape=access.write_shape,
        )
        if access.state_ref == trainable_state.id
        else access
        for access in accesses
    ]
    accesses.append(
        StateAccess.create(
            node_ref=update_id,
            state_ref=trainable_state.id,
            mode=StateAccessMode.READ_WRITE,
            rank=0,
        )
    )

    edges = [
        GraphEdge(
            id=f"{value.id}.edge_to.{consumer.rsplit('.', 1)[-1]}",
            kind=EdgeKind.DATA,
            source_node=value.producer,
            destination_node=consumer,
            value_id=value.id,
        )
        for value in values
        if value.producer is not None
        for consumer in value.consumers
    ]
    edges.append(
        GraphEdge(
            id=f"{ce_forward.id}.control_to.cross_entropy_backward",
            kind=EdgeKind.CONTROL,
            source_node=ce_forward.id,
            destination_node=ce_backward_id,
            value_id=None,
        )
    )
    return IR0.create(
        producer_pass="s2_lite_lm_head_train_expand",
        job=base_graph.job,
        instances=base_graph.instances,
        nodes=tuple(nodes),
        values=tuple(values),
        edges=tuple(edges),
        fusion_candidates=base_graph.fusion_candidates,
        profile=base_graph.profile,
        train=base_graph.train,
        persistent_states=tuple(states),
        state_accesses=tuple(accesses),
    )


def build_s2_lite_lm_head_train_ir0(
    base_spec: ExperimentSpec,
    contract: S2LiteLmHeadTrainContract,
    oracle: S2LiteLmHeadTrainOracle,
) -> S2LiteLmHeadTrainIR0:
    if type(base_spec) is not ExperimentSpec:
        raise SchemaError("must be an ExperimentSpec", path="base_spec")
    if type(contract) is not S2LiteLmHeadTrainContract:
        raise SchemaError("must be an S2LiteLmHeadTrainContract", path="contract")
    if type(oracle) is not S2LiteLmHeadTrainOracle:
        raise SchemaError("must be an S2LiteLmHeadTrainOracle", path="oracle")
    base_spec.validate("base_spec")
    contract.validate("contract")
    oracle.validate_against_contract(contract, path="oracle")
    if contract.source_spec_digest != canonical_digest(base_spec):
        raise SchemaError("contract source digest does not match base_spec", path="contract.source_spec_digest")
    train = base_spec.workload.train
    assert train is not None
    instance = base_spec.parallel.instances[0]
    if (
        train.micro_batch != contract.micro_batch_size
        or train.seq_len != contract.sequence_length
        or train.structure.micro_batch_count != contract.micro_batch_count
        or base_spec.model.H != contract.hidden_size
        or base_spec.model.V != contract.vocabulary_size
        or base_spec.model.dtype is not contract.activation_dtype
        or instance.dp != contract.dp_degree
        or instance.tp != contract.tp_degree
        or instance.pp != contract.pp_degree
        or instance.ep != contract.ep_degree
        or instance.sp
    ):
        raise SchemaError("base_spec geometry does not match the Lite contract", path="base_spec")
    base_graph = build_train_forward_ir0(base_spec)
    graph = _append_s2_lite_lm_head_train(base_graph, contract, oracle)
    from .validate_ir0 import DenseIR0Validator

    DenseIR0Validator.validate(graph, "s2_lite_lm_head_train_ir0.graph")
    return S2LiteLmHeadTrainIR0.create(
        base_graph=base_graph,
        contract=contract,
        oracle=oracle,
        graph=graph,
    )


__all__ = ["build_s2_lite_lm_head_train_ir0"]
