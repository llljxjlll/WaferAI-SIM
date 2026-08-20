"""Build the exact fixed-EP4 down-projection WGRAD backward overlay."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.common import DType
from ..schema.ir0 import OpKind, ReduceOp
from ..schema.ir2 import BufferOwnership
from ..schema.lite_moe import LiteMoeTransferRole
from ..schema.lite_moe_dp4 import S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID
from ..schema.lite_moe_dp4_backward import (
    LiteMoeDp4Backward,
    LiteMoeDp4ExpertReduce,
    LiteMoeDp4ExpertSgdStore,
    LiteMoeDp4RemoteGrad,
    LiteMoeDp4TokenWgrad,
    LiteMoeDp4TrainableDownState,
)
from ..schema.lite_moe_dp4_train_forward import LiteMoeDp4TrainForward
from ..schema.persistent_state import (
    HbmBinding,
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    StateKind,
)
from .lite_moe_dp4_train_forward import validate_lite_moe_dp4_train_forward


def _components(
    train_forward: LiteMoeDp4TrainForward,
) -> tuple[
    tuple[LiteMoeDp4RemoteGrad, ...],
    tuple[LiteMoeDp4TrainableDownState, ...],
    tuple[LiteMoeDp4TokenWgrad, ...],
    tuple[LiteMoeDp4ExpertReduce, ...],
    tuple[LiteMoeDp4ExpertSgdStore, ...],
]:
    validate_lite_moe_dp4_train_forward(train_forward)
    forward = train_forward.forward
    binding_by_id = {item.id: item for item in forward.adapter.p2p_bindings}
    flow_by_binding = {item.p2p_binding_ref: item for item in forward.projection.flows}
    by_token_role = {
        (binding.token_index, binding.role): flow_by_binding[binding.id]
        for binding in forward.adapter.p2p_bindings
    }
    remote_gradients = []
    remote_by_token = {}
    for token in forward.adapter.topology.remote_token_indices:
        assignment = forward.adapter.spec.trace.assignments[token]
        combine = by_token_role.get((token, LiteMoeTransferRole.MOE_COMBINE))
        dispatch = by_token_role.get((token, LiteMoeTransferRole.MOE_DISPATCH))
        if (
            combine is None
            or dispatch is None
            or combine.source_die_id != assignment.expert_index
            or combine.destination_die_id != token % 4
            or dispatch.source_die_id != combine.destination_die_id
            or dispatch.destination_die_id != combine.source_die_id
            or binding_by_id[combine.p2p_binding_ref].bytes != 32
        ):
            raise SchemaError(
                "remote gradient lacks exact reverse forward-combine route",
                path=f"lite_moe_dp4_backward.token{token}",
            )
        prefix = f"S3M4.backward.token{token}.expert{assignment.expert_index}.upstream"
        unit = LiteMoeDp4RemoteGrad.create(
            token_index=token,
            expert_index=assignment.expert_index,
            slot_index=assignment.slot_index,
            forward_combine_flow_ref=combine.id,
            reverse_pair_route_ref=dispatch.pair_route_ref,
            source_die_id=token % 4,
            destination_die_id=assignment.expert_index,
            source_gradient_ref=f"{prefix}.source",
            received_gradient_ref=f"{prefix}.received",
            send_ref=f"{prefix}.send",
            recv_ref=f"{prefix}.recv",
            wait_ref=f"{prefix}.wait",
            bytes=32,
            dtype=DType.FP16,
        )
        remote_gradients.append(unit)
        remote_by_token[token] = unit

    graph = forward.n4.graph
    manifest = graph.persistent_state_manifest
    if manifest is None:
        raise SchemaError("backward requires persistent state manifest", path="n4.graph")
    declarations = {item.id: item for item in manifest.declarations}
    bindings = {item.state_ref: item for item in manifest.bindings}
    accesses = defaultdict(list)
    for access in graph.state_accesses:
        accesses[access.node_ref].append(access)
    down_nodes = tuple(
        sorted(
            (item for item in graph.nodes if item.kind is OpKind.GEMM and item.id.endswith(".down")),
            key=lambda item: int(item.id.split(".token", 1)[1].split(".", 1)[0]),
        )
    )
    if len(down_nodes) != 8:
        raise SchemaError("requires eight down-projection nodes", path="n4.graph.nodes")
    source_by_expert = {}
    for node in down_nodes:
        token = int(node.id.split(".token", 1)[1].split(".", 1)[0])
        expert = forward.adapter.spec.trace.assignments[token].expert_index
        node_accesses = accesses[node.id]
        if len(node_accesses) != 1:
            raise SchemaError("down node must have one state access", path=node.id)
        prior = source_by_expert.setdefault(expert, node_accesses[0].state_ref)
        if prior != node_accesses[0].state_ref:
            raise SchemaError("expert down tokens disagree on state", path=node.id)

    trainable_states = []
    for expert in range(4):
        source_ref = source_by_expert[expert]
        source_decl = declarations[source_ref]
        source_binding = bindings[source_ref]
        if (
            source_decl.identity.kind is not StateKind.PARAMETER
            or source_decl.tensor_bytes != 1024
            or source_binding.die_id != expert
            or source_binding.size_bytes != 1024
        ):
            raise SchemaError("down state is not exact expert-local 1024B", path=f"expert{expert}")
        identity = PersistentStateIdentity.create(
            kind=StateKind.TRAINABLE_PARAMETER,
            instance_ref=source_decl.identity.instance_ref,
            mesh_ref=source_decl.identity.mesh_ref,
            request_ref=source_decl.identity.request_ref,
            layer_index=source_decl.identity.layer_index,
            tensor_ref=source_decl.identity.tensor_ref,
            shard_index=source_decl.identity.shard_index,
            generation=source_decl.identity.generation,
        )
        declaration = PersistentStateDecl.create(
            identity=identity,
            shape=source_decl.shape,
            dtype=source_decl.dtype,
            layout=source_decl.layout,
            lifetime=PersistentStateLifetime.PERSISTENT,
            access=PersistentStateAccess.READ_WRITE,
        )
        binding = HbmBinding.create(
            state_ref=declaration.id,
            die_id=expert,
            address=source_binding.address,
            size_bytes=source_binding.size_bytes,
        )
        item = LiteMoeDp4TrainableDownState(
            expert,
            expert,
            source_ref,
            declaration,
            binding,
        )
        item.validate(f"trainable_down_states[{expert}]")
        trainable_states.append(item)
    state_by_expert = {item.expert_index: item for item in trainable_states}
    tape_by_token = {item.token_index: item for item in train_forward.tape_buffers}

    token_wgrads = []
    for node in down_nodes:
        token = int(node.id.split(".token", 1)[1].split(".", 1)[0])
        assignment = forward.adapter.spec.trace.assignments[token]
        expert = assignment.expert_index
        tape = tape_by_token[token]
        remote = remote_by_token.get(token)
        root = f"S3M4.backward.expert{expert}.wgrad.root"
        token_wgrads.append(
            LiteMoeDp4TokenWgrad.create(
                token_index=token,
                expert_index=expert,
                slot_index=assignment.slot_index,
                home_die_id=expert,
                down_node_ref=node.id,
                tape_buffer_ref=tape.id,
                tape_value_ref=tape.value_ref,
                upstream_gradient_ref=(
                    remote.received_gradient_ref
                    if remote is not None
                    else f"S3M4.backward.token{token}.expert{expert}.upstream.local"
                ),
                trainable_state_ref=state_by_expert[expert].declaration.id,
                root_buffer_ref=root,
                contribution_ref=f"{root}.slot{assignment.slot_index}",
                offset_bytes=assignment.slot_index * 2048,
                size_bytes=2048,
                dtype=DType.FP32,
                deps=(() if remote is None else (remote.id,)),
            )
        )

    expert_reduces = []
    sgd_stores = []
    for expert in range(4):
        contributions = tuple(item for item in token_wgrads if item.expert_index == expert)
        reduce = LiteMoeDp4ExpertReduce.create(
            expert_index=expert,
            home_die_id=expert,
            root_buffer_ref=contributions[0].root_buffer_ref,
            contribution_refs=tuple(item.contribution_ref for item in contributions),
            input_offsets=(0, 2048),
            input_dtype=DType.FP32,
            accumulator_dtype=DType.FP32,
            output_dtype=DType.FP32,
            reduce_op=ReduceOp.SUM,
            input_count=2,
            element_count=512,
            input_stride_bytes=2048,
            source_span_bytes=4096,
            destination_span_bytes=2048,
            output_alias_ref=contributions[0].contribution_ref,
            output_ownership=BufferOwnership.ALIASED,
            deps=tuple(item.id for item in contributions),
        )
        expert_reduces.append(reduce)
        state = state_by_expert[expert]
        sgd_stores.append(
            LiteMoeDp4ExpertSgdStore.create(
                expert_index=expert,
                home_die_id=expert,
                down_weight_state_ref=state.declaration.id,
                down_weight_hbm_binding_ref=state.binding.id,
                reduce_ref=reduce.id,
                gradient_alias_ref=reduce.output_alias_ref,
                updated_weight_ref=f"S3M4.backward.expert{expert}.down_weight.updated",
                hbm_store_ref=f"S3M4.backward.expert{expert}.down_weight.hbm_store",
                weight_read_bytes=1024,
                gradient_read_bytes=2048,
                state_store_bytes=1024,
                learning_rate=0.001,
                momentum=0.0,
                deps=(reduce.id,),
            )
        )
    return (
        tuple(remote_gradients),
        tuple(trainable_states),
        tuple(token_wgrads),
        tuple(expert_reduces),
        tuple(sgd_stores),
    )


def build_lite_moe_dp4_backward(
    train_forward: LiteMoeDp4TrainForward,
) -> LiteMoeDp4Backward:
    remote, states, wgrads, reduces, stores = _components(train_forward)
    result = LiteMoeDp4Backward.create(
        case_id=S3_LITE_MOE_DP4_DOWN_WGRAD_CASE_ID,
        source_topology_id=train_forward.source_topology_id,
        source_oracle_id=train_forward.source_oracle_id,
        train_forward=train_forward,
        remote_gradients=remote,
        trainable_down_states=states,
        token_wgrads=wgrads,
        expert_reduces=reduces,
        sgd_stores=stores,
        updated_weight_refs=tuple(item.updated_weight_ref for item in stores),
    )
    validate_lite_moe_dp4_backward(result, train_forward)
    return result


def validate_lite_moe_dp4_backward(
    result: LiteMoeDp4Backward,
    train_forward: LiteMoeDp4TrainForward | None = None,
) -> None:
    result.validate()
    source = result.train_forward if train_forward is None else train_forward
    expected = _components(source)
    if (
        result.train_forward != source
        or result.source_topology_id != source.source_topology_id
        or result.source_oracle_id != source.source_oracle_id
        or (
            result.remote_gradients,
            result.trainable_down_states,
            result.token_wgrads,
            result.expert_reduces,
            result.sgd_stores,
        )
        != expected
        or result.updated_weight_refs != tuple(item.updated_weight_ref for item in expected[4])
    ):
        raise SchemaError(
            "backward is not the exact TF/topology quotient",
            path="lite_moe_dp4_backward",
        )


__all__ = [
    "build_lite_moe_dp4_backward",
    "validate_lite_moe_dp4_backward",
]
