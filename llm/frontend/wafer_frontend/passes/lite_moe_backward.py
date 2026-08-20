"""Build the exact S3-Lite MoE down-projection backward overlay."""

from __future__ import annotations

from collections import defaultdict

from ..errors import SchemaError
from ..schema.common import DType, stable_artifact_id
from ..schema.ir0 import OpKind
from ..schema.lite_moe import LiteMoeStaticTrace
from ..schema.lite_moe_backward import (
    LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION,
    LiteMoeBackwardContract,
    LiteMoeBackwardOracle,
    LiteMoeBackwardOverlay,
    LiteMoeExpertReduce,
    LiteMoeExpertSgdStore,
    LiteMoeRemoteGradDte,
    LiteMoeTrainableDownState,
    LiteMoeTokenWgrad,
)
from ..schema.lite_moe_execution import (
    LiteMoeGlobalDag,
    LiteMoeProjection,
    LiteMoeScheduled,
)
from ..schema.lite_moe_n4 import LiteMoeN4IR1
from ..schema.lite_moe_n6 import LiteMoeN6Intent
from ..schema.persistent_state import (
    HbmBinding,
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    StateKind,
)
from .lite_moe_execution import (
    validate_lite_moe_global,
    validate_lite_moe_projection,
    validate_lite_moe_schedule,
)
from .lite_moe_n6 import validate_lite_moe_n6_intent


def build_lite_moe_backward_contract(
    trace: LiteMoeStaticTrace,
) -> LiteMoeBackwardContract:
    return LiteMoeBackwardContract.create(trace=trace)


def build_lite_moe_backward_oracle(
    contract: LiteMoeBackwardContract,
) -> LiteMoeBackwardOracle:
    return LiteMoeBackwardOracle.create(contract=contract)


def _token_from_ref(value: str) -> int:
    marker = ".token"
    try:
        suffix = value.split(marker, 1)[1]
        return int(suffix.split(".", 1)[0])
    except (IndexError, ValueError) as error:
        raise SchemaError(
            "cannot derive canonical token index", path="lite_moe_backward"
        ) from error


def _expert_from_ref(value: str) -> int:
    marker = ".expert"
    try:
        suffix = value.split(marker, 1)[1]
        return int(suffix.split(".", 1)[0])
    except (IndexError, ValueError) as error:
        raise SchemaError(
            "cannot derive canonical expert index", path="lite_moe_backward"
        ) from error


def _unit_id(kind: str, semantic: dict[str, object]) -> str:
    return stable_artifact_id(
        f"s3_lite_moe_{kind}",
        semantic,
        schema_version=LITE_MOE_BACKWARD_OVERLAY_SCHEMA_VERSION,
    )


def _components(
    n4: LiteMoeN4IR1,
    projection: LiteMoeProjection,
    trace: LiteMoeStaticTrace,
) -> tuple[
    tuple[LiteMoeTrainableDownState, ...],
    tuple[LiteMoeRemoteGradDte, ...],
    tuple[LiteMoeTokenWgrad, ...],
    tuple[LiteMoeExpertReduce, ...],
    tuple[LiteMoeExpertSgdStore, ...],
]:
    assignments = {item.token_index: item for item in trace.assignments}
    dispatch_flows = tuple(
        sorted(
            (
                flow
                for flow in projection.flows
                if flow.source_value_ref.endswith(".input")
            ),
            key=lambda flow: _token_from_ref(flow.source_value_ref),
        )
    )
    remote_grad_dtes: list[LiteMoeRemoteGradDte] = []
    remote_by_token: dict[int, str] = {}
    for flow in dispatch_flows:
        token = _token_from_ref(flow.source_value_ref)
        assignment = assignments[token]
        semantic = {
            "token_index": token,
            "expert_index": assignment.expert_index,
            "forward_flow_ref": flow.id,
            "pair_route_ref": flow.pair_route_ref,
            "source_die_id": flow.source_die_id,
            "destination_die_id": flow.destination_die_id,
            "bytes": 32,
        }
        unit = LiteMoeRemoteGradDte(
            _unit_id("remote_grad_dte", semantic), **semantic
        )
        remote_grad_dtes.append(unit)
        remote_by_token[token] = unit.id

    node_index = {item.id: item for item in n4.graph.nodes}
    manifest = n4.graph.persistent_state_manifest
    if manifest is None:
        raise SchemaError(
            "backward overlay requires the formal state manifest",
            path="lite_moe_backward.n4.graph.persistent_state_manifest",
        )
    declaration_index = {item.id: item for item in manifest.declarations}
    binding_index = {item.state_ref: item for item in manifest.bindings}
    accesses_by_node: dict[str, list[object]] = defaultdict(list)
    for access in n4.graph.state_accesses:
        accesses_by_node[access.node_ref].append(access)
    down_nodes = tuple(
        sorted(
            (
                node
                for node in n4.graph.nodes
                if node.kind is OpKind.GEMM and node.id.endswith(".down")
            ),
            key=lambda node: _token_from_ref(node.id),
        )
    )
    if len(down_nodes) != 8:
        raise SchemaError(
            "requires exactly eight down-projection nodes",
            path="lite_moe_backward.n4.graph.nodes",
        )
    source_state_by_expert: dict[int, str] = {}
    for node in down_nodes:
        expert = _expert_from_ref(node.id)
        accesses = accesses_by_node[node.id]
        if len(accesses) != 1:
            raise SchemaError(
                "down node must carry one exact parameter state access",
                path=f"lite_moe_backward.{node.id}",
            )
        prior = source_state_by_expert.setdefault(expert, accesses[0].state_ref)
        if prior != accesses[0].state_ref:
            raise SchemaError(
                "expert tokens do not share one down-weight state",
                path=f"lite_moe_backward.expert{expert}",
            )
    trainable_down_states: list[LiteMoeTrainableDownState] = []
    for expert in range(4):
        source_ref = source_state_by_expert[expert]
        source_decl = declaration_index[source_ref]
        source_binding = binding_index[source_ref]
        if (
            source_decl.identity.kind is not StateKind.PARAMETER
            or source_decl.tensor_bytes != 1024
            or source_binding.die_id != expert // 2
            or source_binding.size_bytes != source_decl.tensor_bytes
        ):
            raise SchemaError(
                "down weight must be the exact home-local 1024B parameter",
                path=f"lite_moe_backward.expert{expert}.source_state",
            )
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
            die_id=source_binding.die_id,
            address=source_binding.address,
            size_bytes=source_binding.size_bytes,
        )
        trainable_down_states.append(
            LiteMoeTrainableDownState(
                expert,
                expert // 2,
                source_ref,
                declaration,
                binding,
            )
        )
    trainable_by_expert = {
        item.expert_index: item for item in trainable_down_states
    }
    token_wgrads: list[LiteMoeTokenWgrad] = []
    for node in down_nodes:
        token = _token_from_ref(node.id)
        expert = _expert_from_ref(node.id)
        assignment = assignments.get(token)
        accesses = accesses_by_node[node.id]
        if (
            assignment is None
            or assignment.expert_index != expert
            or len(accesses) != 1
            or len(node.inputs) != 2
        ):
            raise SchemaError(
                "down node does not close trace/state lineage",
                path=f"lite_moe_backward.{node.id}",
            )
        saved_activation_ref = node.inputs[0]
        if saved_activation_ref not in node_index and not any(
            value.id == saved_activation_ref for value in n4.graph.values
        ):
            raise SchemaError(
                "saved activation is absent from N4 graph",
                path=f"lite_moe_backward.{node.id}.inputs[0]",
            )
        root = f"s3_lite.moe_backward.expert{expert}.wgrad.root"
        contribution = (
            f"s3_lite.moe.backward.expert{expert}.token{token}.wgrad"
        )
        semantic = {
            "token_index": token,
            "expert_index": expert,
            "home_die_id": expert // 2,
            "down_node_ref": node.id,
            "saved_activation_ref": saved_activation_ref,
            "upstream_grad_ref": (
                f"s3_lite.moe.backward.token{token}.down_output_grad"
            ),
            "down_weight_state_ref": trainable_by_expert[expert].declaration.id,
            "root_buffer_ref": root,
            "contribution_ref": contribution,
            "offset_bytes": (token % 2) * 2048,
            "size_bytes": 2048,
            "dtype": DType.FP32,
            "deps": ((remote_by_token[token],) if token in remote_by_token else ()),
        }
        token_wgrads.append(
            LiteMoeTokenWgrad(_unit_id("token_wgrad", semantic), **semantic)
        )

    expert_reduces: list[LiteMoeExpertReduce] = []
    sgd_stores: list[LiteMoeExpertSgdStore] = []
    for expert in range(4):
        contributions = tuple(
            item for item in token_wgrads if item.expert_index == expert
        )
        if len(contributions) != 2:
            raise SchemaError(
                "each expert requires two WGRAD contributions",
                path=f"lite_moe_backward.expert{expert}",
            )
        reduce_semantic = {
            "expert_index": expert,
            "home_die_id": expert // 2,
            "root_buffer_ref": contributions[0].root_buffer_ref,
            "contribution_refs": tuple(
                item.contribution_ref for item in contributions
            ),
            "input_offsets": (0, 2048),
            "input_span_bytes": 4096,
            "output_alias_ref": contributions[0].contribution_ref,
            "output_offset_bytes": 0,
            "output_size_bytes": 2048,
            "deps": tuple(item.id for item in contributions),
        }
        reduce = LiteMoeExpertReduce(
            _unit_id("expert_reduce", reduce_semantic), **reduce_semantic
        )
        expert_reduces.append(reduce)
        store_semantic = {
            "expert_index": expert,
            "home_die_id": expert // 2,
            "down_weight_state_ref": contributions[0].down_weight_state_ref,
            "down_weight_hbm_binding_ref": trainable_by_expert[expert].binding.id,
            "reduce_ref": reduce.id,
            "gradient_alias_ref": reduce.output_alias_ref,
            "weight_read_bytes": 1024,
            "gradient_read_bytes": 2048,
            "state_store_bytes": 1024,
            "learning_rate": 0.001,
            "momentum": 0.0,
            "deps": (reduce.id,),
        }
        sgd_stores.append(
            LiteMoeExpertSgdStore(
                _unit_id("expert_sgd_store", store_semantic), **store_semantic
            )
        )
    return (
        tuple(trainable_down_states),
        tuple(remote_grad_dtes),
        tuple(token_wgrads),
        tuple(expert_reduces),
        tuple(sgd_stores),
    )


def build_lite_moe_backward_overlay(
    n4: LiteMoeN4IR1,
    projection: LiteMoeProjection,
    schedule: LiteMoeScheduled,
    global_dag: LiteMoeGlobalDag,
    n6_intent: LiteMoeN6Intent,
    trace: LiteMoeStaticTrace,
) -> LiteMoeBackwardOverlay:
    validate_lite_moe_n6_intent(
        n6_intent, global_dag, schedule, projection, n4
    )
    contract = build_lite_moe_backward_contract(trace)
    oracle = build_lite_moe_backward_oracle(contract)
    trainable, remote, wgrads, reduces, stores = _components(
        n4, projection, trace
    )
    result = LiteMoeBackwardOverlay.create(
        source_n4_id=n4.id,
        source_projection_id=projection.id,
        source_schedule_id=schedule.id,
        source_global_id=global_dag.id,
        source_n6_intent_id=n6_intent.id,
        contract=contract,
        oracle=oracle,
        trainable_down_states=trainable,
        remote_grad_dtes=remote,
        token_wgrads=wgrads,
        expert_reduces=reduces,
        sgd_stores=stores,
    )
    validate_lite_moe_backward_overlay(
        result, n4, projection, schedule, global_dag, n6_intent, trace
    )
    return result


def validate_lite_moe_backward_overlay(
    result: LiteMoeBackwardOverlay,
    n4: LiteMoeN4IR1,
    projection: LiteMoeProjection,
    schedule: LiteMoeScheduled,
    global_dag: LiteMoeGlobalDag,
    n6_intent: LiteMoeN6Intent,
    trace: LiteMoeStaticTrace,
) -> None:
    result.validate()
    validate_lite_moe_projection(projection, n4)
    validate_lite_moe_schedule(schedule, projection, n4)
    validate_lite_moe_global(global_dag, schedule, projection, n4)
    validate_lite_moe_n6_intent(
        n6_intent, global_dag, schedule, projection, n4
    )
    contract = build_lite_moe_backward_contract(trace)
    oracle = build_lite_moe_backward_oracle(contract)
    components = _components(n4, projection, trace)
    if (
        result.source_n4_id != n4.id
        or result.source_projection_id != projection.id
        or result.source_schedule_id != schedule.id
        or result.source_global_id != global_dag.id
        or result.source_n6_intent_id != n6_intent.id
        or result.contract != contract
        or result.oracle != oracle
        or (
            result.remote_grad_dtes,
            result.token_wgrads,
            result.expert_reduces,
            result.sgd_stores,
        )
        != components[1:]
        or result.trainable_down_states != components[0]
    ):
        raise SchemaError(
            "backward overlay is not an exact quotient of formal S3-Lite sources",
            path="lite_moe_backward_overlay",
        )


__all__ = [
    "build_lite_moe_backward_contract",
    "build_lite_moe_backward_oracle",
    "build_lite_moe_backward_overlay",
    "validate_lite_moe_backward_overlay",
]
