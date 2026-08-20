"""Build and independently validate the isolated S3-Lite MoE IR0 graph."""

from __future__ import annotations

from collections import Counter

from ..errors import SchemaError
from ..schema.common import (
    DType,
    MeshAxisName,
    Sharding,
    TensorValue,
    stable_artifact_id,
)
from ..schema.experiment import (
    ExperimentSpec,
    InferOutput,
    InferSource,
    WorkloadMode,
)
from ..schema.ir0 import (
    DeviceMesh,
    EdgeKind,
    EffectKind,
    GemmPartition,
    GemmWorkload,
    GraphEdge,
    IR0,
    JobKind,
    LogicalInstance,
    LogicalNode,
    LogicalRole,
    MeshAxis,
    NodeEffects,
    NodeMath,
    NumericalPolicy,
    OpKind,
    OpPhase,
    P2PByteWorkload,
    ParallelAxes,
    StateAccess,
    StateAccessMode,
    SwiGluWorkload,
)
from ..schema.lite_moe import (
    S3_LITE_STATIC_MOE_CASE_ID,
    LiteMoeOracle,
    LiteMoeSpec,
    LiteMoeTransferRole,
)
from ..schema.lite_moe_graph import (
    LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION,
    LiteMoeIR0Adapter,
    LiteMoeP2PBinding,
)
from ..schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    StateKind,
)
from ..schema.serde import canonical_digest


_INSTANCE_ID = "S3L0"
_MESH_REF = "S3L0.mesh.ep"
_MATH = NodeMath(DType.FP32, NumericalPolicy.BITWISE)
_EFFECTS = NodeEffects(EffectKind.PURE, None, None)


def _node_id(token: int, expert: int, role: str) -> str:
    return f"S3L0.token{token}.expert{expert}.{role}"


def _token_value_id(token: int) -> str:
    return f"S3L0.token{token}.input"


def _weight_value_id(expert: int, role: str) -> str:
    return f"S3L0.expert{expert}.weight.{role}"


def _intermediate_value_id(token: int, expert: int, role: str) -> str:
    return f"S3L0.token{token}.expert{expert}.value.{role}"


def _replicated(rank: int) -> Sharding:
    return Sharding(_MESH_REF, (None,) * rank, ())


def _edge(
    source: str, destination: str, value_id: str
) -> GraphEdge:
    key = {
        "source_node": source,
        "destination_node": destination,
        "value_id": value_id,
    }
    return GraphEdge(
        id=stable_artifact_id(
            "s3_lite_static_moe_ir0_edge",
            key,
            schema_version=LITE_MOE_IR0_ADAPTER_SCHEMA_VERSION,
        ),
        kind=EdgeKind.DATA,
        **key,
    )


def _assignment_map(spec: LiteMoeSpec) -> dict[int, int]:
    return {
        assignment.token_index: assignment.expert_index
        for assignment in spec.trace.assignments
    }


def _common_node(
    *,
    node_id: str,
    kind: OpKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    workload: object,
    impl_ref: str,
) -> LogicalNode:
    return LogicalNode(
        id=node_id,
        instance_id=_INSTANCE_ID,
        kind=kind,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref=_MESH_REF,
        inputs=inputs,
        outputs=outputs,
        workload=workload,
        math=_MATH,
        effects=_EFFECTS,
        impl_ref=impl_ref,
    )


def _parameter_states(
    assignments: dict[int, int],
) -> tuple[
    tuple[PersistentStateDecl, ...],
    tuple[StateAccess, ...],
]:
    declarations: list[PersistentStateDecl] = []
    declaration_by_weight: dict[str, PersistentStateDecl] = {}
    for expert in range(4):
        for role, shape in (
            ("gate", (16, 32)),
            ("up", (16, 32)),
            ("down", (32, 16)),
        ):
            weight_ref = _weight_value_id(expert, role)
            identity = PersistentStateIdentity.create(
                kind=StateKind.PARAMETER,
                instance_ref=_INSTANCE_ID,
                mesh_ref=_MESH_REF,
                request_ref=None,
                layer_index=None,
                tensor_ref=weight_ref,
                # Current IR0 parameter identity is TP-rank based.  EP home is
                # kept only in the typed adapter and must be added by placement.
                shard_index=0,
                generation=0,
            )
            declaration = PersistentStateDecl.create(
                identity=identity,
                shape=shape,
                dtype=DType.FP16,
                layout="KN_expert_local",
                lifetime=PersistentStateLifetime.PERSISTENT,
                access=PersistentStateAccess.READ_ONLY,
            )
            declarations.append(declaration)
            declaration_by_weight[weight_ref] = declaration
    accesses = tuple(
        StateAccess.create(
            node_ref=_node_id(token, expert, role),
            state_ref=declaration_by_weight[
                _weight_value_id(expert, role)
            ].id,
            mode=StateAccessMode.READ,
            rank=0,
        )
        for token, expert in sorted(assignments.items())
        for role in ("gate", "up", "down")
    )
    return tuple(declarations), accesses


def _build_graph(
    experiment: ExperimentSpec,
    spec: LiteMoeSpec,
) -> tuple[IR0, tuple[LiteMoeP2PBinding, ...]]:
    infer = experiment.workload.infer
    assert infer is not None and infer.profile is not None
    assignments = _assignment_map(spec)
    nodes: list[LogicalNode] = []
    values: list[TensorValue] = []
    bindings: list[LiteMoeP2PBinding] = []

    for token, expert in sorted(assignments.items()):
        token_home = token % 2
        expert_home = expert // 2
        remote = token_home != expert_home
        dispatch_id = _node_id(token, expert, "dispatch")
        gate_id = _node_id(token, expert, "gate")
        up_id = _node_id(token, expert, "up")
        values.append(
            TensorValue(
                id=_token_value_id(token),
                shape=(1, 16),
                dtype=DType.FP16,
                logical_layout="MH_token_slice",
                sharding=_replicated(2),
                producer=None,
                consumers=(dispatch_id,) if remote else (gate_id, up_id),
                alias_set=None,
            )
        )

    for expert in range(4):
        expert_tokens = tuple(
            token for token, selected in sorted(assignments.items())
            if selected == expert
        )
        for role, shape in (
            ("gate", (16, 32)),
            ("up", (16, 32)),
            ("down", (32, 16)),
        ):
            values.append(
                TensorValue(
                    id=_weight_value_id(expert, role),
                    shape=shape,
                    dtype=DType.FP16,
                    logical_layout="KN_expert_local",
                    sharding=_replicated(2),
                    producer=None,
                    consumers=tuple(
                        _node_id(token, expert, role)
                        for token in expert_tokens
                    ),
                    alias_set=None,
                )
            )

    for token, expert in sorted(assignments.items()):
        token_home = token % 2
        expert_home = expert // 2
        remote = token_home != expert_home
        token_value = _token_value_id(token)
        dispatch_id = _node_id(token, expert, "dispatch")
        routed_value = _intermediate_value_id(token, expert, "routed")
        gate_id = _node_id(token, expert, "gate")
        up_id = _node_id(token, expert, "up")
        swiglu_id = _node_id(token, expert, "swiglu")
        down_id = _node_id(token, expert, "down")
        combine_id = _node_id(token, expert, "combine")
        activation_value = routed_value if remote else token_value

        if remote:
            nodes.append(
                _common_node(
                    node_id=dispatch_id,
                    kind=OpKind.P2P,
                    inputs=(token_value,),
                    outputs=(routed_value,),
                    workload=P2PByteWorkload(bytes=32, dtype=DType.FP16),
                    impl_ref="s3_lite.moe_dispatch",
                )
            )
            values.append(
                TensorValue(
                    id=routed_value,
                    shape=(1, 16),
                    dtype=DType.FP16,
                    logical_layout="MH_remote_dispatch",
                    sharding=_replicated(2),
                    producer=dispatch_id,
                    consumers=(gate_id, up_id),
                    alias_set=None,
                )
            )
            bindings.append(
                LiteMoeP2PBinding.create(
                    node_ref=dispatch_id,
                    role=LiteMoeTransferRole.MOE_DISPATCH,
                    token_index=token,
                    expert_index=expert,
                    source_die_id=token_home,
                    destination_die_id=expert_home,
                )
            )

        gate_value = _intermediate_value_id(token, expert, "gate")
        up_value = _intermediate_value_id(token, expert, "up")
        swiglu_value = _intermediate_value_id(token, expert, "swiglu")
        down_value = _intermediate_value_id(token, expert, "down")
        nodes.extend(
            (
                _common_node(
                    node_id=gate_id,
                    kind=OpKind.GEMM,
                    inputs=(activation_value, _weight_value_id(expert, "gate")),
                    outputs=(gate_value,),
                    workload=GemmWorkload(
                        logical_shape=(1, 32, 16),
                        rank_shape=(1, 32, 16),
                        partition=GemmPartition.REPLICATED,
                        dtype=DType.FP16,
                    ),
                    impl_ref="matmul_forward",
                ),
                _common_node(
                    node_id=up_id,
                    kind=OpKind.GEMM,
                    inputs=(activation_value, _weight_value_id(expert, "up")),
                    outputs=(up_value,),
                    workload=GemmWorkload(
                        logical_shape=(1, 32, 16),
                        rank_shape=(1, 32, 16),
                        partition=GemmPartition.REPLICATED,
                        dtype=DType.FP16,
                    ),
                    impl_ref="matmul_forward",
                ),
                _common_node(
                    node_id=swiglu_id,
                    kind=OpKind.ELEMENTWISE,
                    inputs=(gate_value, up_value),
                    outputs=(swiglu_value,),
                    workload=SwiGluWorkload(
                        logical_input_shape=(1, 64),
                        logical_output_shape=(1, 32),
                        rank_input_shape=(1, 64),
                        rank_output_shape=(1, 32),
                        dtype=DType.FP16,
                    ),
                    impl_ref="swiglu",
                ),
                _common_node(
                    node_id=down_id,
                    kind=OpKind.GEMM,
                    inputs=(swiglu_value, _weight_value_id(expert, "down")),
                    outputs=(down_value,),
                    workload=GemmWorkload(
                        logical_shape=(1, 16, 32),
                        rank_shape=(1, 16, 32),
                        partition=GemmPartition.REPLICATED,
                        dtype=DType.FP16,
                    ),
                    impl_ref="matmul_forward",
                ),
            )
        )
        values.extend(
            (
                TensorValue(
                    gate_value, (1, 32), DType.FP16, "MI_gate",
                    _replicated(2), gate_id, (swiglu_id,), None,
                ),
                TensorValue(
                    up_value, (1, 32), DType.FP16, "MI_up",
                    _replicated(2), up_id, (swiglu_id,), None,
                ),
                TensorValue(
                    swiglu_value, (1, 32), DType.FP16, "MI_swiglu",
                    _replicated(2), swiglu_id, (down_id,), None,
                ),
                TensorValue(
                    down_value, (1, 16), DType.FP16, "MH_expert_output",
                    _replicated(2), down_id,
                    (combine_id,) if remote else (), None,
                ),
            )
        )
        if remote:
            combine_value = _intermediate_value_id(token, expert, "combined")
            nodes.append(
                _common_node(
                    node_id=combine_id,
                    kind=OpKind.P2P,
                    inputs=(down_value,),
                    outputs=(combine_value,),
                    workload=P2PByteWorkload(bytes=32, dtype=DType.FP16),
                    impl_ref="s3_lite.moe_combine",
                )
            )
            values.append(
                TensorValue(
                    combine_value, (1, 16), DType.FP16, "MH_remote_combine",
                    _replicated(2), combine_id, (), None,
                )
            )
            bindings.append(
                LiteMoeP2PBinding.create(
                    node_ref=combine_id,
                    role=LiteMoeTransferRole.MOE_COMBINE,
                    token_index=token,
                    expert_index=expert,
                    source_die_id=expert_home,
                    destination_die_id=token_home,
                )
            )

    declarations, accesses = _parameter_states(assignments)
    edges = tuple(
        _edge(value.producer, consumer, value.id)
        for value in values
        if value.producer is not None
        for consumer in value.consumers
    )
    graph = IR0.create(
        producer_pass="lite_moe_graph",
        job=JobKind.INFER,
        instances=(
            LogicalInstance(
                id=_INSTANCE_ID,
                role=LogicalRole.PREFILL,
                replicas=1,
                parallel=ParallelAxes(tp=1, sp=False, dp=1, pp=1, ep=2),
                meshes=(
                    DeviceMesh(
                        id=_MESH_REF,
                        axes=(MeshAxis(MeshAxisName.EP, 2),),
                    ),
                ),
            ),
        ),
        nodes=tuple(nodes),
        values=tuple(values),
        edges=edges,
        fusion_candidates=(),
        profile=infer.profile,
        train=None,
        persistent_states=declarations,
        state_accesses=accesses,
    )
    return graph, tuple(bindings)


class LiteMoeIR0Validator:
    """Independent exact validator for the fixed, unrolled S3-Lite graph."""

    @staticmethod
    def validate(
        adapter: LiteMoeIR0Adapter,
        experiment: ExperimentSpec,
        spec: LiteMoeSpec,
        oracle: LiteMoeOracle,
        path: str = "lite_moe_ir0_adapter",
    ) -> None:
        if type(adapter) is not LiteMoeIR0Adapter:
            raise SchemaError("must be a LiteMoeIR0Adapter", path=path)
        if type(experiment) is not ExperimentSpec:
            raise SchemaError("must be an ExperimentSpec", path="experiment")
        if type(spec) is not LiteMoeSpec:
            raise SchemaError("must be a LiteMoeSpec", path="lite_moe_spec")
        if type(oracle) is not LiteMoeOracle:
            raise SchemaError("must be a LiteMoeOracle", path="lite_moe_oracle")
        experiment.validate("experiment")
        spec.validate("lite_moe_spec")
        oracle.validate_against(spec, "lite_moe_oracle")
        adapter.validate(path)
        if (
            adapter.source_experiment_digest != canonical_digest(experiment)
            or adapter.source_moe_spec_id != spec.id
            or adapter.source_moe_spec_digest != canonical_digest(spec)
            or adapter.source_oracle_id != oracle.id
            or adapter.source_oracle_digest != canonical_digest(oracle)
        ):
            raise SchemaError("source provenance mismatch", path=path)
        infer = experiment.workload.infer
        if (
            experiment.workload.mode is not WorkloadMode.INFER
            or infer is None
            or infer.source is not InferSource.STATIC_PROFILE
            or infer.output is not InferOutput.LOGITS
            or infer.profile is None
            or (experiment.model.H, experiment.model.I) != (16, 32)
            or (infer.profile.prefill_tokens, infer.profile.decode_tokens) != (8, 0)
            or spec.trace.token_count != 8
            or spec.trace.expert_histogram != (2, 2, 2, 2)
        ):
            raise SchemaError(
                "inputs must equal the fixed T8/H16/I32 S3-Lite case",
                path=path,
            )
        graph = adapter.graph
        if graph.profile != infer.profile or graph.job is not JobKind.INFER:
            raise SchemaError("graph/profile contract mismatch", path=f"{path}.graph")
        expected_instance = LogicalInstance(
            id=_INSTANCE_ID,
            role=LogicalRole.PREFILL,
            replicas=1,
            parallel=ParallelAxes(tp=1, sp=False, dp=1, pp=1, ep=2),
            meshes=(DeviceMesh(_MESH_REF, (MeshAxis(MeshAxisName.EP, 2),)),),
        )
        if graph.instances != (expected_instance,):
            raise SchemaError(
                "must use the exact EP2 logical instance",
                path=f"{path}.graph.instances",
            )

        assignments = _assignment_map(spec)
        node_index = {node.id: node for node in graph.nodes}
        expected_node_order: list[str] = []
        expected_bindings: list[LiteMoeP2PBinding] = []

        def require_node(
            node_id: str,
            *,
            kind: OpKind,
            inputs: tuple[str, ...],
            outputs: tuple[str, ...],
            workload: object,
            impl_ref: str,
        ) -> None:
            node = node_index.get(node_id)
            if node is None:
                raise SchemaError(
                    f"missing exact node {node_id!r}", path=f"{path}.graph.nodes"
                )
            if (
                node.instance_id != _INSTANCE_ID
                or node.kind is not kind
                or node.phase is not OpPhase.FWD
                or node.stage != 0
                or node.mesh_ref != _MESH_REF
                or node.inputs != inputs
                or node.outputs != outputs
                or node.workload != workload
                or node.math != _MATH
                or node.effects != _EFFECTS
                or node.impl_ref != impl_ref
            ):
                raise SchemaError(
                    f"exact node contract mismatch for {node_id!r}",
                    path=f"{path}.graph.nodes",
                )
            expected_node_order.append(node_id)

        for token, expert in sorted(assignments.items()):
            token_home = token % 2
            expert_home = expert // 2
            remote = token_home != expert_home
            token_value = _token_value_id(token)
            routed_value = _intermediate_value_id(token, expert, "routed")
            activation_value = routed_value if remote else token_value
            dispatch_id = _node_id(token, expert, "dispatch")
            gate_id = _node_id(token, expert, "gate")
            up_id = _node_id(token, expert, "up")
            swiglu_id = _node_id(token, expert, "swiglu")
            down_id = _node_id(token, expert, "down")
            combine_id = _node_id(token, expert, "combine")
            gate_value = _intermediate_value_id(token, expert, "gate")
            up_value = _intermediate_value_id(token, expert, "up")
            swiglu_value = _intermediate_value_id(token, expert, "swiglu")
            down_value = _intermediate_value_id(token, expert, "down")
            if remote:
                require_node(
                    dispatch_id,
                    kind=OpKind.P2P,
                    inputs=(token_value,),
                    outputs=(routed_value,),
                    workload=P2PByteWorkload(bytes=32, dtype=DType.FP16),
                    impl_ref="s3_lite.moe_dispatch",
                )
                expected_bindings.append(
                    LiteMoeP2PBinding.create(
                        node_ref=dispatch_id,
                        role=LiteMoeTransferRole.MOE_DISPATCH,
                        token_index=token,
                        expert_index=expert,
                        source_die_id=token_home,
                        destination_die_id=expert_home,
                    )
                )
            for node_id, role, input_value, output_value, shape in (
                (gate_id, "gate", activation_value, gate_value, (1, 32, 16)),
                (up_id, "up", activation_value, up_value, (1, 32, 16)),
            ):
                require_node(
                    node_id,
                    kind=OpKind.GEMM,
                    inputs=(input_value, _weight_value_id(expert, role)),
                    outputs=(output_value,),
                    workload=GemmWorkload(
                        logical_shape=shape,
                        rank_shape=shape,
                        partition=GemmPartition.REPLICATED,
                        dtype=DType.FP16,
                    ),
                    impl_ref="matmul_forward",
                )
            require_node(
                swiglu_id,
                kind=OpKind.ELEMENTWISE,
                inputs=(gate_value, up_value),
                outputs=(swiglu_value,),
                workload=SwiGluWorkload(
                    logical_input_shape=(1, 64),
                    logical_output_shape=(1, 32),
                    rank_input_shape=(1, 64),
                    rank_output_shape=(1, 32),
                    dtype=DType.FP16,
                ),
                impl_ref="swiglu",
            )
            require_node(
                down_id,
                kind=OpKind.GEMM,
                inputs=(swiglu_value, _weight_value_id(expert, "down")),
                outputs=(down_value,),
                workload=GemmWorkload(
                    logical_shape=(1, 16, 32),
                    rank_shape=(1, 16, 32),
                    partition=GemmPartition.REPLICATED,
                    dtype=DType.FP16,
                ),
                impl_ref="matmul_forward",
            )
            if remote:
                combine_value = _intermediate_value_id(token, expert, "combined")
                require_node(
                    combine_id,
                    kind=OpKind.P2P,
                    inputs=(down_value,),
                    outputs=(combine_value,),
                    workload=P2PByteWorkload(bytes=32, dtype=DType.FP16),
                    impl_ref="s3_lite.moe_combine",
                )
                expected_bindings.append(
                    LiteMoeP2PBinding.create(
                        node_ref=combine_id,
                        role=LiteMoeTransferRole.MOE_COMBINE,
                        token_index=token,
                        expert_index=expert,
                        source_die_id=expert_home,
                        destination_die_id=token_home,
                    )
                )
        if tuple(node.id for node in graph.nodes) != tuple(expected_node_order):
            raise SchemaError(
                "nodes must use exact token/pipeline order",
                path=f"{path}.graph.nodes",
            )

        value_index = {value.id: value for value in graph.values}
        expected_value_order = [
            _token_value_id(token) for token in range(8)
        ]
        expected_value_order.extend(
            _weight_value_id(expert, role)
            for expert in range(4)
            for role in ("gate", "up", "down")
        )

        def require_value(
            value_id: str,
            shape: tuple[int, ...],
            layout: str,
        ) -> None:
            value = value_index.get(value_id)
            if value is None:
                raise SchemaError(
                    f"missing exact value {value_id!r}", path=f"{path}.graph.values"
                )
            if (
                value.shape != shape
                or value.dtype is not DType.FP16
                or value.logical_layout != layout
                or value.sharding != _replicated(len(shape))
                or value.alias_set is not None
            ):
                raise SchemaError(
                    f"exact value contract mismatch for {value_id!r}",
                    path=f"{path}.graph.values",
                )

        for token in range(8):
            require_value(_token_value_id(token), (1, 16), "MH_token_slice")
        for expert in range(4):
            for role, shape in (
                ("gate", (16, 32)),
                ("up", (16, 32)),
                ("down", (32, 16)),
            ):
                require_value(_weight_value_id(expert, role), shape, "KN_expert_local")
        for token, expert in sorted(assignments.items()):
            remote = token % 2 != expert // 2
            if remote:
                value_id = _intermediate_value_id(token, expert, "routed")
                expected_value_order.append(value_id)
                require_value(value_id, (1, 16), "MH_remote_dispatch")
            for role, shape, layout in (
                ("gate", (1, 32), "MI_gate"),
                ("up", (1, 32), "MI_up"),
                ("swiglu", (1, 32), "MI_swiglu"),
                ("down", (1, 16), "MH_expert_output"),
            ):
                value_id = _intermediate_value_id(token, expert, role)
                expected_value_order.append(value_id)
                require_value(value_id, shape, layout)
            if remote:
                value_id = _intermediate_value_id(token, expert, "combined")
                expected_value_order.append(value_id)
                require_value(value_id, (1, 16), "MH_remote_combine")
        if tuple(value.id for value in graph.values) != tuple(expected_value_order):
            raise SchemaError(
                "values must use exact source/weight/token-pipeline order",
                path=f"{path}.graph.values",
            )

        expected_edges = tuple(
            _edge(value.producer, consumer, value.id)
            for value in graph.values
            if value.producer is not None
            for consumer in value.consumers
        )
        if graph.edges != expected_edges or len(graph.edges) != 36:
            raise SchemaError("edge topology mismatch", path=f"{path}.graph.edges")

        state_by_tensor = {
            declaration.identity.tensor_ref: declaration
            for declaration in graph.persistent_states
        }
        expected_weight_refs = {
            _weight_value_id(expert, role)
            for expert in range(4)
            for role in ("gate", "up", "down")
        }
        if set(state_by_tensor) != expected_weight_refs:
            raise SchemaError(
                "expert parameter-state mismatch",
                path=f"{path}.graph.persistent_states",
            )
        for weight_ref, declaration in state_by_tensor.items():
            if (
                declaration.identity.kind is not StateKind.PARAMETER
                or declaration.identity.instance_ref != _INSTANCE_ID
                or declaration.identity.mesh_ref != _MESH_REF
                or declaration.identity.shard_index != 0
                or declaration.identity.generation != 0
                or declaration.shape != value_index[weight_ref].shape
                or declaration.dtype is not DType.FP16
                or declaration.layout != "KN_expert_local"
                or declaration.lifetime is not PersistentStateLifetime.PERSISTENT
                or declaration.access is not PersistentStateAccess.READ_ONLY
            ):
                raise SchemaError(
                    "expert parameter-state fields mismatch",
                    path=f"{path}.graph.persistent_states",
                )
        expected_access_keys = {
            (
                _node_id(token, expert, role),
                state_by_tensor[_weight_value_id(expert, role)].id,
                0,
                StateAccessMode.READ,
            )
            for token, expert in assignments.items()
            for role in ("gate", "up", "down")
        }
        actual_access_keys = {
            (access.node_ref, access.state_ref, access.rank, access.mode)
            for access in graph.state_accesses
        }
        if actual_access_keys != expected_access_keys:
            raise SchemaError(
                "expert parameter access mismatch",
                path=f"{path}.graph.state_accesses",
            )
        if graph.fusion_candidates:
            raise SchemaError(
                "S3-Lite graph must not declare fusion candidates",
                path=f"{path}.graph.fusion_candidates",
            )
        if adapter.p2p_bindings != tuple(expected_bindings):
            raise SchemaError(
                "P2P binding/route mismatch", path=f"{path}.p2p_bindings"
            )

        kinds = Counter(node.kind for node in graph.nodes)
        if kinds != {
            OpKind.GEMM: 24,
            OpKind.ELEMENTWISE: 8,
            OpKind.P2P: 8,
        }:
            raise SchemaError("must contain 32 compute and 8 P2P nodes", path=path)
        if len(graph.persistent_states) != 12 or len(graph.state_accesses) != 24:
            raise SchemaError(
                "must contain 12 parameter states and 24 READ accesses",
                path=path,
            )
        if any(
            access.mode is not StateAccessMode.READ
            for access in graph.state_accesses
        ):
            raise SchemaError("all parameter accesses must be READ", path=path)
        p2p_bytes = sum(
            node.workload.bytes
            for node in graph.nodes
            if type(node.workload) is P2PByteWorkload
        )
        if p2p_bytes != 256 or p2p_bytes != oracle.logical_p2p_bytes:
            raise SchemaError("P2P byte total mismatch", path=path)
        gemm_flops = sum(
            2
            * node.workload.logical_shape[0]
            * node.workload.logical_shape[1]
            * node.workload.logical_shape[2]
            for node in graph.nodes
            if type(node.workload) is GemmWorkload
        )
        if gemm_flops != 24576 or gemm_flops != oracle.total_expert_gemm_flops:
            raise SchemaError("expert FLOP total mismatch", path=path)
        for token, expert in assignments.items():
            expert_home = expert // 2
            remote = token % 2 != expert_home
            bound_roles = {
                binding.role
                for binding in adapter.p2p_bindings
                if binding.token_index == token
            }
            expected_roles = (
                {
                    LiteMoeTransferRole.MOE_DISPATCH,
                    LiteMoeTransferRole.MOE_COMBINE,
                }
                if remote
                else set()
            )
            if bound_roles != expected_roles:
                raise SchemaError(
                    "token route does not match expert home=e//2",
                    path=f"{path}.p2p_bindings",
                )


def build_lite_moe_ir0_adapter(
    experiment: ExperimentSpec,
    spec: LiteMoeSpec,
    oracle: LiteMoeOracle,
) -> LiteMoeIR0Adapter:
    """Build the fixed static MoE IR0 without importing integration fixtures."""

    if type(experiment) is not ExperimentSpec:
        raise TypeError("experiment must be an ExperimentSpec")
    if type(spec) is not LiteMoeSpec:
        raise TypeError("spec must be a LiteMoeSpec")
    if type(oracle) is not LiteMoeOracle:
        raise TypeError("oracle must be a LiteMoeOracle")
    experiment.validate("experiment")
    spec.validate("lite_moe_spec")
    oracle.validate_against(spec, "lite_moe_oracle")
    graph, bindings = _build_graph(experiment, spec)
    result = LiteMoeIR0Adapter.create(
        case_id=S3_LITE_STATIC_MOE_CASE_ID,
        source_experiment_digest=canonical_digest(experiment),
        source_moe_spec_id=spec.id,
        source_moe_spec_digest=canonical_digest(spec),
        source_oracle_id=oracle.id,
        source_oracle_digest=canonical_digest(oracle),
        graph=graph,
        p2p_bindings=bindings,
    )
    LiteMoeIR0Validator.validate(result, experiment, spec, oracle)
    return result


__all__ = ["LiteMoeIR0Validator", "build_lite_moe_ir0_adapter"]
