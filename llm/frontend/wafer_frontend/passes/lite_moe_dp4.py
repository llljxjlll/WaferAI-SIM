"""Build and place the fixed EP4 S3-Lite MoE inference graph."""

from __future__ import annotations

from dataclasses import replace
from collections import Counter

from ..errors import SchemaError
from ..schema.common import DType, MeshAxisName, Sharding, TensorValue, stable_artifact_id
from ..schema.experiment import ExperimentSpec, InferOutput, InferSource, WorkloadMode
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
from ..schema.ir1 import IR1
from ..schema.lite_moe import LiteMoeStaticTrace, LiteMoeTransferRole
from ..schema.lite_moe_dp4 import (
    LITE_MOE_DP4_IR0_ADAPTER_SCHEMA_VERSION,
    LiteMoeDp4IR0Adapter,
    LiteMoeDp4N4IR1,
    LiteMoeDp4Oracle,
    LiteMoeDp4P2PBinding,
    LiteMoeDp4PlacedIR1,
    LiteMoeDp4Spec,
    LiteMoeDp4Topology,
)
from ..schema.n4 import FusionPartitionContext, InterDiePlanningContext
from ..schema.persistent_state import (
    HbmBinding,
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    PersistentStateManifest,
    StateKind,
)
from ..schema.placement import PlacementContext
from ..schema.serde import canonical_digest
from .fusion_partition import partition_ir1
from .group_registry import _expected_group
from .inter_die_plan import plan_ir1
from .placement import _physical_instance, _physical_node


_INSTANCE = "S3M4"
_MESH = "S3M4.mesh.ep"
_MATH = NodeMath(DType.FP32, NumericalPolicy.BITWISE)
_EFFECTS = NodeEffects(EffectKind.PURE, None, None)


def _node(token: int, expert: int, role: str) -> str:
    return f"{_INSTANCE}.token{token}.expert{expert}.{role}"


def _token(token: int) -> str:
    return f"{_INSTANCE}.token{token}.input"


def _weight(expert: int, role: str) -> str:
    return f"{_INSTANCE}.expert{expert}.weight.{role}"


def _value(token: int, expert: int, role: str) -> str:
    return f"{_INSTANCE}.token{token}.expert{expert}.value.{role}"


def _sharding(rank: int) -> Sharding:
    return Sharding(_MESH, (None,) * rank, ())


def _edge(source: str, destination: str, value_ref: str) -> GraphEdge:
    semantic = {
        "source_node": source,
        "destination_node": destination,
        "value_id": value_ref,
    }
    return GraphEdge(
        stable_artifact_id(
            "s3_lite_moe_dp4_edge",
            semantic,
            schema_version=LITE_MOE_DP4_IR0_ADAPTER_SCHEMA_VERSION,
        ),
        EdgeKind.DATA,
        **semantic,
    )


def _common_node(
    node_id: str,
    kind: OpKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    workload: object,
    impl_ref: str,
) -> LogicalNode:
    return LogicalNode(
        node_id,
        _INSTANCE,
        kind,
        OpPhase.FWD,
        0,
        _MESH,
        inputs,
        outputs,
        workload,
        _MATH,
        _EFFECTS,
        impl_ref,
    )


def _parameter_states(
    assignments: tuple[int, ...],
) -> tuple[tuple[PersistentStateDecl, ...], tuple[StateAccess, ...]]:
    declarations = []
    by_weight = {}
    for expert in range(4):
        for role, shape in (
            ("gate", (16, 32)),
            ("up", (16, 32)),
            ("down", (32, 16)),
        ):
            ref = _weight(expert, role)
            identity = PersistentStateIdentity.create(
                kind=StateKind.PARAMETER,
                instance_ref=_INSTANCE,
                mesh_ref=_MESH,
                request_ref=None,
                layer_index=None,
                tensor_ref=ref,
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
            by_weight[ref] = declaration
    accesses = tuple(
        StateAccess.create(
            node_ref=_node(token, expert, role),
            state_ref=by_weight[_weight(expert, role)].id,
            mode=StateAccessMode.READ,
            rank=0,
        )
        for token, expert in enumerate(assignments)
        for role in ("gate", "up", "down")
    )
    return tuple(declarations), accesses


def _build_graph(
    experiment: ExperimentSpec,
    spec: LiteMoeDp4Spec,
) -> tuple[IR0, tuple[LiteMoeDp4P2PBinding, ...]]:
    infer = experiment.workload.infer
    assert infer is not None and infer.profile is not None
    assignments = tuple(item.expert_index for item in spec.trace.assignments)
    slots = tuple(item.slot_index for item in spec.trace.assignments)
    nodes = []
    values = []
    bindings = []

    for token, expert in enumerate(assignments):
        remote = token % 4 != expert
        values.append(
            TensorValue(
                _token(token),
                (1, 16),
                DType.FP16,
                "MH_token_slice",
                _sharding(2),
                None,
                (_node(token, expert, "dispatch"),)
                if remote
                else (_node(token, expert, "gate"), _node(token, expert, "up")),
                None,
            )
        )
    for expert in range(4):
        expert_tokens = tuple(
            token for token, selected in enumerate(assignments) if selected == expert
        )
        for role, shape in (
            ("gate", (16, 32)),
            ("up", (16, 32)),
            ("down", (32, 16)),
        ):
            values.append(
                TensorValue(
                    _weight(expert, role),
                    shape,
                    DType.FP16,
                    "KN_expert_local",
                    _sharding(2),
                    None,
                    tuple(_node(token, expert, role) for token in expert_tokens),
                    None,
                )
            )

    for token, expert in enumerate(assignments):
        remote = token % 4 != expert
        token_ref = _token(token)
        routed = _value(token, expert, "routed")
        activation = routed if remote else token_ref
        dispatch = _node(token, expert, "dispatch")
        gate = _node(token, expert, "gate")
        up = _node(token, expert, "up")
        swiglu = _node(token, expert, "swiglu")
        down = _node(token, expert, "down")
        combine = _node(token, expert, "combine")
        if remote:
            nodes.append(
                _common_node(
                    dispatch,
                    OpKind.P2P,
                    (token_ref,),
                    (routed,),
                    P2PByteWorkload(32, DType.FP16),
                    "s3_lite.dp4.moe_dispatch",
                )
            )
            values.append(
                TensorValue(
                    routed,
                    (1, 16),
                    DType.FP16,
                    "MH_remote_dispatch",
                    _sharding(2),
                    dispatch,
                    (gate, up),
                    None,
                )
            )
            bindings.append(
                LiteMoeDp4P2PBinding.create(
                    node_ref=dispatch,
                    role=LiteMoeTransferRole.MOE_DISPATCH,
                    token_index=token,
                    expert_index=expert,
                    slot_index=slots[token],
                )
            )
        gate_value = _value(token, expert, "gate")
        up_value = _value(token, expert, "up")
        swiglu_value = _value(token, expert, "swiglu")
        down_value = _value(token, expert, "down")
        nodes.extend(
            (
                _common_node(
                    gate,
                    OpKind.GEMM,
                    (activation, _weight(expert, "gate")),
                    (gate_value,),
                    GemmWorkload((1, 32, 16), (1, 32, 16), GemmPartition.REPLICATED, DType.FP16),
                    "matmul_forward",
                ),
                _common_node(
                    up,
                    OpKind.GEMM,
                    (activation, _weight(expert, "up")),
                    (up_value,),
                    GemmWorkload((1, 32, 16), (1, 32, 16), GemmPartition.REPLICATED, DType.FP16),
                    "matmul_forward",
                ),
                _common_node(
                    swiglu,
                    OpKind.ELEMENTWISE,
                    (gate_value, up_value),
                    (swiglu_value,),
                    SwiGluWorkload((1, 64), (1, 32), (1, 64), (1, 32), DType.FP16),
                    "swiglu",
                ),
                _common_node(
                    down,
                    OpKind.GEMM,
                    (swiglu_value, _weight(expert, "down")),
                    (down_value,),
                    GemmWorkload((1, 16, 32), (1, 16, 32), GemmPartition.REPLICATED, DType.FP16),
                    "matmul_forward",
                ),
            )
        )
        values.extend(
            (
                TensorValue(gate_value, (1, 32), DType.FP16, "MI_gate", _sharding(2), gate, (swiglu,), None),
                TensorValue(up_value, (1, 32), DType.FP16, "MI_up", _sharding(2), up, (swiglu,), None),
                TensorValue(swiglu_value, (1, 32), DType.FP16, "MI_swiglu", _sharding(2), swiglu, (down,), None),
                TensorValue(
                    down_value,
                    (1, 16),
                    DType.FP16,
                    "MH_expert_output",
                    _sharding(2),
                    down,
                    (combine,) if remote else (),
                    None,
                ),
            )
        )
        if remote:
            combined = _value(token, expert, "combined")
            nodes.append(
                _common_node(
                    combine,
                    OpKind.P2P,
                    (down_value,),
                    (combined,),
                    P2PByteWorkload(32, DType.FP16),
                    "s3_lite.dp4.moe_combine",
                )
            )
            values.append(
                TensorValue(
                    combined,
                    (1, 16),
                    DType.FP16,
                    "MH_remote_combine",
                    _sharding(2),
                    combine,
                    (),
                    None,
                )
            )
            bindings.append(
                LiteMoeDp4P2PBinding.create(
                    node_ref=combine,
                    role=LiteMoeTransferRole.MOE_COMBINE,
                    token_index=token,
                    expert_index=expert,
                    slot_index=slots[token],
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
        producer_pass="lite_moe_dp4_graph",
        job=JobKind.INFER,
        instances=(
            LogicalInstance(
                _INSTANCE,
                LogicalRole.PREFILL,
                1,
                ParallelAxes(tp=1, sp=False, dp=1, pp=1, ep=4),
                (DeviceMesh(_MESH, (MeshAxis(MeshAxisName.EP, 4),)),),
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


def build_lite_moe_dp4_spec(trace: LiteMoeStaticTrace) -> LiteMoeDp4Spec:
    return LiteMoeDp4Spec.create(trace=trace)


def build_lite_moe_dp4_topology(spec: LiteMoeDp4Spec) -> LiteMoeDp4Topology:
    spec.validate("spec")
    return LiteMoeDp4Topology.create(spec=spec)


def build_lite_moe_dp4_oracle(
    spec: LiteMoeDp4Spec,
    topology: LiteMoeDp4Topology,
) -> LiteMoeDp4Oracle:
    topology.validate_against(spec)
    return LiteMoeDp4Oracle.create(spec=spec, topology=topology)


def _validate_inputs(
    experiment: ExperimentSpec,
    spec: LiteMoeDp4Spec,
    topology: LiteMoeDp4Topology,
    oracle: LiteMoeDp4Oracle,
) -> None:
    experiment.validate("experiment")
    spec.validate("spec")
    topology.validate_against(spec, "topology")
    oracle.validate_against(spec, topology, "oracle")
    infer = experiment.workload.infer
    if (
        experiment.workload.mode is not WorkloadMode.INFER
        or infer is None
        or infer.source is not InferSource.STATIC_PROFILE
        or infer.output is not InferOutput.LOGITS
        or infer.profile is None
        or (experiment.model.H, experiment.model.I) != (16, 32)
        or (infer.profile.prefill_tokens, infer.profile.decode_tokens) != (8, 0)
    ):
        raise SchemaError("requires exact T8/H16/I32 infer host spec", path="experiment")


def build_lite_moe_dp4_ir0_adapter(
    experiment: ExperimentSpec,
    spec: LiteMoeDp4Spec,
    topology: LiteMoeDp4Topology,
    oracle: LiteMoeDp4Oracle,
) -> LiteMoeDp4IR0Adapter:
    _validate_inputs(experiment, spec, topology, oracle)
    graph, bindings = _build_graph(experiment, spec)
    result = LiteMoeDp4IR0Adapter.create(
        experiment_digest=canonical_digest(experiment),
        spec=spec,
        topology=topology,
        oracle=oracle,
        graph=graph,
        p2p_bindings=bindings,
    )
    validate_lite_moe_dp4_ir0_adapter(result, experiment)
    return result


def validate_lite_moe_dp4_ir0_adapter(
    adapter: LiteMoeDp4IR0Adapter,
    experiment: ExperimentSpec,
) -> None:
    adapter.validate()
    _validate_inputs(experiment, adapter.spec, adapter.topology, adapter.oracle)
    expected_graph, expected_bindings = _build_graph(experiment, adapter.spec)
    if (
        adapter.experiment_digest != canonical_digest(experiment)
        or adapter.graph != expected_graph
        or adapter.p2p_bindings != expected_bindings
    ):
        raise SchemaError("adapter is not the exact DP4 graph quotient", path="adapter")
    kinds = Counter(node.kind for node in adapter.graph.nodes)
    if kinds != Counter({OpKind.GEMM: 24, OpKind.ELEMENTWISE: 8, OpKind.P2P: 12}):
        raise SchemaError("graph kind counts changed", path="adapter.graph")
    if sum(
        node.workload.bytes
        for node in adapter.graph.nodes
        if type(node.workload) is P2PByteWorkload
    ) != 384:
        raise SchemaError("graph P2P bytes changed", path="adapter.graph")


def _expert_from_ref(tensor_ref: str) -> int:
    prefix = f"{_INSTANCE}.expert"
    text = tensor_ref[len(prefix):].split(".", 1)[0] if tensor_ref.startswith(prefix) else ""
    if text not in ("0", "1", "2", "3"):
        raise SchemaError("invalid expert tensor ref", path="persistent_states")
    return int(text)


def _placed_state(
    adapter: LiteMoeDp4IR0Adapter,
    context: PlacementContext,
) -> tuple[PersistentStateManifest, tuple[StateAccess, ...]]:
    spaces = {item.die_id: item for item in context.hbm_address_spaces}
    if set(spaces) != {0, 1, 2, 3}:
        raise SchemaError("DP4 placement requires HBM on dies0..3", path="context")
    declarations = []
    remap = {}
    for original in adapter.graph.persistent_states:
        expert = _expert_from_ref(original.identity.tensor_ref or "")
        identity = PersistentStateIdentity.create(
            kind=original.identity.kind,
            instance_ref=original.identity.instance_ref,
            mesh_ref=original.identity.mesh_ref,
            request_ref=None,
            layer_index=None,
            tensor_ref=original.identity.tensor_ref,
            shard_index=expert,
            generation=0,
        )
        placed = PersistentStateDecl.create(
            identity=identity,
            shape=original.shape,
            dtype=original.dtype,
            layout=original.layout,
            lifetime=original.lifetime,
            access=original.access,
        )
        declarations.append(placed)
        remap[original.id] = placed
    accesses = tuple(
        StateAccess.create(
            node_ref=access.node_ref,
            state_ref=remap[access.state_ref].id,
            mode=StateAccessMode.READ,
            rank=remap[access.state_ref].identity.shard_index,
        )
        for access in adapter.graph.state_accesses
    )
    cursors = {die: spaces[die].base_address for die in spaces}
    bindings = []
    for declaration in sorted(
        declarations,
        key=lambda item: (item.identity.shard_index, item.identity.tensor_ref or ""),
    ):
        die = declaration.identity.shard_index
        address = (cursors[die] + 63) // 64 * 64
        bindings.append(
            HbmBinding.create(
                state_ref=declaration.id,
                die_id=die,
                address=address,
                size_bytes=declaration.tensor_bytes,
            )
        )
        cursors[die] = address + declaration.tensor_bytes
    return (
        PersistentStateManifest.create(
            address_spaces=context.hbm_address_spaces,
            declarations=tuple(declarations),
            bindings=tuple(bindings),
        ),
        accesses,
    )


def place_lite_moe_dp4_adapter(
    adapter: LiteMoeDp4IR0Adapter,
    experiment: ExperimentSpec,
    context: PlacementContext,
) -> LiteMoeDp4PlacedIR1:
    validate_lite_moe_dp4_ir0_adapter(adapter, experiment)
    context.validate("context")
    graph = adapter.graph
    instance = graph.instances[0]
    group = replace(
        _expected_group(graph, context, instance, instance.meshes[0]),
        axis=MeshAxisName.EP,
    )
    group.validate("group")
    manifest, accesses = _placed_state(adapter, context)
    placed_graph = IR1.create(
        producer_pass="placement",
        source_ir0_id=graph.id,
        profile=graph.profile,
        fabric=context.fabric,
        instances=(_physical_instance(instance, graph, (group,)),),
        groups=(group,),
        nodes=tuple(_physical_node(node, group.id) for node in graph.nodes),
        values=graph.values,
        edges=graph.edges,
        fusion_candidates=(),
        fused_op_skeletons=(),
        cross_routes=(),
        state_accesses=accesses,
        persistent_state_manifest=manifest,
    )
    result = LiteMoeDp4PlacedIR1.create(
        source=adapter,
        placement_context_id=context.id,
        graph=placed_graph,
        p2p_bindings=adapter.p2p_bindings,
    )
    validate_lite_moe_dp4_placement(result, adapter, context)
    return result


def validate_lite_moe_dp4_placement(
    result: LiteMoeDp4PlacedIR1,
    adapter: LiteMoeDp4IR0Adapter,
    context: PlacementContext,
) -> None:
    result.validate()
    if (
        result.source != adapter
        or result.placement_context_id != context.id
        or result.p2p_bindings != adapter.p2p_bindings
    ):
        raise SchemaError("DP4 placement provenance mismatch", path="placed")
    counts = Counter(
        binding.die_id
        for binding in result.graph.persistent_state_manifest.bindings
    )
    if counts != Counter({0: 3, 1: 3, 2: 3, 3: 3}):
        raise SchemaError("each die must own one expert's three weights", path="placed")


def build_lite_moe_dp4_n4(
    placed: LiteMoeDp4PlacedIR1,
    partition_context: FusionPartitionContext,
    planning_context: InterDiePlanningContext,
) -> LiteMoeDp4N4IR1:
    placed.validate()
    partition_context.validate("partition_context")
    planning_context.validate("planning_context")
    graph = partition_ir1(placed.graph)
    fusion, standalone = plan_ir1(graph, planning_context)
    result = LiteMoeDp4N4IR1.create(
        source=placed,
        partition_context_id=partition_context.id,
        planning_context_id=planning_context.id,
        graph=graph,
        fusion_plans=fusion,
        standalone_plans=standalone,
        p2p_bindings=placed.p2p_bindings,
    )
    validate_lite_moe_dp4_n4(result, placed, partition_context, planning_context)
    return result


def validate_lite_moe_dp4_n4(
    result: LiteMoeDp4N4IR1,
    placed: LiteMoeDp4PlacedIR1,
    partition_context: FusionPartitionContext,
    planning_context: InterDiePlanningContext,
) -> None:
    result.validate()
    if (
        result.source != placed
        or result.partition_context_id != partition_context.id
        or result.planning_context_id != planning_context.id
        or result.graph.nodes != placed.graph.nodes
        or result.graph.values != placed.graph.values
        or result.graph.edges != placed.graph.edges
        or result.graph.groups != placed.graph.groups
        or result.graph.persistent_state_manifest
        != placed.graph.persistent_state_manifest
        or result.graph.state_accesses != placed.graph.state_accesses
    ):
        raise SchemaError("N4 is not exact placed provenance", path="n4")


__all__ = [
    "build_lite_moe_dp4_ir0_adapter",
    "build_lite_moe_dp4_n4",
    "build_lite_moe_dp4_oracle",
    "build_lite_moe_dp4_spec",
    "build_lite_moe_dp4_topology",
    "place_lite_moe_dp4_adapter",
    "validate_lite_moe_dp4_ir0_adapter",
    "validate_lite_moe_dp4_n4",
    "validate_lite_moe_dp4_placement",
]
