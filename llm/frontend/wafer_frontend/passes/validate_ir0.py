"""Semantic cross-validation for a materialized naive Dense IR-0 graph."""

from __future__ import annotations

from collections import deque
from math import prod

from ..errors import SchemaError, UnsupportedFeatureError
from ..schema.common import (
    DType,
    MeshAxisName,
    ProfileKey,
    TensorValue,
    UINT64_MAX,
)
from ..schema.ir0 import (
    AttentionMode,
    AttentionWorkload,
    CollectiveKind,
    CollectiveRole,
    CollectiveWorkload,
    CrossEntropyForwardWorkload,
    CrossEntropyBackwardWorkload,
    CrossEntropyReduction,
    EmbeddingTablePlacement,
    EmbeddingWorkload,
    EdgeKind,
    EffectKind,
    GemmPartition,
    GemmWorkload,
    IR0,
    JobKind,
    LogicalRole,
    LogicalNode,
    NumericalPolicy,
    StateAccess,
    StateAccessMode,
    OpKind,
    OpPhase,
    ReduceOp,
    ResidualWorkload,
    RmsNormWorkload,
    RopeQkWorkload,
    GreedySampleWorkload,
    SwiGluWorkload,
    SgdUpdateWorkload,
)
from ..schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    StateKind,
)
from ..schema.stage3_profile import Stage3ProfileMode


_DTYPE_BYTES = {DType.FP16: 2, DType.FP32: 4, DType.INT32: 4}


def _fail(message: str, path: str) -> None:
    raise SchemaError(message, path=path)


def _tensor_bytes(value: TensorValue, path: str) -> int:
    width = _DTYPE_BYTES.get(value.dtype)
    if width is None:
        _fail("TensorValue dtype is unsupported by the Dense MVP", path)
    result = prod(value.shape) * width
    if result > UINT64_MAX:
        _fail("tensor byte count overflows uint64", path)
    return result


def _rank_shape(
    value: TensorValue,
    axis_sizes: dict[MeshAxisName, int],
    path: str,
) -> tuple[int, ...]:
    result = []
    for index, (extent, axis) in enumerate(
        zip(value.shape, value.sharding.dim_map)
    ):
        if axis is None:
            result.append(extent)
            continue
        size = axis_sizes.get(axis)
        if size is None:
            _fail("sharding axis is absent from the referenced mesh", f"{path}.sharding.dim_map[{index}]")
        if extent % size != 0:
            _fail("sharded tensor extent must divide evenly by its mesh axis", f"{path}.shape[{index}]")
        result.append(extent // size)
    return tuple(result)


def _is_replicated(value: TensorValue) -> bool:
    return all(axis is None for axis in value.sharding.dim_map) and not value.sharding.partial


def _exact_sharding(
    value: TensorValue,
    dim_map: tuple[MeshAxisName | None, ...],
    partial: tuple[MeshAxisName, ...],
    path: str,
) -> None:
    if value.sharding.dim_map != dim_map or value.sharding.partial != partial:
        _fail("tensor sharding does not match the Dense operator contract", f"{path}.sharding")


def _require_pure(node: LogicalNode, path: str) -> None:
    if (
        node.effects.kind is not EffectKind.PURE
        or node.effects.effect_token is not None
        or node.effects.alias_set is not None
    ):
        _fail("Dense compute/collective node must be pure", f"{path}.effects")


def _validate_dependency_dag(graph: IR0, path: str) -> None:
    control_pairs: set[tuple[str, str]] = set()
    indegree = {node.id: 0 for node in graph.nodes}
    dependents: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for index, edge in enumerate(graph.edges):
        if edge.kind is EdgeKind.CONTROL:
            pair = (edge.source_node, edge.destination_node)
            if pair in control_pairs:
                _fail("duplicate control dependency", f"{path}.edges[{index}]")
            control_pairs.add(pair)
        indegree[edge.destination_node] += 1
        dependents[edge.source_node].append(edge.destination_node)
    ready = deque(node.id for node in graph.nodes if indegree[node.id] == 0)
    visited = 0
    while ready:
        current = ready.popleft()
        visited += 1
        for dependent in dependents[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    if visited != len(graph.nodes):
        _fail("node dependency graph contains a cycle", f"{path}.edges")


class DenseIR0Validator:
    """Fail-closed semantic validator for the Dense naive MVP graph."""

    @staticmethod
    def validate(graph: IR0, path: str = "ir0") -> None:
        graph.validate(path)
        _validate_dependency_dag(graph, path)

        nodes = {node.id: node for node in graph.nodes}
        values = {value.id: value for value in graph.values}
        meshes = {
            mesh.id: mesh
            for instance in graph.instances
            for mesh in instance.meshes
        }
        axis_sizes = {
            mesh_id: {axis.name: axis.size for axis in mesh.axes}
            for mesh_id, mesh in meshes.items()
        }
        stage4_profiles: dict[str, set[ProfileKey]] | None = None
        node_profiles: dict[str, ProfileKey] = {}
        if graph.instance_profiles and graph.pd_plan_id is not None:
            stage4_profiles = {}
            for binding in graph.instance_profiles:
                stage4_profiles.setdefault(binding.instance_ref, set()).add(
                    binding.profile
                )
            instance_ids = {instance.id for instance in graph.instances}
            if set(stage4_profiles) != instance_ids:
                _fail(
                    "Stage 4 instance profile bindings must exactly cover instances",
                    f"{path}.instance_profiles",
                )
            node_profiles = {
                binding.node_ref: binding.profile
                for binding in graph.node_profiles
            }
        local_shapes: dict[str, tuple[int, ...]] = {}

        for index, value in enumerate(graph.values):
            value_path = f"{path}.values[{index}]"
            if type(value.dtype) is not DType:
                _fail("TensorValue dtype must be a DType", f"{value_path}.dtype")
            sizes = axis_sizes[value.sharding.mesh_ref]
            for partial_index, axis in enumerate(value.sharding.partial):
                if axis not in sizes:
                    _fail(
                        "partial axis is absent from the referenced mesh",
                        f"{value_path}.sharding.partial[{partial_index}]",
                    )
            local_shapes[value.id] = _rank_shape(value, sizes, value_path)
            endpoint_ids = (
                (() if value.producer is None else (value.producer,))
                + value.consumers
            )
            for endpoint_id in endpoint_ids:
                if nodes[endpoint_id].mesh_ref != value.sharding.mesh_ref:
                    _fail(
                        "value sharding mesh must equal every producer/consumer node mesh",
                        f"{value_path}.sharding.mesh_ref",
                    )

        # Check this before consumer operator contracts so an illegal
        # partial->postprocess edge receives the mathematical-dependency
        # diagnostic rather than a secondary elementwise shape error.
        for index, value in enumerate(graph.values):
            if not value.sharding.partial:
                continue
            value_path = f"{path}.values[{index}]"
            if value.sharding.partial != (MeshAxisName.TP,):
                _fail("Dense MVP partial values must be partial only over TP", f"{value_path}.sharding.partial")
            if value.producer is None:
                _fail("partial value requires a row-parallel GEMM producer", f"{value_path}.producer")
            producer = nodes[value.producer]
            if (
                producer.kind is not OpKind.GEMM
                or not isinstance(producer.workload, GemmWorkload)
                or producer.workload.partition is not GemmPartition.ROW_PARALLEL
            ):
                _fail("partial value requires a row-parallel GEMM producer", f"{value_path}.producer")
            if len(value.consumers) != 1:
                _fail("partial value must have exactly one reduction consumer", f"{value_path}.consumers")
            consumer = nodes[value.consumers[0]]
            if (
                consumer.kind is not OpKind.COLLECTIVE
                or not isinstance(consumer.workload, CollectiveWorkload)
                or consumer.workload.collective is not CollectiveKind.REDUCE_SCATTER
                or consumer.workload.reduce_op is not ReduceOp.SUM
                or consumer.inputs != (value.id,)
            ):
                _fail("partial value must flow directly into one SUM ReduceScatter", f"{value_path}.consumers")

        for index, node in enumerate(graph.nodes):
            node_path = f"{path}.nodes[{index}]"
            profile = (
                graph.profile
                if stage4_profiles is None
                else node_profiles.get(
                    node.id,
                    next(iter(stage4_profiles[node.instance_id])),
                )
            )
            node_values = tuple(values[ref] for ref in (*node.inputs, *node.outputs))
            if any(value.sharding.mesh_ref != node.mesh_ref for value in node_values):
                _fail("node operands must all use the node mesh", f"{node_path}.mesh_ref")
            if node.math.accumulation_dtype is not DType.FP32:
                _fail("Dense MVP requires FP32 accumulation", f"{node_path}.math.accumulation_dtype")
            if type(node.math.numerical_policy) is not NumericalPolicy:
                _fail("must be a NumericalPolicy", f"{node_path}.math.numerical_policy")

            if node.kind is OpKind.GEMM:
                if node.phase is OpPhase.WGRAD:
                    DenseIR0Validator._validate_lm_head_wgrad(
                        node, values, local_shapes, node_path
                    )
                else:
                    DenseIR0Validator._validate_gemm(
                        node, values, local_shapes, axis_sizes[node.mesh_ref], node_path
                    )
            elif node.kind is OpKind.NORM:
                DenseIR0Validator._validate_norm(
                    node,
                    values,
                    local_shapes,
                    node_path,
                    profile=(profile if stage4_profiles is not None else None),
                )
            elif node.kind is OpKind.ELEMENTWISE:
                DenseIR0Validator._validate_elementwise(
                    node, values, local_shapes, node_path
                )
            elif node.kind is OpKind.ATTENTION:
                DenseIR0Validator._validate_attention(
                    node,
                    values,
                    local_shapes,
                    axis_sizes[node.mesh_ref],
                    profile,
                    node_path,
                )
            elif node.kind is OpKind.COLLECTIVE:
                DenseIR0Validator._validate_collective(
                    node, values, local_shapes, axis_sizes[node.mesh_ref], node_path
                )
            elif node.kind is OpKind.EMBEDDING:
                DenseIR0Validator._validate_embedding(
                    node,
                    values,
                    local_shapes,
                    axis_sizes[node.mesh_ref],
                    profile,
                    node_path,
                )
            elif node.kind is OpKind.ROPE:
                DenseIR0Validator._validate_rope(
                    node,
                    values,
                    local_shapes,
                    axis_sizes[node.mesh_ref],
                    profile,
                    node_path,
                )
            elif node.kind is OpKind.SAMPLING:
                DenseIR0Validator._validate_sampling(
                    node, values, local_shapes, profile, node_path
                )
            elif node.kind is OpKind.CE_FORWARD:
                DenseIR0Validator._validate_cross_entropy(
                    node, values, local_shapes, profile, node_path
                )
            elif node.kind is OpKind.CE_BACKWARD:
                DenseIR0Validator._validate_cross_entropy_backward(
                    node, values, local_shapes, profile, node_path
                )
            elif node.kind is OpKind.OPTIMIZER_UPDATE:
                DenseIR0Validator._validate_sgd_update(
                    node, values, local_shapes, node_path
                )
            elif node.kind is OpKind.P2P:
                raise UnsupportedFeatureError(
                    "Dense naive IR-0 does not support P2P nodes", path=node_path
                )
        DenseIR0Validator._validate_persistent_states(
            graph,
            values,
            local_shapes,
            axis_sizes,
            stage4_profiles,
            node_profiles,
            path,
        )
        DenseIR0Validator._validate_job_contract(graph, path)

    @staticmethod
    def _validate_job_contract(graph: IR0, path: str) -> None:
        attentions = tuple(
            node for node in graph.nodes if node.kind is OpKind.ATTENTION
        )
        ce_nodes = tuple(
            node for node in graph.nodes if node.kind is OpKind.CE_FORWARD
        )
        sampling_nodes = tuple(
            node for node in graph.nodes if node.kind is OpKind.SAMPLING
        )
        ce_backward_nodes = tuple(
            node for node in graph.nodes if node.kind is OpKind.CE_BACKWARD
        )
        optimizer_nodes = tuple(
            node for node in graph.nodes if node.kind is OpKind.OPTIMIZER_UPDATE
        )
        if graph.job is JobKind.INFER:
            if ce_nodes or ce_backward_nodes or optimizer_nodes or any(
                isinstance(node.workload, AttentionWorkload)
                and node.workload.mode is AttentionMode.TRAIN_FORWARD
                for node in attentions
            ):
                _fail(
                    "infer graphs cannot contain train-forward attention or cross entropy",
                    f"{path}.nodes",
                )
            return
        if graph.job is not JobKind.TRAIN:
            _fail("unsupported Dense job kind", f"{path}.job")
        if ce_backward_nodes or optimizer_nodes:
            DenseIR0Validator._validate_lite_train_job_contract(graph, path)
            return
        if len(graph.instances) != 1:
            _fail("forward train requires exactly one logical instance", f"{path}.instances")
        instance = graph.instances[0]
        if (
            instance.role is not LogicalRole.TRAIN
            or instance.replicas != 1
            or instance.parallel.pp != 1
            or instance.parallel.ep != 1
        ):
            _fail("forward train instance geometry is not exact", f"{path}.instances[0]")
        if graph.train is None:
            _fail("forward train requires TrainStructure", f"{path}.train")
        if (
            graph.profile.decode_tokens != 0
            or graph.profile.prefill_tokens == 0
            or graph.profile.context_sum != graph.profile.prefill_tokens
            or graph.profile.context_max * graph.profile.num_seqs
            != graph.profile.prefill_tokens
            or graph.profile.kv_pages != 0
        ):
            _fail("forward train profile geometry is not exact", f"{path}.profile")
        if len(ce_nodes) != 1 or sampling_nodes:
            _fail("forward train requires exactly one CE and no sampling", f"{path}.nodes")
        if any(
            node.phase is not OpPhase.FWD or node.stage != 0
            for node in graph.nodes
        ):
            _fail("forward train contains a non-forward node", f"{path}.nodes")
        if any(
            not isinstance(node.workload, AttentionWorkload)
            or node.workload.mode is not AttentionMode.TRAIN_FORWARD
            for node in attentions
        ):
            _fail("forward train attention mode is not exact", f"{path}.nodes")
        if any(
            declaration.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
            for declaration in graph.persistent_states
        ):
            _fail("forward train must not materialize KV state", f"{path}.persistent_states")

    @staticmethod
    def _validate_lite_train_job_contract(graph: IR0, path: str) -> None:
        if len(graph.instances) != 1:
            _fail("S2-Lite requires exactly one logical instance", f"{path}.instances")
        instance = graph.instances[0]
        expected_dp = (
            2
            if graph.producer_pass == "s2_lite_dp2_rooted_ar_source"
            else 1
        )
        if (
            instance.role is not LogicalRole.TRAIN
            or instance.replicas != 1
            or instance.parallel.tp != 1
            or instance.parallel.dp != expected_dp
            or instance.parallel.pp != 1
            or instance.parallel.ep != 1
            or instance.parallel.sp
        ):
            _fail(
                f"S2-Lite instance geometry must be DP={expected_dp}, TP=PP=EP=1",
                f"{path}.instances[0]",
            )
        if graph.train is None or graph.train.micro_batch_count != 1:
            _fail("S2-Lite requires one microbatch", f"{path}.train")
        if (
            graph.profile.decode_tokens != 0
            or graph.profile.prefill_tokens == 0
            or graph.profile.context_sum != graph.profile.prefill_tokens
            or graph.profile.context_max * graph.profile.num_seqs
            != graph.profile.prefill_tokens
            or graph.profile.kv_pages != 0
        ):
            _fail("S2-Lite train profile geometry is not exact", f"{path}.profile")
        if any(node.kind is OpKind.SAMPLING for node in graph.nodes):
            _fail("S2-Lite train forbids sampling", f"{path}.nodes")
        if any(
            type(node.workload) is not AttentionWorkload
            or node.workload.mode is not AttentionMode.TRAIN_FORWARD
            for node in graph.nodes
            if node.kind is OpKind.ATTENTION
        ):
            _fail("S2-Lite attention must remain train-forward", f"{path}.nodes")
        if any(
            state.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
            for state in graph.persistent_states
        ):
            _fail("S2-Lite must not materialize KV state", f"{path}.persistent_states")
        ce_forward = tuple(
            node for node in graph.nodes if node.kind is OpKind.CE_FORWARD
        )
        ce_backward = tuple(
            node for node in graph.nodes if node.kind is OpKind.CE_BACKWARD
        )
        wgrad = tuple(
            node
            for node in graph.nodes
            if node.kind is OpKind.GEMM and node.phase is OpPhase.WGRAD
        )
        optimizer = tuple(
            node for node in graph.nodes if node.kind is OpKind.OPTIMIZER_UPDATE
        )
        if tuple(map(len, (ce_forward, ce_backward, wgrad, optimizer))) != (1, 1, 1, 1):
            _fail(
                "S2-Lite requires one CE forward/backward, LM-head WGRAD and SGD update",
                f"{path}.nodes",
            )
        ce_fwd, ce_bwd, wgrad_node, update = (
            ce_forward[0], ce_backward[0], wgrad[0], optimizer[0]
        )
        forward_nodes = tuple(
            node
            for node in graph.nodes
            if node not in (ce_bwd, wgrad_node, update)
        )
        if any(node.phase is not OpPhase.FWD or node.stage != 0 for node in forward_nodes):
            _fail("S2-Lite backbone must remain forward-only", f"{path}.nodes")
        if (
            ce_bwd.phase is not OpPhase.DGRAD
            or wgrad_node.phase is not OpPhase.WGRAD
            or update.phase is not OpPhase.UPDATE
            or any(node.stage != 0 for node in (ce_bwd, wgrad_node, update))
        ):
            _fail("S2-Lite phases/stages are not exact", f"{path}.nodes")
        values = {value.id: value for value in graph.values}
        lm_head = next(
            (
                node
                for node in graph.nodes
                if node.id.endswith(".lm_head")
                and node.kind is OpKind.GEMM
                and node.phase is OpPhase.FWD
            ),
            None,
        )
        if lm_head is None:
            _fail("S2-Lite requires one canonical LM head", f"{path}.nodes")
        assert lm_head is not None
        logits_id = lm_head.outputs[0]
        labels_id = ce_fwd.inputs[1]
        if (
            ce_bwd.inputs[:2] != (logits_id, labels_id)
            or wgrad_node.inputs
            != (lm_head.inputs[0], ce_bwd.outputs[0])
            or update.inputs != (lm_head.inputs[1], wgrad_node.outputs[0])
            or values[ce_bwd.inputs[2]].producer is not None
            or values[ce_bwd.inputs[2]].consumers != (ce_bwd.id,)
            or values[ce_bwd.outputs[0]].consumers != (wgrad_node.id,)
            or values[wgrad_node.outputs[0]].consumers != (update.id,)
            or values[update.outputs[0]].consumers
        ):
            _fail("S2-Lite backward/update value lineage is not exact", f"{path}.nodes")
        control_edges = tuple(edge for edge in graph.edges if edge.kind is EdgeKind.CONTROL)
        if (
            len(control_edges) != 1
            or control_edges[0].source_node != ce_fwd.id
            or control_edges[0].destination_node != ce_bwd.id
        ):
            _fail("CE forward must control CE backward exactly once", f"{path}.edges")
        if any(
            node.phase in (OpPhase.DGRAD, OpPhase.WGRAD, OpPhase.UPDATE)
            for node in graph.nodes
            if node not in (ce_bwd, wgrad_node, update)
        ):
            _fail("backbone gradients/updates are forbidden", f"{path}.nodes")

    @staticmethod
    def _validate_persistent_states(
        graph: IR0,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        axis_sizes: dict[str, dict[MeshAxisName, int]],
        stage4_profiles: dict[str, set[ProfileKey]] | None,
        node_profiles: dict[str, ProfileKey],
        path: str,
    ) -> None:
        if any(node.kind is OpKind.CE_BACKWARD for node in graph.nodes):
            DenseIR0Validator._validate_lite_persistent_states(
                graph, values, local_shapes, axis_sizes, path
            )
            return
        expected_states: list[PersistentStateDecl] = []
        expected_accesses: list[StateAccess] = []
        for node in graph.nodes:
            profile = (
                graph.profile
                if stage4_profiles is None
                else node_profiles.get(
                    node.id,
                    next(iter(stage4_profiles[node.instance_id])),
                )
            )
            tp = axis_sizes[node.mesh_ref].get(MeshAxisName.TP)
            if tp is None:
                _fail("Dense state requires a TP mesh axis", f"{path}.persistent_states")
            if node.kind in (OpKind.GEMM, OpKind.NORM, OpKind.EMBEDDING):
                if len(node.inputs) != 2:
                    continue
                weight = values[node.inputs[1]]
                for rank in range(tp):
                    identity = PersistentStateIdentity.create(
                        kind=StateKind.PARAMETER,
                        instance_ref=node.instance_id,
                        mesh_ref=node.mesh_ref,
                        request_ref=None,
                        layer_index=None,
                        tensor_ref=weight.id,
                        shard_index=rank,
                        generation=0,
                    )
                    declaration = PersistentStateDecl.create(
                        identity=identity,
                        shape=local_shapes[weight.id],
                        dtype=weight.dtype,
                        layout=weight.logical_layout,
                        lifetime=PersistentStateLifetime.PERSISTENT,
                        access=PersistentStateAccess.READ_ONLY,
                    )
                    expected_states.append(declaration)
                    expected_accesses.append(
                        StateAccess.create(
                            node_ref=node.id,
                            state_ref=declaration.id,
                            mode=StateAccessMode.READ,
                            rank=rank,
                        )
                    )
            elif node.kind is OpKind.ATTENTION:
                workload = node.workload
                assert isinstance(workload, AttentionWorkload)
                if workload.mode is AttentionMode.TRAIN_FORWARD:
                    continue
                marker = ".layer"
                if marker not in node.id:
                    _fail(
                        "Dense attention state requires a canonical layer id",
                        f"{path}.persistent_states",
                    )
                tail = node.id.rsplit(marker, 1)[1]
                layer_text, separator, suffix = tail.partition(".")
                if not separator or suffix != "attention" or not layer_text.isdigit():
                    _fail(
                        "Dense attention state requires a canonical layer id",
                        f"{path}.persistent_states",
                    )
                layer_index = int(layer_text)
                if profile.context_sum == 0:
                    _fail(
                        "persistent KV state requires context_sum > 0",
                        f"{path}.profile.context_sum",
                    )
                layout = "THD_packed_kv_head_tp" if tp > 1 else "THD_packed"

                def append_expected_kv(
                    *,
                    rank: int,
                    request_ref: str,
                    capacity_tokens: int,
                    read_tokens: int,
                    write_start: int,
                    write_tokens: int,
                    explicit_view: bool,
                ) -> None:
                    kv_shape = (
                        capacity_tokens,
                        workload.rank_num_kv_heads,
                        workload.head_dim,
                    )
                    for kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
                        identity = PersistentStateIdentity.create(
                            kind=kind,
                            instance_ref=node.instance_id,
                            mesh_ref=node.mesh_ref,
                            request_ref=request_ref,
                            layer_index=layer_index,
                            tensor_ref=None,
                            shard_index=rank,
                            generation=0,
                        )
                        declaration = PersistentStateDecl.create(
                            identity=identity,
                            shape=kv_shape,
                            dtype=workload.dtype,
                            layout=layout,
                            lifetime=PersistentStateLifetime.PERSISTENT,
                            access=PersistentStateAccess.READ_WRITE,
                        )
                        expected_states.append(declaration)
                        expected_accesses.append(
                            StateAccess.create(
                                node_ref=node.id,
                                state_ref=declaration.id,
                                mode=(
                                    StateAccessMode.READ_WRITE
                                    if read_tokens
                                    else StateAccessMode.WRITE
                                ),
                                rank=rank,
                                read_offset=(0, 0, 0)
                                if explicit_view and read_tokens
                                else None,
                                read_shape=(
                                    read_tokens,
                                    workload.rank_num_kv_heads,
                                    workload.head_dim,
                                )
                                if explicit_view and read_tokens
                                else None,
                                write_offset=(write_start, 0, 0)
                                if explicit_view or read_tokens
                                else None,
                                write_shape=(
                                    write_tokens,
                                    workload.rank_num_kv_heads,
                                    workload.head_dim,
                                )
                                if explicit_view or read_tokens
                                else None,
                            )
                        )
                for rank in range(tp):
                    exact_profile = workload.exact_profile
                    if exact_profile is None:
                        append_expected_kv(
                            rank=rank,
                            request_ref=profile.stable_id(),
                            capacity_tokens=profile.context_sum,
                            read_tokens=0
                            if workload.mode is AttentionMode.PREFILL
                            else (
                                profile.context_sum
                                if stage4_profiles is None
                                or len(stage4_profiles[node.instance_id]) > 1
                                else 0
                            ),
                            write_start=0
                            if workload.mode is AttentionMode.PREFILL
                            else profile.context_sum - profile.decode_tokens,
                            write_tokens=workload.query_tokens,
                            explicit_view=False,
                        )
                    else:
                        for request in exact_profile.requests:
                            append_expected_kv(
                                rank=rank,
                                request_ref=(
                                    request.request_ref
                                    if stage4_profiles is not None
                                    else f"{exact_profile.id}:{request.request_ref}"
                                ),
                                capacity_tokens=request.kv_span.capacity_tokens,
                                read_tokens=(
                                    request.kv_read_tokens
                                    if stage4_profiles is None
                                    or len(stage4_profiles[node.instance_id]) > 1
                                    else 0
                                ),
                                write_start=(
                                    request.context_tokens - request.query_tokens
                                ),
                                write_tokens=request.query_tokens,
                                explicit_view=True,
                            )

        expected_state_index: dict[str, PersistentStateDecl] = {}
        for declaration in expected_states:
            previous = expected_state_index.get(declaration.id)
            if previous is not None and previous != declaration:
                _fail(
                    "fused state identity resolves to incompatible declarations",
                    f"{path}.persistent_states",
                )
            expected_state_index[declaration.id] = declaration
        expected_state_tuple = tuple(
            sorted(
                expected_state_index.values(),
                key=lambda item: (item.identity.id, item.id),
            )
        )
        executable_states = tuple(
            declaration
            for declaration in graph.persistent_states
            if declaration.identity.kind is not StateKind.OPTIMIZER_RESERVED
        )
        if executable_states != expected_state_tuple:
            _fail(
                "Dense parameter/KV state declarations are not exact",
                f"{path}.persistent_states",
            )
        expected_access_tuple = tuple(
            sorted(
                expected_accesses,
                key=lambda item: (item.node_ref, item.state_ref, item.rank, item.id),
            )
        )
        if graph.state_accesses != expected_access_tuple:
            _fail(
                "Dense parameter/KV state accesses are not exact",
                f"{path}.state_accesses",
            )

    @staticmethod
    def _validate_lite_persistent_states(
        graph: IR0,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        axis_sizes: dict[str, dict[MeshAxisName, int]],
        path: str,
    ) -> None:
        updates = tuple(
            node for node in graph.nodes if node.kind is OpKind.OPTIMIZER_UPDATE
        )
        if len(updates) != 1:
            _fail(
                "S2-Lite state requires one optimizer update",
                f"{path}.persistent_states",
            )
        update = updates[0]
        trainable_weight_id = update.inputs[0]
        expected_states: list[PersistentStateDecl] = []
        expected_accesses: list[StateAccess] = []
        for node in graph.nodes:
            if node.phase is not OpPhase.FWD or node.kind not in (
                OpKind.GEMM,
                OpKind.NORM,
                OpKind.EMBEDDING,
            ):
                continue
            if len(node.inputs) != 2:
                continue
            weight = values[node.inputs[1]]
            tp = axis_sizes[node.mesh_ref].get(MeshAxisName.TP)
            if tp != 1:
                _fail("S2-Lite state requires TP1", f"{path}.persistent_states")
            trainable = weight.id == trainable_weight_id
            identity = PersistentStateIdentity.create(
                kind=(
                    StateKind.TRAINABLE_PARAMETER
                    if trainable
                    else StateKind.PARAMETER
                ),
                instance_ref=node.instance_id,
                mesh_ref=node.mesh_ref,
                request_ref=None,
                layer_index=None,
                tensor_ref=weight.id,
                shard_index=0,
                generation=0,
            )
            declaration = PersistentStateDecl.create(
                identity=identity,
                shape=local_shapes[weight.id],
                dtype=weight.dtype,
                layout=weight.logical_layout,
                lifetime=PersistentStateLifetime.PERSISTENT,
                access=(
                    PersistentStateAccess.READ_WRITE
                    if trainable
                    else PersistentStateAccess.READ_ONLY
                ),
            )
            expected_states.append(declaration)
            expected_accesses.append(
                StateAccess.create(
                    node_ref=node.id,
                    state_ref=declaration.id,
                    mode=StateAccessMode.READ,
                    rank=0,
                )
            )
            if trainable:
                expected_accesses.append(
                    StateAccess.create(
                        node_ref=update.id,
                        state_ref=declaration.id,
                        mode=StateAccessMode.READ_WRITE,
                        rank=0,
                    )
                )
        expected_state_tuple = tuple(
            sorted(expected_states, key=lambda item: (item.identity.id, item.id))
        )
        if graph.persistent_states != expected_state_tuple:
            _fail(
                "S2-Lite frozen/trainable parameter declarations are not exact",
                f"{path}.persistent_states",
            )
        expected_access_tuple = tuple(
            sorted(
                expected_accesses,
                key=lambda item: (item.node_ref, item.state_ref, item.rank, item.id),
            )
        )
        if graph.state_accesses != expected_access_tuple:
            _fail(
                "S2-Lite parameter accesses are not exact",
                f"{path}.state_accesses",
            )

    @staticmethod
    def _validate_lm_head_wgrad(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        path: str,
    ) -> None:
        _require_pure(node, path)
        if (
            node.impl_ref != "lm_head_wgrad"
            or len(node.inputs) != 2
            or len(node.outputs) != 1
        ):
            _fail("LM-head WGRAD operator contract is not exact", path)
        workload = node.workload
        assert type(workload) is GemmWorkload
        hidden, logits_gradient = (values[ref] for ref in node.inputs)
        weight_gradient = values[node.outputs[0]]
        rows, hidden_size = hidden.shape
        gradient_rows, vocabulary = logits_gradient.shape
        if (
            gradient_rows != rows
            or weight_gradient.shape != (hidden_size, vocabulary)
            or workload.logical_shape != (hidden_size, vocabulary, rows)
            or workload.rank_shape != workload.logical_shape
            or workload.partition is not GemmPartition.REPLICATED
            or workload.dtype is not DType.FP16
            or hidden.dtype is not DType.FP16
            or logits_gradient.dtype is not DType.FP16
            or weight_gradient.dtype is not DType.FP32
            or local_shapes[hidden.id] != hidden.shape
            or local_shapes[logits_gradient.id] != logits_gradient.shape
            or local_shapes[weight_gradient.id] != weight_gradient.shape
            or not all(
                _is_replicated(value)
                for value in (hidden, logits_gradient, weight_gradient)
            )
        ):
            _fail("LM-head WGRAD geometry/dtypes are not exact", path)

    @staticmethod
    def _validate_cross_entropy_backward(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        profile: ProfileKey,
        path: str,
    ) -> None:
        _require_pure(node, path)
        workload = node.workload
        assert type(workload) is CrossEntropyBackwardWorkload
        if (
            node.impl_ref != "cross_entropy_backward"
            or len(node.inputs) != 3
            or len(node.outputs) != 1
            or workload.profile != profile
            or workload.reduction is not CrossEntropyReduction.NONE
        ):
            _fail("cross entropy backward operator contract is not exact", path)
        logits, labels, loss_gradient = (values[ref] for ref in node.inputs)
        logits_gradient = values[node.outputs[0]]
        if (
            workload.logical_logits_shape != logits.shape
            or workload.rank_logits_shape != local_shapes[logits.id]
            or workload.logical_label_shape != labels.shape
            or workload.rank_label_shape != local_shapes[labels.id]
            or workload.logical_loss_gradient_shape != loss_gradient.shape
            or workload.rank_loss_gradient_shape != local_shapes[loss_gradient.id]
            or workload.logical_logits_gradient_shape != logits_gradient.shape
            or workload.rank_logits_gradient_shape != local_shapes[logits_gradient.id]
            or logits.dtype is not DType.FP16
            or labels.dtype is not DType.INT32
            or loss_gradient.dtype is not DType.FP32
            or logits_gradient.dtype is not DType.FP16
            or loss_gradient.producer is not None
            or not all(
                _is_replicated(value)
                for value in (logits, labels, loss_gradient, logits_gradient)
            )
        ):
            _fail("cross entropy backward tensor boundary is not exact", path)

    @staticmethod
    def _validate_sgd_update(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        path: str,
    ) -> None:
        workload = node.workload
        assert type(workload) is SgdUpdateWorkload
        if (
            node.impl_ref != "sgd_update"
            or len(node.inputs) != 2
            or len(node.outputs) != 1
            or node.effects.kind is not EffectKind.INPLACE
            or node.effects.effect_token != f"{node.id}.effect"
        ):
            _fail("SGD update operator/effect contract is not exact", path)
        weight, gradient = (values[ref] for ref in node.inputs)
        updated = values[node.outputs[0]]
        if (
            node.effects.alias_set != f"trainable:{weight.id}"
            or updated.alias_set != node.effects.alias_set
            or workload.logical_weight_shape != weight.shape
            or workload.rank_weight_shape != local_shapes[weight.id]
            or workload.logical_gradient_shape != gradient.shape
            or workload.rank_gradient_shape != local_shapes[gradient.id]
            or workload.logical_updated_weight_shape != updated.shape
            or workload.rank_updated_weight_shape != local_shapes[updated.id]
            or workload.weight_dtype is not weight.dtype
            or workload.gradient_dtype is not gradient.dtype
            or workload.updated_weight_dtype is not updated.dtype
            or weight.producer is not None
            or updated.consumers
            or not all(_is_replicated(value) for value in (weight, gradient, updated))
        ):
            _fail("SGD weight/gradient/update boundary is not exact", path)

    def _validate_gemm(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        axis_sizes: dict[MeshAxisName, int],
        path: str,
    ) -> None:
        _require_pure(node, path)
        if node.impl_ref != "matmul_forward" or len(node.inputs) != 2 or len(node.outputs) != 1:
            _fail("GEMM requires matmul_forward with two inputs and one output", path)
        workload = node.workload
        assert isinstance(workload, GemmWorkload)
        activation, weight = (values[ref] for ref in node.inputs)
        output = values[node.outputs[0]]
        if len(activation.shape) != 2 or len(weight.shape) != 2 or len(output.shape) != 2:
            _fail("Dense GEMM operands must be rank-2 tensors", path)
        m, k = activation.shape
        weight_k, n = weight.shape
        if weight_k != k or output.shape != (m, n):
            _fail("GEMM activation/weight/output shapes are not MxK, KxN, MxN", f"{path}.workload.logical_shape")
        if workload.logical_shape != (m, n, k):
            _fail("GEMM logical_shape must exactly derive from operand values", f"{path}.workload.logical_shape")
        if any(value.dtype is not workload.dtype for value in (activation, weight, output)):
            _fail("GEMM operand and workload dtypes must match", f"{path}.workload.dtype")
        expected_rank = (
            local_shapes[activation.id][0],
            local_shapes[output.id][1],
            local_shapes[activation.id][1],
        )
        if workload.rank_shape != expected_rank or local_shapes[weight.id] != (expected_rank[2], expected_rank[1]) or local_shapes[output.id] != (expected_rank[0], expected_rank[1]):
            _fail("GEMM rank_shape must exactly derive from rank-local operand values", f"{path}.workload.rank_shape")

        tp = axis_sizes.get(MeshAxisName.TP)
        if tp is None:
            _fail("Dense GEMM requires a TP mesh axis", f"{path}.mesh_ref")
        replicated_a = (None, None)
        if workload.partition is GemmPartition.REPLICATED:
            if not all(_is_replicated(value) for value in (activation, weight, output)):
                _fail("replicated GEMM requires replicated activation, weight and output", f"{path}.workload.partition")
        elif workload.partition is GemmPartition.COLUMN_PARALLEL:
            if tp <= 1:
                _fail("column-parallel GEMM requires TP>1", f"{path}.workload.partition")
            _exact_sharding(activation, replicated_a, (), f"{path}.inputs[0]")
            _exact_sharding(weight, (None, MeshAxisName.TP), (), f"{path}.inputs[1]")
            _exact_sharding(output, (None, MeshAxisName.TP), (), f"{path}.outputs[0]")
        elif workload.partition is GemmPartition.ROW_PARALLEL:
            if tp <= 1:
                _fail("row-parallel GEMM requires TP>1", f"{path}.workload.partition")
            _exact_sharding(activation, (None, MeshAxisName.TP), (), f"{path}.inputs[0]")
            _exact_sharding(weight, (MeshAxisName.TP, None), (), f"{path}.inputs[1]")
            _exact_sharding(output, (None, None), (MeshAxisName.TP,), f"{path}.outputs[0]")
        else:
            if tp <= 1:
                _fail(
                    "sequence-parallel replicated-weight GEMM requires TP>1",
                    f"{path}.workload.partition",
                )
            _exact_sharding(activation, (MeshAxisName.TP, None), (), f"{path}.inputs[0]")
            _exact_sharding(weight, (None, None), (), f"{path}.inputs[1]")
            _exact_sharding(output, (MeshAxisName.TP, None), (), f"{path}.outputs[0]")

    @staticmethod
    def _validate_embedding(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        axis_sizes: dict[MeshAxisName, int],
        profile: ProfileKey,
        path: str,
    ) -> None:
        _require_pure(node, path)
        if (
            node.impl_ref != "embedding_lookup"
            or len(node.inputs) != 2
            or len(node.outputs) != 1
        ):
            _fail(
                "embedding requires embedding_lookup with token/table inputs and one output",
                path,
            )
        workload = node.workload
        assert type(workload) is EmbeddingWorkload
        indices, table = (values[ref] for ref in node.inputs)
        output = values[node.outputs[0]]
        tp = axis_sizes.get(MeshAxisName.TP)
        if tp is None:
            _fail("embedding requires a TP mesh axis", f"{path}.mesh_ref")
        expected_map = (None,) if tp == 1 else (MeshAxisName.TP,)
        output_map = (None, None) if tp == 1 else (MeshAxisName.TP, None)
        if (
            workload.profile != profile
            or workload.logical_index_shape != indices.shape
            or workload.rank_index_shape != local_shapes[indices.id]
            or workload.logical_table_shape != table.shape
            or workload.rank_table_shape != local_shapes[table.id]
            or workload.logical_output_shape != output.shape
            or workload.rank_output_shape != local_shapes[output.id]
            or workload.table_placement is not EmbeddingTablePlacement.REPLICATED
            or workload.index_dtype is not indices.dtype
            or workload.table_dtype is not table.dtype
            or workload.output_dtype is not output.dtype
        ):
            _fail(
                "embedding workload must exactly match profile and operand geometry",
                f"{path}.workload",
            )
        _exact_sharding(indices, expected_map, (), f"{path}.inputs[0]")
        if not _is_replicated(table):
            _fail("embedding table must be replicated", f"{path}.inputs[1].sharding")
        _exact_sharding(output, output_map, (), f"{path}.outputs[0]")

    @staticmethod
    def _validate_rope(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        axis_sizes: dict[MeshAxisName, int],
        profile: ProfileKey,
        path: str,
    ) -> None:
        _require_pure(node, path)
        if (
            node.impl_ref != "rope_qk_exact"
            or len(node.inputs) != 1
            or len(node.outputs) != 1
        ):
            _fail("ROPE requires rope_qk_exact with one input and one output", path)
        workload = node.workload
        assert type(workload) is RopeQkWorkload
        input_value = values[node.inputs[0]]
        output_value = values[node.outputs[0]]
        tp = axis_sizes.get(MeshAxisName.TP)
        if tp is None:
            _fail("ROPE requires a TP mesh axis", f"{path}.mesh_ref")
        expected_map = (None, None) if tp == 1 else (None, MeshAxisName.TP)
        if (
            workload.profile != profile
            or workload.logical_input_shape != input_value.shape
            or workload.rank_input_shape != local_shapes[input_value.id]
            or workload.logical_output_shape != output_value.shape
            or workload.rank_output_shape != local_shapes[output_value.id]
            or workload.rank_num_heads != workload.num_heads // tp
            or workload.rank_num_kv_heads != workload.num_kv_heads // tp
            or workload.dtype is not input_value.dtype
            or workload.dtype is not output_value.dtype
            or input_value.logical_layout != output_value.logical_layout
        ):
            _fail(
                "ROPE workload must exactly preserve packed Q/K/V geometry and TP heads",
                f"{path}.workload",
            )
        _exact_sharding(input_value, expected_map, (), f"{path}.inputs[0]")
        _exact_sharding(output_value, expected_map, (), f"{path}.outputs[0]")

    @staticmethod
    def _validate_sampling(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        profile: ProfileKey,
        path: str,
    ) -> None:
        _require_pure(node, path)
        if (
            node.impl_ref != "greedy_sample"
            or len(node.inputs) != 1
            or len(node.outputs) != 1
        ):
            _fail("greedy sampling requires one logits input and one output", path)
        workload = node.workload
        assert type(workload) is GreedySampleWorkload
        logits = values[node.inputs[0]]
        output = values[node.outputs[0]]
        if (
            workload.profile != profile
            or workload.logical_logits_shape != logits.shape
            or workload.rank_logits_shape != local_shapes[logits.id]
            or workload.logical_output_shape != output.shape
            or workload.rank_output_shape != local_shapes[output.id]
            or workload.logits_dtype is not logits.dtype
            or workload.output_dtype is not output.dtype
            or not _is_replicated(logits)
            or not _is_replicated(output)
        ):
            _fail(
                "greedy sampling workload must exactly match TP1 logits/output values",
                f"{path}.workload",
            )

    @staticmethod
    def _validate_norm(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        path: str,
        *,
        profile: ProfileKey | None = None,
    ) -> None:
        _require_pure(node, path)
        if node.impl_ref != "rms_norm" or len(node.inputs) != 2 or len(node.outputs) != 1:
            _fail("RMSNorm requires rms_norm with activation/scale inputs and one output", path)
        workload = node.workload
        assert type(workload) is RmsNormWorkload
        input_value, weight_value = (values[ref] for ref in node.inputs)
        output_value = values[node.outputs[0]]
        actual = (
            input_value.shape,
            output_value.shape,
            local_shapes[input_value.id],
            local_shapes[output_value.id],
            weight_value.shape,
            local_shapes[weight_value.id],
            input_value.dtype,
            weight_value.dtype,
            output_value.dtype,
        )
        expected = (
            workload.logical_activation_shape,
            workload.logical_output_shape,
            workload.rank_activation_shape,
            workload.rank_output_shape,
            workload.logical_weight_shape,
            workload.rank_weight_shape,
            workload.dtype,
            workload.dtype,
            workload.dtype,
        )
        if (
            actual != expected
            or (
                profile is not None
                and workload.logical_activation_shape[0]
                != profile.prefill_tokens + profile.decode_tokens
            )
            or input_value.sharding != output_value.sharding
            or not _is_replicated(weight_value)
        ):
            _fail("RMSNorm workload/value shapes, dtype and sharding must match exactly", f"{path}.workload")

    @staticmethod
    def _validate_elementwise(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        path: str,
    ) -> None:
        _require_pure(node, path)
        if len(node.outputs) != 1:
            _fail("Dense elementwise requires one output", path)
        workload = node.workload
        inputs = tuple(values[ref] for ref in node.inputs)
        output = values[node.outputs[0]]
        if type(workload) is ResidualWorkload:
            if node.impl_ref != "residual":
                _fail("ResidualWorkload requires impl_ref='residual'", f"{path}.impl_ref")
            if (
                len(inputs) != 2
                or workload.logical_shape != output.shape
                or workload.rank_shape != local_shapes[output.id]
                or any(
                    value.shape != output.shape
                    or local_shapes[value.id] != workload.rank_shape
                    or value.sharding != output.sharding
                    or value.dtype is not workload.dtype
                    for value in inputs
                )
                or output.dtype is not workload.dtype
            ):
                _fail("residual requires two shape/sharding-identical inputs", path)
        elif type(workload) is SwiGluWorkload:
            if (
                node.impl_ref != "swiglu"
                or
                len(inputs) != 1
                or workload.logical_input_shape != inputs[0].shape
                or workload.logical_output_shape != output.shape
                or workload.rank_input_shape != local_shapes[inputs[0].id]
                or workload.rank_output_shape != local_shapes[output.id]
                or inputs[0].sharding != output.sharding
                or inputs[0].dtype is not workload.dtype
                or output.dtype is not workload.dtype
            ):
                _fail("swiglu requires one input with exactly twice the output's final extent and identical sharding", path)
        else:
            _fail("current Dense elementwise requires a typed SwiGLU or Residual workload", f"{path}.workload")

    @staticmethod
    def _validate_attention(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        axis_sizes: dict[MeshAxisName, int],
        profile: ProfileKey,
        path: str,
    ) -> None:
        if node.impl_ref not in ("attention", "attention_forward") or len(node.inputs) != 1 or len(node.outputs) != 1:
            _fail("Attention requires one input and one output with a supported impl_ref", path)
        workload = node.workload
        assert isinstance(workload, AttentionWorkload)
        input_value = values[node.inputs[0]]
        output_value = values[node.outputs[0]]
        exact_profile = workload.exact_profile
        if exact_profile is None and profile.prefill_tokens and profile.decode_tokens:
            raise UnsupportedFeatureError(
                "mixed attention requires an exact static profile",
                path=f"{path}.workload.profile",
            )
        tokens = profile.prefill_tokens + profile.decode_tokens
        qkv_width = (workload.num_heads + 2 * workload.num_kv_heads) * workload.head_dim
        if input_value.shape != (tokens, qkv_width) or output_value.shape != (tokens, workload.hidden_size):
            _fail("Attention value shapes do not match profile/head geometry", f"{path}.workload")
        tp = axis_sizes.get(MeshAxisName.TP)
        if tp is None or workload.num_heads % tp != 0 or workload.num_kv_heads % tp != 0:
            _fail("Attention heads must divide exactly across the TP mesh", f"{path}.workload.rank_num_heads")
        if workload.rank_num_heads != workload.num_heads // tp or workload.rank_num_kv_heads != workload.num_kv_heads // tp:
            _fail("Attention rank head counts must exactly derive from TP", f"{path}.workload.rank_num_heads")
        expected_input_local = (tokens, qkv_width // tp)
        expected_output_local = (tokens, workload.hidden_size // tp)
        if local_shapes[input_value.id] != expected_input_local or local_shapes[output_value.id] != expected_output_local:
            _fail("Attention rank-local value shapes must exactly derive from TP heads", f"{path}.workload")
        expected_map = (None, None) if tp == 1 else (None, MeshAxisName.TP)
        _exact_sharding(input_value, expected_map, (), f"{path}.inputs[0]")
        _exact_sharding(output_value, expected_map, (), f"{path}.outputs[0]")
        if workload.mode is AttentionMode.TRAIN_FORWARD:
            if exact_profile is not None:
                _fail("train-forward attention cannot carry an inference profile", f"{path}.workload.exact_profile")
            expected_pairs = (
                profile.num_seqs
                * profile.context_max
                * (profile.context_max + 1)
                // 2
            )
            expected_mode = AttentionMode.TRAIN_FORWARD
            read_tokens = 0
            write_tokens = 0
        elif exact_profile is not None:
            expected_pairs = exact_profile.capacity.query_key_pairs
            expected_mode = {
                Stage3ProfileMode.PREFILL: AttentionMode.PREFILL,
                Stage3ProfileMode.DECODE: AttentionMode.DECODE,
                Stage3ProfileMode.MIXED: AttentionMode.MIXED,
            }[exact_profile.mode]
            read_tokens = exact_profile.capacity.kv_read_tokens
            write_tokens = exact_profile.capacity.kv_write_tokens
        else:
            expected_pairs = (
                tokens * (tokens + 1) // 2
                if profile.prefill_tokens
                else profile.context_sum
            )
            expected_mode = (
                AttentionMode.PREFILL
                if profile.prefill_tokens
                else AttentionMode.DECODE
            )
            read_tokens = (
                0 if expected_mode is AttentionMode.PREFILL else profile.context_sum
            )
            write_tokens = tokens
        expected_read_bytes = (
            4 * read_tokens * workload.num_kv_heads * workload.head_dim
        )
        expected_write_bytes = (
            4 * write_tokens * workload.num_kv_heads * workload.head_dim
        )
        if (
            workload.profile != profile
            or workload.mode is not expected_mode
            or not workload.causal
            or workload.query_tokens != tokens
            or workload.context_sum != profile.context_sum
            or workload.context_max != profile.context_max
            or workload.query_key_pairs != expected_pairs
            or workload.logical_kv_read_bytes != expected_read_bytes
            or workload.logical_kv_write_bytes != expected_write_bytes
            or workload.rank_kv_read_bytes != expected_read_bytes // tp
            or workload.rank_kv_write_bytes != expected_write_bytes // tp
            or input_value.dtype is not workload.dtype
            or output_value.dtype is not workload.dtype
            or (
                expected_mode is AttentionMode.TRAIN_FORWARD
                and (
                    node.effects.kind is not EffectKind.PURE
                    or node.effects.effect_token is not None
                    or node.effects.alias_set is not None
                )
            )
            or (
                expected_mode is not AttentionMode.TRAIN_FORWARD
                and (
                    node.effects.kind is not EffectKind.STATEFUL
                    or node.effects.effect_token is None
                    or node.effects.alias_set is None
                )
            )
        ):
            _fail("Attention profile, dtype, KV effects or query-key pair count is not exact", f"{path}.workload")

    @staticmethod
    def _validate_cross_entropy(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        profile: ProfileKey,
        path: str,
    ) -> None:
        _require_pure(node, path)
        workload = node.workload
        assert isinstance(workload, CrossEntropyForwardWorkload)
        if (
            node.impl_ref != "cross_entropy_forward"
            or len(node.inputs) != 2
            or len(node.outputs) != 1
            or workload.profile != profile
            or workload.reduction is not CrossEntropyReduction.NONE
        ):
            _fail("cross entropy operator contract is not exact", path)
        logits, labels = (values[ref] for ref in node.inputs)
        loss = values[node.outputs[0]]
        if (
            workload.logical_logits_shape != logits.shape
            or workload.rank_logits_shape != local_shapes[logits.id]
            or workload.logical_label_shape != labels.shape
            or workload.rank_label_shape != local_shapes[labels.id]
            or workload.logical_loss_shape != loss.shape
            or workload.rank_loss_shape != local_shapes[loss.id]
            or logits.dtype is not DType.FP16
            or labels.dtype is not DType.INT32
            or loss.dtype is not DType.FP32
            or labels.producer is not None
            or loss.consumers
            or logits.sharding.dim_map[0] != labels.sharding.dim_map[0]
            or labels.sharding != loss.sharding
            or logits.sharding.dim_map[1] is not None
            or logits.sharding.partial
        ):
            _fail("cross entropy logits/labels/loss boundary is not exact", path)

    @staticmethod
    def _validate_collective(
        node: LogicalNode,
        values: dict[str, TensorValue],
        local_shapes: dict[str, tuple[int, ...]],
        axis_sizes: dict[MeshAxisName, int],
        path: str,
    ) -> None:
        _require_pure(node, path)
        if node.impl_ref != "collective_derived" or len(node.inputs) != 1 or len(node.outputs) != 1:
            _fail("Dense collective requires collective_derived with one input and one output", path)
        workload = node.workload
        assert isinstance(workload, CollectiveWorkload)
        input_value = values[node.inputs[0]]
        output_value = values[node.outputs[0]]
        if input_value.shape != output_value.shape or input_value.dtype is not output_value.dtype or workload.dtype is not input_value.dtype:
            _fail("collective input/output logical shape and dtype must match", f"{path}.workload")
        if workload.input_layout != input_value.logical_layout or workload.output_layout != output_value.logical_layout:
            _fail("collective workload layouts must exactly match TensorValue layouts", f"{path}.workload.input_layout")
        if workload.role is not CollectiveRole.ACTIVATION:
            _fail("Dense forward collective role must be ACTIVATION", f"{path}.workload.role")
        if workload.mesh_axes != (MeshAxisName.TP,):
            _fail("Dense collective must use exactly the TP mesh axis", f"{path}.workload.mesh_axes")
        tp = axis_sizes.get(MeshAxisName.TP)
        if tp is None or tp <= 1 or workload.participant_count != tp:
            _fail("collective participant_count must exactly equal TP>1", f"{path}.workload.participant_count")
        if workload.collective is CollectiveKind.ALL_GATHER:
            axis = workload.gather_tensor_axis
            if axis is None or axis >= len(input_value.shape):
                _fail("AllGather tensor axis is out of range", f"{path}.workload.gather_tensor_axis")
            expected_output_map = list(input_value.sharding.dim_map)
            if input_value.sharding.dim_map[axis] is not MeshAxisName.TP:
                _fail("AllGather input must be sharded by TP on gather_tensor_axis", f"{path}.inputs[0]")
            expected_output_map[axis] = None
            if input_value.sharding.partial or output_value.sharding.partial or output_value.sharding.dim_map != tuple(expected_output_map):
                _fail("AllGather must perform the exact TP-sharded to replicated sharding delta", f"{path}.outputs[0]")
        elif workload.collective is CollectiveKind.REDUCE_SCATTER:
            axis = workload.scatter_tensor_axis
            if axis is None or axis >= len(input_value.shape):
                _fail("ReduceScatter tensor axis is out of range", f"{path}.workload.scatter_tensor_axis")
            expected_output_map = list(input_value.sharding.dim_map)
            if input_value.sharding.partial != (MeshAxisName.TP,) or expected_output_map[axis] is not None:
                _fail("ReduceScatter input must be TP-partial and unsharded on scatter_tensor_axis", f"{path}.inputs[0]")
            expected_output_map[axis] = MeshAxisName.TP
            if (
                workload.reduce_op is not ReduceOp.SUM
                or output_value.sharding.partial
                or output_value.sharding.dim_map != tuple(expected_output_map)
            ):
                _fail("ReduceScatter must perform the exact TP-partial to TP-sharded SUM delta", f"{path}.outputs[0]")
        else:
            raise UnsupportedFeatureError(
                "Dense MVP supports only AllGather and ReduceScatter", path=f"{path}.workload.collective"
            )

        logical_bytes = _tensor_bytes(input_value, f"{path}.workload.logical_tensor_bytes")
        rank_input_bytes = prod(local_shapes[input_value.id]) * _DTYPE_BYTES[input_value.dtype]
        rank_output_bytes = prod(local_shapes[output_value.id]) * _DTYPE_BYTES[output_value.dtype]
        rank_payload_bytes = abs(rank_output_bytes - rank_input_bytes)
        group_payload_bytes = rank_payload_bytes * tp
        if (
            workload.logical_tensor_bytes != logical_bytes
            or workload.rank_input_bytes != rank_input_bytes
            or workload.rank_output_bytes != rank_output_bytes
            or workload.rank_logical_payload_bytes != rank_payload_bytes
            or workload.group_logical_payload_bytes != group_payload_bytes
        ):
            _fail("collective bytes/payloads must exactly derive from TensorValue global/rank-local shapes and participants", f"{path}.workload.logical_tensor_bytes")
