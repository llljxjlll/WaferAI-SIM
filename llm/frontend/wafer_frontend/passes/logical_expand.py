"""Materialize Llama Dense logical graphs for canonical static profiles."""

from __future__ import annotations

from ..errors import UnsupportedFeatureError
from ..schema.common import (
    DType,
    MeshAxisName,
    ProfileKey,
    Sharding,
    TensorValue,
)
from ..schema.ir0 import (
    AttentionMode,
    AttentionWorkload,
    CollectiveKind,
    CollectiveRole,
    CollectiveWorkload,
    CrossEntropyForwardWorkload,
    CrossEntropyReduction,
    EdgeKind,
    EmbeddingTablePlacement,
    EmbeddingWorkload,
    EffectKind,
    FusionCandidate,
    FusionImpl,
    FusionOrigin,
    FusionSemanticContract,
    GemmPartition,
    GemmWorkload,
    GraphEdge,
    IR0,
    JobKind,
    LogicalInstance,
    LogicalNode,
    LogicalRole,
    NodeEffects,
    NodeMath,
    NumericalPolicy,
    OpKind,
    OpPhase,
    PackedQkvLayout,
    ReduceOp,
    ResidualWorkload,
    RmsNormWorkload,
    RopeQkWorkload,
    SampleRowSelection,
    SamplingMode,
    GreedySampleWorkload,
    StateAccess,
    StateAccessMode,
    SwiGluWorkload,
    TrainStructure,
)
from ..schema.experiment import InferOutput
from ..schema.logical import (
    DenseModelShape,
    ExpandedIR0Bundle,
    ExpandedProfileIR0,
    IR0Template,
)
from ..schema.stage3_profile import Stage3ProfileMode, Stage3StaticProfile
from ..schema.persistent_state import (
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    StateKind,
)


def _unsupported(message: str, *, path: str) -> None:
    raise UnsupportedFeatureError(message, path=path)


def _validate_scope(template: IR0Template) -> None:
    parallel = template.instance.parallel
    if parallel.tp == 1 and parallel.sp:
        _unsupported(
            "sequence parallelism over TP=1 is unsupported",
            path="template.instance.parallel.sp",
        )
    if parallel.tp > 1 and not parallel.sp:
        _unsupported(
            "TP>1 logical expansion requires sequence parallelism",
            path="template.instance.parallel.sp",
        )

    model = template.model
    if template.infer_output is InferOutput.GREEDY_SAMPLE and parallel.tp != 1:
        _unsupported(
            "TP-sharded greedy sampling is not implemented",
            path="template.infer_output",
        )
    for index, entry in enumerate(template.profiles):
        profile = entry.key
        exact_profile = entry.exact_profile
        profile_path = f"template.profiles[{index}].key"
        tokens = profile.prefill_tokens + profile.decode_tokens
        if parallel.tp > 1:
            for extent, name, path in (
                (tokens, "tokens", profile_path),
                (model.num_heads, "num_heads", "template.model.num_heads"),
                (
                    model.num_kv_heads,
                    "num_kv_heads",
                    "template.model.num_kv_heads",
                ),
                (
                    model.intermediate_size,
                    "intermediate_size",
                    "template.model.intermediate_size",
                ),
            ):
                if extent % parallel.tp != 0:
                    _unsupported(
                        f"{name} must be divisible by TP for logical expansion",
                        path=path,
                    )
        if exact_profile is not None:
            allowed_roles = {
                Stage3ProfileMode.PREFILL: (
                    LogicalRole.PREFILL,
                    LogicalRole.BOTH,
                ),
                Stage3ProfileMode.DECODE: (
                    LogicalRole.DECODE,
                    LogicalRole.BOTH,
                ),
                Stage3ProfileMode.MIXED: (LogicalRole.BOTH,),
            }[exact_profile.mode]
            if template.instance.role not in allowed_roles:
                _unsupported(
                    "exact profile mode is incompatible with the logical instance role",
                    path="template.instance.role",
                )
            continue
        if profile.prefill_tokens and profile.decode_tokens:
            _unsupported(
                "mixed prefill/decode profiles do not uniquely determine attention work",
                path=profile_path,
            )
        if profile.prefill_tokens:
            if template.instance.role not in (
                LogicalRole.PREFILL,
                LogicalRole.BOTH,
            ):
                _unsupported(
                    "prefill profile is incompatible with the logical instance role",
                    path="template.instance.role",
                )
            if (
                profile.num_seqs != 1
                or profile.context_sum != profile.prefill_tokens
                or profile.context_max != profile.prefill_tokens
            ):
                _unsupported(
                    "prefill requires one sequence and context_sum == context_max == prefill_tokens",
                    path=profile_path,
                )
            continue
        if template.instance.role not in (LogicalRole.DECODE, LogicalRole.BOTH):
            _unsupported(
                "decode profile is incompatible with the logical instance role",
                path="template.instance.role",
            )
        if profile.decode_tokens != profile.num_seqs:
            _unsupported(
                "decode requires exactly one query token per sequence",
                path=f"{profile_path}.decode_tokens",
            )
        if profile.context_sum == 0:
            _unsupported(
                "decode context_sum must be greater than zero",
                path=f"{profile_path}.context_sum",
            )


def _query_key_pairs(
    profile: ProfileKey,
    exact_profile: Stage3StaticProfile | None,
) -> int:
    if exact_profile is not None:
        return exact_profile.capacity.query_key_pairs
    if profile.prefill_tokens:
        return profile.prefill_tokens * (profile.prefill_tokens + 1) // 2
    return profile.context_sum


def _expand_dense_graph(
    model: DenseModelShape,
    instance: LogicalInstance,
    profile: ProfileKey,
    exact_profile: Stage3StaticProfile | None = None,
    *,
    job: JobKind,
    train: TrainStructure | None,
    infer_output: InferOutput | None,
    producer_pass: str,
) -> IR0:
    """Build one independent graph by chaining profile-specialized layers."""

    train_forward = job is JobKind.TRAIN
    tp = instance.parallel.tp
    distributed = tp > 1
    mesh_ref = instance.meshes[0].id
    tokens = profile.prefill_tokens + profile.decode_tokens
    rank_tokens = tokens // tp if distributed else tokens
    if not train_forward and profile.context_sum == 0:
        _unsupported(
            "persistent KV state requires context_sum > 0",
            path="profile.context_sum",
        )
    hidden = model.hidden_size
    intermediate = model.intermediate_size
    qkv_width = (model.num_heads + 2 * model.num_kv_heads) * model.head_dim
    mh = (tokens, hidden)
    mi = (tokens, intermediate)
    m2i = (tokens, 2 * intermediate)
    mqkv = (tokens, qkv_width)
    rank_mh = (rank_tokens, hidden)
    math = NodeMath(DType.FP32, NumericalPolicy.BITWISE)

    def replicated(rank: int = 2) -> Sharding:
        return Sharding(mesh_ref, (None,) * rank, ())

    m_sharded = Sharding(mesh_ref, (MeshAxisName.TP, None), ())
    feature_sharded = Sharding(mesh_ref, (None, MeshAxisName.TP), ())
    partial = Sharding(mesh_ref, (None, None), (MeshAxisName.TP,))

    nodes: list[LogicalNode] = []
    values: list[TensorValue] = []
    candidates: list[FusionCandidate] = []

    persistent_states: list[PersistentStateDecl] = []
    state_accesses: list[StateAccess] = []
    first_prefix = f"{instance.id}.layer0"
    token_ids_id = f"{instance.id}.token_ids"
    embedding_weight_id = f"{instance.id}.tok_embeddings.weight"
    input_id = f"{instance.id}.embedding_out"
    input_consumers = (
        f"{first_prefix}.norm1",
        f"{first_prefix}.residual1",
    )
    values.extend(
        (
            TensorValue(
                id=token_ids_id,
                shape=(tokens,),
                dtype=DType.INT32,
                logical_layout="M_token_ids_shard_tp" if distributed else "M_token_ids",
                sharding=Sharding(mesh_ref, (MeshAxisName.TP,), ())
                if distributed
                else replicated(1),
                producer=None,
                consumers=(f"{instance.id}.embedding",),
                alias_set=None,
            ),
            TensorValue(
                id=embedding_weight_id,
                shape=(model.vocab_size, hidden),
                dtype=model.dtype,
                logical_layout="VH_replicated",
                sharding=replicated(),
                producer=None,
                consumers=(f"{instance.id}.embedding",),
                alias_set=None,
            ),
            TensorValue(
                id=input_id,
                shape=mh,
                dtype=model.dtype,
                logical_layout="MH_shard_tp" if distributed else "MH",
                sharding=m_sharded if distributed else replicated(),
                producer=f"{instance.id}.embedding",
                consumers=input_consumers,
                alias_set=None,
            ),
        )
    )
    nodes.append(
        LogicalNode(
            id=f"{instance.id}.embedding",
            instance_id=instance.id,
            kind=OpKind.EMBEDDING,
            phase=OpPhase.FWD,
            stage=0,
            mesh_ref=mesh_ref,
            inputs=(token_ids_id, embedding_weight_id),
            outputs=(input_id,),
            workload=EmbeddingWorkload(
                profile=profile,
                logical_index_shape=(tokens,),
                rank_index_shape=(rank_tokens,),
                logical_table_shape=(model.vocab_size, hidden),
                rank_table_shape=(model.vocab_size, hidden),
                logical_output_shape=mh,
                rank_output_shape=rank_mh,
                table_placement=EmbeddingTablePlacement.REPLICATED,
                index_dtype=DType.INT32,
                table_dtype=model.dtype,
                output_dtype=model.dtype,
            ),
            math=math,
            effects=NodeEffects(EffectKind.PURE, None, None),
            impl_ref="embedding_lookup",
        )
    )

    def add_layer(
        profile: ProfileKey,
        layer_index: int,
        input_value: str,
    ) -> str:
        """Append one layer and return its residual2 output value ID."""

        prefix = f"{instance.id}.layer{layer_index}"
        op_names = (
            (
                "norm1",
                "ag1",
                "qkv",
                "rope",
                "attention",
                "o",
                "rs1",
                "residual1",
                "norm2",
                "ag2",
                "gate_up",
                "swiglu",
                "down",
                "rs2",
                "residual2",
            )
            if distributed
            else (
                "norm1",
                "qkv",
                "rope",
                "attention",
                "o",
                "residual1",
                "norm2",
                "gate_up",
                "swiglu",
                "down",
                "residual2",
            )
        )
        node_id = {name: f"{prefix}.{name}" for name in op_names}
        next_consumers = (
            (
                f"{instance.id}.layer{layer_index + 1}.norm1",
                f"{instance.id}.layer{layer_index + 1}.residual1",
            )
            if layer_index + 1 < model.num_layers
            else (f"{instance.id}.final_norm",)
        )

        local_values: dict[str, TensorValue] = {}

        def add_value(
            name: str,
            shape: tuple[int, ...],
            layout: str,
            sharding: Sharding,
            producer: str | None,
            consumers: tuple[str, ...],
        ) -> str:
            value = TensorValue(
                id=f"{prefix}.{name}",
                shape=shape,
                dtype=model.dtype,
                logical_layout=layout,
                sharding=sharding,
                producer=None if producer is None else node_id[producer],
                consumers=tuple(node_id.get(item, item) for item in consumers),
                alias_set=None,
            )
            values.append(value)
            local_values[name] = value
            return value.id

        weight_sharding = replicated()
        column_weight = Sharding(mesh_ref, (None, MeshAxisName.TP), ())
        row_weight = Sharding(mesh_ref, (MeshAxisName.TP, None), ())
        add_value(
            "w_norm1",
            (hidden,),
            "H_replicated",
            replicated(1),
            None,
            ("norm1",),
        )
        add_value(
            "w_norm2",
            (hidden,),
            "H_replicated",
            replicated(1),
            None,
            ("norm2",),
        )
        add_value(
            "w_qkv",
            (hidden, qkv_width),
            "KN_column_tp" if distributed else "KN",
            column_weight if distributed else weight_sharding,
            None,
            ("qkv",),
        )
        add_value(
            "w_o",
            (hidden, hidden),
            "KN_row_tp" if distributed else "KN",
            row_weight if distributed else weight_sharding,
            None,
            ("o",),
        )
        add_value(
            "w_gate_up",
            (hidden, 2 * intermediate),
            "KN_column_tp" if distributed else "KN",
            column_weight if distributed else weight_sharding,
            None,
            ("gate_up",),
        )
        add_value(
            "w_down",
            (intermediate, hidden),
            "KN_row_tp" if distributed else "KN",
            row_weight if distributed else weight_sharding,
            None,
            ("down",),
        )

        if distributed:
            add_value("norm1_out", mh, "MH_shard_tp", m_sharded, "norm1", ("ag1",))
            add_value("ag1_out", mh, "MH_replicated", replicated(), "ag1", ("qkv",))
            add_value("qkv_out", mqkv, "MQKV_feature_tp", feature_sharded, "qkv", ("rope",))
            add_value("qkv_rope", mqkv, "MQKV_feature_tp", feature_sharded, "rope", ("attention",))
            add_value("attention_out", mh, "MH_feature_tp", feature_sharded, "attention", ("o",))
            add_value("o_partial", mh, "MH_partial_tp", partial, "o", ("rs1",))
            add_value("rs1_out", mh, "MH_shard_tp", m_sharded, "rs1", ("residual1",))
            add_value(
                "residual1_out",
                mh,
                "MH_shard_tp",
                m_sharded,
                "residual1",
                ("norm2", "residual2"),
            )
            add_value("norm2_out", mh, "MH_shard_tp", m_sharded, "norm2", ("ag2",))
            add_value("ag2_out", mh, "MH_replicated", replicated(), "ag2", ("gate_up",))
            add_value("gate_up_out", m2i, "M2I_feature_tp", feature_sharded, "gate_up", ("swiglu",))
            add_value("swiglu_out", mi, "MI_feature_tp", feature_sharded, "swiglu", ("down",))
            add_value("down_partial", mh, "MH_partial_tp", partial, "down", ("rs2",))
            add_value("rs2_out", mh, "MH_shard_tp", m_sharded, "rs2", ("residual2",))
            output_id = add_value(
                "output",
                mh,
                "MH_shard_tp",
                m_sharded,
                "residual2",
                next_consumers,
            )
        else:
            add_value("norm1_out", mh, "MH", replicated(), "norm1", ("qkv",))
            add_value("qkv_out", mqkv, "MQKV", replicated(), "qkv", ("rope",))
            add_value("qkv_rope", mqkv, "MQKV", replicated(), "rope", ("attention",))
            add_value("attention_out", mh, "MH", replicated(), "attention", ("o",))
            add_value("o_out", mh, "MH", replicated(), "o", ("residual1",))
            add_value(
                "residual1_out",
                mh,
                "MH",
                replicated(),
                "residual1",
                ("norm2", "residual2"),
            )
            add_value("norm2_out", mh, "MH", replicated(), "norm2", ("gate_up",))
            add_value("gate_up_out", m2i, "M2I", replicated(), "gate_up", ("swiglu",))
            add_value("swiglu_out", mi, "MI", replicated(), "swiglu", ("down",))
            add_value("down_out", mh, "MH", replicated(), "down", ("residual2",))
            output_id = add_value(
                "output",
                mh,
                "MH",
                replicated(),
                "residual2",
                next_consumers,
            )

        def value_id(name: str) -> str:
            return local_values[name].id

        def pure() -> NodeEffects:
            return NodeEffects(EffectKind.PURE, None, None)

        def norm(name: str, source: str, weight: str, output: str) -> LogicalNode:
            return LogicalNode(
                id=node_id[name],
                instance_id=instance.id,
                kind=OpKind.NORM,
                phase=OpPhase.FWD,
                stage=0,
                mesh_ref=mesh_ref,
                inputs=(source, value_id(weight)),
                outputs=(value_id(output),),
                workload=RmsNormWorkload(
                    logical_activation_shape=mh,
                    logical_output_shape=mh,
                    rank_activation_shape=rank_mh if distributed else mh,
                    rank_output_shape=rank_mh if distributed else mh,
                    logical_weight_shape=(hidden,),
                    rank_weight_shape=(hidden,),
                    epsilon=model.rms_norm_epsilon,
                    dtype=model.dtype,
                ),
                math=math,
                effects=pure(),
                impl_ref="rms_norm",
            )

        def gemm(
            name: str,
            source: str,
            weight: str,
            output: str,
            logical_shape: tuple[int, int, int],
            rank_shape: tuple[int, int, int],
            partition: GemmPartition,
        ) -> LogicalNode:
            return LogicalNode(
                id=node_id[name],
                instance_id=instance.id,
                kind=OpKind.GEMM,
                phase=OpPhase.FWD,
                stage=0,
                mesh_ref=mesh_ref,
                inputs=(source, value_id(weight)),
                outputs=(value_id(output),),
                workload=GemmWorkload(
                    logical_shape, rank_shape, partition, model.dtype
                ),
                math=math,
                effects=pure(),
                impl_ref="matmul_forward",
            )

        def elementwise(
            name: str,
            inputs: tuple[str, ...],
            output: str,
            logical_inputs: tuple[tuple[int, ...], ...],
            logical_output: tuple[int, ...],
            rank_inputs: tuple[tuple[int, ...], ...],
            rank_output: tuple[int, ...],
            impl_ref: str,
        ) -> LogicalNode:
            if impl_ref == "swiglu":
                workload = SwiGluWorkload(
                    logical_input_shape=logical_inputs[0],
                    logical_output_shape=logical_output,
                    rank_input_shape=rank_inputs[0],
                    rank_output_shape=rank_output,
                    dtype=model.dtype,
                )
            elif impl_ref == "residual":
                workload = ResidualWorkload(
                    logical_shape=logical_output,
                    rank_shape=rank_output,
                    dtype=model.dtype,
                )
            else:
                _unsupported(
                    f"unsupported current-block elementwise implementation {impl_ref!r}",
                    path=f"logical_expand.{name}.impl_ref",
                )
            return LogicalNode(
                id=node_id[name],
                instance_id=instance.id,
                kind=OpKind.ELEMENTWISE,
                phase=OpPhase.FWD,
                stage=0,
                mesh_ref=mesh_ref,
                inputs=inputs,
                outputs=(value_id(output),),
                workload=workload,
                math=math,
                effects=pure(),
                impl_ref=impl_ref,
            )

        def attention() -> LogicalNode:
            mode = AttentionMode.TRAIN_FORWARD if train_forward else (
                {
                    Stage3ProfileMode.PREFILL: AttentionMode.PREFILL,
                    Stage3ProfileMode.DECODE: AttentionMode.DECODE,
                    Stage3ProfileMode.MIXED: AttentionMode.MIXED,
                }[exact_profile.mode]
                if exact_profile is not None
                else (
                    AttentionMode.PREFILL
                    if profile.prefill_tokens
                    else AttentionMode.DECODE
                )
            )
            read_tokens = (
                0
                if train_forward
                else
                exact_profile.capacity.kv_read_tokens
                if exact_profile is not None
                else (0 if mode is AttentionMode.PREFILL else profile.context_sum)
            )
            write_tokens = (
                0
                if train_forward
                else
                exact_profile.capacity.kv_write_tokens
                if exact_profile is not None
                else tokens
            )
            logical_kv_read_bytes = (
                4 * read_tokens * model.num_kv_heads * model.head_dim
            )
            logical_kv_write_bytes = (
                4 * write_tokens * model.num_kv_heads * model.head_dim
            )
            return LogicalNode(
                id=node_id["attention"],
                instance_id=instance.id,
                kind=OpKind.ATTENTION,
                phase=OpPhase.FWD,
                stage=0,
                mesh_ref=mesh_ref,
                inputs=(value_id("qkv_rope"),),
                outputs=(value_id("attention_out"),),
                workload=AttentionWorkload(
                    profile=profile,
                    mode=mode,
                    causal=True,
                    query_tokens=tokens,
                    context_sum=profile.context_sum,
                    context_max=profile.context_max,
                    hidden_size=hidden,
                    num_heads=model.num_heads,
                    num_kv_heads=model.num_kv_heads,
                    head_dim=model.head_dim,
                    rank_num_heads=model.num_heads // tp
                    if distributed
                    else model.num_heads,
                    rank_num_kv_heads=model.num_kv_heads // tp
                    if distributed
                    else model.num_kv_heads,
                    query_key_pairs=(
                        profile.num_seqs
                        * profile.context_max
                        * (profile.context_max + 1)
                        // 2
                        if train_forward
                        else _query_key_pairs(profile, exact_profile)
                    ),
                    logical_kv_read_bytes=logical_kv_read_bytes,
                    logical_kv_write_bytes=logical_kv_write_bytes,
                    rank_kv_read_bytes=logical_kv_read_bytes // tp,
                    rank_kv_write_bytes=logical_kv_write_bytes // tp,
                    dtype=model.dtype,
                    exact_profile=exact_profile,
                ),
                math=math,
                effects=(
                    NodeEffects(EffectKind.PURE, None, None)
                    if train_forward
                    else NodeEffects(
                        EffectKind.STATEFUL,
                        f"kv_effect_layer_{layer_index}",
                        f"kv_alias_layer_{layer_index}",
                    )
                ),
                impl_ref="attention_forward",
            )

        def rope() -> LogicalNode:
            rank_qkv = (tokens, qkv_width // tp) if distributed else mqkv
            return LogicalNode(
                id=node_id["rope"],
                instance_id=instance.id,
                kind=OpKind.ROPE,
                phase=OpPhase.FWD,
                stage=0,
                mesh_ref=mesh_ref,
                inputs=(value_id("qkv_out"),),
                outputs=(value_id("qkv_rope"),),
                workload=RopeQkWorkload(
                    profile=profile,
                    logical_input_shape=mqkv,
                    rank_input_shape=rank_qkv,
                    logical_output_shape=mqkv,
                    rank_output_shape=rank_qkv,
                    packed_layout=PackedQkvLayout.Q_K_V,
                    num_heads=model.num_heads,
                    num_kv_heads=model.num_kv_heads,
                    rank_num_heads=model.num_heads // tp,
                    rank_num_kv_heads=model.num_kv_heads // tp,
                    head_dim=model.head_dim,
                    rotary_dim=model.rotary_dim,
                    rope_theta=model.rope_theta,
                    max_position_embeddings=model.max_position_embeddings,
                    dtype=model.dtype,
                ),
                math=math,
                effects=pure(),
                impl_ref="rope_qk_exact",
            )

        qkv_rank_shape = (
            (tokens, qkv_width // tp, hidden)
            if distributed
            else (tokens, qkv_width, hidden)
        )
        gate_rank_shape = (
            (tokens, 2 * intermediate // tp, hidden)
            if distributed
            else (tokens, 2 * intermediate, hidden)
        )
        o_rank_shape = (
            (tokens, hidden, hidden // tp)
            if distributed
            else (tokens, hidden, hidden)
        )
        down_rank_shape = (
            (tokens, hidden, intermediate // tp)
            if distributed
            else (tokens, hidden, intermediate)
        )

        if distributed:
            activation_bytes = 2 * tokens * hidden
            rank_shard_bytes = activation_bytes // tp
            rank_payload_bytes = activation_bytes - rank_shard_bytes
            group_payload_bytes = rank_payload_bytes * tp

            def collective(
                name: str,
                source: str,
                output: str,
                kind: CollectiveKind,
            ) -> LogicalNode:
                is_ag = kind is CollectiveKind.ALL_GATHER
                return LogicalNode(
                    id=node_id[name],
                    instance_id=instance.id,
                    kind=OpKind.COLLECTIVE,
                    phase=OpPhase.FWD,
                    stage=0,
                    mesh_ref=mesh_ref,
                    inputs=(value_id(source),),
                    outputs=(value_id(output),),
                    workload=CollectiveWorkload(
                        collective=kind,
                        reduce_op=None if is_ag else ReduceOp.SUM,
                        mesh_axes=(MeshAxisName.TP,),
                        participant_count=tp,
                        reduction_mesh_axes=()
                        if is_ag
                        else (MeshAxisName.TP,),
                        scatter_tensor_axis=None if is_ag else 0,
                        gather_tensor_axis=0 if is_ag else None,
                        logical_tensor_bytes=activation_bytes,
                        rank_input_bytes=rank_shard_bytes
                        if is_ag
                        else activation_bytes,
                        rank_output_bytes=activation_bytes
                        if is_ag
                        else rank_shard_bytes,
                        rank_logical_payload_bytes=rank_payload_bytes,
                        group_logical_payload_bytes=group_payload_bytes,
                        dtype=model.dtype,
                        role=CollectiveRole.ACTIVATION,
                        input_layout=local_values[source].logical_layout,
                        output_layout=local_values[output].logical_layout,
                    ),
                    math=math,
                    effects=pure(),
                    impl_ref="collective_derived",
                )

            layer_nodes = (
                norm("norm1", input_value, "w_norm1", "norm1_out"),
                collective(
                    "ag1", "norm1_out", "ag1_out", CollectiveKind.ALL_GATHER
                ),
                gemm(
                    "qkv",
                    value_id("ag1_out"),
                    "w_qkv",
                    "qkv_out",
                    (tokens, qkv_width, hidden),
                    qkv_rank_shape,
                    GemmPartition.COLUMN_PARALLEL,
                ),
                rope(),
                attention(),
                gemm(
                    "o",
                    value_id("attention_out"),
                    "w_o",
                    "o_partial",
                    (tokens, hidden, hidden),
                    o_rank_shape,
                    GemmPartition.ROW_PARALLEL,
                ),
                collective(
                    "rs1",
                    "o_partial",
                    "rs1_out",
                    CollectiveKind.REDUCE_SCATTER,
                ),
                elementwise(
                    "residual1",
                    (input_value, value_id("rs1_out")),
                    "residual1_out",
                    (mh, mh),
                    mh,
                    (rank_mh, rank_mh),
                    rank_mh,
                    "residual",
                ),
                norm("norm2", value_id("residual1_out"), "w_norm2", "norm2_out"),
                collective(
                    "ag2", "norm2_out", "ag2_out", CollectiveKind.ALL_GATHER
                ),
                gemm(
                    "gate_up",
                    value_id("ag2_out"),
                    "w_gate_up",
                    "gate_up_out",
                    (tokens, 2 * intermediate, hidden),
                    gate_rank_shape,
                    GemmPartition.COLUMN_PARALLEL,
                ),
                elementwise(
                    "swiglu",
                    (value_id("gate_up_out"),),
                    "swiglu_out",
                    (m2i,),
                    mi,
                    ((tokens, 2 * intermediate // tp),),
                    (tokens, intermediate // tp),
                    "swiglu",
                ),
                gemm(
                    "down",
                    value_id("swiglu_out"),
                    "w_down",
                    "down_partial",
                    (tokens, hidden, intermediate),
                    down_rank_shape,
                    GemmPartition.ROW_PARALLEL,
                ),
                collective(
                    "rs2",
                    "down_partial",
                    "rs2_out",
                    CollectiveKind.REDUCE_SCATTER,
                ),
                elementwise(
                    "residual2",
                    (value_id("residual1_out"), value_id("rs2_out")),
                    "output",
                    (mh, mh),
                    mh,
                    (rank_mh, rank_mh),
                    rank_mh,
                    "residual",
                ),
            )

            def candidate(
                name: str,
                gemm_name: str,
                rs_name: str,
                input_names: tuple[str, str],
                output_name: str,
            ) -> FusionCandidate:
                boundary_inputs = tuple(value_id(item) for item in input_names)
                return FusionCandidate(
                    id=f"{prefix}.candidate.{name}",
                    members=(node_id[gemm_name], node_id[rs_name]),
                    boundary_inputs=boundary_inputs,
                    boundary_outputs=(value_id(output_name),),
                    semantic_contract=FusionSemanticContract(
                        tile_domain=("M", "N"),
                        reduction_axes=(2,),
                        input_layouts=tuple(
                            local_values[item].logical_layout
                            for item in input_names
                        ),
                        output_layout=local_values[output_name].logical_layout,
                        numerical_policy=NumericalPolicy.BITWISE,
                    ),
                    impl=FusionImpl.NONE,
                    origin=FusionOrigin.DISCOVERED,
                )

            candidates.extend(
                (
                    candidate(
                        "o_rs1",
                        "o",
                        "rs1",
                        ("attention_out", "w_o"),
                        "rs1_out",
                    ),
                    candidate(
                        "down_rs2",
                        "down",
                        "rs2",
                        ("swiglu_out", "w_down"),
                        "rs2_out",
                    ),
                )
            )
        else:
            layer_nodes = (
                norm("norm1", input_value, "w_norm1", "norm1_out"),
                gemm(
                    "qkv",
                    value_id("norm1_out"),
                    "w_qkv",
                    "qkv_out",
                    (tokens, qkv_width, hidden),
                    qkv_rank_shape,
                    GemmPartition.REPLICATED,
                ),
                rope(),
                attention(),
                gemm(
                    "o",
                    value_id("attention_out"),
                    "w_o",
                    "o_out",
                    (tokens, hidden, hidden),
                    o_rank_shape,
                    GemmPartition.REPLICATED,
                ),
                elementwise(
                    "residual1",
                    (input_value, value_id("o_out")),
                    "residual1_out",
                    (mh, mh),
                    mh,
                    (mh, mh),
                    mh,
                    "residual",
                ),
                norm("norm2", value_id("residual1_out"), "w_norm2", "norm2_out"),
                gemm(
                    "gate_up",
                    value_id("norm2_out"),
                    "w_gate_up",
                    "gate_up_out",
                    (tokens, 2 * intermediate, hidden),
                    gate_rank_shape,
                    GemmPartition.REPLICATED,
                ),
                elementwise(
                    "swiglu",
                    (value_id("gate_up_out"),),
                    "swiglu_out",
                    (m2i,),
                    mi,
                    (m2i,),
                    mi,
                    "swiglu",
                ),
                gemm(
                    "down",
                    value_id("swiglu_out"),
                    "w_down",
                    "down_out",
                    (tokens, hidden, intermediate),
                    down_rank_shape,
                    GemmPartition.REPLICATED,
                ),
                elementwise(
                    "residual2",
                    (value_id("residual1_out"), value_id("down_out")),
                    "output",
                    (mh, mh),
                    mh,
                    (mh, mh),
                    mh,
                    "residual",
                ),
            )

        nodes.extend(layer_nodes)
        parameter_specs = (
            ("w_norm1", "norm1", (hidden,)),
            ("w_norm2", "norm2", (hidden,)),
            ("w_qkv", "qkv", (hidden, qkv_width // tp)),
            ("w_o", "o", (hidden // tp, hidden)),
            ("w_gate_up", "gate_up", (hidden, 2 * intermediate // tp)),
            ("w_down", "down", (intermediate // tp, hidden)),
        )
        kv_layout = "THD_packed_kv_head_tp" if distributed else "THD_packed"

        def append_kv_states(
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
                model.num_kv_heads // tp,
                model.head_dim,
            )
            for kind in (StateKind.KV_KEY, StateKind.KV_VALUE):
                identity = PersistentStateIdentity.create(
                    kind=kind,
                    instance_ref=instance.id,
                    mesh_ref=mesh_ref,
                    request_ref=request_ref,
                    layer_index=layer_index,
                    tensor_ref=None,
                    shard_index=rank,
                    generation=0,
                )
                declaration = PersistentStateDecl.create(
                    identity=identity,
                    shape=kv_shape,
                    dtype=model.dtype,
                    layout=kv_layout,
                    lifetime=PersistentStateLifetime.PERSISTENT,
                    access=PersistentStateAccess.READ_WRITE,
                )
                persistent_states.append(declaration)
                state_accesses.append(
                    StateAccess.create(
                        node_ref=node_id["attention"],
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
                            model.num_kv_heads // tp,
                            model.head_dim,
                        )
                        if explicit_view and read_tokens
                        else None,
                        write_offset=(write_start, 0, 0)
                        if explicit_view or read_tokens
                        else None,
                        write_shape=(
                            write_tokens,
                            model.num_kv_heads // tp,
                            model.head_dim,
                        )
                        if explicit_view or read_tokens
                        else None,
                    )
                )
        for rank in range(tp):
            for weight_name, consumer_name, rank_shape in parameter_specs:
                weight = local_values[weight_name]
                identity = PersistentStateIdentity.create(
                    kind=StateKind.PARAMETER,
                    instance_ref=instance.id,
                    mesh_ref=mesh_ref,
                    request_ref=None,
                    layer_index=None,
                    tensor_ref=weight.id,
                    shard_index=rank,
                    generation=0,
                )
                declaration = PersistentStateDecl.create(
                    identity=identity,
                    shape=rank_shape,
                    dtype=weight.dtype,
                    layout=weight.logical_layout,
                    lifetime=PersistentStateLifetime.PERSISTENT,
                    access=PersistentStateAccess.READ_ONLY,
                )
                persistent_states.append(declaration)
                state_accesses.append(
                    StateAccess.create(
                        node_ref=node_id[consumer_name],
                        state_ref=declaration.id,
                        mode=StateAccessMode.READ,
                        rank=rank,
                    )
                )
            if train_forward:
                continue
            if exact_profile is None:
                append_kv_states(
                    rank=rank,
                    request_ref=profile.stable_id(),
                    capacity_tokens=profile.context_sum,
                    read_tokens=0
                    if profile.prefill_tokens
                    else profile.context_sum,
                    write_start=0
                    if profile.prefill_tokens
                    else profile.context_sum - profile.decode_tokens,
                    write_tokens=tokens,
                    explicit_view=False,
                )
            else:
                for request in exact_profile.requests:
                    append_kv_states(
                        rank=rank,
                        request_ref=f"{exact_profile.id}:{request.request_ref}",
                        capacity_tokens=request.kv_span.capacity_tokens,
                        read_tokens=request.kv_read_tokens,
                        write_start=request.context_tokens - request.query_tokens,
                        write_tokens=request.query_tokens,
                        explicit_view=True,
                    )
        return output_id

    layer_input = input_id
    for layer_index in range(model.num_layers):
        layer_input = add_layer(profile, layer_index, layer_input)

    final_norm_weight_id = f"{instance.id}.final_norm.weight"
    final_norm_output_id = f"{instance.id}.final_norm_out"
    lm_head_weight_id = f"{instance.id}.lm_head.weight"
    logits_id = f"{instance.id}.logits"
    terminal_consumer = (
        (f"{instance.id}.cross_entropy",)
        if train_forward
        else (f"{instance.id}.greedy_sample",)
        if infer_output is InferOutput.GREEDY_SAMPLE
        else ()
    )
    values.extend(
        (
            TensorValue(
                id=final_norm_weight_id,
                shape=(hidden,),
                dtype=model.dtype,
                logical_layout="H_replicated",
                sharding=replicated(1),
                producer=None,
                consumers=(f"{instance.id}.final_norm",),
                alias_set=None,
            ),
            TensorValue(
                id=final_norm_output_id,
                shape=mh,
                dtype=model.dtype,
                logical_layout="MH_shard_tp" if distributed else "MH",
                sharding=m_sharded if distributed else replicated(),
                producer=f"{instance.id}.final_norm",
                consumers=(f"{instance.id}.lm_head",),
                alias_set=None,
            ),
            TensorValue(
                id=lm_head_weight_id,
                shape=(hidden, model.vocab_size),
                dtype=model.dtype,
                logical_layout="HV_replicated",
                sharding=replicated(),
                producer=None,
                consumers=(f"{instance.id}.lm_head",),
                alias_set=None,
            ),
            TensorValue(
                id=logits_id,
                shape=(tokens, model.vocab_size),
                dtype=model.dtype,
                logical_layout="MV_shard_tp" if distributed else "MV",
                sharding=m_sharded if distributed else replicated(),
                producer=f"{instance.id}.lm_head",
                consumers=terminal_consumer,
                alias_set=None,
            ),
        )
    )
    nodes.extend(
        (
            LogicalNode(
                id=f"{instance.id}.final_norm",
                instance_id=instance.id,
                kind=OpKind.NORM,
                phase=OpPhase.FWD,
                stage=0,
                mesh_ref=mesh_ref,
                inputs=(layer_input, final_norm_weight_id),
                outputs=(final_norm_output_id,),
                workload=RmsNormWorkload(
                    logical_activation_shape=mh,
                    logical_output_shape=mh,
                    rank_activation_shape=rank_mh,
                    rank_output_shape=rank_mh,
                    logical_weight_shape=(hidden,),
                    rank_weight_shape=(hidden,),
                    epsilon=model.rms_norm_epsilon,
                    dtype=model.dtype,
                ),
                math=math,
                effects=NodeEffects(EffectKind.PURE, None, None),
                impl_ref="rms_norm",
            ),
            LogicalNode(
                id=f"{instance.id}.lm_head",
                instance_id=instance.id,
                kind=OpKind.GEMM,
                phase=OpPhase.FWD,
                stage=0,
                mesh_ref=mesh_ref,
                inputs=(final_norm_output_id, lm_head_weight_id),
                outputs=(logits_id,),
                workload=GemmWorkload(
                    logical_shape=(tokens, model.vocab_size, hidden),
                    rank_shape=(rank_tokens, model.vocab_size, hidden),
                    partition=(
                        GemmPartition.SEQUENCE_PARALLEL_REPLICATED_WEIGHT
                        if distributed
                        else GemmPartition.REPLICATED
                    ),
                    dtype=model.dtype,
                ),
                math=math,
                effects=NodeEffects(EffectKind.PURE, None, None),
                impl_ref="matmul_forward",
            ),
        )
    )
    if infer_output is InferOutput.GREEDY_SAMPLE:
        sampled_ids = f"{instance.id}.sampled_ids"
        values.append(
            TensorValue(
                id=sampled_ids,
                shape=(profile.num_seqs,),
                dtype=DType.INT32,
                logical_layout="S_sample_ids",
                sharding=replicated(1),
                producer=f"{instance.id}.greedy_sample",
                consumers=(),
                alias_set=None,
            )
        )
        nodes.append(
            LogicalNode(
                id=f"{instance.id}.greedy_sample",
                instance_id=instance.id,
                kind=OpKind.SAMPLING,
                phase=OpPhase.FWD,
                stage=0,
                mesh_ref=mesh_ref,
                inputs=(logits_id,),
                outputs=(sampled_ids,),
                workload=GreedySampleWorkload(
                    profile=profile,
                    mode=SamplingMode.GREEDY,
                    row_selection=SampleRowSelection.LAST_PER_SEQUENCE,
                    tp_degree=tp,
                    logical_logits_shape=(tokens, model.vocab_size),
                    rank_logits_shape=(rank_tokens, model.vocab_size),
                    logical_output_shape=(profile.num_seqs,),
                    rank_output_shape=(profile.num_seqs,),
                    sample_count=profile.num_seqs,
                    comparisons=profile.num_seqs * (model.vocab_size - 1),
                    logits_dtype=model.dtype,
                    output_dtype=DType.INT32,
                ),
                math=math,
                effects=NodeEffects(EffectKind.PURE, None, None),
                impl_ref="greedy_sample",
            )
        )

    if train_forward:
        labels_id = f"{instance.id}.labels"
        loss_id = f"{instance.id}.loss"
        row_sharding = (
            Sharding(mesh_ref, (MeshAxisName.TP,), ())
            if distributed
            else replicated(1)
        )
        values.extend(
            (
                TensorValue(
                    id=labels_id,
                    shape=(tokens,),
                    dtype=DType.INT32,
                    logical_layout="M_labels_shard_tp" if distributed else "M_labels",
                    sharding=row_sharding,
                    producer=None,
                    consumers=(f"{instance.id}.cross_entropy",),
                    alias_set=None,
                ),
                TensorValue(
                    id=loss_id,
                    shape=(tokens,),
                    dtype=DType.FP32,
                    logical_layout="M_loss_shard_tp" if distributed else "M_loss",
                    sharding=row_sharding,
                    producer=f"{instance.id}.cross_entropy",
                    consumers=(),
                    alias_set=None,
                ),
            )
        )
        nodes.append(
            LogicalNode(
                id=f"{instance.id}.cross_entropy",
                instance_id=instance.id,
                kind=OpKind.CE_FORWARD,
                phase=OpPhase.FWD,
                stage=0,
                mesh_ref=mesh_ref,
                inputs=(logits_id, labels_id),
                outputs=(loss_id,),
                workload=CrossEntropyForwardWorkload(
                    profile=profile,
                    reduction=CrossEntropyReduction.NONE,
                    logical_logits_shape=(tokens, model.vocab_size),
                    rank_logits_shape=(rank_tokens, model.vocab_size),
                    logical_label_shape=(tokens,),
                    rank_label_shape=(rank_tokens,),
                    logical_loss_shape=(tokens,),
                    rank_loss_shape=(rank_tokens,),
                    logits_dtype=model.dtype,
                    label_dtype=DType.INT32,
                    loss_dtype=DType.FP32,
                ),
                math=math,
                effects=NodeEffects(EffectKind.PURE, None, None),
                impl_ref="cross_entropy_forward",
            )
        )
    for rank in range(tp):
        for tensor_ref, node_ref, shape, layout in (
            (
                embedding_weight_id,
                f"{instance.id}.embedding",
                (model.vocab_size, hidden),
                "VH_replicated",
            ),
            (
                final_norm_weight_id,
                f"{instance.id}.final_norm",
                (hidden,),
                "H_replicated",
            ),
            (
                lm_head_weight_id,
                f"{instance.id}.lm_head",
                (hidden, model.vocab_size),
                "HV_replicated",
            ),
        ):
            identity = PersistentStateIdentity.create(
                kind=StateKind.PARAMETER,
                instance_ref=instance.id,
                mesh_ref=mesh_ref,
                request_ref=None,
                layer_index=None,
                tensor_ref=tensor_ref,
                shard_index=rank,
                generation=0,
            )
            declaration = PersistentStateDecl.create(
                identity=identity,
                shape=shape,
                dtype=model.dtype,
                layout=layout,
                lifetime=PersistentStateLifetime.PERSISTENT,
                access=PersistentStateAccess.READ_ONLY,
            )
            persistent_states.append(declaration)
            state_accesses.append(
                StateAccess.create(
                    node_ref=node_ref,
                    state_ref=declaration.id,
                    mode=StateAccessMode.READ,
                    rank=rank,
                )
            )

    edges = tuple(
        GraphEdge(
            id=(
                f"{value.id}.edge_to."
                f"{consumer.rsplit('.', 1)[-1]}"
            ),
            kind=EdgeKind.DATA,
            source_node=value.producer,
            destination_node=consumer,
            value_id=value.id,
        )
        for value in values
        if value.producer is not None
        for consumer in value.consumers
    )
    graph = IR0.create(
        producer_pass=producer_pass,
        job=job,
        instances=(instance,),
        nodes=tuple(nodes),
        values=tuple(values),
        edges=edges,
        fusion_candidates=tuple(candidates),
        persistent_states=tuple(persistent_states),
        state_accesses=tuple(state_accesses),
        profile=profile,
        train=train,
    )
    graph.validate("expanded_graph")
    return graph


def _expand_profile_graph(
    template: IR0Template,
    profile: ProfileKey,
    exact_profile: Stage3StaticProfile | None = None,
) -> IR0:
    """Compatibility wrapper for the inference expansion entry point."""

    return _expand_dense_graph(
        template.model,
        template.instance,
        profile,
        exact_profile,
        job=template.job,
        train=None,
        infer_output=template.infer_output,
        producer_pass="logical_expand",
    )


def logical_expand(template: IR0Template) -> ExpandedIR0Bundle:
    """Expand every canonical profile into an independent Dense IR-0 graph."""

    template.validate("template")
    _validate_scope(template)
    entries = tuple(
        ExpandedProfileIR0.create(
            source_template_id=template.id,
            weight=profile.weight,
            graph=_expand_profile_graph(
                template,
                profile.key,
                profile.exact_profile,
            ),
        )
        for profile in template.profiles
    )
    result = ExpandedIR0Bundle.create(source_template=template, entries=entries)
    result.validate("expanded_ir0_bundle")
    # Keep the producer independent from validator module import order while
    # still making every returned bundle pass the complete N2 semantic gate.
    from .validate_logical_bundle import DenseLogicalBundleValidator

    DenseLogicalBundleValidator.validate(template, result)
    return result
