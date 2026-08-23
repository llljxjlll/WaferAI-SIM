from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.schema.experiment import EXPERIMENT_SCHEMA_VERSION
from llm.frontend.wafer_frontend.schema.common import (
    DType,
    MeshAxisName,
    ProfileKey,
    Sharding,
    TensorValue,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    CollectiveRole,
    CollectiveWorkload,
    DeviceMesh,
    EdgeKind,
    EffectKind,
    FusionCandidate,
    FusionImpl,
    FusionOrigin,
    FusionSemanticContract,
    FusionPattern,
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
    ParallelAxes,
    ReduceOp,
)
from llm.frontend.wafer_frontend.schema.persistent_state import HbmAddressSpace
from llm.frontend.wafer_frontend.schema.ir1 import (
    C2CPort,
    CoreSpec,
    D2DLink,
    DieSpec,
    Direction,
    FusedOpSkeleton,
    GroupEmbedding,
    IR1,
    PairRoute,
    PhysicalFabric,
    PhysicalGroup,
    PhysicalInstance,
    PhysicalNode,
    RankPlacement,
    ResourceCapacity,
    RouteHop,
    RoutingMode,
    MemoryInitiator,
    SramAllocator,
    SramProfile,
    SramRegionSpec,
)
from llm.frontend.wafer_frontend.schema.n4 import InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.n5 import IntraDieSchedulingContext


def valid_hbm_address_spaces(
    fabric: PhysicalFabric,
    *,
    size_bytes_per_die: int = 1 << 30,
) -> tuple[HbmAddressSpace, ...]:
    """Explicit globally addressed NUMA HBM spaces for frontend unit fixtures."""

    return tuple(
        HbmAddressSpace.create(
            die_id=die.id,
            base_address=die.id * size_bytes_per_die,
            size_bytes=size_bytes_per_die,
            alignment_bytes=64,
        )
        for die in fabric.dies
    )


def naive_inter_die_planning_context(
    producer_pass: str,
) -> InterDiePlanningContext:
    registry = production_registry()
    return InterDiePlanningContext.create(
        producer_pass=producer_pass,
        fused_policy=registry.instantiate(
            RegistryKind.INTER_DIE, "naive"
        ).selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE,
            "direct_all_gather",
        ).selection,
    )


def naive_intra_die_scheduling_context(
    producer_pass: str,
) -> IntraDieSchedulingContext:
    registry = production_registry()
    return IntraDieSchedulingContext.create(
        producer_pass=producer_pass,
        policy=registry.instantiate(
            RegistryKind.INTRA_DIE, "naive"
        ).selection,
    )


_VALID_SPEC = {
    "schema_version": EXPERIMENT_SCHEMA_VERSION,
    "model": {
        "source": "analytic",
        "arch": "llama",
        "V": 512,
        "H": 256,
        "I": 512,
        "NH": 4,
        "KVH": 2,
        "DH": 64,
        "rotary_dim": 64,
        "L": 1,
        "dtype": "fp16",
        "tie_word_embeddings": False,
        "rms_norm_epsilon": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "moe": None,
    },
    "hardware": {"ref": "notes/frontend/examples/hardware_2x1.json"},
    "workload": {
        "mode": "infer",
        "infer": {
            "source": "static_profile",
            "output": "logits",
            "profile": {
                "prefill_tokens": 32,
                "decode_tokens": 0,
                "num_seqs": 1,
                "context_sum": 32,
                "context_max": 32,
                "kv_pages": 2,
                "expert_load": None,
            },
        },
    },
    "parallel": {
        "instances": [
            {
                "id": "P0",
                "role": "prefill",
                "tp": 2,
                "sp": True,
                "replicas": 1,
            }
        ]
    },
    "placement": {"strategy": "compact", "groups": []},
    "policy": {
        "partition": "gemm_coll",
        "inter_die": "naive",
        "intra_die": "naive",
    },
    "backend": {
        "execution": "unified_stream",
        "ordinary_lowering": "json_coarse",
        "fused_lowering": "isa_region",
        "standalone_collective_lowering": "strict_actions",
        "reduction_contract": {
            "accumulate": "fp32",
            "rounding": "rne",
            "validation": "timing",
        },
        "transport": "strict",
        "static_link": True,
        "dynamic_region_dispatch": False,
    },
}


def valid_spec() -> dict[str, object]:
    return deepcopy(_VALID_SPEC)


def static_profile() -> ProfileKey:
    return ProfileKey(
        prefill_tokens=32,
        decode_tokens=0,
        num_seqs=1,
        context_sum=32,
        context_max=32,
        kv_pages=2,
        expert_load=None,
    )


def valid_ir0() -> IR0:
    mesh = DeviceMesh("mesh_tp", (MeshAxis(MeshAxisName.TP, 2),))
    instance = LogicalInstance(
        id="P0",
        role=LogicalRole.PREFILL,
        replicas=1,
        parallel=ParallelAxes(tp=2, sp=True, dp=1, pp=1, ep=1),
        meshes=(mesh,),
    )
    gemm = LogicalNode(
        id="gemm_0",
        instance_id="P0",
        kind=OpKind.GEMM,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref="mesh_tp",
        inputs=("v_in",),
        outputs=("v_partial",),
        workload=GemmWorkload(
            logical_shape=(32, 128, 256),
            rank_shape=(32, 128, 128),
            partition=GemmPartition.ROW_PARALLEL,
            dtype=DType.FP16,
        ),
        math=NodeMath(
            accumulation_dtype=DType.FP32,
            numerical_policy=NumericalPolicy.BITWISE,
        ),
        effects=NodeEffects(kind=EffectKind.PURE, effect_token=None, alias_set=None),
        impl_ref="matmul_forward",
    )
    reduce_scatter = LogicalNode(
        id="rs_0",
        instance_id="P0",
        kind=OpKind.COLLECTIVE,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref="mesh_tp",
        inputs=("v_partial",),
        outputs=("v_out",),
        workload=CollectiveWorkload(
            collective=CollectiveKind.REDUCE_SCATTER,
            reduce_op=ReduceOp.SUM,
            mesh_axes=(MeshAxisName.TP,),
            participant_count=2,
            reduction_mesh_axes=(MeshAxisName.TP,),
            scatter_tensor_axis=0,
            gather_tensor_axis=None,
            logical_tensor_bytes=8192,
            rank_input_bytes=8192,
            rank_output_bytes=4096,
            rank_logical_payload_bytes=4096,
            group_logical_payload_bytes=8192,
            dtype=DType.FP16,
            role=CollectiveRole.ACTIVATION,
            input_layout="MN_partial_tp",
            output_layout="MN_shard_tp",
        ),
        math=NodeMath(
            accumulation_dtype=DType.FP32,
            numerical_policy=NumericalPolicy.BITWISE,
        ),
        effects=NodeEffects(kind=EffectKind.PURE, effect_token=None, alias_set=None),
        impl_ref="collective_derived",
    )
    values = (
        TensorValue(
            id="v_in",
            shape=(32, 256),
            dtype=DType.FP16,
            logical_layout="MK",
            sharding=Sharding("mesh_tp", (None, MeshAxisName.TP), ()),
            producer=None,
            consumers=("gemm_0",),
            alias_set=None,
        ),
        TensorValue(
            id="v_partial",
            shape=(32, 128),
            dtype=DType.FP16,
            logical_layout="MN_partial_tp",
            sharding=Sharding("mesh_tp", (None, None), (MeshAxisName.TP,)),
            producer="gemm_0",
            consumers=("rs_0",),
            alias_set=None,
        ),
        TensorValue(
            id="v_out",
            shape=(32, 128),
            dtype=DType.FP16,
            logical_layout="MN_shard_tp",
            sharding=Sharding("mesh_tp", (MeshAxisName.TP, None), ()),
            producer="rs_0",
            consumers=(),
            alias_set=None,
        ),
    )
    candidate = FusionCandidate(
        id="fusion_0",
        members=("gemm_0", "rs_0"),
        boundary_inputs=("v_in",),
        boundary_outputs=("v_out",),
        semantic_contract=FusionSemanticContract(
            pattern=FusionPattern.GEMM_RS,
            tile_domain=("M", "N"),
            reduction_axes=(1,),
            input_layouts=("MK",),
            output_layout="MN_shard_tp",
            numerical_policy=NumericalPolicy.TOLERANCE,
        ),
        impl=FusionImpl.NONE,
        origin=FusionOrigin.DECLARED,
    )
    result = IR0.create(
        producer_pass="analytic_fixture",
        job=JobKind.INFER,
        instances=(instance,),
        nodes=(gemm, reduce_scatter),
        values=values,
        edges=(GraphEdge("edge_partial", EdgeKind.DATA, "gemm_0", "rs_0", "v_partial"),),
        fusion_candidates=(candidate,),
        profile=static_profile(),
    )
    result.validate()
    return result


def valid_ir1() -> IR1:
    ir0 = valid_ir0()
    sram_profile = SramProfile(
        id="sram_default",
        capacity_bytes=1 << 20,
        allocation_alignment_bytes=64,
        bank_count=4,
        bank_interleave_bytes=64,
        real_data_path=True,
        manual_regions=True,
        manual_memory_schedule=True,
        regions=(
            SramRegionSpec(
                id="sram_main",
                name="sram",
                base_bytes=0,
                size_bytes=1 << 20,
                allocator=SramAllocator.BLOCK,
                spillable=False,
                access=(
                    MemoryInitiator.COMPUTE,
                    MemoryInitiator.DTE,
                    MemoryInitiator.LSU,
                    MemoryInitiator.NOC_RX,
                    MemoryInitiator.LEGACY,
                ),
            ),
        ),
    )

    def cores(die_id: int) -> tuple[CoreSpec, ...]:
        return tuple(
            CoreSpec(
                id=f"core_{die_id}_{local_core_id}",
                local_core_id=local_core_id,
                runtime_core_id=die_id * 16 + local_core_id,
                noc_coord=(local_core_id % 4, local_core_id // 4),
                sram_profile_ref=sram_profile.id,
            )
            for local_core_id in range(16)
        )

    dies = (
        DieSpec(
            id=0,
            coord=(0, 0),
            noc_grid=(4, 4),
            noc_bytes_per_cycle=128,
            hbm_bytes_per_cycle=128,
            cores=cores(0),
            ports=(
                C2CPort(
                    id="east_0",
                    runtime_port_id=0,
                    side=Direction.EAST,
                    direction=Direction.EAST,
                    noc_coord=(3, 1),
                    egress_resource_id="port_0_east",
                    bytes_per_cycle=64,
                    buffer_packets=8,
                ),
            ),
        ),
        DieSpec(
            id=1,
            coord=(1, 0),
            noc_grid=(4, 4),
            noc_bytes_per_cycle=128,
            hbm_bytes_per_cycle=128,
            cores=cores(1),
            ports=(
                C2CPort(
                    id="west_1",
                    runtime_port_id=0,
                    side=Direction.WEST,
                    direction=Direction.WEST,
                    noc_coord=(0, 1),
                    egress_resource_id="port_1_west",
                    bytes_per_cycle=64,
                    buffer_packets=8,
                ),
            ),
        ),
    )
    fabric = PhysicalFabric(
        routing_mode=RoutingMode.BACKEND_XY_V1,
        die_grid=(2, 1),
        sram_profiles=(sram_profile,),
        dies=dies,
        links=(
            D2DLink(
                id="link_0_1",
                source_die=0,
                source_port_ref="east_0",
                destination_die=1,
                destination_port_ref="west_1",
                bytes_per_cycle=64,
                latency_cycles=2,
                resource_id="d2d_0_1",
                link_group_ref="cut_0_1",
            ),
            D2DLink(
                id="link_1_0",
                source_die=1,
                source_port_ref="west_1",
                destination_die=0,
                destination_port_ref="east_0",
                bytes_per_cycle=64,
                latency_cycles=2,
                resource_id="d2d_1_0",
                link_group_ref="cut_1_0",
            ),
        ),
    )
    embedding = GroupEmbedding(
        routes=(
            PairRoute(
                id="route_0_1",
                source_rank=0,
                destination_rank=1,
                die_path=(0, 1),
                hops=(
                    RouteHop(
                        index=0,
                        link_ref="link_0_1",
                        source_die=0,
                        source_port_ref="east_0",
                        destination_die=1,
                        destination_port_ref="west_1",
                        resource_ids=("port_0_east", "d2d_0_1", "cut_0_1"),
                    ),
                ),
                resource_ids=("port_0_east", "d2d_0_1", "cut_0_1"),
            ),
            PairRoute(
                id="route_1_0",
                source_rank=1,
                destination_rank=0,
                die_path=(1, 0),
                hops=(
                    RouteHop(
                        index=0,
                        link_ref="link_1_0",
                        source_die=1,
                        source_port_ref="west_1",
                        destination_die=0,
                        destination_port_ref="east_0",
                        resource_ids=("port_1_west", "d2d_1_0", "cut_1_0"),
                    ),
                ),
                resource_ids=("port_1_west", "d2d_1_0", "cut_1_0"),
            ),
        ),
        resource_capacities=(
            ResourceCapacity("port_0_east", 64),
            ResourceCapacity("port_1_west", 64),
            ResourceCapacity("d2d_0_1", 64),
            ResourceCapacity("d2d_1_0", 64),
            ResourceCapacity("cut_0_1", 64),
            ResourceCapacity("cut_1_0", 64),
        ),
        canonical_profiles=(),
    )
    group = PhysicalGroup(
        id="group_tp",
        instance_id="P0.r0",
        mesh_ref="mesh_tp",
        axis=MeshAxisName.TP,
        logical_shape=(2,),
        placements=(RankPlacement(0, 0, (0,)), RankPlacement(1, 1, (1,))),
        embedding=embedding,
    )
    gemm = PhysicalNode(
        id="p_gemm_0",
        origin_node_id="gemm_0",
        instance_id="P0.r0",
        kind=OpKind.GEMM,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref="mesh_tp",
        execution_group_ref="group_tp",
        inputs=("p_v_in",),
        outputs=("p_v_partial",),
        workload=GemmWorkload(
            logical_shape=(32, 128, 256),
            rank_shape=(32, 128, 128),
            partition=GemmPartition.ROW_PARALLEL,
            dtype=DType.FP16,
        ),
        math=NodeMath(DType.FP32, NumericalPolicy.BITWISE),
        effects=NodeEffects(EffectKind.PURE, None, None),
        impl_ref="matmul_forward",
    )
    rs = PhysicalNode(
        id="p_rs_0",
        origin_node_id="rs_0",
        instance_id="P0.r0",
        kind=OpKind.COLLECTIVE,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref="mesh_tp",
        execution_group_ref="group_tp",
        inputs=("p_v_partial",),
        outputs=("p_v_out",),
        workload=CollectiveWorkload(
            collective=CollectiveKind.REDUCE_SCATTER,
            reduce_op=ReduceOp.SUM,
            mesh_axes=(MeshAxisName.TP,),
            participant_count=2,
            reduction_mesh_axes=(MeshAxisName.TP,),
            scatter_tensor_axis=0,
            gather_tensor_axis=None,
            logical_tensor_bytes=8192,
            rank_input_bytes=8192,
            rank_output_bytes=4096,
            rank_logical_payload_bytes=4096,
            group_logical_payload_bytes=8192,
            dtype=DType.FP16,
            role=CollectiveRole.ACTIVATION,
            input_layout="MN_partial_tp",
            output_layout="MN_shard_tp",
        ),
        math=NodeMath(DType.FP32, NumericalPolicy.BITWISE),
        effects=NodeEffects(EffectKind.PURE, None, None),
        impl_ref="collective_derived",
    )
    values = (
        TensorValue("p_v_in", (32, 256), DType.FP16, "MK", Sharding("mesh_tp", (None, MeshAxisName.TP), ()), None, ("p_gemm_0",), None),
        TensorValue("p_v_partial", (32, 128), DType.FP16, "MN_partial_tp", Sharding("mesh_tp", (None, None), (MeshAxisName.TP,)), "p_gemm_0", ("p_rs_0",), None),
        TensorValue("p_v_out", (32, 128), DType.FP16, "MN_shard_tp", Sharding("mesh_tp", (MeshAxisName.TP, None), ()), "p_rs_0", (), None),
    )
    instance = PhysicalInstance(
        id="P0.r0",
        origin_instance_id="P0",
        role=LogicalRole.PREFILL,
        die_region=(0, 1),
        group_ids=("group_tp",),
        node_ids=("p_gemm_0", "p_rs_0"),
    )
    skeleton = FusedOpSkeleton(
        id="p_fusion_0",
        fusion_ref="fusion_0",
        instance_id="P0.r0",
        member_node_ids=("p_gemm_0", "p_rs_0"),
        boundary_inputs=("p_v_in",),
        boundary_outputs=("p_v_out",),
        semantic_contract=ir0.fusion_candidates[0].semantic_contract,
        impl=FusionImpl.NAIVE,
    )
    candidate = replace(
        ir0.fusion_candidates[0],
        members=("p_gemm_0", "p_rs_0"),
        boundary_inputs=("p_v_in",),
        boundary_outputs=("p_v_out",),
    )
    result = IR1.create(
        producer_pass="placement_fixture",
        source_ir0_id=ir0.id,
        profile=static_profile(),
        fabric=fabric,
        instances=(instance,),
        groups=(group,),
        nodes=(gemm, rs),
        values=values,
        edges=(GraphEdge("p_edge_partial", EdgeKind.DATA, "p_gemm_0", "p_rs_0", "p_v_partial"),),
        fusion_candidates=(candidate,),
        fused_op_skeletons=(skeleton,),
    )
    result.validate()
    return result
