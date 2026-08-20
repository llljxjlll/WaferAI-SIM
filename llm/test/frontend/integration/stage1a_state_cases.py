"""Independent Stage1a persistent-state integration cases.

The builders in this module intentionally do not import unit-test fixtures.
They construct the smallest legal upstream IR and then exercise the production
projection, scheduling, global-action, lowering, lifecycle, linker, and
ProgramIo implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from llm.frontend.wafer_frontend.lowering.coarse import NaiveCoarseLowering
from llm.frontend.wafer_frontend.lowering.context import LoweringContext
from llm.frontend.wafer_frontend.lowering.lifecycle import (
    add_fixed_sram_lifecycle,
)
from llm.frontend.wafer_frontend.lowering.linker import NaiveManifestLinker
from llm.frontend.wafer_frontend.lowering.state import NaiveStateDmaLowering
from llm.frontend.wafer_frontend.passes import (
    build_ir0,
    hbm_address_spaces_from_data,
    link_profile,
    logical_expand,
    lower_profile,
    physical_fabric_from_data,
    place_ir0,
)
from llm.frontend.wafer_frontend.passes.global_action import (
    build_global_action_dag,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_timing_program_io,
    _resolved_abis,
)
from llm.frontend.wafer_frontend.policies.naive_fusion_partition import (
    NaiveFusionPartition,
)
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    LinkedProgramManifest,
)
from llm.frontend.wafer_frontend.schema.common import (
    DType,
    MeshAxisName,
    ProfileKey,
    Sharding,
    TensorValue,
    stable_artifact_id,
)
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.schema.ir0 import (
    EffectKind,
    GemmPartition,
    GemmWorkload,
    CollectiveKind,
    LogicalRole,
    NodeEffects,
    NodeMath,
    NumericalPolicy,
    OpKind,
    OpPhase,
    StateAccess,
    StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.ir1 import (
    CoreSpec,
    DieSpec,
    GroupEmbedding,
    IR1,
    MemoryInitiator,
    PhysicalFabric,
    PhysicalGroup,
    PhysicalInstance,
    PhysicalNode,
    RankPlacement,
    RoutingMode,
    SramAllocator,
    SramProfile,
    SramRegionSpec,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    IR2ProjectionResult,
    IntraDieScheduleSet,
    SemanticTaskKind,
    StateIoOrigin,
    StateTransferOrigin,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
)
from llm.frontend.wafer_frontend.schema.n4 import (
    INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
    InterDiePlannedProfile,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
    GlobalActionProfile,
    IntraDieSchedulingContext,
    ProjectToIR2Context,
    ProjectedProfileIR2,
    ScheduledProfileIR2,
)
from llm.frontend.wafer_frontend.schema.n6 import (
    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LinkedProgramProfile,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.persistent_state import (
    HbmAddressSpace,
    HbmBinding,
    PersistentStateAccess,
    PersistentStateDecl,
    PersistentStateIdentity,
    PersistentStateLifetime,
    PersistentStateManifest,
    StateKind,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramIoContract,
)
from llm.frontend.wafer_frontend.schema.state_transfer import (
    StateTransferContract,
)
from llm.frontend.wafer_frontend.schema.serde import from_data
from llm.frontend.wafer_frontend.schema.policy import RegistryKind
from llm.frontend.wafer_frontend.policies.registry import production_registry


_P1_HBM_BYTES_PER_CYCLE = 16
_P1_HBM_BASE = 0x1000
_P1_SEED = bytes(range(128))
_P1_ARTIFACT_SHA256 = "10" * 32
_ROOT = Path(__file__).resolve().parents[4]
_K1_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x1.json"
_K1_HBM_BYTES_PER_CYCLE = 16
_K1_PAYLOAD_BYTES = (0x11, 0x22, 0x33, 0x44)
_K1_ARTIFACT_SHA256 = "20" * 32
_PD1_ARTIFACT_SHA256 = "30" * 32
_PD1_SOURCE_BYTES = {
    StateKind.KV_KEY: bytes([0x5A]) * 32,
    StateKind.KV_VALUE: bytes([0xA5]) * 32,
}




@dataclass(frozen=True, slots=True)
class ParameterFoundationCase:
    graph: IR1
    projection: IR2ProjectionResult
    schedule_set: IntraDieScheduleSet
    global_dag: GlobalActionDAG
    lowering_context: LoweringContext
    fragments: tuple[CommandFragment, ...]
    manifest: LinkedProgramManifest
    linked_profile: LinkedProgramProfile
    program_io: ProgramIoContract
    state_ref: str
    hbm_binding_ref: str
    seed: bytes
    gemm_flops: int
    hbm_capacity_floor_cycles: int



@dataclass(frozen=True, slots=True)
class CrossActionKvFoundationCase:
    graph: IR1
    projection: IR2ProjectionResult
    schedule_set: IntraDieScheduleSet
    global_dag: GlobalActionDAG
    lowering_context: LoweringContext
    manifest: LinkedProgramManifest
    linked_profile: LinkedProgramProfile
    program_io: ProgramIoContract
    state_payloads: tuple[tuple[str, bytes], ...]
    write_access_ids: tuple[str, ...]
    read_access_ids: tuple[str, ...]
    write_staging_ids: tuple[str, ...]
    read_staging_ids: tuple[str, ...]
    hbm_write_bytes: int
    hbm_read_bytes: int
    d2d_bytes: int
    middle_action_count: int
    hbm_capacity_floor_cycles: int


@dataclass(frozen=True, slots=True)
class Pd1FoundationCase:
    graph: IR1
    contracts: tuple[StateTransferContract, ...]
    projection: IR2ProjectionResult
    schedule_set: IntraDieScheduleSet
    global_dag: GlobalActionDAG
    lowering_context: LoweringContext
    manifest: LinkedProgramManifest
    linked_profile: LinkedProgramProfile
    program_io: ProgramIoContract
    source_payloads: tuple[tuple[str, bytes], ...]
    destination_expected: tuple[tuple[str, bytes], ...]
    source_access_ids: tuple[str, ...]
    destination_access_ids: tuple[str, ...]
    bridge_action_ids: tuple[str, ...]
    d2d_bytes: int


def _p1_fabric() -> PhysicalFabric:
    region = SramRegionSpec(
        id="p1.comm",
        name="comm",
        base_bytes=0,
        size_bytes=1024,
        allocator=SramAllocator.BLOCK,
        spillable=False,
        access=(MemoryInitiator.COMPUTE, MemoryInitiator.LSU),
    )
    profile = SramProfile(
        id="p1.sram",
        capacity_bytes=1024,
        allocation_alignment_bytes=64,
        bank_count=4,
        bank_interleave_bytes=64,
        real_data_path=True,
        manual_regions=True,
        manual_memory_schedule=True,
        regions=(region,),
    )
    core = CoreSpec(
        id="p1.core0",
        local_core_id=0,
        runtime_core_id=0,
        noc_coord=(0, 0),
        sram_profile_ref=profile.id,
    )
    die = DieSpec(
        id=0,
        coord=(0, 0),
        noc_grid=(1, 1),
        noc_bytes_per_cycle=16,
        hbm_bytes_per_cycle=_P1_HBM_BYTES_PER_CYCLE,
        cores=(core,),
        ports=(),
    )
    fabric = PhysicalFabric(
        routing_mode=RoutingMode.BACKEND_XY_V1,
        die_grid=(1, 1),
        sram_profiles=(profile,),
        dies=(die,),
        links=(),
    )
    fabric.validate("p1.fabric")
    return fabric


def _p1_graph() -> tuple[IR1, PersistentStateDecl, HbmBinding]:
    fabric = _p1_fabric()
    profile = ProfileKey(
        prefill_tokens=1,
        decode_tokens=0,
        num_seqs=1,
        context_sum=1,
        context_max=1,
        kv_pages=1,
        expert_load=None,
    )
    group = PhysicalGroup(
        id="p1.group",
        instance_id="p1.instance",
        mesh_ref="p1.mesh",
        axis=MeshAxisName.TP,
        logical_shape=(1,),
        placements=(RankPlacement(rank=0, die_id=0, logical_coord=(0,)),),
        embedding=GroupEmbedding(
            routes=(),
            resource_capacities=(),
            canonical_profiles=(),
        ),
    )
    node = PhysicalNode(
        id="p1.matmul",
        origin_node_id="p1.matmul.logical",
        instance_id="p1.instance",
        kind=OpKind.GEMM,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref="p1.mesh",
        execution_group_ref=group.id,
        inputs=("p1.x", "p1.w"),
        outputs=("p1.y",),
        workload=GemmWorkload(
            logical_shape=(1, 8, 8),
            rank_shape=(1, 8, 8),
            partition=GemmPartition.REPLICATED,
            dtype=DType.FP16,
        ),
        math=NodeMath(
            accumulation_dtype=DType.FP32,
            numerical_policy=NumericalPolicy.BITWISE,
        ),
        effects=NodeEffects(
            kind=EffectKind.PURE,
            effect_token=None,
            alias_set=None,
        ),
        impl_ref="matmul_forward",
    )
    unsharded = Sharding("p1.mesh", (None, None), ())
    values = (
        TensorValue(
            id="p1.x",
            shape=(1, 8),
            dtype=DType.FP16,
            logical_layout="MK",
            sharding=unsharded,
            producer=None,
            consumers=(node.id,),
            alias_set=None,
        ),
        TensorValue(
            id="p1.w",
            shape=(8, 8),
            dtype=DType.FP16,
            logical_layout="KN",
            sharding=unsharded,
            producer=None,
            consumers=(node.id,),
            alias_set=None,
        ),
        TensorValue(
            id="p1.y",
            shape=(1, 8),
            dtype=DType.FP16,
            logical_layout="MN",
            sharding=unsharded,
            producer=node.id,
            consumers=(),
            alias_set=None,
        ),
    )
    instance = PhysicalInstance(
        id="p1.instance",
        origin_instance_id="p1.instance.logical",
        role=LogicalRole.PREFILL,
        die_region=(0,),
        group_ids=(group.id,),
        node_ids=(node.id,),
    )

    identity = PersistentStateIdentity.create(
        kind=StateKind.PARAMETER,
        instance_ref=instance.id,
        mesh_ref=group.mesh_ref,
        request_ref=None,
        layer_index=None,
        tensor_ref="p1.w",
        shard_index=0,
        generation=0,
    )
    declaration = PersistentStateDecl.create(
        identity=identity,
        shape=(8, 8),
        dtype=DType.FP16,
        layout="KN",
        lifetime=PersistentStateLifetime.PERSISTENT,
        access=PersistentStateAccess.READ_ONLY,
    )
    address_space = HbmAddressSpace.create(
        die_id=0,
        base_address=_P1_HBM_BASE,
        size_bytes=4096,
        alignment_bytes=64,
    )
    binding = HbmBinding.create(
        state_ref=declaration.id,
        die_id=0,
        address=_P1_HBM_BASE,
        size_bytes=declaration.tensor_bytes,
    )
    state_manifest = PersistentStateManifest.create(
        address_spaces=(address_space,),
        declarations=(declaration,),
        bindings=(binding,),
    )
    state_access = StateAccess.create(
        node_ref=node.id,
        state_ref=declaration.id,
        mode=StateAccessMode.READ,
        rank=0,
    )
    graph = IR1.create(
        producer_pass="stage1a_state_cases",
        source_ir0_id="p1.ir0",
        profile=profile,
        fabric=fabric,
        instances=(instance,),
        groups=(group,),
        nodes=(node,),
        values=values,
        edges=(),
        state_accesses=(state_access,),
        persistent_state_manifest=state_manifest,
    )
    graph.validate("p1.graph")
    return graph, declaration, binding


def _linked_profile(
    context: LoweringContext,
    manifest: LinkedProgramManifest,
) -> LinkedProgramProfile:
    leaf_fragments = tuple(
        sorted(
            (
                linked.fragment
                if hasattr(linked, "fragment")
                else linked
                for linked in manifest.fragments
            ),
            key=lambda fragment: fragment.id,
        )
    )
    if any(type(fragment) is not CommandFragment for fragment in leaf_fragments):
        raise AssertionError("P1 must lower only command-fragment leaves")
    semantic_key = {
        "source_lowered_entry_id": "p1.lowered",
        "source_global_action_entry_id": "p1.global",
        "source_scheduled_entry_id": "p1.scheduled",
        "source_projected_entry_id": "p1.projected",
        "source_planned_entry_id": "p1.planned",
        "source_partitioned_entry_id": "p1.partitioned",
        "source_ir1_id": context.ir1.id,
        "projection_context_id": "p1.projection.context",
        "scheduling_context_id": "p1.scheduling.context",
        "profile_id": context.ir1.profile.stable_id(),
        "weight": 1.0,
        "lowering_context": context,
        "leaf_fragments": leaf_fragments,
        "manifest": manifest,
    }
    linked = LinkedProgramProfile(
        id=stable_artifact_id(
            "linked_program_profile",
            semantic_key,
            schema_version=LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
        ),
        **semantic_key,
    )
    linked.validate("p1.linked_profile")
    return linked


def build_parameter_foundation_case() -> ParameterFoundationCase:
    """Build P1: one 128-byte parameter load feeding one timing MATMUL."""

    graph, declaration, binding = _p1_graph()
    projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
    projection.validate_against(graph, (), ())
    schedule_set = NaiveIntraDiePolicy().schedule(projection, graph)
    schedule_set.validate_against(projection, graph)
    global_dag = build_global_action_dag(graph, projection, schedule_set)
    global_dag.validate_against(graph, projection, schedule_set)

    context = LoweringContext(
        ir1=graph,
        fusion_plans=(),
        standalone_plans=(),
        projection=projection,
        schedule_set=schedule_set,
        global_dag=global_dag,
    )
    context.validate("p1.lowering_context")

    state_lowerer = NaiveStateDmaLowering()
    coarse_lowerer = NaiveCoarseLowering()
    fragments = tuple(
        add_fixed_sram_lifecycle(
            (
                state_lowerer.lower(action, context)
                if action.task_kind is SemanticTaskKind.DMA_IN
                else coarse_lowerer.lower(action, context)
            ),
            context,
        )
        for action in global_dag.actions
    )
    manifest = NaiveManifestLinker().link(context, fragments)
    linked_profile = _linked_profile(context, manifest)
    program_io = build_timing_program_io(
        linked_profile,
        _P1_ARTIFACT_SHA256,
        state_seed_overrides={declaration.id: _P1_SEED},
    )
    gemm_flops = 2 * 1 * 8 * 8
    hbm_capacity_floor_cycles = (
        declaration.tensor_bytes + _P1_HBM_BYTES_PER_CYCLE - 1
    ) // _P1_HBM_BYTES_PER_CYCLE
    return ParameterFoundationCase(
        graph=graph,
        projection=projection,
        schedule_set=schedule_set,
        global_dag=global_dag,
        lowering_context=context,
        fragments=fragments,
        manifest=manifest,
        linked_profile=linked_profile,
        program_io=program_io,
        state_ref=declaration.id,
        hbm_binding_ref=binding.id,
        seed=_P1_SEED,
        gemm_flops=gemm_flops,
        hbm_capacity_floor_cycles=hbm_capacity_floor_cycles,
    )


def _k1_spec() -> ExperimentSpec:
    raw = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "model": {
            "source": "analytic",
            "arch": "llama",
            "V": 32,
            "H": 8,
            "I": 16,
            "NH": 2,
            "KVH": 2,
            "DH": 4,
            "rotary_dim": 4,
            "L": 2,
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
                    "prefill_tokens": 2,
                    "decode_tokens": 0,
                    "num_seqs": 1,
                    "context_sum": 2,
                    "context_max": 2,
                    "kv_pages": 1,
                    "expert_load": None,
                },
            },
        },
        "parallel": {
            "instances": [
                {
                    "id": "P0",
                    "role": "prefill",
                    "tp": 1,
                    "sp": False,
                    "replicas": 1,
                    "dp": 1,
                    "pp": 1,
                    "ep": 1,
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
    return from_data(ExperimentSpec, raw, path="k1.spec")


def _k1_graph() -> tuple[
    IR1,
    tuple[PersistentStateDecl, ...],
    tuple[StateAccess, ...],
    tuple[StateAccess, ...],
]:
    spec = _k1_spec()
    hardware = json.loads(_K1_HARDWARE.read_text(encoding="utf-8"))
    fabric = physical_fabric_from_data(hardware)
    hbm_address_spaces = hbm_address_spaces_from_data(hardware)
    logical = logical_expand(build_ir0(spec)).entries[0].graph
    placement_context = PlacementContext.create(
        producer_pass="stage1a_state_cases",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces,
    )
    placed = place_ir0(logical, placement_context)
    manifest = placed.persistent_state_manifest
    if manifest is None:
        raise AssertionError("K1 placement must produce persistent state")
    kv_declarations = tuple(
        state
        for state in manifest.declarations
        if state.identity.kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
    )
    if len(kv_declarations) != 4 or any(
        state.shape != (2, 2, 4) or state.tensor_bytes != 32
        for state in kv_declarations
    ):
        raise AssertionError("K1 must contain four 32-byte KV states")
    kv_refs = {state.id for state in kv_declarations}
    state_manifest = PersistentStateManifest.create(
        address_spaces=manifest.address_spaces,
        declarations=kv_declarations,
        bindings=tuple(
            binding
            for binding in manifest.bindings
            if binding.state_ref in kv_refs
        ),
    )
    attention_nodes = tuple(
        sorted(
            (node for node in placed.nodes if node.kind is OpKind.ATTENTION),
            key=lambda node: (node.stage, node.id),
        )
    )
    if len(attention_nodes) != 2:
        raise AssertionError("K1 requires exactly two attention nodes")
    writes = tuple(
        StateAccess.create(
            node_ref=attention_nodes[0].id,
            state_ref=state.id,
            mode=StateAccessMode.WRITE,
            rank=0,
        )
        for state in kv_declarations
    )
    reads = tuple(
        StateAccess.create(
            node_ref=attention_nodes[1].id,
            state_ref=state.id,
            mode=StateAccessMode.READ,
            rank=0,
        )
        for state in kv_declarations
    )
    placed_fields = placed._semantic_key()
    placed_fields.update(
        state_accesses=tuple(
            sorted(
                (*writes, *reads),
                key=lambda item: (item.node_ref, item.state_ref, item.rank, item.id),
            )
        ),
        persistent_state_manifest=state_manifest,
    )
    stateful = IR1.create(producer_pass="placement", **placed_fields)
    stateful.validate("k1.placed")
    partition_fields = stateful._semantic_key()
    partition_fields["fused_op_skeletons"] = NaiveFusionPartition().run(stateful)
    graph = IR1.create(producer_pass="fusion_partition", **partition_fields)
    graph.validate("k1.graph")
    return graph, kv_declarations, writes, reads


def build_cross_action_kv_foundation_case() -> CrossActionKvFoundationCase:
    """Build K1: four exact KV payloads stored then reloaded across actions."""

    graph, declarations, writes, reads = _k1_graph()
    fusion_plans = tuple(
        NaiveInterDiePolicy().plan(graph, skeleton, graph.profile)
        for skeleton in graph.fused_op_skeletons
    )
    fused_members = {
        node_id
        for skeleton in graph.fused_op_skeletons
        for node_id in skeleton.member_node_ids
    }
    standalone_plans = tuple(
        DirectAllGatherPolicy().plan(graph, node, graph.profile)
        for node in graph.nodes
        if node.kind is OpKind.COLLECTIVE
        and node.id not in fused_members
        and getattr(node.workload, "collective", None)
        is CollectiveKind.ALL_GATHER
    )
    planned_semantic_key = {
        "source_partitioned_entry_id": "k1.partitioned",
        "source_ir1_id": graph.id,
        "planning_context_id": "k1.planning.context",
        "profile_id": graph.profile.stable_id(),
        "weight": 1.0,
        "graph_id": graph.id,
        "fusion_plans": fusion_plans,
        "standalone_plans": standalone_plans,
    }
    planned = InterDiePlannedProfile(
        id=stable_artifact_id(
            "inter_die_planned_profile",
            planned_semantic_key,
            schema_version=INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
        ),
        source_partitioned_entry_id="k1.partitioned",
        source_ir1_id=graph.id,
        planning_context_id="k1.planning.context",
        profile_id=graph.profile.stable_id(),
        weight=1.0,
        graph=graph,
        fusion_plans=fusion_plans,
        standalone_plans=standalone_plans,
    )
    planned.validate("k1.planned")

    projection_context = ProjectToIR2Context.create(
        producer_pass="stage1a_state_cases",
        state_transfers=(),
    )
    projection = NaiveProjectToIR2().run(
        graph,
        fusion_plans,
        standalone_plans,
        state_transfers=(),
    )
    projection.validate_against(graph, fusion_plans, standalone_plans)
    projected = ProjectedProfileIR2.create(
        source=planned,
        context=projection_context,
        projection=projection,
    )
    projected.validate_against(planned, projection_context)

    scheduling_context = IntraDieSchedulingContext.create(
        producer_pass="stage1a_state_cases",
        policy=production_registry().instantiate(
            RegistryKind.INTRA_DIE,
            "naive",
        ).selection,
    )
    schedule_set = NaiveIntraDiePolicy().schedule(projection, graph)
    schedule_set.validate_against(projection, graph)
    scheduled = ScheduledProfileIR2.create(
        source=projected,
        context=scheduling_context,
        schedule_set=schedule_set,
    )
    scheduled.validate_against(projected, scheduling_context)
    global_dag = build_global_action_dag(graph, projection, schedule_set)
    global_dag.validate_against(graph, projection, schedule_set)
    global_profile = GlobalActionProfile.create(
        source=scheduled,
        global_dag=global_dag,
    )
    global_profile.validate_against(scheduled)
    lowered = lower_profile(global_profile)
    linked_profile = link_profile(lowered)
    manifest = linked_profile.manifest

    staging_by_access = {
        staging.state_access_ref: staging.id
        for dag in projection.dags
        for staging in dag.state_staging_values
    }
    write_staging_ids = tuple(staging_by_access[item.id] for item in writes)
    read_staging_ids = tuple(staging_by_access[item.id] for item in reads)
    resolved_abis = _resolved_abis(linked_profile)

    def abi_id(value_id: str) -> str:
        matches = tuple(
            item for item in resolved_abis if item.abi.value_id == value_id
        )
        if len(matches) != 1:
            raise AssertionError(
                "K1 state staging must resolve to exactly one BufferABI"
            )
        return matches[0].abi.id

    state_payloads = tuple(
        (
            declaration.id,
            bytes([payload_byte]) * declaration.tensor_bytes,
        )
        for declaration, payload_byte in zip(declarations, _K1_PAYLOAD_BYTES)
    )
    payload_by_state = dict(state_payloads)
    write_seed_overrides = {
        abi_id(staging_id): payload_by_state[access.state_ref]
        for staging_id, access in zip(write_staging_ids, writes)
    }
    read_expected_overrides = {
        abi_id(staging_id): payload_by_state[access.state_ref]
        for staging_id, access in zip(read_staging_ids, reads)
    }
    program_io = build_timing_program_io(
        linked_profile,
        _K1_ARTIFACT_SHA256,
        sram_seed_overrides=write_seed_overrides,
        sram_expected_overrides=read_expected_overrides,
        state_seed_overrides={},
        state_expected_overrides=payload_by_state,
    )

    write_access_ids = tuple(item.id for item in writes)
    read_access_ids = tuple(item.id for item in reads)
    write_actions = tuple(
        action
        for action in global_dag.actions
        if action.task_kind is SemanticTaskKind.DMA_OUT
        and getattr(action.origin_ref, "state_access_ref", None)
        in write_access_ids
    )
    read_actions = tuple(
        action
        for action in global_dag.actions
        if action.task_kind is SemanticTaskKind.DMA_IN
        and getattr(action.origin_ref, "state_access_ref", None)
        in read_access_ids
    )
    write_end = max(
        action.core_order_index
        for action in write_actions
        if action.core_order_index is not None
    )
    read_start = min(
        action.core_order_index
        for action in read_actions
        if action.core_order_index is not None
    )
    middle_action_count = sum(
        action.core_order_index is not None
        and write_end < action.core_order_index < read_start
        for action in global_dag.actions
    )
    hbm_write_bytes = sum(action.bytes for action in write_actions)
    hbm_read_bytes = sum(action.bytes for action in read_actions)
    d2d_bytes = sum(flow.bytes for dag in projection.dags for flow in dag.flows)
    hbm_capacity_floor_cycles = (
        hbm_write_bytes + hbm_read_bytes + _K1_HBM_BYTES_PER_CYCLE - 1
    ) // _K1_HBM_BYTES_PER_CYCLE
    return CrossActionKvFoundationCase(
        graph=graph,
        projection=projection,
        schedule_set=schedule_set,
        global_dag=global_dag,
        lowering_context=global_profile.lowering_context(),
        manifest=manifest,
        linked_profile=linked_profile,
        program_io=program_io,
        state_payloads=state_payloads,
        write_access_ids=write_access_ids,
        read_access_ids=read_access_ids,
        write_staging_ids=write_staging_ids,
        read_staging_ids=read_staging_ids,
        hbm_write_bytes=hbm_write_bytes,
        hbm_read_bytes=hbm_read_bytes,
        d2d_bytes=d2d_bytes,
        middle_action_count=middle_action_count,
        hbm_capacity_floor_cycles=hbm_capacity_floor_cycles,
    )


def _pd1_spec() -> ExperimentSpec:
    base = _k1_spec()
    instance = replace(
        base.parallel.instances[0],
        tp=2,
        sp=True,
    )
    result = replace(
        base,
        model=replace(base.model, L=1),
        parallel=replace(base.parallel, instances=(instance,)),
    )
    result.validate("pd1.spec")
    return result


def _pd1_graph() -> tuple[
    IR1,
    tuple[StateTransferContract, ...],
    dict[tuple[int, StateKind], StateAccess],
]:
    spec = _pd1_spec()
    hardware = json.loads(_K1_HARDWARE.read_text(encoding="utf-8"))
    fabric = physical_fabric_from_data(hardware)
    logical = logical_expand(build_ir0(spec)).entries[0].graph
    placement_context = PlacementContext.create(
        producer_pass="stage1a_state_cases",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(hardware),
    )
    placed = place_ir0(logical, placement_context)
    manifest = placed.persistent_state_manifest
    if manifest is None:
        raise AssertionError("PD1 placement must produce persistent state")
    attention_nodes = tuple(
        node for node in placed.nodes if node.kind is OpKind.ATTENTION
    )
    if len(attention_nodes) != 1:
        raise AssertionError("PD1 requires exactly one ATTENTION anchor")
    attention = attention_nodes[0]
    declaration_index = {
        declaration.id: declaration for declaration in manifest.declarations
    }
    selected_accesses = tuple(
        access
        for access in placed.state_accesses
        if access.node_ref == attention.id
        and declaration_index[access.state_ref].identity.kind
        in (StateKind.KV_KEY, StateKind.KV_VALUE)
    )
    selected_state_refs = {access.state_ref for access in selected_accesses}
    exact_keys = {
        (0, StateKind.KV_KEY),
        (0, StateKind.KV_VALUE),
        (1, StateKind.KV_KEY),
        (1, StateKind.KV_VALUE),
    }
    actual_keys = {
        (access.rank, declaration_index[access.state_ref].identity.kind)
        for access in selected_accesses
    }
    if len(selected_accesses) != 4 or actual_keys != exact_keys:
        raise AssertionError("PD1 requires rank0/rank1 K/V state accesses")

    replacement_by_ref = {
        declaration.id: PersistentStateDecl.create(
            identity=declaration.identity,
            shape=(16,),
            dtype=declaration.dtype,
            layout=declaration.layout,
            lifetime=declaration.lifetime,
            access=declaration.access,
        )
        for declaration in manifest.declarations
        if declaration.id in selected_state_refs
    }
    declarations = tuple(
        sorted(replacement_by_ref.values(), key=lambda item: item.id)
    )
    bindings = tuple(
        sorted(
            (
                HbmBinding.create(
                    state_ref=replacement_by_ref[binding.state_ref].id,
                    die_id=binding.die_id,
                    address=binding.address,
                    size_bytes=replacement_by_ref[
                        binding.state_ref
                    ].tensor_bytes,
                )
                for binding in manifest.bindings
                if binding.state_ref in selected_state_refs
            ),
            key=lambda item: (item.die_id, item.address, item.state_ref),
        )
    )
    state_manifest = PersistentStateManifest.create(
        address_spaces=manifest.address_spaces,
        declarations=declarations,
        bindings=bindings,
    )
    rewritten_accesses = tuple(
        sorted(
            (
                StateAccess.create(
                    node_ref=access.node_ref,
                    state_ref=replacement_by_ref[access.state_ref].id,
                    mode=(
                        StateAccessMode.READ
                        if access.rank == 0
                        else StateAccessMode.WRITE
                    ),
                    rank=access.rank,
                )
                for access in selected_accesses
            ),
            key=lambda item: (
                item.node_ref,
                item.state_ref,
                item.rank,
                item.id,
            ),
        )
    )
    attention_value_ids = {*attention.inputs, *attention.outputs}
    attention_values = tuple(
        replace(
            value,
            producer=(attention.id if value.id in attention.outputs else None),
            consumers=((attention.id,) if value.id in attention.inputs else ()),
        )
        for value in placed.values
        if value.id in attention_value_ids
    )
    placed_fields = placed._semantic_key()
    placed_fields.update(
        instances=tuple(
            replace(instance, node_ids=(attention.id,))
            for instance in placed.instances
        ),
        nodes=(attention,),
        values=attention_values,
        edges=(),
        fusion_candidates=(),
        fused_op_skeletons=(),
        state_accesses=rewritten_accesses,
        persistent_state_manifest=state_manifest,
    )
    stateful = IR1.create(producer_pass="placement", **placed_fields)
    stateful.validate("pd1.placed")
    partition_fields = stateful._semantic_key()
    # PD1 isolates the persistent KV bridge.  Dense nodes intentionally stay
    # ordinary so parameter state does not enter this foundation accounting.
    partition_fields["fused_op_skeletons"] = ()
    graph = IR1.create(
        producer_pass="fusion_partition", **partition_fields
    )
    graph.validate("pd1.graph")

    declaration_by_id = {
        declaration.id: declaration for declaration in declarations
    }
    accesses = {
        (access.rank, declaration_by_id[access.state_ref].identity.kind): access
        for access in graph.state_accesses
    }
    group = next(
        item
        for item in graph.groups
        if item.id == attention.execution_group_ref
    )
    forward_route = next(
        route
        for route in group.embedding.routes
        if (route.source_rank, route.destination_rank) == (0, 1)
    )
    contracts = tuple(
        sorted(
            (
                StateTransferContract.create(
                    producer_pass="stage1a_state_cases",
                    source_ir1_id=graph.id,
                    source_state_access_ref=accesses[(0, kind)].id,
                    destination_state_access_ref=accesses[(1, kind)].id,
                    pair_route_ref=forward_route.id,
                )
                for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
            ),
            key=lambda item: (
                item.source_ir1_id,
                item.source_state_access_ref,
                item.destination_state_access_ref,
                item.pair_route_ref,
                item.id,
            ),
        )
    )
    for index, contract in enumerate(contracts):
        contract.validate_against(graph, f"pd1.contracts[{index}]")
    return graph, contracts, accesses


def build_pd1_case() -> Pd1FoundationCase:
    # Two exact 32-byte K/V transfers from TP rank 0 to TP rank 1.
    graph, contracts, access_by_rank_kind = _pd1_graph()
    fusion_plans = ()
    standalone_plans = tuple(
        DirectAllGatherPolicy().plan(graph, node, graph.profile)
        for node in graph.nodes
        if node.kind is OpKind.COLLECTIVE
        and getattr(node.workload, "collective", None)
        is CollectiveKind.ALL_GATHER
    )
    projection = NaiveProjectToIR2().run(
        graph,
        fusion_plans,
        standalone_plans,
        state_transfers=contracts,
    )
    projection.validate_against(graph, fusion_plans, standalone_plans)
    schedule_set = NaiveIntraDiePolicy().schedule(projection, graph)
    schedule_set.validate_against(projection, graph)
    global_dag = build_global_action_dag(graph, projection, schedule_set)
    global_dag.validate_against(graph, projection, schedule_set)
    profile_fields = {
        "source_scheduled_entry_id": "pd1.scheduled",
        "source_projected_entry_id": "pd1.projected",
        "source_planned_entry_id": "pd1.planned",
        "source_partitioned_entry_id": "pd1.partitioned",
        "source_ir1_id": graph.id,
        "projection_context_id": "pd1.projection.context",
        "scheduling_context_id": "pd1.scheduling.context",
        "profile_id": graph.profile.stable_id(),
        "weight": 1.0,
        "graph": graph,
        "fusion_plans": fusion_plans,
        "standalone_plans": standalone_plans,
        "projection": projection,
        "schedule_set": schedule_set,
        "global_dag": global_dag,
    }
    profile_semantic_key = {
        **{
            name: profile_fields[name]
            for name in (
                "source_scheduled_entry_id",
                "source_projected_entry_id",
                "source_planned_entry_id",
                "source_partitioned_entry_id",
                "source_ir1_id",
                "projection_context_id",
                "scheduling_context_id",
                "profile_id",
                "weight",
            )
        },
        "graph_id": graph.id,
        "fusion_plan_ids": (),
        "standalone_plan_ids": tuple(
            plan.id for plan in standalone_plans
        ),
        "projection_id": projection.id,
        "schedule_set_id": schedule_set.id,
        "global_dag_id": global_dag.id,
    }
    global_profile = GlobalActionProfile(
        id=stable_artifact_id(
            "global_action_profile",
            profile_semantic_key,
            schema_version=GLOBAL_ACTION_BUNDLE_SCHEMA_VERSION,
        ),
        **profile_fields,
    )
    global_profile.validate("pd1.global_profile")
    lowered = lower_profile(global_profile)
    lowered.validate_against(global_profile)
    linked_profile = link_profile(lowered)
    linked_profile.validate_against(lowered)
    manifest = linked_profile.manifest

    state_manifest = graph.persistent_state_manifest
    assert state_manifest is not None
    manifest_by_id = {
        declaration.id: declaration
        for declaration in state_manifest.declarations
    }
    source_payloads = tuple(
        (
            access_by_rank_kind[(0, kind)].state_ref,
            _PD1_SOURCE_BYTES[kind],
        )
        for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
    )
    destination_expected = tuple(
        (
            access_by_rank_kind[(1, kind)].state_ref,
            _PD1_SOURCE_BYTES[kind],
        )
        for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
    )
    if any(
        manifest_by_id[state_ref].tensor_bytes != len(payload)
        for state_ref, payload in (*source_payloads, *destination_expected)
    ):
        raise AssertionError("PD1 ProgramIo payload must cover whole state")
    program_io = build_timing_program_io(
        linked_profile,
        _PD1_ARTIFACT_SHA256,
        state_seed_overrides=dict(source_payloads),
        state_expected_overrides=dict(destination_expected),
    )

    endpoint_access_ids = {
        access_ref
        for contract in contracts
        for access_ref in (
            contract.source_state_access_ref,
            contract.destination_state_access_ref,
        )
    }
    contract_ids = {contract.id for contract in contracts}
    bridge_task_ids: set[str] = set()
    for dag in projection.dags:
        for task in dag.tasks:
            if (
                isinstance(task.origin_ref, StateIoOrigin)
                and task.origin_ref.state_access_ref in endpoint_access_ids
            ):
                bridge_task_ids.add(task.id)
                assert task.dma is not None
                bridge_task_ids.update(task.dma.access_task_refs)
            elif (
                isinstance(task.origin_ref, StateTransferOrigin)
                and task.origin_ref.state_transfer_ref in contract_ids
            ):
                bridge_task_ids.add(task.id)
    bridge_action_ids = tuple(
        action.id
        for action in global_dag.actions
        if action.source.task_id in bridge_task_ids
    )
    if len(bridge_action_ids) != 12:
        raise AssertionError("PD1 bridge must contain exactly 12 actions")
    access_index = {access.id: access for access in graph.state_accesses}
    d2d_bytes = sum(
        manifest_by_id[
            access_index[contract.source_state_access_ref].state_ref
        ].tensor_bytes
        for contract in contracts
    )
    if d2d_bytes != 64:
        raise AssertionError("PD1 unique logical D2D payload must be 64 bytes")
    return Pd1FoundationCase(
        graph=graph,
        contracts=contracts,
        projection=projection,
        schedule_set=schedule_set,
        global_dag=global_dag,
        lowering_context=global_profile.lowering_context(),
        manifest=manifest,
        linked_profile=linked_profile,
        program_io=program_io,
        source_payloads=source_payloads,
        destination_expected=destination_expected,
        source_access_ids=tuple(
            access_by_rank_kind[(0, kind)].id
            for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        ),
        destination_access_ids=tuple(
            access_by_rank_kind[(1, kind)].id
            for kind in (StateKind.KV_KEY, StateKind.KV_VALUE)
        ),
        bridge_action_ids=bridge_action_ids,
        d2d_bytes=d2d_bytes,
    )


__all__ = [
    "CrossActionKvFoundationCase",
    "ParameterFoundationCase",
    "Pd1FoundationCase",
    "build_cross_action_kv_foundation_case",
    "build_parameter_foundation_case",
    "build_pd1_case",
]
