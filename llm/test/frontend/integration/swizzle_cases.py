"""Self-contained discovery-to-adapter Swizzle integration cases.

The Dense cases use the production ExperimentSpec, logical expansion and N3
placement path.  The synthetic AllReduce case is intentionally stopped out of
production Dense placement: it still uses a schema-valid IR0, production
fusion discovery, the production group/route builder, and an exact local IR1
placement adapter.  This keeps the Dense validator fail-closed while exposing
the first typed GEMM+AR planner integration case.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    build_ir0,
    hbm_address_spaces_from_data,
    logical_expand,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.discover_fusion import (
    discover_fusion_candidates,
    with_discovered_fusion_candidates,
)
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.group_registry import build_group_registry
from llm.frontend.wafer_frontend.passes.placement import place_ir0
from llm.frontend.wafer_frontend.policies.naive_fusion_partition import (
    NaiveFusionPartition,
)
from swizzle_forced import ForcedSwizzlePlanAdapter, materialize_forced_swizzle
from llm.frontend.wafer_frontend.policies.swizzle.wang_1d import (
    generate_wang_1d_drafts,
)
from llm.frontend.wafer_frontend.policies.swizzle_topo import (
    SwizzleFusionPartition,
    SwizzlePlanner,
)
from llm.frontend.wafer_frontend.schema.common import (
    DType,
    MeshAxisName,
    ProfileKey,
    Sharding,
    TensorValue,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    CollectiveRole,
    CollectiveWorkload,
    DeviceMesh,
    EdgeKind,
    EffectKind,
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
from llm.frontend.wafer_frontend.schema.ir1 import (
    IR1,
    PhysicalInstance,
    PhysicalNode,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import from_data
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleConstraints,
    SwizzleDecision,
    SwizzleEfficiencyPoint,
    SwizzleHardwareProfile,
)


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x1.json"


@dataclass(frozen=True, slots=True)
class SwizzleIntegrationCase:
    name: str
    pattern: FusionPattern
    synthetic_placement: bool
    source_graph: IR0
    placement_context: PlacementContext
    placed_graph: IR1
    partitioned_graph: IR1
    naive_fusion_refs: tuple[str, ...]
    swizzle_fusion_refs: tuple[str, ...]
    skeleton_ref: str
    decision: SwizzleDecision
    adapter: ForcedSwizzlePlanAdapter

    def validate(self, path: str = "swizzle_integration_case") -> None:
        if not self.name:
            raise SchemaError("name must be non-empty", path=f"{path}.name")
        if type(self.pattern) is not FusionPattern:
            raise SchemaError("must use a FusionPattern", path=f"{path}.pattern")
        if type(self.synthetic_placement) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.synthetic_placement")
        self.source_graph.validate(f"{path}.source_graph")
        self.placement_context.validate(f"{path}.placement_context")
        self.placed_graph.validate(f"{path}.placed_graph")
        self.partitioned_graph.validate(f"{path}.partitioned_graph")
        self.decision.validate(f"{path}.decision")
        self.adapter.validate(f"{path}.adapter")
        if self.placed_graph.source_ir0_id != self.source_graph.id:
            raise SchemaError("placement source provenance is not exact", path=path)
        skeleton = next(
            (
                item
                for item in self.partitioned_graph.fused_op_skeletons
                if item.id == self.skeleton_ref
            ),
            None,
        )
        if skeleton is None or skeleton.semantic_contract.pattern is not self.pattern:
            raise SchemaError("case skeleton/pattern is not exact", path=f"{path}.skeleton_ref")
        if skeleton.fusion_ref not in self.swizzle_fusion_refs:
            raise SchemaError("skeleton is absent from Swizzle selection", path=path)
        if self.decision.problem.source_ir1_id != self.partitioned_graph.id:
            raise SchemaError("decision references a different IR1", path=f"{path}.decision")
        if self.decision.problem.fused_op_id != skeleton.id:
            raise SchemaError("decision references a different skeleton", path=f"{path}.decision")
        if self.adapter.decision != self.decision or self.adapter.pattern is not self.pattern:
            raise SchemaError("adapter lost decision/pattern provenance", path=f"{path}.adapter")
        if self.adapter.candidate.algorithm is SwizzleAlgorithm.UNFUSED:
            raise SchemaError("integration adapter must materialize a fused candidate", path=f"{path}.adapter")


def _dense_spec() -> ExperimentSpec:
    raw = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "model": {
            "source": "analytic",
            "arch": "llama",
            "V": 32,
            "H": 16,
            "I": 32,
            "NH": 4,
            "KVH": 4,
            "DH": 4,
            "rotary_dim": 4,
            "L": 1,
            "dtype": "fp16",
            "tie_word_embeddings": False,
            "rms_norm_epsilon": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
            "moe": None,
        },
        "hardware": {"ref": "notes/frontend/examples/hardware_2x1.json"},
        "workload": {
            "mode": "infer",
            "infer": {
                "source": "static_profile",
                "output": "logits",
                "profile": {
                    "prefill_tokens": 8,
                    "decode_tokens": 0,
                    "num_seqs": 1,
                    "context_sum": 8,
                    "context_max": 8,
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
                    "tp": 2,
                    "sp": True,
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
    return from_data(ExperimentSpec, raw, path="swizzle_dense.spec")


def _hardware() -> tuple[dict[str, object], object]:
    raw = json.loads(_HARDWARE.read_text(encoding="utf-8"))
    return raw, physical_fabric_from_data(raw, path="swizzle.hardware")


def _placement_context(spec: ExperimentSpec) -> PlacementContext:
    raw, fabric = _hardware()
    return PlacementContext.create(
        producer_pass="swizzle_cases",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(raw, path="swizzle.hardware"),
    )


def _profile() -> SwizzleHardwareProfile:
    return SwizzleHardwareProfile.create(
        peak_flops_per_cycle=1024.0,
        confidence_fraction=0.05,
        efficiency_points=(SwizzleEfficiencyPoint(4, 4, 4, 0.9),),
        dte_launch_cycles=2,
        dte_sync_cycles=1,
        hop_latency_cycles=1,
        lane_bytes_per_cycle=32.0,
        max_inflight_dte=4,
        min_transfer_bytes=16,
        efficient_tile_floor=(1, 1, 1),
        sram_budget_bytes=1 << 20,
        double_buffer_supported=True,
    )


def _planner() -> SwizzlePlanner:
    constraints = SwizzleConstraints(
        allowed_algorithms=(
            SwizzleAlgorithm.UNFUSED,
            SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL,
        ),
        max_candidates=32,
        max_actions=4096,
        max_buffers=256,
        max_chunk_count=2,
        allow_unroll_two=True,
    )
    return SwizzlePlanner(
        hardware_profile=_profile(),
        constraints=constraints,
        generators=(generate_wang_1d_drafts,),
    )


def _partition_selections(placed: IR1) -> tuple[tuple[str, ...], IR1]:
    naive = NaiveFusionPartition().run(placed)
    partitioned = partition_ir1(placed, policy=SwizzleFusionPartition())
    return (
        tuple(item.fusion_ref for item in naive),
        partitioned,
    )


def _case(
    *,
    name: str,
    pattern: FusionPattern,
    synthetic: bool,
    source: IR0,
    context: PlacementContext,
    placed: IR1,
    partitioned: IR1,
    naive_refs: tuple[str, ...],
) -> SwizzleIntegrationCase:
    skeleton = next(
        item
        for item in partitioned.fused_op_skeletons
        if item.semantic_contract.pattern is pattern
    )
    decision = _planner().decide(partitioned, skeleton)
    adapter = materialize_forced_swizzle(decision)
    result = SwizzleIntegrationCase(
        name=name,
        pattern=pattern,
        synthetic_placement=synthetic,
        source_graph=source,
        placement_context=context,
        placed_graph=placed,
        partitioned_graph=partitioned,
        naive_fusion_refs=naive_refs,
        swizzle_fusion_refs=tuple(
            item.fusion_ref for item in partitioned.fused_op_skeletons
        ),
        skeleton_ref=skeleton.id,
        decision=decision,
        adapter=adapter,
    )
    result.validate()
    return result


def build_dense_tp_swizzle_cases() -> tuple[SwizzleIntegrationCase, SwizzleIntegrationCase]:
    """Build production Dense TP AG+GEMM and GEMM+RS from one graph."""

    spec = _dense_spec()
    source = logical_expand(build_ir0(spec)).entries[0].graph
    if source.fusion_candidates != discover_fusion_candidates(source):
        raise SchemaError("logical expansion lost canonical discovery", path="dense.fusion_candidates")
    context = _placement_context(spec)
    placed = place_ir0(source, context)
    naive_refs, partitioned = _partition_selections(placed)
    return (
        _case(
            name="production_dense_tp_ag_gemm",
            pattern=FusionPattern.AG_GEMM,
            synthetic=False,
            source=source,
            context=context,
            placed=placed,
            partitioned=partitioned,
            naive_refs=naive_refs,
        ),
        _case(
            name="production_dense_tp_gemm_rs",
            pattern=FusionPattern.GEMM_RS,
            synthetic=False,
            source=source,
            context=context,
            placed=placed,
            partitioned=partitioned,
            naive_refs=naive_refs,
        ),
    )


def _synthetic_ar_source() -> IR0:
    mesh = DeviceMesh("synthetic.tp", (MeshAxis(MeshAxisName.TP, 2),))
    instance = LogicalInstance(
        id="SYN0",
        role=LogicalRole.PREFILL,
        replicas=1,
        parallel=ParallelAxes(tp=2, sp=False, dp=1, pp=1, ep=1),
        meshes=(mesh,),
    )
    gemm = LogicalNode(
        id="synthetic.gemm",
        instance_id=instance.id,
        kind=OpKind.GEMM,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref=mesh.id,
        inputs=("synthetic.lhs", "synthetic.rhs"),
        outputs=("synthetic.partial",),
        workload=GemmWorkload(
            logical_shape=(8, 8, 8),
            rank_shape=(8, 8, 4),
            partition=GemmPartition.ROW_PARALLEL,
            dtype=DType.FP16,
        ),
        math=NodeMath(DType.FP32, NumericalPolicy.BITWISE),
        effects=NodeEffects(EffectKind.PURE, None, None),
        impl_ref="matmul_forward",
    )
    all_reduce = LogicalNode(
        id="synthetic.ar",
        instance_id=instance.id,
        kind=OpKind.COLLECTIVE,
        phase=OpPhase.FWD,
        stage=0,
        mesh_ref=mesh.id,
        inputs=("synthetic.partial",),
        outputs=("synthetic.output",),
        workload=CollectiveWorkload(
            collective=CollectiveKind.ALL_REDUCE,
            reduce_op=ReduceOp.SUM,
            mesh_axes=(MeshAxisName.TP,),
            participant_count=2,
            reduction_mesh_axes=(MeshAxisName.TP,),
            scatter_tensor_axis=None,
            gather_tensor_axis=None,
            logical_tensor_bytes=128,
            rank_input_bytes=128,
            rank_output_bytes=128,
            rank_logical_payload_bytes=128,
            group_logical_payload_bytes=256,
            dtype=DType.FP16,
            role=CollectiveRole.ACTIVATION,
            input_layout="MN_partial_tp",
            output_layout="MN_replicated",
        ),
        math=NodeMath(DType.FP32, NumericalPolicy.BITWISE),
        effects=NodeEffects(EffectKind.PURE, None, None),
        impl_ref="collective_derived",
    )
    values = (
        TensorValue(
            "synthetic.lhs", (8, 8), DType.FP16, "MK_shard_tp",
            Sharding(mesh.id, (None, MeshAxisName.TP), ()), None, (gemm.id,), None,
        ),
        TensorValue(
            "synthetic.rhs", (8, 8), DType.FP16, "KN_shard_tp",
            Sharding(mesh.id, (MeshAxisName.TP, None), ()), None, (gemm.id,), None,
        ),
        TensorValue(
            "synthetic.partial", (8, 8), DType.FP16, "MN_partial_tp",
            Sharding(mesh.id, (None, None), (MeshAxisName.TP,)), gemm.id, (all_reduce.id,), None,
        ),
        TensorValue(
            "synthetic.output", (8, 8), DType.FP16, "MN_replicated",
            Sharding(mesh.id, (None, None), ()), all_reduce.id, (), None,
        ),
    )
    graph = IR0.create(
        producer_pass="synthetic_swizzle_ar_source",
        job=JobKind.INFER,
        instances=(instance,),
        nodes=(gemm, all_reduce),
        values=values,
        edges=(GraphEdge("synthetic.edge", EdgeKind.DATA, gemm.id, all_reduce.id, "synthetic.partial"),),
        fusion_candidates=(),
        profile=ProfileKey(8, 0, 1, 8, 8, 0, None),
    )
    graph.validate("synthetic_ar_source")
    discovered = with_discovered_fusion_candidates(graph)
    if (
        len(discovered.fusion_candidates) != 1
        or discovered.fusion_candidates[0].semantic_contract.pattern
        is not FusionPattern.GEMM_AR
    ):
        raise SchemaError("synthetic AR discovery did not close", path="synthetic_ar")
    return discovered


def _place_synthetic_ar(source: IR0, context: PlacementContext) -> IR1:
    """Exact typed placement without weakening the production Dense validator."""

    groups = build_group_registry(source, context)
    group_by_owner = {(item.instance_id, item.mesh_ref): item for item in groups}
    nodes = tuple(
        PhysicalNode(
            id=node.id,
            origin_node_id=node.id,
            instance_id=node.instance_id,
            kind=node.kind,
            phase=node.phase,
            stage=node.stage,
            mesh_ref=node.mesh_ref,
            execution_group_ref=group_by_owner[(node.instance_id, node.mesh_ref)].id,
            inputs=node.inputs,
            outputs=node.outputs,
            workload=node.workload,
            math=node.math,
            effects=node.effects,
            impl_ref=node.impl_ref,
        )
        for node in source.nodes
    )
    instances = tuple(
        PhysicalInstance(
            id=instance.id,
            origin_instance_id=instance.id,
            role=instance.role,
            die_region=tuple(
                placement.die_id
                for group in groups
                if group.instance_id == instance.id
                for placement in group.placements
            ),
            group_ids=tuple(group.id for group in groups if group.instance_id == instance.id),
            node_ids=tuple(node.id for node in nodes if node.instance_id == instance.id),
        )
        for instance in source.instances
    )
    result = IR1.create(
        producer_pass="synthetic_swizzle_placement",
        source_ir0_id=source.id,
        profile=source.profile,
        fabric=context.fabric,
        instances=instances,
        groups=groups,
        nodes=nodes,
        values=source.values,
        edges=source.edges,
        fusion_candidates=source.fusion_candidates,
    )
    result.validate("synthetic_swizzle_placement")
    return result


def _partition_synthetic(placed: IR1) -> tuple[tuple[str, ...], IR1]:
    naive = NaiveFusionPartition().run(placed)
    selected = SwizzleFusionPartition().run(placed)
    result = IR1.create(
        producer_pass="synthetic_swizzle_fusion_partition",
        source_ir0_id=placed.source_ir0_id,
        profile=placed.profile,
        fabric=placed.fabric,
        instances=placed.instances,
        groups=placed.groups,
        nodes=placed.nodes,
        values=placed.values,
        edges=placed.edges,
        fusion_candidates=placed.fusion_candidates,
        fused_op_skeletons=selected,
    )
    result.validate("synthetic_swizzle_fusion_partition")
    return tuple(item.fusion_ref for item in naive), result


def build_synthetic_gemm_ar_swizzle_case() -> SwizzleIntegrationCase:
    source = _synthetic_ar_source()
    spec = _dense_spec()
    context = _placement_context(spec)
    placed = _place_synthetic_ar(source, context)
    naive_refs, partitioned = _partition_synthetic(placed)
    return _case(
        name="synthetic_gemm_ar",
        pattern=FusionPattern.GEMM_AR,
        synthetic=True,
        source=source,
        context=context,
        placed=placed,
        partitioned=partitioned,
        naive_refs=naive_refs,
    )


def build_swizzle_integration_cases() -> tuple[SwizzleIntegrationCase, ...]:
    dense_ag, dense_rs = build_dense_tp_swizzle_cases()
    return dense_ag, dense_rs, build_synthetic_gemm_ar_swizzle_case()


__all__ = [
    "SwizzleIntegrationCase",
    "build_dense_tp_swizzle_cases",
    "build_synthetic_gemm_ar_swizzle_case",
    "build_swizzle_integration_cases",
]
