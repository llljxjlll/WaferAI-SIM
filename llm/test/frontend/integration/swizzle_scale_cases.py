"""Production ExperimentSpec-to-IR1 Swizzle scale cases for W1/W2."""

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
)
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.placement import place_ir0
from llm.frontend.wafer_frontend.policies.swizzle_topo import (
    SwizzleFusionPartition,
    SwizzlePlanner,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
)
from llm.frontend.wafer_frontend.schema.ir0 import (
    FusionPattern,
    GemmWorkload,
    CollectiveWorkload,
    IR0,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import from_data
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleConstraints,
    SwizzleDecision,
    SwizzleEfficiencyPoint,
    SwizzleHardwareProfile,
    SwizzleTopologyKind,
)
from llm.frontend.wafer_frontend.schema.swizzle_scale import SwizzleScalePoint


_ROOT = Path(__file__).resolve().parents[4]
_PATTERN_ORDER = (FusionPattern.AG_GEMM, FusionPattern.GEMM_RS)


@dataclass(frozen=True, slots=True)
class SwizzleScaleCase:
    point: SwizzleScalePoint
    spec: ExperimentSpec
    hardware_path: Path
    source_graph: IR0
    placement_context: PlacementContext
    placed_graph: IR1
    partitioned_graph: IR1
    decisions: tuple[SwizzleDecision, ...]

    def validate(self, path: str = "swizzle_scale_case") -> None:
        self.point.validate_against(
            self.spec,
            self.source_graph,
            self.placed_graph,
            self.partitioned_graph,
            f"{path}.point",
        )
        if not self.hardware_path.is_file():
            raise SchemaError(
                "hardware path must identify a real production input",
                path=f"{path}.hardware_path",
            )
        if self.hardware_path != (_ROOT / self.spec.hardware.ref).resolve():
            raise SchemaError(
                "hardware path must come from ExperimentSpec.hardware.ref",
                path=f"{path}.hardware_path",
            )
        expected_source = logical_expand(build_ir0(self.spec)).entries[0].graph
        if self.source_graph != expected_source:
            raise SchemaError(
                "IR0 is not the exact ExperimentSpec producer result",
                path=f"{path}.source_graph",
            )
        if self.source_graph.fusion_candidates != discover_fusion_candidates(
            self.source_graph
        ):
            raise SchemaError(
                "IR0 fusion discovery is not canonical",
                path=f"{path}.source_graph.fusion_candidates",
            )
        if place_ir0(self.source_graph, self.placement_context) != self.placed_graph:
            raise SchemaError(
                "placed IR1 is not the exact production result",
                path=f"{path}.placed_graph",
            )
        expected_partitioned = partition_ir1(
            self.placed_graph, policy=SwizzleFusionPartition()
        )
        if self.partitioned_graph != expected_partitioned:
            raise SchemaError(
                "partitioned IR1 is not the exact production result",
                path=f"{path}.partitioned_graph",
            )
        if len(self.decisions) != len(_PATTERN_ORDER):
            raise SchemaError(
                "case must carry one canonical AG+GEMM and GEMM+RS decision",
                path=f"{path}.decisions",
            )
        for index, (pattern, decision) in enumerate(
            zip(_PATTERN_ORDER, self.decisions)
        ):
            decision_path = f"{path}.decisions[{index}]"
            decision.validate(decision_path)
            if (
                decision.problem.pattern is not pattern
                or decision.problem.source_ir1_id != self.partitioned_graph.id
            ):
                raise SchemaError(
                    "decision problem provenance/pattern drifted",
                    path=decision_path,
                )
            node_index = {node.id: node for node in self.partitioned_graph.nodes}
            gemm_node = node_index.get(decision.problem.gemm.node_ref)
            collective_node = node_index.get(decision.problem.collective.node_ref)
            if (
                gemm_node is None
                or type(gemm_node.workload) is not GemmWorkload
                or collective_node is None
                or type(collective_node.workload) is not CollectiveWorkload
            ):
                raise SchemaError(
                    "problem must resolve to typed IR1 GEMM/collective members",
                    path=decision_path,
                )
            gemm_workload = gemm_node.workload
            collective_workload = collective_node.workload
            if (
                decision.problem.gemm.m,
                decision.problem.gemm.n,
                decision.problem.gemm.k,
            ) != gemm_workload.logical_shape:
                raise SchemaError(
                    "problem GEMM shape must come from logical IR1 workload",
                    path=f"{decision_path}.problem.gemm",
                )
            expected_flops = (
                2
                * gemm_workload.logical_shape[0]
                * gemm_workload.logical_shape[1]
                * gemm_workload.logical_shape[2]
            )
            if (
                decision.problem.gemm.flops != expected_flops
                or decision.problem.gemm.dtype is not self.point.dtype
            ):
                raise SchemaError(
                    "problem GEMM FLOPs/dtype drifted from IR1",
                    path=f"{decision_path}.problem.gemm",
                )
            if (
                decision.problem.collective.logical_bytes,
                decision.problem.collective.rank_input_bytes,
                decision.problem.collective.rank_output_bytes,
            ) != (
                collective_workload.logical_tensor_bytes,
                collective_workload.rank_input_bytes,
                collective_workload.rank_output_bytes,
            ):
                raise SchemaError(
                    "problem collective bytes drifted from IR1",
                    path=f"{decision_path}.problem.collective",
                )
            if decision.problem.collective.participant_ranks != tuple(
                range(self.point.tp)
            ):
                raise SchemaError(
                    "collective must cover every physical TP rank",
                    path=f"{decision_path}.problem.collective.participant_ranks",
                )
            fused = tuple(
                candidate
                for candidate in decision.ranked_candidates
                if candidate.algorithm is not SwizzleAlgorithm.UNFUSED
            )
            if not fused:
                raise SchemaError(
                    "scale problem must discover at least one fused candidate",
                    path=f"{decision_path}.ranked_candidates",
                )
            if any(
                candidate.algorithm is not SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL
                for candidate in fused
            ):
                raise SchemaError(
                    "W1/W2 admits only currently production-ready Wang candidates",
                    path=f"{decision_path}.ranked_candidates",
                )
            for candidate in fused:
                split_axis = candidate.split_axis
                if (
                    split_axis is None
                    or split_axis.extent % candidate.chunk_count
                    or len(candidate.rank_programs) != self.point.tp
                ):
                    raise SchemaError(
                        "candidate decomposition/rank coverage is not exact",
                        path=f"{decision_path}.ranked_candidates",
                    )
            if self.point.tp == 4:
                kinds = {candidate.topology_witness.kind for candidate in fused}
                if kinds != {
                    SwizzleTopologyKind.BIDIRECTIONAL_LINE,
                    SwizzleTopologyKind.HAMILTONIAN_RING,
                }:
                    raise SchemaError(
                        "2x2 Wang discovery requires exact line and real ring witnesses",
                        path=f"{decision_path}.ranked_candidates",
                    )
                ring_orders = {
                    candidate.topology_witness.rank_order
                    for candidate in fused
                    if candidate.topology_witness.kind
                    is SwizzleTopologyKind.HAMILTONIAN_RING
                }
                if ring_orders != {(0, 1, 3, 2)}:
                    raise SchemaError(
                        "2x2 Wang ring must use the physical snake cycle",
                        path=f"{decision_path}.ranked_candidates",
                    )


def build_swizzle_scale_points() -> tuple[SwizzleScalePoint, ...]:
    """Return the complete immutable S0-S4 search matrix."""

    return tuple(
        SwizzleScalePoint.create(
            name=name,
            tokens=tokens,
            hidden_size=hidden,
            intermediate_size=intermediate,
            tp=tp,
            mesh_rows=rows,
            mesh_columns=columns,
            dtype=DType.FP16,
        )
        for name, tokens, hidden, intermediate, tp, rows, columns in (
            ("S0", 8, 16, 32, 2, 1, 2),
            ("S1", 32, 64, 256, 4, 2, 2),
            ("S2", 64, 64, 256, 4, 2, 2),
            ("S3", 128, 64, 256, 4, 2, 2),
            ("S4", 256, 512, 2048, 4, 2, 2),
        )
    )


def _spec(point: SwizzleScalePoint) -> ExperimentSpec:
    heads = max(4, point.hidden_size // 16)
    head_dim = point.hidden_size // heads
    hardware_ref = (
        "notes/frontend/examples/hardware_2x1.json"
        if point.tp == 2
        else "notes/frontend/examples/hardware_2x2.json"
    )
    raw = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "model": {
            "source": "analytic",
            "arch": "llama",
            "V": 32,
            "H": point.hidden_size,
            "I": point.intermediate_size,
            "NH": heads,
            "KVH": heads,
            "DH": head_dim,
            "rotary_dim": head_dim,
            "L": 1,
            "dtype": point.dtype.value,
            "tie_word_embeddings": False,
            "rms_norm_epsilon": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": max(128, point.tokens),
            "moe": None,
        },
        "hardware": {"ref": hardware_ref},
        "workload": {
            "mode": "infer",
            "infer": {
                "source": "static_profile",
                "output": "logits",
                "profile": {
                    "prefill_tokens": point.tokens,
                    "decode_tokens": 0,
                    "num_seqs": 1,
                    "context_sum": point.tokens,
                    "context_max": point.tokens,
                    "kv_pages": max(1, (point.tokens + 15) // 16),
                    "expert_load": None,
                },
            },
        },
        "parallel": {
            "instances": [
                {
                    "id": "P0",
                    "role": "prefill",
                    "tp": point.tp,
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
    return from_data(ExperimentSpec, raw, path=f"swizzle_scale.{point.name}.spec")


def _placement_context(spec: ExperimentSpec) -> tuple[Path, PlacementContext]:
    hardware_path = (_ROOT / spec.hardware.ref).resolve()
    data = json.loads(hardware_path.read_text(encoding="utf-8"))
    context = PlacementContext.create(
        producer_pass="swizzle_scale_cases",
        fabric=physical_fabric_from_data(
            data, path=f"swizzle_scale.{hardware_path.name}"
        ),
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(
            data, path=f"swizzle_scale.{hardware_path.name}"
        ),
    )
    return hardware_path, context


def _profile(point: SwizzleScalePoint) -> SwizzleHardwareProfile:
    """Keep S0 exact and apply the W7 calibration floor only to scale probes."""

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
        efficient_tile_floor=(
            (1, 1, 1) if point.name == "S0" else (4, 16, 16)
        ),
        sram_budget_bytes=1 << 20,
        double_buffer_supported=True,
    )


def _planner(point: SwizzleScalePoint) -> SwizzlePlanner:
    return SwizzlePlanner(
        hardware_profile=_profile(point),
        constraints=SwizzleConstraints(
            allowed_algorithms=(
                SwizzleAlgorithm.MESHSLICE_2D_OS,
                SwizzleAlgorithm.UNFUSED,
                SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL,
            ),
            max_candidates=64,
            max_actions=4096,
            max_buffers=256,
            max_chunk_count=point.tp if point.name == "S0" else 32,
            allow_unroll_two=True,
        ),
    )


def build_swizzle_scale_case(point: SwizzleScalePoint) -> SwizzleScaleCase:
    """Build one scale only through formal frontend producers."""

    point.validate()
    spec = _spec(point)
    source = logical_expand(build_ir0(spec)).entries[0].graph
    hardware_path, context = _placement_context(spec)
    placed = place_ir0(source, context)
    partitioned = partition_ir1(placed, policy=SwizzleFusionPartition())
    planner = _planner(point)
    decisions = tuple(
        planner.decide(
            partitioned,
            next(
                skeleton
                for skeleton in partitioned.fused_op_skeletons
                if skeleton.semantic_contract.pattern is pattern
            ),
        )
        for pattern in _PATTERN_ORDER
    )
    result = SwizzleScaleCase(
        point=point,
        spec=spec,
        hardware_path=hardware_path,
        source_graph=source,
        placement_context=context,
        placed_graph=placed,
        partitioned_graph=partitioned,
        decisions=decisions,
    )
    result.validate()
    return result


def build_first_green_swizzle_scale_cases() -> tuple[SwizzleScaleCase, ...]:
    """Build S0-S3 token scaling; S4 retains the first weight-capacity gate."""

    return tuple(
        build_swizzle_scale_case(point)
        for point in build_swizzle_scale_points()
        if point.name != "S4"
    )


__all__ = [
    "SwizzleScaleCase",
    "build_first_green_swizzle_scale_cases",
    "build_swizzle_scale_case",
    "build_swizzle_scale_points",
]
