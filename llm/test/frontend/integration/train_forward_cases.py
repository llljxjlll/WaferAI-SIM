"""Self-contained production builder for the tiny DP2 x TP2 train case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm.frontend.wafer_frontend.passes import (
    build_train_forward_ir0,
    build_train_forward_oracle,
    build_train_global_action,
    load_physical_fabric_and_hbm_address_spaces,
    link_train,
    lower_train,
    partition_train_forward,
    place_train_forward_ir0,
    plan_train_forward,
    project_train_forward,
    schedule_train_forward,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
)
from llm.frontend.wafer_frontend.schema.ir0 import IR0
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
    TrainFusionPartitionedIR1,
    TrainInterDiePlannedIR1,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    IntraDieSchedulingContext,
    ProjectToIR2Context,
    TrainProjectedIR2,
    TrainScheduledIR2,
)
from llm.frontend.wafer_frontend.schema.placed_ir1 import TrainPlacedIR1
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import from_data
from llm.frontend.wafer_frontend.schema.train_forward_oracle import (
    TrainForwardOracle,
)
from llm.frontend.wafer_frontend.schema.train_global_action import (
    TrainGlobalAction,
)
from llm.frontend.wafer_frontend.schema.train_n6 import (
    TrainLinkedProgram,
    TrainLoweredProgram,
)


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x2.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_PRODUCER = "train_forward_cases"


@dataclass(frozen=True, slots=True)
class TrainForwardRuntimeInputs:
    source_hardware_path: Path
    source_mapping_path: Path
    hardware_json: str
    mapping_text: str


@dataclass(frozen=True, slots=True)
class TrainForwardCase:
    spec: ExperimentSpec
    oracle: TrainForwardOracle
    logical_graph: IR0
    placement_context: PlacementContext
    placed: TrainPlacedIR1
    partition_context: FusionPartitionContext
    partitioned: TrainFusionPartitionedIR1
    planning_context: InterDiePlanningContext
    planned: TrainInterDiePlannedIR1
    projection_context: ProjectToIR2Context
    projected: TrainProjectedIR2
    scheduling_context: IntraDieSchedulingContext
    scheduled: TrainScheduledIR2
    global_action: TrainGlobalAction
    lowered: TrainLoweredProgram
    linked: TrainLinkedProgram
    runtime_inputs: TrainForwardRuntimeInputs


def _spec() -> ExperimentSpec:
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
            "L": 2,
            "dtype": "fp16",
            "tie_word_embeddings": False,
            "rms_norm_epsilon": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 64,
            "moe": None,
        },
        "hardware": {
            "ref": "notes/frontend/examples/hardware_2x2.json",
        },
        "workload": {
            "mode": "train",
            "infer": None,
            "train": {
                "global_batch": 4,
                "micro_batch": 1,
                "seq_len": 8,
                "backward": False,
                "optimizer": "none",
                "structure": {
                    "micro_batch_count": 2,
                    "pp_schedule": "gpipe",
                    "interleave_chunks": 1,
                    "recompute": "none",
                },
            },
        },
        "parallel": {
            "instances": [
                {
                    "id": "T0",
                    "role": "train",
                    "tp": 2,
                    "sp": True,
                    "replicas": 1,
                    "dp": 2,
                    "pp": 1,
                    "ep": 1,
                }
            ],
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
    return from_data(ExperimentSpec, raw, path="train_forward_case.spec")


def _runtime_inputs() -> TrainForwardRuntimeInputs:
    return TrainForwardRuntimeInputs(
        source_hardware_path=_HARDWARE,
        source_mapping_path=_MAPPING,
        hardware_json=_HARDWARE.read_text(encoding="utf-8"),
        mapping_text=_MAPPING.read_text(encoding="utf-8"),
    )


def build_train_forward_case() -> TrainForwardCase:
    """Build the one formal pre-link F-TF case without unit-test fixtures."""

    spec = _spec()
    oracle = build_train_forward_oracle(spec)
    logical_graph = build_train_forward_ir0(spec)
    oracle.validate_against_ir0(spec, logical_graph)
    runtime_inputs = _runtime_inputs()
    fabric, hbm_address_spaces = (
        load_physical_fabric_and_hbm_address_spaces(
            runtime_inputs.source_hardware_path,
            runtime_inputs.source_mapping_path,
        )
    )
    placement_context = PlacementContext.create(
        producer_pass=_PRODUCER,
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=hbm_address_spaces,
    )
    placed = place_train_forward_ir0(logical_graph, placement_context)
    partition_context = FusionPartitionContext.create(
        producer_pass=_PRODUCER,
    )
    partitioned = partition_train_forward(placed, partition_context)
    registry = production_registry()
    planning_context = InterDiePlanningContext.create(
        producer_pass=_PRODUCER,
        fused_policy=registry.instantiate(
            RegistryKind.INTER_DIE,
            "naive",
        ).selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE,
            "direct_all_gather",
        ).selection,
    )
    planned = plan_train_forward(partitioned, planning_context)
    projection_context = ProjectToIR2Context.create(
        producer_pass=_PRODUCER,
        state_transfers=(),
    )
    projected = project_train_forward(planned, projection_context)
    scheduling_context = IntraDieSchedulingContext.create(
        producer_pass=_PRODUCER,
        policy=registry.instantiate(
            RegistryKind.INTRA_DIE,
            "naive",
        ).selection,
    )
    scheduled = schedule_train_forward(projected, scheduling_context)
    global_action = build_train_global_action(scheduled)
    lowered = lower_train(global_action)
    lowered.validate_against(global_action)
    linked = link_train(lowered)
    linked.validate_against(lowered)
    return TrainForwardCase(
        spec=spec,
        oracle=oracle,
        logical_graph=logical_graph,
        placement_context=placement_context,
        placed=placed,
        partition_context=partition_context,
        partitioned=partitioned,
        planning_context=planning_context,
        planned=planned,
        projection_context=projection_context,
        projected=projected,
        scheduling_context=scheduling_context,
        scheduled=scheduled,
        global_action=global_action,
        lowered=lowered,
        linked=linked,
        runtime_inputs=runtime_inputs,
    )


__all__ = [
    "TrainForwardCase",
    "TrainForwardRuntimeInputs",
    "build_train_forward_case",
]
