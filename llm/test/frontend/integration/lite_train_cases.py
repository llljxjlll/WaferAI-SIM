"""Self-contained production source builder for the S2-Lite training case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm.frontend.wafer_frontend.passes import (
    build_s2_lite_train_global_action,
    build_s2_lite_lm_head_train_ir0,
    build_s2_lite_lm_head_train_oracle,
    load_physical_fabric_and_hbm_address_spaces,
    partition_train_forward,
    place_train_forward_ir0,
    plan_train_forward,
    project_train_forward,
    schedule_train_forward,
)
from llm.frontend.wafer_frontend.policies.registry import RegistryKind, production_registry
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.experiment import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentSpec,
    TrainOptimizer,
)
from llm.frontend.wafer_frontend.schema.lite_train import (
    S2_LITE_LM_HEAD_TRAIN_CASE_ID,
    S2LiteLmHeadTrainContract,
    S2LiteLmHeadTrainOracle,
    S2LiteTrainCoverage,
    S2LiteTrainStage,
)
from llm.frontend.wafer_frontend.schema.lite_train_graph import S2LiteLmHeadTrainIR0
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
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, from_data
from llm.frontend.wafer_frontend.schema.train_global_action import S2LiteTrainGlobalAction


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x1.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"


@dataclass(frozen=True, slots=True)
class S2LiteRuntimeInputs:
    source_hardware_path: Path
    source_mapping_path: Path
    hardware_json: str
    mapping_text: str


@dataclass(frozen=True, slots=True)
class S2LiteSourceCase:
    spec: ExperimentSpec
    contract: S2LiteLmHeadTrainContract
    oracle: S2LiteLmHeadTrainOracle
    logical: S2LiteLmHeadTrainIR0
    runtime_inputs: S2LiteRuntimeInputs

    def validate(self) -> None:
        self.spec.validate("s2_lite_case.spec")
        self.contract.validate("s2_lite_case.contract")
        self.oracle.validate_against_contract(
            self.contract, path="s2_lite_case.oracle"
        )
        self.logical.validate("s2_lite_case.logical")
        if self.contract.source_spec_digest != canonical_digest(self.spec):
            raise ValueError("S2-Lite contract is not bound to the source spec")
        if self.logical.contract != self.contract or self.logical.oracle != self.oracle:
            raise ValueError("S2-Lite logical carrier does not preserve its sources")
        rebuilt = build_s2_lite_lm_head_train_ir0(
            self.spec, self.contract, self.oracle
        )
        if rebuilt != self.logical:
            raise ValueError("S2-Lite logical carrier is not the production quotient")
        if not self.runtime_inputs.hardware_json or not self.runtime_inputs.mapping_text:
            raise ValueError("S2-Lite runtime inputs must be non-empty")


@dataclass(frozen=True, slots=True)
class S2LiteProductionCase:
    source: S2LiteSourceCase
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
    global_action: S2LiteTrainGlobalAction

    def validate(self) -> None:
        self.source.validate()
        self.placed.validate()
        self.partitioned.validate()
        self.planned.validate()
        self.projected.validate()
        self.scheduled.validate()
        self.global_action.validate_against(self.scheduled)
        if len(self.global_action.global_dags) != 1:
            raise ValueError("S2-Lite requires exactly one GlobalAction DAG")


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
        "hardware": {"ref": "notes/frontend/examples/hardware_2x1.json"},
        "workload": {
            "mode": "train",
            "infer": None,
            "train": {
                "global_batch": 1,
                "micro_batch": 1,
                "seq_len": 8,
                "backward": False,
                "optimizer": "none",
                "structure": {
                    "micro_batch_count": 1,
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
    return from_data(ExperimentSpec, raw, path="s2_lite_case.spec")


def build_s2_lite_source_case() -> S2LiteSourceCase:
    spec = _spec()
    contract = S2LiteLmHeadTrainContract.create(
        case_id=S2_LITE_LM_HEAD_TRAIN_CASE_ID,
        source_spec_digest=canonical_digest(spec),
        coverage=S2LiteTrainCoverage.LM_HEAD_ONLY,
        backbone_frozen=True,
        embedding_frozen=True,
        optimizer=TrainOptimizer.SGD,
        learning_rate=0.001,
        momentum=0.0,
        dp_degree=1,
        tp_degree=1,
        pp_degree=1,
        ep_degree=1,
        micro_batch_count=1,
        step_count=1,
        micro_batch_size=1,
        sequence_length=8,
        hidden_size=16,
        vocabulary_size=32,
        activation_dtype=DType.FP16,
        label_dtype=DType.INT32,
        loss_gradient_dtype=DType.FP32,
        weight_dtype=DType.FP16,
        weight_gradient_dtype=DType.FP32,
        stages=(
            S2LiteTrainStage.CE_BACKWARD,
            S2LiteTrainStage.LM_HEAD_WGRAD,
            S2LiteTrainStage.SGD_UPDATE,
        ),
    )
    oracle = build_s2_lite_lm_head_train_oracle(contract)
    logical = build_s2_lite_lm_head_train_ir0(spec, contract, oracle)
    result = S2LiteSourceCase(
        spec=spec,
        contract=contract,
        oracle=oracle,
        logical=logical,
        runtime_inputs=S2LiteRuntimeInputs(
            source_hardware_path=_HARDWARE,
            source_mapping_path=_MAPPING,
            hardware_json=_HARDWARE.read_text(encoding="utf-8"),
            mapping_text=_MAPPING.read_text(encoding="utf-8"),
        ),
    )
    result.validate()
    return result


def build_s2_lite_production_case() -> S2LiteProductionCase:
    source = build_s2_lite_source_case()
    fabric, hbm_address_spaces = load_physical_fabric_and_hbm_address_spaces(
        source.runtime_inputs.source_hardware_path,
        source.runtime_inputs.source_mapping_path,
    )
    placement_context = PlacementContext.create(
        producer_pass="lite_train_cases",
        fabric=fabric,
        placement=source.spec.placement,
        hbm_address_spaces=hbm_address_spaces,
    )
    placed = place_train_forward_ir0(source.logical.graph, placement_context)
    partition_context = FusionPartitionContext.create(
        producer_pass="lite_train_cases"
    )
    partitioned = partition_train_forward(placed, partition_context)
    registry = production_registry()
    planning_context = InterDiePlanningContext.create(
        producer_pass="lite_train_cases",
        fused_policy=registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
        ).selection,
    )
    planned = plan_train_forward(partitioned, planning_context)
    projection_context = ProjectToIR2Context.create(
        producer_pass="lite_train_cases", state_transfers=()
    )
    projected = project_train_forward(planned, projection_context)
    scheduling_context = IntraDieSchedulingContext.create(
        producer_pass="lite_train_cases",
        policy=registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection,
    )
    scheduled = schedule_train_forward(projected, scheduling_context)
    global_action = build_s2_lite_train_global_action(scheduled)
    result = S2LiteProductionCase(
        source=source,
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
    )
    result.validate()
    return result


__all__ = [
    "S2LiteProductionCase",
    "S2LiteRuntimeInputs",
    "S2LiteSourceCase",
    "build_s2_lite_production_case",
    "build_s2_lite_source_case",
]
