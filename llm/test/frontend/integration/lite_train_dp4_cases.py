"""Self-contained production case through the DP4 tree-AR GlobalAction boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from llm.test.frontend.integration.lite_train_cases import (
    S2LiteRuntimeInputs,
    S2LiteSourceCase,
    build_s2_lite_source_case,
)

from llm.frontend.wafer_frontend.passes.fusion_partition import partition_train_forward
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import schedule_train_forward
from llm.frontend.wafer_frontend.passes.lite_train_dp4 import (
    build_s2_lite_dp4_tree_ar_global_action,
    build_s2_lite_dp4_tree_ar_source,
)
from llm.frontend.wafer_frontend.passes.lite_train_dp4_link_program import (
    link_s2_lite_dp4_tree_ar,
)
from llm.frontend.wafer_frontend.passes.lite_train_dp4_lower_program import (
    lower_s2_lite_dp4_tree_ar,
)
from llm.frontend.wafer_frontend.passes.lite_train_dp4_n6 import (
    build_s2_lite_dp4_tree_ar_n6_intent,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    load_physical_fabric_and_hbm_address_spaces,
)
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_train_forward
from llm.frontend.wafer_frontend.policies.registry import RegistryKind, production_registry
from llm.frontend.wafer_frontend.schema.lite_train_dp4 import (
    S2_LITE_DP4_TREE_AR_CASE_ID,
    S2LiteDp4TreeArGlobalAction,
    S2LiteDp4TreeArSource,
)
from llm.frontend.wafer_frontend.schema.lite_train_dp4_n6 import (
    S2LiteDp4TreeArLinkedProgram,
    S2LiteDp4TreeArLoweredProgram,
    S2LiteDp4TreeArN6Intent,
)
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


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x2.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_PRODUCER = "lite_train_dp4_cases"


@dataclass(frozen=True, slots=True)
class S2LiteDp4TreeArCase:
    case_id: str
    source: S2LiteSourceCase
    dp4_source: S2LiteDp4TreeArSource
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
    global_action: S2LiteDp4TreeArGlobalAction
    n6_intent: S2LiteDp4TreeArN6Intent
    lowered: S2LiteDp4TreeArLoweredProgram
    linked: S2LiteDp4TreeArLinkedProgram

    @property
    def hardware_path(self) -> Path:
        return self.source.runtime_inputs.source_hardware_path

    @property
    def mapping_path(self) -> Path:
        return self.source.runtime_inputs.source_mapping_path

    def validate(self) -> None:
        if self.case_id != S2_LITE_DP4_TREE_AR_CASE_ID:
            raise ValueError("wrong S2-Lite DP4 tree-AR case id")
        self.source.validate()
        inputs = self.source.runtime_inputs
        if (
            inputs.source_hardware_path != _HARDWARE
            or inputs.source_mapping_path != _MAPPING
            or inputs.hardware_json != _HARDWARE.read_text(encoding="utf-8")
            or inputs.mapping_text != _MAPPING.read_text(encoding="utf-8")
        ):
            raise ValueError("DP4 runtime hardware/mapping inputs drifted")
        self.dp4_source.validate("dp4_case.dp4_source")
        if self.dp4_source.base != self.source.logical:
            raise ValueError("DP4 source does not preserve the production Lite graph")
        self.placement_context.validate("dp4_case.placement_context")
        self.placed.validate("dp4_case.placed")
        if (
            self.placed.source_ir0_id != self.dp4_source.graph.id
            or self.placed.placement_context_id != self.placement_context.id
        ):
            raise ValueError("DP4 placement provenance is not source-exact")
        self.partitioned.validate_against(self.placed, self.partition_context, "dp4_case.partitioned")
        self.planned.validate_against(self.partitioned, self.planning_context, "dp4_case.planned")
        self.projected.validate_against(self.planned, self.projection_context, "dp4_case.projected")
        self.scheduled.validate_against(self.projected, self.scheduling_context, "dp4_case.scheduled")
        self.global_action.validate("dp4_case.global_action")
        if self.global_action.source != self.dp4_source or self.global_action.scheduled != self.scheduled:
            raise ValueError("DP4 GlobalAction does not preserve source/schedule")
        self.n6_intent.validate("dp4_case.n6_intent")
        self.lowered.validate("dp4_case.lowered")
        self.linked.validate("dp4_case.linked")
        if (
            self.n6_intent.source != self.global_action
            or self.lowered.intent != self.n6_intent
            or self.linked.source != self.lowered
        ):
            raise ValueError("DP4 N6/link chain does not preserve exact provenance")


def build_s2_lite_dp4_tree_ar_case() -> S2LiteDp4TreeArCase:
    base = build_s2_lite_source_case()
    source = replace(
        base,
        runtime_inputs=S2LiteRuntimeInputs(
            source_hardware_path=_HARDWARE,
            source_mapping_path=_MAPPING,
            hardware_json=_HARDWARE.read_text(encoding="utf-8"),
            mapping_text=_MAPPING.read_text(encoding="utf-8"),
        ),
    )
    source.validate()
    dp4_source = build_s2_lite_dp4_tree_ar_source(source.logical)
    fabric, hbm_address_spaces = load_physical_fabric_and_hbm_address_spaces(_HARDWARE, _MAPPING)
    placement_context = PlacementContext.create(
        producer_pass=_PRODUCER,
        fabric=fabric,
        placement=source.spec.placement,
        hbm_address_spaces=hbm_address_spaces,
    )
    placed = place_train_forward_ir0(dp4_source.graph, placement_context)
    partition_context = FusionPartitionContext.create(producer_pass=_PRODUCER)
    partitioned = partition_train_forward(placed, partition_context)
    registry = production_registry()
    planning_context = InterDiePlanningContext.create(
        producer_pass=_PRODUCER,
        fused_policy=registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
        ).selection,
    )
    planned = plan_train_forward(partitioned, planning_context)
    projection_context = ProjectToIR2Context.create(producer_pass=_PRODUCER, state_transfers=())
    projected = project_train_forward(planned, projection_context)
    scheduling_context = IntraDieSchedulingContext.create(
        producer_pass=_PRODUCER,
        policy=registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection,
    )
    scheduled = schedule_train_forward(projected, scheduling_context)
    global_action = build_s2_lite_dp4_tree_ar_global_action(dp4_source, scheduled)
    n6_intent = build_s2_lite_dp4_tree_ar_n6_intent(global_action)
    lowered = lower_s2_lite_dp4_tree_ar(n6_intent)
    linked = link_s2_lite_dp4_tree_ar(lowered)
    result = S2LiteDp4TreeArCase(
        S2_LITE_DP4_TREE_AR_CASE_ID,
        source,
        dp4_source,
        placement_context,
        placed,
        partition_context,
        partitioned,
        planning_context,
        planned,
        projection_context,
        projected,
        scheduling_context,
        scheduled,
        global_action,
        n6_intent,
        lowered,
        linked,
    )
    result.validate()
    return result


__all__ = [
    "S2LiteDp4TreeArCase",
    "build_s2_lite_dp4_tree_ar_case",
]
