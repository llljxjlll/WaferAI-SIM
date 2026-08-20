"""Self-contained production case for S2-Lite DP2 rooted all-reduce."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm.test.frontend.integration.lite_train_cases import (
    S2LiteSourceCase,
    build_s2_lite_source_case,
)

from llm.frontend.wafer_frontend.passes.fusion_partition import (
    partition_train_forward,
)
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import (
    schedule_train_forward,
)
from llm.frontend.wafer_frontend.passes.lite_train_dp2 import (
    build_s2_lite_dp2_rooted_ar_global_action,
    build_s2_lite_dp2_rooted_ar_source,
)
from llm.frontend.wafer_frontend.passes.lite_train_rooted_ar_n6 import (
    build_s2_lite_rooted_ar_n6_intent,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    load_physical_fabric_and_hbm_address_spaces,
)
from llm.frontend.wafer_frontend.passes.placement import (
    place_train_forward_ir0,
)
from llm.frontend.wafer_frontend.passes.project_to_ir2 import (
    project_train_forward,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.schema.lite_train_dp2 import (
    S2LiteDp2RootedArGlobalAction,
    S2LiteDp2RootedArSource,
)
from llm.frontend.wafer_frontend.schema.lite_train_rooted_ar_n6 import (
    S2LiteRootedArN6Intent,
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


S2_LITE_DP2_ROOTED_AR_CASE_ID = (
    "case.s2_lite.dp2_tp1.rooted_allreduce"
)

_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "notes/frontend/examples/hardware_2x1.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"
_PRODUCER = "lite_train_rooted_ar_cases"


@dataclass(frozen=True, slots=True)
class S2LiteDp2RootedArCase:
    case_id: str
    source: S2LiteSourceCase
    rooted_source: S2LiteDp2RootedArSource
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
    global_action: S2LiteDp2RootedArGlobalAction
    n6_intent: S2LiteRootedArN6Intent

    def validate(self) -> None:
        if self.case_id != S2_LITE_DP2_ROOTED_AR_CASE_ID:
            raise ValueError("wrong S2-Lite DP2 rooted-AR case id")
        self.source.validate()
        if (
            self.source.runtime_inputs.source_hardware_path != _HARDWARE
            or self.source.runtime_inputs.source_mapping_path != _MAPPING
            or self.source.runtime_inputs.hardware_json
            != _HARDWARE.read_text(encoding="utf-8")
            or self.source.runtime_inputs.mapping_text
            != _MAPPING.read_text(encoding="utf-8")
        ):
            raise ValueError("rooted-AR runtime hardware/mapping inputs drifted")
        self.rooted_source.validate("rooted_ar_case.rooted_source")
        if self.rooted_source.base != self.source.logical:
            raise ValueError("rooted-AR source does not preserve the production Lite graph")

        self.placement_context.validate("rooted_ar_case.placement_context")
        self.placed.validate("rooted_ar_case.placed")
        if (
            self.placed.source_ir0_id != self.rooted_source.graph.id
            or self.placed.placement_context_id != self.placement_context.id
        ):
            raise ValueError("placement provenance is not source-exact")
        self.partitioned.validate_against(
            self.placed,
            self.partition_context,
            "rooted_ar_case.partitioned",
        )
        self.planned.validate_against(
            self.partitioned,
            self.planning_context,
            "rooted_ar_case.planned",
        )
        self.projected.validate_against(
            self.planned,
            self.projection_context,
            "rooted_ar_case.projected",
        )
        self.scheduled.validate_against(
            self.projected,
            self.scheduling_context,
            "rooted_ar_case.scheduled",
        )
        self.global_action.validate("rooted_ar_case.global_action")
        if self.global_action.source != self.rooted_source:
            raise ValueError("GlobalAction does not preserve the rooted source")
        if self.global_action.scheduled != self.scheduled:
            raise ValueError("GlobalAction does not preserve the exact schedule")
        self.n6_intent.validate("rooted_ar_case.n6_intent")
        if self.n6_intent.source != self.global_action:
            raise ValueError("N6 intent does not preserve the exact GlobalAction")


def build_s2_lite_dp2_rooted_ar_case() -> S2LiteDp2RootedArCase:
    source = build_s2_lite_source_case()
    rooted_source = build_s2_lite_dp2_rooted_ar_source(source.logical)
    fabric, hbm_address_spaces = load_physical_fabric_and_hbm_address_spaces(
        source.runtime_inputs.source_hardware_path,
        source.runtime_inputs.source_mapping_path,
    )
    placement_context = PlacementContext.create(
        producer_pass=_PRODUCER,
        fabric=fabric,
        placement=source.spec.placement,
        hbm_address_spaces=hbm_address_spaces,
    )
    placed = place_train_forward_ir0(rooted_source.graph, placement_context)
    partition_context = FusionPartitionContext.create(producer_pass=_PRODUCER)
    partitioned = partition_train_forward(placed, partition_context)

    registry = production_registry()
    planning_context = InterDiePlanningContext.create(
        producer_pass=_PRODUCER,
        fused_policy=registry.instantiate(
            RegistryKind.INTER_DIE, "naive"
        ).selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
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
            RegistryKind.INTRA_DIE, "naive"
        ).selection,
    )
    scheduled = schedule_train_forward(projected, scheduling_context)
    global_action = build_s2_lite_dp2_rooted_ar_global_action(
        rooted_source,
        scheduled,
    )
    n6_intent = build_s2_lite_rooted_ar_n6_intent(global_action)
    result = S2LiteDp2RootedArCase(
        case_id=S2_LITE_DP2_ROOTED_AR_CASE_ID,
        source=source,
        rooted_source=rooted_source,
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
        n6_intent=n6_intent,
    )
    result.validate()
    return result


__all__ = [
    "S2_LITE_DP2_ROOTED_AR_CASE_ID",
    "S2LiteDp2RootedArCase",
    "build_s2_lite_dp2_rooted_ar_case",
]
