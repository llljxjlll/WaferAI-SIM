from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    partition_stage4 as public_partition_stage4,
    place_stage4_carrier as public_place_stage4_carrier,
    plan_stage4 as public_plan_stage4,
)
from llm.frontend.wafer_frontend.passes.fusion_partition import (
    partition_bundle,
    partition_stage4,
)
from llm.frontend.wafer_frontend.passes.inter_die_plan import (
    plan_bundle,
    plan_stage4,
)
from llm.frontend.wafer_frontend.passes.placement import place_stage4_carrier
from llm.frontend.wafer_frontend.passes.stage4_logical_expand import (
    build_stage4_fused_ir0,
)
from llm.frontend.wafer_frontend.schema import (
    Stage4FusionPartitionedIR1 as PublicStage4FusionPartitionedIR1,
    Stage4InterDiePlannedIR1 as PublicStage4InterDiePlannedIR1,
    Stage4PlacedIR1 as PublicStage4PlacedIR1,
)
from llm.frontend.wafer_frontend.schema.ir0 import IR0, LogicalRole
from llm.frontend.wafer_frontend.schema.n4 import (
    STAGE4_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
    STAGE4_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
    FusionPartitionContext,
    FusionPartitionedIR1Bundle,
    Stage4FusionPartitionedIR1,
    Stage4InterDiePlannedIR1,
)
from llm.frontend.wafer_frontend.schema.placed_ir1 import (
    STAGE4_PLACED_IR1_SCHEMA_VERSION,
    PlacedIR1Bundle,
    Stage4PlacedIR1,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass

from _fixtures import (
    naive_inter_die_planning_context,
    valid_hbm_address_spaces,
    valid_ir1,
)
from test_stage4_fused_ir0 import _case as _fused_case
from test_stage4_inter_die_plan import _stage4_case


def _chain(prefill_tp: int, decode_tp: int):
    graph, placement_context, pd_plan = _stage4_case(prefill_tp, decode_tp)
    placed = place_stage4_carrier(graph, placement_context, pd_plan)
    partition_context = FusionPartitionContext.create(producer_pass="stage4_test")
    partitioned = partition_stage4(placed, partition_context)
    planning_context = naive_inter_die_planning_context("stage4_test")
    planned = plan_stage4(partitioned, planning_context)
    return (
        graph,
        placement_context,
        partition_context,
        planning_context,
        placed,
        partitioned,
        planned,
    )


def _fused_chain():
    spec, pd_plan = _fused_case()
    graph = build_stage4_fused_ir0(spec, pd_plan)
    fabric = valid_ir1().fabric
    placement_context = PlacementContext.create(
        producer_pass="stage4_fused_carrier_test",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    placed = place_stage4_carrier(graph, placement_context, pd_plan)
    partition_context = FusionPartitionContext.create(
        producer_pass="stage4_fused_carrier_test"
    )
    partitioned = partition_stage4(placed, partition_context)
    planning_context = naive_inter_die_planning_context(
        "stage4_fused_carrier_test"
    )
    planned = plan_stage4(partitioned, planning_context)
    return (
        graph,
        placement_context,
        partition_context,
        planning_context,
        placed,
        partitioned,
        planned,
    )


class Stage4CarrierTest(unittest.TestCase):
    def test_public_exports_resolve_exact_carrier_apis(self) -> None:
        self.assertIs(PublicStage4PlacedIR1, Stage4PlacedIR1)
        self.assertIs(
            PublicStage4FusionPartitionedIR1,
            Stage4FusionPartitionedIR1,
        )
        self.assertIs(
            PublicStage4InterDiePlannedIR1,
            Stage4InterDiePlannedIR1,
        )
        self.assertIs(public_place_stage4_carrier, place_stage4_carrier)
        self.assertIs(public_partition_stage4, partition_stage4)
        self.assertIs(public_plan_stage4, plan_stage4)

    def test_tp1_and_heterogeneous_exact_chain_round_trip(self) -> None:
        self.assertEqual(
            STAGE4_PLACED_IR1_SCHEMA_VERSION,
            "wafer_frontend.stage4_placed_ir1/v1alpha2",
        )
        self.assertEqual(
            STAGE4_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
            "wafer_frontend.stage4_fusion_partitioned_ir1/v1alpha2",
        )
        self.assertEqual(
            STAGE4_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
            "wafer_frontend.stage4_inter_die_planned_ir1/v1alpha2",
        )
        for prefill_tp, decode_tp, expected in ((1, 1, (0, 0)), (2, 1, (4, 4))):
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                (
                    graph,
                    placement_context,
                    partition_context,
                    planning_context,
                    placed,
                    partitioned,
                    planned,
                ) = _chain(prefill_tp, decode_tp)
                placed.validate_against(graph, placement_context)
                partitioned.validate_against(placed, partition_context)
                planned.validate_against(partitioned, planning_context)
                self.assertEqual(
                    (len(planned.fusion_plans), len(planned.standalone_plans)),
                    expected,
                )
                self.assertEqual(partitioned.pd_plan, placed.pd_plan)
                self.assertEqual(planned.pd_plan, placed.pd_plan)
                self.assertEqual(planned.graph, partitioned.graph)
                self.assertEqual(placed.graph.node_profiles, graph.node_profiles)
                self.assertEqual(
                    partitioned.graph.node_profiles,
                    placed.graph.node_profiles,
                )
                self.assertEqual(
                    planned.graph.node_profiles,
                    partitioned.graph.node_profiles,
                )
                for artifact_type, artifact in (
                    (Stage4PlacedIR1, placed),
                    (Stage4FusionPartitionedIR1, partitioned),
                    (Stage4InterDiePlannedIR1, planned),
                ):
                    self.assertEqual(
                        loads_dataclass(artifact_type, canonical_json(artifact)),
                        artifact,
                    )

    def test_fused_tp1_exact_chain_has_one_owner_two_profiles_and_no_routes(self) -> None:
        (
            graph,
            placement_context,
            partition_context,
            planning_context,
            placed,
            partitioned,
            planned,
        ) = _fused_chain()
        placed.validate_against(graph, placement_context)
        partitioned.validate_against(placed, partition_context)
        planned.validate_against(partitioned, planning_context)
        self.assertEqual(
            (
                len(placed.graph.instances),
                len(placed.graph.groups),
                len(placed.graph.instance_profiles),
                len(placed.graph.node_profiles),
                len(placed.graph.cross_routes),
                len(placed.pd_plan.handoffs),
                len(partitioned.graph.fused_op_skeletons),
                len(planned.fusion_plans),
                len(planned.standalone_plans),
            ),
            (1, 1, 2, 50, 0, 0, 0, 0, 0),
        )
        self.assertEqual(placed.graph.instances[0].id, "F0")
        self.assertIs(placed.graph.instances[0].role, LogicalRole.BOTH)
        self.assertEqual(
            {binding.instance_ref for binding in placed.graph.instance_profiles},
            {"F0"},
        )
        self.assertEqual(placed.graph.instance_profiles, graph.instance_profiles)
        self.assertEqual(placed.graph.node_profiles, graph.node_profiles)
        self.assertEqual(partitioned.graph.node_profiles, graph.node_profiles)
        self.assertEqual(planned.graph.node_profiles, graph.node_profiles)
        for artifact_type, artifact in (
            (Stage4PlacedIR1, placed),
            (Stage4FusionPartitionedIR1, partitioned),
            (Stage4InterDiePlannedIR1, planned),
        ):
            self.assertEqual(
                loads_dataclass(artifact_type, canonical_json(artifact)),
                artifact,
            )
        self.assertEqual(_fused_chain()[4:], (placed, partitioned, planned))

    def test_provenance_plan_coverage_and_old_bundles_fail_closed(self) -> None:
        (
            graph,
            placement_context,
            partition_context,
            planning_context,
            placed,
            partitioned,
            planned,
        ) = _chain(2, 1)
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(placed, schema_version="wafer_frontend.stage4_placed_ir1/v0").validate()
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                partitioned,
                schema_version="wafer_frontend.stage4_fusion_partitioned_ir1/v1alpha1",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                planned,
                schema_version="wafer_frontend.stage4_inter_die_planned_ir1/v1alpha1",
            ).validate()
        with self.assertRaisesRegex(SchemaError, "exactly cover skeletons"):
            replace(planned, fusion_plans=planned.fusion_plans[:-1]).validate()
        other_graph, _other_context, _other_plan = _stage4_case(1, 1)
        with self.assertRaisesRegex(SchemaError, "must match source IR0"):
            placed.validate_against(other_graph, placement_context)
        with self.assertRaisesRegex(SchemaError, "must be a PlacedIR1Bundle"):
            partition_bundle(placed, partition_context)  # type: ignore[arg-type]
        with self.assertRaisesRegex(SchemaError, "must be a FusionPartitionedIR1Bundle"):
            plan_bundle(partitioned, planning_context)  # type: ignore[arg-type]
        with self.assertRaises(SchemaError):
            loads_dataclass(PlacedIR1Bundle, canonical_json(placed))
        with self.assertRaises(SchemaError):
            loads_dataclass(FusionPartitionedIR1Bundle, canonical_json(partitioned))

        (
            fused_graph,
            fused_placement_context,
            _fused_partition_context,
            _fused_planning_context,
            fused_placed,
            _fused_partitioned,
            _fused_planned,
        ) = _fused_chain()
        with self.assertRaisesRegex(SchemaError, "profile"):
            replace(
                fused_placed,
                graph=replace(
                    fused_placed.graph,
                    instance_profiles=fused_placed.graph.instance_profiles[:1],
                ),
            ).validate()
        wrong_source_fields = fused_graph._semantic_key()
        wrong_source_fields["pd_plan_id"] = placed.pd_plan.id
        wrong_source = IR0.create(
            producer_pass=fused_graph.producer_pass,
            **wrong_source_fields,
        )
        with self.assertRaisesRegex(SchemaError, "must match source IR0"):
            fused_placed.validate_against(
                wrong_source,
                fused_placement_context,
            )


if __name__ == "__main__":
    unittest.main()
