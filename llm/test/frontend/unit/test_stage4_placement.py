from __future__ import annotations

from pathlib import Path
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes import (
    load_physical_fabric,
    place_stage4_ir0,
)
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.group_registry import (
    stage4_route_endpoint_pairs,
)
from llm.frontend.wafer_frontend.passes.placement import place_ir0
from llm.frontend.wafer_frontend.passes.stage4_logical_expand import (
    build_stage4_separated_ir0,
)
from llm.frontend.wafer_frontend.passes.stage4_pd import build_stage4_pd_plan
from llm.frontend.wafer_frontend.schema.placement import PlacementContext

from _fixtures import valid_hbm_address_spaces
from test_stage4_pd import _profile, _spec


_ROOT = Path(__file__).resolve().parents[4]
_HARDWARE = _ROOT / "llm/test/sram/hardware_numa.json"
_MAPPING = _ROOT / "llm/test/default/mapping.spec"


def _tp1_case():
    spec = _spec(1, 1)
    plan = build_stage4_pd_plan(
        spec,
        prefill_profile=_profile(prefill=True),
        decode_profile=_profile(prefill=False),
    )
    graph = build_stage4_separated_ir0(spec, plan)
    fabric = load_physical_fabric(_HARDWARE, _MAPPING)
    context = PlacementContext.create(
        producer_pass="stage4_placement_test",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    return graph, context, plan


class Stage4PlacementTest(unittest.TestCase):
    def test_tp1_exact_route_provenance_and_partition_preservation(self) -> None:
        graph, context, plan = _tp1_case()

        generic = place_ir0(graph, context)
        self.assertEqual(generic.instance_profiles, graph.instance_profiles)
        self.assertEqual(generic.pd_plan_id, plan.id)
        self.assertEqual(generic.cross_routes, ())

        placed = place_stage4_ir0(graph, context, plan)
        self.assertEqual(placed.source_ir0_id, graph.id)
        self.assertEqual(placed.instance_profiles, graph.instance_profiles)
        self.assertEqual(placed.pd_plan_id, plan.id)
        self.assertEqual(len(plan.handoffs), 2)
        self.assertEqual(sum(len(item.flows) for item in plan.handoffs), 2)
        self.assertEqual(len(placed.cross_routes), 1)
        route = placed.cross_routes[0]
        groups = {group.id: group for group in placed.groups}
        self.assertEqual(
            (
                groups[route.source_group_ref].instance_id,
                route.source_rank,
                groups[route.destination_group_ref].instance_id,
                route.destination_rank,
                route.die_path,
            ),
            ("P0", 0, "D0", 0, (0, 1)),
        )
        route.validate_against(placed.fabric, groups, "route")

        partitioned = partition_ir1(placed)
        self.assertEqual(partitioned.instance_profiles, placed.instance_profiles)
        self.assertEqual(partitioned.pd_plan_id, placed.pd_plan_id)
        self.assertEqual(partitioned.cross_routes, placed.cross_routes)

    def test_endpoint_pairs_follow_flows_without_cartesian_expansion(self) -> None:
        cases = (
            (1, 1, ((0, 0),)),
            (2, 1, ((0, 0), (1, 0))),
            (1, 2, ((0, 0), (0, 1))),
            (2, 2, ((0, 0), (1, 1))),
            (4, 2, ((0, 0), (1, 0), (2, 1), (3, 1))),
            (2, 4, ((0, 0), (0, 1), (1, 2), (1, 3))),
            (4, 4, ((0, 0), (1, 1), (2, 2), (3, 3))),
        )
        for prefill_tp, decode_tp, expected in cases:
            with self.subTest(prefill_tp=prefill_tp, decode_tp=decode_tp):
                spec = _spec(prefill_tp, decode_tp)
                plan = build_stage4_pd_plan(
                    spec,
                    prefill_profile=_profile(prefill=True),
                    decode_profile=_profile(prefill=False),
                )
                self.assertEqual(stage4_route_endpoint_pairs(plan), expected)

    def test_plan_and_graph_lineage_fail_closed(self) -> None:
        graph, context, plan = _tp1_case()
        fields = graph._semantic_key()
        fields["pd_plan_id"] = "stage4_pd_plan_wrong"
        wrong_graph = type(graph).create(
            producer_pass=graph.producer_pass,
            **fields,
        )
        with self.assertRaisesRegex(SchemaError, "supplied Stage 4 plan id"):
            place_stage4_ir0(wrong_graph, context, plan)

        fused_spec = _spec(1, 1, fused=True)
        fused_plan = build_stage4_pd_plan(
            fused_spec,
            prefill_profile=_profile(prefill=True),
            decode_profile=_profile(prefill=False),
        )
        with self.assertRaisesRegex(SchemaError, "supplied Stage 4 plan id"):
            place_stage4_ir0(graph, context, fused_plan)


if __name__ == "__main__":
    unittest.main()
