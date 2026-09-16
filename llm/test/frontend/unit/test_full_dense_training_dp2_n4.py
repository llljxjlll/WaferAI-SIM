"""Real TP2/DP2 N4 plans carry both local TP and cross-die FP32 SUM work."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_train_forward
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.load_fabric import hbm_address_spaces_from_data, physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.policies.registry import RegistryKind, production_registry
from llm.frontend.wafer_frontend.schema.action import FusionActionKind
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext, InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec


class FullDenseDP2N4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_flexible_dense_train_plan(_spec(2, 2), RectMeshSpec(2, 2))
        cls.graph = build_full_dense_training_two_step_ir0(cls.plan)
        hardware = _hardware(2, 2)
        cls.placement = PlacementContext.create(
            producer_pass="dp2_production_n4",
            fabric=physical_fabric_from_data(hardware),
            placement=cls.plan.source_experiment.placement,
            hbm_address_spaces=hbm_address_spaces_from_data(hardware),
        )
        placed = place_train_forward_ir0(cls.graph, cls.placement)
        cls.partitioned = partition_train_forward(
            placed, FusionPartitionContext.create(producer_pass="dp2_production_n4"),
        )
        registry = production_registry()
        cls.planning = InterDiePlanningContext.create(
            producer_pass="dp2_production_n4",
            fused_policy=registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
            standalone_policy=registry.instantiate(
                RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
            ).selection,
        )
        cls.result = plan_train_forward(
            cls.partitioned, cls.planning,
            dense_dp2_plan=cls.plan, dp2_placement_context=cls.placement,
        )

    def test_all_true_tp_and_dp_collectives_are_covered_once(self) -> None:
        result = self.result
        result.validate_against(self.partitioned, self.planning)
        self.assertEqual(len(result.replicas), 2)
        self.assertEqual(len(result.dp_gradient_routes.gradients), 60)
        self.assertEqual(
            [(len(item.standalone_plans), len(item.dp_sync_refs))
             for item in result.replicas],
            [(24, 60), (24, 60)],
        )
        for gradient in result.dp_gradient_routes.gradients:
            actions = tuple(action for program in gradient.rank_programs
                            for action in program.actions)
            self.assertEqual(
                [action.kind for action in actions].count(FusionActionKind.REDUCE), 1,
            )
            self.assertEqual(
                [action.kind for action in actions].count(FusionActionKind.SEND), 2,
            )

    def test_missing_or_truncated_cross_replica_plan_fails_closed(self) -> None:
        with self.assertRaisesRegex(SchemaError, "exact plan and physical N3 placement"):
            plan_train_forward(self.partitioned, self.planning)
        route_plan = self.result.dp_gradient_routes
        with self.assertRaisesRegex(SchemaError, "exact two-DP route coverage"):
            replace(self.result, dp_gradient_routes=replace(
                route_plan, gradients=route_plan.gradients[1:],
            )).validate()
        with self.assertRaisesRegex(SchemaError, "unused cross-replica DP route plan|requires exact"):
            replace(self.result, dp_gradient_routes=None).validate()


if __name__ == "__main__":
    unittest.main()
