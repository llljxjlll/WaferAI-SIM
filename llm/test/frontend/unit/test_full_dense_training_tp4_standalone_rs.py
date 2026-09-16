"""Real TP4 backward SUM ReduceScatter must retain all routed contributions."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_train_forward
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data, hbm_address_spaces_from_data
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.policies.registry import production_registry, RegistryKind
from llm.frontend.wafer_frontend.schema.action import FusionActionKind, StandaloneCollectivePlan
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, OpPhase
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext, InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec


class FullDenseTP4StandaloneRSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = build_flexible_dense_train_plan(_spec(1, 4), RectMeshSpec(1, 4))
        g = build_full_dense_training_two_step_ir0(p)
        h = _hardware(1, 4)
        ir1 = place_train_forward_ir0(g, PlacementContext.create(
            producer_pass="dense_tp4_rs_test", fabric=physical_fabric_from_data(h),
            placement=p.source_experiment.placement,
            hbm_address_spaces=hbm_address_spaces_from_data(h)))
        partition = partition_train_forward(ir1, FusionPartitionContext.create(
            producer_pass="dense_tp4_rs_test"))
        registry = production_registry()
        planned = plan_train_forward(partition, InterDiePlanningContext.create(
            producer_pass="dense_tp4_rs_test",
            fused_policy=registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
            standalone_policy=registry.instantiate(RegistryKind.STANDALONE_COLLECTIVE,
                                                   "direct_all_gather").selection))
        cls.ir1 = planned.replicas[0].graph
        cls.plan = next(p for p in planned.replicas[0].standalone_plans
                        if next(n for n in cls.ir1.nodes if n.id == p.op_id).phase is OpPhase.DGRAD
                        and next(n for n in cls.ir1.nodes if n.id == p.op_id).workload.collective
                        is CollectiveKind.REDUCE_SCATTER)

    @staticmethod
    def _rebuild(plan, rank_programs):
        return StandaloneCollectivePlan.create(
            producer_pass=plan.producer_pass, source_ir1_id=plan.source_ir1_id,
            op_id=plan.op_id, algorithm=plan.algorithm, group_ref=plan.group_ref,
            profile_key=plan.profile_key, chunk_dim=plan.chunk_dim,
            chunk_slices=plan.chunk_slices, rank_programs=rank_programs)

    def test_real_source_routes_waits_and_all_four_owner_reductions(self):
        plan = self.plan
        plan.validate_against(self.ir1)
        for rank, program in enumerate(plan.rank_programs):
            self.assertEqual(program.rank, rank)
            self.assertEqual(sum(a.kind is FusionActionKind.LOCAL_COPY for a in program.actions), 1)
            self.assertEqual(sum(a.kind is FusionActionKind.SEND for a in program.actions), 3)
            self.assertEqual(sum(a.kind is FusionActionKind.RECV for a in program.actions), 3)
            self.assertEqual(sum(a.kind is FusionActionKind.WAIT for a in program.actions), 3)
            self.assertEqual(sum(a.kind is FusionActionKind.REDUCE for a in program.actions), 1)
            reduce = next(a for a in program.actions if a.kind is FusionActionKind.REDUCE)
            self.assertEqual(reduce.reduction.input_ranks, (0, 1, 2, 3))
            self.assertEqual(len(reduce.deps), 4)

    def test_missing_wait_and_forged_contribution_source_fail_closed(self):
        plan = self.plan
        owner = plan.rank_programs[0]
        wait = next(a for a in owner.actions if a.kind is FusionActionKind.WAIT)
        programs = (replace(owner, actions=tuple(a for a in owner.actions if a != wait)),
                    *plan.rank_programs[1:])
        with self.assertRaises(SchemaError):
            self._rebuild(plan, programs).validate_against(self.ir1)
        sender = plan.rank_programs[1]
        send = next(a for a in sender.actions if a.kind is FusionActionKind.SEND)
        programs = (plan.rank_programs[0], replace(sender, actions=tuple(
            replace(a, reads=(plan.op_id,)) if a == send else a for a in sender.actions)),
            *plan.rank_programs[2:])
        with self.assertRaisesRegex(SchemaError, "source partial tensor"):
            self._rebuild(plan, programs).validate_against(self.ir1)


if __name__ == "__main__":
    unittest.main()
