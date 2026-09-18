"""EP2 execution ranks and physical transfers must follow true source owners."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_moe import MoeRectFlowStage
from llm.frontend.wafer_frontend.passes.moe_full_train_ep2_rank_plan import (
    build_moe_ep2_rank_plan,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_ir1_source import (
    build_moe_ep_shared_reverse_ir1_candidate,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_placement import (
    build_moe_full_train_ep_placement,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_ir0 import (
    build_moe_full_train_forward_ir0,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeEp2RankPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.candidates = []
        for step in (0, 1):
            phase = Fixture.phase if step == 0 else build_moe_full_train_forward_ir0(
                Fixture.dense, Fixture.sequence, step=step,
            )
            placement = Fixture.placement if step == 0 else build_moe_full_train_ep_placement(
                phase, original_dense=Fixture.dense,
                dense_manifest=Fixture.manifest, sequence=Fixture.sequence,
                context=Fixture.context,
            )
            cls.candidates.append(build_moe_ep_shared_reverse_ir1_candidate(
                phase, original_dense=Fixture.dense,
                sequence=Fixture.sequence, placement=placement,
                context=Fixture.context, dense_manifest=Fixture.manifest,
            ))

    def test_both_steps_have_exact_owner_ranks_and_six_remote_values(self):
        plans = [build_moe_ep2_rank_plan(source, Fixture.sequence) for source in self.candidates]
        self.assertNotEqual(plans[0].source_ir1_id, plans[1].source_ir1_id)
        for source, plan in zip(self.candidates, plans, strict=True):
            plan.validate_against(source, Fixture.sequence)
            self.assertEqual(len(plan.node_ranks), 38)
            self.assertEqual(len(plan.transfers), 6)
            self.assertEqual({(item.source_die, item.destination_die)
                              for item in plan.transfers}, {(0, 1), (1, 0)})
            self.assertTrue(all(item.bytes == 16 and item.die_path ==
                                (item.source_die, item.destination_die)
                                for item in plan.transfers))
            self.assertEqual(sum(item.source_flow_ref is not None
                                 for item in plan.transfers), 4)
            ranks = {item.node_ref: item.rank for item in plan.node_ranks}
            self.assertEqual({ref for ref, rank in ranks.items() if rank == 1},
                             {f"T0.layer{layer}.moe.expert1" for layer in (0, 1)})
            for layer in (0, 1):
                remote = {item.value_ref: item for item in plan.transfers
                          if item.value_ref.startswith(f"T0.layer{layer}.")}
                self.assertEqual(set(remote), {
                    f"T0.layer{layer}.moe.router.weight.ep1",
                    f"T0.layer{layer}.moe.dispatch1",
                    f"T0.layer{layer}.moe.expert1.output",
                })
                self.assertIsNotNone(remote[
                    f"T0.layer{layer}.moe.router.weight.ep1"].source_state_ref)
                step = 0 if source is self.candidates[0] else 1
                unit = next(unit for unit in Fixture.sequence.units
                            if (unit.step, unit.layer) == (step, layer))
                for value_ref, stage in (
                    (f"T0.layer{layer}.moe.dispatch1", MoeRectFlowStage.DISPATCH),
                    (f"T0.layer{layer}.moe.expert1.output", MoeRectFlowStage.COMBINE),
                ):
                    actual = remote[value_ref]
                    self.assertEqual(
                        actual.source_flow_ref,
                        next(flow.id for flow in unit.plan.flows
                             if flow.stage is stage
                             and flow.source_rank == actual.source_rank
                             and flow.destination_rank == actual.destination_rank),
                    )

    def test_forged_byte_count_or_missing_return_is_rejected(self):
        source = self.candidates[0]
        plan = build_moe_ep2_rank_plan(source, Fixture.sequence)
        first = replace(plan.transfers[0], bytes=plan.transfers[0].bytes + 2)
        for forged in (
            replace(plan, transfers=(first, *plan.transfers[1:])),
            replace(plan, transfers=plan.transfers[:-1]),
            replace(plan, transfers=(replace(
                plan.transfers[0], source_flow_ref="forged_flow",
            ), *plan.transfers[1:])),
        ):
            with self.subTest(forged=forged), self.assertRaisesRegex(
                    SchemaError, "differs from source/owner route"):
                forged.validate_against(source, Fixture.sequence)


if __name__ == "__main__":
    unittest.main()
