"""True frozen EP2 route and full gate/up/down work before IR0 materialization."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.moe_full_training_block_workload import (
    MoeForwardBlockKind, MoeFullTrainingBlockWorkload,
)
from llm.test.frontend.unit.test_full_training_timeline_linker import (
    FullTrainingTimelineLinkerTest,
)


class MoeFullTrainingBlockWorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FullTrainingTimelineLinkerTest.setUpClass()
        cls.sequence = FullTrainingTimelineLinkerTest.moe
        cls.unit = next(unit for unit in cls.sequence.units
                        if (unit.step, unit.layer) == (0, 0))
        cls.trace = next(trace for trace in
                         cls.sequence.materialization.logical_graph.route_traces
                         if (trace.step, trace.layer) == (0, 0))

    def _source(self, kind: MoeForwardBlockKind,
                expert: int | None = None) -> MoeFullTrainingBlockWorkload:
        binding = self.unit.operation_binding
        ids = {
            MoeForwardBlockKind.ROUTER: binding.router_operation_ref,
            MoeForwardBlockKind.ROUTE_FREEZE: binding.route_freeze_operation_ref,
            MoeForwardBlockKind.DISPATCH: binding.dispatch_operation_ref,
            MoeForwardBlockKind.COMBINE: binding.combine_operation_ref,
            MoeForwardBlockKind.EXPERT:
                (binding.expert_forward_operation_refs[expert]
                 if expert is not None else None),
        }
        model = self.sequence.materialization.request.model
        return MoeFullTrainingBlockWorkload(
            kind, self.sequence.materialization.request.case_id,
            ids[kind], self.trace.id, self.unit.route_trace_digest,
            self.unit.step, self.unit.layer, expert,
            self.trace.token_count, model.hidden_size,
            model.intermediate_size, model.num_experts,
            self.trace.expert_token_counts, self.trace.expert_by_token,
            tuple(item.slot_index for item in sorted(
                self.unit.spec.trace.assignments,
                key=lambda item: item.token_index)),
        )

    def test_frozen_two_expert_full_three_projection_work(self):
        from llm.frontend.wafer_frontend.schema.flexible_moe import (
            MoeRectActionKind,
        )
        actual = {
            kind: tuple(action.flops for action in self.unit.plan.actions
                        if action.kind is kind)
            for kind in (MoeRectActionKind.GATE,
                         MoeRectActionKind.EXPERT_FORWARD,
                         MoeRectActionKind.WEIGHTED_COMBINE)
        }
        for kind in MoeForwardBlockKind:
            if kind is MoeForwardBlockKind.EXPERT:
                for expert in range(2):
                    action = self._source(kind, expert)
                    action.validate()
                    self.assertEqual((action.owned_token_count,
                                      action.projected_matmul_flops,
                                      action.expert_projection_parameter_bytes),
                                     (2, 384, 192))
                    self.assertEqual(action.projected_matmul_flops,
                                     actual[MoeRectActionKind.EXPERT_FORWARD][expert])
            else:
                action = self._source(kind)
                action.validate()
                self.assertEqual(action.projected_matmul_flops,
                                 64 if kind is MoeForwardBlockKind.ROUTER else 0)
                if kind is MoeForwardBlockKind.ROUTER:
                    self.assertEqual(action.projected_matmul_flops,
                                     actual[MoeRectActionKind.GATE][0])
                if kind is MoeForwardBlockKind.COMBINE:
                    self.assertEqual(action.projected_vector_flops,
                                     actual[MoeRectActionKind.WEIGHTED_COMBINE][0])

    def test_route_or_owner_tampering_does_not_satisfy_source(self):
        original = self._source(MoeForwardBlockKind.EXPERT, 1)
        for forged in (
            replace(original, expert=2),
            replace(original, expert=None),
            replace(original, expert_histogram=(4, 0)),
            replace(original, frozen_expert_by_token=(0, 0, 0, 1)),
            replace(original, frozen_slot_by_token=(0, 0, 0, 0)),
        ):
            with self.assertRaises(SchemaError):
                forged.validate()
        with self.assertRaisesRegex(SchemaError, "cannot claim expert"):
            replace(self._source(MoeForwardBlockKind.DISPATCH),
                    expert=1).validate()


if __name__ == "__main__":
    unittest.main()
