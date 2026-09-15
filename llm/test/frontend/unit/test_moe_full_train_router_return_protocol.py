"""Real P2 top1 transport/reassembly offset proof and old overlap rejection."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_router_return_protocol import (
    build_moe_router_signed_return_protocol,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_router_score_source import (
    build_moe_trainable_signed_router_requirements,
)
from llm.frontend.wafer_frontend.schema.flexible_moe import MoeRectFlowStage
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeFullTrainRouterReturnProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.score = build_moe_trainable_signed_router_requirements(
            Fixture.sequence)
        cls.placement = build_moe_router_signed_return_protocol(
            cls.score, Fixture.sequence)

    def test_four_signed_source_rank_return_tapes_keep_each_token_and_slot(self):
        self.placement.validate_against(self.score, Fixture.sequence)
        self.assertEqual(len(self.placement.placements), 4)
        for placement in self.placement.placements:
            self.assertEqual(placement.source_rank, 0)
            self.assertEqual(placement.score_tape_bytes, 16)
            self.assertEqual(placement.expert_return_bytes, 32)
            self.assertEqual(placement.combined_output_bytes, 32)
            self.assertEqual([
                (group.expert_home_rank, group.expert_index,
                 group.offset_bytes, group.size_bytes,
                 group.assignment_refs)
                for group in placement.segments], [
                    (0, 0, 0, 16, ("assignment.0", "assignment.2")),
                    (1, 1, 16, 16, ("assignment.1", "assignment.3")),
                ])
            self.assertIsNone(placement.segments[0].combine_flow_ref)
            self.assertIsNotNone(placement.segments[1].combine_flow_ref)
            self.assertEqual([
                (lane.token_index, lane.expert_home_rank,
                 lane.signed_score_offset_bytes,
                 lane.expert_output_offset_bytes,
                 lane.combined_output_offset_bytes)
                for lane in placement.lanes], [
                    (0, 0, 0, 0, 0), (1, 1, 6, 16, 8),
                    (2, 0, 8, 8, 16), (3, 1, 14, 24, 24),
                ])

    def test_remote_return_is_one_real_p2_flow_with_full_expert_bytes(self):
        for placement in self.placement.placements:
            unit = next(unit for unit in Fixture.sequence.units
                        if (unit.step, unit.layer)
                        == (placement.step, placement.layer))
            remote = placement.segments[1]
            flow = next(flow for flow in unit.plan.flows
                        if flow.id == remote.combine_flow_ref)
            self.assertIs(flow.stage, MoeRectFlowStage.COMBINE)
            self.assertEqual((flow.source_rank, flow.destination_rank),
                             (1, 0))
            self.assertEqual(flow.assignment_refs, remote.assignment_refs)
            self.assertEqual(flow.logical_bytes, remote.size_bytes)

    def test_old_remote_recv_does_not_land_after_local_expert_staging(self):
        with self.assertRaisesRegex(SchemaError,
                                    "RETURN lacks one source-owned rank-major physical SRAM tape"):
            self.placement.require_physical_return_transport(
                self.score, Fixture.sequence)

    def test_forged_source_score_identity_cannot_change_return_tape(self):
        wrong = replace(self.score, dynamic_score_case_ref=
                        self.score.original_static_case_ref)
        with self.assertRaisesRegex(SchemaError,
                                    "source/hardware/route/score contract"):
            build_moe_router_signed_return_protocol(
                wrong, Fixture.sequence)

    def test_dropped_remote_home_or_overlapping_reception_rejected(self):
        path = self.placement.placements[0]
        segment = path.segments[1]
        for changed in (replace(segment, offset_bytes=0),
                        replace(segment, size_bytes=8),
                        replace(segment, combine_flow_ref=None)):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                    SchemaError, "route table, EP return bytes"):
                replace(self.placement,
                    placements=(replace(path,
                        segments=(*path.segments[:1], changed)),
                        *self.placement.placements[1:])).validate_against(
                            self.score, Fixture.sequence)


if __name__ == "__main__":
    unittest.main()
