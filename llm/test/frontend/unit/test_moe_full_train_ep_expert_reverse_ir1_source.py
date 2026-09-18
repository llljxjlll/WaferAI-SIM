"""EP2 expert reverse must retain real source tensors and physical owners."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_expert_reverse_ir1_source import (
    build_moe_ep2_expert_reverse_ir1_candidate,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep2_reverse_rank_plan import (
    build_moe_ep2_reverse_rank_plan,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep2_backward_native_candidate import (
    build_moe_ep2_backward_native_candidate,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_placement import (
    build_moe_full_train_ep_placement,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_ir0 import (
    build_moe_full_train_forward_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeEp2ExpertReverseIr1SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.cases = []
        for step in (0, 1):
            phase = (Fixture.phase if step == 0 else
                     build_moe_full_train_forward_ir0(
                         Fixture.dense, Fixture.sequence, step=step))
            placement = (Fixture.placement if step == 0 else
                         build_moe_full_train_ep_placement(
                             phase, original_dense=Fixture.dense,
                             dense_manifest=Fixture.manifest,
                             sequence=Fixture.sequence,
                             context=Fixture.context))
            candidate = build_moe_ep2_expert_reverse_ir1_candidate(
                phase, original_dense=Fixture.dense,
                sequence=Fixture.sequence, placement=placement,
                context=Fixture.context, dense_manifest=Fixture.manifest,
            )
            cls.cases.append((phase, placement, candidate))

    def test_two_steps_have_official_ir1_and_distinct_expert_owner_tapes(self):
        for step, (phase, placement, candidate) in enumerate(self.cases):
            with self.subTest(step=step):
                source = candidate.source_ir0
                ir1 = candidate.physical_ir1
                self.assertEqual(len(source.nodes), 41)
                self.assertEqual(ir1.source_ir0_id, source.id)
                self.assertEqual(sum(node.kind is OpKind.MOE_EXPERT_BACKWARD
                                     for node in source.nodes), 2)
                self.assertEqual(tuple(place.die_id for place in
                                       ir1.groups[0].placements), (0, 1))
                self.assertEqual(ir1.persistent_state_manifest,
                                 placement.persistent_state_manifest)
                candidate.validate_source_against(
                    phase, original_dense=Fixture.dense,
                    sequence=Fixture.sequence, placement=placement,
                    context=Fixture.context, dense_manifest=Fixture.manifest,
                )
                ir1.validate()

    def test_remote_payloads_reuse_forward_return_and_expose_dexpert1(self):
        for step, (_, _, candidate) in enumerate(self.cases):
            with self.subTest(step=step):
                plan = build_moe_ep2_reverse_rank_plan(
                    candidate, Fixture.sequence)
                self.assertEqual(len(plan.node_ranks), 41)
                self.assertEqual(len(plan.remote_payloads), 7)
                self.assertEqual(sum(len(item.consumer_node_refs)
                                     for item in plan.remote_payloads), 9)
                self.assertEqual(sum(item.lacks_source_bound_transport
                                     for item in plan.remote_payloads), 3)
                by_value = {item.value_ref: item
                            for item in plan.remote_payloads}
                dispatched = by_value["T0.layer1.moe.dispatch1"]
                self.assertEqual(dispatched.consumer_node_refs,
                                 ("T0.layer1.moe.expert1",
                                  "backward::T0.layer1.moe.expert1"))
                self.assertIsNotNone(dispatched.existing_flow_ref)
                returned = by_value["T0.layer1.moe.expert1.output"]
                self.assertEqual(returned.consumer_node_refs,
                                 ("T0.layer1.moe.combine",
                                  "backward::T0.layer1.moe.combine"))
                self.assertIsNotNone(returned.existing_flow_ref)
                gradient = by_value[
                    "backward::T0.layer1.moe.combine.dexpert1"]
                self.assertEqual((gradient.source_rank,
                                  gradient.destination_rank,
                                  gradient.bytes), (0, 1, 16))
                self.assertEqual(gradient.consumer_node_refs,
                                 ("backward::T0.layer1.moe.expert1",))
                self.assertTrue(gradient.lacks_source_bound_transport)
                self.assertIsNotNone(gradient.candidate_backward_flow_ref)
                self.assertIsNotNone(gradient.candidate_backward_send_action_ref)
                self.assertIsNotNone(gradient.candidate_backward_recv_action_ref)
                self.assertIsNotNone(gradient.candidate_backward_wait_action_ref)
                native = build_moe_ep2_backward_native_candidate(
                    plan, candidate, Fixture.sequence)
                self.assertEqual((native.step, native.layer, native.bytes),
                                 (step, 1, 16))
                self.assertEqual(native.p2_flow_ref,
                                 gradient.candidate_backward_flow_ref)
                self.assertIs(native.send.opcode, RecordOpcode.DTE_SEND)
                self.assertIs(native.recv.opcode, RecordOpcode.DTE_RECV)
                self.assertIs(native.wait.opcode, RecordOpcode.DTE_WAIT)
                self.assertEqual((native.send.core_die,
                                  native.recv.core_die), (0, 1))
                native.validate_against(plan, candidate, Fixture.sequence)
                with self.assertRaisesRegex(
                        SchemaError, "candidate differs from source/P2"):
                    replace(native, recv=replace(
                        native.recv, record_index=999,
                    )).validate_against(plan, candidate, Fixture.sequence)
                plan.validate_against(candidate, Fixture.sequence)
                forged = replace(plan, remote_payloads=(
                    replace(plan.remote_payloads[0], bytes=99),
                    *plan.remote_payloads[1:],
                ))
                with self.assertRaisesRegex(
                        SchemaError, "differ from source and signed P2"):
                    forged.validate_against(candidate, Fixture.sequence)

    def test_physical_owner_or_source_drift_is_rejected(self):
        phase, placement, candidate = self.cases[0]
        for forged in (
            replace(candidate, source_ir0=candidate.shared.source_ir0),
            replace(candidate, physical_ir1=replace(
                candidate.physical_ir1,
                persistent_state_manifest=None,
            )),
        ):
            with self.subTest(forged=forged), self.assertRaisesRegex(
                    SchemaError, "differs from signed source or HBM owners"):
                forged.validate_source_against(
                    phase, original_dense=Fixture.dense,
                    sequence=Fixture.sequence, placement=placement,
                    context=Fixture.context, dense_manifest=Fixture.manifest,
                )


if __name__ == "__main__":
    unittest.main()
