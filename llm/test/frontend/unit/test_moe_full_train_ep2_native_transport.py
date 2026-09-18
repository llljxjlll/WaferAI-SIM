"""EP2 source dispatch/return must bind real DTE records from signed P2."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ep2_native_transport import (
    build_moe_ep2_native_transport_proof,
)
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
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeEp2NativeTransportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.cases = []
        for step in (0, 1):
            phase = Fixture.phase if step == 0 else build_moe_full_train_forward_ir0(
                Fixture.dense, Fixture.sequence, step=step,
            )
            placement = Fixture.placement if step == 0 else build_moe_full_train_ep_placement(
                phase, original_dense=Fixture.dense,
                dense_manifest=Fixture.manifest, sequence=Fixture.sequence,
                context=Fixture.context,
            )
            source = build_moe_ep_shared_reverse_ir1_candidate(
                phase, original_dense=Fixture.dense,
                sequence=Fixture.sequence, placement=placement,
                context=Fixture.context, dense_manifest=Fixture.manifest,
            )
            cls.cases.append((source, build_moe_ep2_rank_plan(source, Fixture.sequence)))

    def test_two_steps_bind_four_signed_flows_to_executable_dte_records(self):
        for step, (source, plan) in enumerate(self.cases):
            proof = build_moe_ep2_native_transport_proof(
                plan, source, Fixture.sequence,
            )
            self.assertEqual(len(proof.bindings), 4)
            self.assertEqual({item.step for item in proof.bindings}, {step})
            self.assertEqual({item.layer for item in proof.bindings}, {0, 1})
            self.assertEqual({item.bytes for item in proof.bindings}, {16})
            for item in proof.bindings:
                self.assertIs(item.send.opcode, RecordOpcode.DTE_SEND)
                self.assertIs(item.recv.opcode, RecordOpcode.DTE_RECV)
                self.assertIs(item.wait.opcode, RecordOpcode.DTE_WAIT)
                self.assertNotEqual(item.send.core_die, item.recv.core_die)
                self.assertGreaterEqual(item.send.local_core_id, 0)
                self.assertGreaterEqual(item.recv.local_core_id, 0)
            proof.validate_against(plan, source, Fixture.sequence)

    def test_missing_or_forged_native_record_binding_is_rejected(self):
        source, plan = self.cases[0]
        proof = build_moe_ep2_native_transport_proof(plan, source, Fixture.sequence)
        for forged in (
            replace(proof, bindings=proof.bindings[:-1]),
            replace(proof, bindings=(replace(
                proof.bindings[0], recv=replace(
                    proof.bindings[0].recv, record_index=999,
                ),
            ), *proof.bindings[1:])),
        ):
            with self.subTest(forged=forged), self.assertRaisesRegex(
                    SchemaError, "differ from signed source flow"):
                forged.validate_against(plan, source, Fixture.sequence)


if __name__ == "__main__":
    unittest.main()
