"""Source-proven EP1 identity dispatch and native DTE copy rejection tests."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.lifecycle import add_fixed_sram_lifecycle
from llm.frontend.wafer_frontend.lowering.moe_full_train_dispatch import lower_moe_dispatch
from llm.frontend.wafer_frontend.schema.artifact_manifest import CommandFragment, RecordOpcode
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_expert_microplan import (
    MoeFullTrainExpertMicroplanTest as Fixture,
)


class MoeDispatchNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.actions = tuple(action for action in Fixture.dag.actions
                            if action.op_kind is OpKind.MOE_DISPATCH)

    def test_both_layers_have_signed_identity_slots_and_native_copy(self):
        self.assertEqual(len(self.actions), 2)
        for action in self.actions:
            workload = action.compute.workload
            self.assertEqual(workload.frozen_slot_by_token, tuple(range(4)))
            self.assertEqual(workload.frozen_expert_by_token, (0,) * 4)
            self.assertIsNotNone(action.runtime_binding.token_symbol)
            fragment = lower_moe_dispatch(action, Fixture.context)
            fragment.validate_against(Fixture.dag)
            self.assertEqual([record.opcode for record in fragment.core_streams[0].records],
                             [RecordOpcode.DTE_ISSUE, RecordOpcode.DTE_WAIT])
            decorated = add_fixed_sram_lifecycle(fragment, Fixture.context)
            decorated.validate_against(Fixture.dag)
            self.assertEqual(len(decorated.core_streams[0].records), 4)

    def test_nonidentity_permutation_is_valid_source_but_cannot_use_copy(self):
        workload = self.actions[0].compute.workload
        nonidentity = replace(workload, frozen_slot_by_token=(1, 0, 2, 3))
        nonidentity.validate()
        self.assertNotEqual(nonidentity.frozen_slot_by_token,
                            tuple(range(nonidentity.token_count)))

    def test_phantom_short_dte_payload_is_rejected(self):
        fragment = lower_moe_dispatch(self.actions[0], Fixture.context)
        stream = fragment.core_streams[0]
        issue = stream.records[0]
        forged = CommandFragment.create(
            producer_pass=fragment.producer_pass,
            **{**fragment._semantic_key(), "core_streams": (
                replace(stream, records=(replace(issue, operands=(
                    *issue.operands[:3],
                    replace(issue.operands[3], literal_value=16),
                    *issue.operands[4:])), stream.records[1])),)},
        )
        with self.assertRaisesRegex(SchemaError, "dispatch DTE payload/token"):
            forged.validate_against(Fixture.dag)


if __name__ == "__main__":
    unittest.main()
