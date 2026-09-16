"""EP1 router score physical MATMUL and fail-closed source binding."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.lifecycle import add_fixed_sram_lifecycle
from llm.frontend.wafer_frontend.lowering.moe_full_train_router import (
    lower_moe_router_score_fragment,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment, RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_expert_microplan import (
    MoeFullTrainExpertMicroplanTest as Fixture,
)


class MoeRouterScoreNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.actions = tuple(action for action in Fixture.dag.actions
                            if action.op_kind is OpKind.MOE_ROUTER)

    def test_two_layers_have_real_score_matmul_and_lifecycle(self):
        self.assertEqual(len(self.actions), 2)
        for action in self.actions:
            fragment = lower_moe_router_score_fragment(
                action, Fixture.schedule, Fixture.graph,
                source_global_dag_id=Fixture.dag.id)
            fragment.validate_against(Fixture.dag)
            self.assertEqual([record.opcode for record in fragment.core_streams[0].records],
                             [RecordOpcode.SRAM_BIND, RecordOpcode.MATMUL])
            self.assertEqual(fragment.core_streams[0].records[1].operands[-1].literal_value,
                             (1, 4, 4, 1))
            decorated = add_fixed_sram_lifecycle(fragment, Fixture.context)
            decorated.validate_against(Fixture.dag)
            self.assertEqual(len(decorated.core_streams[0].records), 4)

    def test_parameter_or_weight_extent_drift_is_rejected(self):
        action = self.actions[0]
        fragment = lower_moe_router_score_fragment(
            action, Fixture.schedule, Fixture.graph,
            source_global_dag_id=Fixture.dag.id)
        stream = fragment.core_streams[0]
        record = stream.records[1]
        forged_record = replace(record, operands=(*record.operands[:-1],
                                                  replace(record.operands[-1],
                                                          literal_value=(1, 4, 4, 2))))
        forged = CommandFragment.create(
            producer_pass=fragment.producer_pass,
            **{**fragment._semantic_key(),
               "core_streams": (replace(stream, records=(stream.records[0],
                                                        forged_record)),)},
        )
        with self.assertRaisesRegex(SchemaError, "router score MATMUL parameters"):
            forged.validate_against(Fixture.dag)
        weight = next(binding for binding in Fixture.schedule.buffer_bindings
                      if binding.id == next(use.binding_id for use in action.buffer_uses
                                            if use.role.value == "comp_input"
                                            and use.operand_index == 1))
        short_schedule = replace(Fixture.schedule, buffer_bindings=tuple(
            replace(binding, size_bytes=binding.size_bytes-2)
            if binding.id == weight.id else binding
            for binding in Fixture.schedule.buffer_bindings))
        with self.assertRaisesRegex(SchemaError, "physical FP16 extents"):
            lower_moe_router_score_fragment(
                action, short_schedule, Fixture.graph,
                source_global_dag_id=Fixture.dag.id)


if __name__ == "__main__":
    unittest.main()
