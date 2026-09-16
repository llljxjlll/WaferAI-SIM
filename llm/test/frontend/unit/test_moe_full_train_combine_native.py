"""Source-bound score, native weighted combine and full EP1 forward link."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.lifecycle import add_fixed_sram_lifecycle
from llm.frontend.wafer_frontend.lowering.linker import NaiveManifestLinker
from llm.frontend.wafer_frontend.lowering.moe_full_train_combine import (
    lower_moe_weighted_combine,
)
from llm.frontend.wafer_frontend.passes.lower_program import (
    _lower_fragments, _resolve_dependencies,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment, RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.test.frontend.unit.test_moe_full_train_expert_microplan import (
    MoeFullTrainExpertMicroplanTest as Fixture,
)


class MoeFullTrainCombineNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.actions = tuple(action for action in Fixture.dag.actions
                            if action.op_kind is OpKind.MOE_COMBINE)

    def test_both_layers_consume_dynamic_score_and_native_weighted_opcode(self):
        self.assertEqual(len(self.actions), 2)
        for action in self.actions:
            self.assertEqual([use.tensor_slice.shape for use in action.buffer_uses],
                             [(4, 4), (4, 5), (4, 1), (4, 4)])
            fragment = lower_moe_weighted_combine(action, Fixture.context)
            fragment.validate_against(Fixture.dag)
            self.assertEqual([record.opcode for record in fragment.core_streams[0].records],
                             [RecordOpcode.SRAM_BIND,
                              RecordOpcode.MOE_SCORE_WEIGHTED_FORWARD])
            decorated = add_fixed_sram_lifecycle(fragment, Fixture.context)
            decorated.validate_against(Fixture.dag)
            self.assertEqual(len(decorated.core_streams[0].records), 6)

    def test_wrong_expert_count_literal_is_rejected(self):
        fragment = lower_moe_weighted_combine(self.actions[0], Fixture.context)
        stream = fragment.core_streams[0]
        weighted = stream.records[1]
        forged_weighted = replace(weighted, operands=(
            *weighted.operands[:-2],
            replace(weighted.operands[-2], literal_value=2),
            weighted.operands[-1],
        ))
        forged = CommandFragment.create(
            producer_pass=fragment.producer_pass,
            **{**fragment._semantic_key(),
               "core_streams": (replace(stream, records=(stream.records[0],
                                                        forged_weighted)),)},
        )
        with self.assertRaises(SchemaError):
            forged.validate_against(Fixture.dag)

    def test_entire_two_layer_forward_lowers_and_links(self):
        leaves = _lower_fragments(Fixture.context,
                                  _resolve_dependencies(None, None, None, None, None))
        self.assertEqual(len(leaves), 51)
        linked = NaiveManifestLinker().link(Fixture.context, leaves)
        linked.validate_against(
            Fixture.context.ir1, Fixture.context.fusion_plans,
            Fixture.context.standalone_plans, Fixture.context.projection,
            Fixture.context.schedule_set, Fixture.context.global_dag,
            linked.fragments,
        )
        self.assertEqual(len(linked.core_streams), 1)
        self.assertEqual(sum(len(stream.records) for stream in linked.core_streams),
                         sum(len(stream.records) for leaf in leaves
                             for stream in leaf.core_streams))


if __name__ == "__main__":
    unittest.main()
