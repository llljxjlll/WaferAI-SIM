"""EP2 expert reverse must retain real source tensors and physical owners."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_expert_reverse_ir1_source import (
    build_moe_ep2_expert_reverse_ir1_candidate,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_placement import (
    build_moe_full_train_ep_placement,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_ir0 import (
    build_moe_full_train_forward_ir0,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
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
