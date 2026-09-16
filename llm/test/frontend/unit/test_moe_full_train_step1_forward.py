"""Step1 MoE forward must consume P2 v1 parameter and route sources."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_ir0 import (
    build_moe_full_train_forward_ir0,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_forward_validator import (
    MoeFullTrainForwardValidator,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_route_table_source import (
    build_moe_full_train_route_table_source,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeFullTrainStep1ForwardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.zero, cls.sequence, _, _ = build_single_die_moe_train_physical_source(
            Fixture, step=0,
        )
        cls.one, _, cls.placement, cls.context = (
            build_single_die_moe_train_physical_source(Fixture, step=1)
        )

    def test_step1_consumes_sgd_version1_and_distinct_frozen_routes(self):
        self.assertNotEqual(self.zero.graph.id, self.one.graph.id)
        self.assertEqual(self.one.step, 1)
        states = {state.id: state for state in
                  self.sequence.materialization.logical_graph.state_versions}
        self.assertEqual({states[owner.source_e2e_state_ref].version
                          for owner in self.one.ep_state_owners}, {1})
        self.assertEqual({states[owner.source_e2e_state_ref].version
                          for owner in self.zero.ep_state_owners}, {0})
        self.assertNotEqual(
            build_moe_full_train_route_table_source(
                self.zero, self.sequence,
            ).id,
            build_moe_full_train_route_table_source(
                self.one, self.sequence,
            ).id,
        )
        self.one.validate_against(Fixture.dense, self.sequence)
        MoeFullTrainForwardValidator.validate(
            self.one, original_dense=Fixture.dense, sequence=self.sequence,
        )
        self.placement.validate(
            self.one, Fixture.dense, Fixture.manifest,
            self.sequence, self.context,
        )

    def test_step1_rejects_replay_of_version0_or_step0_route(self):
        forged_owner = replace(
            self.one.ep_state_owners[0],
            source_e2e_state_ref=self.zero.ep_state_owners[0].source_e2e_state_ref,
        )
        with self.assertRaises(SchemaError):
            replace(self.one, ep_state_owners=(forged_owner,
                    *self.one.ep_state_owners[1:])).validate_against(
                        Fixture.dense, self.sequence,
                    )
        with self.assertRaises(SchemaError):
            replace(self.one, source_route_trace_refs=
                    self.zero.source_route_trace_refs).validate_against(
                        Fixture.dense, self.sequence,
                    )
        with self.assertRaises(SchemaError):
            build_moe_full_train_forward_ir0(Fixture.dense, self.sequence, step=2)


if __name__ == "__main__":
    unittest.main()
