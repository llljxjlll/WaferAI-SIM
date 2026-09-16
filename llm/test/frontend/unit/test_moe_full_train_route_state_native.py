"""Real EP1 signed route HBM state to official IR2 and native LSU_LOAD."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.context import LoweringContext
from llm.frontend.wafer_frontend.lowering.state import NaiveStateDmaLowering
from llm.frontend.wafer_frontend.lowering.moe_full_train_route_freeze import lower_moe_route_freeze
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.global_action import build_global_action_dag
from llm.frontend.wafer_frontend.passes.lower_program import _decorate_fragment
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_ir1_source import (
    build_moe_ep_placed_ir1_candidate,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_route_table_source import (
    build_moe_full_train_route_table_source,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import NaiveProjectToIR2
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment, RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind
from llm.frontend.wafer_frontend.schema.persistent_state import (
    PersistentStateAccess, PersistentStateLifetime, StateKind,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeFullTrainRouteStateNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.phase, cls.sequence, cls.placement, cls.context = (
            build_single_die_moe_train_physical_source(Fixture))
        cls.route_source = build_moe_full_train_route_table_source(
            cls.phase, cls.sequence)
        cls.candidate = build_moe_ep_placed_ir1_candidate(
            cls.phase, original_dense=Fixture.dense,
            sequence=cls.sequence, placement=cls.placement,
            context=cls.context, dense_manifest=Fixture.manifest,
        )
        cls.graph = partition_ir1(IR1.create(
            producer_pass="placement", **cls.candidate.physical_ir1._semantic_key()))
        cls.projection = NaiveProjectToIR2().run(
            cls.graph, (), (), state_transfers=())
        cls.schedules = NaiveIntraDiePolicy().schedule(cls.projection, cls.graph)
        cls.dag = build_global_action_dag(cls.graph, cls.projection, cls.schedules)
        cls.lowering = LoweringContext(cls.graph, (), (), cls.projection,
                                       cls.schedules, cls.dag)

    def test_two_signed_80_byte_route_states_are_real_die0_lsu_loads(self):
        self.lowering.validate()
        states = {state.id: state for state in self.phase.graph.persistent_states}
        homes = self.placement.hbm_layout.routes
        self.assertEqual(len(homes), 2)
        self.assertEqual([home.source_seed_sha256 for home in homes],
                         [seed.payload_sha256 for seed in self.route_source.seeds])
        self.assertLessEqual(homes[0].physical_address + 80,
                             homes[1].physical_address)
        for layer, state_ref in enumerate(self.phase.route_state_refs):
            state = states[state_ref]
            self.assertEqual((state.identity.kind, state.lifetime, state.access,
                              state.shape, state.tensor_bytes),
                             (StateKind.MOE_STATIC_ROUTE,
                              PersistentStateLifetime.STEP,
                              PersistentStateAccess.READ_ONLY, (4, 5), 80))
            dma = [action for action in self.dag.actions
                   if action.task_kind is SemanticTaskKind.DMA_IN
                   and action.dma is not None
                   and action.dma.state_ref == state_ref]
            self.assertEqual(len(dma), 1)
            fragment = NaiveStateDmaLowering().lower(dma[0], self.lowering)
            record, = fragment.core_streams[0].records
            self.assertEqual((record.opcode, record.operands[1].literal_value),
                             (RecordOpcode.LSU_LOAD, 80))
            self.assertEqual((fragment.state_abi[0].kind,
                              fragment.state_abi[0].address),
                             (StateKind.MOE_STATIC_ROUTE,
                              homes[layer].physical_address))
            freeze = next(action for action in self.dag.actions
                          if action.op_kind is OpKind.MOE_ROUTE_FREEZE
                          and action.compute.workload.layer == layer)
            staging = dma[0].dma.local_value_ref
            self.assertEqual(tuple(operand.value_id for operand in
                                   freeze.compute.inputs)[1], staging)
            self.assertIn(dma[0].id, freeze.deps)

    def test_two_route_freezes_have_real_native_local_copy_and_lifecycle(self):
        for layer in range(2):
            action = next(action for action in self.dag.actions
                          if action.op_kind is OpKind.MOE_ROUTE_FREEZE
                          and action.compute.workload.layer == layer)
            fragment = lower_moe_route_freeze(action, self.lowering)
            fragment.validate_against(self.dag)
            records = fragment.core_streams[0].records
            self.assertEqual([record.opcode for record in records],
                             [RecordOpcode.DTE_ISSUE, RecordOpcode.DTE_WAIT])
            self.assertEqual(records[0].operands[3].literal_value, 80)
            self.assertEqual(records[0].operands[1].symbol_ref,
                             records[1].operands[0].symbol_ref)
            self.assertEqual(fragment.runtime_symbols[0].source_ref,
                             action.runtime_binding.token_symbol)
            decorated = _decorate_fragment(fragment, self.lowering,
                                           path="test.route_freeze", validate=True)
            decorated.validate_against(self.dag)
            self.assertEqual(sum(record.opcode is RecordOpcode.DTE_ISSUE
                                 for record in decorated.core_streams[0].records), 1)

            tampered = replace(records[0], operands=tuple(
                replace(operand, literal_value=40)
                if index == 3 else operand
                for index, operand in enumerate(records[0].operands)))
            stream = replace(fragment.core_streams[0],
                             records=(tampered, records[1]))
            forged = CommandFragment.create(
                producer_pass=fragment.producer_pass,
                **{**fragment._semantic_key(), "core_streams": (stream,)})
            with self.assertRaisesRegex(SchemaError, "payload/token"):
                forged.validate_against(self.dag)
            missing_score = CommandFragment.create(
                producer_pass=fragment.producer_pass,
                **{**fragment._semantic_key(), "buffer_abi": tuple(
                    abi for abi in fragment.buffer_abi
                    if abi.binding_id != action.buffer_uses[0].binding_id)})
            with self.assertRaisesRegex(SchemaError, "lacks exact BufferABI"):
                missing_score.validate_against(self.dag)

    def test_route_hbm_collision_and_missing_source_read_fail_closed(self):
        homes = self.placement.hbm_layout.routes
        forged_home = replace(homes[1], physical_address=homes[0].physical_address)
        with self.assertRaisesRegex(SchemaError, "overlap"):
            replace(self.placement.hbm_layout,
                    routes=(homes[0], forged_home)).validate(
                        self.phase, Fixture.manifest, self.sequence)
        with self.assertRaisesRegex(SchemaError, "source op/state removal"):
            replace(self.phase, route_state_refs=()).validate()


if __name__ == "__main__":
    unittest.main()
