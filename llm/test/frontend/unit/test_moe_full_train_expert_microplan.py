"""Real EP1 expert native operation plan and BufferABI rejection tests."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_ir1_source import (
    build_moe_ep_placed_ir1_candidate,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_expert_microplan import (
    plan_moe_expert_native_forward,
)
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.global_action import build_global_action_dag
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import NaiveProjectToIR2
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import BufferOwnership
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeFullTrainExpertMicroplanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        phase, sequence, placement, context = build_single_die_moe_train_physical_source(Fixture)
        source = build_moe_ep_placed_ir1_candidate(
            phase, original_dense=Fixture.dense, sequence=sequence,
            placement=placement, context=context, dense_manifest=Fixture.manifest,
        )
        graph = partition_ir1(IR1.create(
            producer_pass="placement", **source.physical_ir1._semantic_key(),
        ))
        projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
        schedule_set = NaiveIntraDiePolicy().schedule(projection, graph)
        cls.schedule = schedule_set.schedules[0]
        cls.graph = graph
        dag = build_global_action_dag(graph, projection, schedule_set)
        cls.actions = tuple(action for action in dag.actions
                            if action.op_kind is OpKind.MOE_EXPERT_FORWARD)

    def test_both_real_layers_need_exact_four_native_operations_and_distinct_scratch(self):
        self.assertEqual(len(self.actions), 2)
        plans = tuple(plan_moe_expert_native_forward(action, self.schedule, self.graph)
                      for action in self.actions)
        self.assertNotEqual(plans[0].id, plans[1].id)
        for plan in plans:
            self.assertEqual([op.opcode for op in plan.operations],
                             [RecordOpcode.MATMUL, RecordOpcode.MATMUL,
                              RecordOpcode.SWIGLU, RecordOpcode.MATMUL])
            self.assertEqual([op.parameters for op in plan.operations],
                             [(1, 4, 4, 8), (1, 4, 4, 8), (32,), (1, 4, 8, 4)])
            self.assertEqual((plan.concat_scratch_bytes, plan.activated_scratch_bytes),
                             (128, 64))
            self.assertEqual(plan.source_ir1_ref, self.graph.id)
            self.assertIs(plan.scratch_ownership, BufferOwnership.OWNED)
            self.assertEqual(plan.core_order_index,
                             self.actions[plans.index(plan)].core_order_index)
            self.assertEqual(plan.concat_region_offset_bytes % 64, 0)
            self.assertEqual(plan.activated_region_offset_bytes % 64, 0)
            self.assertGreaterEqual(plan.activated_region_offset_bytes,
                                    plan.concat_region_offset_bytes + plan.concat_scratch_bytes)
            scratch_start = plan.concat_region_offset_bytes
            scratch_end = plan.activated_region_offset_bytes + plan.activated_scratch_bytes
            for binding in self.schedule.buffer_bindings:
                if (binding.core_id == plan.runtime_core_id
                        and binding.region_ref == plan.scratch_region_ref):
                    self.assertTrue(binding.region_offset_bytes + binding.size_bytes <= scratch_start
                                    or binding.region_offset_bytes >= scratch_end)
            self.assertEqual([op.output_offset_bytes for op in plan.operations],
                             [0, 64, 0, 0])
            self.assertEqual(plan.operations[0].output_ref,
                             plan.operations[1].output_ref)
            self.assertNotEqual(plan.operations[0].output_ref,
                                plan.operations[2].output_ref)
            self.assertEqual(plan.operations[2].output_ref,
                             plan.operations[3].input_ref)
            self.assertEqual(plan.operations[3].output_ref,
                             next(use.binding_id for use in self.actions[plans.index(plan)].buffer_uses
                                  if use.role.value == "comp_output"))

    def test_short_weight_and_scratch_alias_fail_closed(self):
        action = self.actions[0]
        weight_use = next(use for use in action.buffer_uses
                          if use.role.value == "comp_input" and use.operand_index == 2)
        bindings = self.schedule.buffer_bindings
        short = tuple(replace(binding, size_bytes=binding.size_bytes-2)
                      if binding.id == weight_use.binding_id else binding
                      for binding in bindings)
        with self.assertRaisesRegex(SchemaError, "exact FP16 projection extents"):
            plan_moe_expert_native_forward(action,
                                           replace(self.schedule, buffer_bindings=short), self.graph)
        activation = next(binding for binding in bindings
                          if binding.id == next(use.binding_id for use in action.buffer_uses
                                                if use.role.value == "comp_input"
                                                and use.operand_index == 0))
        aliased = tuple(replace(binding, storage_id=activation.storage_id)
                        if binding.id == weight_use.binding_id else binding
                        for binding in bindings)
        with self.assertRaisesRegex(SchemaError, "cannot alias"):
            plan_moe_expert_native_forward(action,
                                           replace(self.schedule, buffer_bindings=aliased), self.graph)

    def test_forged_action_or_schedule_is_rejected(self):
        action = self.actions[0]
        with self.assertRaisesRegex(SchemaError, "scheduled physical MoE expert"):
            plan_moe_expert_native_forward(replace(action, op_kind=OpKind.GEMM),
                                           self.schedule, self.graph)
        with self.assertRaisesRegex(SchemaError, "scheduled physical MoE expert"):
            plan_moe_expert_native_forward(action,
                                           replace(self.schedule, die_id=1), self.graph)

    def test_scratch_does_not_silently_overflow_region_or_isa_address(self):
        action = self.actions[0]
        plan = plan_moe_expert_native_forward(action, self.schedule, self.graph)
        binding = next(binding for binding in self.schedule.buffer_bindings
                       if binding.core_id == plan.runtime_core_id
                       and binding.region_ref == plan.scratch_region_ref)
        oversized = tuple(replace(candidate, region_offset_bytes=(1 << 16))
                          if candidate.id == binding.id else candidate
                          for candidate in self.schedule.buffer_bindings)
        with self.assertRaisesRegex(SchemaError, "exceed physical SRAM"):
            plan_moe_expert_native_forward(
                action, replace(self.schedule, buffer_bindings=oversized), self.graph)


if __name__ == "__main__":
    unittest.main()
