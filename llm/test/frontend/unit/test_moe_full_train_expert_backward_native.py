"""Physical EP1 expert reverse rejects forged scratch views and bind arity."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.context import LoweringContext
from llm.frontend.wafer_frontend.lowering.lifecycle import add_fixed_sram_lifecycle
from llm.frontend.wafer_frontend.lowering.moe_full_train_expert_backward import (
    lower_moe_expert_backward_fragment,
)
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.global_action import build_global_action_dag
from llm.frontend.wafer_frontend.passes.moe_full_train_ce_backward_ir0 import append_moe_full_train_ce_backward_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_head_backward_ir0 import append_moe_full_train_head_backward_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_shared_reverse_ir0 import append_moe_full_train_shared_reverse_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_combine_backward_ir0 import append_moe_full_train_combine_backward_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_router_wgrad_ir0 import append_moe_full_train_router_wgrad_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_expert_backward_ir0 import append_moe_full_train_expert_backward_ir0
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_ir1_source import build_moe_ep_placed_ir1_candidate
from llm.frontend.wafer_frontend.passes.placement import _physical_node
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import NaiveProjectToIR2
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment, RecordOpcode, RecordOperand, SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeExpertBackwardNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        phase, sequence, placement, physical = build_single_die_moe_train_physical_source(Fixture)
        source = append_moe_full_train_expert_backward_ir0(
            append_moe_full_train_router_wgrad_ir0(
                append_moe_full_train_combine_backward_ir0(
                    append_moe_full_train_shared_reverse_ir0(
                        append_moe_full_train_head_backward_ir0(
                            append_moe_full_train_ce_backward_ir0(phase))))))
        base = build_moe_ep_placed_ir1_candidate(
            phase, original_dense=Fixture.dense, sequence=sequence,
            placement=placement, context=physical,
            dense_manifest=Fixture.manifest,
        ).physical_ir1
        reverse = source.nodes[len(phase.graph.nodes):]
        instance = replace(base.instances[0], node_ids=(
            *base.instances[0].node_ids, *(node.id for node in reverse)))
        ir1 = IR1.create(
            producer_pass="placement", source_ir0_id=source.id,
            profile=source.profile, fabric=physical.fabric,
            instances=(instance,), groups=base.groups,
            nodes=(*base.nodes, *(_physical_node(node, base.groups[0].id)
                                  for node in reverse)),
            values=source.values, edges=source.edges,
            fusion_candidates=source.fusion_candidates,
            state_accesses=source.state_accesses,
            persistent_state_manifest=base.persistent_state_manifest,
            instance_profiles=source.instance_profiles,
            node_profiles=source.node_profiles, pd_plan_id=source.pd_plan_id,
        )
        graph = partition_ir1(ir1)
        projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
        schedules = NaiveIntraDiePolicy().schedule(projection, graph)
        dag = build_global_action_dag(graph, projection, schedules)
        action = next(item for item in dag.actions
                      if item.op_kind is OpKind.MOE_EXPERT_BACKWARD)
        schedule = next(item for item in schedules.schedules
                        if item.id == action.source.schedule_id)
        cls.dag = dag
        cls.context = LoweringContext(graph, (), (), projection, schedules, dag)
        cls.fragment = lower_moe_expert_backward_fragment(
            action, schedule, graph, source_global_dag_id=dag.id)

    def test_exact_owned_five_scratch_and_twenty_one_physical_records(self):
        fragment = self.fragment
        fragment.validate_against(self.dag)
        records = fragment.core_streams[0].records
        self.assertEqual(len(records), 21)
        self.assertEqual(sum(item.opcode is RecordOpcode.SRAM_BIND for item in records), 10)
        self.assertIs(records[-1].opcode, RecordOpcode.LOCAL_REDUCE)
        self.assertEqual(sum(item.opcode is RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING
                             for item in records), 3)
        self.assertEqual(sum(item.opcode is RecordOpcode.GEMM_DX_TIMING
                             for item in records), 3)
        self.assertEqual(sum(':backward_' in abi.value_id for abi in fragment.buffer_abi), 5)
        add_fixed_sram_lifecycle(fragment, self.context).validate_against(self.dag)

    def test_second_projection_offset_and_second_input_bind_fail_closed(self):
        stream = self.fragment.core_streams[0]
        offset = next(item for item in stream.address_relocations
                      if item.record_index == 3
                      and item.operand_id is SemanticOperandId.COMPUTE_OUTPUT_ADDRESS)
        self.assertGreater(offset.addend, 0)
        forged = replace(stream, address_relocations=tuple(
            replace(item, addend=0) if item == offset else item
            for item in stream.address_relocations))
        candidate = CommandFragment.create(
            producer_pass=self.fragment.producer_pass,
            **{**self.fragment._semantic_key(), 'core_streams': (forged,)})
        with self.assertRaisesRegex(SchemaError, 'wrong source/tape'):
            candidate.validate_against(self.dag)
        bind = stream.records[6]
        self.assertIs(bind.opcode, RecordOpcode.SRAM_BIND)
        self.assertEqual(bind.operands[0].literal_value, 2)
        bad_bind = replace(bind, operands=(RecordOperand.literal('input_count', 1),
                                           *bind.operands[1:]))
        forged = replace(stream, records=tuple(
            bad_bind if index == 6 else record
            for index, record in enumerate(stream.records)))
        candidate = CommandFragment.create(
            producer_pass=self.fragment.producer_pass,
            **{**self.fragment._semantic_key(), 'core_streams': (forged,)})
        with self.assertRaises(SchemaError):
            candidate.validate_against(self.dag)


if __name__ == '__main__':
    unittest.main()
