from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    LoweringContext,
    NaiveStandaloneCollectiveLowering,
)
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    DirectAllGatherPolicy,
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    OperandKind,
    PlanBarrierEventPhase,
    RecordOpcode,
    RuntimeOperandField,
    RuntimeSymbolKind,
    canonical_plan_barrier_core_symbol,
    canonical_plan_barrier_event_symbol,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, OpKind
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferUseRole,
    FlowRouteRole,
    SemanticTaskKind,
    StandaloneNodeOrigin,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from test_global_action_schema import _create_global
from test_naive_intra_die import _complete_projection


def _context(tp: int):
    graph, projection = _complete_projection(tp=tp, large_sram=True)
    fusion_plans = tuple(
        NaiveInterDiePolicy().plan(graph, skeleton, graph.profile)
        for skeleton in graph.fused_op_skeletons
    )
    fused_members = {
        member_id
        for skeleton in graph.fused_op_skeletons
        for member_id in skeleton.member_node_ids
    }
    standalone_plans = tuple(
        DirectAllGatherPolicy().plan(graph, node, graph.profile)
        for node in graph.nodes
        if node.id not in fused_members
        and node.kind is OpKind.COLLECTIVE
        and node.workload.collective is CollectiveKind.ALL_GATHER
    )
    schedule_set = NaiveIntraDiePolicy().schedule(projection, graph)
    global_dag = _create_global(graph, projection, schedule_set)
    context = LoweringContext(
        graph,
        fusion_plans,
        standalone_plans,
        projection,
        schedule_set,
        global_dag,
    )
    context.validate()
    plan = standalone_plans[0]
    actions = tuple(
        action
        for action in global_dag.actions
        if isinstance(action.origin_ref, StandaloneNodeOrigin)
        and action.origin_ref.collective_plan_id == plan.id
    )
    return context, plan, actions


def _records_by_action(fragment):
    result = {}
    for stream in fragment.core_streams:
        for record in stream.records:
            result.setdefault(record.source_global_action_id, []).append(record)
    return {action_id: tuple(records) for action_id, records in result.items()}


class NaiveStandaloneCollectiveLoweringTest(unittest.TestCase):
    def test_tp2_axis0_views_emit_exact_nonzero_addends_and_reject_zero(self) -> None:
        context, _plan, actions = _context(2)
        fragment = NaiveStandaloneCollectiveLowering().lower(actions, context)
        addends = tuple(
            relocation.addend
            for stream in fragment.core_streams
            for relocation in stream.address_relocations
        )
        self.assertEqual(set(addends), {0, 8192})

        stream_index, relocation_index = next(
            (stream_index, relocation_index)
            for stream_index, stream in enumerate(fragment.core_streams)
            for relocation_index, relocation in enumerate(stream.address_relocations)
            if relocation.addend == 8192
        )
        streams = list(fragment.core_streams)
        relocations = list(streams[stream_index].address_relocations)
        relocations[relocation_index] = replace(
            relocations[relocation_index], addend=0
        )
        streams[stream_index] = replace(
            streams[stream_index], address_relocations=tuple(relocations)
        )
        forged = CommandFragment.create(
            producer_pass=fragment.producer_pass,
            **{
                **fragment._semantic_key(),
                "core_streams": tuple(streams),
            },
        )
        with self.assertRaisesRegex(SchemaError, "ActionBufferUse view"):
            forged.validate_against(context.global_dag)

    def test_tp2_tp4_counts_determinism_transit_and_public_api(self) -> None:
        lowerer = NaiveStandaloneCollectiveLowering()
        for tp, source_count, claimed_count, record_count in (
            (2, 8, 8, 12),
            (4, 36, 32, 44),
        ):
            with self.subTest(tp=tp):
                context, _plan, actions = _context(tp)
                before = (canonical_digest(context), canonical_digest(actions))
                fragment = lowerer.lower(actions, context)
                self.assertEqual(fragment, lowerer.lower(actions, context))
                self.assertEqual(
                    before,
                    (canonical_digest(context), canonical_digest(actions)),
                )
                self.assertEqual(len(actions), source_count)
                self.assertEqual(len(fragment.claimed_action_ids), claimed_count)
                self.assertEqual(
                    sum(len(stream.records) for stream in fragment.core_streams),
                    record_count,
                )
                transit_ids = {
                    action.id
                    for action in actions
                    if action.task_kind is SemanticTaskKind.TRANSIT
                }
                self.assertEqual(len(transit_ids), 0 if tp == 2 else 4)
                self.assertTrue(
                    transit_ids.isdisjoint(fragment.claimed_action_ids)
                )
                fragment.validate_against(context.global_dag)

    def test_copy_transport_routes_sync_bytes_and_buffer_abi_are_exact(self) -> None:
        context, _plan, actions = _context(4)
        fragment = NaiveStandaloneCollectiveLowering().lower(actions, context)
        records_by_action = _records_by_action(fragment)
        runtime_symbols = {
            symbol.id: symbol for symbol in fragment.runtime_symbols
        }
        program_symbols = {
            symbol.id: symbol for symbol in fragment.program_symbols
        }
        abi_bindings = {abi.binding_id for abi in fragment.buffer_abi}

        for action in actions:
            if action.task_kind is SemanticTaskKind.TRANSIT:
                self.assertNotIn(action.id, records_by_action)
                self.assertIs(action.flow_route.role, FlowRouteRole.TRANSIT)
                continue
            records = records_by_action[action.id]
            if action.task_kind is SemanticTaskKind.LOCAL_COPY:
                self.assertEqual(
                    tuple(record.opcode for record in records),
                    (RecordOpcode.DTE_ISSUE, RecordOpcode.DTE_WAIT),
                )
                issue, wait = records
                operands = {operand.name: operand for operand in issue.operands}
                self.assertEqual(operands["payload_bits"].literal_value, action.bytes * 8)
                self.assertEqual(operands["size_bytes"].literal_value, action.bytes)
                self.assertEqual(
                    operands["token"].symbol_ref,
                    wait.operands[0].symbol_ref,
                )
                token = runtime_symbols[operands["token"].symbol_ref]
                self.assertIs(token.kind, RuntimeSymbolKind.DTE_TOKEN)
                self.assertEqual(token.source_ref, action.id)
                expected_bindings = {
                    use.binding_id
                    for use in action.buffer_uses
                    if use.role
                    in (
                        BufferUseRole.LOCAL_COPY_SOURCE,
                        BufferUseRole.LOCAL_COPY_DESTINATION,
                    )
                }
                actual_bindings = {
                    program_symbols[operand.symbol_ref].source_ref
                    for operand in issue.operands
                    if operand.kind is OperandKind.ADDRESS_SYMBOL
                }
                self.assertEqual(actual_bindings, expected_bindings)
                self.assertTrue(expected_bindings.issubset(abi_bindings))
            elif action.task_kind in (
                SemanticTaskKind.SEND,
                SemanticTaskKind.RECV,
            ):
                self.assertEqual(len(records), 1)
                record = records[0]
                operands = {operand.name: operand for operand in record.operands}
                self.assertEqual(operands["completion"].literal_value, 1)
                self.assertIs(operands["token"].kind, OperandKind.LITERAL)
                self.assertEqual(operands["token"].literal_value, 0)
                self.assertEqual(operands["length_bytes"].literal_value, action.bytes)
                expected_role = (
                    FlowRouteRole.SOURCE
                    if action.task_kind is SemanticTaskKind.SEND
                    else FlowRouteRole.DESTINATION
                )
                self.assertIs(action.flow_route.role, expected_role)
                self.assertEqual(action.flow_route.flow_id, action.flow_id)
                fsm = runtime_symbols[operands["fsm_id"].symbol_ref]
                self.assertIs(fsm.kind, RuntimeSymbolKind.DTE_FSM)
                self.assertEqual(
                    fsm.source_ref, action.runtime_binding.channel_symbol
                )
                address = next(
                    operand
                    for operand in record.operands
                    if operand.kind is OperandKind.ADDRESS_SYMBOL
                )
                expected_role = (
                    BufferUseRole.SEND_SOURCE
                    if action.task_kind is SemanticTaskKind.SEND
                    else BufferUseRole.RECV_DESTINATION
                )
                binding_id = next(
                    use.binding_id
                    for use in action.buffer_uses
                    if use.role is expected_role
                )
                self.assertEqual(
                    program_symbols[address.symbol_ref].source_ref, binding_id
                )
                self.assertIn(binding_id, abi_bindings)

    def test_tp2_tp4_plan_barrier_records_and_symbols_are_canonical(self) -> None:
        for tp in (2, 4):
            with self.subTest(tp=tp):
                context, _plan, actions = _context(tp)
                fragment = NaiveStandaloneCollectiveLowering().lower(
                    actions, context
                )
                barriers = tuple(
                    action
                    for action in actions
                    if action.task_kind is SemanticTaskKind.BARRIER
                )
                contract = barriers[0].sync.barrier
                by_rank = {
                    action.origin_ref.rank: action for action in barriers
                }
                participants = tuple(
                    by_rank[rank] for rank in contract.participant_ranks
                )
                leader, *peers = participants
                records = _records_by_action(fragment)
                self.assertEqual(
                    tuple(record.opcode for record in records[leader.id]),
                    (
                        *((RecordOpcode.EVENT_WAIT,) * (tp - 1)),
                        *((RecordOpcode.EVENT_SET,) * (tp - 1)),
                    ),
                )
                for peer in peers:
                    self.assertEqual(
                        tuple(record.opcode for record in records[peer.id]),
                        (RecordOpcode.EVENT_SET, RecordOpcode.EVENT_WAIT),
                    )
                expected_core_symbols = {
                    canonical_plan_barrier_core_symbol(
                        context.global_dag.id, action
                    ).id
                    for action in participants
                }
                expected_event_symbols = {
                    *(
                        canonical_plan_barrier_event_symbol(
                            context.global_dag.id,
                            PlanBarrierEventPhase.ARRIVE,
                            peer,
                            leader,
                        ).id
                        for peer in peers
                    ),
                    *(
                        canonical_plan_barrier_event_symbol(
                            context.global_dag.id,
                            PlanBarrierEventPhase.RELEASE,
                            leader,
                            peer,
                        ).id
                        for peer in peers
                    ),
                }
                actual_core_symbols = {
                    symbol.id
                    for symbol in fragment.runtime_symbols
                    if symbol.kind is RuntimeSymbolKind.RUNTIME_CORE
                    and symbol.source_ref == contract.id
                }
                actual_event_symbols = {
                    symbol.id
                    for symbol in fragment.runtime_symbols
                    if symbol.kind is RuntimeSymbolKind.EVENT_TAG
                }
                self.assertEqual(actual_core_symbols, expected_core_symbols)
                self.assertEqual(actual_event_symbols, expected_event_symbols)
                event_relocations = tuple(
                    relocation
                    for stream in fragment.core_streams
                    for relocation in stream.runtime_relocations
                    if relocation.field is RuntimeOperandField.EVENT_TAG
                )
                self.assertEqual(len(event_relocations), 4 * (tp - 1))

    def test_rejects_reordered_missing_mixed_and_forged_actions(self) -> None:
        context, _plan, actions = _context(4)
        lowerer = NaiveStandaloneCollectiveLowering()
        cases = (
            tuple(reversed(actions)),
            tuple(
                action
                for action in actions
                if action.task_kind is not SemanticTaskKind.TRANSIT
            ),
            (replace(actions[0], bytes=actions[0].bytes + 2), *actions[1:]),
        )
        for changed in cases:
            with self.subTest(size=len(changed)), self.assertRaisesRegex(
                SchemaError, "exactly preserve"
            ):
                lowerer.lower(changed, context)

        other_plan_id = context.standalone_plans[1].id
        other = next(
            action
            for action in context.global_dag.actions
            if isinstance(action.origin_ref, StandaloneNodeOrigin)
            and action.origin_ref.collective_plan_id == other_plan_id
        )
        with self.assertRaisesRegex(SchemaError, "one standalone"):
            lowerer.lower((*actions, other), context)


if __name__ == "__main__":
    unittest.main()
