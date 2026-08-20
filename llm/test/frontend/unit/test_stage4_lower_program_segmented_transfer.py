from __future__ import annotations

from collections import Counter
from dataclasses import replace
from functools import lru_cache
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    LoweringContext,
    NaiveStateTransferLowering,
)
from llm.frontend.wafer_frontend.passes.global_action_dag import (
    build_stage4_global_action,
)
from llm.frontend.wafer_frontend.passes.link_program import link_stage4
from llm.frontend.wafer_frontend.passes.lower_program import lower_stage4
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    RecordOpcode,
    RuntimeOperandField,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.global_action import GlobalActionDAG
from llm.frontend.wafer_frontend.schema.ir2 import (
    SemanticTaskKind,
    StateTransferOrigin,
)

from test_stage4_global_action_carrier import _scheduled_pdr


@lru_cache(maxsize=1)
def _pipeline():
    source = build_stage4_global_action(_scheduled_pdr())
    lowered = lower_stage4(source)
    linked = link_stage4(lowered)
    return source, lowered, linked


def _transfer_fragments(lowered):
    return tuple(
        fragment
        for fragment in lowered.fragments
        if type(fragment) is CommandFragment
        and fragment.kind is FragmentKind.STATE_TRANSFER
    )


def _recreate(fragment: CommandFragment, **changes: object) -> CommandFragment:
    fields = fragment._semantic_key()
    fields.update(changes)
    return CommandFragment.create(
        producer_pass=fragment.producer_pass,
        **fields,
    )


class Stage4LowerProgramSegmentedTransferTest(unittest.TestCase):
    def test_pdr_tp2_to_tp1_lowers_and_links_exact_segment_units(self) -> None:
        source, lowered, linked = _pipeline()
        lowered.validate_against(source)
        linked.validate_against(lowered)
        contracts = source.projection.state_transfers
        fragments = _transfer_fragments(lowered)
        actions = {action.id: action for action in source.global_dag.actions}
        records = tuple(
            record
            for fragment in fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        transport_records = tuple(
            record
            for record in records
            if record.opcode in (RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV)
        )

        self.assertEqual((len(contracts), sum(len(item.segments) for item in contracts)), (8, 64))
        self.assertEqual((len(lowered.fragments), len(fragments)), (152, 16))
        self.assertEqual(sum(len(item.claimed_action_ids) for item in fragments), 192)
        self.assertEqual(len(records), 314)
        self.assertEqual(
            Counter(record.opcode for record in records),
            Counter(
                {
                    RecordOpcode.DTE_SEND: 64,
                    RecordOpcode.DTE_RECV: 64,
                    RecordOpcode.DTE_WAIT: 64,
                    RecordOpcode.EVENT_SET: 55,
                    RecordOpcode.EVENT_WAIT: 55,
                    RecordOpcode.SRAM_ALLOC_AT: 12,
                }
            ),
        )
        self.assertEqual(
            (
                sum(len(stream.address_relocations) for item in fragments for stream in item.core_streams),
                sum(len(stream.runtime_relocations) for item in fragments for stream in item.core_streams),
                sum(len(item.runtime_symbols) for item in fragments),
                sum(len(item.program_symbols) for item in fragments),
                sum(len(item.buffer_abi) for item in fragments),
            ),
            (152, 714, 650, 40, 16),
        )
        self.assertEqual(
            (
                len(linked.manifest.fragments),
                len(linked.manifest.address_operand_bindings),
                len(linked.manifest.runtime_symbol_definitions),
                len(linked.manifest.program_symbol_definitions),
            ),
            (152, 1210, 504, 432),
        )
        self.assertEqual(
            Counter(
                next(
                    operand.literal_value
                    for operand in record.operands
                    if operand.name == "length_bytes"
                )
                for record in transport_records
            ),
            Counter({16: 128}),
        )
        self.assertEqual(
            sum(
                next(
                    operand.literal_value
                    for operand in record.operands
                    if operand.name == "length_bytes"
                )
                for record in transport_records
            ),
            2048,
        )

        by_contract: dict[str, list[CommandFragment]] = {}
        source_addends = []
        destination_addends = []
        for fragment in fragments:
            self.assertEqual((len(fragment.core_streams), len(fragment.buffer_abi), fragment.state_abi), (1, 1, ()))
            stream = fragment.core_streams[0]
            claimed = tuple(actions[action_id] for action_id in fragment.claimed_action_ids)
            transfer_refs = {
                action.origin_ref.state_transfer_ref
                for action in claimed
                if isinstance(action.origin_ref, StateTransferOrigin)
            }
            self.assertEqual(len(transfer_refs), 1)
            transfer_ref = next(iter(transfer_refs))
            by_contract.setdefault(transfer_ref, []).append(fragment)
            payload = tuple(
                record
                for record in stream.records
                if record.opcode
                in (RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV, RecordOpcode.DTE_WAIT)
            )
            payload_semantics = tuple(
                (
                    actions[record.source_global_action_id].origin_ref.segment_index,
                    record.opcode,
                )
                for record in payload
            )
            is_source = any(record.opcode is RecordOpcode.DTE_SEND for record in payload)
            expected = (
                tuple((index, RecordOpcode.DTE_SEND) for index in range(8))
                if is_source
                else tuple(
                    item
                    for index in range(8)
                    for item in (
                        (index, RecordOpcode.DTE_RECV),
                        (index, RecordOpcode.DTE_WAIT),
                    )
                )
            )
            self.assertEqual(payload_semantics, expected)
            address_addends = tuple(
                relocation.addend
                for relocation in stream.address_relocations
                if relocation.operand_id
                in (
                    SemanticOperandId.SOURCE_ADDRESS,
                    SemanticOperandId.DESTINATION_ADDRESS,
                )
            )
            (source_addends if is_source else destination_addends).append(address_addends)
            fsm_refs = {
                relocation.symbol_ref
                for relocation in stream.runtime_relocations
                if relocation.field is RuntimeOperandField.DTE_FSM
            }
            peer_refs = {
                relocation.symbol_ref
                for relocation in stream.runtime_relocations
                if relocation.field is RuntimeOperandField.PEER_CORE
            }
            self.assertEqual((len(fsm_refs), len(peer_refs)), (8, 8))
            if not is_source:
                token_refs = tuple(
                    relocation.symbol_ref
                    for relocation in stream.runtime_relocations
                    if relocation.field is RuntimeOperandField.DTE_TOKEN
                )
                self.assertEqual((len(token_refs), len(set(token_refs))), (16, 8))
                self.assertTrue(
                    all(
                        token_refs[index] == token_refs[index + 1]
                        for index in range(0, 16, 2)
                    )
                )
        self.assertEqual(set(by_contract), {contract.id for contract in contracts})
        self.assertTrue(all(len(items) == 2 for items in by_contract.values()))
        self.assertEqual(
            Counter(source_addends),
            Counter({(0, 16, 32, 48, 64, 80, 96, 112): 8}),
        )
        self.assertEqual(
            Counter(destination_addends),
            Counter(
                {
                    (0, 32, 64, 96, 128, 160, 192, 224): 4,
                    (16, 48, 80, 112, 144, 176, 208, 240): 4,
                }
            ),
        )

    def test_missing_reorder_address_token_and_segment_tamper_fail_closed(self) -> None:
        source, lowered, _linked = _pipeline()
        context = LoweringContext(
            source.graph,
            source.fusion_plans,
            source.standalone_plans,
            source.projection,
            source.schedule_set,
            source.global_dag,
        )
        fragments = _transfer_fragments(lowered)
        actions = {action.id: action for action in source.global_dag.actions}
        source_fragment = next(
            item
            for item in fragments
            if any(
                record.opcode is RecordOpcode.DTE_SEND
                for record in item.core_streams[0].records
            )
        )
        source_actions = tuple(
            action
            for action in source.global_dag.actions
            if action.id in source_fragment.claimed_action_ids
        )
        lowerer = NaiveStateTransferLowering()
        with self.assertRaisesRegex(SchemaError, "exactly preserve"):
            lowerer.lower(source_actions[:-1], context)
        with self.assertRaisesRegex(SchemaError, "exactly preserve"):
            lowerer.lower(tuple(reversed(source_actions)), context)

        source_stream = source_fragment.core_streams[0]
        relocation_index = next(
            index
            for index, relocation in enumerate(source_stream.address_relocations)
            if relocation.operand_id is SemanticOperandId.SOURCE_ADDRESS
        )
        bad_relocations = list(source_stream.address_relocations)
        bad_relocations[relocation_index] = replace(
            bad_relocations[relocation_index],
            addend=bad_relocations[relocation_index].addend + 2,
        )
        bad_address = _recreate(
            source_fragment,
            core_streams=(
                replace(
                    source_stream,
                    address_relocations=tuple(bad_relocations),
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "addend"):
            bad_address.validate_against(source.global_dag)

        destination_fragment = next(
            item
            for item in fragments
            if item.core_streams[0].records[0].opcode is RecordOpcode.DTE_RECV
        )
        destination_stream = destination_fragment.core_streams[0]
        recv_indices = tuple(
            index
            for index, record in enumerate(destination_stream.records)
            if record.opcode is RecordOpcode.DTE_RECV
        )
        wait_indices = tuple(
            index
            for index, record in enumerate(destination_stream.records)
            if record.opcode is RecordOpcode.DTE_WAIT
        )
        token_by_record = {
            relocation.record_index: relocation.symbol_ref
            for relocation in destination_stream.runtime_relocations
            if relocation.field is RuntimeOperandField.DTE_TOKEN
        }
        wrong_token = token_by_record[recv_indices[1]]
        bad_records = list(destination_stream.records)
        bad_records[wait_indices[0]] = replace(
            bad_records[wait_indices[0]],
            operands=(
                replace(
                    bad_records[wait_indices[0]].operands[0],
                    symbol_ref=wrong_token,
                ),
            ),
        )
        bad_runtime = tuple(
            replace(relocation, symbol_ref=wrong_token)
            if relocation.record_index == wait_indices[0]
            and relocation.field is RuntimeOperandField.DTE_TOKEN
            else relocation
            for relocation in destination_stream.runtime_relocations
        )
        bad_token = _recreate(
            destination_fragment,
            core_streams=(
                replace(
                    destination_stream,
                    records=tuple(bad_records),
                    runtime_relocations=bad_runtime,
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "token"):
            bad_token.validate_against(source.global_dag)

        remap = {recv_indices[0]: recv_indices[1], wait_indices[0]: wait_indices[1], recv_indices[1]: recv_indices[0], wait_indices[1]: wait_indices[0]}
        reordered_records = list(destination_stream.records)
        reordered_records[0:4] = reordered_records[2:4] + reordered_records[0:2]
        reordered_runtime = tuple(
            sorted(
                (
                    replace(
                        relocation,
                        record_index=remap.get(relocation.record_index, relocation.record_index),
                    )
                    for relocation in destination_stream.runtime_relocations
                ),
                key=lambda relocation: (
                    relocation.record_index,
                    tuple(RuntimeOperandField).index(relocation.field),
                ),
            )
        )
        reordered_address = tuple(
            sorted(
                (
                    replace(
                        relocation,
                        record_index=remap.get(relocation.record_index, relocation.record_index),
                    )
                    for relocation in destination_stream.address_relocations
                ),
                key=lambda relocation: (
                    relocation.record_index,
                    int(relocation.operand_id),
                ),
            )
        )
        bad_order = _recreate(
            destination_fragment,
            core_streams=(
                replace(
                    destination_stream,
                    records=tuple(reordered_records),
                    runtime_relocations=reordered_runtime,
                    address_relocations=reordered_address,
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "order|runtime relocation"):
            bad_order.validate_against(source.global_dag)

        first_send = next(
            action
            for action in source_actions
            if isinstance(action.origin_ref, StateTransferOrigin)
            and action.origin_ref.segment_index == 0
            and action.task_kind is SemanticTaskKind.SEND
        )
        forged_send = replace(
            first_send,
            origin_ref=replace(first_send.origin_ref, segment_index=1),
        )
        dag_fields = source.global_dag._semantic_key()
        dag_fields["actions"] = tuple(
            forged_send if action.id == first_send.id else action
            for action in source.global_dag.actions
        )
        forged_dag = GlobalActionDAG.create(
            producer_pass=source.global_dag.producer_pass,
            **dag_fields,
        )
        forged_fragment = _recreate(
            source_fragment,
            source_global_dag_id=forged_dag.id,
        )
        with self.assertRaisesRegex(SchemaError, "contiguous segmented"):
            forged_fragment.validate_against(forged_dag)


if __name__ == "__main__":
    unittest.main()
