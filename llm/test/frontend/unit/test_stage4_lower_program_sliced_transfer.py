from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import NaiveManifestLinker
from llm.frontend.wafer_frontend.passes.global_action_dag import (
    build_stage4_global_action,
)
from llm.frontend.wafer_frontend.passes.link_program import link_stage4
from llm.frontend.wafer_frontend.passes.lower_program import lower_stage4
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    CommandFragment,
    FragmentKind,
    LinkedProgramManifest,
    ProgramControlEnvelope,
    RecordOpcode,
    RuntimeSymbolKind,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    SemanticTaskKind,
    StateTransferOrigin,
)
from llm.frontend.wafer_frontend.schema.n5 import Stage4GlobalAction

from test_stage4_global_action_carrier import _scheduled


def _profile() -> Stage4GlobalAction:
    return build_stage4_global_action(_scheduled(fused=False))


class Stage4LowerProgramSlicedTransferTest(unittest.TestCase):
    def test_tp1_cross_group_endpoints_lower_exactly_once(self) -> None:
        profile = _profile()
        lowered = lower_stage4(profile)
        lowered.validate_against(profile)
        fragments = tuple(
            fragment
            for fragment in lowered.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_TRANSFER
        )
        self.assertEqual(len(lowered.fragments), 96)
        self.assertEqual(len(fragments), 8)
        self.assertEqual(
            sum(
                len(stream.records)
                for fragment in fragments
                for stream in fragment.core_streams
            ),
            20,
        )
        self.assertEqual(
            sum(len(fragment.claimed_action_ids) for fragment in fragments),
            12,
        )
        actions = {action.id: action for action in profile.global_dag.actions}
        by_transfer: dict[str, list[CommandFragment]] = {}
        for fragment in fragments:
            origin = actions[fragment.claimed_action_ids[0]].origin_ref
            self.assertIsInstance(origin, StateTransferOrigin)
            assert isinstance(origin, StateTransferOrigin)
            by_transfer.setdefault(origin.state_transfer_ref, []).append(fragment)
            self.assertEqual(len(fragment.buffer_abi), 1)
            self.assertEqual(fragment.state_abi, ())
        self.assertEqual(
            set(by_transfer),
            {item.id for item in profile.projection.state_transfers},
        )
        self.assertTrue(all(len(items) == 2 for items in by_transfer.values()))
        for items in by_transfer.values():
            records = {
                tuple(
                    record.opcode
                    for stream in fragment.core_streams
                    for record in stream.records
                )
                for fragment in items
            }
            self.assertEqual(
                records,
                {
                    (RecordOpcode.SRAM_ALLOC_AT, RecordOpcode.DTE_SEND),
                    (
                        RecordOpcode.SRAM_ALLOC_AT,
                        RecordOpcode.DTE_RECV,
                        RecordOpcode.DTE_WAIT,
                    ),
                },
            )
            claimed_kinds = {
                frozenset(
                    actions[action_id].task_kind
                    for action_id in fragment.claimed_action_ids
                )
                for fragment in items
            }
            self.assertEqual(
                claimed_kinds,
                {
                    frozenset((SemanticTaskKind.SEND,)),
                    frozenset(
                        (SemanticTaskKind.RECV, SemanticTaskKind.WAIT)
                    ),
                },
            )

    def test_tp1_linker_closes_transfer_addresses_runtime_and_control(self) -> None:
        profile = _profile()
        lowered = lower_stage4(profile)
        linked = link_stage4(lowered)
        context = lowered.lowering_context
        manifest = linked.manifest
        manifest.validate_against(
            context.ir1,
            context.fusion_plans,
            context.standalone_plans,
            context.projection,
            context.schedule_set,
            context.global_dag,
            manifest.fragments,
        )
        self.assertEqual(
            (
                len(manifest.fragments),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
                len(manifest.fragment_interfaces),
                len(manifest.core_streams),
            ),
            (96, 564, 38, 18, 219, 96, 2),
        )
        self.assertIsInstance(manifest.envelope, ProgramControlEnvelope)
        self.assertEqual(len(manifest.envelope.active_cores), 2)
        self.assertEqual(len(manifest.envelope.start_events), 2)
        self.assertEqual(
            manifest.envelope.terminal_cores,
            manifest.envelope.active_cores,
        )
        self.assertEqual(
            manifest.envelope.expected_ack_cores,
            manifest.envelope.active_cores,
        )
        self.assertEqual(
            manifest.envelope.expected_done_cores,
            manifest.envelope.active_cores,
        )

        actions = {action.id: action for action in profile.global_dag.actions}
        transfer_fragments = tuple(
            fragment
            for fragment in lowered.fragments
            if type(fragment) is CommandFragment
            and fragment.kind is FragmentKind.STATE_TRANSFER
        )
        transfer_ids = {fragment.id for fragment in transfer_fragments}
        self.assertEqual(
            sum(
                binding.fragment_id in transfer_ids
                for binding in manifest.address_operand_bindings
            ),
            24,
        )
        self.assertTrue(
            all(
                binding.fragment_id not in transfer_ids
                for binding in manifest.state_operand_bindings
            )
        )

        runtime_definitions = manifest.runtime_symbol_definitions
        for contract in profile.projection.state_transfers:
            transfer_actions = tuple(
                action
                for action in actions.values()
                if isinstance(action.origin_ref, StateTransferOrigin)
                and action.origin_ref.state_transfer_ref == contract.id
            )
            send = next(
                action
                for action in transfer_actions
                if action.task_kind is SemanticTaskKind.SEND
            )
            recv = next(
                action
                for action in transfer_actions
                if action.task_kind is SemanticTaskKind.RECV
            )
            wait = next(
                action
                for action in transfer_actions
                if action.task_kind is SemanticTaskKind.WAIT
            )
            send_fragment = next(
                fragment
                for fragment in transfer_fragments
                if send.id in fragment.claimed_action_ids
            )
            destination_fragment = next(
                fragment
                for fragment in transfer_fragments
                if recv.id in fragment.claimed_action_ids
            )
            for action, fragment, opcode, operand_id in (
                (
                    send,
                    send_fragment,
                    RecordOpcode.DTE_SEND,
                    SemanticOperandId.SOURCE_ADDRESS,
                ),
                (
                    recv,
                    destination_fragment,
                    RecordOpcode.DTE_RECV,
                    SemanticOperandId.DESTINATION_ADDRESS,
                ),
            ):
                stream = fragment.core_streams[0]
                record_index = next(
                    index
                    for index, record in enumerate(stream.records)
                    if record.opcode is opcode
                )
                relocation = next(
                    item
                    for item in stream.address_relocations
                    if item.record_index == record_index
                    and item.operand_id is operand_id
                )
                self.assertEqual(relocation.addend, 0)
                binding = next(
                    item
                    for item in manifest.address_operand_bindings
                    if item.fragment_id == fragment.id
                    and item.fragment_record_index == record_index
                    and item.operand_id is operand_id
                )
                self.assertEqual(
                    binding.buffer_abi_ids,
                    (fragment.buffer_abi[0].id,),
                )
                self.assertEqual(binding.tensor_slices, (action.tensor_slice,))

            fsm = tuple(
                item
                for item in runtime_definitions
                if item.symbol.kind is RuntimeSymbolKind.DTE_FSM
                and item.source_action_id == send.id
                and item.destination_action_id == recv.id
            )
            token = tuple(
                item
                for item in runtime_definitions
                if item.symbol.kind is RuntimeSymbolKind.DTE_TOKEN
                and item.source_action_id == recv.id
                and item.destination_action_id == wait.id
            )
            self.assertEqual((len(fsm), len(token)), (1, 1))
            for fragment, expected_symbol, expected_count in (
                (send_fragment, fsm[0].symbol.id, 1),
                (destination_fragment, fsm[0].symbol.id, 1),
                (destination_fragment, token[0].symbol.id, 2),
            ):
                self.assertEqual(
                    sum(
                        relocation.symbol_ref == expected_symbol
                        for stream in fragment.core_streams
                        for relocation in stream.runtime_relocations
                    ),
                    expected_count,
                )

        victim = transfer_fragments[0]
        with self.assertRaisesRegex(SchemaError, "cover every executable"):
            NaiveManifestLinker().link(
                context,
                tuple(
                    fragment
                    for fragment in lowered.fragments
                    if fragment.id != victim.id
                ),
            )
        with self.assertRaisesRegex(SchemaError, "duplicate fragment"):
            NaiveManifestLinker().link(
                context,
                lowered.fragments + (victim,),
            )

        bindings = list(manifest.address_operand_bindings)
        binding_index = next(
            index
            for index, binding in enumerate(bindings)
            if binding.fragment_id in transfer_ids
            and binding.operand_id
            in (
                SemanticOperandId.SOURCE_ADDRESS,
                SemanticOperandId.DESTINATION_ADDRESS,
            )
        )
        bindings[binding_index] = replace(
            bindings[binding_index],
            buffer_abi_ids=("buffer_abi_forged",),
        )
        fields = manifest._semantic_key()
        fields["address_operand_bindings"] = tuple(bindings)
        broken = LinkedProgramManifest.create(
            producer_pass=manifest.producer_pass,
            **fields,
        )
        with self.assertRaisesRegex(SchemaError, "BufferABI|address"):
            broken.validate_against(
                context.ir1,
                context.fusion_plans,
                context.standalone_plans,
                context.projection,
                context.schedule_set,
                context.global_dag,
                broken.fragments,
            )

        token_index = next(
            index
            for index, item in enumerate(manifest.runtime_symbol_definitions)
            if item.symbol.kind is RuntimeSymbolKind.DTE_TOKEN
        )
        fields = manifest._semantic_key()
        fields["runtime_symbol_definitions"] = (
            manifest.runtime_symbol_definitions[:token_index]
            + manifest.runtime_symbol_definitions[token_index + 1 :]
        )
        missing_token = LinkedProgramManifest.create(
            producer_pass=manifest.producer_pass,
            **fields,
        )
        with self.assertRaisesRegex(SchemaError, "runtime.*definition"):
            missing_token.validate_against(
                context.ir1,
                context.fusion_plans,
                context.standalone_plans,
                context.projection,
                context.schedule_set,
                context.global_dag,
                missing_token.fragments,
            )


if __name__ == "__main__":
    unittest.main()
