from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    LoweringContext,
    NaiveStateTransferLowering,
)
from llm.frontend.wafer_frontend.passes.global_action import (
    build_global_action_dag,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    COMMAND_FRAGMENT_SCHEMA_VERSION,
    CommandFragment,
    FragmentKind,
    RecordOpcode,
    RuntimeOperandField,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    SemanticTaskKind,
    StateTransferOrigin,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind

from test_naive_intra_die_state_transfer import _scheduled


def _case(kinds: tuple[StateKind, ...]):
    graph, fusion, standalone, contracts, projection, schedules = _scheduled(
        kinds
    )
    global_dag = build_global_action_dag(graph, projection, schedules)
    context = LoweringContext(
        graph,
        fusion,
        standalone,
        projection,
        schedules,
        global_dag,
    )
    groups = {
        (contract.id, die_id): tuple(
            action
            for action in global_dag.actions
            if isinstance(action.origin_ref, StateTransferOrigin)
            and action.origin_ref.state_transfer_ref == contract.id
            and action.logical_core is not None
            and action.logical_core.die_id == die_id
        )
        for contract in contracts
        for die_id in (0, 1)
    }
    return context, contracts, groups


def _recreate(
    fragment: CommandFragment, **changes: object
) -> CommandFragment:
    fields = fragment._semantic_key()
    fields.update(changes)
    return CommandFragment.create(
        producer_pass=fragment.producer_pass,
        **fields,
    )


class NaiveStateTransferLoweringTest(unittest.TestCase):
    def test_single_k_endpoints_are_exact_and_deterministic(self) -> None:
        context, contracts, groups = _case((StateKind.KV_KEY,))
        contract = contracts[0]
        lowerer = NaiveStateTransferLowering()
        source = lowerer.lower(groups[(contract.id, 0)], context)
        destination = lowerer.lower(groups[(contract.id, 1)], context)

        self.assertEqual(
            source,
            lowerer.lower(groups[(contract.id, 0)], context),
        )
        self.assertEqual(
            destination,
            lowerer.lower(groups[(contract.id, 1)], context),
        )
        self.assertEqual(
            COMMAND_FRAGMENT_SCHEMA_VERSION,
            "wafer_frontend.command_fragment/v1alpha13",
        )
        self.assertTrue(
            all(
                fragment.kind is FragmentKind.STATE_TRANSFER
                and len(fragment.core_streams) == 1
                and len(fragment.program_symbols) == 1
                and len(fragment.buffer_abi) == 1
                and not fragment.state_abi
                for fragment in (source, destination)
            )
        )
        self.assertEqual(
            tuple(
                record.opcode for record in source.core_streams[0].records
            ),
            (RecordOpcode.DTE_SEND,),
        )
        self.assertEqual(
            tuple(
                record.opcode
                for record in destination.core_streams[0].records
            ),
            (RecordOpcode.DTE_RECV, RecordOpcode.DTE_WAIT),
        )
        self.assertEqual(
            (len(source.runtime_symbols), len(destination.runtime_symbols)),
            (2, 3),
        )
        destination_tokens = tuple(
            relocation.symbol_ref
            for relocation in destination.core_streams[0].runtime_relocations
            if relocation.field is RuntimeOperandField.DTE_TOKEN
        )
        self.assertEqual(len(destination_tokens), 2)
        self.assertEqual(len(set(destination_tokens)), 1)
        source.validate_against(context.global_dag)
        destination.validate_against(context.global_dag)

    def test_kv_pair_freezes_fragment_record_and_byte_counts(self) -> None:
        context, contracts, groups = _case(
            (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        lowerer = NaiveStateTransferLowering()
        fragments = tuple(
            lowerer.lower(groups[(contract.id, die_id)], context)
            for contract in contracts
            for die_id in (0, 1)
        )
        records = tuple(
            record
            for fragment in fragments
            for stream in fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(len(fragments), 4)
        self.assertEqual(len({fragment.id for fragment in fragments}), 4)
        self.assertEqual(
            tuple(record.opcode for record in records).count(
                RecordOpcode.DTE_SEND
            ),
            2,
        )
        self.assertEqual(
            tuple(record.opcode for record in records).count(
                RecordOpcode.DTE_RECV
            ),
            2,
        )
        self.assertEqual(
            tuple(record.opcode for record in records).count(
                RecordOpcode.DTE_WAIT
            ),
            2,
        )
        transport_records = tuple(
            record
            for record in records
            if record.opcode in (RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV)
        )
        self.assertEqual(
            {
                next(
                    operand.literal_value
                    for operand in record.operands
                    if operand.name == "length_bytes"
                )
                for record in transport_records
            },
            {32},
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
            128,
        )
        self.assertEqual(sum(len(item.buffer_abi) for item in fragments), 4)
        self.assertEqual(sum(len(item.state_abi) for item in fragments), 0)
        for fragment in fragments:
            fragment.validate_against(context.global_dag)

    def test_group_and_artifact_tampering_fail_closed(self) -> None:
        context, contracts, groups = _case(
            (StateKind.KV_KEY, StateKind.KV_VALUE)
        )
        lowerer = NaiveStateTransferLowering()
        first = contracts[0]
        second = contracts[1]
        destination_actions = groups[(first.id, 1)]
        for label, actions in (
            ("partial", destination_actions[:1]),
            ("reversed", tuple(reversed(destination_actions))),
            (
                "mixed_contract",
                (
                    groups[(first.id, 0)][0],
                    groups[(second.id, 0)][0],
                ),
            ),
        ):
            with self.subTest(label=label), self.assertRaises(SchemaError):
                lowerer.lower(actions, context)

        forged_transit = replace(
            groups[(first.id, 0)][0],
            task_kind=SemanticTaskKind.TRANSIT,
            logical_core=None,
            core_order_index=None,
        )
        with self.assertRaisesRegex(SchemaError, "TRANSIT emits no fragment"):
            lowerer.lower((forged_transit,), context)

        source = lowerer.lower(groups[(first.id, 0)], context)
        source_stream = source.core_streams[0]
        bad_relocation = replace(
            source_stream.address_relocations[0],
            addend=source_stream.address_relocations[0].addend + 2,
        )
        bad_addend = _recreate(
            source,
            core_streams=(
                replace(
                    source_stream,
                    address_relocations=(bad_relocation,),
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "addend"):
            bad_addend.validate_against(context.global_dag)

        destination = lowerer.lower(destination_actions, context)
        recv = next(
            action
            for action in destination_actions
            if action.task_kind is SemanticTaskKind.RECV
        )
        destination_stream = destination.core_streams[0]
        missing_wait = _recreate(
            destination,
            claimed_action_ids=(recv.id,),
            core_streams=(
                replace(
                    destination_stream,
                    records=destination_stream.records[:1],
                    runtime_relocations=tuple(
                        relocation
                        for relocation in destination_stream.runtime_relocations
                        if relocation.record_index == 0
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "exactly claim"):
            missing_wait.validate_against(context.global_dag)


if __name__ == "__main__":
    unittest.main()
