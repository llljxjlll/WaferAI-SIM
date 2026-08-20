from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import (
    LoweringContext,
    NaiveIsaRegionLowering,
)
from llm.frontend.wafer_frontend.passes.global_action import (
    build_global_action_dag,
)
from llm.frontend.wafer_frontend.policies.naive_inter_die import (
    NaiveInterDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import (
    NaiveIntraDiePolicy,
)
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import (
    NaiveProjectToIR2,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    OperandKind,
    RecordOpcode,
    RuntimeOperandField,
    RuntimeSymbolKind,
)
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import (
    FusedNodeOrigin,
    SemanticTaskKind,
)

from test_naive_project_to_ir2 import _fused_graph


_LARGE_SRAM_BYTES = 64 * 1024 * 1024


def _context(tp: int) -> tuple[LoweringContext, object, tuple]:
    graph = _fused_graph(tp)
    profiles = tuple(
        replace(
            profile,
            capacity_bytes=_LARGE_SRAM_BYTES,
            regions=tuple(
                replace(region, size_bytes=_LARGE_SRAM_BYTES)
                for region in profile.regions
            ),
        )
        for profile in graph.fabric.sram_profiles
    )
    graph = IR1.create(
        producer_pass=graph.producer_pass,
        **{
            **graph._semantic_key(),
            "fabric": replace(graph.fabric, sram_profiles=profiles),
        },
    )
    plan = NaiveInterDiePolicy().plan(
        graph, graph.fused_op_skeletons[0], graph.profile
    )
    projection = NaiveProjectToIR2().run(graph, (plan,), (), state_transfers=())
    schedule_set = NaiveIntraDiePolicy().schedule(projection, graph)
    global_dag = build_global_action_dag(graph, projection, schedule_set)
    context = LoweringContext(
        graph, (plan,), (), projection, schedule_set, global_dag
    )
    context.validate()
    actions = tuple(
        action
        for action in global_dag.actions
        if isinstance(action.origin_ref, FusedNodeOrigin)
        and action.origin_ref.plan_id == plan.id
        and action.task_kind is not SemanticTaskKind.TRANSIT
    )
    return context, plan, actions


class NaiveIsaRegionLoweringTest(unittest.TestCase):
    def test_tp2_single_plan_has_exact_local_records_and_async_tokens(self) -> None:
        context, plan, actions = _context(2)
        lowerer = NaiveIsaRegionLowering()
        manifests = lowerer.lower(plan, actions, context)
        self.assertEqual(manifests, lowerer.lower(plan, actions, context))
        self.assertEqual(tuple(item.target_dies for item in manifests), ((0,), (1,)))
        self.assertEqual(
            tuple(len(item.fragment.claimed_action_ids) for item in manifests),
            (6, 6),
        )
        self.assertEqual(
            tuple(
                sum(len(stream.records) for stream in item.fragment.core_streams)
                for item in manifests
            ),
            (8, 8),
        )

        action_index = {action.id: action for action in actions}
        all_records = tuple(
            record
            for manifest in manifests
            for stream in manifest.fragment.core_streams
            for record in stream.records
        )
        self.assertEqual(
            tuple(record.opcode for record in all_records).count(RecordOpcode.MATMUL),
            4,
        )
        self.assertEqual(
            tuple(record.opcode for record in all_records).count(RecordOpcode.DTE_SEND),
            2,
        )
        self.assertEqual(
            tuple(record.opcode for record in all_records).count(RecordOpcode.DTE_RECV),
            2,
        )
        self.assertEqual(
            tuple(record.opcode for record in all_records).count(RecordOpcode.DTE_WAIT),
            2,
        )
        self.assertEqual(
            tuple(record.opcode for record in all_records).count(RecordOpcode.LOCAL_REDUCE),
            2,
        )
        for record in all_records:
            action = action_index[record.source_global_action_id]
            if record.opcode is RecordOpcode.MATMUL:
                rank_m, rank_n, rank_k = action.compute.workload.rank_shape
                self.assertEqual(
                    record.operands[-1].literal_value,
                    (1, rank_m, rank_k, rank_n),
                )
            elif record.opcode in (RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV):
                length = next(
                    operand.literal_value
                    for operand in record.operands
                    if operand.name == "length_bytes"
                )
                self.assertEqual(length, action.bytes)
                self.assertEqual(length, plan.chunk_slices[action.chunk_id].bytes)

        for manifest in manifests:
            records = {
                record.source_global_action_id: (index, record)
                for stream in manifest.fragment.core_streams
                for index, record in enumerate(stream.records)
            }
            symbols = {
                symbol.id: symbol for symbol in manifest.fragment.runtime_symbols
            }
            for action in actions:
                if action.id not in records:
                    continue
                _record_index, record = records[action.id]
                if action.task_kind is SemanticTaskKind.SEND:
                    self.assertEqual(record.operands[2].literal_value, 1)
                    self.assertEqual(record.operands[6].literal_value, 0)
                elif action.task_kind is SemanticTaskKind.RECV:
                    self.assertEqual(record.operands[1].literal_value, 0)
                    self.assertIs(record.operands[5].kind, OperandKind.RUNTIME_SYMBOL)
                    token_id = record.operands[5].symbol_ref
                    self.assertIs(symbols[token_id].kind, RuntimeSymbolKind.DTE_TOKEN)
                    wait = next(
                        candidate
                        for candidate in actions
                        if candidate.task_kind is SemanticTaskKind.WAIT
                        and action.id in candidate.deps
                    )
                    _wait_index, wait_record = records[wait.id]
                    self.assertIs(wait_record.opcode, RecordOpcode.DTE_WAIT)
                    self.assertEqual(wait_record.operands[0].symbol_ref, token_id)
                    token_relocations = tuple(
                        relocation
                        for stream in manifest.fragment.core_streams
                        for relocation in stream.runtime_relocations
                        if relocation.field is RuntimeOperandField.DTE_TOKEN
                        and relocation.symbol_ref == token_id
                    )
                    self.assertEqual(len(token_relocations), 2)
            manifest.validate_against(context.global_dag)

    def test_tp4_is_per_die_canonical_and_excludes_transit_actions(self) -> None:
        context, plan, actions = _context(4)
        lowerer = NaiveIsaRegionLowering()
        manifests = lowerer.lower(plan, actions, context)

        self.assertEqual(manifests, lowerer.lower(plan, actions, context))
        self.assertEqual(len(manifests), 4)
        self.assertEqual(
            tuple(manifest.target_dies for manifest in manifests),
            ((0,), (1,), (2,), (3,)),
        )
        self.assertEqual(
            tuple(len(manifest.fragment.claimed_action_ids) for manifest in manifests),
            (14, 14, 14, 14),
        )
        self.assertEqual(
            tuple(
                sum(
                    len(stream.records)
                    for stream in manifest.fragment.core_streams
                )
                for manifest in manifests
            ),
            (18, 18, 18, 18),
        )
        claimed = tuple(
            action_id
            for manifest in manifests
            for action_id in manifest.fragment.claimed_action_ids
        )
        self.assertEqual(set(claimed), {action.id for action in actions})
        self.assertEqual(len(claimed), len(set(claimed)))
        transit_ids = {
            action.id
            for action in context.global_dag.actions
            if isinstance(action.origin_ref, FusedNodeOrigin)
            and action.origin_ref.plan_id == plan.id
            and action.task_kind is SemanticTaskKind.TRANSIT
        }
        self.assertTrue(transit_ids)
        self.assertTrue(transit_ids.isdisjoint(claimed))
        for manifest in manifests:
            manifest.validate_against(context.global_dag)

    def test_rejects_reordered_or_forged_actions(self) -> None:
        context, plan, actions = _context(2)
        lowerer = NaiveIsaRegionLowering()

        with self.assertRaisesRegex(SchemaError, "exactly preserve"):
            lowerer.lower(plan, tuple(reversed(actions)), context)
        forged = (replace(actions[0], member_id="forged"), *actions[1:])
        with self.assertRaisesRegex(SchemaError, "exactly preserve"):
            lowerer.lower(plan, forged, context)

    def test_rejects_forged_plan_even_when_stable_id_is_unchanged(self) -> None:
        context, plan, actions = _context(2)
        forged = replace(plan, producer_pass="forged")

        self.assertEqual(forged.id, plan.id)
        with self.assertRaisesRegex(SchemaError, "exactly equal"):
            NaiveIsaRegionLowering().lower(forged, actions, context)


if __name__ == "__main__":
    unittest.main()
