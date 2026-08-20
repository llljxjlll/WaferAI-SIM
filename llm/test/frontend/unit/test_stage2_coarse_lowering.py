from __future__ import annotations

import unittest
from dataclasses import replace
from collections import Counter

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.link_program import link_bundle
from llm.frontend.wafer_frontend.passes.lower_program import lower_bundle
from llm.frontend.wafer_frontend.schema.n6 import (
    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
    LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
)

from llm.frontend.wafer_frontend.lowering.coarse import NaiveCoarseLowering
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
    RecordOpcode,
    SemanticOperandId,
)
from test_stage2_dense_forward_graph import _schedule_tiny


class Stage2CoarseLoweringTest(unittest.TestCase):
    def test_real_tp1_fixed_records_and_rms_data_are_exact(self) -> None:
        _planned, _projected, _policy, _scheduled, bundle = _schedule_tiny(1)
        context = bundle.entries[0].lowering_context()
        actions = {
            action.compute.impl_ref: action
            for action in context.global_dag.actions
            if action.compute is not None
        }
        expected = {
            "embedding_lookup": (
                RecordOpcode.EMBEDDING_LOOKUP,
                (
                    "index_datatype",
                    "table_datatype",
                    "output_datatype",
                    "placement",
                    "indices_address",
                    "table_address",
                    "output_address",
                    "logical_rows",
                    "rank_rows",
                    "tp_degree",
                    "vocab_size",
                    "hidden_size",
                ),
            ),
            "rope_qk_exact": (
                RecordOpcode.ROPE_QK_EXACT,
                (
                    "datatype",
                    "packed_layout",
                    "input_address",
                    "output_address",
                    "logical_tokens",
                    "tp_degree",
                    "num_heads",
                    "num_kv_heads",
                    "rank_num_heads",
                    "rank_num_kv_heads",
                    "head_dim",
                    "rotary_dim",
                    "max_position_embeddings",
                    "context_max",
                    "rope_theta_f64_bits",
                ),
            ),
            "attention_forward": (
                RecordOpcode.ATTENTION_EXACT,
                (
                    "datatype",
                    "mode",
                    "packed_layout",
                    "causal",
                    "input_address",
                    "output_address",
                    "query_tokens",
                    "tp_degree",
                    "num_heads",
                    "num_kv_heads",
                    "rank_num_heads",
                    "rank_num_kv_heads",
                    "head_dim",
                    "context_sum",
                    "context_max",
                    "query_key_pairs",
                    "rank_kv_read_bytes",
                    "rank_kv_write_bytes",
                ),
            ),
        }
        lowerer = NaiveCoarseLowering()
        for impl_ref, (opcode, names) in expected.items():
            with self.subTest(impl_ref=impl_ref):
                fragment = lowerer.lower(actions[impl_ref], context)
                record = fragment.core_streams[0].records[1]
                self.assertEqual(record.opcode, opcode)
                self.assertEqual(tuple(operand.name for operand in record.operands), names)
                self.assertNotIn("parameters", names)
                fragment.validate_against(context.global_dag)

        rms = lowerer.lower(actions["rms_norm"], context)
        rms_record = rms.core_streams[0].records[1]
        self.assertEqual(rms_record.opcode, RecordOpcode.RMSNORM)
        self.assertEqual(
            tuple(
                relocation.operand_id
                for relocation in rms.core_streams[0].address_relocations
                if relocation.record_index == 1
            ),
            (
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
                SemanticOperandId.COMPUTE_DATA_ADDRESS,
                SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
            ),
        )

    def test_real_tp1_greedy_record_is_exact(self) -> None:
        _planned, _projected, _policy, _scheduled, bundle = _schedule_tiny(
            1,
            output="greedy_sample",
        )
        context = bundle.entries[0].lowering_context()
        action = next(
            action
            for action in context.global_dag.actions
            if action.compute is not None
            and action.compute.impl_ref == "greedy_sample"
        )
        fragment = NaiveCoarseLowering().lower(action, context)
        record = fragment.core_streams[0].records[1]
        self.assertEqual(record.opcode, RecordOpcode.GREEDY_SAMPLE)
        literals = {
            operand.name: operand.literal_value
            for operand in record.operands
            if operand.literal_value is not None
        }
        self.assertEqual(literals["sample_count"], 1)
        self.assertEqual(literals["comparisons"], 31)
        self.assertEqual(literals["output_datatype"], 2)

    def test_tp1_linker_rejects_rms_data_as_activation_closure(self) -> None:
        _p, _r, _c, _s, source = _schedule_tiny(
            1,
            block_allocator=True,
        )
        lowered = lower_bundle(source)
        manifest = link_bundle(lowered).entries[0].manifest
        leaves = {
            (
                fragment.fragment.id
                if hasattr(fragment, "fragment")
                else fragment.id
            ): (
                fragment.fragment
                if hasattr(fragment, "fragment")
                else fragment
            )
            for fragment in manifest.fragments
        }

        def record_for(binding):
            fragment = leaves[binding.fragment_id]
            stream = next(
                stream
                for stream in fragment.core_streams
                if stream.logical_core == binding.logical_core
            )
            return stream.records[binding.fragment_record_index]

        data = next(
            binding
            for binding in manifest.address_operand_bindings
            if binding.operand_id is SemanticOperandId.COMPUTE_DATA_ADDRESS
            and record_for(binding).opcode is RecordOpcode.RMSNORM
        )
        activation = next(
            binding
            for binding in manifest.address_operand_bindings
            if (
                binding.fragment_id,
                binding.logical_core,
                binding.fragment_record_index,
                binding.operand_id,
            )
            == (
                data.fragment_id,
                data.logical_core,
                data.fragment_record_index,
                SemanticOperandId.COMPUTE_INPUT_ADDRESS,
            )
        )
        forged_binding = replace(
            data,
            buffer_abi_ids=activation.buffer_abi_ids,
            tensor_slices=activation.tensor_slices,
        )
        fields = manifest._semantic_key()
        fields["address_operand_bindings"] = tuple(
            forged_binding if binding == data else binding
            for binding in manifest.address_operand_bindings
        )
        forged = LinkedProgramManifest.create(
            producer_pass=manifest.producer_pass,
            **fields,
        )
        context = source.entries[0].lowering_context()
        with self.assertRaisesRegex(SchemaError, "exact|closure"):
            forged.validate_against(
                context.ir1,
                context.fusion_plans,
                context.standalone_plans,
                context.projection,
                context.schedule_set,
                context.global_dag,
                lowered.entries[0].fragments,
            )

    def test_tp1_tp2_tp4_whole_lower_link_lifecycle_goldens(self) -> None:
        expected = {
            1: (
                44,
                44,
                159,
                278,
                {
                    "ATTENTION_EXACT": 2,
                    "EMBEDDING_LOOKUP": 1,
                    "LSU_LOAD": 15,
                    "LSU_STORE": 4,
                    "MATMUL": 9,
                    "RESIDUAL": 4,
                    "RMSNORM": 5,
                    "ROPE_QK_EXACT": 2,
                    "SRAM_ALLOC_AT": 45,
                    "SRAM_BIND": 25,
                    "SRAM_FREE": 45,
                    "SWIGLU": 2,
                },
            ),
            2: (
                160,
                92,
                510,
                804,
                {
                    "ATTENTION_EXACT": 4,
                    "DTE_ISSUE": 8,
                    "DTE_RECV": 16,
                    "DTE_SEND": 16,
                    "DTE_WAIT": 16,
                    "EMBEDDING_LOOKUP": 2,
                    "EVENT_SET": 8,
                    "EVENT_WAIT": 8,
                    "LOCAL_REDUCE": 8,
                    "LSU_LOAD": 30,
                    "LSU_STORE": 8,
                    "MATMUL": 26,
                    "RESIDUAL": 8,
                    "RMSNORM": 10,
                    "ROPE_QK_EXACT": 4,
                    "SRAM_ALLOC_AT": 138,
                    "SRAM_BIND": 58,
                    "SRAM_FREE": 138,
                    "SWIGLU": 4,
                },
            ),
            4: (
                544,
                180,
                1452,
                2184,
                {
                    "ATTENTION_EXACT": 8,
                    "DTE_ISSUE": 16,
                    "DTE_RECV": 96,
                    "DTE_SEND": 96,
                    "DTE_WAIT": 64,
                    "EMBEDDING_LOOKUP": 4,
                    "EVENT_SET": 24,
                    "EVENT_WAIT": 24,
                    "LOCAL_REDUCE": 16,
                    "LSU_LOAD": 60,
                    "LSU_STORE": 16,
                    "MATMUL": 84,
                    "RESIDUAL": 16,
                    "RMSNORM": 20,
                    "ROPE_QK_EXACT": 8,
                    "SRAM_ALLOC_AT": 372,
                    "SRAM_BIND": 148,
                    "SRAM_FREE": 372,
                    "SWIGLU": 8,
                },
            ),
        }
        for tp, (
            action_count,
            fragment_count,
            record_count,
            closure_count,
            opcode_counts,
        ) in expected.items():
            with self.subTest(tp=tp):
                _p, _r, _c, _s, source = _schedule_tiny(
                    tp,
                    block_allocator=True,
                )
                lowered = lower_bundle(source)
                linked = link_bundle(lowered)
                profile = linked.entries[0]
                manifest = profile.manifest
                leaves = tuple(
                    fragment.fragment
                    if hasattr(fragment, "fragment")
                    else fragment
                    for fragment in manifest.fragments
                )
                actual_opcodes = Counter(
                    record.opcode.name
                    for fragment in leaves
                    for stream in fragment.core_streams
                    for record in stream.records
                )
                self.assertEqual(
                    lowered.schema_version,
                    LOWERED_PROGRAM_BUNDLE_SCHEMA_VERSION,
                )
                self.assertEqual(
                    linked.schema_version,
                    LINKED_PROGRAM_BUNDLE_SCHEMA_VERSION,
                )
                self.assertEqual(len(source.entries[0].global_dag.actions), action_count)
                self.assertEqual(len(leaves), fragment_count)
                self.assertEqual(sum(actual_opcodes.values()), record_count)
                self.assertEqual(
                    len(manifest.address_operand_bindings),
                    closure_count,
                )
                self.assertEqual(dict(actual_opcodes), opcode_counts)
                self.assertEqual(
                    actual_opcodes["SRAM_ALLOC_AT"],
                    actual_opcodes["SRAM_FREE"],
                )



if __name__ == "__main__":
    unittest.main()
