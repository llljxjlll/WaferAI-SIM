from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering import NaiveCoarseLowering
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    OperandKind,
    ProgramSymbolKind,
    RecordOpcode,
    SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferUseRole,
    RegionLowering,
    SemanticTaskKind,
)

from test_linked_program_manifest_schema import valid_exact_linked_manifest
from test_n5_pipeline import _compile_through_n5


class NaiveCoarseLoweringTest(unittest.TestCase):
    def test_matmul_is_exact_deterministic_and_self_validating(self) -> None:
        context, _manifest = valid_exact_linked_manifest()
        action = context.global_dag.actions[0]
        lowerer = NaiveCoarseLowering()
        fragment = lowerer.lower(action, context)
        self.assertEqual(fragment, lowerer.lower(action, context))
        self.assertEqual(fragment.claimed_action_ids, (action.id,))
        self.assertEqual(
            tuple(record.opcode for record in fragment.core_streams[0].records),
            (RecordOpcode.SRAM_BIND, RecordOpcode.MATMUL),
        )
        bind, matmul = fragment.core_streams[0].records
        self.assertEqual(bind.operands[0].literal_value, 1)
        rank_m, rank_n, rank_k = action.compute.workload.rank_shape
        self.assertEqual(matmul.operands[-1].literal_value, (1, rank_m, rank_k, rank_n))
        self.assertEqual(
            {abi.binding_id for abi in fragment.buffer_abi},
            {use.binding_id for use in action.buffer_uses},
        )
        label_relocations = tuple(
            relocation
            for relocation in fragment.core_streams[0].address_relocations
            if relocation.record_index == 0
        )
        self.assertEqual(
            tuple(relocation.operand_id for relocation in label_relocations),
            (
                SemanticOperandId.SRAM_BIND_INPUT_0,
                SemanticOperandId.SRAM_BIND_OUTPUT,
            ),
        )
        symbols = {symbol.id: symbol for symbol in fragment.program_symbols}
        self.assertTrue(
            all(
                symbols[relocation.symbol_ref].kind is ProgramSymbolKind.SRAM_LABEL
                for relocation in label_relocations
            )
        )
        fragment.validate_against(context.global_dag)

    def test_rejects_an_action_that_is_not_the_exact_context_member(self) -> None:
        context, _manifest = valid_exact_linked_manifest()
        action = context.global_dag.actions[0]
        forged = replace(action, member_id="impostor")
        with self.assertRaisesRegex(SchemaError, "exactly equal"):
            NaiveCoarseLowering().lower(forged, context)

    def test_matmul_uses_have_frozen_activation_weight_output_roles(self) -> None:
        context, _manifest = valid_exact_linked_manifest()
        action = context.global_dag.actions[0]
        roles = tuple(
            (use.role, use.operand_index) for use in action.buffer_uses
        )
        self.assertEqual(
            roles,
            (
                (BufferUseRole.COMP_INPUT, 0),
                (BufferUseRole.COMP_INPUT, 1),
                (BufferUseRole.COMP_OUTPUT, 0),
            ),
        )
        NaiveCoarseLowering().lower(action, context)

    def test_real_tp2_dense_compute_abis_are_exact(self) -> None:
        context = _compile_through_n5()[-1].entries[0].lowering_context()
        actions = tuple(
            action
            for action in context.global_dag.actions
            if action.task_kind is SemanticTaskKind.COMP
            and action.lowering is RegionLowering.JSON_COARSE
        )
        first_by_impl = {}
        for action in actions:
            assert action.compute is not None
            first_by_impl.setdefault(action.compute.impl_ref, action)
        expected = {
            "embedding_lookup": (
                RecordOpcode.EMBEDDING_LOOKUP,
                2,
                None,
                True,
            ),
            "matmul_forward": (
                RecordOpcode.MATMUL,
                1,
                (1, 32, 256, 256),
                True,
            ),
            "attention_forward": (
                RecordOpcode.ATTENTION_EXACT,
                1,
                None,
                False,
            ),
            "rms_norm": (
                RecordOpcode.RMSNORM,
                1,
                (1, 16, 256),
                True,
            ),
            "rope_qk_exact": (
                RecordOpcode.ROPE_QK_EXACT,
                1,
                None,
                False,
            ),
            "swiglu": (
                RecordOpcode.SWIGLU,
                1,
                (8192,),
                False,
            ),
            "residual": (
                RecordOpcode.RESIDUAL,
                2,
                (4096,),
                True,
            ),
        }
        self.assertEqual(set(first_by_impl), set(expected))
        lowerer = NaiveCoarseLowering()
        for impl_ref, (
            opcode,
            input_count,
            parameters,
            addressed_data,
        ) in expected.items():
            with self.subTest(impl_ref=impl_ref):
                action = first_by_impl[impl_ref]
                fragment = lowerer.lower(action, context)
                self.assertEqual(fragment, lowerer.lower(action, context))
                bind, compute = fragment.core_streams[0].records
                self.assertEqual(
                    (bind.opcode, compute.opcode),
                    (RecordOpcode.SRAM_BIND, opcode),
                )
                self.assertEqual(bind.operands[0].literal_value, input_count)
                if parameters is None:
                    self.assertNotIn(
                        "parameters",
                        tuple(operand.name for operand in compute.operands),
                    )
                else:
                    self.assertEqual(
                        compute.operands[-1].literal_value, parameters
                    )
                    self.assertEqual(
                        compute.operands[2].kind,
                        (
                            OperandKind.ADDRESS_SYMBOL
                            if addressed_data
                            else OperandKind.LITERAL
                        ),
                    )
                    self.assertEqual(
                        compute.operands[2].literal_value,
                        None if addressed_data else 0,
                    )
                expected_binding_ids = {
                    use.binding_id for use in action.buffer_uses
                }
                self.assertEqual(
                    {abi.binding_id for abi in fragment.buffer_abi},
                    expected_binding_ids,
                )
                fragment.validate_against(context.global_dag)


if __name__ == "__main__":
    unittest.main()
