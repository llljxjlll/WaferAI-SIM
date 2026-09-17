"""AdamW carries five independently versioned in-place state buffers."""

from __future__ import annotations

from dataclasses import replace
import struct
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.action import (
    ComputeContract, ComputeOperand, canonical_compute_operand_roles,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    ProgramSymbol, ProgramSymbolKind, RecordOpcode, RelocatableRecord,
    SemanticOperandId, _compute_record_abi, _fixed_compute_literals,
)
from llm.frontend.wafer_frontend.lowering.coarse import _fixed_compute_operands
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir0 import (
    AdamwUpdateWorkload, EffectKind, NodeEffects, NodeMath,
    NumericalPolicy, OpKind,
)


def _compute(*, step: int = 1) -> ComputeContract:
    workload = AdamwUpdateWorkload(
        logical_weight_shape=(8, 4),
        rank_weight_shape=(8, 4),
        element_count=32,
        step=step,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )
    reads, writes = canonical_compute_operand_roles(
        OpKind.OPTIMIZER_UPDATE, workload, tiled=False,
    )
    return ComputeContract(
        op_kind=OpKind.OPTIMIZER_UPDATE,
        workload=workload,
        math=NodeMath(DType.FP32, NumericalPolicy.TOLERANCE),
        effects=NodeEffects(EffectKind.INPLACE, "adamw.effect", "weight"),
        impl_ref="adamw_update",
        inputs=tuple(ComputeOperand(f"read.{index}", role)
                     for index, role in enumerate(reads)),
        outputs=tuple(ComputeOperand(f"write.{index}", role)
                      for index, role in enumerate(writes)),
    )


class AdamwComputeContractTests(unittest.TestCase):
    def test_compiles_exact_six_to_five_fixed_public_record(self) -> None:
        compute = _compute(step=2)
        compute.validate("adamw")
        abi = _compute_record_abi(compute, path="adamw")
        self.assertIs(abi.opcode, RecordOpcode.ADAMW_UPDATE)
        self.assertEqual((abi.bind_input_count, abi.data_input_index), (6, 1))
        self.assertEqual(
            tuple(item.role for item in compute.outputs),
            ("updated_weight", "updated_master_weight", "updated_first_moment",
             "updated_second_moment", "updated_step_counter"),
        )
        literals = _fixed_compute_literals(compute, abi.opcode, path="adamw")
        self.assertEqual((literals["element_count"], literals["step"]), (32, 2))
        self.assertEqual(
            literals["weight_decay_f64_bits"],
            struct.unpack("<Q", struct.pack("<d", 0.01))[0],
        )

    def test_native_element_count_supports_real_vector_gamma(self) -> None:
        matrix = _compute()
        vector = replace(matrix, workload=replace(
            matrix.workload,
            logical_weight_shape=(32,), rank_weight_shape=(32,),
        ))
        vector.validate("adamw.gamma")
        abi = _compute_record_abi(vector, path="adamw.gamma")
        self.assertIs(abi.opcode, RecordOpcode.ADAMW_UPDATE)
        self.assertEqual(
            _fixed_compute_literals(vector, abi.opcode, path="adamw.gamma")["element_count"],
            32,
        )
        for bad in (
            replace(vector.workload, logical_weight_shape=(2, 4, 4),
                    rank_weight_shape=(2, 4, 4)),
            replace(vector.workload, rank_weight_shape=(1, 32)),
            replace(vector.workload, element_count=31),
        ):
            with self.subTest(workload=bad):
                with self.assertRaises(SchemaError):
                    bad.validate("adamw.gamma")

    def test_rejects_missing_state_role_or_zero_step_and_wrong_dtype(self) -> None:
        compute = _compute()
        with self.assertRaises(SchemaError):
            replace(compute, inputs=compute.inputs[:-1]).validate("adamw")
        with self.assertRaises(SchemaError):
            replace(compute, outputs=compute.outputs[:-1]).validate("adamw")
        for changed in (
            replace(compute.workload, step=0),
            replace(compute.workload, state_dtype=DType.FP16),
            replace(compute.workload, beta1=1.0),
            replace(compute.workload, rank_weight_shape=(4, 4)),
        ):
            with self.subTest(workload=changed):
                with self.assertRaises(SchemaError):
                    changed.validate("adamw")

    def test_coarse_record_preserves_all_state_symbols_and_exact_aliases(self) -> None:
        compute = _compute()
        abi = _compute_record_abi(compute, path="adamw")
        addresses = tuple(
            ProgramSymbol(f"p.{index}", ProgramSymbolKind.ABSOLUTE_ADDRESS,
                          f"binding.{index}")
            for index in range(6)
        )
        output_indices = (0, 2, 3, 4, 5)
        operands = _fixed_compute_operands(
            compute, abi, addresses[0], addresses[1], None, addresses[0],
            adamw_inputs=addresses,
            adamw_outputs=tuple(addresses[index] for index in output_indices),
        )
        record = RelocatableRecord("adamw", RecordOpcode.ADAMW_UPDATE, operands)
        record.validate("adamw")
        self.assertEqual(len(operands), 23)
        self.assertEqual(
            tuple((item.operand_id, item.symbol_ref) for item in operands[5:16]),
            (
                (SemanticOperandId.COMPUTE_INPUT_ADDRESS, "p.0"),
                (SemanticOperandId.COMPUTE_DATA_ADDRESS, "p.1"),
                (SemanticOperandId.COMPUTE_MASTER_ADDRESS, "p.2"),
                (SemanticOperandId.COMPUTE_FIRST_MOMENT_ADDRESS, "p.3"),
                (SemanticOperandId.COMPUTE_SECOND_MOMENT_ADDRESS, "p.4"),
                (SemanticOperandId.COMPUTE_STEP_ADDRESS, "p.5"),
                (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS, "p.0"),
                (SemanticOperandId.COMPUTE_UPDATED_MASTER_ADDRESS, "p.2"),
                (SemanticOperandId.COMPUTE_UPDATED_FIRST_MOMENT_ADDRESS, "p.3"),
                (SemanticOperandId.COMPUTE_UPDATED_SECOND_MOMENT_ADDRESS, "p.4"),
                (SemanticOperandId.COMPUTE_UPDATED_STEP_ADDRESS, "p.5"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
