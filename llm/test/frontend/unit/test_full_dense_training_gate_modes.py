from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.full_dense_gradient_physical_gate import (
    dense_dx_upstream_output_operands,
)
from llm.frontend.wafer_frontend.lowering.full_training_program_merger import (
    require_full_training_opcode_matrix,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    RecordOpcode, SemanticOperandId,
)
from llm.frontend.wafer_frontend.schema.full_training_physical_dag import (
    FullTrainingPhysicalDAG, PhysicalTrainingAction,
)
from llm.frontend.wafer_frontend.schema.global_action import LogicalCoreRef


def _dense_dag() -> FullTrainingPhysicalDAG:
    actions = []
    previous = None
    index = 0
    for step in (0, 1):
        for layer in (0, 1):
            opcodes = [
                RecordOpcode.ATTENTION_EXACT,
                RecordOpcode.RMSNORM, RecordOpcode.RMSNORM,
                RecordOpcode.RESIDUAL, RecordOpcode.RESIDUAL,
                RecordOpcode.SWIGLU,
                RecordOpcode.SWIGLU_BACKWARD_TIMING,
                RecordOpcode.RMSNORM_BACKWARD_TIMING,
                RecordOpcode.RMSNORM_BACKWARD_TIMING,
                RecordOpcode.ATTENTION_BACKWARD_TIMING,
                RecordOpcode.ROPE_BACKWARD_TIMING,
                RecordOpcode.RESIDUAL_BACKWARD_TIMING,
                RecordOpcode.RESIDUAL_BACKWARD_TIMING,
                RecordOpcode.GEMM_DX_TIMING,
                RecordOpcode.GEMM_DX_TIMING,
                RecordOpcode.GEMM_DX_TIMING,
                RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
                RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
                RecordOpcode.GEMM_WEIGHT_WGRAD_TIMING,
                RecordOpcode.NORM_GAMMA_WGRAD_TIMING,
                RecordOpcode.NORM_GAMMA_WGRAD_TIMING,
                RecordOpcode.SGD_UPDATE,
            ]
            if layer == 0:
                opcodes.extend((RecordOpcode.CROSS_ENTROPY_FORWARD,
                                RecordOpcode.CROSS_ENTROPY_BACKWARD,
                                RecordOpcode.LSU_LOAD,
                                RecordOpcode.LSU_STORE))
            identifier = f"a{index:02d}"
            actions.append(PhysicalTrainingAction(
                identifier, LogicalCoreRef(0, 0), identifier,
                f"step{step}.layer{layer}",
                "backward", step, layer,
                tuple((f"fragment{index:02d}", offset, opcode)
                      for offset, opcode in enumerate(opcodes)),
                () if previous is None else (previous,),
            ))
            previous = identifier
            index += 1
    return FullTrainingPhysicalDAG.create(
        source_artifact_ids=("source",), actions=tuple(actions)
    )


class FullDenseTrainingGateModesTest(unittest.TestCase):
    def test_dense_matrix_requires_native_reverse_without_moe_transport(self) -> None:
        dag = _dense_dag()
        require_full_training_opcode_matrix(dag, require_moe=False)
        with self.assertRaisesRegex(SchemaError, "MoE backward"):
            require_full_training_opcode_matrix(dag)
        victim = dag.actions[0]
        records = tuple(item for item in victim.executable_records
                        if item[2] is not RecordOpcode.ROPE_BACKWARD_TIMING)
        broken = replace(dag, actions=(replace(victim, executable_records=records),
                                      *dag.actions[1:]))
        with self.assertRaisesRegex(SchemaError, "Dense layer lacks"):
            require_full_training_opcode_matrix(broken, require_moe=False)

    def test_all_real_backbone_outputs_may_feed_gemm_dx(self) -> None:
        for opcode in (
            RecordOpcode.CROSS_ENTROPY_BACKWARD,
            RecordOpcode.GEMM_DX_TIMING,
            RecordOpcode.SWIGLU_BACKWARD_TIMING,
            RecordOpcode.RMSNORM_BACKWARD_TIMING,
            RecordOpcode.ATTENTION_BACKWARD_TIMING,
            RecordOpcode.ROPE_BACKWARD_TIMING,
        ):
            self.assertEqual(dense_dx_upstream_output_operands(opcode),
                             (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,))
        self.assertEqual(
            dense_dx_upstream_output_operands(
                RecordOpcode.RESIDUAL_BACKWARD_TIMING),
            (SemanticOperandId.COMPUTE_OUTPUT_ADDRESS,
             SemanticOperandId.COMPUTE_AUX_ADDRESS),
        )
        with self.assertRaisesRegex(SchemaError, "not a native Dense"):
            dense_dx_upstream_output_operands(RecordOpcode.MATMUL)


if __name__ == "__main__":
    unittest.main()
