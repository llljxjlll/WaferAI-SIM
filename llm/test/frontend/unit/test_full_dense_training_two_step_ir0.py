"""Both Dense iterations must retain exact source gradient and state edges."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_ir0 import (
    build_full_dense_training_two_step_ir0,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir0 import EdgeKind, OpKind, OpPhase
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec


class FullDenseTrainingTwoStepSourceTest(unittest.TestCase):
    def _build(self):
        plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
        return plan, build_full_dense_training_two_step_ir0(plan)

    def test_two_complete_steps_and_all_parameter_version_edges(self) -> None:
        plan, graph = self._build()
        DenseIR0Validator.validate(graph, "two_step")
        source = {state.identity.tensor_ref for state in graph.persistent_states}
        self.assertEqual(len(source), 15)
        for step in (0, 1):
            scoped = tuple(node for node in graph.nodes
                           if f"::step{step}" in node.id)
            self.assertEqual(sum(node.kind is OpKind.CE_FORWARD for node in scoped), 1)
            self.assertEqual(sum(node.kind is OpKind.CE_BACKWARD for node in scoped), 1)
            self.assertEqual(sum(node.phase is OpPhase.WGRAD for node in scoped), 15)
            self.assertEqual(sum(node.kind is OpKind.OPTIMIZER_UPDATE
                                 for node in scoped), 15)
            self.assertEqual(sum(node.phase is OpPhase.DGRAD and
                                 node.kind is not OpKind.CE_BACKWARD and
                                 not node.id.startswith("gradient_sum::")
                                 for node in scoped), 24)
        state_edges = tuple(edge for edge in graph.edges
                            if edge.kind is EdgeKind.CONTROL
                            and edge.source_node.startswith("sgd_update::"))
        self.assertEqual(len(state_edges), 15)
        self.assertEqual({edge.source_node.split("::")[1] for edge in state_edges},
                         source)

    def test_missing_or_duplicate_ce_and_wrong_version_edge_fail_closed(self) -> None:
        _, graph = self._build()
        ce = next(node for node in graph.nodes
                  if node.kind is OpKind.CE_FORWARD and "::step1" in node.id)
        with self.assertRaisesRegex(SchemaError, "one CE forward/backward per step"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, nodes=tuple(node for node in graph.nodes
                                           if node.id != ce.id)), "missing_ce")
        with self.assertRaisesRegex(SchemaError, "one CE forward/backward per step"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, nodes=(*graph.nodes, ce)), "duplicate_ce")
        state_edge = next(edge for edge in graph.edges
                          if edge.kind is EdgeKind.CONTROL
                          and edge.source_node.startswith("sgd_update::"))
        with self.assertRaisesRegex(SchemaError, "step0 SGD STORE"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, edges=tuple(edge for edge in graph.edges
                                           if edge != state_edge)), "missing_version")

    def test_real_projection_global_dag_native_optimizer_and_state_io(self) -> None:
        from llm.frontend.wafer_frontend.lowering.full_dense_gradient_physical_gate import (
            require_full_dense_physical_gradient_paths,
        )
        from llm.frontend.wafer_frontend.lowering.full_dense_two_step_physical_dag import (
            dense_two_step_native_opcode_contract,
        )
        from llm.frontend.wafer_frontend.lowering.full_training_program_merger import (
            require_independent_ce_loss_gradient_seed,
        )
        from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_runtime import (
            compile_full_dense_two_step_native,
        )

        plan, _ = self._build()
        compilation = compile_full_dense_two_step_native(plan, _hardware(1, 1))
        linked = compilation.program
        physical = Counter(
            record.opcode for fragment in linked.manifest.fragments
            for stream in fragment.core_streams for record in stream.records
        )
        self.assertEqual(physical[RecordOpcode.CROSS_ENTROPY_BACKWARD], 2)
        self.assertEqual(physical[RecordOpcode.SGD_UPDATE], 30)
        self.assertEqual(physical[RecordOpcode.LSU_STORE], 30)
        self.assertEqual(physical[RecordOpcode.LSU_LOAD], 80)
        self.assertEqual(len(linked.manifest.fragments), 280)
        self.assertEqual(len(compilation.physical_dag.actions), 280)
        self.assertEqual(len(compilation.physical_dag.state_version_edges), 15)
        self.assertEqual(len(set(compilation.loss_gradient_seed_abi_by_step.values())), 2)

        seeds = compilation.loss_gradient_seed_abi_by_step
        with self.assertRaisesRegex(SchemaError, "independent per-row FP32 seed"):
            require_independent_ce_loss_gradient_seed(
                linked.manifest, compilation.physical_dag,
                seed_abi_by_step={0: seeds[1], 1: seeds[0]},
            )
        missing_version = replace(
            compilation.physical_dag,
            state_version_edges=compilation.physical_dag.state_version_edges[1:],
        )
        backward, wgrad = dense_two_step_native_opcode_contract(plan)
        with self.assertRaisesRegex(
                SchemaError, "exact next-step HBM LOAD version"):
            require_full_dense_physical_gradient_paths(
                linked.manifest, plan, compilation.requirements, missing_version,
                required_backward_opcodes=backward,
                required_wgrad_opcodes=wgrad,
            )


if __name__ == "__main__":
    unittest.main()
