"""Source state and optimizer coverage is exact, including repeated steps."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.ir0 import EdgeKind, GraphEdge, OpKind, StateAccess, StateAccessMode
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FullDenseTrainingTpShardValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
        cls.graph = build_full_dense_training_two_step_ir0(plan)

    def test_wrong_optimizer_state_shard_is_rejected(self) -> None:
        graph = self.graph
        update = next(node for node in graph.nodes if node.kind is OpKind.OPTIMIZER_UPDATE)
        access = next(access for access in graph.state_accesses if access.node_ref == update.id)
        forged = StateAccess.create(node_ref=update.id, state_ref=access.state_ref,
                                    mode=StateAccessMode.READ_WRITE, rank=1)
        altered = replace(graph, state_accesses=tuple(
            forged if current is access else current for current in graph.state_accesses
        ))
        with self.assertRaisesRegex(SchemaError, "its own FP32 WGRAD"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(altered, "wrong_rank")

    def test_missing_or_duplicate_state_version_edge_is_rejected(self) -> None:
        graph = self.graph
        edge = next(edge for edge in graph.edges if edge.kind is EdgeKind.CONTROL
                    and edge.source_node.startswith("sgd_update::"))
        with self.assertRaisesRegex(SchemaError, "step0 SGD STORE"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, edges=tuple(x for x in graph.edges if x is not edge)),
                "missing_version",
            )
        duplicate = GraphEdge(edge.id + ".duplicate", edge.kind,
                              edge.source_node, edge.destination_node, None)
        with self.assertRaisesRegex(SchemaError, "step0 SGD STORE"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, edges=(*graph.edges, duplicate)), "duplicate_version",
            )


class FullDenseTrainingTp4ShardValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        plan = build_flexible_dense_train_plan(_spec(1, 4), RectMeshSpec(1, 4))
        cls.graph = build_full_dense_training_two_step_ir0(plan)

    def test_full_source_has_all_parameter_shards_both_steps(self) -> None:
        DenseIR0Validator.validate(self.graph, "tp4")
        self.assertEqual(len(self.graph.persistent_states), 60)
        self.assertEqual(sum(node.kind is OpKind.OPTIMIZER_UPDATE
                             for node in self.graph.nodes), 120)

    def test_missing_one_rank_gradient_read_fails(self) -> None:
        graph = self.graph
        victim = next(access for access in graph.state_accesses
                      if "wgrad::" in access.node_ref and "::tp3::step1" in access.node_ref)
        with self.assertRaisesRegex(SchemaError, "WGRAD reads the wrong TP parameter shard"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, state_accesses=tuple(
                    access for access in graph.state_accesses if access is not victim)),
                "missing_tp3_read",
            )

    def test_rank_three_update_cannot_write_rank_zero_state(self) -> None:
        graph = self.graph
        access = next(access for access in graph.state_accesses
                      if access.node_ref.startswith("sgd_update::")
                      and "::tp3::step0" in access.node_ref)
        states = {state.id: state for state in graph.persistent_states}
        weight = states[access.state_ref].identity.tensor_ref
        rank_zero = next(state for state in states.values()
                         if state.identity.tensor_ref == weight
                         and state.identity.shard_index == 0)
        forged = StateAccess.create(node_ref=access.node_ref, state_ref=rank_zero.id,
                                    mode=StateAccessMode.READ_WRITE, rank=3)
        with self.assertRaisesRegex(SchemaError, "its own FP32 WGRAD"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, state_accesses=tuple(
                    forged if current is access else current for current in graph.state_accesses)),
                "wrong_tp3_state",
            )

    def test_rank_three_version_edge_cannot_be_skipped(self) -> None:
        graph = self.graph
        victim = next(edge for edge in graph.edges if edge.kind is EdgeKind.CONTROL
                      and edge.source_node.startswith("sgd_update::")
                      and "::tp3::step0" in edge.source_node)
        with self.assertRaisesRegex(SchemaError, "step0 SGD STORE"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, edges=tuple(edge for edge in graph.edges
                                           if edge is not victim)), "missing_tp3_version",
            )


if __name__ == "__main__":
    unittest.main()
