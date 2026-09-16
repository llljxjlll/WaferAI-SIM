"""TP4 reverse graph must preserve each parameter shard and unfused tape."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.discover_fusion import discover_fusion_candidates
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from llm.frontend.wafer_frontend.passes.validate_fusion import FusionSemanticValidator
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.ir0 import EdgeKind, OpKind, OpPhase, StateAccessMode
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _spec


class FullDenseTP4SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = build_flexible_dense_train_plan(_spec(1, 4), RectMeshSpec(1, 4))
        cls.graph = build_full_dense_training_two_step_ir0(cls.plan)

    def test_every_tp_shard_has_real_reverse_parameter_and_version_lineage(self):
        graph = self.graph
        DenseIR0Validator.validate(graph, "tp4")
        FusionSemanticValidator.validate(graph, "tp4")
        self.assertEqual(len(graph.persistent_states), 60)
        self.assertEqual(len(graph.nodes), 382)
        self.assertEqual(len(graph.state_accesses), 432)
        by_node = {node.id: node for node in graph.nodes}
        state_by_shard = {(s.identity.tensor_ref, s.identity.shard_index): s.id
                          for s in graph.persistent_states}
        reads = {(a.node_ref, a.state_ref, a.rank) for a in graph.state_accesses
                 if a.mode is StateAccessMode.READ}
        for step in (0, 1):
            nodes = tuple(node for node in graph.nodes if f"::step{step}" in node.id)
            self.assertEqual(sum(n.kind is OpKind.CE_FORWARD for n in nodes), 1)
            self.assertEqual(sum(n.kind is OpKind.CE_BACKWARD for n in nodes), 1)
            self.assertEqual(sum(n.phase is OpPhase.WGRAD for n in nodes), 60)
            self.assertEqual(sum(n.kind is OpKind.OPTIMIZER_UPDATE for n in nodes), 60)
            for t in self.plan.parameter_templates:
                wgrad = f"{t.wgrad_ref}::step{step}"
                sgd = f"sgd_update::{t.tensor_ref}::tp{t.tp_shard_index}::step{step}"
                self.assertIn(wgrad, by_node)
                self.assertIn(sgd, by_node)
                self.assertIn((wgrad, state_by_shard[(t.tensor_ref, t.tp_shard_index)],
                               t.tp_shard_index), reads)
        edges = tuple(e for e in graph.edges if e.kind is EdgeKind.CONTROL and
                      e.source_node.startswith("sgd_update::") and
                      e.source_node.endswith("::step0"))
        self.assertEqual(len(edges), 60)
        self.assertEqual(len({(e.source_node, e.destination_node) for e in edges}), 60)

    def test_only_tape_safe_fusion_candidates_remain(self):
        graph = self.graph
        self.assertEqual(graph.fusion_candidates, discover_fusion_candidates(graph))
        self.assertEqual(len(graph.fusion_candidates), 8)
        for candidate in graph.fusion_candidates:
            self.assertEqual(len(candidate.boundary_outputs), 1)
            shared = next(value for value in graph.values if value.producer == candidate.members[0])
            self.assertEqual(shared.consumers, (candidate.members[1],))
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            FusionSemanticValidator.validate(replace(graph, fusion_candidates=()), "missing")

    def test_removed_tp_state_read_or_version_edge_is_rejected(self):
        graph = self.graph
        shard = next(a for a in graph.state_accesses
                     if "::step1" in a.node_ref and a.rank == 3 and
                     a.node_ref.startswith("sgd_update::") and
                     a.mode is StateAccessMode.READ_WRITE)
        with self.assertRaises(SchemaError):
            DenseIR0Validator.validate(replace(
                graph, state_accesses=tuple(a for a in graph.state_accesses if a != shard)), "missing")
        edge = next(e for e in graph.edges if e.kind is EdgeKind.CONTROL and
                    e.source_node.startswith("sgd_update::") and "::tp3::step0" in e.source_node)
        with self.assertRaisesRegex(SchemaError, "step0 SGD STORE"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, edges=tuple(e for e in graph.edges if e != edge)), "missing")
