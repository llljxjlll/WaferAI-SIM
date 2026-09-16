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
        from llm.frontend.wafer_frontend.passes.fusion_partition import partition_train_forward
        from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
        from llm.frontend.wafer_frontend.passes.intra_die_schedule import schedule_train_forward
        from llm.frontend.wafer_frontend.passes.load_fabric import (
            hbm_address_spaces_from_data, physical_fabric_from_data,
        )
        from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
        from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_train_forward
        from llm.frontend.wafer_frontend.passes.train_global_action import build_train_global_action
        from llm.frontend.wafer_frontend.passes.train_link_program import link_train
        from llm.frontend.wafer_frontend.passes.train_lower_program import lower_train
        from llm.frontend.wafer_frontend.policies.registry import RegistryKind, production_registry
        from llm.frontend.wafer_frontend.schema.n4 import (
            FusionPartitionContext, InterDiePlanningContext,
        )
        from llm.frontend.wafer_frontend.schema.n5 import (
            IntraDieSchedulingContext, ProjectToIR2Context,
        )
        from llm.frontend.wafer_frontend.schema.placement import PlacementContext

        plan, graph = self._build()
        hardware = _hardware(1, 1)
        producer = "test_full_dense_two_step_native"
        placed = place_train_forward_ir0(graph, PlacementContext.create(
            producer_pass=producer, fabric=physical_fabric_from_data(hardware),
            placement=plan.source_experiment.placement,
            hbm_address_spaces=hbm_address_spaces_from_data(hardware),
        ))
        partitioned = partition_train_forward(placed,
            FusionPartitionContext.create(producer_pass=producer))
        registry = production_registry()
        planned = plan_train_forward(partitioned, InterDiePlanningContext.create(
            producer_pass=producer,
            fused_policy=registry.instantiate(RegistryKind.INTER_DIE,
                                               "naive").selection,
            standalone_policy=registry.instantiate(
                RegistryKind.STANDALONE_COLLECTIVE,
                "direct_all_gather").selection,
        ))
        projected = project_train_forward(planned, ProjectToIR2Context.create(
            producer_pass=producer, state_transfers=()))
        scheduled = schedule_train_forward(projected,
            IntraDieSchedulingContext.create(
                producer_pass=producer,
                policy=registry.instantiate(RegistryKind.INTRA_DIE,
                                            "naive").selection,
            ))
        source_dag = build_train_global_action(scheduled)
        linked = link_train(lower_train(source_dag))
        physical = Counter(
            record.opcode for fragment in linked.manifest.fragments
            for stream in fragment.core_streams for record in stream.records
        )
        self.assertEqual(physical[RecordOpcode.CROSS_ENTROPY_BACKWARD], 2)
        self.assertEqual(physical[RecordOpcode.SGD_UPDATE], 30)
        self.assertEqual(physical[RecordOpcode.LSU_STORE], 30)
        self.assertEqual(physical[RecordOpcode.LSU_LOAD], 80)
        self.assertEqual(len(linked.manifest.fragments), 280)


if __name__ == "__main__":
    unittest.main()
