"""Official full Dense AdamW source must project five physical state paths."""
from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_adamw_ir0 import build_full_dense_training_two_step_adamw_ir0
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_train_forward
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import schedule_train_forward
from llm.frontend.wafer_frontend.passes.load_fabric import hbm_address_spaces_from_data, physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_train_forward
from llm.frontend.wafer_frontend.policies.registry import RegistryKind, production_registry
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.ir0 import OpKind, StateAccessMode
from llm.frontend.wafer_frontend.schema.dense_adamw_state_version import dense_adamw_two_step_state_access_pairs
from llm.frontend.wafer_frontend.schema.ir2 import (
    BufferOwnership, SemanticTaskKind, StateIoOrigin, canonical_state_task_id,
)
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext, InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.n5 import (
    IntraDieSchedulingContext, ProjectToIR2Context,
    _validate_full_dense_sgd_projection_state,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec


class FullDenseTwoStepAdamwN5Test(unittest.TestCase):
    @classmethod
    @builder_validation_session()
    def setUpClass(cls) -> None:
        producer = "full_dense_training_two_step_adamw_native"
        plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
        source = build_full_dense_training_two_step_adamw_ir0(plan)
        hardware = _hardware(1, 1)
        placed = place_train_forward_ir0(source, PlacementContext.create(
            producer_pass=producer,
            fabric=physical_fabric_from_data(hardware),
            placement=plan.source_experiment.placement,
            hbm_address_spaces=hbm_address_spaces_from_data(hardware),
        ))
        partitioned = partition_train_forward(
            placed, FusionPartitionContext.create(producer_pass=producer),
        )
        registry = production_registry()
        planned = plan_train_forward(partitioned, InterDiePlanningContext.create(
            producer_pass=producer,
            fused_policy=registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
            standalone_policy=registry.instantiate(
                RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather",
            ).selection,
        ))
        projected = project_train_forward(planned, ProjectToIR2Context.create(
            producer_pass=producer, state_transfers=(),
        ))
        scheduled = schedule_train_forward(projected, IntraDieSchedulingContext.create(
            producer_pass=producer,
            policy=registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection,
        ))
        cls.scheduled = scheduled
        cls.ir1 = placed.replicas[0].graph
        cls.projection = projected.replicas[0].projection
        cls.schedule = scheduled.replicas[0].schedule_set.schedules[0]
        cls.planned = planned.replicas[0]

    def test_150_optimizer_state_writes_have_real_n5_dma_and_aliases(self) -> None:
        dag = self.projection.dags[0]
        self.assertEqual(len(self.ir1.nodes), 170)
        self.assertEqual(len(self.ir1.persistent_state_manifest.declarations), 75)
        self.assertEqual(len(self.ir1.state_accesses), 200)
        self.assertEqual(sum(task.kind is SemanticTaskKind.COMP for task in dag.tasks), 170)
        self.assertEqual(sum(task.kind is SemanticTaskKind.DMA_IN for task in dag.tasks), 200)
        self.assertEqual(sum(task.kind is SemanticTaskKind.DMA_OUT for task in dag.tasks), 150)
        self.assertEqual(sum(binding.ownership is BufferOwnership.ALIASED
                             for binding in self.schedule.buffer_bindings), 150)

    def test_75_source_bound_store0_to_load1_tasks(self) -> None:
        pairs = dense_adamw_two_step_state_access_pairs(self.ir1)
        self.assertEqual(len(pairs), 75)
        task_by_id = {task.id: task for task in self.projection.dags[0].tasks}
        for old_access, new_access in pairs:
            store_ref = canonical_state_task_id(old_access, SemanticTaskKind.DMA_OUT)
            load_ref = canonical_state_task_id(new_access, SemanticTaskKind.DMA_IN)
            self.assertEqual(task_by_id[load_ref].deps, (store_ref,))
            self.assertIsInstance(task_by_id[load_ref].origin_ref, StateIoOrigin)
            self.assertEqual(task_by_id[load_ref].dma.state_ref,
                             task_by_id[store_ref].dma.state_ref)
        access = next(item for item in self.ir1.state_accesses
                      if item.id == pairs[0][1])
        node0 = next(item for item in self.ir1.nodes
                     if item.id == next(old.node_ref for old in self.ir1.state_accesses
                                        if old.id == pairs[0][0]))
        node1 = next(item for item in self.ir1.nodes if item.id == access.node_ref)
        forged = replace(self.ir1, edges=tuple(
            edge for edge in self.ir1.edges
            if not (edge.source_node == node0.id and edge.destination_node == node1.id)
        ))
        with self.assertRaisesRegex(SchemaError, "source update0-to-update1 edge"):
            dense_adamw_two_step_state_access_pairs(forged)

    def test_missing_step_counter_write_or_wrong_owner_fails_closed(self) -> None:
        graph = self.ir1
        access = next(access for access in graph.state_accesses
                      if access.mode is StateAccessMode.READ_WRITE
                      and access.node_ref.startswith("adamw_update::")
                      and any(state.id == access.state_ref and
                              state.identity.kind.value == "optimizer_step"
                              for state in graph.persistent_state_manifest.declarations))
        missing = replace(graph, state_accesses=tuple(
            item for item in graph.state_accesses if item.id != access.id
        ))
        with self.assertRaisesRegex(SchemaError, "five distinct physical states"):
            _validate_full_dense_sgd_projection_state(
                missing, self.projection, "missing_step_counter",
            )
        other = next(item for item in graph.state_accesses
                     if item.node_ref != access.node_ref
                     and item.state_ref != access.state_ref
                     and item.mode is StateAccessMode.READ_WRITE
                     and any(state.id == item.state_ref and
                             state.identity.kind.value == "optimizer_step"
                             for state in graph.persistent_state_manifest.declarations))
        forged = replace(graph, state_accesses=tuple(
            replace(item, state_ref=other.state_ref)
            if item.id == access.id else item for item in graph.state_accesses
        ))
        with self.assertRaisesRegex(SchemaError, "its own weight/master/m/v/step"):
            _validate_full_dense_sgd_projection_state(
                forged, self.projection, "wrong_step_counter",
            )


if __name__ == "__main__":
    unittest.main()
