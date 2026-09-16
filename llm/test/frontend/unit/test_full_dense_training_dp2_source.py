"""DP2 training must synchronize genuine FP32 gradients before both SGD copies."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.action import FusionActionKind, _validate_rank_programs
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_ir0 import build_full_dense_training_two_step_ir0
from llm.frontend.wafer_frontend.passes.full_dense_training_dp2_routes import build_dense_dp2_route_plan
from llm.frontend.wafer_frontend.passes.load_fabric import hbm_address_spaces_from_data, physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.common import DType, MeshAxisName
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind, EdgeKind, GraphEdge, IR0, OpKind, ReduceOp, StateAccessMode,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec


class FullDenseTrainingDP2SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_flexible_dense_train_plan(_spec(2, 2), RectMeshSpec(2, 2))
        cls.graph = build_full_dense_training_two_step_ir0(cls.plan)

    def test_every_real_tp_gradient_has_two_dp_producers_and_sum_before_sgd(self) -> None:
        graph = self.graph
        DenseIR0Validator.validate(graph, "dp2")
        self.assertEqual(graph.producer_pass, "full_dense_training_two_step_dp2_source")
        self.assertEqual(len(graph.nodes), 322)
        self.assertEqual(len(graph.state_accesses), 320)
        self.assertEqual(Counter(a.rank for a in graph.state_accesses),
                         Counter({rank: 80 for rank in range(4)}))
        values = {value.id: value for value in graph.values}
        nodes = {node.id: node for node in graph.nodes}
        trainable_states = {
            (state.identity.tensor_ref, state.identity.shard_index): state
            for state in graph.persistent_states
        }
        for step in (0, 1):
            for template in self.plan.parameter_templates:
                wgrad_ref = f"{template.wgrad_ref}::step{step}"
                local_ref = f"{template.wgrad_ref}.output::step{step}"
                state = trainable_states[(template.tensor_ref, template.tp_shard_index)]
                sync_ref = f"dp_sync::{state.id}::tp{template.tp_shard_index}::step{step}"
                synchronized_ref = f"dp_sync::{state.id}::tp{template.tp_shard_index}.output::step{step}"
                sgd_ref = f"sgd_update::{template.tensor_ref}::tp{template.tp_shard_index}::step{step}"
                gradient, synchronized = values[local_ref], values[synchronized_ref]
                sync, sgd = nodes[sync_ref], nodes[sgd_ref]
                self.assertEqual(gradient.producer, wgrad_ref)
                self.assertEqual((gradient.dtype, gradient.sharding.partial),
                                 (DType.FP32, (MeshAxisName.DP,)))
                self.assertEqual((sync.kind, sync.inputs, sync.outputs),
                                 (OpKind.COLLECTIVE, (local_ref,), (synchronized_ref,)))
                self.assertEqual((sync.workload.collective, sync.workload.reduce_op),
                                 (CollectiveKind.ALL_REDUCE, ReduceOp.SUM))
                self.assertEqual((sync.workload.mesh_axes, sync.workload.rank_input_bytes),
                                 ((MeshAxisName.DP,), template.gradient_bytes))
                self.assertEqual((synchronized.dtype, synchronized.sharding.partial),
                                 (DType.FP32, ()))
                self.assertEqual(sgd.inputs[1], synchronized_ref)
                self.assertEqual({a.rank for a in graph.state_accesses
                                  if a.node_ref == sgd_ref
                                  and a.mode is StateAccessMode.READ_WRITE},
                                 set(template.owner_ranks))
        controls = tuple(edge for edge in graph.edges
                         if edge.kind is EdgeKind.CONTROL and ".store0_to." in edge.id)
        self.assertEqual(len(controls), 30)

    def test_dp_sync_is_required_and_cannot_consume_other_parameter_gradient(self) -> None:
        graph = self.graph
        nodes = list(graph.nodes)
        sync = next(node for node in nodes if node.id.startswith("dp_sync::"))
        original = next(value for value in graph.values
                        if value.id == sync.inputs[0])
        wrong_gradient = next(value.id for value in graph.values
                              if value.id.startswith("wgrad::")
                              and value.id.endswith(".output::step0")
                              and value.id != sync.inputs[0]
                              and value.shape == original.shape
                              and value.sharding == original.sharding)
        nodes[nodes.index(sync)] = replace(sync, inputs=(wrong_gradient,))
        consumers = {value.id: [] for value in graph.values}
        for node in nodes:
            for value_ref in node.inputs:
                consumers[value_ref].append(node.id)
        values = tuple(replace(value, consumers=tuple(consumers[value.id]))
                       for value in graph.values)
        data = tuple(
            GraphEdge(f"{value.id}.edge_to.{consumer}", EdgeKind.DATA,
                      value.producer, consumer, value.id)
            for value in values if value.producer is not None
            for consumer in value.consumers
        )
        controls = tuple(edge for edge in graph.edges
                         if edge.kind is EdgeKind.CONTROL)
        # Keep every IR0 data edge and value-consumer list internally valid:
        # the semantic validator must reject a wrong but physically plausible
        # FP32 partial producer.
        tampered = IR0.create(
            producer_pass=graph.producer_pass,
            job=graph.job, instances=graph.instances, nodes=tuple(nodes),
            values=values, edges=(*data, *controls),
            fusion_candidates=(), profile=graph.profile, train=graph.train,
            persistent_states=graph.persistent_states,
            state_accesses=graph.state_accesses,
        )
        with self.assertRaises(SchemaError):
            DenseIR0Validator.validate(tampered, "wrong_parameter")

    def test_placement_binds_each_replica_state_to_its_physical_die(self) -> None:
        graph = self.graph
        hardware = _hardware(2, 2)
        context = PlacementContext.create(
            producer_pass="dp2_exact_placement",
            fabric=physical_fabric_from_data(hardware),
            placement=self.plan.source_experiment.placement,
            hbm_address_spaces=hbm_address_spaces_from_data(hardware),
        )
        placed = place_train_forward_ir0(graph, context)
        self.assertEqual(len(placed.replicas), 2)
        for dp, replica in enumerate(placed.replicas):
            ir1 = replica.graph
            self.assertEqual(tuple(item.die_id for item in ir1.groups[0].placements),
                             (dp * 2, dp * 2 + 1))
            self.assertEqual({item.rank for item in ir1.state_accesses}, {0, 1})
            manifest = ir1.persistent_state_manifest
            self.assertIsNotNone(manifest)
            self.assertEqual({item.die_id for item in manifest.bindings},
                             {dp * 2, dp * 2 + 1})
            self.assertEqual(len(manifest.bindings), 30)
            self.assertEqual(sum(node.kind is OpKind.OPTIMIZER_UPDATE
                                 for node in ir1.nodes), 60)
        routes = build_dense_dp2_route_plan(self.plan, placed, context)
        routes.validate_against(self.plan, placed, context)
        self.assertEqual(len(routes.gradients), 60)
        self.assertEqual(
            tuple(tuple(item.die_id for item in group.placements)
                  for group in routes.dp_groups),
            ((0, 2), (1, 3)),
        )
        for gradient in routes.gradients:
            self.assertEqual(gradient.reduce_route.die_path[0],
                             routes.dp_groups[gradient.tp_shard].placements[1].die_id)
            self.assertEqual(gradient.reduce_route.die_path[-1],
                             routes.dp_groups[gradient.tp_shard].placements[0].die_id)
            self.assertEqual(gradient.broadcast_route.die_path[0],
                             routes.dp_groups[gradient.tp_shard].placements[0].die_id)
            self.assertEqual(gradient.broadcast_route.die_path[-1],
                             routes.dp_groups[gradient.tp_shard].placements[1].die_id)
        first = routes.gradients[0]
        for gradient in routes.gradients:
            actions = tuple(action for program in gradient.rank_programs
                            for action in program.actions)
            self.assertEqual(
                Counter(action.kind for action in actions),
                Counter({
                    FusionActionKind.LOCAL_COPY: 1,
                    FusionActionKind.SEND: 2,
                    FusionActionKind.RECV: 2,
                    FusionActionKind.WAIT: 2,
                    FusionActionKind.REDUCE: 1,
                }),
            )
            self.assertEqual(
                next(action for action in actions
                     if action.kind is FusionActionKind.REDUCE).reduction.input_ranks,
                (0, 1),
            )
            _validate_rank_programs(
                gradient.rank_programs, (gradient.chunk,),
                path="dp2_true_transport",
            )
        missing_child_send = replace(
            first.rank_programs[1],
            actions=first.rank_programs[1].actions[1:],
        )
        with self.assertRaisesRegex(SchemaError, "one SEND and one RECV"):
            _validate_rank_programs(
                (first.rank_programs[0], missing_child_send),
                (first.chunk,), path="dp2_missing_dte",
            )
        with self.assertRaisesRegex(SchemaError, "gradient route/source/owner/bytes"):
            replace(routes, gradients=(
                replace(first, gradient_bytes=2048), *routes.gradients[1:],
            )).validate_against(self.plan, placed, context)


if __name__ == "__main__":
    unittest.main()
