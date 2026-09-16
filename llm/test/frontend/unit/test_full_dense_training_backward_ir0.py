"""Typed complete one-step Dense backward source construction."""

from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
)
from llm.frontend.wafer_frontend.passes.full_dense_training_backward_ir0 import (
    build_full_dense_training_backward_ir0,
)
from llm.frontend.wafer_frontend.passes.fusion_partition import (
    partition_train_forward,
)
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import (
    schedule_train_forward,
)
from llm.frontend.wafer_frontend.passes.train_link_program import link_train
from llm.frontend.wafer_frontend.passes.train_lower_program import lower_train
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_train_forward
from llm.frontend.wafer_frontend.passes.train_global_action import (
    build_train_global_action,
)
from llm.frontend.wafer_frontend.policies.registry import (
    RegistryKind,
    production_registry,
)
from llm.frontend.wafer_frontend.passes.validate_ir0 import DenseIR0Validator
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.full_dense_gradient_requirements import (
    build_dense_full_train_requirements,
)
from llm.frontend.wafer_frontend.schema.ir0 import OpKind, OpPhase
from llm.frontend.wafer_frontend.schema.n4 import (
    FusionPartitionContext,
    InterDiePlanningContext,
)
from llm.frontend.wafer_frontend.schema.n5 import (
    IntraDieSchedulingContext,
    ProjectToIR2Context,
)
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec


class FullDenseTrainingBackwardIr0Test(unittest.TestCase):
    def _build(self):
        plan = build_flexible_dense_train_plan(
            _spec(1, 1), RectMeshSpec(1, 1)
        )
        return plan, build_full_dense_training_backward_ir0(plan)

    def test_two_layer_graph_has_every_typed_reverse_and_wgrad(self) -> None:
        plan, graph = self._build()
        DenseIR0Validator.validate(graph, "full_dense_backward")
        requirements = build_dense_full_train_requirements(plan, steps=2)
        reverse = {
            node.id for node in graph.nodes
            if node.phase is OpPhase.DGRAD
            and node.kind is not OpKind.CE_BACKWARD
            and not node.id.startswith("gradient_sum::")
        }
        self.assertEqual(
            reverse, set(requirements.required_backbone_backward_refs)
        )
        self.assertEqual(
            {node.id for node in graph.nodes if node.phase is OpPhase.WGRAD},
            {template.wgrad_ref for template in plan.parameter_templates},
        )
        values = {value.id: value for value in graph.values}
        self.assertTrue(all(
            values[ref].dtype is DType.FP16
            for node in graph.nodes if node.phase is OpPhase.DGRAD
            for ref in node.outputs
        ))
        self.assertTrue(all(
            values[ref].dtype is DType.FP32
            for node in graph.nodes if node.phase is OpPhase.WGRAD
            for ref in node.outputs
        ))
        residuals = tuple(
            node for node in graph.nodes
            if node.kind is OpKind.RESIDUAL_BACKWARD
        )
        self.assertEqual(len(residuals), 4)
        self.assertTrue(all(
            len(node.outputs) == 2
            and all(values[ref].consumers for ref in node.outputs)
            for node in residuals
        ))

    def test_standard_ir1_projection_schedule_global_dag_is_source_backed(self) -> None:
        plan, graph = self._build()
        hardware = _hardware(1, 1)
        fabric = physical_fabric_from_data(hardware)
        spaces = hbm_address_spaces_from_data(hardware)
        producer = "test_full_dense_backward_lineage"
        placed = place_train_forward_ir0(
            graph,
            PlacementContext.create(
                producer_pass=producer,
                fabric=fabric,
                placement=plan.source_experiment.placement,
                hbm_address_spaces=spaces,
            ),
        )
        partitioned = partition_train_forward(
            placed, FusionPartitionContext.create(producer_pass=producer)
        )
        registry = production_registry()
        planned = plan_train_forward(
            partitioned,
            InterDiePlanningContext.create(
                producer_pass=producer,
                fused_policy=registry.instantiate(
                    RegistryKind.INTER_DIE, "naive"
                ).selection,
                standalone_policy=registry.instantiate(
                    RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather"
                ).selection,
            ),
        )
        projected = project_train_forward(
            planned,
            ProjectToIR2Context.create(
                producer_pass=producer, state_transfers=()
            ),
        )
        scheduled = schedule_train_forward(
            projected,
            IntraDieSchedulingContext.create(
                producer_pass=producer,
                policy=registry.instantiate(
                    RegistryKind.INTRA_DIE, "naive"
                ).selection,
            ),
        )
        global_action = build_train_global_action(scheduled)
        replica = global_action.replicas[0]
        origins = {action.member_id.removesuffix("__dp0")
                   for action in replica.global_dag.actions
                   if action.member_id is not None}
        required = {
            node.id for node in graph.nodes
            if node.phase in (OpPhase.DGRAD, OpPhase.WGRAD)
        }
        self.assertLessEqual(required, origins)
        self.assertEqual(
            len(replica.global_dag.actions),
            len(projected.replicas[0].projection.dags[0].tasks),
        )
        linked = link_train(lower_train(global_action))
        self.assertTrue(linked.manifest.fragments)

    def test_missing_reverse_and_fp32_activation_gradient_fail_closed(self) -> None:
        _, graph = self._build()
        victim = next(
            node for node in graph.nodes
            if node.kind is OpKind.ATTENTION_BACKWARD
        )
        with self.assertRaisesRegex(SchemaError, "omits or fabricates"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(
                    graph,
                    nodes=tuple(node for node in graph.nodes
                                if node.id != victim.id),
                ),
                "forged",
            )
        dgrad = next(
            node for node in graph.nodes
            if node.kind is OpKind.GEMM_INPUT_DX
        )
        forged_values = tuple(
            replace(value, dtype=DType.FP32)
            if value.id == dgrad.outputs[0] else value
            for value in graph.values
        )
        with self.assertRaisesRegex(SchemaError, "must remain FP16"):
            DenseIR0Validator._validate_full_dense_backward_job_contract(
                replace(graph, values=forged_values), "forged"
            )


if __name__ == "__main__":
    unittest.main()
