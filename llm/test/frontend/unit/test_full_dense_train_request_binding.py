from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_train_request_binding import (
    bind_dense_full_train_request,
)
from llm.frontend.wafer_frontend.schema.full_dense_gradient_requirements import (
    build_dense_full_train_requirements,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadExecutionSpec,
    WorkloadFamily,
    WorkloadMeshSpec,
    WorkloadModelArchitecture,
    WorkloadModelSpec,
    WorkloadOptimizerKind,
    WorkloadOptimizerSpec,
    WorkloadParallelSpec,
    WorkloadRunRequest,
    WorkloadStepSpec,
    WorkloadTrainingSteps,
)
from llm.test.frontend.unit.test_flexible_dense_train import _spec


def _request(plan, **changes):
    model = plan.source_experiment.model
    original = {
        "family": WorkloadFamily.DENSE_TRAINING,
        "model": WorkloadModelSpec(
            architecture=WorkloadModelArchitecture.LLAMA_DENSE,
            vocabulary_size=model.V,
            hidden_size=model.H,
            intermediate_size=model.I,
            num_layers=model.L,
            num_attention_heads=model.NH,
            num_kv_heads=model.KVH,
            head_dim=model.DH,
            max_sequence_length=model.max_position_embeddings,
            dtype=model.dtype,
        ),
        "steps": WorkloadStepSpec(
            training=WorkloadTrainingSteps(
                step_count=2,
                global_batch_size=plan.spec.dp_degree,
                micro_batch_size=1,
                micro_batch_count=1,
                sequence_length=plan.spec.tp_degree,
            )
        ),
        "mesh": WorkloadMeshSpec(plan.spec.mesh.rows, plan.spec.mesh.columns),
        "parallel": WorkloadParallelSpec(tp=plan.spec.tp_degree, dp=plan.spec.dp_degree),
        "optimizer": WorkloadOptimizerSpec(
            kind=WorkloadOptimizerKind.SGD,
            learning_rate=plan.spec.learning_rate,
        ),
        "execution": WorkloadExecutionSpec(independent_repeats=2),
    }
    return WorkloadRunRequest.create(**(original | changes))


class DenseTrainRequestBindingTest(unittest.TestCase):
    def test_two_step_public_case_bound_to_actual_source_and_oracle(self) -> None:
        plan = build_flexible_dense_train_plan(_spec(2, 2), RectMeshSpec(2, 2))
        requirements = build_dense_full_train_requirements(plan)
        request = _request(plan)
        binding = bind_dense_full_train_request(request, plan, requirements)
        self.assertEqual(binding.case_id, request.case_id)
        self.assertEqual(binding.source_plan_id, plan.id)
        self.assertEqual(binding, bind_dense_full_train_request(request, plan, requirements))
        self.assertFalse(hasattr(binding, "runtime_verified"))

    def test_changing_model_or_step_fails_even_with_valid_new_case_id(self) -> None:
        plan = build_flexible_dense_train_plan(_spec(2, 2), RectMeshSpec(2, 2))
        requirements = build_dense_full_train_requirements(plan)
        baseline = _request(plan)
        modified_model = _request(
            plan, model=replace(baseline.model, vocabulary_size=baseline.model.vocabulary_size + 8)
        )
        self.assertNotEqual(modified_model.case_id, baseline.case_id)
        with self.assertRaisesRegex(SchemaError, "model differs"):
            bind_dense_full_train_request(modified_model, plan, requirements)
        modified_steps = _request(
            plan, steps=WorkloadStepSpec(
                training=replace(baseline.steps.training, step_count=3)
            ),
        )
        with self.assertRaisesRegex(SchemaError, "steps/batch"):
            bind_dense_full_train_request(modified_steps, plan, requirements)

    def test_physical_remap_and_optimizer_drift_rejected(self) -> None:
        plan = build_flexible_dense_train_plan(_spec(2, 2), RectMeshSpec(2, 2))
        requirements = build_dense_full_train_requirements(plan)
        changed_placement = _request(
            plan, parallel=WorkloadParallelSpec(
                tp=2, dp=2, active_die_ids=(2, 0, 3, 1)
            ),
        )
        with self.assertRaisesRegex(SchemaError, "mapping differs"):
            bind_dense_full_train_request(changed_placement, plan, requirements)
        wrong_lr = _request(
            plan, optimizer=WorkloadOptimizerSpec(
                kind=WorkloadOptimizerKind.SGD,
                learning_rate=plan.spec.learning_rate + 1.0e-3,
            ),
        )
        with self.assertRaisesRegex(SchemaError, "SGD contract"):
            bind_dense_full_train_request(wrong_lr, plan, requirements)

    def test_unimplemented_full_train_functional_request_fails_closed(self) -> None:
        plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
        requirements = build_dense_full_train_requirements(plan)
        request = _request(
            plan, execution=WorkloadExecutionSpec(
                timing=True, functional=True, independent_repeats=2
            ),
        )
        with self.assertRaisesRegex(UnsupportedFeatureError, "functional oracle"):
            bind_dense_full_train_request(request, plan, requirements)


if __name__ == "__main__":
    unittest.main()
