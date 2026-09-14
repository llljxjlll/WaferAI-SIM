from __future__ import annotations

import json
import unittest
from dataclasses import replace

from llm.frontend import wafer_frontend
from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.workload_run import (
    WORKLOAD_RUN_REQUEST_SCHEMA_VERSION,
    WorkloadCapabilityLevel,
    WorkloadExecutionSpec,
    WorkloadFamily,
    WorkloadFamilyCapability,
    WorkloadInferenceSteps,
    WorkloadMemoryMode,
    WorkloadMemoryPolicy,
    WorkloadMeshSpec,
    WorkloadModelArchitecture,
    WorkloadModelSpec,
    WorkloadOptimizerKind,
    WorkloadOptimizerSpec,
    WorkloadParallelSpec,
    WorkloadRunCapability,
    WorkloadRunRequest,
    WorkloadStepSpec,
    WorkloadTrainingSteps,
)


def _model(*, moe: bool) -> WorkloadModelSpec:
    return WorkloadModelSpec(
        architecture=(
            WorkloadModelArchitecture.LLAMA_MOE
            if moe
            else WorkloadModelArchitecture.LLAMA_DENSE
        ),
        vocabulary_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=4,
        head_dim=4,
        max_sequence_length=128,
        dtype=DType.FP16,
        num_experts=4 if moe else 0,
        experts_per_token=1 if moe else 0,
    )


def _request(family: WorkloadFamily) -> WorkloadRunRequest:
    training = family.is_training
    moe = family.is_moe
    mesh = WorkloadMeshSpec(2, 4 if training and moe else 2)
    parallel = WorkloadParallelSpec(
        tp=2,
        dp=2 if training else 1,
        ep=2 if moe else 1,
    )
    if training:
        steps = WorkloadStepSpec(
            training=WorkloadTrainingSteps(
                step_count=2,
                global_batch_size=2,
                micro_batch_size=1,
                micro_batch_count=1,
                sequence_length=8,
            )
        )
        optimizer = WorkloadOptimizerSpec(
            kind=WorkloadOptimizerKind.SGD,
            learning_rate=0.01,
        )
    else:
        steps = WorkloadStepSpec(
            inference=WorkloadInferenceSteps(
                prefill_tokens=8,
                decode_steps=2,
                request_count=2,
            )
        )
        optimizer = None
    return WorkloadRunRequest.create(
        family=family,
        model=_model(moe=moe),
        steps=steps,
        mesh=mesh,
        parallel=parallel,
        optimizer=optimizer,
    )


def _family_capability(
    family: WorkloadFamily,
    *,
    functional: WorkloadCapabilityLevel = WorkloadCapabilityLevel.SUPPORTED,
    external_offload: WorkloadCapabilityLevel = WorkloadCapabilityLevel.SUPPORTED,
) -> WorkloadFamilyCapability:
    supported = WorkloadCapabilityLevel.SUPPORTED
    return WorkloadFamilyCapability(
        family=family,
        full_model=supported,
        motif=supported,
        baseline=supported,
        optimized=WorkloadCapabilityLevel.NOT_MEASURED,
        lowering=supported,
        runtime=supported,
        timing=supported,
        functional=functional,
        capacity=supported,
        multi_step=supported,
        remote_hbm=supported,
        external_offload=external_offload,
        sgd_optimizer=supported,
        adamw_optimizer=WorkloadCapabilityLevel.NOT_MEASURED,
        repeatability=supported,
    )


def _capability(
    *,
    functional: WorkloadCapabilityLevel = WorkloadCapabilityLevel.SUPPORTED,
    external_offload: WorkloadCapabilityLevel = WorkloadCapabilityLevel.SUPPORTED,
) -> WorkloadRunCapability:
    return WorkloadRunCapability.create(
        max_mesh_rows=10,
        max_mesh_columns=10,
        max_mesh_ranks=100,
        families=tuple(
            _family_capability(
                family,
                functional=functional,
                external_offload=external_offload,
            )
            for family in WorkloadFamily
        ),
    )


class WorkloadRunSchemaTest(unittest.TestCase):
    def test_all_four_families_round_trip_with_stable_ids(self) -> None:
        for family in WorkloadFamily:
            with self.subTest(family=family):
                request = _request(family)
                decoded = loads_dataclass(
                    WorkloadRunRequest,
                    canonical_json(request),
                    path="request",
                )
                self.assertEqual(decoded, request)
                self.assertEqual(decoded.case_id, _request(family).case_id)
                self.assertEqual(len(decoded.digest), 64)

    def test_semantic_change_changes_id_and_stale_id_fails_closed(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        assert request.steps.inference is not None
        changed_steps = replace(
            request.steps,
            inference=replace(request.steps.inference, decode_steps=3),
        )
        changed = WorkloadRunRequest.create(
            family=request.family,
            model=request.model,
            steps=changed_steps,
            mesh=request.mesh,
            parallel=request.parallel,
        )
        self.assertNotEqual(request.case_id, changed.case_id)
        with self.assertRaisesRegex(SchemaError, "unstable case id"):
            replace(request, steps=changed_steps).validate()

    def test_strict_serde_rejects_missing_unknown_enum_and_version(self) -> None:
        raw = json.loads(canonical_json(_request(WorkloadFamily.DENSE_INFERENCE)))
        raw["unexpected"] = True
        with self.assertRaisesRegex(SchemaError, "unknown field"):
            loads_dataclass(WorkloadRunRequest, json.dumps(raw), path="request")

        raw = json.loads(canonical_json(_request(WorkloadFamily.DENSE_INFERENCE)))
        del raw["model"]
        with self.assertRaisesRegex(SchemaError, "missing required field"):
            loads_dataclass(WorkloadRunRequest, json.dumps(raw), path="request")

        raw = json.loads(canonical_json(_request(WorkloadFamily.DENSE_INFERENCE)))
        raw["family"] = "dense-infer"
        with self.assertRaisesRegex(SchemaError, "unknown value"):
            loads_dataclass(WorkloadRunRequest, json.dumps(raw), path="request")

        raw = json.loads(canonical_json(_request(WorkloadFamily.DENSE_INFERENCE)))
        raw["schema_version"] = "wafer_frontend.workload_run_request/v0"
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            loads_dataclass(WorkloadRunRequest, json.dumps(raw), path="request")

    def test_family_steps_optimizer_and_parallel_combinations_are_strict(self) -> None:
        infer = _request(WorkloadFamily.DENSE_INFERENCE)
        with self.assertRaisesRegex(SchemaError, "absent for inference"):
            WorkloadRunRequest.create(
                family=infer.family,
                model=infer.model,
                steps=infer.steps,
                mesh=infer.mesh,
                parallel=infer.parallel,
                optimizer=WorkloadOptimizerSpec(
                    WorkloadOptimizerKind.SGD,
                    0.01,
                ),
            )
        with self.assertRaisesRegex(SchemaError, "Dense workloads require ep=1"):
            WorkloadRunRequest.create(
                family=infer.family,
                model=infer.model,
                steps=infer.steps,
                mesh=infer.mesh,
                parallel=WorkloadParallelSpec(tp=1, ep=2),
            )
        with self.assertRaisesRegex(SchemaError, "family and model architecture"):
            WorkloadRunRequest.create(
                family=WorkloadFamily.MOE_INFERENCE,
                model=infer.model,
                steps=infer.steps,
                mesh=infer.mesh,
                parallel=WorkloadParallelSpec(tp=2, ep=2),
            )
        with self.assertRaisesRegex(SchemaError, "AdamW requires"):
            WorkloadOptimizerSpec(
                WorkloadOptimizerKind.ADAMW,
                0.001,
            ).validate()

    def test_memory_policy_and_explicit_active_die_mapping_are_strict(self) -> None:
        with self.assertRaisesRegex(SchemaError, "external_tier_ref"):
            WorkloadMemoryPolicy(
                mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD
            ).validate()
        with self.assertRaisesRegex(SchemaError, "duplicate die"):
            WorkloadParallelSpec(
                tp=2,
                active_die_ids=(1, 1),
            ).validate_against_mesh(WorkloadMeshSpec(2, 2))
        with self.assertRaisesRegex(SchemaError, "outside the mesh"):
            WorkloadParallelSpec(
                tp=2,
                active_die_ids=(0, 4),
            ).validate_against_mesh(WorkloadMeshSpec(2, 2))

    def test_capability_round_trip_and_dependency_rules(self) -> None:
        capability = _capability()
        self.assertEqual(
            loads_dataclass(
                WorkloadRunCapability,
                canonical_json(capability),
                path="capability",
            ),
            capability,
        )
        self.assertEqual(capability.unsupported_requirements(_request(
            WorkloadFamily.MOE_TRAINING
        )), ())
        broken = replace(
            capability.families[0],
            lowering=WorkloadCapabilityLevel.UNSUPPORTED,
        )
        with self.assertRaisesRegex(SchemaError, "requires lowering"):
            broken.validate()
        with self.assertRaisesRegex(SchemaError, "canonical order"):
            WorkloadRunCapability.create(
                max_mesh_rows=10,
                max_mesh_columns=10,
                max_mesh_ranks=100,
                families=tuple(reversed(capability.families)),
            )

    def test_capability_does_not_promote_unmeasured_or_schema_only(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        request = WorkloadRunRequest.create(
            family=request.family,
            model=request.model,
            steps=request.steps,
            mesh=request.mesh,
            parallel=request.parallel,
            memory=WorkloadMemoryPolicy(
                mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
                external_tier_ref="host0",
            ),
            execution=WorkloadExecutionSpec(timing=True, functional=True),
        )
        capability = _capability(
            functional=WorkloadCapabilityLevel.NOT_MEASURED,
            external_offload=WorkloadCapabilityLevel.SCHEMA_ONLY,
        )
        self.assertEqual(
            capability.unsupported_requirements(request),
            ("family.functional", "family.external_offload"),
        )
        with self.assertRaisesRegex(
            UnsupportedFeatureError, "family.functional.*family.external_offload"
        ):
            capability.require_supported(request)

    def test_mesh_schema_is_not_the_measured_envelope(self) -> None:
        request = WorkloadRunRequest.create(
            family=WorkloadFamily.DENSE_INFERENCE,
            model=_model(moe=False),
            steps=WorkloadStepSpec(
                inference=WorkloadInferenceSteps(8, 0, 1)
            ),
            mesh=WorkloadMeshSpec(12, 12),
            parallel=WorkloadParallelSpec(tp=1),
        )
        self.assertIn("mesh.rows", _capability().unsupported_requirements(request))
        self.assertIn(
            "mesh.rank_count", _capability().unsupported_requirements(request)
        )

    def test_contract_is_exported_from_public_packages(self) -> None:
        self.assertIs(wafer_frontend.WorkloadRunRequest, WorkloadRunRequest)
        self.assertEqual(
            wafer_frontend.WORKLOAD_RUN_REQUEST_SCHEMA_VERSION,
            WORKLOAD_RUN_REQUEST_SCHEMA_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
