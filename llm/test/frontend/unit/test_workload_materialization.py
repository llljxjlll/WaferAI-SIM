from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.passes.validate_parallel_transport import (
    validate_parallel_transport_workload_bindings,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.e2e_workload_graph import E2EOperationKind
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryPlanExecution,
    MemoryTier,
    MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.parallel_transport import (
    ParallelCommunicationKind,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    loads_dataclass,
)
from llm.frontend.wafer_frontend.schema.workload_materialization import (
    WorkloadArtifactStatus,
    WorkloadMaterializationManifest,
    WorkloadMaterializationStatus,
)
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadCapabilityLevel,
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
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware


def _model(moe: bool) -> WorkloadModelSpec:
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
    parallel = WorkloadParallelSpec(
        tp=2,
        dp=2 if training and not moe else 1,
        ep=2 if moe else 1,
    )
    if training:
        steps = WorkloadStepSpec(
            training=WorkloadTrainingSteps(2, parallel.dp, 1, 1, 8)
        )
        optimizer = WorkloadOptimizerSpec(WorkloadOptimizerKind.SGD, 0.01)
    else:
        steps = WorkloadStepSpec(
            inference=WorkloadInferenceSteps(8, 2, 2)
        )
        optimizer = None
    return WorkloadRunRequest.create(
        family=family,
        model=_model(moe),
        steps=steps,
        mesh=WorkloadMeshSpec(2, 2),
        parallel=parallel,
        optimizer=optimizer,
    )


def _family_capability(
    family: WorkloadFamily, *, supported: bool
) -> WorkloadFamilyCapability:
    yes = WorkloadCapabilityLevel.SUPPORTED
    partial = WorkloadCapabilityLevel.SCHEMA_ONLY
    runtime = yes if supported else WorkloadCapabilityLevel.NOT_MEASURED
    return WorkloadFamilyCapability(
        family=family,
        full_model=yes if supported else partial,
        motif=yes,
        baseline=yes,
        optimized=WorkloadCapabilityLevel.NOT_MEASURED,
        lowering=yes if supported else partial,
        runtime=runtime,
        timing=runtime,
        functional=WorkloadCapabilityLevel.NOT_MEASURED,
        capacity=yes,
        multi_step=yes if supported else partial,
        remote_hbm=WorkloadCapabilityLevel.UNSUPPORTED,
        external_offload=WorkloadCapabilityLevel.SCHEMA_ONLY,
        sgd_optimizer=yes,
        adamw_optimizer=WorkloadCapabilityLevel.NOT_MEASURED,
        repeatability=WorkloadCapabilityLevel.NOT_MEASURED,
    )


def _capability(*, supported: bool) -> WorkloadRunCapability:
    return WorkloadRunCapability.create(
        max_mesh_rows=10,
        max_mesh_columns=10,
        max_mesh_ranks=100,
        families=tuple(
            _family_capability(family, supported=supported)
            for family in WorkloadFamily
        ),
    )


def _capacities(size: int = 1 << 20) -> tuple[MemoryTierCapacity, ...]:
    return tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{die}",
            base_address=0,
            capacity_bytes=size,
            alignment_bytes=16,
        )
        for die in range(4)
    )


class WorkloadMaterializationTest(unittest.TestCase):
    def test_four_families_produce_real_partial_manifests(self) -> None:
        expected_ops = {
            WorkloadFamily.DENSE_INFERENCE: {
                E2EOperationKind.QKV,
                E2EOperationKind.LOGITS,
            },
            WorkloadFamily.DENSE_TRAINING: {
                E2EOperationKind.LOSS,
                E2EOperationKind.SGD_UPDATE,
            },
            WorkloadFamily.MOE_INFERENCE: {
                E2EOperationKind.ROUTER,
                E2EOperationKind.DISPATCH,
            },
            WorkloadFamily.MOE_TRAINING: {
                E2EOperationKind.EXPERT_BACKWARD,
                E2EOperationKind.GRAD_DISPATCH,
            },
        }
        for family in WorkloadFamily:
            with self.subTest(family=family):
                manifest = materialize_workload_preflight(
                    _request(family),
                    _capability(supported=True),
                    capacities=_capacities(),
                )
                self.assertIs(manifest.status, WorkloadMaterializationStatus.PARTIAL)
                self.assertIs(
                    manifest.lowering_status,
                    WorkloadArtifactStatus.NOT_MATERIALIZED,
                )
                self.assertIs(
                    manifest.runtime_status,
                    WorkloadArtifactStatus.NOT_MATERIALIZED,
                )
                self.assertTrue(manifest.placement.ownership_domains)
                self.assertTrue(manifest.logical_graph.operations)
                self.assertEqual(manifest.logical_graph.request, manifest.request)
                self.assertEqual(
                    manifest.logical_graph_digest,
                    canonical_digest(manifest.logical_graph),
                )
                self.assertTrue(manifest.state_inventory)
                self.assertTrue(manifest.memory_plan.allocations)
                self.assertTrue(expected_ops[family].issubset(
                    {item.kind for item in manifest.logical_graph.operations}
                ))

    def test_manifest_round_trip_and_stable_digest(self) -> None:
        request = _request(WorkloadFamily.MOE_TRAINING)
        first = materialize_workload_preflight(
            request,
            _capability(supported=True),
            capacities=_capacities(),
        )
        second = materialize_workload_preflight(
            request,
            _capability(supported=True),
            capacities=tuple(reversed(_capacities())),
        )
        self.assertEqual(first, second)
        decoded = loads_dataclass(
            WorkloadMaterializationManifest,
            canonical_json(first),
            path="manifest",
        )
        self.assertEqual(decoded, first)
        self.assertEqual(decoded.digest, first.digest)

    def test_schema_only_capability_stays_unsupported_with_real_preflight(self) -> None:
        manifest = materialize_workload_preflight(
            _request(WorkloadFamily.DENSE_INFERENCE),
            _capability(supported=False),
            capacities=_capacities(),
        )
        self.assertIs(manifest.status, WorkloadMaterializationStatus.UNSUPPORTED)
        self.assertIn("family.full_model", manifest.unsupported_requirements)
        self.assertIn("family.runtime", manifest.unsupported_requirements)
        self.assertTrue(manifest.logical_graph.operations)
        self.assertTrue(manifest.memory_plan.allocations)

    def test_optional_transport_plan_binds_requests_and_placement(self) -> None:
        request = _request(WorkloadFamily.DENSE_TRAINING)
        fabric = physical_fabric_from_data(minimal_hardware(2, 2))
        manifest = materialize_workload_preflight(
            request,
            _capability(supported=True),
            capacities=_capacities(),
            fabric=fabric,
        )
        self.assertIsNotNone(manifest.transport_plan)
        assert manifest.transport_plan is not None
        self.assertEqual(
            manifest.transport_plan.placement_digest,
            manifest.placement.digest,
        )
        self.assertEqual(
            manifest.transport_plan.requests,
            manifest.transport_requests,
        )
        self.assertTrue(manifest.transport_plan.transfers)
        requests = {
            item.id: item for item in manifest.transport_requests
        }
        for transfer in manifest.transport_plan.transfers:
            parent = requests[transfer.request_id]
            self.assertEqual(
                (
                    transfer.workload_phase,
                    transfer.workload_step,
                    transfer.workload_layer,
                    transfer.logical_operation_ref,
                    transfer.payload_value_refs,
                    transfer.bytes,
                ),
                (
                    parent.workload_phase,
                    parent.workload_step,
                    parent.workload_layer,
                    parent.logical_operation_ref,
                    parent.payload_value_refs,
                    parent.transfer_bytes,
                ),
            )

    def test_transport_requests_follow_graph_communication_points(self) -> None:
        for family in WorkloadFamily:
            with self.subTest(family=family):
                manifest = materialize_workload_preflight(
                    _request(family),
                    _capability(supported=True),
                    capacities=_capacities(),
                )
                operations = {
                    item.id: item for item in manifest.logical_graph.operations
                }
                values = {
                    item.id: item for item in manifest.logical_graph.tensor_values
                }
                request_operation_kinds = set()
                for transport in manifest.transport_requests:
                    operation = operations[transport.logical_operation_ref]
                    request_operation_kinds.add(operation.kind)
                    self.assertEqual(
                        (
                            transport.workload_case_id,
                            transport.workload_request_digest,
                        ),
                        (manifest.request.case_id, manifest.request.digest),
                    )
                    self.assertEqual(
                        (
                            transport.workload_phase,
                            transport.workload_step,
                            transport.workload_layer,
                        ),
                        (operation.phase, operation.step, operation.layer),
                    )
                    self.assertTrue(transport.payload_value_refs)
                    self.assertTrue(
                        all(
                            values[value_ref].size_bytes
                            == transport.transfer_bytes
                            for value_ref in transport.payload_value_refs
                        )
                    )
                    self.assertEqual(
                        transport.is_noop,
                        transport.transfer_bytes == 0,
                    )

                self.assertIn(E2EOperationKind.QKV, request_operation_kinds)
                if family.is_training:
                    self.assertIn(
                        E2EOperationKind.GRADIENT_SYNC,
                        request_operation_kinds,
                    )
                    sync_steps = {
                        item.workload_step
                        for item in manifest.transport_requests
                        if item.kind is ParallelCommunicationKind.ALL_REDUCE
                    }
                    self.assertEqual(sync_steps, {0, 1})
                if family.is_moe:
                    self.assertTrue(
                        {
                            E2EOperationKind.DISPATCH,
                            E2EOperationKind.COMBINE,
                        }.issubset(request_operation_kinds)
                    )
                if family is WorkloadFamily.DENSE_INFERENCE:
                    phases = {
                        item.workload_phase
                        for item in manifest.transport_requests
                    }
                    self.assertEqual(phases, {"prefill", "decode"})
                if family is WorkloadFamily.MOE_INFERENCE:
                    self.assertTrue(
                        any(item.is_noop for item in manifest.transport_requests)
                    )

                validate_parallel_transport_workload_bindings(
                    manifest.transport_requests,
                    manifest.logical_graph,
                    manifest.placement,
                )

    def test_transport_oracle_rejects_forged_payload_bytes(self) -> None:
        manifest = materialize_workload_preflight(
            _request(WorkloadFamily.DENSE_INFERENCE),
            _capability(supported=True),
            capacities=_capacities(),
        )
        first = manifest.transport_requests[0]
        forged = (
            replace(first, transfer_bytes=first.transfer_bytes + 1),
            *manifest.transport_requests[1:],
        )
        with self.assertRaisesRegex(SchemaError, "payload bytes disagree"):
            validate_parallel_transport_workload_bindings(
                forged,
                manifest.logical_graph,
                manifest.placement,
            )

    def test_capacity_failure_is_reported_by_real_planner(self) -> None:
        with self.assertRaisesRegex(SchemaError, "capacity exceeded") as caught:
            materialize_workload_preflight(
                _request(WorkloadFamily.DENSE_TRAINING),
                _capability(supported=True),
                capacities=_capacities(16),
            )
        self.assertEqual(caught.exception.code, "memory_capacity_exceeded")

    def test_remote_hbm_and_missing_capacity_fail_closed(self) -> None:
        source = _request(WorkloadFamily.DENSE_INFERENCE)
        remote = WorkloadRunRequest.create(
            family=source.family,
            model=source.model,
            steps=source.steps,
            mesh=source.mesh,
            parallel=source.parallel,
            memory=WorkloadMemoryPolicy(mode=WorkloadMemoryMode.REMOTE_HBM),
        )
        with self.assertRaisesRegex(UnsupportedFeatureError, "not materialized"):
            materialize_workload_preflight(
                remote,
                _capability(supported=True),
                capacities=_capacities(),
            )
        with self.assertRaisesRegex(SchemaError, "has no hbm capacity") as caught:
            materialize_workload_preflight(
                source,
                _capability(supported=True),
                capacities=_capacities()[:1],
            )
        self.assertEqual(caught.exception.code, "memory_capacity_missing")

    def test_manifest_rejects_forged_graph_and_request_digests(self) -> None:
        manifest = materialize_workload_preflight(
            _request(WorkloadFamily.DENSE_INFERENCE),
            _capability(supported=True),
            capacities=_capacities(),
        )
        with self.assertRaisesRegex(SchemaError, "does not match logical graph"):
            replace(manifest, logical_graph_digest="0" * 64).validate()
        with self.assertRaisesRegex(SchemaError, "does not match request"):
            replace(manifest, request_digest="0" * 64).validate()

    def test_materialization_fails_when_independent_graph_oracle_rejects(self) -> None:
        with patch(
            "llm.frontend.wafer_frontend.passes.workload_materialization."
            "validate_e2e_workload_coverage",
            side_effect=SchemaError("oracle rejected graph", path="graph"),
        ):
            with self.assertRaisesRegex(SchemaError, "oracle rejected graph"):
                materialize_workload_preflight(
                    _request(WorkloadFamily.DENSE_INFERENCE),
                    _capability(supported=True),
                    capacities=_capacities(),
                )

    def test_external_plan_remains_schema_only(self) -> None:
        source = _request(WorkloadFamily.MOE_INFERENCE)
        request = WorkloadRunRequest.create(
            family=source.family,
            model=source.model,
            steps=source.steps,
            mesh=source.mesh,
            parallel=source.parallel,
            memory=WorkloadMemoryPolicy(
                mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
                external_tier_ref="host:0",
            ),
        )
        external = MemoryTierCapacity.create(
            tier=MemoryTier.EXTERNAL,
            location_ref="host:0",
            base_address=0,
            capacity_bytes=1 << 20,
            alignment_bytes=16,
        )
        manifest = materialize_workload_preflight(
            request,
            _capability(supported=True),
            capacities=(*_capacities(), external),
        )
        self.assertIs(
            manifest.memory_plan.execution,
            MemoryPlanExecution.SCHEMA_ONLY_EXTERNAL,
        )
        self.assertIs(manifest.status, WorkloadMaterializationStatus.UNSUPPORTED)
        self.assertIn("family.external_offload", manifest.unsupported_requirements)


if __name__ == "__main__":
    unittest.main()
