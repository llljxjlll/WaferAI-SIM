from __future__ import annotations

import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadCapabilityLevel,
    WorkloadExecutionSpec,
    WorkloadFamily,
)
from llm.frontend.wafer_frontend.workload_capability_evidence import (
    WorkloadCapabilityArtifact,
    WorkloadCapabilityArtifactKind,
    WorkloadCapabilityDerivation,
    WorkloadCapabilityEvidence,
    WorkloadEvidenceScope,
    build_workload_capability_from_evidence,
)
from llm.test.frontend.unit.test_workload_materialization import _capability, _request


SOURCE_DIGEST = "1" * 64
BINARY_DIGEST = "2" * 64
TOOLCHAIN_DIGEST = "3" * 64
EXECUTION_DIGEST = "e" * 64


def _evidence(
    request,
    *,
    scope: WorkloadEvidenceScope = WorkloadEvidenceScope.FULL_MODEL,
    kinds: tuple[WorkloadCapabilityArtifactKind, ...] = tuple(
        WorkloadCapabilityArtifactKind
    ),
    executions: tuple[str, ...] = (EXECUTION_DIGEST, EXECUTION_DIGEST),
) -> WorkloadCapabilityEvidence:
    artifacts = tuple(
        WorkloadCapabilityArtifact.create(
            kind=kind,
            artifact_digest=f"{index + 4:x}" * 64,
        )
        for index, kind in enumerate(kinds)
    )
    return WorkloadCapabilityEvidence.create(
        request=request,
        scope=scope,
        source_digest=SOURCE_DIGEST,
        binary_digest=BINARY_DIGEST,
        toolchain_digest=TOOLCHAIN_DIGEST,
        artifacts=artifacts,
        independent_execution_digests=executions,
    )


def _build(*evidence: WorkloadCapabilityEvidence) -> WorkloadCapabilityDerivation:
    return build_workload_capability_from_evidence(
        tuple(evidence),
        max_mesh_rows=10,
        max_mesh_columns=10,
        max_mesh_ranks=100,
    )


def _recreate(request, *, execution=None):
    return type(request).create(
        family=request.family,
        model=request.model,
        steps=request.steps,
        mesh=request.mesh,
        parallel=request.parallel,
        memory=request.memory,
        optimizer=request.optimizer,
        execution=execution or request.execution,
    )


def _reasons(derivation, request) -> tuple[str, ...]:
    return derivation.readiness_reasons(
        request,
        source_digest=SOURCE_DIGEST,
        binary_digest=BINARY_DIGEST,
        toolchain_digest=TOOLCHAIN_DIGEST,
    )


class WorkloadCapabilityEvidenceTest(unittest.TestCase):
    def test_legal_full_model_evidence_is_ready_and_round_trips(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        derivation = _build(_evidence(request))

        derivation.capability.require_supported(request)
        self.assertFalse(_reasons(derivation, request))
        restored = loads_dataclass(
            WorkloadCapabilityDerivation,
            canonical_json(derivation),
        )
        self.assertEqual(restored, derivation)
        self.assertEqual(restored.digest, derivation.digest)

    def test_cross_case_evidence_cannot_make_request_ready(self) -> None:
        measured = _request(WorkloadFamily.DENSE_INFERENCE)
        requested = _recreate(
            measured,
            execution=WorkloadExecutionSpec(timing=True, functional=True),
        )
        derivation = _build(_evidence(measured))

        self.assertEqual(_reasons(derivation, requested), ("exact_case_evidence",))

    def test_missing_runtime_artifact_cannot_upgrade_runtime(self) -> None:
        request = _request(WorkloadFamily.MOE_INFERENCE)
        kinds = tuple(
            kind
            for kind in WorkloadCapabilityArtifactKind
            if kind is not WorkloadCapabilityArtifactKind.RUNTIME
        )
        derivation = _build(_evidence(request, kinds=kinds))
        family = derivation.capability.families[tuple(WorkloadFamily).index(request.family)]

        self.assertIs(family.runtime, WorkloadCapabilityLevel.NOT_MEASURED)
        self.assertIn("runtime_artifact", _reasons(derivation, request))

    def test_motif_only_evidence_cannot_upgrade_full_model(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        derivation = _build(
            _evidence(request, scope=WorkloadEvidenceScope.MOTIF)
        )
        family = derivation.capability.families[tuple(WorkloadFamily).index(request.family)]

        self.assertIs(family.motif, WorkloadCapabilityLevel.SUPPORTED)
        self.assertIs(family.full_model, WorkloadCapabilityLevel.NOT_MEASURED)
        self.assertIn("full_model_evidence", _reasons(derivation, request))

    def test_single_execution_cannot_upgrade_readiness(self) -> None:
        request = _request(WorkloadFamily.DENSE_TRAINING)
        derivation = _build(
            _evidence(request, executions=(EXECUTION_DIGEST,))
        )
        family = derivation.capability.families[tuple(WorkloadFamily).index(request.family)]

        self.assertIs(family.repeatability, WorkloadCapabilityLevel.NOT_MEASURED)
        self.assertIn("independent_executions", _reasons(derivation, request))

    def test_required_functional_and_capacity_artifacts_gate_readiness(self) -> None:
        base = _request(WorkloadFamily.DENSE_INFERENCE)
        request = _recreate(
            base,
            execution=replace(base.execution, functional=True),
        )
        for missing, reason in (
            (WorkloadCapabilityArtifactKind.FUNCTIONAL, "functional_artifact"),
            (WorkloadCapabilityArtifactKind.CAPACITY, "capacity_artifact"),
        ):
            with self.subTest(missing=missing):
                kinds = tuple(
                    kind for kind in WorkloadCapabilityArtifactKind if kind is not missing
                )
                derivation = _build(_evidence(request, kinds=kinds))
                self.assertIn(reason, _reasons(derivation, request))

    def test_derivation_rejects_caller_substituted_supported_capability(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        derivation = _build(_evidence(request))
        forged = replace(
            derivation,
            capability=_capability(supported=True),
        )
        with self.assertRaisesRegex(SchemaError, "not evidence-derived"):
            forged.validate()


if __name__ == "__main__":
    unittest.main()
