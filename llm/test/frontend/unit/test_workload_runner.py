from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.workload_capability_evidence import (
    WorkloadCapabilityArtifact,
    WorkloadCapabilityArtifactKind,
    WorkloadCapabilityEvidence,
    WorkloadEvidenceScope,
    build_workload_capability_from_evidence,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, load_json_dataclass
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadExecutionSpec,
    WorkloadFamily,
)
from llm.frontend.wafer_frontend.workload_runner import (
    WorkloadBackendAdapter,
    WorkloadRunnerError,
    WorkloadRunnerReadiness,
    WorkloadRunnerStage,
    WorkloadRunnerState,
    WorkloadRunnerStatus,
    WorkloadStageArtifact,
    run_workload,
)
from llm.test.frontend.unit.test_workload_materialization import (
    _capacities,
    _capability,
    _request,
)


def _evidence_derivation(request):
    artifacts = tuple(
        WorkloadCapabilityArtifact.create(
            kind=kind,
            artifact_digest=character * 64,
        )
        for kind, character in (
            (WorkloadCapabilityArtifactKind.LOWERING, "3"),
            (WorkloadCapabilityArtifactKind.RUNTIME, "4"),
            (WorkloadCapabilityArtifactKind.CAPACITY, "5"),
            (WorkloadCapabilityArtifactKind.FUNCTIONAL, "6"),
        )
    )
    evidence = WorkloadCapabilityEvidence.create(
        request=request,
        scope=WorkloadEvidenceScope.FULL_MODEL,
        source_digest="1" * 64,
        binary_digest="2" * 64,
        toolchain_digest="a" * 64,
        artifacts=artifacts,
        independent_execution_digests=("e" * 64, "e" * 64),
    )
    return build_workload_capability_from_evidence(
        (evidence,),
        max_mesh_rows=10,
        max_mesh_columns=10,
        max_mesh_ranks=100,
    )


def _adapter(
    *,
    fail_at: WorkloadRunnerStage | None = None,
    diverge_runtime: bool = False,
    tamper_after_capture: WorkloadRunnerStage | None = None,
    toolchain_digest: str = "a" * 64,
) -> WorkloadBackendAdapter:
    def callback(stage: WorkloadRunnerStage):
        def invoke(context):
            if stage is fail_at:
                raise RuntimeError(f"injected {stage.value} failure")
            output = {
                "stage": stage.value,
                "manifest_digest": context.manifest.digest,
                "previous": context.input_digest,
                "runtime_variant": (
                    context.execution_index
                    if diverge_runtime and stage is WorkloadRunnerStage.RUNTIME
                    else 0
                ),
            }
            artifact_dir = context.work_dir / "artifacts"
            artifact_dir.mkdir(exist_ok=True)
            relative_path = f"artifacts/{stage.value}.json"
            payload_path = context.work_dir / relative_path
            payload_path.write_text(canonical_json(output) + "\n", encoding="utf-8")
            artifact = WorkloadStageArtifact.create(
                stage=stage,
                input_digest=context.input_digest,
                work_dir=context.work_dir,
                relative_paths=(relative_path,),
                one_shot_workload_end=stage is WorkloadRunnerStage.RUNTIME,
            )
            if stage is tamper_after_capture:
                payload_path.write_bytes(b"tampered")
            return artifact

        return invoke

    return WorkloadBackendAdapter(
        id="test.backend/v1",
        source_digest="1" * 64,
        binary_digest="2" * 64,
        toolchain_digest=toolchain_digest,
        adapter=callback(WorkloadRunnerStage.ADAPTER),
        finalizer=callback(WorkloadRunnerStage.FINALIZE),
        resolver=callback(WorkloadRunnerStage.RESOLVE),
        runtime=callback(WorkloadRunnerStage.RUNTIME),
    )


class WorkloadRunnerTest(unittest.TestCase):
    def test_materialize_only_is_partial_and_exact_resume_is_stable(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "case"
            first = run_workload(
                request,
                _capability(supported=True),
                capacities=_capacities(),
                output_dir=output,
            )
            resumed = run_workload(
                request,
                _capability(supported=True),
                capacities=tuple(reversed(_capacities())),
                output_dir=output,
                resume=True,
            )

            self.assertIs(first.status.state, WorkloadRunnerState.PARTIAL)
            self.assertIs(
                first.status.completed_stage, WorkloadRunnerStage.MATERIALIZE
            )
            self.assertFalse(first.status.runtime_evidence_materialized)
            self.assertTrue(resumed.resumed)
            self.assertEqual(resumed.binding, first.binding)
            self.assertEqual(resumed.manifest.digest, first.manifest.digest)
            self.assertEqual(resumed.status, first.status)
            self.assertEqual(
                {item.name for item in output.iterdir()},
                {"request.json", "input_binding.json", "manifest.json", "status.json"},
            )
            self.assertFalse((output / "SUCCESS").exists())

    def test_resume_rejects_stale_request_digest(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        changed = replace(
            request,
            execution=WorkloadExecutionSpec(timing=False, functional=True),
        )
        changed = type(request).create(
            family=changed.family,
            model=changed.model,
            steps=changed.steps,
            mesh=changed.mesh,
            parallel=changed.parallel,
            memory=changed.memory,
            optimizer=changed.optimizer,
            execution=changed.execution,
        )
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "case"
            run_workload(
                request,
                _capability(supported=True),
                capacities=_capacities(),
                output_dir=output,
            )
            with self.assertRaisesRegex(SchemaError, "workload_resume_stale"):
                run_workload(
                    changed,
                    _capability(supported=True),
                    capacities=_capacities(),
                    output_dir=output,
                    resume=True,
                )

    def test_resume_rejects_changed_backend_toolchain_digest(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        derivation = _evidence_derivation(request)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "case"
            run_workload(
                request,
                derivation.capability,
                capacities=_capacities(),
                output_dir=output,
                adapter=_adapter(toolchain_digest="a" * 64),
                capability_derivation=derivation,
            )
            with self.assertRaisesRegex(SchemaError, "workload_resume_stale"):
                run_workload(
                    request,
                    derivation.capability,
                    capacities=_capacities(),
                    output_dir=output,
                    adapter=_adapter(toolchain_digest="b" * 64),
                    capability_derivation=derivation,
                    resume=True,
                )

    def test_two_executions_are_isolated_and_digest_identical(self) -> None:
        original = _request(WorkloadFamily.DENSE_INFERENCE)
        request = type(original).create(
            family=original.family,
            model=original.model,
            steps=original.steps,
            mesh=original.mesh,
            parallel=original.parallel,
            memory=original.memory,
            optimizer=original.optimizer,
            execution=replace(original.execution, independent_repeats=2),
        )
        derivation = _evidence_derivation(request)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "case"
            result = run_workload(
                request,
                derivation.capability,
                capacities=_capacities(),
                output_dir=output,
                adapter=_adapter(),
                capability_derivation=derivation,
            )

            self.assertIs(result.status.state, WorkloadRunnerState.PARTIAL)
            self.assertIs(result.status.readiness, WorkloadRunnerReadiness.READY)
            self.assertFalse(result.status.readiness_reasons)
            self.assertIs(
                result.status.completed_stage, WorkloadRunnerStage.REPEATABILITY
            )
            self.assertEqual(len(result.status.execution_digests), 2)
            self.assertEqual(len(set(result.status.execution_digests)), 1)
            for execution_index in range(2):
                execution_dir = output / f"execution_{execution_index}"
                for stage in (
                    WorkloadRunnerStage.ADAPTER,
                    WorkloadRunnerStage.FINALIZE,
                    WorkloadRunnerStage.RESOLVE,
                    WorkloadRunnerStage.RUNTIME,
                ):
                    artifact = load_json_dataclass(
                        WorkloadStageArtifact,
                        execution_dir / f"{stage.value}.json",
                    )
                    self.assertIs(artifact.stage, stage)
                    self.assertEqual(len(artifact.files), 1)
                    self.assertEqual(
                        artifact.files[0].relative_path,
                        f"artifacts/{stage.value}.json",
                    )
                    artifact.verify_files(execution_dir)
                    self.assertEqual(
                        artifact.one_shot_workload_end,
                        stage is WorkloadRunnerStage.RUNTIME,
                    )
            self.assertFalse((output / "SUCCESS").exists())

    def test_forged_supported_capability_cannot_enter_adapter(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "case"
            result = run_workload(
                request,
                _capability(supported=True),
                capacities=_capacities(),
                output_dir=output,
                adapter=_adapter(),
            )
            self.assertIs(result.status.state, WorkloadRunnerState.PARTIAL)
            self.assertIs(
                result.status.readiness,
                WorkloadRunnerReadiness.NON_READY,
            )
            self.assertEqual(
                result.status.readiness_reasons,
                ("capability_evidence",),
            )
            self.assertFalse((output / "execution_0").exists())

    def test_unsupported_manifest_never_calls_adapter(self) -> None:
        called = False

        def forbidden(_context):
            nonlocal called
            called = True
            raise AssertionError("unsupported request reached backend")

        adapter = WorkloadBackendAdapter(
            id="forbidden.backend/v1",
            source_digest="1" * 64,
            binary_digest="2" * 64,
            toolchain_digest="f" * 64,
            adapter=forbidden,
            finalizer=forbidden,
            resolver=forbidden,
            runtime=forbidden,
        )
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "case"
            result = run_workload(
                _request(WorkloadFamily.DENSE_INFERENCE),
                _capability(supported=False),
                capacities=_capacities(),
                output_dir=output,
                adapter=adapter,
            )
            self.assertFalse(called)
            self.assertIs(result.status.state, WorkloadRunnerState.UNSUPPORTED)
            self.assertFalse((output / "execution_0").exists())

    def test_materialization_capacity_failure_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "case"
            with self.assertRaises(Exception):
                run_workload(
                    _request(WorkloadFamily.DENSE_INFERENCE),
                    _capability(supported=True),
                    capacities=_capacities(size=16),
                    output_dir=output,
                )
            error = load_json_dataclass(WorkloadRunnerError, output / "error.json")
            status = load_json_dataclass(WorkloadRunnerStatus, output / "status.json")
            self.assertIs(error.stage, WorkloadRunnerStage.MATERIALIZE)
            self.assertIs(status.state, WorkloadRunnerState.FAILED)
            self.assertFalse((output / "manifest.json").exists())
            self.assertFalse((output / "SUCCESS").exists())

    def test_each_backend_stage_failure_is_retained(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        derivation = _evidence_derivation(request)
        for failed_stage in (
            WorkloadRunnerStage.ADAPTER,
            WorkloadRunnerStage.FINALIZE,
            WorkloadRunnerStage.RESOLVE,
            WorkloadRunnerStage.RUNTIME,
        ):
            with self.subTest(stage=failed_stage), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "case"
                with self.assertRaisesRegex(RuntimeError, failed_stage.value):
                    run_workload(
                        request,
                        derivation.capability,
                        capacities=_capacities(),
                        output_dir=output,
                        adapter=_adapter(fail_at=failed_stage),
                        capability_derivation=derivation,
                    )
                error = load_json_dataclass(
                    WorkloadRunnerError, output / "error.json"
                )
                status = load_json_dataclass(
                    WorkloadRunnerStatus, output / "status.json"
                )
                self.assertIs(error.stage, failed_stage)
                self.assertIs(status.state, WorkloadRunnerState.FAILED)
                self.assertTrue((output / "manifest.json").exists())
                self.assertFalse((output / "SUCCESS").exists())

    def test_stage_completion_rejects_tampered_adapter_file(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        derivation = _evidence_derivation(request)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "case"
            with self.assertRaisesRegex(SchemaError, "workload_artifact_integrity"):
                run_workload(
                    request,
                    derivation.capability,
                    capacities=_capacities(),
                    output_dir=output,
                    adapter=_adapter(tamper_after_capture=WorkloadRunnerStage.ADAPTER),
                    capability_derivation=derivation,
                )
            error = load_json_dataclass(WorkloadRunnerError, output / "error.json")
            self.assertIs(error.stage, WorkloadRunnerStage.ADAPTER)
            self.assertTrue((output / "execution_0" / "artifacts" / "adapter.json").exists())

    def test_resume_rejects_missing_and_truncated_stage_files(self) -> None:
        request = _request(WorkloadFamily.DENSE_INFERENCE)
        derivation = _evidence_derivation(request)
        for mutation in (
            "payload_missing",
            "payload_truncated",
            "manifest_missing",
            "manifest_truncated",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                output = Path(root) / "case"
                run_workload(
                    request,
                    derivation.capability,
                    capacities=_capacities(),
                    output_dir=output,
                    adapter=_adapter(),
                    capability_derivation=derivation,
                )
                execution = output / "execution_0"
                payload = execution / "artifacts" / "finalize.json"
                stage_manifest = execution / "finalize.json"
                target = (
                    payload
                    if mutation.startswith("payload")
                    else stage_manifest
                )
                if mutation.endswith("missing"):
                    target.unlink()
                else:
                    target.write_bytes(target.read_bytes()[:3])
                with self.assertRaisesRegex(
                    SchemaError, "workload_artifact_integrity"
                ):
                    run_workload(
                        request,
                        derivation.capability,
                        capacities=_capacities(),
                        output_dir=output,
                        adapter=_adapter(),
                        capability_derivation=derivation,
                        resume=True,
                    )

    def test_repeatability_mismatch_fails_closed(self) -> None:
        original = _request(WorkloadFamily.DENSE_INFERENCE)
        request = type(original).create(
            family=original.family,
            model=original.model,
            steps=original.steps,
            mesh=original.mesh,
            parallel=original.parallel,
            memory=original.memory,
            optimizer=original.optimizer,
            execution=replace(original.execution, independent_repeats=2),
        )
        derivation = _evidence_derivation(request)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "case"
            with self.assertRaisesRegex(SchemaError, "workload_repeatability_mismatch"):
                run_workload(
                    request,
                    derivation.capability,
                    capacities=_capacities(),
                    output_dir=output,
                    adapter=_adapter(diverge_runtime=True),
                    capability_derivation=derivation,
                )
            error = load_json_dataclass(WorkloadRunnerError, output / "error.json")
            status = load_json_dataclass(WorkloadRunnerStatus, output / "status.json")
            self.assertIs(error.stage, WorkloadRunnerStage.REPEATABILITY)
            self.assertIs(status.state, WorkloadRunnerState.FAILED)
            self.assertFalse(status.execution_digests)


if __name__ == "__main__":
    unittest.main()
