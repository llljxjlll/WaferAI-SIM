"""WorkloadRunner adapter for a versioned external-DMA action sidecar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from .errors import SchemaError
from .passes.external_dma_action_graph import (
    build_external_dma_action_graph,
    validate_external_dma_action_graph,
)
from .passes.external_dma_program import (
    load_external_dma_program,
    write_external_dma_program,
)
from .schema.external_dma_action_graph import (
    ExternalDmaActionGraph,
    ExternalDmaRuntimeBinding,
)
from .schema.external_dma_program import ExternalDmaProgram
from .schema.offload import BlockingOffloadPlan
from .schema.serde import canonical_json, load_json_dataclass
from .schema.workload_materialization import WorkloadMaterializationManifest
from .workload_runner import (
    WorkloadBackendAdapter,
    WorkloadRunnerStage,
    WorkloadStageArtifact,
)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SchemaError(str(error), path="runtime_binary") from error
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _artifact(context, stage: WorkloadRunnerStage, relative_path: str):
    return WorkloadStageArtifact.create(
        stage=stage,
        input_digest=context.input_digest,
        work_dir=context.work_dir,
        relative_paths=(relative_path,),
        one_shot_workload_end=stage is WorkloadRunnerStage.RUNTIME,
    )


def create_external_dma_workload_adapter(
    *,
    manifest: WorkloadMaterializationManifest,
    plan: BlockingOffloadPlan,
    program: ExternalDmaProgram,
    runtime_binary: Path,
    source_digest: str,
    toolchain_digest: str,
) -> WorkloadBackendAdapter:
    """Create a runner adapter whose runtime stage waits for DMA drain."""

    if not isinstance(runtime_binary, Path) or not runtime_binary.is_file():
        raise SchemaError("must be an existing file", path="runtime_binary")
    action_graph = build_external_dma_action_graph(
        manifest=manifest, plan=plan, program=program
    )
    binary_digest = _file_digest(runtime_binary)

    def require_manifest(context) -> None:
        if context.manifest.digest != manifest.digest:
            raise SchemaError(
                "adapter was bound to another workload manifest",
                path="context.manifest",
                code="external_dma_action_source_mismatch",
            )

    def require_persisted_action_graph(context) -> None:
        persisted = load_json_dataclass(
            ExternalDmaActionGraph,
            context.work_dir / "artifacts/external_dma_action_graph.json",
            path="external_dma_action_graph",
        )
        if persisted != action_graph:
            raise SchemaError(
                "persisted action graph differs from adapter binding",
                path="external_dma_action_graph",
                code="external_dma_action_source_mismatch",
            )
        validate_external_dma_action_graph(
            persisted,
            manifest=manifest,
            plan=plan,
            program=program,
        )

    def adapter_stage(context):
        require_manifest(context)
        artifacts = context.work_dir / "artifacts"
        artifacts.mkdir(exist_ok=True)
        relative = "artifacts/external_dma_action_graph.json"
        _write(context.work_dir / relative, action_graph)
        return _artifact(context, WorkloadRunnerStage.ADAPTER, relative)

    def finalizer_stage(context):
        require_manifest(context)
        require_persisted_action_graph(context)
        relative = "artifacts/external_dma_program.json"
        write_external_dma_program(program, context.work_dir / relative)
        return _artifact(context, WorkloadRunnerStage.FINALIZE, relative)

    def resolver_stage(context):
        require_manifest(context)
        require_persisted_action_graph(context)
        persisted_program = load_external_dma_program(
            context.work_dir / "artifacts/external_dma_program.json"
        )
        if persisted_program != program:
            raise SchemaError(
                "finalized DMA program differs from bound program",
                path="external_dma_program",
                code="external_dma_action_source_mismatch",
            )
        binding = ExternalDmaRuntimeBinding.create(
            action_graph_digest=action_graph.digest,
            program_relative_path="artifacts/external_dma_program.json",
            case_digest=program.case_digest,
            request_digest=program.request_digest,
            logical_graph_digest=program.logical_graph_digest,
            source_memory_plan_digest=program.source_memory_plan_digest,
            blocking_offload_plan_digest=program.blocking_offload_plan_digest,
        )
        relative = "artifacts/external_dma_runtime_binding.json"
        _write(context.work_dir / relative, binding)
        return _artifact(context, WorkloadRunnerStage.RESOLVE, relative)

    def runtime_stage(context):
        require_manifest(context)
        require_persisted_action_graph(context)
        if _file_digest(runtime_binary) != binary_digest:
            raise SchemaError(
                "runtime binary changed after adapter binding",
                path="runtime_binary",
                code="external_dma_runtime_binary_mismatch",
            )
        binding = load_json_dataclass(
            ExternalDmaRuntimeBinding,
            context.work_dir / "artifacts/external_dma_runtime_binding.json",
            path="external_dma_runtime_binding",
        )
        if binding.action_graph_digest != action_graph.digest:
            raise SchemaError(
                "runtime binding does not match action graph",
                path="external_dma_runtime_binding.action_graph_digest",
            )
        relative = "artifacts/external_dma_runtime_report.json"
        report_path = context.work_dir / relative
        command = [
            str(runtime_binary),
            str(context.work_dir / binding.program_relative_path),
            str(report_path),
            binding.action_graph_digest,
            binding.case_digest,
            binding.request_digest,
            binding.logical_graph_digest,
            binding.source_memory_plan_digest,
            binding.blocking_offload_plan_digest,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=context.work_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SchemaError(str(error), path="external_dma_runtime") from error
        if completed.returncode != 0:
            raise SchemaError(
                f"runtime failed with {completed.returncode}: {completed.stdout.strip()}",
                path="external_dma_runtime",
            )
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SchemaError(str(error), path="external_dma_runtime_report") from error
        expected_keys = {
            "schema_version", "action_graph_digest", "program_ref", "completed",
            "pending_requests", "submitted_requests", "completed_requests",
            "failed_requests", "external_read_bytes", "external_write_bytes",
            "hbm_read_bytes", "hbm_write_bytes", "all_probes_matched",
        }
        if type(report) is not dict or set(report) != expected_keys:
            raise SchemaError("unexpected report fields", path="external_dma_runtime_report")
        if (
            report["schema_version"] != "npusim.external_dma_runtime_report/v1alpha1"
            or report["action_graph_digest"] != action_graph.digest
            or report["program_ref"] != program.id
            or report["completed"] is not True
            or report["pending_requests"] != 0
            or report["failed_requests"] != 0
            or report["completed_requests"] != len(program.descriptors)
            or report["submitted_requests"] != len(program.descriptors)
            or report["all_probes_matched"] is not True
        ):
            raise SchemaError(
                "runtime did not complete and drain the bound action program",
                path="external_dma_runtime_report",
                code="external_dma_runtime_not_drained",
            )
        return _artifact(context, WorkloadRunnerStage.RUNTIME, relative)

    return WorkloadBackendAdapter(
        id="npusim.external_dma_action_adapter/v1alpha1",
        source_digest=source_digest,
        binary_digest=binary_digest,
        toolchain_digest=toolchain_digest,
        adapter=adapter_stage,
        finalizer=finalizer_stage,
        resolver=resolver_stage,
        runtime=runtime_stage,
    )


__all__ = ["create_external_dma_workload_adapter"]
