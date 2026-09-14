"""MeshSlice adapter for the strict flexible-Mesh release runner."""

from __future__ import annotations

import hashlib

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.flexible_mesh_compiler import (
    compile_flexible_mesh_workload,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    _build_timing_program_io_prevalidated,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RuntimeSymbolKind
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseFamily,
    FlexibleMeshReleaseResidual,
    expected_completion_markers,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_runtime import (
    FlexibleMeshRuntimeCase,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_workload import (
    FlexibleMeshSliceOperation,
    FlexibleMeshWorkloadSpec,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from flexible_mesh_runtime_markers import (
    parse_flexible_mesh_runtime_marker,
    validate_credit_balance,
)
from flexible_mesh_release_hardware import (
    p5_large_hardware_template_json,
    specialize_release_hardware,
)
from flexible_mesh_release_profiles import release_trace_model_digest
from flexible_mesh_runtime_meshslice import ProductionMeshSliceRuntimeMaterializer
from run_flexible_mesh_release import (
    FlexibleMeshReleaseMaterialized,
    FlexibleMeshReleaseObservation,
)


_FAMILY_OPERATION = {
    FlexibleMeshReleaseFamily.MESHSLICE_AG:
        FlexibleMeshSliceOperation.AG_GEMM,
    FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK:
        FlexibleMeshSliceOperation.GEMM_RS,
    FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK:
        FlexibleMeshSliceOperation.GEMM_AR,
}


class FlexibleMeshSliceReleaseAdapter:
    def __init__(
        self,
        family: FlexibleMeshReleaseFamily,
        *,
        mapping_text: str,
        hardware_template_json: str | None = None,
    ) -> None:
        if family not in _FAMILY_OPERATION:
            raise SchemaError("requires a MeshSlice release family", path="family")
        self.family = family
        self._hardware_template_json = (
            p5_large_hardware_template_json()
            if hardware_template_json is None
            else hardware_template_json
        )
        specialize_release_hardware(self._hardware_template_json, 1, 1)
        self._mapping_text = mapping_text
        self._materializer = ProductionMeshSliceRuntimeMaterializer(
            mapping_text=mapping_text
        )
        self._old_cases: dict[str, FlexibleMeshRuntimeCase] = {}
        self._sources: dict[str, object] = {}

    def _hardware_json(self, case: FlexibleMeshReleaseCase) -> str:
        return specialize_release_hardware(
            self._hardware_template_json,
            case.mesh.rows,
            case.mesh.columns,
        )

    @property
    def mapping_sha256(self) -> str:
        return hashlib.sha256(self._mapping_text.encode("utf-8")).hexdigest()

    def expected_hardware_sha256(self, case: FlexibleMeshReleaseCase) -> str:
        if case.family is not self.family:
            raise SchemaError("adapter family drifted", path="case.family")
        if case.trace_model_digest != release_trace_model_digest(case.family):
            raise SchemaError(
                "adapter trace/model profile drifted",
                path="case.trace_model_digest",
            )
        return hashlib.sha256(
            self._hardware_json(case).encode("utf-8")
        ).hexdigest()

    def _runtime_case(
        self, case: FlexibleMeshReleaseCase
    ) -> FlexibleMeshRuntimeCase:
        if case.family is not self.family:
            raise SchemaError("adapter family drifted", path="case.family")
        if case.trace_model_digest != release_trace_model_digest(case.family):
            raise SchemaError("adapter trace/model profile drifted", path="case.trace_model_digest")
        workload = FlexibleMeshWorkloadSpec.dense_infer(case.mesh)
        return FlexibleMeshRuntimeCase.create(
            workload,
            _FAMILY_OPERATION[self.family],
            repeat_count=1,
        )

    def materialize(
        self,
        case: FlexibleMeshReleaseCase,
    ) -> FlexibleMeshReleaseMaterialized:
        runtime_case = self._runtime_case(case)
        executable = self._materializer.materialize(runtime_case)
        compilation = compile_flexible_mesh_workload(runtime_case.workload)
        runtime_symbols = tuple(
            symbol
            for fragment in executable.manifest.fragments
            for symbol in fragment.runtime_symbols
        )
        tag_count = len(
            {
                symbol.id for symbol in runtime_symbols
                if symbol.kind is RuntimeSymbolKind.DTE_TOKEN
            }
        )
        self._old_cases[case.id] = runtime_case
        self._sources[case.id] = executable.linked_source
        result = FlexibleMeshReleaseMaterialized(
            case_id=case.id,
            spec_digest=runtime_case.workload.digest,
            plan_digest=canonical_digest(compilation),
            manifest=executable.manifest,
            hardware_json=self._hardware_json(case),
            mapping_text=executable.mapping_text,
            peak_sessions_per_core_per_wave=(
                0 if case.mesh.rank_count == 1 else 2
            ),
            transport_tag_count=tag_count,
        )
        result.validate(case)
        return result

    def build_program_io(
        self,
        materialized: FlexibleMeshReleaseMaterialized,
        artifact_sha256: str,
    ) -> ProgramIoContract:
        source = self._sources.get(materialized.case_id)
        if source is None:
            raise SchemaError("case was not materialized", path="materialized.case_id")
        return _build_timing_program_io_prevalidated(
            source, artifact_sha256,
        )

    def observe(
        self,
        case: FlexibleMeshReleaseCase,
        materialized: FlexibleMeshReleaseMaterialized,
        contract: ProgramIoContract,
        artifact_sha256: str,
        npusim_output: str,
    ) -> FlexibleMeshReleaseObservation:
        runtime_case = self._old_cases.get(case.id)
        if runtime_case is None:
            raise SchemaError("case was not materialized", path="case.id")
        marker = parse_flexible_mesh_runtime_marker(
            npusim_output,
            case=runtime_case,
            manifest=materialized.manifest,
            contract=contract,
            artifact_sha256=artifact_sha256,
        )
        validate_credit_balance(npusim_output)
        result = FlexibleMeshReleaseObservation(
            makespan_cycles=marker.makespan_cycles,
            marker_digest=marker.marker_digest,
            residual=FlexibleMeshReleaseResidual(
                active_endpoints=marker.residual.active_endpoints,
                active_sessions=marker.residual.active_sessions,
                outstanding_tags=marker.residual.outstanding_tags,
                incomplete_barriers=marker.residual.incomplete_barriers,
                pending_state_writes=marker.residual.pending_state_writes,
                proto_wait_count=marker.residual.proto_wait_count,
            ),
            rank_coverage=marker.rank_coverage,
            core_coverage=tuple(range(case.mesh.rank_count)),
            completion_markers=expected_completion_markers(case.family),
        )
        result.validate(case)
        return result


__all__ = ["FlexibleMeshSliceReleaseAdapter"]
