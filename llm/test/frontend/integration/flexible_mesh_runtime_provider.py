"""Reusable preparation provider for flexible-Mesh integration tests.

The provider stops at real hardware loading and bounded topology construction.
It never labels fixture preparation as runtime execution evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Protocol

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.policies.swizzle.rect_mesh_topology import (
    build_rect_mesh_topology,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.flexible_mesh_compiler import (
    compile_flexible_mesh_workload,
)
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_runtime import (
    FlexibleMeshArtifactCapacityEvidence,
    FlexibleMeshRuntimeCase,
    FlexibleMeshRuntimeEvidence,
    FlexibleMeshRuntimeFailure,
    FlexibleMeshRuntimeStage,
    FlexibleMeshRuntimeStageStatus,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.frontend.wafer_frontend.schema.swizzle_standard import (
    SwizzleStandardLinkedProgram,
)
from llm.frontend.wafer_frontend.schema.swizzle_unfused_standard import (
    UnfusedComparisonStandardLinkedProgram,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware

from flexible_mesh_cases import FlexibleMeshCase, validate_flexible_mesh_case_matrix
from flexible_mesh_runtime_markers import parse_flexible_mesh_runtime_marker


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class PreparedFlexibleMeshCase:
    """Deterministic, non-runtime preparation evidence for one case."""

    case: FlexibleMeshCase
    fabric_digest: str
    hbm_address_spaces_digest: str
    topology_digest: str
    die_count: int
    directed_link_count: int
    expected_ordered_pair_route_count: int
    max_hop_count: int
    row_rank_orders: tuple[tuple[int, ...], ...]
    column_rank_orders: tuple[tuple[int, ...], ...]
    snake_rank_order: tuple[int, ...]
    hamiltonian_cycle_rank_order: tuple[int, ...]

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "prepared_flexible_mesh_case") -> None:
        if type(self.case) is not FlexibleMeshCase:
            raise SchemaError("must carry a FlexibleMeshCase", path=f"{path}.case")
        self.case.validate(f"{path}.case")
        for name in (
            "fabric_digest",
            "hbm_address_spaces_digest",
            "topology_digest",
        ):
            _validate_digest(getattr(self, name), f"{path}.{name}")
        spec = self.case.mesh
        actual = (
            self.die_count,
            self.directed_link_count,
            self.expected_ordered_pair_route_count,
            self.max_hop_count,
            self.row_rank_orders,
            self.column_rank_orders,
            self.snake_rank_order,
        )
        expected = (
            spec.rank_count,
            spec.directed_link_count,
            spec.rank_count * (spec.rank_count - 1),
            spec.max_hop_count,
            spec.row_rank_orders,
            spec.column_rank_orders,
            spec.snake_rank_order,
        )
        if actual != expected:
            raise SchemaError(
                "prepared metrics drifted from the rectangular Mesh contract",
                path=path,
            )
        cycle = self.hamiltonian_cycle_rank_order
        if bool(cycle) != spec.has_hamiltonian_cycle:
            raise SchemaError("Hamiltonian-cycle existence drifted", path=path)
        if cycle and (
            len(cycle) != spec.rank_count
            or set(cycle) != set(range(spec.rank_count))
        ):
            raise SchemaError("Hamiltonian cycle must cover every rank once", path=path)


class FlexibleMeshRuntimeProvider:
    """Shared preparation entry point for later runtime and evidence suites."""

    def prepare(self, case: FlexibleMeshCase) -> PreparedFlexibleMeshCase:
        if type(case) is not FlexibleMeshCase:
            raise SchemaError("must be a FlexibleMeshCase", path="provider.case")
        case.validate("provider.case")
        spec = case.mesh
        hardware = minimal_hardware(spec.columns, spec.rows)
        fabric = physical_fabric_from_data(hardware)
        address_spaces = hbm_address_spaces_from_data(hardware)
        topology = build_rect_mesh_topology(
            {spec.coordinate(rank): rank for rank in range(spec.rank_count)}
        )
        if (
            fabric.die_grid != spec.physical_shape
            or len(fabric.dies) != spec.rank_count
            or len(fabric.links) != spec.directed_link_count
            or len(address_spaces) != spec.rank_count
            or not topology.is_complete_rectangle
            or topology.physical_shape != spec.physical_shape
        ):
            raise SchemaError(
                "production hardware preparation does not match RectMeshSpec",
                path="provider",
            )
        result = PreparedFlexibleMeshCase(
            case=case,
            fabric_digest=canonical_digest(fabric),
            hbm_address_spaces_digest=canonical_digest(address_spaces),
            topology_digest=canonical_digest(topology),
            die_count=len(fabric.dies),
            directed_link_count=len(fabric.links),
            expected_ordered_pair_route_count=len(spec.ordered_rank_pairs),
            max_hop_count=spec.max_hop_count,
            row_rank_orders=topology.row_rank_orders,
            column_rank_orders=topology.column_rank_orders,
            snake_rank_order=topology.snake_rank_order,
            hamiltonian_cycle_rank_order=topology.hamiltonian_cycle_rank_order,
        )
        result.validate()
        return result

    def prepare_matrix(
        self, cases: tuple[FlexibleMeshCase, ...]
    ) -> tuple[PreparedFlexibleMeshCase, ...]:
        validate_flexible_mesh_case_matrix(cases, path="provider.cases")
        return tuple(self.prepare(case) for case in cases)


_LinkedSource = (
    SwizzleStandardLinkedProgram | UnfusedComparisonStandardLinkedProgram
)


@dataclass(frozen=True, slots=True)
class FlexibleMeshRuntimeExecutable:
    case_id: str
    compilation_id: str
    linked_source_ref: str
    linked_source: _LinkedSource
    hardware_json: str
    mapping_text: str

    @property
    def manifest(self) -> LinkedProgramManifest:
        return self.linked_source.manifest

    def validate(self, case: FlexibleMeshRuntimeCase) -> None:
        case.validate()
        if self.case_id != case.id:
            raise SchemaError("executable case provenance drifted", path="runtime.executable")
        for name in (
            "compilation_id",
            "linked_source_ref",
            "hardware_json",
            "mapping_text",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise SchemaError("must be non-empty", path=f"runtime.executable.{name}")
        if self.linked_source_ref != self.linked_source.id:
            raise SchemaError("linked source reference drifted", path="runtime.executable")
        self.linked_source.validate_against()


class FlexibleMeshRuntimeMaterializer(Protocol):
    def materialize(
        self,
        case: FlexibleMeshRuntimeCase,
    ) -> FlexibleMeshRuntimeExecutable: ...


class _ExternalStageError(RuntimeError):
    def __init__(self, stage: FlexibleMeshRuntimeStage, output: str) -> None:
        self.stage = stage
        self.output = output
        super().__init__(output)


def _run_external(
    command: tuple[str, ...],
    *,
    cwd: Path,
    timeout: int,
    stage: FlexibleMeshRuntimeStage,
) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise _ExternalStageError(
            stage,
            f"returncode={completed.returncode}: {' '.join(command)}\n{completed.stdout}",
        )
    return completed.stdout


def _failure(
    case: FlexibleMeshRuntimeCase,
    stage: FlexibleMeshRuntimeStage,
    reason: str,
) -> FlexibleMeshRuntimeFailure:
    stages = tuple(FlexibleMeshRuntimeStage)
    result = FlexibleMeshRuntimeFailure(
        case_id=case.id,
        failed_stage=stage,
        reason=reason or "unknown runtime stage failure",
        completed_stages=stages[:stages.index(stage)],
    )
    result.validate()
    return result


class FlexibleMeshExecutionProvider:
    """Run one materialized case through official external binaries only."""

    def __init__(self, materializer: FlexibleMeshRuntimeMaterializer) -> None:
        self._materializer = materializer

    def run(
        self,
        case: FlexibleMeshRuntimeCase,
        *,
        finalizer: Path,
        resolver: Path,
        npusim: Path,
        simulation: Path,
        runtime_root: Path,
        timeout: int = 600,
    ) -> FlexibleMeshRuntimeEvidence | FlexibleMeshRuntimeFailure:
        try:
            case.validate()
        except Exception as error:
            return _failure(case, FlexibleMeshRuntimeStage.SCHEMA, str(error))
        try:
            compilation = compile_flexible_mesh_workload(case.workload)
            compilation.validate()
        except Exception as error:
            return _failure(case, FlexibleMeshRuntimeStage.CANDIDATE, str(error))
        try:
            executable = self._materializer.materialize(case)
            executable.validate(case)
            if executable.compilation_id != compilation.id:
                raise SchemaError(
                    "materializer compilation provenance drifted",
                    path="runtime.executable.compilation_id",
                )
        except Exception as error:
            return _failure(case, FlexibleMeshRuntimeStage.LOWER_LINK, str(error))
        try:
            preflight = build_timing_program_io(
                executable.linked_source,
                "0" * 64,
            )
            preflight.validate_against(executable.manifest)
        except Exception as error:
            return _failure(case, FlexibleMeshRuntimeStage.PROGRAM_IO, str(error))

        directory = runtime_root / case.id
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        artifact_path = directory / "program.npup"
        report_path = directory / "finalizer.json"
        sidecar_path = directory / "program_io.json"
        manifest_path.write_text(canonical_json(executable.manifest), encoding="utf-8")
        hardware_path.write_text(executable.hardware_json, encoding="utf-8")
        mapping_path.write_text(executable.mapping_text, encoding="utf-8")
        try:
            output = _run_external(
                (
                    str(finalizer),
                    "--input",
                    str(manifest_path),
                    "--output",
                    str(artifact_path),
                    "--report",
                    str(report_path),
                ),
                cwd=runtime_root,
                timeout=120,
                stage=FlexibleMeshRuntimeStage.FINALIZER,
            )
            (directory / "finalizer.stdout.txt").write_text(output, encoding="utf-8")
            artifact = artifact_path.read_bytes()
            artifact_sha = hashlib.sha256(artifact).hexdigest()
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                type(report) is not dict
                or report.get("artifact_sha256") != artifact_sha
                or report.get("artifact_bytes") != len(artifact)
                or report.get("linked_manifest_id") != executable.manifest.id
                or report.get("linked_manifest_digest")
                != canonical_digest(executable.manifest)
            ):
                raise SchemaError(
                    "finalizer report does not close actual artifact",
                    path="runtime.finalizer",
                )
            capacity = FlexibleMeshArtifactCapacityEvidence(
                record_count=report.get("record_count", -1),
                relocation_count=report.get("relocation_count", -1),
                runtime_symbol_count=len(
                    executable.manifest.runtime_symbol_definitions
                ),
                artifact_file_bytes=len(artifact),
                core_count=report.get("core_count", -1),
            )
            capacity.validate(case)
            contract = build_timing_program_io(
                executable.linked_source,
                artifact_sha,
            )
            contract.validate_against(executable.manifest)
            sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        except Exception as error:
            return _failure(case, FlexibleMeshRuntimeStage.FINALIZER, str(error))
        try:
            resolver_output = _run_external(
                (
                    str(resolver),
                    "--resolve",
                    str(manifest_path),
                    str(artifact_path),
                    str(sidecar_path),
                ),
                cwd=runtime_root,
                timeout=120,
                stage=FlexibleMeshRuntimeStage.RESOLVER,
            )
            (directory / "resolver.stdout.txt").write_text(
                resolver_output,
                encoding="utf-8",
            )
            if (
                f"initializations={len(contract.initializations)}"
                not in resolver_output
                or f"probes={len(contract.output_probes)}" not in resolver_output
            ):
                raise SchemaError(
                    "resolver lost exact ProgramIO counts",
                    path="runtime.resolver",
                )
            resolver_digest = hashlib.sha256(
                resolver_output.encode("utf-8")
            ).hexdigest()
        except Exception as error:
            return _failure(case, FlexibleMeshRuntimeStage.RESOLVER, str(error))
        try:
            markers = []
            for index in range(case.repeat_count):
                runtime_output = _run_external(
                    (
                        str(npusim),
                        "--program",
                        str(artifact_path),
                        "--linked-manifest",
                        str(manifest_path),
                        "--program-io",
                        str(sidecar_path),
                        "--hardware-config",
                        str(hardware_path),
                        "--simulation-config",
                        str(simulation),
                        "--mapping-config",
                        str(mapping_path),
                        "--trace-window",
                        "1000000",
                    ),
                    cwd=runtime_root,
                    timeout=timeout,
                    stage=FlexibleMeshRuntimeStage.NPUSIM,
                )
                (directory / f"npusim.{index}.stdout.txt").write_text(
                    runtime_output,
                    encoding="utf-8",
                )
                markers.append(parse_flexible_mesh_runtime_marker(
                    runtime_output,
                    case=case,
                    manifest=executable.manifest,
                    contract=contract,
                    artifact_sha256=artifact_sha,
                ))
        except Exception as error:
            return _failure(case, FlexibleMeshRuntimeStage.NPUSIM, str(error))
        if case.repeat_count == 2 and markers[0] != markers[1]:
            return _failure(
                case,
                FlexibleMeshRuntimeStage.REPEATABILITY,
                "NpuSim marker/makespan repeat changed",
            )
        stages = tuple(
            (
                stage,
                (
                    FlexibleMeshRuntimeStageStatus.VERIFIED
                    if stage is not FlexibleMeshRuntimeStage.REPEATABILITY
                    or case.repeat_count == 2
                    else FlexibleMeshRuntimeStageStatus.NOT_MEASURED
                ),
            )
            for stage in FlexibleMeshRuntimeStage
        )
        result = FlexibleMeshRuntimeEvidence.create(
            case=case,
            compilation_id=compilation.id,
            linked_source_ref=executable.linked_source_ref,
            manifest_id=executable.manifest.id,
            manifest_digest=canonical_digest(executable.manifest),
            artifact_sha256=artifact_sha,
            program_io_digest=canonical_digest(contract),
            resolver_digest=resolver_digest,
            npusim_exit_code=0,
            capacity=capacity,
            markers=tuple(markers),
            stages=stages,
            timing_execution=True,
            functional_execution=False,
        )
        (directory / "runtime_evidence.json").write_text(
            canonical_json(result),
            encoding="utf-8",
        )
        return result


__all__ = [
    "FlexibleMeshExecutionProvider",
    "FlexibleMeshRuntimeExecutable",
    "FlexibleMeshRuntimeMaterializer",
    "FlexibleMeshRuntimeProvider",
    "PreparedFlexibleMeshCase",
]
