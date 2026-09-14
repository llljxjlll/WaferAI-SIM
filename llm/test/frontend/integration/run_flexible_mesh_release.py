"""Sharded real-tool runner for the flexible-Mesh 100-shape release matrix.

The runner is deliberately adapter based: workload adapters own compilation,
ProgramIO construction and typed marker parsing, while this module owns the
two independent materialize/finalize/resolve/npusim executions and their exact
repeatability comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Protocol

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.program_io import (
    _program_io_manifest_digest_context,
)
from llm.frontend.wafer_frontend.schema._validation_session import (
    builder_validation_session,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (
    LinkedProgramManifest,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES,
    FlexibleMeshCompletionMarker,
    FlexibleMeshIndependentRun,
    FlexibleMeshProgramIOPhase,
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseCapacityEvidence,
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseExecutionEvidence,
    FlexibleMeshReleaseFamily,
    FlexibleMeshReleaseResidual,
    FlexibleMeshReleaseRuntimeStage,
    expected_completion_markers,
)
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _write(path: Path, value: str | bytes) -> None:
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _run(
    command: tuple[str, ...],
    *,
    cwd: Path,
    timeout: int,
) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseMaterialized:
    """Workload-owned executable inputs consumed by the common runner."""

    case_id: str
    spec_digest: str
    plan_digest: str
    manifest: LinkedProgramManifest
    hardware_json: str
    mapping_text: str
    peak_sessions_per_core_per_wave: int
    transport_tag_count: int

    def validate(self, case: FlexibleMeshReleaseCase) -> None:
        case.validate("release_case")
        if self.case_id != case.id:
            raise SchemaError("materialization case drifted", path="materialized")
        for name in ("spec_digest", "plan_digest"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SchemaError("must be a SHA-256 digest", path=f"materialized.{name}")
        self.manifest.validate("materialized.manifest")
        if type(self.hardware_json) is not str or not self.hardware_json:
            raise SchemaError("hardware JSON is empty", path="materialized.hardware_json")
        try:
            parsed = json.loads(self.hardware_json)
        except json.JSONDecodeError as error:
            raise SchemaError("hardware JSON is invalid", path="materialized.hardware_json") from error
        if type(parsed) is not dict:
            raise SchemaError("hardware JSON must be an object", path="materialized.hardware_json")
        if type(self.mapping_text) is not str or not self.mapping_text:
            raise SchemaError("mapping text is empty", path="materialized.mapping_text")
        for name in (
            "peak_sessions_per_core_per_wave",
            "transport_tag_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SchemaError("must be a non-negative integer", path=f"materialized.{name}")


@dataclass(frozen=True, slots=True)
class FlexibleMeshReleaseObservation:
    """Typed, workload-specific parse of one real NpuSim stdout."""

    makespan_cycles: int
    marker_digest: str
    residual: FlexibleMeshReleaseResidual
    rank_coverage: tuple[int, ...]
    core_coverage: tuple[int, ...]
    completion_markers: tuple[FlexibleMeshCompletionMarker, ...]

    def validate(self, case: FlexibleMeshReleaseCase) -> None:
        if type(self.makespan_cycles) is not int or self.makespan_cycles <= 0:
            raise SchemaError("makespan must be positive", path="observation.makespan")
        if (
            type(self.marker_digest) is not str
            or len(self.marker_digest) != 64
            or any(character not in "0123456789abcdef" for character in self.marker_digest)
        ):
            raise SchemaError("marker digest is invalid", path="observation.marker_digest")
        self.residual.validate("observation.residual")
        expected = tuple(range(case.mesh.rank_count))
        if self.rank_coverage != expected or self.core_coverage != expected:
            raise SchemaError("coverage is not exact", path="observation")
        if self.completion_markers != expected_completion_markers(case.family):
            raise SchemaError("completion marker closure drifted", path="observation")


class FlexibleMeshReleaseAdapter(Protocol):
    family: FlexibleMeshReleaseFamily

    def materialize(
        self,
        case: FlexibleMeshReleaseCase,
    ) -> FlexibleMeshReleaseMaterialized: ...

    def build_program_io(
        self,
        materialized: FlexibleMeshReleaseMaterialized,
        artifact_sha256: str,
    ) -> ProgramIoContract: ...

    def observe(
        self,
        case: FlexibleMeshReleaseCase,
        materialized: FlexibleMeshReleaseMaterialized,
        contract: ProgramIoContract,
        artifact_sha256: str,
        npusim_output: str,
    ) -> FlexibleMeshReleaseObservation: ...


def _symbolic_record_count(manifest: LinkedProgramManifest) -> int:
    return sum(
        len(stream.records)
        for fragment in manifest.fragments
        for stream in (
            fragment.fragment.core_streams
            if hasattr(fragment, "fragment")
            else fragment.core_streams
        )
    )


def _max_runtime_core_id(manifest: LinkedProgramManifest) -> int:
    return max((binding.runtime_core_id for binding in manifest.core_bindings), default=0)


def _transport_tag_count(manifest: LinkedProgramManifest) -> int:
    tags = {
        operand.literal_value
        for fragment in manifest.fragments
        for stream in (
            fragment.fragment.core_streams
            if hasattr(fragment, "fragment")
            else fragment.core_streams
        )
        for record in stream.records
        if record.opcode in (RecordOpcode.DTE_SEND, RecordOpcode.DTE_RECV)
        for operand in record.operands
        if operand.name in ("tag", "transport_tag")
        and type(operand.literal_value) is int
    }
    return len(tags)


class FlexibleMeshReleaseRunner:
    def __init__(
        self,
        *,
        binding: FlexibleMeshReleaseBinding,
        adapters: tuple[FlexibleMeshReleaseAdapter, ...],
        finalizer: Path,
        resolver: Path,
        npusim: Path,
        simulation: Path,
        runtime_root: Path,
        timeout: int = 600,
    ) -> None:
        binding.validate("binding")
        self._binding = binding
        self._adapters = {adapter.family: adapter for adapter in adapters}
        if set(self._adapters) != set(FlexibleMeshReleaseFamily):
            raise SchemaError("adapters must cover all six families", path="adapters")
        self._finalizer = finalizer.resolve()
        self._resolver = resolver.resolve()
        self._npusim = npusim.resolve()
        self._simulation = simulation.resolve()
        self._runtime_root = runtime_root.resolve()
        self._timeout = timeout
        if type(timeout) is not int or timeout <= 0:
            raise SchemaError("timeout must be positive", path="timeout")
        self._runtime_root.mkdir(parents=True, exist_ok=True)
        expected_paths = tuple(tool.binary_path for tool in binding.tools)
        actual_paths = tuple(
            str(path) for path in (self._finalizer, self._resolver, self._npusim)
        )
        if expected_paths != actual_paths:
            raise SchemaError("tool paths drifted from release binding", path="binding.tools")
        for tool, path in zip(binding.tools, (self._finalizer, self._resolver, self._npusim)):
            if not path.is_file() or _sha256_bytes(path.read_bytes()) != tool.sha256:
                raise SchemaError("tool bytes drifted from release binding", path="binding.tools")
        if (
            not self._simulation.is_file()
            or _sha256_bytes(self._simulation.read_bytes())
            != binding.simulation_config_sha256
        ):
            raise SchemaError("simulation config drifted", path="binding.simulation_config_sha256")

    def _execute_once(
        self,
        case: FlexibleMeshReleaseCase,
        execution_index: int,
    ) -> FlexibleMeshReleaseExecutionEvidence:
        adapter = self._adapters[case.family]
        directory = self._runtime_root / case.id / f"execution_{execution_index}"
        directory.mkdir(parents=True, exist_ok=True)
        with builder_validation_session():
            materialized = adapter.materialize(case)
            materialized.validate(case)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        artifact_path = directory / "program.npup"
        report_path = directory / "finalizer.json"
        sidecar_path = directory / "program_io.json"
        manifest_bytes = canonical_json(materialized.manifest).encode("utf-8")
        manifest_digest = _sha256_bytes(manifest_bytes)
        if len(manifest_bytes) > FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES:
            raise SchemaError(
                "linked manifest byte limit exceeded: "
                f"observed={len(manifest_bytes)} "
                f"limit={FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES}",
                path=f"case.{case.id}.capacity.linked_manifest_file_bytes",
            )
        _write(manifest_path, manifest_bytes)
        _write(hardware_path, materialized.hardware_json)
        _write(mapping_path, materialized.mapping_text)

        with builder_validation_session():
            with _program_io_manifest_digest_context(
                materialized.manifest,
                manifest_digest,
            ):
                preflight = adapter.build_program_io(materialized, "0" * 64)
        finalizer_code, finalizer_output = _run(
            (
                str(self._finalizer), "--input", str(manifest_path),
                "--output", str(artifact_path), "--report", str(report_path),
            ),
            # Tool-owned relative configuration paths (for example
            # ../DRAMSys in the P5 simulation profile) must resolve from the
            # build/tool directory, not from an arbitrary evidence root.
            cwd=self._finalizer.parent,
            timeout=min(self._timeout, 120),
        )
        _write(directory / "finalizer.stdout.txt", finalizer_output)
        if finalizer_code != 0:
            raise SchemaError("finalizer failed", path=f"case.{case.id}.finalizer")
        artifact = artifact_path.read_bytes()
        artifact_sha = _sha256_bytes(artifact)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            type(report) is not dict
            or report.get("artifact_sha256") != artifact_sha
            or report.get("artifact_bytes") != len(artifact)
            or report.get("linked_manifest_id") != materialized.manifest.id
            or report.get("linked_manifest_digest") != manifest_digest
        ):
            raise SchemaError("finalizer report does not close", path=f"case.{case.id}.finalizer")
        with builder_validation_session():
            contract = preflight.bind_program_artifact_sha256(artifact_sha)
            contract.validate_against(materialized.manifest)
            sidecar_bytes = canonical_json(contract).encode("utf-8")
        program_io_digest = _sha256_bytes(sidecar_bytes)
        _write(sidecar_path, sidecar_bytes)
        resolver_code, resolver_output = _run(
            (
                str(self._resolver), "--resolve", str(manifest_path),
                str(artifact_path), str(sidecar_path),
            ),
            cwd=self._resolver.parent,
            timeout=min(self._timeout, 120),
        )
        _write(directory / "resolver.stdout.txt", resolver_output)
        if resolver_code != 0:
            raise SchemaError("resolver failed", path=f"case.{case.id}.resolver")
        expected_resolver = (
            f"initializations={len(contract.initializations)}",
            f"probes={len(contract.output_probes)}",
        )
        if any(marker not in resolver_output for marker in expected_resolver):
            raise SchemaError("resolver counts drifted", path=f"case.{case.id}.resolver")
        npusim_code, npusim_output = _run(
            (
                str(self._npusim), "--program-one-shot",
                "--program", str(artifact_path),
                "--linked-manifest", str(manifest_path),
                "--program-io", str(sidecar_path),
                "--hardware-config", str(hardware_path),
                "--simulation-config", str(self._simulation),
                "--mapping-config", str(mapping_path),
                "--trace-window", "1000000",
            ),
            cwd=self._npusim.parent,
            timeout=self._timeout,
        )
        _write(directory / "npusim.stdout.txt", npusim_output)
        if npusim_code != 0:
            raise SchemaError("npusim failed", path=f"case.{case.id}.npusim")
        observation = adapter.observe(
            case,
            materialized,
            contract,
            artifact_sha,
            npusim_output,
        )
        observation.validate(case)
        symbolic_records = _symbolic_record_count(materialized.manifest)
        exact_records = report.get("record_count")
        if type(exact_records) is not int:
            raise SchemaError("finalizer record count is absent", path="finalizer.report")
        capacity = FlexibleMeshReleaseCapacityEvidence(
            rank_count=case.mesh.rank_count,
            peak_sessions_per_core_per_wave=(
                materialized.peak_sessions_per_core_per_wave
            ),
            symbolic_record_count=symbolic_records,
            exact_record_count=exact_records,
            linked_manifest_file_bytes=len(manifest_bytes),
            artifact_file_bytes=len(artifact),
            max_runtime_core_id=_max_runtime_core_id(materialized.manifest),
            transport_tag_count=max(
                materialized.transport_tag_count,
                _transport_tag_count(materialized.manifest),
            ),
        )
        capacity.validate(case)
        semantic_resolver = resolver_output.replace(str(directory), "<EXECUTION>")
        run_prefix = f"{case.id}.execution{execution_index}"
        evidence = FlexibleMeshReleaseExecutionEvidence.create(
            case=case,
            binding=self._binding,
            execution_index=execution_index,
            run=FlexibleMeshIndependentRun(
                materialization_id=f"{run_prefix}.materialize",
                finalizer_run_id=f"{run_prefix}.finalizer",
                resolver_run_id=f"{run_prefix}.resolver",
                npusim_run_id=f"{run_prefix}.npusim",
            ),
            spec_digest=materialized.spec_digest,
            plan_digest=materialized.plan_digest,
            manifest_digest=manifest_digest,
            hardware_config_sha256=_sha256_text(materialized.hardware_json),
            simulation_config_sha256=_sha256_bytes(self._simulation.read_bytes()),
            mapping_config_sha256=_sha256_text(materialized.mapping_text),
            artifact_sha256=artifact_sha,
            artifact_file_bytes=len(artifact),
            program_io_artifact_sha256=contract.program_artifact_sha256,
            program_io_digest=program_io_digest,
            resolver_digest=_sha256_text(semantic_resolver),
            makespan_cycles=observation.makespan_cycles,
            marker_digest=observation.marker_digest,
            capacity=capacity,
            residual=observation.residual,
            rank_coverage=observation.rank_coverage,
            core_coverage=observation.core_coverage,
            completion_markers=observation.completion_markers,
            stage_exit_codes=tuple(
                (stage, 0) for stage in FlexibleMeshReleaseRuntimeStage
            ),
            program_io_phases=tuple(FlexibleMeshProgramIOPhase),
        )
        _write(directory / "execution_evidence.json", canonical_json(evidence))
        return evidence

    def run_case(
        self,
        case: FlexibleMeshReleaseCase,
    ) -> FlexibleMeshReleaseCaseEvidence:
        case.validate("case")
        if case.runtime_profile_version != self._binding.runtime_profile_version:
            raise SchemaError("runtime profile drifted", path="case")
        executions = tuple(self._execute_once(case, index) for index in (0, 1))
        result = FlexibleMeshReleaseCaseEvidence.create(
            case=case,
            binding=self._binding,
            executions=executions,
        )
        if not result.runtime_verified or not result.repeatability_verified:
            raise SchemaError("case repeatability failed", path=f"case.{case.id}")
        _write(
            self._runtime_root / case.id / "case_evidence.json",
            canonical_json(result),
        )
        return result

    def run_shard(
        self,
        cases: tuple[FlexibleMeshReleaseCase, ...],
        *,
        shard_index: int,
        shard_count: int,
    ) -> tuple[FlexibleMeshReleaseCaseEvidence, ...]:
        if (
            type(shard_index) is not int
            or type(shard_count) is not int
            or shard_count <= 0
            or not 0 <= shard_index < shard_count
        ):
            raise SchemaError("invalid shard", path="shard")
        selected = tuple(
            case for index, case in enumerate(cases)
            if index % shard_count == shard_index
        )
        result = tuple(self.run_case(case) for case in selected)
        _write(
            self._runtime_root / f"release_shard_{shard_index}_of_{shard_count}.json",
            canonical_json(result),
        )
        return result


__all__ = [
    "FlexibleMeshReleaseAdapter",
    "FlexibleMeshReleaseMaterialized",
    "FlexibleMeshReleaseObservation",
    "FlexibleMeshReleaseRunner",
]
