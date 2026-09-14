"""Fail-closed workspace runner for unified workload preflight artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Callable

from .errors import SchemaError
from .passes.workload_materialization import materialize_workload_preflight
from .workload_capability_evidence import WorkloadCapabilityDerivation
from .schema.common import stable_artifact_id, validate_nonempty
from .schema.ir1 import PhysicalFabric
from .schema.memory_plan import MemoryTierCapacity
from .schema.serde import (
    canonical_digest,
    canonical_json,
    load_json_dataclass,
)
from .schema.workload_materialization import (
    WorkloadMaterializationManifest,
    WorkloadMaterializationStatus,
)
from .schema.workload_run import WorkloadRunCapability, WorkloadRunRequest


WORKLOAD_RUNNER_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.workload_runner_binding/v1alpha2"
)
WORKLOAD_STAGE_ARTIFACT_SCHEMA_VERSION = (
    "wafer_frontend.workload_stage_artifact/v1alpha2"
)
WORKLOAD_RUNNER_STATUS_SCHEMA_VERSION = (
    "wafer_frontend.workload_runner_status/v1alpha2"
)
WORKLOAD_RUNNER_ERROR_SCHEMA_VERSION = (
    "wafer_frontend.workload_runner_error/v1alpha1"
)


class WorkloadRunnerStage(str, Enum):
    MATERIALIZE = "materialize"
    ADAPTER = "adapter"
    FINALIZE = "finalize"
    RESOLVE = "resolve"
    RUNTIME = "runtime"
    REPEATABILITY = "repeatability"


class WorkloadRunnerState(str, Enum):
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class WorkloadRunnerReadiness(str, Enum):
    NON_READY = "non_ready"
    READY = "ready"


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class WorkloadRunnerBinding:
    schema_version: str
    id: str
    request_digest: str
    capability_digest: str
    capacities_digest: str
    fabric_digest: str | None
    capability_derivation_digest: str | None
    adapter_id: str | None
    adapter_source_digest: str | None
    adapter_binary_digest: str | None
    adapter_toolchain_digest: str | None

    @classmethod
    def create(
        cls,
        *,
        request: WorkloadRunRequest,
        capability: WorkloadRunCapability,
        capacities: tuple[MemoryTierCapacity, ...],
        fabric: PhysicalFabric | None,
        capability_derivation_digest: str | None,
        adapter_id: str | None,
        adapter_source_digest: str | None,
        adapter_binary_digest: str | None,
        adapter_toolchain_digest: str | None,
    ) -> "WorkloadRunnerBinding":
        key = {
            "request_digest": request.digest,
            "capability_digest": capability.digest,
            "capacities_digest": canonical_digest(capacities),
            "fabric_digest": canonical_digest(fabric) if fabric is not None else None,
            "capability_derivation_digest": capability_derivation_digest,
            "adapter_id": adapter_id,
            "adapter_source_digest": adapter_source_digest,
            "adapter_binary_digest": adapter_binary_digest,
            "adapter_toolchain_digest": adapter_toolchain_digest,
        }
        result = cls(
            schema_version=WORKLOAD_RUNNER_BINDING_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_runner_binding",
                key,
                schema_version=WORKLOAD_RUNNER_BINDING_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "workload_runner_binding") -> None:
        if self.schema_version != WORKLOAD_RUNNER_BINDING_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in ("request_digest", "capability_digest", "capacities_digest"):
            _digest(getattr(self, name), f"{path}.{name}")
        if self.fabric_digest is not None:
            _digest(self.fabric_digest, f"{path}.fabric_digest")
        if self.capability_derivation_digest is not None:
            _digest(
                self.capability_derivation_digest,
                f"{path}.capability_derivation_digest",
            )
        adapter_bindings = (
            self.adapter_source_digest,
            self.adapter_binary_digest,
            self.adapter_toolchain_digest,
        )
        if self.adapter_id is not None:
            validate_nonempty(self.adapter_id, f"{path}.adapter_id")
        if (
            self.adapter_id is None
            and any(item is not None for item in adapter_bindings)
        ) or (
            self.adapter_id is not None
            and any(item is None for item in adapter_bindings)
        ):
            raise SchemaError(
                "adapter id and all source/binary/toolchain digests must appear together",
                path=path,
            )
        for name in (
            "adapter_source_digest",
            "adapter_binary_digest",
            "adapter_toolchain_digest",
        ):
            value = getattr(self, name)
            if value is not None:
                _digest(value, f"{path}.{name}")
        expected = stable_artifact_id(
            "workload_runner_binding",
            self._key(),
            schema_version=WORKLOAD_RUNNER_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable binding id", path=f"{path}.id")


def _validate_relative_file_path(value: str, path: str) -> None:
    validate_nonempty(value, path)
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or value != pure.as_posix()
        or "\\" in value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise SchemaError(
            "must be a normalized relative POSIX path",
            path=path,
            code="workload_artifact_integrity",
        )


def _inspect_stage_file(
    work_dir: Path,
    relative_path: str,
    path: str,
) -> tuple[int, str]:
    _validate_relative_file_path(relative_path, path)
    try:
        root = work_dir.resolve(strict=True)
        candidate = work_dir.joinpath(*PurePosixPath(relative_path).parts)
        cursor = work_dir
        for part in PurePosixPath(relative_path).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise SchemaError(
                    "symbolic links are not valid stage artifacts",
                    path=path,
                    code="workload_artifact_integrity",
                )
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise SchemaError(
                "stage artifact is not a regular file inside the execution directory",
                path=path,
                code="workload_artifact_integrity",
            )
        size_bytes = 0
        content_hasher = hashlib.sha256()
        with resolved.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                size_bytes += len(chunk)
                content_hasher.update(chunk)
        return size_bytes, content_hasher.hexdigest()
    except SchemaError:
        raise
    except (OSError, RuntimeError) as error:
        raise SchemaError(
            f"cannot read stage artifact: {error}",
            path=path,
            code="workload_artifact_integrity",
        ) from error


@dataclass(frozen=True, slots=True)
class WorkloadStageFile:
    relative_path: str
    size_bytes: int
    content_digest: str

    @classmethod
    def capture(
        cls,
        *,
        work_dir: Path,
        relative_path: str,
        path: str = "workload_stage_file",
    ) -> "WorkloadStageFile":
        size_bytes, content_digest = _inspect_stage_file(
            work_dir,
            relative_path,
            f"{path}.relative_path",
        )
        result = cls(
            relative_path=relative_path,
            size_bytes=size_bytes,
            content_digest=content_digest,
        )
        result.validate(path)
        return result

    def validate(self, path: str = "workload_stage_file") -> None:
        _validate_relative_file_path(self.relative_path, f"{path}.relative_path")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise SchemaError("must be a non-negative int", path=f"{path}.size_bytes")
        _digest(self.content_digest, f"{path}.content_digest")


def _stage_output_digest(
    stage: WorkloadRunnerStage,
    input_digest: str,
    files: tuple[WorkloadStageFile, ...],
) -> str:
    return canonical_digest(
        {
            "stage": stage,
            "input_digest": input_digest,
            "files": files,
        }
    )


@dataclass(frozen=True, slots=True)
class WorkloadStageArtifact:
    schema_version: str
    id: str
    stage: WorkloadRunnerStage
    input_digest: str
    files: tuple[WorkloadStageFile, ...]
    output_digest: str
    one_shot_workload_end: bool

    @classmethod
    def create(
        cls,
        *,
        stage: WorkloadRunnerStage,
        input_digest: str,
        work_dir: Path,
        relative_paths: tuple[str, ...],
        one_shot_workload_end: bool = False,
    ) -> "WorkloadStageArtifact":
        if not isinstance(work_dir, Path):
            raise SchemaError("must be a pathlib.Path", path="work_dir")
        if type(relative_paths) is not tuple or not relative_paths:
            raise SchemaError(
                "must be a non-empty tuple",
                path="relative_paths",
                code="workload_artifact_integrity",
            )
        files = tuple(
            WorkloadStageFile.capture(
                work_dir=work_dir,
                relative_path=relative_path,
                path=f"relative_paths[{index}]",
            )
            for index, relative_path in enumerate(relative_paths)
        )
        files = tuple(sorted(files, key=lambda item: item.relative_path))
        key = {
            "stage": stage,
            "input_digest": input_digest,
            "files": files,
            "output_digest": _stage_output_digest(stage, input_digest, files),
            "one_shot_workload_end": one_shot_workload_end,
        }
        result = cls(
            schema_version=WORKLOAD_STAGE_ARTIFACT_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_stage_artifact",
                key,
                schema_version=WORKLOAD_STAGE_ARTIFACT_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "workload_stage_artifact") -> None:
        if self.schema_version != WORKLOAD_STAGE_ARTIFACT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.stage) is not WorkloadRunnerStage:
            raise SchemaError("must be a WorkloadRunnerStage", path=f"{path}.stage")
        _digest(self.input_digest, f"{path}.input_digest")
        if type(self.files) is not tuple or not self.files:
            raise SchemaError("must be a non-empty tuple", path=f"{path}.files")
        for index, artifact_file in enumerate(self.files):
            if type(artifact_file) is not WorkloadStageFile:
                raise SchemaError(
                    "must be a WorkloadStageFile", path=f"{path}.files[{index}]"
                )
            artifact_file.validate(f"{path}.files[{index}]")
        if self.files != tuple(sorted(self.files, key=lambda item: item.relative_path)):
            raise SchemaError("must use canonical path order", path=f"{path}.files")
        paths = tuple(item.relative_path for item in self.files)
        if len(set(paths)) != len(paths):
            raise SchemaError("must not contain duplicate paths", path=f"{path}.files")
        _digest(self.output_digest, f"{path}.output_digest")
        expected_output = _stage_output_digest(
            self.stage,
            self.input_digest,
            self.files,
        )
        if self.output_digest != expected_output:
            raise SchemaError(
                "does not match the stage file manifest",
                path=f"{path}.output_digest",
                code="workload_artifact_integrity",
            )
        if type(self.one_shot_workload_end) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.one_shot_workload_end")
        if self.one_shot_workload_end != (self.stage is WorkloadRunnerStage.RUNTIME):
            raise SchemaError(
                "only the final runtime stage may end the workload one-shot",
                path=f"{path}.one_shot_workload_end",
            )
        expected = stable_artifact_id(
            "workload_stage_artifact",
            self._key(),
            schema_version=WORKLOAD_STAGE_ARTIFACT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable stage artifact id", path=f"{path}.id")

    def verify_files(
        self,
        work_dir: Path,
        path: str = "workload_stage_artifact",
    ) -> None:
        self.validate(path)
        observed = tuple(
            WorkloadStageFile.capture(
                work_dir=work_dir,
                relative_path=item.relative_path,
                path=f"{path}.files[{index}]",
            )
            for index, item in enumerate(self.files)
        )
        if observed != self.files:
            raise SchemaError(
                "stage artifact file size or content digest changed",
                path=f"{path}.files",
                code="workload_artifact_integrity",
            )


@dataclass(frozen=True, slots=True)
class WorkloadStageContext:
    execution_index: int
    work_dir: Path
    manifest: WorkloadMaterializationManifest
    input_digest: str


WorkloadStageCallable = Callable[[WorkloadStageContext], WorkloadStageArtifact]


@dataclass(frozen=True, slots=True)
class WorkloadBackendAdapter:
    id: str
    source_digest: str
    binary_digest: str
    toolchain_digest: str
    adapter: WorkloadStageCallable
    finalizer: WorkloadStageCallable
    resolver: WorkloadStageCallable
    runtime: WorkloadStageCallable

    def validate(self, path: str = "adapter") -> None:
        validate_nonempty(self.id, f"{path}.id")
        _digest(self.source_digest, f"{path}.source_digest")
        _digest(self.binary_digest, f"{path}.binary_digest")
        _digest(self.toolchain_digest, f"{path}.toolchain_digest")
        for name in ("adapter", "finalizer", "resolver", "runtime"):
            if not callable(getattr(self, name)):
                raise SchemaError("must be callable", path=f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class WorkloadRunnerStatus:
    schema_version: str
    id: str
    state: WorkloadRunnerState
    readiness: WorkloadRunnerReadiness
    readiness_reasons: tuple[str, ...]
    binding_digest: str
    manifest_digest: str | None
    completed_stage: WorkloadRunnerStage | None
    execution_digests: tuple[str, ...]
    runtime_evidence_materialized: bool

    @classmethod
    def create(
        cls,
        *,
        state: WorkloadRunnerState,
        readiness: WorkloadRunnerReadiness,
        readiness_reasons: tuple[str, ...],
        binding: WorkloadRunnerBinding,
        manifest_digest: str | None,
        completed_stage: WorkloadRunnerStage | None,
        execution_digests: tuple[str, ...] = (),
    ) -> "WorkloadRunnerStatus":
        key = {
            "state": state,
            "readiness": readiness,
            "readiness_reasons": tuple(sorted(readiness_reasons)),
            "binding_digest": canonical_digest(binding),
            "manifest_digest": manifest_digest,
            "completed_stage": completed_stage,
            "execution_digests": execution_digests,
            "runtime_evidence_materialized": False,
        }
        result = cls(
            schema_version=WORKLOAD_RUNNER_STATUS_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_runner_status",
                key,
                schema_version=WORKLOAD_RUNNER_STATUS_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "workload_runner_status") -> None:
        if self.schema_version != WORKLOAD_RUNNER_STATUS_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.state) is not WorkloadRunnerState:
            raise SchemaError("must be a WorkloadRunnerState", path=f"{path}.state")
        if type(self.readiness) is not WorkloadRunnerReadiness:
            raise SchemaError(
                "must be a WorkloadRunnerReadiness", path=f"{path}.readiness"
            )
        if type(self.readiness_reasons) is not tuple:
            raise SchemaError("must be a tuple", path=f"{path}.readiness_reasons")
        for index, reason in enumerate(self.readiness_reasons):
            validate_nonempty(reason, f"{path}.readiness_reasons[{index}]")
        if self.readiness_reasons != tuple(sorted(set(self.readiness_reasons))):
            raise SchemaError(
                "must be sorted and unique", path=f"{path}.readiness_reasons"
            )
        if (self.readiness is WorkloadRunnerReadiness.READY) == bool(
            self.readiness_reasons
        ):
            raise SchemaError(
                "ready requires no reasons and non-ready requires reasons",
                path=f"{path}.readiness_reasons",
            )
        if (
            self.readiness is WorkloadRunnerReadiness.NON_READY
            and self.execution_digests
        ):
            raise SchemaError(
                "non-ready status cannot publish executions", path=path
            )
        _digest(self.binding_digest, f"{path}.binding_digest")
        if self.manifest_digest is not None:
            _digest(self.manifest_digest, f"{path}.manifest_digest")
        if self.completed_stage is not None and type(self.completed_stage) is not WorkloadRunnerStage:
            raise SchemaError("must be a WorkloadRunnerStage", path=f"{path}.completed_stage")
        for index, digest in enumerate(self.execution_digests):
            _digest(digest, f"{path}.execution_digests[{index}]")
        if self.state is WorkloadRunnerState.FAILED and self.execution_digests:
            raise SchemaError("failed status cannot publish executions", path=path)
        if self.state in (
            WorkloadRunnerState.PARTIAL,
            WorkloadRunnerState.UNSUPPORTED,
        ) and self.manifest_digest is None:
            raise SchemaError(
                "non-failed status requires a manifest digest", path=path
            )
        if self.state is WorkloadRunnerState.PARTIAL:
            if (
                self.readiness is WorkloadRunnerReadiness.READY
                and self.completed_stage is not WorkloadRunnerStage.REPEATABILITY
            ):
                raise SchemaError(
                    "ready partial status requires completed executions", path=path
                )
            if (
                self.readiness is WorkloadRunnerReadiness.NON_READY
                and self.completed_stage is not WorkloadRunnerStage.MATERIALIZE
            ):
                raise SchemaError(
                    "non-ready partial status must stop after materialization",
                    path=path,
                )
        if self.state is WorkloadRunnerState.UNSUPPORTED:
            if self.readiness is not WorkloadRunnerReadiness.NON_READY:
                raise SchemaError("unsupported status cannot be ready", path=path)
            if self.completed_stage is not WorkloadRunnerStage.MATERIALIZE:
                raise SchemaError(
                    "unsupported status must stop after materialization", path=path
                )
            if self.execution_digests:
                raise SchemaError("unsupported status cannot publish executions", path=path)
        if (
            self.state is WorkloadRunnerState.PARTIAL
            and self.completed_stage is WorkloadRunnerStage.REPEATABILITY
            and not self.execution_digests
        ):
            raise SchemaError(
                "repeatability completion requires execution digests", path=path
            )
        if self.runtime_evidence_materialized is not False:
            raise SchemaError(
                "P0.4 cannot materialize runtime evidence",
                path=f"{path}.runtime_evidence_materialized",
            )
        expected = stable_artifact_id(
            "workload_runner_status",
            self._key(),
            schema_version=WORKLOAD_RUNNER_STATUS_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable runner status id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class WorkloadRunnerError:
    schema_version: str
    id: str
    stage: WorkloadRunnerStage
    binding_digest: str
    error_type: str
    message: str

    @classmethod
    def create(
        cls,
        *,
        stage: WorkloadRunnerStage,
        binding: WorkloadRunnerBinding,
        error: BaseException,
    ) -> "WorkloadRunnerError":
        key = {
            "stage": stage,
            "binding_digest": canonical_digest(binding),
            "error_type": type(error).__name__,
            "message": str(error),
        }
        result = cls(
            schema_version=WORKLOAD_RUNNER_ERROR_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_runner_error",
                key,
                schema_version=WORKLOAD_RUNNER_ERROR_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "workload_runner_error") -> None:
        if self.schema_version != WORKLOAD_RUNNER_ERROR_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.stage) is not WorkloadRunnerStage:
            raise SchemaError("must be a WorkloadRunnerStage", path=f"{path}.stage")
        _digest(self.binding_digest, f"{path}.binding_digest")
        validate_nonempty(self.error_type, f"{path}.error_type")
        validate_nonempty(self.message, f"{path}.message")
        expected = stable_artifact_id(
            "workload_runner_error",
            self._key(),
            schema_version=WORKLOAD_RUNNER_ERROR_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable runner error id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class WorkloadRunnerResult:
    output_dir: Path
    binding: WorkloadRunnerBinding
    manifest: WorkloadMaterializationManifest
    status: WorkloadRunnerStatus
    resumed: bool


def _write(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _canonical_capacities(
    capacities: tuple[MemoryTierCapacity, ...],
) -> tuple[MemoryTierCapacity, ...]:
    if type(capacities) is not tuple or not capacities:
        raise SchemaError("must be a non-empty tuple", path="capacities")
    for index, capacity in enumerate(capacities):
        if type(capacity) is not MemoryTierCapacity:
            raise SchemaError(
                "must be a MemoryTierCapacity", path=f"capacities[{index}]"
            )
        capacity.validate(f"capacities[{index}]")
    return tuple(sorted(capacities, key=lambda item: (item.tier.value, item.location_ref)))


def _load_stage_artifact(
    execution_dir: Path,
    stage: WorkloadRunnerStage,
) -> WorkloadStageArtifact:
    path = f"{execution_dir.name}.{stage.value}"
    try:
        return load_json_dataclass(
            WorkloadStageArtifact,
            execution_dir / f"{stage.value}.json",
            path=path,
        )
    except SchemaError as error:
        if error.code == "workload_artifact_integrity":
            raise
        raise SchemaError(
            f"cannot load stage artifact manifest: {error}",
            path=path,
            code="workload_artifact_integrity",
        ) from error


def _verify_execution(
    execution_dir: Path,
    *,
    manifest_digest: str,
) -> str:
    input_digest = manifest_digest
    for stage in (
        WorkloadRunnerStage.ADAPTER,
        WorkloadRunnerStage.FINALIZE,
        WorkloadRunnerStage.RESOLVE,
        WorkloadRunnerStage.RUNTIME,
    ):
        artifact = _load_stage_artifact(execution_dir, stage)
        if artifact.stage is not stage:
            raise SchemaError(
                "stage artifact manifest has the wrong stage",
                path=f"{execution_dir.name}.{stage.value}",
                code="workload_artifact_integrity",
            )
        if artifact.input_digest != input_digest:
            raise SchemaError(
                "stage artifact digest chain is broken",
                path=f"{execution_dir.name}.{stage.value}.input_digest",
                code="workload_artifact_integrity",
            )
        artifact.verify_files(
            execution_dir,
            path=f"{execution_dir.name}.{stage.value}",
        )
        input_digest = artifact.output_digest
    return input_digest


def _execution_directory_names(output_dir: Path) -> set[str]:
    try:
        return {
            child.name
            for child in output_dir.iterdir()
            if child.name.startswith("execution_")
        }
    except OSError as error:
        raise SchemaError(
            f"cannot inspect execution directories: {error}",
            path="resume.executions",
            code="workload_artifact_integrity",
        ) from error

def _resume(output_dir: Path, binding: WorkloadRunnerBinding) -> WorkloadRunnerResult:
    old_binding = load_json_dataclass(
        WorkloadRunnerBinding,
        output_dir / "input_binding.json",
        path="input_binding",
    )
    if old_binding != binding:
        raise SchemaError(
            "resume input binding digest mismatch",
            path="resume",
            code="workload_resume_stale",
        )
    derivation_path = output_dir / "capability_derivation.json"
    if binding.capability_derivation_digest is None:
        if derivation_path.exists():
            raise SchemaError(
                "unexpected capability derivation artifact",
                path="resume.capability_derivation",
                code="workload_resume_stale",
            )
    else:
        saved_derivation = load_json_dataclass(
            WorkloadCapabilityDerivation,
            derivation_path,
            path="capability_derivation",
        )
        if saved_derivation.digest != binding.capability_derivation_digest:
            raise SchemaError(
                "capability derivation digest mismatch",
                path="resume.capability_derivation",
                code="workload_resume_stale",
            )
    saved_request = load_json_dataclass(
        WorkloadRunRequest,
        output_dir / "request.json",
        path="request",
    )
    if saved_request.digest != binding.request_digest:
        raise SchemaError(
            "request artifact digest mismatch",
            path="resume.request",
            code="workload_resume_stale",
        )
    manifest = load_json_dataclass(
        WorkloadMaterializationManifest,
        output_dir / "manifest.json",
        path="manifest",
    )
    status = load_json_dataclass(
        WorkloadRunnerStatus,
        output_dir / "status.json",
        path="status",
    )
    if status.binding_digest != canonical_digest(binding):
        raise SchemaError("status binding mismatch", path="resume.status")
    if status.manifest_digest != manifest.digest:
        raise SchemaError("status manifest mismatch", path="resume.status")
    if (
        manifest.request != saved_request
        or manifest.request_digest != binding.request_digest
        or manifest.capability_digest != binding.capability_digest
    ):
        raise SchemaError(
            "manifest input binding mismatch",
            path="resume.manifest",
            code="workload_resume_stale",
        )
    if status.state is WorkloadRunnerState.FAILED:
        raise SchemaError(
            "failed workspaces cannot be resumed as completed results",
            path="resume.status",
            code="workload_resume_failed",
        )
    expected_execution_names: set[str] = set()
    if status.completed_stage is WorkloadRunnerStage.REPEATABILITY:
        if binding.adapter_id is None:
            raise SchemaError(
                "execution artifacts require an adapter binding",
                path="resume.executions",
                code="workload_artifact_integrity",
            )
        expected_execution_names = {
            f"execution_{index}"
            for index in range(saved_request.execution.independent_repeats)
        }
        observed_digests = tuple(
            _verify_execution(
                output_dir / f"execution_{index}",
                manifest_digest=manifest.digest,
            )
            for index in range(saved_request.execution.independent_repeats)
        )
        if observed_digests != status.execution_digests:
            raise SchemaError(
                "execution artifact digests do not match status",
                path="resume.executions",
                code="workload_artifact_integrity",
            )
    if _execution_directory_names(output_dir) != expected_execution_names:
        raise SchemaError(
            "execution directory set does not match status",
            path="resume.executions",
            code="workload_artifact_integrity",
        )
    return WorkloadRunnerResult(output_dir, binding, manifest, status, True)


def _invoke(
    callback: WorkloadStageCallable,
    *,
    expected_stage: WorkloadRunnerStage,
    execution_index: int,
    work_dir: Path,
    manifest: WorkloadMaterializationManifest,
    input_digest: str,
) -> WorkloadStageArtifact:
    result = callback(
        WorkloadStageContext(
            execution_index=execution_index,
            work_dir=work_dir,
            manifest=manifest,
            input_digest=input_digest,
        )
    )
    if type(result) is not WorkloadStageArtifact:
        raise SchemaError("adapter returned an untyped artifact", path=expected_stage.value)
    result.validate(expected_stage.value)
    if result.stage is not expected_stage:
        raise SchemaError("adapter returned the wrong stage", path=expected_stage.value)
    if result.input_digest != input_digest:
        raise SchemaError("adapter input digest mismatch", path=expected_stage.value)
    result.verify_files(work_dir, path=expected_stage.value)
    return result


def _runner_readiness(
    request: WorkloadRunRequest,
    capability: WorkloadRunCapability,
    capability_derivation: WorkloadCapabilityDerivation | None,
    adapter: WorkloadBackendAdapter | None,
) -> tuple[WorkloadRunnerReadiness, tuple[str, ...]]:
    if adapter is None:
        return WorkloadRunnerReadiness.NON_READY, ("backend_adapter",)
    if capability_derivation is None:
        return WorkloadRunnerReadiness.NON_READY, ("capability_evidence",)
    if capability_derivation.capability != capability:
        return (
            WorkloadRunnerReadiness.NON_READY,
            ("capability_not_evidence_derived",),
        )
    reasons = capability_derivation.readiness_reasons(
        request,
        source_digest=adapter.source_digest,
        binary_digest=adapter.binary_digest,
        toolchain_digest=adapter.toolchain_digest,
    )
    if reasons:
        return WorkloadRunnerReadiness.NON_READY, reasons
    return WorkloadRunnerReadiness.READY, ()


def run_workload(
    request: WorkloadRunRequest,
    capability: WorkloadRunCapability,
    *,
    capacities: tuple[MemoryTierCapacity, ...],
    output_dir: Path,
    fabric: PhysicalFabric | None = None,
    adapter: WorkloadBackendAdapter | None = None,
    capability_derivation: WorkloadCapabilityDerivation | None = None,
    resume: bool = False,
) -> WorkloadRunnerResult:
    """Materialize one workload workspace and optionally invoke a typed backend."""

    if not isinstance(output_dir, Path):
        raise SchemaError("must be a pathlib.Path", path="output_dir")
    if type(resume) is not bool:
        raise SchemaError("must be a bool", path="resume")
    request.validate("request")
    capability.validate("capability")
    capacities = _canonical_capacities(capacities)
    if fabric is not None:
        if type(fabric) is not PhysicalFabric:
            raise SchemaError("must be a PhysicalFabric", path="fabric")
        fabric.validate("fabric")
    if adapter is not None:
        if type(adapter) is not WorkloadBackendAdapter:
            raise SchemaError("must be a WorkloadBackendAdapter", path="adapter")
        adapter.validate()
    if capability_derivation is not None:
        if type(capability_derivation) is not WorkloadCapabilityDerivation:
            raise SchemaError(
                "must be a WorkloadCapabilityDerivation",
                path="capability_derivation",
            )
        capability_derivation.validate("capability_derivation")
    readiness, readiness_reasons = _runner_readiness(
        request,
        capability,
        capability_derivation,
        adapter,
    )
    binding = WorkloadRunnerBinding.create(
        request=request,
        capability=capability,
        capacities=capacities,
        fabric=fabric,
        capability_derivation_digest=(
            capability_derivation.digest
            if capability_derivation is not None
            else None
        ),
        adapter_id=adapter.id if adapter is not None else None,
        adapter_source_digest=(
            adapter.source_digest if adapter is not None else None
        ),
        adapter_binary_digest=(
            adapter.binary_digest if adapter is not None else None
        ),
        adapter_toolchain_digest=(
            adapter.toolchain_digest if adapter is not None else None
        ),
    )
    if output_dir.exists():
        if not resume:
            raise SchemaError("output directory already exists", path="output_dir")
        return _resume(output_dir, binding)
    output_dir.mkdir(parents=True)
    _write(output_dir / "request.json", request)
    _write(output_dir / "input_binding.json", binding)
    if capability_derivation is not None:
        _write(
            output_dir / "capability_derivation.json",
            capability_derivation,
        )
    stage = WorkloadRunnerStage.MATERIALIZE
    manifest: WorkloadMaterializationManifest | None = None
    try:
        manifest = materialize_workload_preflight(
            request,
            capability,
            capacities=capacities,
            fabric=fabric,
        )
        _write(output_dir / "manifest.json", manifest)
        if manifest.status is WorkloadMaterializationStatus.UNSUPPORTED:
            status = WorkloadRunnerStatus.create(
                state=WorkloadRunnerState.UNSUPPORTED,
                readiness=WorkloadRunnerReadiness.NON_READY,
                readiness_reasons=("manifest_unsupported",),
                binding=binding,
                manifest_digest=manifest.digest,
                completed_stage=WorkloadRunnerStage.MATERIALIZE,
            )
            _write(output_dir / "status.json", status)
            return WorkloadRunnerResult(output_dir, binding, manifest, status, False)
        if readiness is WorkloadRunnerReadiness.NON_READY:
            status = WorkloadRunnerStatus.create(
                state=WorkloadRunnerState.PARTIAL,
                readiness=readiness,
                readiness_reasons=readiness_reasons,
                binding=binding,
                manifest_digest=manifest.digest,
                completed_stage=WorkloadRunnerStage.MATERIALIZE,
            )
            _write(output_dir / "status.json", status)
            return WorkloadRunnerResult(output_dir, binding, manifest, status, False)

        assert adapter is not None
        callbacks = (
            (WorkloadRunnerStage.ADAPTER, adapter.adapter),
            (WorkloadRunnerStage.FINALIZE, adapter.finalizer),
            (WorkloadRunnerStage.RESOLVE, adapter.resolver),
            (WorkloadRunnerStage.RUNTIME, adapter.runtime),
        )
        for execution_index in range(request.execution.independent_repeats):
            execution_dir = output_dir / f"execution_{execution_index}"
            execution_dir.mkdir()
            input_digest = manifest.digest
            for stage, callback in callbacks:
                artifact = _invoke(
                    callback,
                    expected_stage=stage,
                    execution_index=execution_index,
                    work_dir=execution_dir,
                    manifest=manifest,
                    input_digest=input_digest,
                )
                artifact_path = execution_dir / f"{stage.value}.json"
                _write(artifact_path, artifact)
                persisted = _load_stage_artifact(execution_dir, stage)
                if persisted != artifact:
                    raise SchemaError(
                        "persisted stage manifest differs from adapter result",
                        path=f"execution_{execution_index}.{stage.value}",
                        code="workload_artifact_integrity",
                    )
                persisted.verify_files(
                    execution_dir,
                    path=f"execution_{execution_index}.{stage.value}",
                )
                input_digest = persisted.output_digest
        stage = WorkloadRunnerStage.REPEATABILITY
        runtime_digests = [
            _verify_execution(
                output_dir / f"execution_{execution_index}",
                manifest_digest=manifest.digest,
            )
            for execution_index in range(request.execution.independent_repeats)
        ]
        if len(set(runtime_digests)) != 1:
            raise SchemaError(
                "independent execution digests differ",
                path="executions",
                code="workload_repeatability_mismatch",
            )
        status = WorkloadRunnerStatus.create(
            state=WorkloadRunnerState.PARTIAL,
            readiness=WorkloadRunnerReadiness.READY,
            readiness_reasons=(),
            binding=binding,
            manifest_digest=manifest.digest,
            completed_stage=WorkloadRunnerStage.REPEATABILITY,
            execution_digests=tuple(runtime_digests),
        )
        _write(output_dir / "status.json", status)
        return WorkloadRunnerResult(output_dir, binding, manifest, status, False)
    except Exception as error:
        failure = WorkloadRunnerError.create(stage=stage, binding=binding, error=error)
        _write(output_dir / "error.json", failure)
        status = WorkloadRunnerStatus.create(
            state=WorkloadRunnerState.FAILED,
            readiness=readiness,
            readiness_reasons=readiness_reasons,
            binding=binding,
            manifest_digest=manifest.digest if manifest is not None else None,
            completed_stage=stage,
        )
        _write(output_dir / "status.json", status)
        raise


__all__ = [
    "WORKLOAD_RUNNER_BINDING_SCHEMA_VERSION",
    "WORKLOAD_RUNNER_ERROR_SCHEMA_VERSION",
    "WORKLOAD_RUNNER_STATUS_SCHEMA_VERSION",
    "WORKLOAD_STAGE_ARTIFACT_SCHEMA_VERSION",
    "WorkloadBackendAdapter",
    "WorkloadRunnerBinding",
    "WorkloadRunnerError",
    "WorkloadRunnerResult",
    "WorkloadRunnerReadiness",
    "WorkloadRunnerStage",
    "WorkloadRunnerState",
    "WorkloadRunnerStatus",
    "WorkloadStageArtifact",
    "WorkloadStageFile",
    "WorkloadStageCallable",
    "WorkloadStageContext",
    "run_workload",
]
