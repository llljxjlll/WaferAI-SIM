"""Typed evidence for real flexible-Mesh timing execution.

The carriers in this module deliberately distinguish compilation from an
external finalizer/resolver/NpuSim observation.  In particular, no constructor
can produce ``runtime_verified`` without a zero-exit runtime marker and a
zero-valued residual parsed from simulator output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .flexible_mesh_workload import (
    FlexibleMeshSliceOperation,
    FlexibleMeshWorkloadSpec,
)


FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_runtime/v1alpha1"
)
FLEXIBLE_MESH_RUNTIME_MARKER_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_runtime_marker/v1alpha1"
)


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


class FlexibleMeshRuntimeBaseline(str, Enum):
    MESHSLICE_STANDARD = "MESHSLICE_STANDARD"
    UNFUSED_FALLBACK = "UNFUSED_FALLBACK"


class FlexibleMeshRuntimeStage(str, Enum):
    SCHEMA = "schema"
    CANDIDATE = "candidate"
    LOWER_LINK = "lower_link"
    PROGRAM_IO = "program_io"
    FINALIZER = "finalizer"
    RESOLVER = "resolver"
    NPUSIM = "npusim"
    REPEATABILITY = "repeatability"


class FlexibleMeshRuntimeStageStatus(str, Enum):
    VERIFIED = "verified"
    NOT_MEASURED = "not_measured"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FlexibleMeshRuntimeCase:
    schema_version: str
    id: str
    workload: FlexibleMeshWorkloadSpec
    operation: FlexibleMeshSliceOperation
    selected_baseline: FlexibleMeshRuntimeBaseline
    fallback_reason: str | None
    expected_rank_count: int
    expected_state_owners: tuple[int, ...]
    repeat_count: int

    @classmethod
    def create(
        cls,
        workload: FlexibleMeshWorkloadSpec,
        operation: FlexibleMeshSliceOperation,
        *,
        repeat_count: int = 1,
    ) -> "FlexibleMeshRuntimeCase":
        fallback = operation in (
            FlexibleMeshSliceOperation.GEMM_RS,
            FlexibleMeshSliceOperation.GEMM_AR,
        )
        semantic = {
            "workload": workload,
            "operation": operation,
            "selected_baseline": (
                FlexibleMeshRuntimeBaseline.UNFUSED_FALLBACK
                if fallback
                else FlexibleMeshRuntimeBaseline.MESHSLICE_STANDARD
            ),
            "fallback_reason": (
                "STRICT_TWO_INPUT_REDUCE_ABI" if fallback else None
            ),
            "expected_rank_count": workload.mesh.rank_count,
            "expected_state_owners": (),
            "repeat_count": repeat_count,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_runtime_case",
                semantic,
                schema_version=FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "flexible_mesh_runtime_case") -> None:
        if self.schema_version != FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.workload.validate(f"{path}.workload")
        if type(self.operation) is not FlexibleMeshSliceOperation:
            raise SchemaError("must be a MeshSlice operation", path=f"{path}.operation")
        if self.operation not in self.workload.meshslice.operations:
            raise SchemaError("operation is disabled by workload", path=f"{path}.operation")
        fallback = self.operation in (
            FlexibleMeshSliceOperation.GEMM_RS,
            FlexibleMeshSliceOperation.GEMM_AR,
        )
        expected = (
            FlexibleMeshRuntimeBaseline.UNFUSED_FALLBACK
            if fallback
            else FlexibleMeshRuntimeBaseline.MESHSLICE_STANDARD
        )
        if self.selected_baseline is not expected:
            raise SchemaError("selected baseline does not match operation", path=path)
        expected_reason = "STRICT_TWO_INPUT_REDUCE_ABI" if fallback else None
        if self.fallback_reason != expected_reason:
            raise SchemaError("fallback reason is not exact", path=f"{path}.fallback_reason")
        if self.expected_rank_count != self.workload.mesh.rank_count:
            raise SchemaError("rank count drifted from Mesh", path=f"{path}.expected_rank_count")
        if self.expected_state_owners != ():
            raise SchemaError("MeshSlice has no persistent state owners", path=f"{path}.expected_state_owners")
        if type(self.repeat_count) is not int or self.repeat_count not in (1, 2):
            raise SchemaError("repeat count must be one or two", path=f"{path}.repeat_count")
        expected_id = stable_artifact_id(
            "flexible_mesh_runtime_case",
            self._semantic_key(),
            schema_version=FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable runtime case id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMeshRuntimeResidual:
    active_endpoints: int
    active_sessions: int
    outstanding_tags: int
    incomplete_barriers: int
    pending_state_writes: int
    proto_wait_count: int
    credit_residual: int

    def validate(self, path: str = "flexible_mesh_runtime_residual") -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")

    @property
    def is_zero(self) -> bool:
        self.validate()
        return all(getattr(self, name) == 0 for name in self.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class FlexibleMeshArtifactCapacityEvidence:
    record_count: int
    relocation_count: int
    runtime_symbol_count: int
    artifact_file_bytes: int
    core_count: int

    def validate(
        self,
        case: FlexibleMeshRuntimeCase,
        path: str = "flexible_mesh_artifact_capacity",
    ) -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.record_count > case.workload.capacity.max_symbolic_records
            or self.artifact_file_bytes
            > case.workload.capacity.max_artifact_file_bytes
            or self.core_count != case.expected_rank_count
            or self.core_count > 0xFFFF
        ):
            raise SchemaError("final artifact exceeds exact capacity", path=path)


@dataclass(frozen=True, slots=True)
class FlexibleMeshRuntimeMarker:
    schema_version: str
    mesh_digest: str
    workload_digest: str
    manifest_digest: str
    program_io_digest: str
    makespan_cycles: int
    rank_coverage: tuple[int, ...]
    core_coverage: tuple[int, ...]
    active_routes: tuple[str, ...]
    state_completion: tuple[int, ...]
    residual: FlexibleMeshRuntimeResidual
    marker_digest: str

    def validate(
        self,
        case: FlexibleMeshRuntimeCase,
        path: str = "flexible_mesh_runtime_marker",
    ) -> None:
        if self.schema_version != FLEXIBLE_MESH_RUNTIME_MARKER_SCHEMA_VERSION:
            raise SchemaError("unsupported marker schema", path=f"{path}.schema_version")
        for name in (
            "mesh_digest",
            "workload_digest",
            "manifest_digest",
            "program_io_digest",
            "marker_digest",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        if (
            self.mesh_digest != case.workload.mesh.digest
            or self.workload_digest != case.workload.digest
        ):
            raise SchemaError("marker input digest drifted", path=path)
        validate_uint64(self.makespan_cycles, f"{path}.makespan_cycles")
        if self.makespan_cycles == 0:
            raise SchemaError("makespan must be positive", path=f"{path}.makespan_cycles")
        expected_ranks = tuple(range(case.expected_rank_count))
        if self.rank_coverage != expected_ranks:
            raise SchemaError("rank coverage is not exact", path=f"{path}.rank_coverage")
        if (
            len(self.core_coverage) != case.expected_rank_count
            or self.core_coverage != tuple(sorted(set(self.core_coverage)))
        ):
            raise SchemaError("core coverage is not exact", path=f"{path}.core_coverage")
        if self.active_routes != tuple(sorted(set(self.active_routes))):
            raise SchemaError("active routes must be canonical", path=f"{path}.active_routes")
        if case.expected_rank_count == 1 and self.active_routes:
            raise SchemaError("LOCAL mode cannot report transport", path=f"{path}.active_routes")
        if self.state_completion != case.expected_state_owners:
            raise SchemaError("state completion drifted", path=f"{path}.state_completion")
        self.residual.validate(f"{path}.residual")
        if not self.residual.is_zero:
            raise SchemaError("runtime residual must be zero", path=f"{path}.residual")


@dataclass(frozen=True, slots=True)
class FlexibleMeshRuntimeEvidence:
    schema_version: str
    id: str
    case: FlexibleMeshRuntimeCase
    compilation_id: str
    linked_source_ref: str
    manifest_id: str
    manifest_digest: str
    artifact_sha256: str
    program_io_digest: str
    resolver_digest: str
    npusim_exit_code: int
    capacity: FlexibleMeshArtifactCapacityEvidence
    markers: tuple[FlexibleMeshRuntimeMarker, ...]
    stages: tuple[tuple[FlexibleMeshRuntimeStage, FlexibleMeshRuntimeStageStatus], ...]
    timing_execution: bool
    functional_execution: bool

    @property
    def runtime_verified(self) -> bool:
        try:
            if self.schema_version != FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION:
                return False
            self.case.validate("flexible_mesh_runtime_evidence.case")
            for name in ("compilation_id", "linked_source_ref", "manifest_id"):
                validate_nonempty(
                    getattr(self, name),
                    f"flexible_mesh_runtime_evidence.{name}",
                )
            for name in (
                "manifest_digest",
                "artifact_sha256",
                "program_io_digest",
                "resolver_digest",
            ):
                _digest(
                    getattr(self, name),
                    f"flexible_mesh_runtime_evidence.{name}",
                )
            self.capacity.validate(
                self.case,
                "flexible_mesh_runtime_evidence.capacity",
            )
            if (
                self.npusim_exit_code != 0
                or len(self.markers) != self.case.repeat_count
                or self.timing_execution is not True
                or self.functional_execution is not False
            ):
                return False
            for marker in self.markers:
                marker.validate(self.case)
                if (
                    marker.manifest_digest != self.manifest_digest
                    or marker.program_io_digest != self.program_io_digest
                ):
                    return False
            expected_stages = tuple(
                (
                    stage,
                    (
                        FlexibleMeshRuntimeStageStatus.VERIFIED
                        if stage is not FlexibleMeshRuntimeStage.REPEATABILITY
                        or self.case.repeat_count == 2
                        else FlexibleMeshRuntimeStageStatus.NOT_MEASURED
                    ),
                )
                for stage in FlexibleMeshRuntimeStage
            )
            if self.stages != expected_stages:
                return False
            expected_id = stable_artifact_id(
                "flexible_mesh_runtime_evidence",
                self._semantic_key(),
                schema_version=FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION,
            )
            return self.id == expected_id
        except (SchemaError, AttributeError, TypeError, ValueError):
            return False

    @property
    def repeatability_verified(self) -> bool:
        return (
            self.runtime_verified
            and self.case.repeat_count == 2
            and len(self.markers) == 2
            and self.markers[0] == self.markers[1]
        )

    @classmethod
    def create(cls, **semantic: object) -> "FlexibleMeshRuntimeEvidence":
        result = cls(
            schema_version=FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_runtime_evidence",
                semantic,
                schema_version=FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(self, path: str = "flexible_mesh_runtime_evidence") -> None:
        if self.schema_version != FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.case.validate(f"{path}.case")
        for name in ("compilation_id", "linked_source_ref", "manifest_id"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in (
            "manifest_digest",
            "artifact_sha256",
            "program_io_digest",
            "resolver_digest",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        if self.npusim_exit_code != 0:
            raise SchemaError("runtime evidence requires exit code zero", path=f"{path}.npusim_exit_code")
        self.capacity.validate(self.case, f"{path}.capacity")
        if len(self.markers) != self.case.repeat_count:
            raise SchemaError("marker count must equal repeat count", path=f"{path}.markers")
        for index, marker in enumerate(self.markers):
            marker.validate(self.case, f"{path}.markers[{index}]")
            if (
                marker.manifest_digest != self.manifest_digest
                or marker.program_io_digest != self.program_io_digest
            ):
                raise SchemaError("marker provenance drifted", path=f"{path}.markers[{index}]")
        expected_stages = tuple(
            (
                stage,
                (
                    FlexibleMeshRuntimeStageStatus.VERIFIED
                    if stage is not FlexibleMeshRuntimeStage.REPEATABILITY
                    or self.case.repeat_count == 2
                    else FlexibleMeshRuntimeStageStatus.NOT_MEASURED
                ),
            )
            for stage in FlexibleMeshRuntimeStage
        )
        if self.stages != expected_stages:
            raise SchemaError("stage status is not exact", path=f"{path}.stages")
        if self.timing_execution is not True or self.functional_execution is not False:
            raise SchemaError("runtime v1 is timing-only", path=path)
        if not self.runtime_verified:
            raise SchemaError("runtime evidence is not verified", path=path)
        if self.case.repeat_count == 2 and not self.repeatability_verified:
            raise SchemaError("repeatability evidence changed", path=f"{path}.markers")
        expected_id = stable_artifact_id(
            "flexible_mesh_runtime_evidence",
            self._semantic_key(),
            schema_version=FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable runtime evidence id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMeshRuntimeFailure:
    case_id: str
    failed_stage: FlexibleMeshRuntimeStage
    reason: str
    completed_stages: tuple[FlexibleMeshRuntimeStage, ...]
    runtime_verified: bool = False

    def validate(self, path: str = "flexible_mesh_runtime_failure") -> None:
        validate_nonempty(self.case_id, f"{path}.case_id")
        if type(self.failed_stage) is not FlexibleMeshRuntimeStage:
            raise SchemaError("must name a runtime stage", path=f"{path}.failed_stage")
        validate_nonempty(self.reason, f"{path}.reason")
        prefix = tuple(FlexibleMeshRuntimeStage)[: tuple(FlexibleMeshRuntimeStage).index(self.failed_stage)]
        if self.completed_stages != prefix:
            raise SchemaError("completed stages must be an exact prefix", path=f"{path}.completed_stages")
        if self.runtime_verified is not False:
            raise SchemaError("failure cannot be runtime verified", path=f"{path}.runtime_verified")


__all__ = [
    "FLEXIBLE_MESH_RUNTIME_MARKER_SCHEMA_VERSION",
    "FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION",
    "FlexibleMeshArtifactCapacityEvidence",
    "FlexibleMeshRuntimeBaseline",
    "FlexibleMeshRuntimeCase",
    "FlexibleMeshRuntimeEvidence",
    "FlexibleMeshRuntimeFailure",
    "FlexibleMeshRuntimeMarker",
    "FlexibleMeshRuntimeResidual",
    "FlexibleMeshRuntimeStage",
    "FlexibleMeshRuntimeStageStatus",
]
