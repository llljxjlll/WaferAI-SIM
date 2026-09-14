"""Versioned P5 schema/preflight coverage for the 1..10 rectangular envelope."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .workload_materialization import WorkloadMaterializationStatus
from .workload_run import (
    WorkloadCapabilityLevel,
    WorkloadFamily,
    WorkloadRunRequest,
)


WORKLOAD_SHAPE_CASE_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.workload_shape_case_report/v1alpha1"
)
WORKLOAD_SHAPE_MATRIX_SCHEMA_VERSION = (
    "wafer_frontend.workload_shape_matrix/v1alpha1"
)
P5_MAX_MESH_ROWS = 10
P5_MAX_MESH_COLUMNS = 10


class WorkloadShapeMappingMode(str, Enum):
    MESH_SCALED_ALL_DIES = "mesh_scaled_all_dies"
    FIXED_FOUR_RANK_IDLE_DIES = "fixed_four_rank_idle_dies"


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class WorkloadLogicalWork:
    layer_count: int
    workload_step_count: int
    token_count: int
    logical_rank_count: int
    operation_count: int
    tensor_value_count: int
    tensor_value_bytes: int
    transport_request_count: int
    transport_payload_bytes: int
    memory_state_count: int
    memory_reserved_bytes: int

    def validate(self, path: str = "logical_work") -> None:
        positive = {
            "layer_count",
            "workload_step_count",
            "token_count",
            "logical_rank_count",
            "operation_count",
            "tensor_value_count",
            "memory_state_count",
            "memory_reserved_bytes",
        }
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            validate_uint64(value, f"{path}.{name}")
            if name in positive and value == 0:
                raise SchemaError("must be greater than zero", path=f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class WorkloadShapeCaseReport:
    schema_version: str
    producer_pass: str
    id: str
    request: WorkloadRunRequest
    request_digest: str
    mapping_mode: WorkloadShapeMappingMode
    active_die_ids: tuple[int, ...]
    idle_die_ids: tuple[int, ...]
    logical_work: WorkloadLogicalWork
    materialization_id: str
    materialization_digest: str
    placement_digest: str
    logical_graph_digest: str
    transport_requests_digest: str
    memory_plan_digest: str
    materialization_status: WorkloadMaterializationStatus
    unsupported_requirements: tuple[str, ...]
    runtime_status: WorkloadCapabilityLevel
    is_large_mesh_canary: bool

    @classmethod
    def create(
        cls,
        *,
        request: WorkloadRunRequest,
        mapping_mode: WorkloadShapeMappingMode,
        active_die_ids: tuple[int, ...],
        idle_die_ids: tuple[int, ...],
        logical_work: WorkloadLogicalWork,
        materialization_id: str,
        materialization_digest: str,
        placement_digest: str,
        logical_graph_digest: str,
        transport_requests_digest: str,
        memory_plan_digest: str,
        materialization_status: WorkloadMaterializationStatus,
        unsupported_requirements: tuple[str, ...],
        is_large_mesh_canary: bool,
    ) -> "WorkloadShapeCaseReport":
        key = {
            "request": request,
            "request_digest": request.digest,
            "mapping_mode": mapping_mode,
            "active_die_ids": active_die_ids,
            "idle_die_ids": idle_die_ids,
            "logical_work": logical_work,
            "materialization_id": materialization_id,
            "materialization_digest": materialization_digest,
            "placement_digest": placement_digest,
            "logical_graph_digest": logical_graph_digest,
            "transport_requests_digest": transport_requests_digest,
            "memory_plan_digest": memory_plan_digest,
            "materialization_status": materialization_status,
            "unsupported_requirements": unsupported_requirements,
            "runtime_status": WorkloadCapabilityLevel.NOT_MEASURED,
            "is_large_mesh_canary": is_large_mesh_canary,
        }
        result = cls(
            schema_version=WORKLOAD_SHAPE_CASE_REPORT_SCHEMA_VERSION,
            producer_pass="workload_shape_matrix_preflight",
            id=stable_artifact_id(
                "workload_shape_case_report",
                key,
                schema_version=WORKLOAD_SHAPE_CASE_REPORT_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    @property
    def matrix_key(self) -> tuple[int, int, WorkloadFamily]:
        return self.request.mesh.rows, self.request.mesh.columns, self.request.family

    def validate(self, path: str = "workload_shape_case_report") -> None:
        if self.schema_version != WORKLOAD_SHAPE_CASE_REPORT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "workload_shape_matrix_preflight":
            raise SchemaError("unexpected producer", path=f"{path}.producer_pass")
        if type(self.request) is not WorkloadRunRequest:
            raise SchemaError("must be a WorkloadRunRequest", path=f"{path}.request")
        self.request.validate(f"{path}.request")
        if self.request_digest != self.request.digest:
            raise SchemaError("does not match request", path=f"{path}.request_digest")
        if type(self.mapping_mode) is not WorkloadShapeMappingMode:
            raise SchemaError("must be a mapping mode", path=f"{path}.mapping_mode")
        rank_count = self.request.mesh.rank_count
        expected_dies = set(range(rank_count))
        for name, dies in (
            ("active_die_ids", self.active_die_ids),
            ("idle_die_ids", self.idle_die_ids),
        ):
            if type(dies) is not tuple or len(dies) != len(set(dies)):
                raise SchemaError("must contain unique Die IDs", path=f"{path}.{name}")
            if tuple(sorted(dies)) != dies or not set(dies).issubset(expected_dies):
                raise SchemaError("must be sorted and within the mesh", path=f"{path}.{name}")
        if set(self.active_die_ids).intersection(self.idle_die_ids) or (
            set(self.active_die_ids).union(self.idle_die_ids) != expected_dies
        ):
            raise SchemaError("active and idle Dies must partition the mesh", path=path)
        if len(self.active_die_ids) != self.request.parallel.logical_rank_count:
            raise SchemaError("active Dies must match logical ranks", path=f"{path}.active_die_ids")
        if tuple(sorted(self.request.parallel.active_die_ids)) != self.active_die_ids:
            raise SchemaError("does not match request mapping", path=f"{path}.active_die_ids")
        if self.mapping_mode is WorkloadShapeMappingMode.MESH_SCALED_ALL_DIES:
            if self.active_die_ids != tuple(range(rank_count)) or self.idle_die_ids:
                raise SchemaError("all-die mode must activate the full mesh", path=path)
        elif len(self.active_die_ids) != 4 or not self.idle_die_ids:
            raise SchemaError("idle-die mode requires four active ranks", path=path)
        if type(self.logical_work) is not WorkloadLogicalWork:
            raise SchemaError("must be WorkloadLogicalWork", path=f"{path}.logical_work")
        self.logical_work.validate(f"{path}.logical_work")
        if self.logical_work.layer_count != self.request.model.num_layers:
            raise SchemaError("layer count disagrees with request", path=f"{path}.logical_work")
        if self.logical_work.logical_rank_count != len(self.active_die_ids):
            raise SchemaError("rank count disagrees with mapping", path=f"{path}.logical_work")
        validate_nonempty(self.materialization_id, f"{path}.materialization_id")
        for name in (
            "materialization_digest",
            "placement_digest",
            "logical_graph_digest",
            "transport_requests_digest",
            "memory_plan_digest",
        ):
            _validate_digest(getattr(self, name), f"{path}.{name}")
        if self.materialization_status is not WorkloadMaterializationStatus.UNSUPPORTED:
            raise SchemaError(
                "schema preflight must not claim materialized runtime support",
                path=f"{path}.materialization_status",
            )
        if type(self.unsupported_requirements) is not tuple or (
            "family.runtime" not in self.unsupported_requirements
        ):
            raise SchemaError(
                "must retain missing runtime capability",
                path=f"{path}.unsupported_requirements",
            )
        if self.runtime_status is not WorkloadCapabilityLevel.NOT_MEASURED:
            raise SchemaError("must remain not_measured", path=f"{path}.runtime_status")
        if type(self.is_large_mesh_canary) is not bool or self.is_large_mesh_canary != (
            self.request.mesh.rows == P5_MAX_MESH_ROWS
            and self.request.mesh.columns == P5_MAX_MESH_COLUMNS
        ):
            raise SchemaError(
                "does not match the canonical 10x10 canary",
                path=f"{path}.is_large_mesh_canary",
            )
        expected_id = stable_artifact_id(
            "workload_shape_case_report",
            self._key(),
            schema_version=WORKLOAD_SHAPE_CASE_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable report id", path=f"{path}.id")


_FAMILY_RANK = {family: index for index, family in enumerate(WorkloadFamily)}


@dataclass(frozen=True, slots=True)
class WorkloadShapeMatrix:
    schema_version: str
    producer_pass: str
    id: str
    max_mesh_rows: int
    max_mesh_columns: int
    reports: tuple[WorkloadShapeCaseReport, ...]
    large_mesh_canary_case_ids: tuple[str, ...]
    runtime_status: WorkloadCapabilityLevel

    @classmethod
    def create(
        cls, *, reports: tuple[WorkloadShapeCaseReport, ...]
    ) -> "WorkloadShapeMatrix":
        canonical = tuple(
            sorted(
                reports,
                key=lambda item: (
                    item.request.mesh.rows,
                    item.request.mesh.columns,
                    _FAMILY_RANK[item.request.family],
                ),
            )
        )
        canaries = tuple(
            item.request.case_id for item in canonical if item.is_large_mesh_canary
        )
        key = {
            "max_mesh_rows": P5_MAX_MESH_ROWS,
            "max_mesh_columns": P5_MAX_MESH_COLUMNS,
            "reports": canonical,
            "large_mesh_canary_case_ids": canaries,
            "runtime_status": WorkloadCapabilityLevel.NOT_MEASURED,
        }
        result = cls(
            schema_version=WORKLOAD_SHAPE_MATRIX_SCHEMA_VERSION,
            producer_pass="workload_shape_matrix_preflight",
            id=stable_artifact_id(
                "workload_shape_matrix",
                key,
                schema_version=WORKLOAD_SHAPE_MATRIX_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "workload_shape_matrix") -> None:
        if self.schema_version != WORKLOAD_SHAPE_MATRIX_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "workload_shape_matrix_preflight":
            raise SchemaError("unexpected producer", path=f"{path}.producer_pass")
        if (self.max_mesh_rows, self.max_mesh_columns) != (
            P5_MAX_MESH_ROWS,
            P5_MAX_MESH_COLUMNS,
        ):
            raise SchemaError("must describe the exact 1..10 envelope", path=path)
        if type(self.reports) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.reports")
        actual_keys: list[tuple[int, int, WorkloadFamily]] = []
        report_ids: list[str] = []
        case_ids: list[str] = []
        request_digests: list[str] = []
        for index, report in enumerate(self.reports):
            if type(report) is not WorkloadShapeCaseReport:
                raise SchemaError("must be a case report", path=f"{path}.reports[{index}]")
            report.validate(f"{path}.reports[{index}]")
            actual_keys.append(report.matrix_key)
            report_ids.append(report.id)
            case_ids.append(report.request.case_id)
            request_digests.append(report.request_digest)
        expected_keys = [
            (row, column, family)
            for row in range(1, P5_MAX_MESH_ROWS + 1)
            for column in range(1, P5_MAX_MESH_COLUMNS + 1)
            for family in WorkloadFamily
        ]
        if actual_keys != expected_keys:
            raise SchemaError(
                "reports must cover each shape/family exactly once",
                path=f"{path}.reports",
            )
        for name, values in (
            ("report id", report_ids),
            ("case id", case_ids),
            ("request digest", request_digests),
        ):
            if len(values) != len(set(values)):
                raise SchemaError(f"duplicate {name}", path=f"{path}.reports")
        expected_canaries = tuple(
            report.request.case_id
            for report in self.reports
            if report.is_large_mesh_canary
        )
        if self.large_mesh_canary_case_ids != expected_canaries or len(expected_canaries) != 4:
            raise SchemaError(
                "must reference all four 10x10 canaries",
                path=f"{path}.large_mesh_canary_case_ids",
            )
        if self.runtime_status is not WorkloadCapabilityLevel.NOT_MEASURED:
            raise SchemaError("must remain not_measured", path=f"{path}.runtime_status")
        if not {item.mapping_mode for item in self.reports} == set(WorkloadShapeMappingMode):
            raise SchemaError("must contain all-die and idle-die mappings", path=f"{path}.reports")
        expected_id = stable_artifact_id(
            "workload_shape_matrix",
            self._key(),
            schema_version=WORKLOAD_SHAPE_MATRIX_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable matrix id", path=f"{path}.id")


__all__ = [
    "P5_MAX_MESH_COLUMNS",
    "P5_MAX_MESH_ROWS",
    "WORKLOAD_SHAPE_CASE_REPORT_SCHEMA_VERSION",
    "WORKLOAD_SHAPE_MATRIX_SCHEMA_VERSION",
    "WorkloadLogicalWork",
    "WorkloadShapeCaseReport",
    "WorkloadShapeMappingMode",
    "WorkloadShapeMatrix",
]
