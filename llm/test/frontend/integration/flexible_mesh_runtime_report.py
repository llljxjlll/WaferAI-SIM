"""Fail-closed capability report skeleton for flexible rectangular Meshes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import stable_artifact_id

from flexible_mesh_runtime_provider import PreparedFlexibleMeshCase


FLEXIBLE_MESH_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_runtime_report/v1alpha1"
)


class FlexibleMeshCapabilityStatus(str, Enum):
    VERIFIED = "verified"
    NOT_MEASURED = "not_measured"
    NOT_APPLICABLE = "not_applicable"
    OUT_OF_SCOPE = "out_of_scope"


_COMPLETION_NAMES = (
    "mesh_foundation_complete",
    "mesh_runtime_complete",
    "dense_workload_complete",
    "wang_scale_complete",
    "meshslice_rect_complete",
    "flexible_mesh_dense_complete",
)

_F7_CAPABILITIES = (
    "dense_training",
    "moe_flexible_mesh",
    "functional_execution",
    "subrectangle_placement",
    "cross_die_barrier_timing",
    "plan_template_reuse",
    "nonrectangular_or_fault_mesh",
)


@dataclass(frozen=True, slots=True)
class FlexibleMeshCaseCapability:
    case_name: str
    rows: int
    columns: int
    preparation_digest: str
    meshslice_mode: str
    foundation_fixture: FlexibleMeshCapabilityStatus
    runtime_execution: FlexibleMeshCapabilityStatus
    meshslice_execution: FlexibleMeshCapabilityStatus

    def validate(self, path: str) -> None:
        if self.case_name != f"mesh_{self.rows}x{self.columns}":
            raise SchemaError("case identity drifted", path=f"{path}.case_name")
        _validate_digest(self.preparation_digest, f"{path}.preparation_digest")
        if self.foundation_fixture is not FlexibleMeshCapabilityStatus.VERIFIED:
            raise SchemaError("prepared fixture must remain verified", path=path)
        if self.runtime_execution is not FlexibleMeshCapabilityStatus.NOT_MEASURED:
            raise SchemaError(
                "preparation report cannot claim runtime execution",
                path=f"{path}.runtime_execution",
            )
        expected_mode = (
            "local" if self.rows == self.columns == 1 else
            "row_only" if self.rows == 1 else
            "column_only" if self.columns == 1 else
            "full_2d"
        )
        if self.meshslice_mode != expected_mode:
            raise SchemaError("MeshSlice mode drifted", path=f"{path}.meshslice_mode")
        if self.meshslice_execution is not FlexibleMeshCapabilityStatus.NOT_MEASURED:
            raise SchemaError("MeshSlice shape eligibility drifted", path=path)


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("invalid digest", path=path)


@dataclass(frozen=True, slots=True)
class FlexibleMeshRuntimeReport:
    id: str
    schema_version: str
    measurement_scope: str
    timing_execution: bool
    functional_execution: bool
    cases: tuple[FlexibleMeshCaseCapability, ...]
    completion_states: tuple[tuple[str, FlexibleMeshCapabilityStatus], ...]
    f7_capabilities: tuple[tuple[str, FlexibleMeshCapabilityStatus], ...]

    @classmethod
    def create(
        cls, prepared: tuple[PreparedFlexibleMeshCase, ...]
    ) -> "FlexibleMeshRuntimeReport":
        rows = tuple(
            FlexibleMeshCaseCapability(
                case_name=item.case.name,
                rows=item.case.mesh.rows,
                columns=item.case.mesh.columns,
                preparation_digest=item.digest,
                meshslice_mode=item.case.meshslice_mode.value,
                foundation_fixture=FlexibleMeshCapabilityStatus.VERIFIED,
                runtime_execution=FlexibleMeshCapabilityStatus.NOT_MEASURED,
                meshslice_execution=FlexibleMeshCapabilityStatus.NOT_MEASURED,
            )
            for item in prepared
        )
        semantic = {
            "schema_version": FLEXIBLE_MESH_RUNTIME_REPORT_SCHEMA_VERSION,
            "measurement_scope": "fixture_foundation_only",
            "timing_execution": True,
            "functional_execution": False,
            "cases": rows,
            "completion_states": tuple(
                (name, FlexibleMeshCapabilityStatus.NOT_MEASURED)
                for name in _COMPLETION_NAMES
            ),
            "f7_capabilities": tuple(
                (name, FlexibleMeshCapabilityStatus.OUT_OF_SCOPE)
                for name in _F7_CAPABILITIES
            ),
        }
        result = cls(
            id=stable_artifact_id(
                "flexible_mesh_runtime_report",
                semantic,
                schema_version=FLEXIBLE_MESH_RUNTIME_REPORT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def validate(self, path: str = "flexible_mesh_runtime_report") -> None:
        if self.schema_version != FLEXIBLE_MESH_RUNTIME_REPORT_SCHEMA_VERSION:
            raise SchemaError("schema version drifted", path=f"{path}.schema_version")
        if (
            self.measurement_scope != "fixture_foundation_only"
            or self.timing_execution is not True
            or self.functional_execution is not False
        ):
            raise SchemaError("v1alpha1 scope drifted", path=path)
        if type(self.cases) is not tuple or not self.cases:
            raise SchemaError("requires at least one prepared case", path=f"{path}.cases")
        for index, row in enumerate(self.cases):
            row.validate(f"{path}.cases[{index}]")
        names = tuple(row.case_name for row in self.cases)
        if len(set(names)) != len(names):
            raise SchemaError("case rows must be unique", path=f"{path}.cases")
        expected_completion = tuple(
            (name, FlexibleMeshCapabilityStatus.NOT_MEASURED)
            for name in _COMPLETION_NAMES
        )
        if self.completion_states != expected_completion:
            raise SchemaError(
                "fixture-only report cannot claim a plan completion state",
                path=f"{path}.completion_states",
            )
        expected_f7 = tuple(
            (name, FlexibleMeshCapabilityStatus.OUT_OF_SCOPE)
            for name in _F7_CAPABILITIES
        )
        if self.f7_capabilities != expected_f7:
            raise SchemaError(
                "F7 capabilities must remain explicitly out of scope",
                path=f"{path}.f7_capabilities",
            )
        semantic = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }
        expected_id = stable_artifact_id(
            "flexible_mesh_runtime_report",
            semantic,
            schema_version=FLEXIBLE_MESH_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("report id does not match its contents", path=f"{path}.id")


__all__ = [
    "FLEXIBLE_MESH_RUNTIME_REPORT_SCHEMA_VERSION",
    "FlexibleMeshCapabilityStatus",
    "FlexibleMeshCaseCapability",
    "FlexibleMeshRuntimeReport",
]
