"""Deterministic case matrices for the flexible rectangular-Mesh suite.

This module contains no compiler or simulator policy. It is a shared case
vocabulary that unit, lower/link and runtime suites can consume without
freezing one golden artifact per Mesh shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_workload import (
    FlexibleMeshSliceMode,
    FlexibleMeshSliceSpec,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec


@dataclass(frozen=True, slots=True)
class FlexibleMeshCase:
    """One named complete-rectangle case and its coverage categories."""

    name: str
    mesh: RectMeshSpec
    coverage: tuple[str, ...]

    @classmethod
    def create(
        cls, rows: int, columns: int, *coverage: str
    ) -> "FlexibleMeshCase":
        result = cls(
            name=f"mesh_{rows}x{columns}",
            mesh=RectMeshSpec(rows=rows, columns=columns),
            coverage=tuple(sorted(coverage)),
        )
        result.validate()
        return result

    def validate(self, path: str = "flexible_mesh_case") -> None:
        if type(self.mesh) is not RectMeshSpec:
            raise SchemaError("must carry a RectMeshSpec", path=f"{path}.mesh")
        self.mesh.validate(f"{path}.mesh")
        if self.name != f"mesh_{self.mesh.rows}x{self.mesh.columns}":
            raise SchemaError(
                "name must be derived from rows and columns", path=f"{path}.name"
            )
        if (
            type(self.coverage) is not tuple
            or not self.coverage
            or self.coverage != tuple(sorted(set(self.coverage)))
            or any(type(item) is not str or not item for item in self.coverage)
        ):
            raise SchemaError(
                "coverage must be a non-empty sorted unique string tuple",
                path=f"{path}.coverage",
            )

    @property
    def is_linear(self) -> bool:
        return self.mesh.rows == 1 or self.mesh.columns == 1

    @property
    def meshslice_eligible_shape(self) -> bool:
        """Return shape eligibility, not a support or performance claim."""

        return True

    @property
    def meshslice_mode(self) -> FlexibleMeshSliceMode:
        return FlexibleMeshSliceSpec.mode_for(self.mesh)


FLEXIBLE_MESH_REPRESENTATIVE_CASES = (
    FlexibleMeshCase.create(1, 1, "single_die"),
    FlexibleMeshCase.create(1, 10, "linear", "maximum_width"),
    FlexibleMeshCase.create(10, 1, "linear", "maximum_height"),
    FlexibleMeshCase.create(2, 3, "non_power_of_two", "rectangular"),
    FlexibleMeshCase.create(3, 2, "non_power_of_two", "rectangular", "transpose"),
    FlexibleMeshCase.create(3, 3, "odd_by_odd", "square"),
    FlexibleMeshCase.create(10, 10, "maximum_area", "square"),
)


def all_flexible_mesh_cases() -> tuple[FlexibleMeshCase, ...]:
    """Return the complete 10x10 envelope without checked-in goldens."""

    return tuple(
        FlexibleMeshCase.create(rows, columns, "envelope")
        for rows in range(1, 11)
        for columns in range(1, 11)
    )


def validate_flexible_mesh_case_matrix(
    cases: tuple[FlexibleMeshCase, ...],
    *,
    path: str = "flexible_mesh_cases",
) -> None:
    if type(cases) is not tuple or not cases:
        raise SchemaError("must be a non-empty tuple", path=path)
    for index, case in enumerate(cases):
        if type(case) is not FlexibleMeshCase:
            raise SchemaError("must be a FlexibleMeshCase", path=f"{path}[{index}]")
        case.validate(f"{path}[{index}]")
    keys = tuple((case.mesh.rows, case.mesh.columns) for case in cases)
    if len(set(keys)) != len(keys):
        raise SchemaError("Mesh shapes must be unique", path=path)


validate_flexible_mesh_case_matrix(FLEXIBLE_MESH_REPRESENTATIVE_CASES)


__all__ = [
    "all_flexible_mesh_cases",
    "FLEXIBLE_MESH_REPRESENTATIVE_CASES",
    "FlexibleMeshCase",
    "validate_flexible_mesh_case_matrix",
]
