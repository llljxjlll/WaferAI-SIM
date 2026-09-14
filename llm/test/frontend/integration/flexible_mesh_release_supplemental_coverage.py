"""Canonical supplemental release scope and primary-matrix witnesses."""

from __future__ import annotations

from collections.abc import Iterable

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseFamily,
)


REPRESENTATIVE_RELEASE_SHAPES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 10),
    (10, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 10),
    (10, 2),
    (5, 6),
    (6, 5),
    (9, 9),
    (9, 10),
    (10, 9),
    (10, 10),
)

_DENSE_WITNESSES = {
    "compute_only": (1, 1),
    "tp_only": (1, 2),
    "dp_only": (2, 1),
    "dp_x_tp": (2, 2),
}
_MESHSLICE_MODE_WITNESSES = {
    "local": (1, 1),
    "row_only": (1, 10),
    "column_only": (10, 1),
    "full_2d": (2, 3),
}
_MESHSLICE_FAMILIES = (
    FlexibleMeshReleaseFamily.MESHSLICE_AG,
    FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,
    FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,
)


def supplemental_coverage_catalog() -> dict[str, object]:
    """Return declarations only; runtime truth still comes from case evidence."""

    return {
        "representative_shape_allowlist": REPRESENTATIVE_RELEASE_SHAPES,
        "moe": {
            "supplemental_runtime_traces": ("all-local", "hot-empty"),
            "primary_runtime_trace": "balanced",
            "families": ("moe_inference", "moe_train"),
            "functional_correctness": "out_of_scope_timing_only",
        },
        "dense_primary_witnesses": tuple(sorted(_DENSE_WITNESSES.items())),
        "meshslice_primary_witnesses": {
            "execution_modes": tuple(sorted(_MESHSLICE_MODE_WITNESSES.items())),
            "families": tuple(item.value for item in _MESHSLICE_FAMILIES),
            "capacity_boundary_shape": (10, 10),
            "automatic_fallback_families": (
                FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK.value,
                FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK.value,
            ),
            "automatic_fallback_reason": "STRICT_TWO_INPUT_REDUCE_ABI",
            "optimized_auto_performance_selection": "out_of_scope",
        },
    }


def audit_primary_supplemental_coverage(
    cases: Iterable[FlexibleMeshReleaseCase],
) -> None:
    """Fail closed unless primary cases provide every declared witness."""

    items = tuple(cases)
    keys = {
        (case.family, case.mesh.rows, case.mesh.columns)
        for case in items
    }
    if len(keys) != len(items):
        raise SchemaError("primary cases are duplicated", path="primary_cases")

    missing: list[str] = []
    for name, (rows, columns) in _DENSE_WITNESSES.items():
        if (FlexibleMeshReleaseFamily.DENSE_TRAIN, rows, columns) not in keys:
            missing.append(f"dense:{name}:{rows}x{columns}")
    for mode, (rows, columns) in _MESHSLICE_MODE_WITNESSES.items():
        for family in _MESHSLICE_FAMILIES:
            if (family, rows, columns) not in keys:
                missing.append(f"meshslice:{mode}:{family.value}:{rows}x{columns}")
    for family in _MESHSLICE_FAMILIES:
        if (family, 10, 10) not in keys:
            missing.append(f"meshslice:capacity:{family.value}:10x10")
    for family in (
        FlexibleMeshReleaseFamily.MOE_INFERENCE,
        FlexibleMeshReleaseFamily.MOE_TRAIN,
    ):
        for rows, columns in REPRESENTATIVE_RELEASE_SHAPES:
            if (family, rows, columns) not in keys:
                missing.append(f"moe:balanced:{family.value}:{rows}x{columns}")
    if missing:
        raise SchemaError(
            "primary supplemental witnesses are missing: " + ",".join(missing),
            path="primary_cases",
        )


__all__ = [
    "REPRESENTATIVE_RELEASE_SHAPES",
    "audit_primary_supplemental_coverage",
    "supplemental_coverage_catalog",
]
