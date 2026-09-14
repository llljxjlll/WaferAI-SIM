"""Shared production limits for flexible rectangular-Mesh workloads.

The v2 workload carriers use this module for cheap, deterministic preflight.
It deliberately reports encoded artifact bytes as an upper bound: the C++
finalizer remains the authority for the exact on-disk size.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import validate_uint64


FLEXIBLE_MESH_CAPACITY_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_capacity/v1alpha1"
)


@dataclass(frozen=True, slots=True)
class FlexibleMeshCapacityProfile:
    max_ranks: int = 100
    max_actions: int = 80_000
    max_buffers: int = 100_000
    max_symbolic_records: int = 1_048_576
    max_artifact_file_bytes: int = 64 * 1024 * 1024
    max_sessions_per_core_per_wave: int = 3
    max_transport_tags: int = 65_535
    max_state_bytes: int = 64 * 1024 * 1024 * 1024

    def validate(self, path: str = "flexible_mesh_capacity") -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            validate_uint64(value, f"{path}.{name}")
            if value == 0:
                raise SchemaError("must be positive", path=f"{path}.{name}")
        if self.max_ranks > 100:
            raise SchemaError("must not exceed 100", path=f"{path}.max_ranks")
        if self.max_sessions_per_core_per_wave > 3:
            raise SchemaError(
                "must preserve the production session limit of 3",
                path=f"{path}.max_sessions_per_core_per_wave",
            )
        if self.max_symbolic_records > 1_048_576:
            raise SchemaError(
                "must preserve the production record limit",
                path=f"{path}.max_symbolic_records",
            )
        if self.max_artifact_file_bytes > 64 * 1024 * 1024:
            raise SchemaError(
                "must preserve the production file limit",
                path=f"{path}.max_artifact_file_bytes",
            )


@dataclass(frozen=True, slots=True)
class FlexibleMeshCapacityDemand:
    ranks: int
    actions: int
    buffers: int
    symbolic_records: int
    artifact_file_upper_bound_bytes: int
    peak_sessions_per_core_per_wave: int
    transport_tags: int
    state_bytes: int

    def validate(self, path: str = "flexible_mesh_capacity_demand") -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")

    def exceeded_limits(
        self, profile: FlexibleMeshCapacityProfile
    ) -> tuple[str, ...]:
        self.validate()
        profile.validate()
        fields = (
            ("ranks", "max_ranks"),
            ("actions", "max_actions"),
            ("buffers", "max_buffers"),
            ("symbolic_records", "max_symbolic_records"),
            (
                "artifact_file_upper_bound_bytes",
                "max_artifact_file_bytes",
            ),
            (
                "peak_sessions_per_core_per_wave",
                "max_sessions_per_core_per_wave",
            ),
            ("transport_tags", "max_transport_tags"),
            ("state_bytes", "max_state_bytes"),
        )
        return tuple(
            demand_name
            for demand_name, limit_name in fields
            if getattr(self, demand_name) > getattr(profile, limit_name)
        )

    def require_fits(
        self,
        profile: FlexibleMeshCapacityProfile,
        path: str = "flexible_mesh_capacity_demand",
    ) -> None:
        exceeded = self.exceeded_limits(profile)
        if exceeded:
            raise SchemaError(
                "capacity exceeded: " + ", ".join(exceeded),
                path=path,
                code="flexible_mesh_capacity_exceeded",
            )


__all__ = [
    "FLEXIBLE_MESH_CAPACITY_SCHEMA_VERSION",
    "FlexibleMeshCapacityDemand",
    "FlexibleMeshCapacityProfile",
]
