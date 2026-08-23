"""Typed parameterization for Swizzle benefit scale experiments."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64
from .experiment import ExperimentSpec, WorkloadMode
from .ir0 import IR0
from .ir1 import IR1


SWIZZLE_SCALE_POINT_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_scale_point/v1alpha1"
)


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be greater than zero", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleScalePoint:
    """One immutable Dense TP workload/mesh scale point."""

    schema_version: str
    id: str
    name: str
    tokens: int
    hidden_size: int
    intermediate_size: int
    tp: int
    mesh_rows: int
    mesh_columns: int
    dtype: DType

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleScalePoint":
        result = cls(
            schema_version=SWIZZLE_SCALE_POINT_SCHEMA_VERSION,
            id=stable_artifact_id(
                "swizzle_scale_point",
                semantic,
                schema_version=SWIZZLE_SCALE_POINT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "name",
                "tokens",
                "hidden_size",
                "intermediate_size",
                "tp",
                "mesh_rows",
                "mesh_columns",
                "dtype",
            )
        }

    def validate(self, path: str = "swizzle_scale_point") -> None:
        if self.schema_version != SWIZZLE_SCALE_POINT_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        validate_nonempty(self.name, f"{path}.name")
        for name in (
            "tokens",
            "hidden_size",
            "intermediate_size",
            "tp",
            "mesh_rows",
            "mesh_columns",
        ):
            _positive(getattr(self, name), f"{path}.{name}")
        if type(self.dtype) is not DType:
            raise SchemaError("must use a typed dtype", path=f"{path}.dtype")
        if self.tp != self.mesh_rows * self.mesh_columns:
            raise SchemaError(
                "tp must equal mesh_rows * mesh_columns", path=f"{path}.tp"
            )
        for name, extent in (
            ("tokens", self.tokens),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
        ):
            if extent % self.tp:
                raise SchemaError(
                    f"{name} must divide exactly by tp",
                    path=f"{path}.{name}",
                )
        expected = stable_artifact_id(
            "swizzle_scale_point",
            self._semantic_key(),
            schema_version=SWIZZLE_SCALE_POINT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        spec: ExperimentSpec,
        source: IR0,
        placed: IR1,
        partitioned: IR1,
        path: str = "swizzle_scale_point",
    ) -> None:
        """Prove ExperimentSpec provenance and physical rectangle placement."""

        self.validate(path)
        spec.validate(f"{path}.spec")
        source.validate(f"{path}.source")
        placed.validate(f"{path}.placed")
        partitioned.validate(f"{path}.partitioned")
        if spec.workload.mode is not WorkloadMode.INFER:
            raise SchemaError(
                "scale points require an inference ExperimentSpec",
                path=f"{path}.spec.workload.mode",
            )
        infer = spec.workload.infer
        if infer is None or infer.profile is None:
            raise SchemaError(
                "scale points require a static inference profile",
                path=f"{path}.spec.workload.infer",
            )
        if (
            spec.model.H,
            spec.model.I,
            spec.model.dtype,
            infer.profile.prefill_tokens,
        ) != (
            self.hidden_size,
            self.intermediate_size,
            self.dtype,
            self.tokens,
        ):
            raise SchemaError(
                "ExperimentSpec model/profile drifted from the scale point",
                path=f"{path}.spec",
            )
        if len(spec.parallel.instances) != 1:
            raise SchemaError(
                "scale points require one Dense TP instance",
                path=f"{path}.spec.parallel.instances",
            )
        instance = spec.parallel.instances[0]
        if instance.tp != self.tp:
            raise SchemaError(
                "ExperimentSpec TP drifted from the scale point",
                path=f"{path}.spec.parallel.instances[0].tp",
            )
        if source.profile.prefill_tokens != self.tokens:
            raise SchemaError(
                "IR0 profile tokens drifted from the ExperimentSpec",
                path=f"{path}.source.profile",
            )
        if placed.source_ir0_id != source.id:
            raise SchemaError(
                "placed IR1 lost exact IR0 provenance",
                path=f"{path}.placed.source_ir0_id",
            )
        if (
            partitioned.producer_pass != "fusion_partition"
            or partitioned.source_ir0_id != source.id
        ):
            raise SchemaError(
                "partitioned IR1 lost exact placement provenance",
                path=f"{path}.partitioned",
            )
        expected_grid = (self.mesh_columns, self.mesh_rows)
        if placed.fabric.die_grid != expected_grid:
            raise SchemaError(
                f"physical fabric must be exact rectangle {expected_grid!r}",
                path=f"{path}.placed.fabric.die_grid",
            )
        if len(placed.groups) != 1:
            raise SchemaError(
                "scale points require one physical TP group",
                path=f"{path}.placed.groups",
            )
        group = placed.groups[0]
        rank_to_die = {
            item.rank: item.die_id for item in group.placements
        }
        if (
            group.logical_shape != (self.tp,)
            or tuple(sorted(rank_to_die)) != tuple(range(self.tp))
            or tuple(rank_to_die[index] for index in range(self.tp))
            != tuple(range(self.tp))
        ):
            raise SchemaError(
                "production placement must map dense TP ranks to the rectangle",
                path=f"{path}.placed.groups[0].placements",
            )
        routes = group.embedding.routes
        expected_pairs = {
            (source_rank, destination_rank)
            for source_rank in range(self.tp)
            for destination_rank in range(self.tp)
            if source_rank != destination_rank
        }
        if {
            (route.source_rank, route.destination_rank) for route in routes
        } != expected_pairs or len(routes) != len(expected_pairs):
            raise SchemaError(
                "group must contain every ordered PairRoute exactly once",
                path=f"{path}.placed.groups[0].embedding.routes",
            )
        for index, route in enumerate(routes):
            route.validate_against(
                placed.fabric,
                rank_to_die,
                f"{path}.placed.groups[0].embedding.routes[{index}]",
            )


__all__ = [
    "SWIZZLE_SCALE_POINT_SCHEMA_VERSION",
    "SwizzleScalePoint",
]
