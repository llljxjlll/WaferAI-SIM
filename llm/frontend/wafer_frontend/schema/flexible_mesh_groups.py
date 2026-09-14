"""Canonical physical and logical groups for a rectangular Die Mesh."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty
from .rect_mesh import RectMeshSpec


FLEXIBLE_MESH_GROUP_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_group/v1alpha1"
)
FLEXIBLE_MESH_GROUP_REGISTRY_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_group_registry/v1alpha1"
)


class FlexibleMeshGroupKind(str, Enum):
    FULL = "full"
    ROW = "row"
    COLUMN = "column"
    DP = "dp"
    TP = "tp"
    EP = "ep"


@dataclass(frozen=True, slots=True)
class FlexibleMeshAxisMapping:
    dp_degree: int
    tp_degree: int
    ep_degree: int = 1
    pp_degree: int = 1
    transposed: bool = False

    @classmethod
    def dense(cls, mesh: RectMeshSpec, *, transposed: bool = False):
        mesh.validate()
        return cls(
            dp_degree=mesh.columns if transposed else mesh.rows,
            tp_degree=mesh.rows if transposed else mesh.columns,
            transposed=transposed,
        )

    @classmethod
    def moe(cls, mesh: RectMeshSpec):
        mesh.validate()
        return cls(dp_degree=1, tp_degree=1, ep_degree=mesh.rank_count)

    def validate(
        self,
        mesh: RectMeshSpec | str | None = None,
        *,
        require_dense: bool = False,
        require_moe: bool = False,
        path: str = "axis_mapping",
    ) -> None:
        if type(mesh) is str:
            path = mesh
            mesh = None
        for name in ("dp_degree", "tp_degree", "ep_degree", "pp_degree"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise SchemaError("must be a positive int", path=f"{path}.{name}")
        if type(self.transposed) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.transposed")
        if self.pp_degree != 1:
            raise SchemaError("v1 requires PP=1", path=f"{path}.pp_degree")
        if mesh is None:
            if require_dense or require_moe:
                raise SchemaError("Mesh context is required", path=path)
            return
        if type(mesh) is not RectMeshSpec:
            raise SchemaError("must be a RectMeshSpec", path=f"{path}.mesh")
        mesh.validate(f"{path}.mesh")
        if require_dense:
            expected = FlexibleMeshAxisMapping.dense(
                mesh, transposed=self.transposed
            )
            if self != expected:
                raise SchemaError(
                    "dense mapping must be the canonical DPxTP rectangle",
                    path=path,
                    code="invalid_axis_mapping",
                )
        if require_moe and self != FlexibleMeshAxisMapping.moe(mesh):
            raise SchemaError(
                "MoE v1 requires EP=R and DP=TP=PP=1",
                path=path,
                code="invalid_axis_mapping",
            )


@dataclass(frozen=True, slots=True)
class FlexibleMeshGroup:
    schema_version: str
    id: str
    kind: FlexibleMeshGroupKind
    index: int
    ranks: tuple[int, ...]

    @classmethod
    def create(
        cls,
        *,
        mesh: RectMeshSpec,
        kind: FlexibleMeshGroupKind,
        index: int,
        ranks: tuple[int, ...],
    ) -> "FlexibleMeshGroup":
        semantic = {
            "mesh_digest": mesh.digest,
            "kind": kind,
            "index": index,
            "ranks": ranks,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_GROUP_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_group",
                semantic,
                schema_version=FLEXIBLE_MESH_GROUP_SCHEMA_VERSION,
            ),
            kind=kind,
            index=index,
            ranks=ranks,
        )
        result.validate(mesh)
        return result

    def validate(
        self,
        mesh: RectMeshSpec | str | None = None,
        path: str = "flexible_mesh_group",
    ) -> None:
        if type(mesh) is str:
            path = mesh
            mesh = None
        if self.schema_version != FLEXIBLE_MESH_GROUP_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.id, f"{path}.id")
        if type(self.kind) is not FlexibleMeshGroupKind:
            raise SchemaError("must be a group kind", path=f"{path}.kind")
        if type(self.index) is not int or self.index < 0:
            raise SchemaError("must be non-negative", path=f"{path}.index")
        if (
            type(self.ranks) is not tuple
            or not self.ranks
            or len(set(self.ranks)) != len(self.ranks)
            or any(type(rank) is not int or rank < 0 for rank in self.ranks)
        ):
            raise SchemaError(
                "must contain unique non-negative ranks", path=f"{path}.ranks"
            )
        if mesh is None:
            return
        if type(mesh) is not RectMeshSpec:
            raise SchemaError("must be a RectMeshSpec", path=f"{path}.mesh")
        mesh.validate(f"{path}.mesh")
        if any(rank >= mesh.rank_count for rank in self.ranks):
            raise SchemaError(
                "rank lies outside the Mesh", path=f"{path}.ranks"
            )
        semantic = {
            "mesh_digest": mesh.digest,
            "kind": self.kind,
            "index": self.index,
            "ranks": self.ranks,
        }
        expected = stable_artifact_id(
            "flexible_mesh_group",
            semantic,
            schema_version=FLEXIBLE_MESH_GROUP_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable group id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMeshGroupRegistry:
    schema_version: str
    mesh: RectMeshSpec
    axis_mapping: FlexibleMeshAxisMapping
    groups: tuple[FlexibleMeshGroup, ...]

    @classmethod
    def create(
        cls, mesh: RectMeshSpec, axis_mapping: FlexibleMeshAxisMapping
    ) -> "FlexibleMeshGroupRegistry":
        mesh.validate()
        # Workload-specific validation is owned by the envelope.  The registry
        # only requires a structurally valid PP1 mapping.
        axis_mapping.validate(mesh)
        groups: list[FlexibleMeshGroup] = []

        def add(kind: FlexibleMeshGroupKind, index: int, ranks: tuple[int, ...]):
            groups.append(
                FlexibleMeshGroup.create(
                    mesh=mesh, kind=kind, index=index, ranks=ranks
                )
            )

        add(FlexibleMeshGroupKind.FULL, 0, tuple(range(mesh.rank_count)))
        for index, ranks in enumerate(mesh.row_rank_orders):
            add(FlexibleMeshGroupKind.ROW, index, ranks)
        for index, ranks in enumerate(mesh.column_rank_orders):
            add(FlexibleMeshGroupKind.COLUMN, index, ranks)

        if axis_mapping.ep_degree == mesh.rank_count:
            add(FlexibleMeshGroupKind.EP, 0, tuple(range(mesh.rank_count)))
        elif axis_mapping.dp_degree * axis_mapping.tp_degree == mesh.rank_count:
            tp_groups = (
                mesh.column_rank_orders
                if axis_mapping.transposed
                else mesh.row_rank_orders
            )
            dp_groups = (
                mesh.row_rank_orders
                if axis_mapping.transposed
                else mesh.column_rank_orders
            )
            for index, ranks in enumerate(tp_groups):
                add(FlexibleMeshGroupKind.TP, index, ranks)
            for index, ranks in enumerate(dp_groups):
                add(FlexibleMeshGroupKind.DP, index, ranks)

        result = cls(
            schema_version=FLEXIBLE_MESH_GROUP_REGISTRY_SCHEMA_VERSION,
            mesh=mesh,
            axis_mapping=axis_mapping,
            groups=tuple(groups),
        )
        result.validate()
        return result

    def validate(self, path: str = "flexible_mesh_group_registry") -> None:
        if self.schema_version != FLEXIBLE_MESH_GROUP_REGISTRY_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.mesh.validate(f"{path}.mesh")
        self.axis_mapping.validate(self.mesh, path=f"{path}.axis_mapping")
        if type(self.groups) is not tuple or not self.groups:
            raise SchemaError("must contain groups", path=f"{path}.groups")
        for index, group in enumerate(self.groups):
            if type(group) is not FlexibleMeshGroup:
                raise SchemaError("must be a group", path=f"{path}.groups[{index}]")
            group.validate(self.mesh, f"{path}.groups[{index}]")
        keys = tuple((group.kind, group.index) for group in self.groups)
        if len(set(keys)) != len(keys):
            raise SchemaError("group keys must be unique", path=f"{path}.groups")

    def select(self, kind: FlexibleMeshGroupKind) -> tuple[FlexibleMeshGroup, ...]:
        return tuple(group for group in self.groups if group.kind is kind)


__all__ = [
    "FLEXIBLE_MESH_GROUP_REGISTRY_SCHEMA_VERSION",
    "FLEXIBLE_MESH_GROUP_SCHEMA_VERSION",
    "FlexibleMeshAxisMapping",
    "FlexibleMeshGroup",
    "FlexibleMeshGroupKind",
    "FlexibleMeshGroupRegistry",
]
