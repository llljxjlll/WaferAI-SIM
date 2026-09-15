"""Typed capability envelope for flexible rectangular Die Mesh execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import validate_uint64
from .serde import canonical_digest


RECT_MESH_MAX_ROWS = 10
RECT_MESH_MAX_COLUMNS = 10
RECT_MESH_MAX_RANKS = 100
RECT_MESH_IMPLEMENTATION_MAX_ROWS = 16
RECT_MESH_IMPLEMENTATION_MAX_COLUMNS = 16
RECT_MESH_IMPLEMENTATION_MAX_RANKS = 256


class RectMeshRankOrder(str, Enum):
    ROW_MAJOR = "row_major"


class RectMeshRoutePolicy(str, Enum):
    X_FIRST = "x_first"


class RectMeshWorkloadScope(str, Enum):
    DENSE_FIRST = "dense_first"


@dataclass(frozen=True, slots=True)
class RectMeshSpec:
    """The deliberately small v1 contract for flexible Mesh execution.

    Coordinates use ``(x, y) == (column, row)`` throughout the frontend.  The
    public dimensions retain the more readable ``rows``/``columns`` names;
    ``physical_shape`` exposes the backend's canonical ``(columns, rows)``
    order explicitly.
    """

    rows: int
    columns: int
    rank_order: RectMeshRankOrder = RectMeshRankOrder.ROW_MAJOR
    route_policy: RectMeshRoutePolicy = RectMeshRoutePolicy.X_FIRST
    ranks_per_die: int = 1
    origin: tuple[int, int] = (0, 0)
    timing_execution: bool = True
    functional_execution: bool = False
    workload_scope: RectMeshWorkloadScope = RectMeshWorkloadScope.DENSE_FIRST

    def validate(self, path: str = "rect_mesh") -> None:
        for field_name, maximum in (
            ("rows", RECT_MESH_IMPLEMENTATION_MAX_ROWS),
            ("columns", RECT_MESH_IMPLEMENTATION_MAX_COLUMNS),
        ):
            value = getattr(self, field_name)
            validate_uint64(value, f"{path}.{field_name}")
            if value == 0 or value > maximum:
                raise SchemaError(
                    f"must be in [1, {maximum}]",
                    path=f"{path}.{field_name}",
                )
        if self.rank_count > RECT_MESH_IMPLEMENTATION_MAX_RANKS:
            raise SchemaError(
                "rank count must not exceed "
                f"{RECT_MESH_IMPLEMENTATION_MAX_RANKS}",
                path=path,
            )
        if type(self.rank_order) is not RectMeshRankOrder:
            raise SchemaError(
                "must be a RectMeshRankOrder", path=f"{path}.rank_order"
            )
        if self.rank_order is not RectMeshRankOrder.ROW_MAJOR:
            raise SchemaError("only row_major is supported", path=f"{path}.rank_order")
        if type(self.route_policy) is not RectMeshRoutePolicy:
            raise SchemaError(
                "must be a RectMeshRoutePolicy", path=f"{path}.route_policy"
            )
        if self.route_policy is not RectMeshRoutePolicy.X_FIRST:
            raise SchemaError("only x_first is supported", path=f"{path}.route_policy")
        if type(self.ranks_per_die) is not int or self.ranks_per_die != 1:
            raise SchemaError(
                "v1 requires exactly one rank per Die", path=f"{path}.ranks_per_die"
            )
        if type(self.origin) is not tuple or self.origin != (0, 0):
            raise SchemaError("v1 requires origin (0, 0)", path=f"{path}.origin")
        if self.timing_execution is not True:
            raise SchemaError(
                "v1 requires timing execution", path=f"{path}.timing_execution"
            )
        if self.functional_execution is not False:
            raise SchemaError(
                "v1 does not support functional execution",
                path=f"{path}.functional_execution",
            )
        if type(self.workload_scope) is not RectMeshWorkloadScope:
            raise SchemaError(
                "must be a RectMeshWorkloadScope", path=f"{path}.workload_scope"
            )
        if self.workload_scope is not RectMeshWorkloadScope.DENSE_FIRST:
            raise SchemaError(
                "v1 supports the dense-first scope only",
                path=f"{path}.workload_scope",
            )

    @property
    def rank_count(self) -> int:
        return self.rows * self.columns

    @property
    def within_release_envelope(self) -> bool:
        """Whether this shape belongs to the frozen 1..10 release matrix."""

        return (
            self.rows <= RECT_MESH_MAX_ROWS
            and self.columns <= RECT_MESH_MAX_COLUMNS
            and self.rank_count <= RECT_MESH_MAX_RANKS
        )

    @property
    def physical_shape(self) -> tuple[int, int]:
        """Return ``(width, height)`` as consumed by PhysicalFabric."""

        return (self.columns, self.rows)

    @property
    def row_rank_orders(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(self.rank(row, column) for column in range(self.columns))
            for row in range(self.rows)
        )

    @property
    def column_rank_orders(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(self.rank(row, column) for row in range(self.rows))
            for column in range(self.columns)
        )

    @property
    def snake_rank_order(self) -> tuple[int, ...]:
        return tuple(
            rank
            for row_index, row in enumerate(self.row_rank_orders)
            for rank in (row if row_index % 2 == 0 else tuple(reversed(row)))
        )

    @property
    def has_hamiltonian_cycle(self) -> bool:
        return self.rows > 1 and self.columns > 1 and self.rank_count % 2 == 0

    @property
    def ordered_rank_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (source, destination)
            for source in range(self.rank_count)
            for destination in range(self.rank_count)
            if source != destination
        )

    @property
    def directed_link_count(self) -> int:
        return 2 * (
            self.rows * (self.columns - 1)
            + self.columns * (self.rows - 1)
        )

    @property
    def max_hop_count(self) -> int:
        return self.rows + self.columns - 2

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def rank(self, row: int, column: int) -> int:
        if type(row) is not int or row < 0 or row >= self.rows:
            raise SchemaError("row lies outside the Mesh", path="rect_mesh.row")
        if type(column) is not int or column < 0 or column >= self.columns:
            raise SchemaError(
                "column lies outside the Mesh", path="rect_mesh.column"
            )
        return row * self.columns + column

    def coordinate(self, rank: int) -> tuple[int, int]:
        if type(rank) is not int or rank < 0 or rank >= self.rank_count:
            raise SchemaError("rank lies outside the Mesh", path="rect_mesh.rank")
        row, column = divmod(rank, self.columns)
        return (column, row)

    def validate_participant_count(
        self, participant_count: int, path: str = "participant_count"
    ) -> None:
        if type(participant_count) is not int or participant_count != self.rank_count:
            raise SchemaError(
                f"must equal rectangular Mesh rank count {self.rank_count}", path=path
            )

    def validate_rank_coordinates(
        self,
        rank_coordinates: tuple[tuple[int, int], ...],
        path: str = "rank_coordinates",
    ) -> None:
        expected = tuple(self.coordinate(rank) for rank in range(self.rank_count))
        if type(rank_coordinates) is not tuple or rank_coordinates != expected:
            raise SchemaError(
                "must be the complete, hole-free row-major placement", path=path
            )


__all__ = [
    "RECT_MESH_IMPLEMENTATION_MAX_COLUMNS",
    "RECT_MESH_IMPLEMENTATION_MAX_RANKS",
    "RECT_MESH_IMPLEMENTATION_MAX_ROWS",
    "RECT_MESH_MAX_COLUMNS",
    "RECT_MESH_MAX_RANKS",
    "RECT_MESH_MAX_ROWS",
    "RectMeshRankOrder",
    "RectMeshRoutePolicy",
    "RectMeshSpec",
    "RectMeshWorkloadScope",
]
