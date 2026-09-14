"""Deterministic O(R) topology construction for complete rectangular Meshes."""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import SchemaError


@dataclass(frozen=True, slots=True)
class RectMeshTopology:
    physical_origin: tuple[int, int]
    physical_shape: tuple[int, int]
    is_complete_rectangle: bool
    row_rank_orders: tuple[tuple[int, ...], ...]
    column_rank_orders: tuple[tuple[int, ...], ...]
    snake_rank_order: tuple[int, ...]
    hamiltonian_cycle_rank_order: tuple[int, ...]

    @property
    def rank_count(self) -> int:
        return sum(len(row) for row in self.row_rank_orders)

    @property
    def has_hamiltonian_cycle(self) -> bool:
        return bool(self.hamiltonian_cycle_rank_order)


def _canonical_cycle(order: tuple[int, ...]) -> tuple[int, ...]:
    """Anchor a cycle at its smallest rank and choose its stable direction."""

    if not order:
        return ()
    start = min(order)
    offset = order.index(start)
    forward = order[offset:] + order[:offset]
    reversed_order = tuple(reversed(order))
    reverse_offset = reversed_order.index(start)
    backward = reversed_order[reverse_offset:] + reversed_order[:reverse_offset]
    return min(forward, backward)


def _cycle_coordinates(width: int, height: int) -> tuple[tuple[int, int], ...]:
    """Construct a Hamiltonian cycle, transposing when only width is even."""

    if width <= 1 or height <= 1 or (width * height) % 2:
        return ()
    if height % 2:
        transposed = _cycle_coordinates(height, width)
        return tuple((y, x) for x, y in transposed)

    result = [(x, 0) for x in range(width)]
    for y in range(1, height):
        xs = range(width - 1, 0, -1) if y % 2 else range(1, width)
        result.extend((x, y) for x in xs)
    result.extend((0, y) for y in range(height - 1, 0, -1))
    return tuple(result)


def build_rect_mesh_topology(
    rank_by_coord: dict[tuple[int, int], int],
) -> RectMeshTopology:
    """Build rows, columns, snake and cycle without graph search.

    Coordinates are physical ``(x, y)`` pairs.  Row/column lists remain useful
    for incomplete selections, while cycle construction fails closed unless
    the coordinates form one complete, contiguous rectangle.
    """

    if not rank_by_coord:
        raise SchemaError("must contain at least one rank", path="rank_by_coord")
    if len(set(rank_by_coord.values())) != len(rank_by_coord):
        raise SchemaError("contains duplicate ranks", path="rank_by_coord")
    for coord, rank in rank_by_coord.items():
        if (
            type(coord) is not tuple
            or len(coord) != 2
            or any(type(value) is not int or value < 0 for value in coord)
        ):
            raise SchemaError(
                "coordinates must be non-negative integer pairs",
                path="rank_by_coord",
            )
        if type(rank) is not int or rank < 0:
            raise SchemaError(
                "ranks must be non-negative integers", path="rank_by_coord"
            )

    xs = tuple(sorted({x for x, _ in rank_by_coord}))
    ys = tuple(sorted({y for _, y in rank_by_coord}))
    origin = (xs[0], ys[0])
    width = xs[-1] - xs[0] + 1
    height = ys[-1] - ys[0] + 1
    rectangle_coords = {
        (x, y)
        for x in range(xs[0], xs[-1] + 1)
        for y in range(ys[0], ys[-1] + 1)
    }
    complete = set(rank_by_coord) == rectangle_coords
    rows = tuple(
        tuple(rank_by_coord[(x, y)] for x in xs if (x, y) in rank_by_coord)
        for y in ys
    )
    columns = tuple(
        tuple(rank_by_coord[(x, y)] for y in ys if (x, y) in rank_by_coord)
        for x in xs
    )
    snake = tuple(
        rank
        for row_index, row in enumerate(rows)
        for rank in (row if row_index % 2 == 0 else tuple(reversed(row)))
    )
    cycle: tuple[int, ...] = ()
    if complete:
        coordinate_cycle = _cycle_coordinates(width, height)
        translated = tuple(
            rank_by_coord[(origin[0] + x, origin[1] + y)]
            for x, y in coordinate_cycle
        )
        cycle = _canonical_cycle(translated)
    return RectMeshTopology(
        physical_origin=origin,
        physical_shape=(width, height),
        is_complete_rectangle=complete,
        row_rank_orders=rows,
        column_rank_orders=columns,
        snake_rank_order=snake,
        hamiltonian_cycle_rank_order=cycle,
    )


__all__ = ["RectMeshTopology", "build_rect_mesh_topology"]
