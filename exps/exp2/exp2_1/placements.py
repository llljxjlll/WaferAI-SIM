#!/usr/bin/env python3
"""Canonical physical placements for exp2-1.

Coordinates are always ``(x, y)`` and die ids are row-major.  Routing is
X-first/XY and returns directed links, so callers can aggregate contention
without multiplying payload by a guessed hop factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


WAFER_WIDTH = 6
WAFER_HEIGHT = 6
HBM_STACK_CAPACITY_BYTES = 16 * 1024**3
HBM_STACK_BANDWIDTH_BPS = 256.0e9


@dataclass(frozen=True, order=True, slots=True)
class Coordinate:
    x: int
    y: int

    def __post_init__(self) -> None:
        if not (0 <= self.x < WAFER_WIDTH and 0 <= self.y < WAFER_HEIGHT):
            raise ValueError(f"coordinate outside 6x6 wafer: {(self.x, self.y)}")

    @property
    def die_id(self) -> int:
        return self.y * WAFER_WIDTH + self.x

    def as_list(self) -> list[int]:
        return [self.x, self.y]


def coordinate_to_die_id(x: int, y: int) -> int:
    return Coordinate(x, y).die_id


def die_id_to_coordinate(die_id: int) -> Coordinate:
    if type(die_id) is not int or not (0 <= die_id < WAFER_WIDTH * WAFER_HEIGHT):
        raise ValueError(f"invalid die id: {die_id}")
    y, x = divmod(die_id, WAFER_WIDTH)
    return Coordinate(x, y)


@dataclass(frozen=True, order=True, slots=True)
class DirectedLink:
    source_die: int
    destination_die: int

    def __post_init__(self) -> None:
        source = die_id_to_coordinate(self.source_die)
        destination = die_id_to_coordinate(self.destination_die)
        if abs(source.x - destination.x) + abs(source.y - destination.y) != 1:
            raise ValueError("directed links must connect adjacent dies")

    @property
    def resource_id(self) -> str:
        return f"d2d.die{self.source_die}.to.die{self.destination_die}"


def xy_route(source_die: int, destination_die: int) -> tuple[DirectedLink, ...]:
    """Return the exact X-then-Y directed route."""

    source = die_id_to_coordinate(source_die)
    destination = die_id_to_coordinate(destination_die)
    x, y = source.x, source.y
    links: list[DirectedLink] = []
    while x != destination.x:
        next_x = x + (1 if destination.x > x else -1)
        links.append(DirectedLink(coordinate_to_die_id(x, y), coordinate_to_die_id(next_x, y)))
        x = next_x
    while y != destination.y:
        next_y = y + (1 if destination.y > y else -1)
        links.append(DirectedLink(coordinate_to_die_id(x, y), coordinate_to_die_id(x, next_y)))
        y = next_y
    return tuple(links)


@dataclass(frozen=True, slots=True)
class Placement:
    name: str
    role: str
    rows: tuple[int, ...]
    columns: tuple[int, ...]

    @property
    def coordinates(self) -> tuple[Coordinate, ...]:
        return tuple(Coordinate(x, y) for y in self.rows for x in self.columns)

    @property
    def die_ids(self) -> tuple[int, ...]:
        return tuple(coordinate.die_id for coordinate in self.coordinates)

    @property
    def anchor_die(self) -> int:
        return self.die_ids[0]

    def manifest_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "rows": list(self.rows),
            "columns": list(self.columns),
            "coordinates_xy": [coordinate.as_list() for coordinate in self.coordinates],
            "die_ids": list(self.die_ids),
        }


TRAINING_GROUPS = (
    Placement("DP00", "training_tp", (0, 1, 2), (0, 1, 2)),
    Placement("DP01", "training_tp", (0, 1, 2), (3, 4, 5)),
    Placement("DP10", "training_tp", (3, 4, 5), (0, 1, 2)),
    Placement("DP11", "training_tp", (3, 4, 5), (3, 4, 5)),
)

INFERENCE_INSTANCES = (
    Placement("P0", "prefill", (0, 1), (0, 1, 2)),
    Placement("P1", "prefill", (0, 1), (3, 4, 5)),
    Placement("D0", "decode", (2, 3), (0, 1, 2)),
    Placement("D1", "decode", (2, 3), (3, 4, 5)),
    Placement("P2", "prefill", (4, 5), (0, 1, 2)),
    Placement("P3", "prefill", (4, 5), (3, 4, 5)),
)

PD_HANDOFFS = (("P0", "D0"), ("P2", "D0"), ("P1", "D1"), ("P3", "D1"))


@dataclass(frozen=True, slots=True)
class HbmStack:
    stack_id: int
    side: str
    home_die: int
    address_base: int
    capacity_bytes: int = HBM_STACK_CAPACITY_BYTES
    bandwidth_Bps: float = HBM_STACK_BANDWIDTH_BPS

    @property
    def resource_id(self) -> str:
        return f"hbm.stack{self.stack_id}"

    @property
    def address_end(self) -> int:
        return self.address_base + self.capacity_bytes

    def manifest_dict(self) -> dict[str, object]:
        coordinate = die_id_to_coordinate(self.home_die)
        return {
            "stack_id": self.stack_id,
            "side": self.side,
            "home_die": self.home_die,
            "home_coordinate_xy": coordinate.as_list(),
            "address_base": self.address_base,
            "address_end_exclusive": self.address_end,
            "capacity_bytes": self.capacity_bytes,
            "bandwidth_Bps": self.bandwidth_Bps,
        }


HBM_STACKS = tuple(
    HbmStack(index, side, coordinate_to_die_id(x, y), index * HBM_STACK_CAPACITY_BYTES)
    for index, (side, x, y) in enumerate(
        (("S", 1, 0), ("S", 4, 0), ("N", 1, 5), ("N", 4, 5))
    )
)


def placement_by_name(name: str) -> Placement:
    for placement in TRAINING_GROUPS + INFERENCE_INSTANCES:
        if placement.name == name:
            return placement
    raise KeyError(name)


def validate_placements() -> None:
    """Raise if the plan's placements no longer form exact wafer partitions."""

    all_dies = set(range(WAFER_WIDTH * WAFER_HEIGHT))
    for family in (TRAINING_GROUPS, INFERENCE_INSTANCES):
        flattened = [die for placement in family for die in placement.die_ids]
        if len(flattened) != len(set(flattened)) or set(flattened) != all_dies:
            raise AssertionError("placement family must partition all 36 dies exactly")
    ranges = [(stack.address_base, stack.address_end) for stack in HBM_STACKS]
    if sum(stack.capacity_bytes for stack in HBM_STACKS) != 64 * 1024**3:
        raise AssertionError("four stacks must expose exactly 64 GiB")
    for index, first in enumerate(ranges):
        for second in ranges[index + 1 :]:
            if max(first[0], second[0]) < min(first[1], second[1]):
                raise AssertionError("HBM address ranges overlap")


def directed_route_resources(source_die: int, destination_die: int) -> tuple[str, ...]:
    return tuple(link.resource_id for link in xy_route(source_die, destination_die))


def all_directed_links(routes: Iterable[tuple[int, int]]) -> tuple[DirectedLink, ...]:
    links = {link for source, destination in routes for link in xy_route(source, destination)}
    return tuple(sorted(links))


validate_placements()

