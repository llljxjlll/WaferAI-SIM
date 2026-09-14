#!/usr/bin/env python3
"""Deterministic placement of exp-2's 36 logical ranks on a variable wafer mesh."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from typing import Any, Iterable


LOGICAL_RANKS = 36


class TopologyCapacityError(ValueError):
    """A candidate cannot host one complete 36-rank workload replica."""


@dataclass(frozen=True, order=True, slots=True)
class Module:
    x: int
    y: int
    module_id: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RankPlacement:
    wafer_nx: int
    wafer_ny: int
    topology_status: str
    rank_to_module: tuple[int, ...]
    rank_to_coordinate: tuple[tuple[int, int], ...]
    training_groups: tuple[tuple[int, ...], ...]
    inference_groups: tuple[tuple[int, ...], ...]
    inference_group_names: tuple[str, ...]
    pd_handoffs: tuple[tuple[str, str], ...]
    selected_boundary_edges: int
    mean_pairwise_manhattan: float
    replica_count: int
    mapping_digest: str

    def manifest_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rank_to_module"] = list(self.rank_to_module)
        result["rank_to_coordinate"] = [list(v) for v in self.rank_to_coordinate]
        result["training_groups"] = [list(v) for v in self.training_groups]
        result["inference_groups"] = [list(v) for v in self.inference_groups]
        result["pd_handoffs"] = [list(v) for v in self.pd_handoffs]
        return result


def _canonical_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _adjacent(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1


def _connected(cells: Iterable[tuple[int, int]]) -> bool:
    remaining = set(cells)
    if not remaining:
        return False
    todo = [next(iter(remaining))]
    seen = set(todo)
    while todo:
        x, y = todo.pop()
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if neighbor in remaining and neighbor not in seen:
                seen.add(neighbor)
                todo.append(neighbor)
    return seen == remaining


def _boundary(cells: set[tuple[int, int]]) -> int:
    return sum(
        neighbor not in cells
        for x, y in cells
        for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
    )


def _mean_pair_distance(cells: Iterable[tuple[int, int]]) -> float:
    points = tuple(cells)
    total = sum(
        abs(a[0] - b[0]) + abs(a[1] - b[1])
        for index, a in enumerate(points)
        for b in points[index + 1 :]
    )
    return total / math.comb(len(points), 2)


def _serpentine_path(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...] | None:
    """Return a Hamiltonian row/column serpentine when one exists."""

    candidates: list[tuple[tuple[int, int], ...]] = []
    for transpose in (False, True):
        transformed = {(y, x) if transpose else (x, y) for x, y in cells}
        ys = range(min(y for _, y in transformed), max(y for _, y in transformed) + 1)
        for reverse_rows in (False, True):
            rows = tuple(reversed(tuple(ys))) if reverse_rows else tuple(ys)
            for first_reverse in (False, True):
                path: list[tuple[int, int]] = []
                valid = True
                for row_index, y in enumerate(rows):
                    xs = sorted(x for x, cy in transformed if cy == y)
                    if not xs or xs != list(range(xs[0], xs[-1] + 1)):
                        valid = False
                        break
                    if first_reverse ^ bool(row_index % 2):
                        xs.reverse()
                    path.extend((x, y) for x in xs)
                if valid and all(_adjacent(a, b) for a, b in zip(path, path[1:])):
                    restored = tuple((y, x) if transpose else (x, y) for x, y in path)
                    candidates.append(restored)
    return min(candidates) if candidates else None


def _compositions(total: int, parts: int):
    """Yield deterministic weak compositions of total into parts."""
    for bars in itertools.combinations(range(total + parts - 1), parts - 1):
        marks = (-1,) + bars + (total + parts - 1,)
        yield tuple(marks[i + 1] - marks[i] - 1 for i in range(parts))


def _select_shape(nx: int, ny: int) -> tuple[tuple[int, int], ...]:
    """Exact compact-shape search over minimal bounding rectangles.

    The first objective is exposed boundary, the second is mean Manhattan
    distance, and the final tie-break is the lexicographic global module-id
    sequence.  Requiring a serpentine Hamiltonian path makes both planned
    partition sizes (9 and 6) connected without changing the selected set.
    """

    best: tuple[tuple[Any, ...], tuple[tuple[int, int], ...]] | None = None
    rectangles = []
    for height in range(1, ny + 1):
        width = math.ceil(LOGICAL_RANKS / height)
        excess = width * height - LOGICAL_RANKS
        if width <= nx and excess <= 8:
            rectangles.append((2 * (width + height), excess, height, width))
    for lower_boundary, excess, height, width in sorted(rectangles):
        if best is not None and lower_boundary > best[0][0]:
            continue
        for trims in _compositions(excess, 2 * height):
            cells = set()
            valid_intervals = True
            for y in range(height):
                left, right = trims[2 * y], trims[2 * y + 1]
                if left + right >= width:
                    valid_intervals = False
                    break
                cells.update((x, y) for x in range(left, width - right))
            if not valid_intervals:
                continue
            if not _connected(cells):
                continue
            path = _serpentine_path(cells)
            if path is None:
                continue
            module_ids = tuple(y * nx + x for x, y in path)
            score = (_boundary(cells), _mean_pair_distance(cells), tuple(sorted(module_ids)), module_ids)
            if best is None or score < best[0]:
                best = (score, path)
    if best is None:
        raise TopologyCapacityError(f"cannot construct connected 36-rank map on {nx}x{ny}")
    return best[1]


def _group_distance(
    first: tuple[int, ...], second: tuple[int, ...], coordinates: tuple[tuple[int, int], ...]
) -> int:
    return min(
        abs(coordinates[a][0] - coordinates[b][0]) + abs(coordinates[a][1] - coordinates[b][1])
        for a in first for b in second
    )


def map_logical_ranks(wafer_nx: int, wafer_ny: int) -> RankPlacement:
    """Map one exp-2 replica, or raise with ``topology_capacity_infeasible``."""

    if type(wafer_nx) is not int or type(wafer_ny) is not int or wafer_nx <= 0 or wafer_ny <= 0:
        raise ValueError("wafer dimensions must be positive integers")
    if wafer_nx * wafer_ny < LOGICAL_RANKS:
        raise TopologyCapacityError(
            f"topology_capacity_infeasible: {wafer_nx}x{wafer_ny} has fewer than 36 modules"
        )
    path = _select_shape(wafer_nx, wafer_ny)
    module_ids = tuple(y * wafer_nx + x for x, y in path)
    training = tuple(tuple(range(start, start + 9)) for start in range(0, 36, 9))
    raw_inference = tuple(tuple(range(start, start + 6)) for start in range(0, 36, 6))

    # Pick the two decoder chunks that minimize four P->D handoffs.  Ties use
    # chunk indices, and P/D names then preserve the exp-2 public convention.
    best_roles: tuple[Any, ...] | None = None
    for decoder_set in itertools.combinations(range(6), 2):
        prefills = tuple(index for index in range(6) if index not in decoder_set)
        for decoders in (decoder_set, tuple(reversed(decoder_set))):
            for first_pair in itertools.combinations(prefills, 2):
                second_pair = tuple(index for index in prefills if index not in first_pair)
                assignment_cost = sum(
                    _group_distance(raw_inference[p], raw_inference[decoders[0]], path)
                    for p in first_pair
                ) + sum(
                    _group_distance(raw_inference[p], raw_inference[decoders[1]], path)
                    for p in second_pair
                )
                ordered_indices = (
                    first_pair[0], second_pair[0], decoders[0], decoders[1],
                    first_pair[1], second_pair[1],
                )
                score = (assignment_cost, ordered_indices)
                if best_roles is None or score < best_roles[0]:
                    best_roles = (score, ordered_indices)
    assert best_roles is not None
    ordered_indices = best_roles[1]
    names = ("P0", "P1", "D0", "D1", "P2", "P3")
    inference = tuple(raw_inference[index] for index in ordered_indices)
    handoffs = (("P0", "D0"), ("P2", "D0"), ("P1", "D1"), ("P3", "D1"))

    cells = set(path)
    payload = {
        "schema_version": "exp4.rank_mapping.v1",
        "wafer_nx": wafer_nx,
        "wafer_ny": wafer_ny,
        "rank_to_module": module_ids,
        "training_groups": training,
        "inference_groups": inference,
        "inference_group_names": names,
        "pd_handoffs": handoffs,
    }
    return RankPlacement(
        wafer_nx=wafer_nx,
        wafer_ny=wafer_ny,
        topology_status="capacity_feasible",
        rank_to_module=module_ids,
        rank_to_coordinate=path,
        training_groups=training,
        inference_groups=inference,
        inference_group_names=names,
        pd_handoffs=handoffs,
        selected_boundary_edges=_boundary(cells),
        mean_pairwise_manhattan=_mean_pair_distance(cells),
        replica_count=(wafer_nx * wafer_ny) // LOGICAL_RANKS,
        mapping_digest=_canonical_digest(payload),
    )


def topology_status(wafer_nx: int, wafer_ny: int) -> str:
    return "capacity_feasible" if wafer_nx * wafer_ny >= LOGICAL_RANKS else "topology_capacity_infeasible"


def group_is_connected(group: Iterable[int], placement: RankPlacement) -> bool:
    return _connected(placement.rank_to_coordinate[rank] for rank in group)


__all__ = [
    "LOGICAL_RANKS",
    "Module",
    "RankPlacement",
    "TopologyCapacityError",
    "group_is_connected",
    "map_logical_ranks",
    "topology_status",
]
