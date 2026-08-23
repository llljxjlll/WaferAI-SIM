"""Deterministic candidate enumeration for inter-die Swizzle planning."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from ...errors import SchemaError
from ...schema.ir0 import FusionPattern
from ...schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleBufferRequirement,
    SwizzleCandidate,
    SwizzleFeasibilityWitness,
    SwizzleProblem,
    SwizzleRankProgramWitness,
    SwizzleSemanticWitness,
    SwizzleTensorAxis,
    SwizzleTopologyWitness,
)


@dataclass(frozen=True, slots=True)
class SwizzleCandidateDraft:
    """A fully typed action plan before analytical cost materialization."""

    problem_ref: str
    pattern: FusionPattern
    algorithm: SwizzleAlgorithm
    split_axis: SwizzleTensorAxis
    chunk_count: int
    unroll_degree: int
    rank_programs: tuple[SwizzleRankProgramWitness, ...]
    buffer_requirements: tuple[SwizzleBufferRequirement, ...]
    topology_witness: SwizzleTopologyWitness
    semantic_witness: SwizzleSemanticWitness
    feasibility_witness: SwizzleFeasibilityWitness
    tile_shape: tuple[int, int, int]

    def validate_against(self, problem: SwizzleProblem) -> None:
        if self.problem_ref != problem.id or self.pattern is not problem.pattern:
            raise SchemaError("draft does not belong to problem", path="swizzle_draft")
        if self.algorithm is SwizzleAlgorithm.UNFUSED:
            raise SchemaError("UNFUSED is built directly, not as a draft", path="swizzle_draft.algorithm")
        if self.chunk_count <= 0 or self.unroll_degree not in (1, 2):
            raise SchemaError("invalid decomposition parameters", path="swizzle_draft")
        if self.split_axis != self.semantic_witness.split_axis:
            raise SchemaError("split-axis witness mismatch", path="swizzle_draft.split_axis")
        if not self.feasibility_witness.feasible:
            raise SchemaError("cannot materialize an infeasible draft", path="swizzle_draft.feasibility")
        if len(self.tile_shape) != 3 or any(type(item) is not int or item <= 0 for item in self.tile_shape):
            raise SchemaError("tile_shape must be a positive M/N/K triple", path="swizzle_draft.tile_shape")


CandidateGenerator = Callable[
    [SwizzleProblem, SwizzleSemanticWitness],
    tuple[SwizzleCandidateDraft, ...],
]


def legal_divisors(extent: int, maximum: int) -> tuple[int, ...]:
    """Return canonical positive divisors without scanning past ``maximum``."""

    if type(extent) is not int or extent <= 0:
        raise SchemaError("extent must be a positive integer", path="extent")
    if type(maximum) is not int or maximum <= 0:
        raise SchemaError("maximum must be a positive integer", path="maximum")
    return tuple(value for value in range(1, min(extent, maximum) + 1) if extent % value == 0)


def draft_sort_key(draft: SwizzleCandidateDraft) -> tuple[object, ...]:
    axis = draft.split_axis
    return (
        draft.algorithm.value,
        draft.topology_witness.kind.value,
        axis.tensor_ref,
        axis.index,
        draft.chunk_count,
        draft.unroll_degree,
        draft.topology_witness.rank_order,
    )


def level1_q_score(
    problem: SwizzleProblem,
    draft: SwizzleCandidateDraft,
) -> float:
    """Return the deterministic distance/compute/bandwidth ordering score."""

    route_by_id = {route.id: route for route in problem.group.routes}
    used_routes = []
    for route_ref in draft.topology_witness.route_refs:
        route = route_by_id.get(route_ref)
        if route is None:
            raise SchemaError(
                "topology witness references an unknown route",
                path="swizzle_draft.topology_witness.route_refs",
            )
        used_routes.append(route)
    average_hops = (
        sum(len(route.die_path) - 1 for route in used_routes) / len(used_routes)
        if used_routes
        else 1.0
    )
    compute_intensity = problem.gemm.flops / max(1, problem.collective.logical_bytes)
    return average_hops * compute_intensity / problem.hardware_profile.lane_bytes_per_cycle


@lru_cache(maxsize=256)
def enumerate_drafts(
    problem: SwizzleProblem,
    semantic_witness: SwizzleSemanticWitness,
    generators: tuple[CandidateGenerator, ...],
) -> tuple[SwizzleCandidateDraft, ...]:
    """Run Level 0--2 generators and cap the canonical concrete shortlist."""

    problem.validate("swizzle_problem")
    semantic_witness.validate("semantic_witness")
    if semantic_witness.pattern is not problem.pattern:
        raise SchemaError("semantic witness pattern mismatch", path="semantic_witness.pattern")
    drafts = tuple(
        draft
        for generator in generators
        for draft in generator(problem, semantic_witness)
    )
    for draft in drafts:
        draft.validate_against(problem)
        if draft.algorithm not in problem.constraints.allowed_algorithms:
            raise SchemaError(
                "generator returned a disabled algorithm",
                path="swizzle_drafts.algorithm",
            )
    ordered = tuple(
        sorted(
            drafts,
            key=lambda draft: (
                level1_q_score(problem, draft), draft_sort_key(draft)
            ),
        )
    )
    unique: list[SwizzleCandidateDraft] = []
    seen: set[tuple[object, ...]] = set()
    for draft in ordered:
        key = draft_sort_key(draft)
        if key in seen:
            continue
        seen.add(key)
        unique.append(draft)
        if len(unique) >= min(32, problem.constraints.max_candidates):
            break
    return tuple(unique)


def materialize_drafts(
    problem: SwizzleProblem,
    drafts: tuple[SwizzleCandidateDraft, ...],
) -> tuple[SwizzleCandidate, ...]:
    """Evaluate typed action DAGs and produce stable candidate artifacts."""

    from .cost import evaluate_action_dag

    candidates: list[SwizzleCandidate] = []
    for draft in drafts:
        draft.validate_against(problem)
        cost = evaluate_action_dag(
            problem,
            draft.rank_programs,
            draft.buffer_requirements,
            tile_shape=draft.tile_shape,
        )
        candidates.append(
            SwizzleCandidate.create(
                problem_ref=draft.problem_ref,
                pattern=draft.pattern,
                algorithm=draft.algorithm,
                split_axis=draft.split_axis,
                chunk_count=draft.chunk_count,
                unroll_degree=draft.unroll_degree,
                rank_programs=draft.rank_programs,
                buffer_requirements=draft.buffer_requirements,
                topology_witness=draft.topology_witness,
                semantic_witness=draft.semantic_witness,
                feasibility_witness=draft.feasibility_witness,
                cost=cost,
            )
        )
    return tuple(candidates)


__all__ = [
    "CandidateGenerator",
    "SwizzleCandidateDraft",
    "draft_sort_key",
    "enumerate_drafts",
    "legal_divisors",
    "level1_q_score",
    "materialize_drafts",
]
