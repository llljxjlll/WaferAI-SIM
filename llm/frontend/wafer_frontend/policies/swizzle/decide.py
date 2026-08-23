"""Deterministic Level-3 Swizzle candidate selection."""

from __future__ import annotations

from functools import cmp_to_key

from ...errors import SchemaError
from ...schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleDecision,
    SwizzleDecisionReason,
    SwizzleProblem,
)


def _compare(left: SwizzleCandidate, right: SwizzleCandidate) -> int:
    """Order by separated intervals, then documented deterministic ties."""

    if left.cost.upper_cycles < right.cost.lower_cycles:
        return -1
    if right.cost.upper_cycles < left.cost.lower_cycles:
        return 1
    left_key = (
        -left.cost.direction_port_utilization,
        left.cost.control_action_count,
        left.cost.sram_high_water_bytes,
        left.id,
    )
    right_key = (
        -right.cost.direction_port_utilization,
        right.cost.control_action_count,
        right.cost.sram_high_water_bytes,
        right.id,
    )
    return (left_key > right_key) - (left_key < right_key)


def rank_fused_candidates(
    candidates: tuple[SwizzleCandidate, ...],
) -> tuple[SwizzleCandidate, ...]:
    if any(candidate.algorithm is SwizzleAlgorithm.UNFUSED for candidate in candidates):
        raise SchemaError("fused ranking cannot contain UNFUSED", path="candidates")
    if len({candidate.id for candidate in candidates}) != len(candidates):
        raise SchemaError("contains duplicate candidate ids", path="candidates")
    canonical = tuple(sorted(candidates, key=lambda candidate: candidate.id))
    return tuple(sorted(canonical, key=cmp_to_key(_compare)))


def decide_swizzle(
    problem: SwizzleProblem,
    baseline: SwizzleCandidate,
    fused_candidates: tuple[SwizzleCandidate, ...],
) -> SwizzleDecision:
    """Apply the profitability gate and produce a stable ranked decision."""

    problem.validate("swizzle_problem")
    baseline.validate("baseline")
    if baseline.algorithm is not SwizzleAlgorithm.UNFUSED:
        raise SchemaError("baseline must be UNFUSED", path="baseline.algorithm")
    for index, candidate in enumerate(fused_candidates):
        candidate.validate(f"fused_candidates[{index}]")
        if candidate.problem_ref != problem.id or candidate.pattern is not problem.pattern:
            raise SchemaError("candidate does not belong to problem", path=f"fused_candidates[{index}]")
    ordered = rank_fused_candidates(fused_candidates)
    profitable = tuple(
        candidate
        for candidate in ordered
        if candidate.cost.estimated_cycles < baseline.cost.estimated_cycles
    )
    if not ordered:
        ranked = (baseline,)
        reason = SwizzleDecisionReason.BASELINE_ONLY
    elif not profitable:
        ranked = (baseline,) + ordered
        reason = SwizzleDecisionReason.NO_PROFITABLE_FUSION
    else:
        selected = profitable[0]
        ranked = (selected,) + tuple(item for item in ordered if item.id != selected.id) + (baseline,)
        runner_up = ranked[1]
        if selected.cost.upper_cycles < runner_up.cost.lower_cycles:
            reason = SwizzleDecisionReason.NON_OVERLAPPING_INTERVAL
        elif not (
            selected.cost.upper_cycles < runner_up.cost.lower_cycles
            or runner_up.cost.upper_cycles < selected.cost.lower_cycles
        ):
            reason = SwizzleDecisionReason.INTERVAL_TIE_BREAK
        else:
            reason = SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES
    return SwizzleDecision.create(
        problem=problem,
        baseline=baseline,
        ranked_candidates=ranked,
        selected_candidate_ref=ranked[0].id,
        decision_reason=reason,
    )


__all__ = ["decide_swizzle", "rank_fused_candidates"]
