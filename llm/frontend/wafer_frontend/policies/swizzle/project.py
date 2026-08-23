"""Fail-closed compatibility audit for projecting Swizzle plans to IR-2.

This module deliberately does not project a candidate.  The current N5 pass
accepts :class:`FusionPlan` objects and proves an exact correspondence between
their ``FusionAction`` objects and IR-2 tasks.  A ``SwizzleCandidate`` is an
earlier, analytical witness and does not contain all of that executable
contract.  The preflight records which facts can already be proved and stops
at the first boundary that needs the W7 materializer or a shared-schema patch.

Keeping this audit separate makes the interim state honest: matching action
enum spellings are useful, but are not evidence that a candidate is executable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

from ...errors import SchemaError
from ...schema.action import FusionActionKind
from ...schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleProblem,
)


class SwizzleProjectionGate(str, Enum):
    """Ordered proof obligations at the Swizzle-to-IR2 boundary."""

    CANDIDATE_PROVENANCE = "candidate_provenance"
    FUSED_EXECUTION = "fused_execution"
    ACTION_OPCODE_COVERAGE = "action_opcode_coverage"
    TRANSPORT_ROUTE_CLOSURE = "transport_route_closure"
    FUSION_ACTION_CARRIER = "fusion_action_carrier"
    TEMP_VALUE_LINEAGE = "temp_value_lineage"
    PATTERN_REDUCTION_COVERAGE = "pattern_reduction_coverage"
    OUTPUT_OWNERSHIP = "output_ownership"
    NAIVE_INTRA_DIE_SCHEDULE = "naive_intra_die_schedule"


class SwizzleProjectionCheckStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class SwizzleProjectionCheck:
    gate: SwizzleProjectionGate
    status: SwizzleProjectionCheckStatus
    detail: str

    def validate(self, path: str = "swizzle_projection_check") -> None:
        if type(self.gate) is not SwizzleProjectionGate:
            raise SchemaError("must be a SwizzleProjectionGate", path=f"{path}.gate")
        if type(self.status) is not SwizzleProjectionCheckStatus:
            raise SchemaError(
                "must be a SwizzleProjectionCheckStatus",
                path=f"{path}.status",
            )
        if type(self.detail) is not str or not self.detail:
            raise SchemaError("must be a non-empty string", path=f"{path}.detail")


@dataclass(frozen=True, slots=True)
class SwizzleProjectionPreflight:
    """Immutable, ordered evidence for the current N5 projector boundary."""

    problem_ref: str
    candidate_ref: str
    checks: tuple[SwizzleProjectionCheck, ...]

    def validate(self, path: str = "swizzle_projection_preflight") -> None:
        for name in ("problem_ref", "candidate_ref"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise SchemaError("must be a non-empty string", path=f"{path}.{name}")
        if type(self.checks) is not tuple or not self.checks:
            raise SchemaError("must contain immutable checks", path=f"{path}.checks")
        gates = []
        blocked_seen = False
        for index, check in enumerate(self.checks):
            check.validate(f"{path}.checks[{index}]")
            gates.append(check.gate)
            if check.status is SwizzleProjectionCheckStatus.BLOCKED:
                if blocked_seen:
                    raise SchemaError(
                        "must contain exactly one first blocker",
                        path=f"{path}.checks[{index}]",
                    )
                blocked_seen = True
            elif blocked_seen and check.status is not SwizzleProjectionCheckStatus.DEFERRED:
                raise SchemaError(
                    "checks after the first blocker must be deferred",
                    path=f"{path}.checks[{index}]",
                )
        if len(set(gates)) != len(gates):
            raise SchemaError("contains duplicate gates", path=f"{path}.checks")

    @property
    def first_blocker(self) -> SwizzleProjectionCheck | None:
        return next(
            (
                check
                for check in self.checks
                if check.status is SwizzleProjectionCheckStatus.BLOCKED
            ),
            None,
        )

    @property
    def ready(self) -> bool:
        return all(
            check.status is SwizzleProjectionCheckStatus.PASSED
            for check in self.checks
        )


_EXECUTABLE_ACTION_FIELDS = (
    "member_id",
    "slice_ref",
    "logical_channel",
    "expected_route",
    "dtype",
    "compute",
    "reduction",
    "sync",
)


def _transport_route_error(
    problem: SwizzleProblem,
    candidate: SwizzleCandidate,
) -> str | None:
    route_index = {route.id: route for route in problem.group.routes}
    for route_ref in candidate.topology_witness.route_refs:
        if route_ref not in route_index:
            return f"topology references unknown route {route_ref!r}"

    sends: Counter[tuple[int, int, str, int | None, int]] = Counter()
    recvs: Counter[tuple[int, int, str, int | None, int]] = Counter()
    for program in candidate.rank_programs:
        for action in program.actions:
            if action.kind not in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
                continue
            if action.route_ref is None or action.peer_rank is None:
                return f"transport action {action.id!r} lacks route or peer"
            route = route_index.get(action.route_ref)
            if route is None:
                return f"action {action.id!r} references unknown route {action.route_ref!r}"
            source, destination = (
                (action.rank, action.peer_rank)
                if action.kind is SwizzleActionKind.SEND
                else (action.peer_rank, action.rank)
            )
            if (route.source_rank, route.destination_rank) != (source, destination):
                return f"action {action.id!r} route endpoints disagree with its ranks"
            key = (
                source,
                destination,
                action.route_ref,
                action.chunk_index,
                action.logical_bytes,
            )
            (sends if action.kind is SwizzleActionKind.SEND else recvs)[key] += 1
    if sends != recvs:
        return "SEND/RECV witnesses do not pair by route, chunk, and payload"
    return None


def preflight_current_ir2_projection(
    problem: SwizzleProblem,
    candidate: SwizzleCandidate,
) -> SwizzleProjectionPreflight:
    """Audit a candidate against the *current* FusionPlan/IR2 projector.

    The function is intentionally non-mutating and never manufactures missing
    executable semantics.  Today every valid fused candidate reaches the
    ``FUSION_ACTION_CARRIER`` blocker.  Earlier blockers remain useful for
    diagnosing malformed cross-artifact provenance or route pairing.
    """

    if type(problem) is not SwizzleProblem:
        raise SchemaError("must be a SwizzleProblem", path="problem")
    if type(candidate) is not SwizzleCandidate:
        raise SchemaError("must be a SwizzleCandidate", path="candidate")
    problem.validate("swizzle_problem")
    candidate.validate("swizzle_candidate")

    checks: list[SwizzleProjectionCheck] = []
    blocked = False

    def add(
        gate: SwizzleProjectionGate,
        status: SwizzleProjectionCheckStatus,
        detail: str,
    ) -> None:
        nonlocal blocked
        if blocked:
            status = SwizzleProjectionCheckStatus.DEFERRED
        elif status is SwizzleProjectionCheckStatus.BLOCKED:
            blocked = True
        checks.append(SwizzleProjectionCheck(gate, status, detail))

    provenance_closed = (
        candidate.problem_ref == problem.id
        and candidate.pattern is problem.pattern
    )
    add(
        SwizzleProjectionGate.CANDIDATE_PROVENANCE,
        (
            SwizzleProjectionCheckStatus.PASSED
            if provenance_closed
            else SwizzleProjectionCheckStatus.BLOCKED
        ),
        (
            "candidate belongs to the problem and pattern"
            if provenance_closed
            else "candidate problem_ref/pattern disagrees with the problem"
        ),
    )

    is_fused = candidate.algorithm is not SwizzleAlgorithm.UNFUSED
    add(
        SwizzleProjectionGate.FUSED_EXECUTION,
        SwizzleProjectionCheckStatus.PASSED if is_fused else SwizzleProjectionCheckStatus.BLOCKED,
        (
            "candidate carries a decomposed fused action DAG"
            if is_fused
            else "UNFUSED is a cost baseline, not a FusionPlan projection input"
        ),
    )

    unsupported = tuple(
        sorted(
            {
                action.kind.value
                for program in candidate.rank_programs
                for action in program.actions
                if action.kind.value not in {kind.value for kind in FusionActionKind}
            }
        )
    )
    add(
        SwizzleProjectionGate.ACTION_OPCODE_COVERAGE,
        (
            SwizzleProjectionCheckStatus.PASSED
            if not unsupported
            else SwizzleProjectionCheckStatus.BLOCKED
        ),
        (
            "all Swizzle action kinds have existing FusionAction/IR2 task opcodes"
            if not unsupported
            else f"missing executable action kinds: {unsupported!r}"
        ),
    )

    route_error = _transport_route_error(problem, candidate) if is_fused else None
    add(
        SwizzleProjectionGate.TRANSPORT_ROUTE_CLOSURE,
        (
            SwizzleProjectionCheckStatus.PASSED
            if route_error is None
            else SwizzleProjectionCheckStatus.BLOCKED
        ),
        route_error or "real routes and SEND/RECV payload pairs are closed",
    )

    missing = ", ".join(_EXECUTABLE_ACTION_FIELDS)
    add(
        SwizzleProjectionGate.FUSION_ACTION_CARRIER,
        SwizzleProjectionCheckStatus.BLOCKED,
        "current N5 accepts exact FusionPlan/FusionAction inputs; W7 must "
        f"materialize typed {missing} without inference in the projector",
    )
    add(
        SwizzleProjectionGate.TEMP_VALUE_LINEAGE,
        SwizzleProjectionCheckStatus.DEFERRED,
        "projector must bind each planned temporary to an explicit IR1 origin; "
        "its current blanket RECV-to-first-member-output rule is RS-specific",
    )
    add(
        SwizzleProjectionGate.PATTERN_REDUCTION_COVERAGE,
        SwizzleProjectionCheckStatus.DEFERRED,
        "IR2 validation must dispatch AG/RS/AR writer coverage by typed pattern; "
        "the current unconditional full REDUCE cover is RS-specific",
    )
    add(
        SwizzleProjectionGate.OUTPUT_OWNERSHIP,
        SwizzleProjectionCheckStatus.DEFERRED,
        "first executable version must retain identity physical/logical output "
        "ownership or add an explicit permutation lowering",
    )
    add(
        SwizzleProjectionGate.NAIVE_INTRA_DIE_SCHEDULE,
        SwizzleProjectionCheckStatus.DEFERRED,
        "run the existing intra-die scheduler only after the projected task/value DAG closes",
    )

    result = SwizzleProjectionPreflight(problem.id, candidate.id, tuple(checks))
    result.validate()
    return result


__all__ = [
    "SwizzleProjectionCheck",
    "SwizzleProjectionCheckStatus",
    "SwizzleProjectionGate",
    "SwizzleProjectionPreflight",
    "preflight_current_ir2_projection",
]
