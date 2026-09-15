"""Build deterministic M1, M2, and M3 primary release-matrix plans."""

from __future__ import annotations

from ..schema.workload_release_matrix import (
    WorkloadReleaseCapacityProfile,
    WorkloadReleaseCase,
    WorkloadReleaseExpectedOutcome,
    WorkloadReleaseMilestone,
    WorkloadReleasePlan,
)
from ..schema.workload_run import (
    WorkloadExecutionSpec,
    WorkloadFamily,
    WorkloadMemoryMode,
    WorkloadMemoryPolicy,
    WorkloadRunRequest,
)
from .workload_shape_matrix import build_workload_shape_request


PRIMARY_RELEASE_SHAPES = (
    (1, 1),
    (1, 4),
    (4, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (10, 10),
)


def _request_with(
    request: WorkloadRunRequest,
    *,
    memory: WorkloadMemoryPolicy,
    repeats: int,
) -> WorkloadRunRequest:
    execution = WorkloadExecutionSpec(
        timing=request.execution.timing,
        functional=request.execution.functional,
        independent_repeats=repeats,
        strategy=request.execution.strategy,
    )
    return WorkloadRunRequest.create(
        family=request.family,
        model=request.model,
        steps=request.steps,
        mesh=request.mesh,
        parallel=request.parallel,
        memory=memory,
        optimizer=request.optimizer,
        execution=execution,
    )


def _resident_success(request: WorkloadRunRequest) -> WorkloadReleaseCase:
    return WorkloadReleaseCase.create(
        request=_request_with(
            request,
            memory=WorkloadMemoryPolicy(mode=WorkloadMemoryMode.RESIDENT_HBM),
            repeats=2,
        ),
        capacity_profile=WorkloadReleaseCapacityProfile.RESIDENT_SUFFICIENT,
        expected_outcome=WorkloadReleaseExpectedOutcome.RUNTIME_PASS,
        execution_count=2,
    )


def _bounded_pair(
    request: WorkloadRunRequest,
) -> tuple[WorkloadReleaseCase, WorkloadReleaseCase]:
    resident_reject = WorkloadReleaseCase.create(
        request=_request_with(
            request,
            memory=WorkloadMemoryPolicy(mode=WorkloadMemoryMode.RESIDENT_HBM),
            repeats=1,
        ),
        capacity_profile=WorkloadReleaseCapacityProfile.OFFLOAD_BOUNDED,
        expected_outcome=WorkloadReleaseExpectedOutcome.CAPACITY_REJECT,
        execution_count=1,
    )
    offload_success = WorkloadReleaseCase.create(
        request=_request_with(
            request,
            memory=WorkloadMemoryPolicy(
                mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
                external_tier_ref="external:host0",
            ),
            repeats=2,
        ),
        capacity_profile=WorkloadReleaseCapacityProfile.OFFLOAD_BOUNDED,
        expected_outcome=WorkloadReleaseExpectedOutcome.RUNTIME_PASS,
        execution_count=2,
    )
    return resident_reject, offload_success


def build_workload_release_plan(
    milestone: WorkloadReleaseMilestone,
    *,
    source_digest: str,
    binary_digest: str,
    toolchain_digest: str,
    shard_count: int,
) -> WorkloadReleasePlan:
    """Freeze the milestone primary matrix without claiming any runtime result."""

    shapes = (
        PRIMARY_RELEASE_SHAPES
        if milestone in (WorkloadReleaseMilestone.M1, WorkloadReleaseMilestone.M2)
        else tuple((row, column) for row in range(1, 11) for column in range(1, 11))
    )
    cases: list[WorkloadReleaseCase] = []
    for rows, columns in shapes:
        for family in WorkloadFamily:
            _mode, request = build_workload_shape_request(family, rows, columns)
            cases.append(_resident_success(request))
            if milestone is not WorkloadReleaseMilestone.M1:
                cases.extend(_bounded_pair(request))
    return WorkloadReleasePlan.create(
        milestone=milestone,
        source_digest=source_digest,
        binary_digest=binary_digest,
        toolchain_digest=toolchain_digest,
        shard_count=shard_count,
        cases=tuple(cases),
    )


__all__ = ["PRIMARY_RELEASE_SHAPES", "build_workload_release_plan"]
