"""Typed primary-matrix plans and observations for M1, M2, and M3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty
from .serde import canonical_digest
from .workload_run import WorkloadFamily, WorkloadMemoryMode, WorkloadRunRequest


WORKLOAD_RELEASE_CASE_SCHEMA_VERSION = (
    "wafer_frontend.workload_release_case/v1alpha1"
)
WORKLOAD_RELEASE_PLAN_SCHEMA_VERSION = (
    "wafer_frontend.workload_release_plan/v1alpha1"
)
WORKLOAD_RELEASE_CASE_RESULT_SCHEMA_VERSION = (
    "wafer_frontend.workload_release_case_result/v1alpha1"
)
WORKLOAD_RELEASE_SHARD_BINDING_SCHEMA_VERSION = (
    "wafer_frontend.workload_release_shard_binding/v1alpha1"
)
WORKLOAD_RELEASE_SUMMARY_SCHEMA_VERSION = (
    "wafer_frontend.workload_release_summary/v1alpha1"
)


class WorkloadReleaseMilestone(str, Enum):
    M1 = "m1"
    M2 = "m2"
    M3 = "m3"


class WorkloadReleaseCapacityProfile(str, Enum):
    RESIDENT_SUFFICIENT = "resident_sufficient"
    OFFLOAD_BOUNDED = "offload_bounded"


class WorkloadReleaseExpectedOutcome(str, Enum):
    RUNTIME_PASS = "runtime_pass"
    CAPACITY_REJECT = "capacity_reject"


class WorkloadReleaseObservedOutcome(str, Enum):
    RUNTIME_PASS = "runtime_pass"
    CAPACITY_REJECT = "capacity_reject"
    FAILED = "failed"
    BLOCKED = "blocked"


_FAMILY_RANK = {family: index for index, family in enumerate(WorkloadFamily)}
_MEMORY_RANK = {
    WorkloadMemoryMode.RESIDENT_HBM: 0,
    WorkloadMemoryMode.EXTERNAL_OFFLOAD: 1,
    WorkloadMemoryMode.REMOTE_HBM: 2,
}
_PROFILE_RANK = {
    WorkloadReleaseCapacityProfile.RESIDENT_SUFFICIENT: 0,
    WorkloadReleaseCapacityProfile.OFFLOAD_BOUNDED: 1,
}


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class WorkloadReleaseCase:
    schema_version: str
    id: str
    request: WorkloadRunRequest
    request_digest: str
    capacity_profile: WorkloadReleaseCapacityProfile
    expected_outcome: WorkloadReleaseExpectedOutcome
    execution_count: int

    @classmethod
    def create(
        cls,
        *,
        request: WorkloadRunRequest,
        capacity_profile: WorkloadReleaseCapacityProfile,
        expected_outcome: WorkloadReleaseExpectedOutcome,
        execution_count: int,
    ) -> "WorkloadReleaseCase":
        key = {
            "request": request,
            "request_digest": request.digest,
            "capacity_profile": capacity_profile,
            "expected_outcome": expected_outcome,
            "execution_count": execution_count,
        }
        result = cls(
            schema_version=WORKLOAD_RELEASE_CASE_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_release_case",
                key,
                schema_version=WORKLOAD_RELEASE_CASE_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    @property
    def matrix_key(self) -> tuple[int, int, WorkloadFamily]:
        return self.request.mesh.rows, self.request.mesh.columns, self.request.family

    @property
    def shard_key(self) -> int:
        return int(canonical_digest(self)[:16], 16)

    def validate(self, path: str = "workload_release_case") -> None:
        if self.schema_version != WORKLOAD_RELEASE_CASE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.request) is not WorkloadRunRequest:
            raise SchemaError("must be a WorkloadRunRequest", path=f"{path}.request")
        self.request.validate(f"{path}.request")
        if self.request_digest != self.request.digest:
            raise SchemaError("does not match request", path=f"{path}.request_digest")
        if type(self.capacity_profile) is not WorkloadReleaseCapacityProfile:
            raise SchemaError("must be a capacity profile", path=f"{path}.capacity_profile")
        if type(self.expected_outcome) is not WorkloadReleaseExpectedOutcome:
            raise SchemaError("must be an expected outcome", path=f"{path}.expected_outcome")
        if type(self.execution_count) is not int or self.execution_count <= 0:
            raise SchemaError("must be a positive int", path=f"{path}.execution_count")
        if self.expected_outcome is WorkloadReleaseExpectedOutcome.RUNTIME_PASS:
            if self.execution_count != 2:
                raise SchemaError("runtime pass requires two executions", path=f"{path}.execution_count")
            if self.request.execution.independent_repeats != 2:
                raise SchemaError("request must require two repeats", path=f"{path}.request.execution")
        else:
            if self.execution_count != 1:
                raise SchemaError("capacity rejection requires one execution", path=f"{path}.execution_count")
            if self.request.execution.independent_repeats != 1:
                raise SchemaError("rejection request requires one attempt", path=f"{path}.request.execution")
        mode = self.request.memory.mode
        if self.capacity_profile is WorkloadReleaseCapacityProfile.RESIDENT_SUFFICIENT:
            if mode is not WorkloadMemoryMode.RESIDENT_HBM or self.expected_outcome is not WorkloadReleaseExpectedOutcome.RUNTIME_PASS:
                raise SchemaError("resident-sufficient cases must pass resident runtime", path=path)
        elif mode is WorkloadMemoryMode.RESIDENT_HBM:
            if self.expected_outcome is not WorkloadReleaseExpectedOutcome.CAPACITY_REJECT:
                raise SchemaError("bounded resident case must reject capacity", path=path)
        elif mode is WorkloadMemoryMode.EXTERNAL_OFFLOAD:
            if self.expected_outcome is not WorkloadReleaseExpectedOutcome.RUNTIME_PASS:
                raise SchemaError("bounded offload case must pass runtime", path=path)
        else:
            raise SchemaError("primary matrix does not include remote HBM", path=path)
        expected = stable_artifact_id(
            "workload_release_case",
            self._key(),
            schema_version=WORKLOAD_RELEASE_CASE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable case id", path=f"{path}.id")


def _case_sort_key(case: WorkloadReleaseCase) -> tuple[int, int, int, int, int]:
    return (
        case.request.mesh.rows,
        case.request.mesh.columns,
        _FAMILY_RANK[case.request.family],
        _PROFILE_RANK[case.capacity_profile],
        _MEMORY_RANK[case.request.memory.mode],
    )


@dataclass(frozen=True, slots=True)
class WorkloadReleasePlan:
    schema_version: str
    id: str
    milestone: WorkloadReleaseMilestone
    source_digest: str
    binary_digest: str
    toolchain_digest: str
    shard_count: int
    cases: tuple[WorkloadReleaseCase, ...]
    success_case_count: int
    independent_execution_count: int

    @classmethod
    def create(
        cls,
        *,
        milestone: WorkloadReleaseMilestone,
        source_digest: str,
        binary_digest: str,
        toolchain_digest: str,
        shard_count: int,
        cases: tuple[WorkloadReleaseCase, ...],
    ) -> "WorkloadReleasePlan":
        canonical = tuple(sorted(cases, key=_case_sort_key))
        success_cases = tuple(
            case for case in canonical
            if case.expected_outcome is WorkloadReleaseExpectedOutcome.RUNTIME_PASS
        )
        key = {
            "milestone": milestone,
            "source_digest": source_digest,
            "binary_digest": binary_digest,
            "toolchain_digest": toolchain_digest,
            "shard_count": shard_count,
            "cases": canonical,
            "success_case_count": len(success_cases),
            "independent_execution_count": sum(
                case.execution_count for case in success_cases
            ),
        }
        result = cls(
            schema_version=WORKLOAD_RELEASE_PLAN_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_release_plan",
                key,
                schema_version=WORKLOAD_RELEASE_PLAN_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def cases_for_shard(self, shard_index: int) -> tuple[WorkloadReleaseCase, ...]:
        if type(shard_index) is not int or not 0 <= shard_index < self.shard_count:
            raise SchemaError("shard index out of range", path="shard_index")
        return tuple(case for case in self.cases if case.shard_key % self.shard_count == shard_index)

    def validate(self, path: str = "workload_release_plan") -> None:
        if self.schema_version != WORKLOAD_RELEASE_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.milestone) is not WorkloadReleaseMilestone:
            raise SchemaError("must be a milestone", path=f"{path}.milestone")
        for name in ("source_digest", "binary_digest", "toolchain_digest"):
            _digest(getattr(self, name), f"{path}.{name}")
        if type(self.shard_count) is not int or self.shard_count <= 0:
            raise SchemaError("must be a positive int", path=f"{path}.shard_count")
        if type(self.cases) is not tuple or not self.cases:
            raise SchemaError("must be a non-empty tuple", path=f"{path}.cases")
        for index, case in enumerate(self.cases):
            if type(case) is not WorkloadReleaseCase:
                raise SchemaError("must be a release case", path=f"{path}.cases[{index}]")
            case.validate(f"{path}.cases[{index}]")
        if self.cases != tuple(sorted(self.cases, key=_case_sort_key)):
            raise SchemaError("must use canonical case order", path=f"{path}.cases")
        ids = tuple(case.id for case in self.cases)
        if len(ids) != len(set(ids)):
            raise SchemaError("contains duplicate cases", path=f"{path}.cases")
        shape_count = 8 if self.milestone in (WorkloadReleaseMilestone.M1, WorkloadReleaseMilestone.M2) else 100
        modes_per_key = 1 if self.milestone is WorkloadReleaseMilestone.M1 else 3
        expected_total = shape_count * len(WorkloadFamily) * modes_per_key
        expected_success = shape_count * len(WorkloadFamily) * (1 if self.milestone is WorkloadReleaseMilestone.M1 else 2)
        if len(self.cases) != expected_total:
            raise SchemaError("case count does not match milestone primary matrix", path=f"{path}.cases")
        if self.success_case_count != expected_success:
            raise SchemaError("success case count does not match milestone", path=f"{path}.success_case_count")
        if self.independent_execution_count != expected_success * 2:
            raise SchemaError("execution count does not match double-run contract", path=f"{path}.independent_execution_count")
        grouped: dict[tuple[int, int, WorkloadFamily], set[tuple[WorkloadReleaseCapacityProfile, WorkloadMemoryMode, WorkloadReleaseExpectedOutcome]]] = {}
        for case in self.cases:
            grouped.setdefault(case.matrix_key, set()).add(
                (case.capacity_profile, case.request.memory.mode, case.expected_outcome)
            )
        expected_group = {
            (
                WorkloadReleaseCapacityProfile.RESIDENT_SUFFICIENT,
                WorkloadMemoryMode.RESIDENT_HBM,
                WorkloadReleaseExpectedOutcome.RUNTIME_PASS,
            )
        }
        if self.milestone is not WorkloadReleaseMilestone.M1:
            expected_group |= {
                (
                    WorkloadReleaseCapacityProfile.OFFLOAD_BOUNDED,
                    WorkloadMemoryMode.RESIDENT_HBM,
                    WorkloadReleaseExpectedOutcome.CAPACITY_REJECT,
                ),
                (
                    WorkloadReleaseCapacityProfile.OFFLOAD_BOUNDED,
                    WorkloadMemoryMode.EXTERNAL_OFFLOAD,
                    WorkloadReleaseExpectedOutcome.RUNTIME_PASS,
                ),
            }
        if len(grouped) != shape_count * len(WorkloadFamily) or any(value != expected_group for value in grouped.values()):
            raise SchemaError("shape/family memory pairing is incomplete", path=f"{path}.cases")
        expected = stable_artifact_id(
            "workload_release_plan",
            self._key(),
            schema_version=WORKLOAD_RELEASE_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable plan id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class WorkloadReleaseShardBinding:
    schema_version: str
    id: str
    plan_id: str
    plan_digest: str
    shard_index: int
    case_ids: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        plan: WorkloadReleasePlan,
        shard_index: int,
    ) -> "WorkloadReleaseShardBinding":
        cases = plan.cases_for_shard(shard_index)
        key = {
            "plan_id": plan.id,
            "plan_digest": plan.digest,
            "shard_index": shard_index,
            "case_ids": tuple(case.id for case in cases),
        }
        result = cls(
            schema_version=WORKLOAD_RELEASE_SHARD_BINDING_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_release_shard_binding",
                key,
                schema_version=WORKLOAD_RELEASE_SHARD_BINDING_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate_against(plan)
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate_against(
        self,
        plan: WorkloadReleasePlan,
        path: str = "workload_release_shard_binding",
    ) -> None:
        plan.validate(f"{path}.plan")
        if self.schema_version != WORKLOAD_RELEASE_SHARD_BINDING_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.plan_id != plan.id or self.plan_digest != plan.digest:
            raise SchemaError("does not match plan", path=path)
        expected_case_ids = tuple(
            case.id for case in plan.cases_for_shard(self.shard_index)
        )
        if self.case_ids != expected_case_ids:
            raise SchemaError("case ids do not match shard", path=f"{path}.case_ids")
        expected = stable_artifact_id(
            "workload_release_shard_binding",
            self._key(),
            schema_version=WORKLOAD_RELEASE_SHARD_BINDING_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable shard binding id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class WorkloadReleaseCaseResult:
    schema_version: str
    id: str
    plan_id: str
    case_id: str
    observed_outcome: WorkloadReleaseObservedOutcome
    execution_digests: tuple[str, ...]
    diagnostic_code: str | None

    @classmethod
    def create(
        cls,
        *,
        plan: WorkloadReleasePlan,
        case: WorkloadReleaseCase,
        observed_outcome: WorkloadReleaseObservedOutcome,
        execution_digests: tuple[str, ...] = (),
        diagnostic_code: str | None = None,
    ) -> "WorkloadReleaseCaseResult":
        key = {
            "plan_id": plan.id,
            "case_id": case.id,
            "observed_outcome": observed_outcome,
            "execution_digests": execution_digests,
            "diagnostic_code": diagnostic_code,
        }
        result = cls(
            schema_version=WORKLOAD_RELEASE_CASE_RESULT_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_release_case_result",
                key,
                schema_version=WORKLOAD_RELEASE_CASE_RESULT_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate_against(plan, case)
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate_against(
        self,
        plan: WorkloadReleasePlan,
        case: WorkloadReleaseCase,
        path: str = "workload_release_case_result",
    ) -> None:
        if self.schema_version != WORKLOAD_RELEASE_CASE_RESULT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.plan_id != plan.id or self.case_id != case.id:
            raise SchemaError("result binding mismatch", path=path)
        if type(self.observed_outcome) is not WorkloadReleaseObservedOutcome:
            raise SchemaError("must be an observed outcome", path=f"{path}.observed_outcome")
        for index, digest in enumerate(self.execution_digests):
            _digest(digest, f"{path}.execution_digests[{index}]")
        if self.observed_outcome is WorkloadReleaseObservedOutcome.RUNTIME_PASS:
            if case.expected_outcome is not WorkloadReleaseExpectedOutcome.RUNTIME_PASS:
                raise SchemaError("runtime pass contradicts expected rejection", path=path)
            if len(self.execution_digests) != case.execution_count or len(set(self.execution_digests)) != 1:
                raise SchemaError("runtime pass requires identical independent execution digests", path=f"{path}.execution_digests")
            if self.diagnostic_code is not None:
                raise SchemaError("runtime pass cannot have a diagnostic", path=f"{path}.diagnostic_code")
        elif self.observed_outcome is WorkloadReleaseObservedOutcome.CAPACITY_REJECT:
            if case.expected_outcome is not WorkloadReleaseExpectedOutcome.CAPACITY_REJECT:
                raise SchemaError("capacity rejection contradicts expected runtime", path=path)
            if self.execution_digests or self.diagnostic_code != "memory_capacity_exceeded":
                raise SchemaError("capacity rejection requires the stable diagnostic", path=path)
        else:
            if self.execution_digests:
                raise SchemaError("failed or blocked results cannot publish executions", path=path)
            if self.diagnostic_code is None:
                raise SchemaError("failed or blocked result requires a diagnostic", path=f"{path}.diagnostic_code")
            validate_nonempty(self.diagnostic_code, f"{path}.diagnostic_code")
        expected = stable_artifact_id(
            "workload_release_case_result",
            self._key(),
            schema_version=WORKLOAD_RELEASE_CASE_RESULT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable result id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class WorkloadReleaseSummary:
    schema_version: str
    id: str
    plan_id: str
    results: tuple[WorkloadReleaseCaseResult, ...]
    missing_case_ids: tuple[str, ...]
    runtime_pass_count: int
    capacity_reject_count: int
    failed_count: int
    blocked_count: int
    all_planned_cases_complete: bool

    @classmethod
    def create(
        cls,
        *,
        plan: WorkloadReleasePlan,
        results: tuple[WorkloadReleaseCaseResult, ...],
    ) -> "WorkloadReleaseSummary":
        by_case = {case.id: case for case in plan.cases}
        unknown = tuple(result.case_id for result in results if result.case_id not in by_case)
        if unknown:
            raise SchemaError("unknown result case", path="workload_release_summary.results")
        canonical = tuple(sorted(results, key=lambda item: _case_sort_key(by_case[item.case_id])))
        observed_ids = {result.case_id for result in canonical}
        missing = tuple(case.id for case in plan.cases if case.id not in observed_ids)
        counts = {
            outcome: sum(result.observed_outcome is outcome for result in canonical)
            for outcome in WorkloadReleaseObservedOutcome
        }
        complete = not missing and not counts[WorkloadReleaseObservedOutcome.FAILED] and not counts[WorkloadReleaseObservedOutcome.BLOCKED]
        key = {
            "plan_id": plan.id,
            "results": canonical,
            "missing_case_ids": missing,
            "runtime_pass_count": counts[WorkloadReleaseObservedOutcome.RUNTIME_PASS],
            "capacity_reject_count": counts[WorkloadReleaseObservedOutcome.CAPACITY_REJECT],
            "failed_count": counts[WorkloadReleaseObservedOutcome.FAILED],
            "blocked_count": counts[WorkloadReleaseObservedOutcome.BLOCKED],
            "all_planned_cases_complete": complete,
        }
        result = cls(
            schema_version=WORKLOAD_RELEASE_SUMMARY_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_release_summary",
                key,
                schema_version=WORKLOAD_RELEASE_SUMMARY_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate_against(plan)
        return result

    def _key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate_against(self, plan: WorkloadReleasePlan, path: str = "workload_release_summary") -> None:
        if self.schema_version != WORKLOAD_RELEASE_SUMMARY_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.plan_id != plan.id:
            raise SchemaError("summary plan binding mismatch", path=f"{path}.plan_id")
        by_case = {case.id: case for case in plan.cases}
        seen: set[str] = set()
        for index, result in enumerate(self.results):
            if result.case_id not in by_case or result.case_id in seen:
                raise SchemaError("unknown or duplicate result case", path=f"{path}.results[{index}]")
            result.validate_against(plan, by_case[result.case_id], f"{path}.results[{index}]")
            seen.add(result.case_id)
        missing = tuple(case.id for case in plan.cases if case.id not in seen)
        if self.missing_case_ids != missing:
            raise SchemaError("missing case set mismatch", path=f"{path}.missing_case_ids")
        counts = {outcome: sum(result.observed_outcome is outcome for result in self.results) for outcome in WorkloadReleaseObservedOutcome}
        expected_fields = (
            counts[WorkloadReleaseObservedOutcome.RUNTIME_PASS],
            counts[WorkloadReleaseObservedOutcome.CAPACITY_REJECT],
            counts[WorkloadReleaseObservedOutcome.FAILED],
            counts[WorkloadReleaseObservedOutcome.BLOCKED],
        )
        if expected_fields != (self.runtime_pass_count, self.capacity_reject_count, self.failed_count, self.blocked_count):
            raise SchemaError("summary counts mismatch", path=path)
        complete = not missing and self.failed_count == 0 and self.blocked_count == 0
        if self.all_planned_cases_complete is not complete:
            raise SchemaError("completion flag mismatch", path=f"{path}.all_planned_cases_complete")
        expected_id = stable_artifact_id(
            "workload_release_summary",
            self._key(),
            schema_version=WORKLOAD_RELEASE_SUMMARY_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable summary id", path=f"{path}.id")


__all__ = [
    "WORKLOAD_RELEASE_CASE_RESULT_SCHEMA_VERSION",
    "WORKLOAD_RELEASE_SHARD_BINDING_SCHEMA_VERSION",
    "WORKLOAD_RELEASE_CASE_SCHEMA_VERSION",
    "WORKLOAD_RELEASE_PLAN_SCHEMA_VERSION",
    "WORKLOAD_RELEASE_SUMMARY_SCHEMA_VERSION",
    "WorkloadReleaseCapacityProfile",
    "WorkloadReleaseCase",
    "WorkloadReleaseCaseResult",
    "WorkloadReleaseExpectedOutcome",
    "WorkloadReleaseMilestone",
    "WorkloadReleaseObservedOutcome",
    "WorkloadReleasePlan",
    "WorkloadReleaseShardBinding",
    "WorkloadReleaseSummary",
]
