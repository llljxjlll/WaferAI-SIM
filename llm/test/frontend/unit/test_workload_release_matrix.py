from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.workload_release_matrix import (
    PRIMARY_RELEASE_SHAPES,
    build_workload_release_plan,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.frontend.wafer_frontend.schema.workload_release_matrix import (
    WorkloadReleaseCapacityProfile,
    WorkloadReleaseCaseResult,
    WorkloadReleaseExpectedOutcome,
    WorkloadReleaseMilestone,
    WorkloadReleaseObservedOutcome,
    WorkloadReleaseCase,
    WorkloadReleasePlan,
)
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily,
    WorkloadMemoryMode,
    WorkloadParallelSpec,
    WorkloadRunRequest,
)
from llm.frontend.wafer_frontend.workload_release_runner import (
    merge_workload_release_shards,
    run_workload_release_shard,
)


SOURCE = "1" * 64
BINARY = "2" * 64
TOOLCHAIN = "3" * 64


def _plan(milestone: WorkloadReleaseMilestone, shards: int = 4):
    return build_workload_release_plan(
        milestone,
        source_digest=SOURCE,
        binary_digest=BINARY,
        toolchain_digest=TOOLCHAIN,
        shard_count=shards,
    )


def _passing_result(plan, case):
    if case.expected_outcome is WorkloadReleaseExpectedOutcome.CAPACITY_REJECT:
        return WorkloadReleaseCaseResult.create(
            plan=plan,
            case=case,
            observed_outcome=WorkloadReleaseObservedOutcome.CAPACITY_REJECT,
            diagnostic_code="memory_capacity_exceeded",
        )
    digest = hashlib.sha256(case.id.encode("utf-8")).hexdigest()
    return WorkloadReleaseCaseResult.create(
        plan=plan,
        case=case,
        observed_outcome=WorkloadReleaseObservedOutcome.RUNTIME_PASS,
        execution_digests=(digest, digest),
    )


class WorkloadReleaseMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m1 = _plan(WorkloadReleaseMilestone.M1)
        cls.m2 = _plan(WorkloadReleaseMilestone.M2)
        cls.m3 = _plan(WorkloadReleaseMilestone.M3)

    def test_primary_matrix_cardinalities_and_canonical_identity(self) -> None:
        self.assertEqual((len(self.m1.cases), self.m1.success_case_count, self.m1.independent_execution_count), (32, 32, 64))
        self.assertEqual((len(self.m2.cases), self.m2.success_case_count, self.m2.independent_execution_count), (96, 64, 128))
        self.assertEqual((len(self.m3.cases), self.m3.success_case_count, self.m3.independent_execution_count), (1200, 800, 1600))
        self.assertEqual(
            {(case.request.mesh.rows, case.request.mesh.columns) for case in self.m1.cases},
            set(PRIMARY_RELEASE_SHAPES),
        )
        rebuilt = _plan(WorkloadReleaseMilestone.M3)
        self.assertEqual(rebuilt.id, self.m3.id)
        self.assertEqual(canonical_json(rebuilt), canonical_json(self.m3))

    def test_m2_has_resident_success_reject_and_offload_success_per_key(self) -> None:
        for rows, columns in PRIMARY_RELEASE_SHAPES:
            for family in WorkloadFamily:
                cases = [case for case in self.m2.cases if case.matrix_key == (rows, columns, family)]
                self.assertEqual(len(cases), 3)
                observed = {
                    (case.capacity_profile, case.request.memory.mode, case.expected_outcome)
                    for case in cases
                }
                self.assertEqual(
                    observed,
                    {
                        (WorkloadReleaseCapacityProfile.RESIDENT_SUFFICIENT, WorkloadMemoryMode.RESIDENT_HBM, WorkloadReleaseExpectedOutcome.RUNTIME_PASS),
                        (WorkloadReleaseCapacityProfile.OFFLOAD_BOUNDED, WorkloadMemoryMode.RESIDENT_HBM, WorkloadReleaseExpectedOutcome.CAPACITY_REJECT),
                        (WorkloadReleaseCapacityProfile.OFFLOAD_BOUNDED, WorkloadMemoryMode.EXTERNAL_OFFLOAD, WorkloadReleaseExpectedOutcome.RUNTIME_PASS),
                    },
                )

    def test_bounded_pair_rejects_different_model_or_placement(self) -> None:
        bounded = next(
            case for case in self.m2.cases
            if case.request.memory.mode is WorkloadMemoryMode.EXTERNAL_OFFLOAD
            and case.request.mesh.rank_count == 4
        )
        for change in ("model", "parallel"):
            with self.subTest(change=change):
                request = bounded.request
                replacement = (
                    {"model": replace(request.model, intermediate_size=16)}
                    if change == "model"
                    else {"parallel": WorkloadParallelSpec(
                        tp=request.parallel.tp,
                        dp=request.parallel.dp,
                        ep=request.parallel.ep,
                        active_die_ids=tuple(reversed(request.parallel.active_die_ids)),
                    )}
                )
                changed_request = WorkloadRunRequest.create(
                    family=request.family,
                    model=replacement.get("model", request.model),
                    steps=request.steps,
                    mesh=request.mesh,
                    parallel=replacement.get("parallel", request.parallel),
                    memory=request.memory,
                    optimizer=request.optimizer,
                    execution=request.execution,
                )
                changed_case = WorkloadReleaseCase.create(
                    request=changed_request,
                    capacity_profile=bounded.capacity_profile,
                    expected_outcome=bounded.expected_outcome,
                    execution_count=bounded.execution_count,
                )
                cases = tuple(
                    changed_case if case.id == bounded.id else case
                    for case in self.m2.cases
                )
                with self.assertRaisesRegex(
                    SchemaError, "same logical workload and placement"
                ):
                    WorkloadReleasePlan.create(
                        milestone=self.m2.milestone,
                        source_digest=SOURCE,
                        binary_digest=BINARY,
                        toolchain_digest=TOOLCHAIN,
                        shard_count=self.m2.shard_count,
                        cases=cases,
                    )

    def test_shards_are_an_exact_stable_partition(self) -> None:
        partitions = [self.m3.cases_for_shard(index) for index in range(self.m3.shard_count)]
        case_ids = [case.id for partition in partitions for case in partition]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        self.assertEqual(set(case_ids), {case.id for case in self.m3.cases})
        self.assertEqual(partitions, [self.m3.cases_for_shard(index) for index in range(self.m3.shard_count)])

    def test_success_requires_two_identical_execution_digests(self) -> None:
        case = self.m1.cases[0]
        digest = "a" * 64
        _passing_result(self.m1, case)
        with self.assertRaises(SchemaError):
            WorkloadReleaseCaseResult.create(
                plan=self.m1,
                case=case,
                observed_outcome=WorkloadReleaseObservedOutcome.RUNTIME_PASS,
                execution_digests=(digest, "b" * 64),
            )

    def test_capacity_rejection_requires_stable_diagnostic(self) -> None:
        case = next(case for case in self.m2.cases if case.expected_outcome is WorkloadReleaseExpectedOutcome.CAPACITY_REJECT)
        with self.assertRaises(SchemaError):
            WorkloadReleaseCaseResult.create(
                plan=self.m2,
                case=case,
                observed_outcome=WorkloadReleaseObservedOutcome.CAPACITY_REJECT,
                diagnostic_code="out_of_memory",
            )

    def test_resume_skips_verified_results_and_rejects_changed_plan(self) -> None:
        calls: list[str] = []

        def executor(plan, case):
            calls.append(case.id)
            return _passing_result(plan, case)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "shard"
            first = run_workload_release_shard(self.m1, 0, output, executor)
            first_calls = tuple(calls)
            second = run_workload_release_shard(self.m1, 0, output, executor, resume=True)
            self.assertEqual(first, second)
            self.assertEqual(tuple(calls), first_calls)
            changed = build_workload_release_plan(
                WorkloadReleaseMilestone.M1,
                source_digest="4" * 64,
                binary_digest=BINARY,
                toolchain_digest=TOOLCHAIN,
                shard_count=4,
            )
            with self.assertRaises(SchemaError):
                run_workload_release_shard(changed, 0, output, executor, resume=True)

    def test_resume_rejects_extra_result_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "shard"
            run_workload_release_shard(self.m1, 0, output, _passing_result)
            (output / "results" / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(SchemaError):
                run_workload_release_shard(self.m1, 0, output, _passing_result, resume=True)

    def test_full_shard_merge_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_dirs = tuple(root / f"shard-{index}" for index in range(self.m1.shard_count))
            for index, directory in enumerate(shard_dirs):
                run_workload_release_shard(self.m1, index, directory, _passing_result)
            summary = merge_workload_release_shards(self.m1, shard_dirs)
            self.assertTrue(summary.all_planned_cases_complete)
            self.assertEqual(summary.runtime_pass_count, 32)
            self.assertFalse(summary.missing_case_ids)

    def test_executor_failure_is_persisted_and_prevents_completion(self) -> None:
        failed_case_id = self.m1.cases_for_shard(0)[0].id

        def executor(plan, case):
            if case.id == failed_case_id:
                raise RuntimeError("injected backend failure")
            return _passing_result(plan, case)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_dirs = tuple(
                root / f"shard-{index}"
                for index in range(self.m1.shard_count)
            )
            for index, directory in enumerate(shard_dirs):
                run_workload_release_shard(self.m1, index, directory, executor)
            summary = merge_workload_release_shards(self.m1, shard_dirs)
            self.assertFalse(summary.all_planned_cases_complete)
            self.assertEqual(summary.failed_count, 1)
            self.assertFalse(summary.missing_case_ids)
            failure = next(
                result
                for result in summary.results
                if result.case_id == failed_case_id
            )
            self.assertIs(
                failure.observed_outcome,
                WorkloadReleaseObservedOutcome.FAILED,
            )
            self.assertEqual(failure.diagnostic_code, "executor.RuntimeError")


if __name__ == "__main__":
    unittest.main()
