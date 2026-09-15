"""Strict, resumable shard execution for M1/M2/M3 release matrices."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from .errors import SchemaError
from .schema.serde import canonical_json, load_json_dataclass
from .schema.workload_release_matrix import (
    WorkloadReleaseCase,
    WorkloadReleaseCaseResult,
    WorkloadReleaseObservedOutcome,
    WorkloadReleasePlan,
    WorkloadReleaseShardBinding,
    WorkloadReleaseSummary,
)


WorkloadReleaseExecutor = Callable[
    [WorkloadReleasePlan, WorkloadReleaseCase], WorkloadReleaseCaseResult
]


def _write_json(path: Path, value: object, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(value))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            raise SchemaError("artifact already exists", path=str(path))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_plan(directory: Path, expected: WorkloadReleasePlan) -> None:
    path = directory / "plan.json"
    loaded = load_json_dataclass(WorkloadReleasePlan, path, path="plan")
    if loaded != expected or loaded.digest != expected.digest:
        raise SchemaError("plan differs from requested plan", path=str(path))


def _load_binding(
    directory: Path,
    plan: WorkloadReleasePlan,
) -> WorkloadReleaseShardBinding:
    path = directory / "shard_binding.json"
    binding = load_json_dataclass(
        WorkloadReleaseShardBinding, path, path="shard_binding"
    )
    binding.validate_against(plan)
    return binding


def _load_results(
    directory: Path,
    plan: WorkloadReleasePlan,
    binding: WorkloadReleaseShardBinding,
) -> dict[str, WorkloadReleaseCaseResult]:
    results_dir = directory / "results"
    if results_dir.is_symlink() or not results_dir.is_dir():
        raise SchemaError("results must be a real directory", path=str(results_dir))
    by_case = {case.id: case for case in plan.cases}
    allowed = set(binding.case_ids)
    loaded: dict[str, WorkloadReleaseCaseResult] = {}
    for artifact in sorted(results_dir.iterdir()):
        if artifact.is_symlink() or not artifact.is_file() or artifact.suffix != ".json":
            raise SchemaError("unexpected result artifact", path=str(artifact))
        case_id = artifact.stem
        if case_id not in allowed or case_id in loaded:
            raise SchemaError("result does not belong to shard", path=str(artifact))
        result = load_json_dataclass(
            WorkloadReleaseCaseResult,
            artifact,
            path=f"results.{case_id}",
        )
        if result.case_id != case_id:
            raise SchemaError("filename and result case differ", path=str(artifact))
        result.validate_against(plan, by_case[case_id])
        loaded[case_id] = result
    return loaded


def run_workload_release_shard(
    plan: WorkloadReleasePlan,
    shard_index: int,
    output_dir: Path,
    executor: WorkloadReleaseExecutor,
    *,
    resume: bool = False,
) -> WorkloadReleaseSummary:
    """Run or resume one immutable shard without trusting stale artifacts."""

    plan.validate()
    output_dir = Path(output_dir)
    binding = WorkloadReleaseShardBinding.create(plan=plan, shard_index=shard_index)
    if output_dir.exists():
        if not resume:
            raise SchemaError("output directory already exists", path=str(output_dir))
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise SchemaError("output must be a real directory", path=str(output_dir))
        allowed_entries = {"plan.json", "shard_binding.json", "results", "summary.json"}
        extras = sorted(entry.name for entry in output_dir.iterdir() if entry.name not in allowed_entries)
        if extras:
            raise SchemaError("unexpected shard artifact", path=str(output_dir / extras[0]))
        _load_plan(output_dir, plan)
        loaded_binding = _load_binding(output_dir, plan)
        if loaded_binding != binding:
            raise SchemaError("shard binding differs", path=str(output_dir / "shard_binding.json"))
    else:
        output_dir.mkdir(parents=True)
        (output_dir / "results").mkdir()
        _write_json(output_dir / "plan.json", plan)
        _write_json(output_dir / "shard_binding.json", binding)

    results = _load_results(output_dir, plan, binding)
    cases = {case.id: case for case in plan.cases_for_shard(shard_index)}
    for case_id in binding.case_ids:
        if case_id in results:
            continue
        case = cases[case_id]
        try:
            result = executor(plan, case)
        except Exception as error:
            result = WorkloadReleaseCaseResult.create(
                plan=plan,
                case=case,
                observed_outcome=WorkloadReleaseObservedOutcome.FAILED,
                diagnostic_code=f"executor.{type(error).__name__}",
            )
        if type(result) is not WorkloadReleaseCaseResult:
            raise SchemaError("executor must return WorkloadReleaseCaseResult", path=case_id)
        result.validate_against(plan, case)
        _write_json(output_dir / "results" / f"{case_id}.json", result)
        results[case_id] = result

    summary = WorkloadReleaseSummary.create(plan=plan, results=tuple(results.values()))
    _write_json(output_dir / "summary.json", summary, replace=True)
    return summary


def merge_workload_release_shards(
    plan: WorkloadReleasePlan,
    shard_dirs: tuple[Path, ...],
    *,
    output_path: Path | None = None,
) -> WorkloadReleaseSummary:
    """Validate and merge exactly one directory for every planned shard."""

    plan.validate()
    if len(shard_dirs) != plan.shard_count:
        raise SchemaError("must provide exactly one directory per shard", path="shard_dirs")
    by_index: dict[int, tuple[Path, WorkloadReleaseShardBinding]] = {}
    for index, raw_directory in enumerate(shard_dirs):
        directory = Path(raw_directory)
        if directory.is_symlink() or not directory.is_dir():
            raise SchemaError("shard must be a real directory", path=f"shard_dirs[{index}]")
        _load_plan(directory, plan)
        binding = _load_binding(directory, plan)
        if binding.shard_index in by_index:
            raise SchemaError("duplicate shard index", path=f"shard_dirs[{index}]")
        by_index[binding.shard_index] = (directory, binding)
    if set(by_index) != set(range(plan.shard_count)):
        raise SchemaError("shard index set is incomplete", path="shard_dirs")

    merged: list[WorkloadReleaseCaseResult] = []
    seen: set[str] = set()
    for shard_index in range(plan.shard_count):
        directory, binding = by_index[shard_index]
        results = _load_results(directory, plan, binding)
        duplicate = seen.intersection(results)
        if duplicate:
            raise SchemaError("case appears in multiple shards", path=str(directory))
        seen.update(results)
        merged.extend(results.values())
    summary = WorkloadReleaseSummary.create(plan=plan, results=tuple(merged))
    if output_path is not None:
        _write_json(Path(output_path), summary, replace=True)
    return summary


__all__ = [
    "WorkloadReleaseExecutor",
    "merge_workload_release_shards",
    "run_workload_release_shard",
]
