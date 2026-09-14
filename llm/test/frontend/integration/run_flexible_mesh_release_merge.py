"""Merge workload runtime roots into strict exhaustive or representative completion."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path
from typing import Any

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_representative_completion import (
    FlexibleMeshValidationScope,
    select_flexible_mesh_representative_cases,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FLEXIBLE_MESH_RELEASE_CAPACITY_POLICY_VERSION,
    FLEXIBLE_MESH_RELEASE_MAX_ARTIFACT_BYTES,
    FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES,
    FLEXIBLE_MESH_RELEASE_MAX_RECORDS,
    FLEXIBLE_MESH_RELEASE_MAX_RUNTIME_CORE_ID,
    FLEXIBLE_MESH_RELEASE_MAX_SESSIONS_PER_CORE_PER_WAVE,
    FLEXIBLE_MESH_RELEASE_MAX_TRANSPORT_TAGS,
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseFamily,
    generate_flexible_mesh_release_cases,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_json,
    load_json_dataclass,
)

from flexible_mesh_release_report import (
    aggregate_flexible_mesh_representative_roots,
    aggregate_flexible_mesh_release_roots,
    write_flexible_mesh_representative_completion_report,
    write_flexible_mesh_completion_report,
)
from flexible_mesh_release_profiles import release_trace_model_digests


_ROOT_FAMILIES = (
    (FlexibleMeshReleaseFamily.DENSE_TRAIN,),
    (
        FlexibleMeshReleaseFamily.MOE_INFERENCE,
        FlexibleMeshReleaseFamily.MOE_TRAIN,
    ),
    (
        FlexibleMeshReleaseFamily.MESHSLICE_AG,
        FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,
        FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,
    ),
)


def audit_flexible_mesh_release_roots(
    *,
    binding: FlexibleMeshReleaseBinding,
    cases: tuple[FlexibleMeshReleaseCase, ...],
    runtime_roots: tuple[Path, ...],
) -> tuple[dict[str, Any], tuple[FlexibleMeshReleaseCaseEvidence, ...]]:
    """Inventory evidence without ever deriving completion from a partial set."""

    binding.validate("binding")
    case_by_id = {case.id: case for case in cases}
    observed: dict[str, tuple[FlexibleMeshReleaseCaseEvidence, Path]] = {}
    missing_roots: list[str] = []
    invalid: list[dict[str, str]] = []
    drifted: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    root_binding_sha256: dict[str, str] = {}
    scanned_files = 0
    expected_binding_bytes = canonical_json(binding).encode("utf-8")
    if type(runtime_roots) is not tuple or len(runtime_roots) != len(_ROOT_FAMILIES):
        raise ValueError("runtime_roots must be Dense, MoE, MeshSlice in that order")
    for root_index, root in enumerate(runtime_roots):
        if not root.is_dir():
            missing_roots.append(str(root))
            continue
        binding_path = root / "release_binding.json"
        try:
            binding_bytes = binding_path.read_bytes()
            root_binding_sha256[str(root)] = hashlib.sha256(
                binding_bytes
            ).hexdigest()
            root_binding = load_json_dataclass(
                FlexibleMeshReleaseBinding,
                binding_path,
                path=f"root_binding[{root}]",
            )
            root_binding.validate(f"root_binding[{root}]")
            if root_binding != binding or binding_bytes != expected_binding_bytes:
                raise ValueError("root binding bytes/digest drifted")
        except (OSError, ValueError, TypeError, SchemaError) as error:
            invalid.append({"path": str(binding_path), "error": str(error)})
            continue
        for path in sorted(root.glob("*/case_evidence.json")):
            scanned_files += 1
            try:
                row = load_json_dataclass(
                    FlexibleMeshReleaseCaseEvidence,
                    path,
                    path=f"case_evidence[{path.parent.name}]",
                )
            except (OSError, ValueError, TypeError, SchemaError) as error:
                invalid.append({"path": str(path), "error": str(error)})
                continue
            expected = case_by_id.get(row.case.id)
            if expected is None or row.case != expected:
                drifted.append({"path": str(path), "case_id": row.case.id})
                continue
            try:
                row.validate_against(binding)
            except (SchemaError, ValueError, TypeError) as error:
                invalid.append({"path": str(path), "error": str(error)})
                continue
            previous = observed.get(row.case.id)
            if previous is not None:
                duplicates.append({
                    "case_id": row.case.id,
                    "first": str(previous[1]),
                    "duplicate": str(path),
                })
                continue
            if row.case.family not in _ROOT_FAMILIES[root_index]:
                drifted.append({
                    "path": str(path),
                    "case_id": row.case.id,
                    "error": "case is owned by a different workload root",
                })
                continue
            observed[row.case.id] = (row, path)
    missing = tuple(case.id for case in cases if case.id not in observed)
    missing_by_family = {
        family.value: sum(
            case.id in missing for case in cases if case.family is family
        )
        for family in FlexibleMeshReleaseFamily
    }
    observed_by_family = {
        family.value: sum(
            case.family is family for case_id, (row, _path) in observed.items()
            for case in (case_by_id[case_id],)
        )
        for family in FlexibleMeshReleaseFamily
    }
    clean = not any((missing_roots, invalid, drifted, duplicates, missing))
    audit = {
        "schema_version": "flexible_mesh_release_missing_audit/v1",
        "clean": clean,
        "binding_id": binding.id,
        "binding_digest": binding.digest,
        "runtime_roots": tuple(str(root) for root in runtime_roots),
        "root_binding_sha256": root_binding_sha256,
        "expected_cases": len(cases),
        "expected_executions": 2 * len(cases),
        "scanned_case_files": scanned_files,
        "valid_unique_cases": len(observed),
        "valid_executions": 2 * len(observed),
        "observed_by_family": observed_by_family,
        "missing_by_family": missing_by_family,
        "missing_root_count": len(missing_roots),
        "missing_roots": tuple(missing_roots),
        "missing_case_count": len(missing),
        "missing_case_ids": missing,
        "duplicate_count": len(duplicates),
        "duplicates": tuple(duplicates),
        "drifted_count": len(drifted),
        "drifted": tuple(drifted),
        "invalid_count": len(invalid),
        "invalid": tuple(invalid),
        "completion_generated": False,
    }
    ordered = tuple(observed[case.id][0] for case in cases if case.id in observed)
    return audit, ordered


def _summary(completion) -> dict[str, Any]:
    rows = completion.evidence_matrix.case_evidence
    executions = tuple(execution for row in rows for execution in row.executions)
    family_counts = Counter(row.case.family.value for row in rows)
    capacity = tuple(execution.capacity for execution in executions)
    capacity_limits = {
        "rank_count": 100,
        "peak_sessions_per_core_per_wave": (
            FLEXIBLE_MESH_RELEASE_MAX_SESSIONS_PER_CORE_PER_WAVE
        ),
        "symbolic_record_count": FLEXIBLE_MESH_RELEASE_MAX_RECORDS,
        "exact_record_count": FLEXIBLE_MESH_RELEASE_MAX_RECORDS,
        "linked_manifest_file_bytes": FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES,
        "artifact_file_bytes": FLEXIBLE_MESH_RELEASE_MAX_ARTIFACT_BYTES,
        "max_runtime_core_id": FLEXIBLE_MESH_RELEASE_MAX_RUNTIME_CORE_ID,
        "transport_tag_count": FLEXIBLE_MESH_RELEASE_MAX_TRANSPORT_TAGS,
    }
    capacity_max = {
        "rank_count": max(item.rank_count for item in capacity),
        "peak_sessions_per_core_per_wave": max(
            item.peak_sessions_per_core_per_wave for item in capacity
        ),
        "symbolic_record_count": max(
            item.symbolic_record_count for item in capacity
        ),
        "exact_record_count": max(item.exact_record_count for item in capacity),
        "linked_manifest_file_bytes": max(
            item.linked_manifest_file_bytes for item in capacity
        ),
        "artifact_file_bytes": max(item.artifact_file_bytes for item in capacity),
        "max_runtime_core_id": max(item.max_runtime_core_id for item in capacity),
        "transport_tag_count": max(item.transport_tag_count for item in capacity),
    }
    capacity_by_family = {}
    for family in FlexibleMeshReleaseFamily:
        items = tuple(
            execution.capacity
            for row in rows if row.case.family is family
            for execution in row.executions
        )
        capacity_by_family[family.value] = {
            name: max(getattr(item, name) for item in items)
            for name in capacity_max
        }
    residual_nonzero = sum(
        any(getattr(item.residual, field) != 0 for field in item.residual.__dataclass_fields__)
        for item in executions
    )
    stage_failure_count = sum(
        code != 0 for item in executions for _stage, code in item.stage_exit_codes
    )
    failed_case_ids = list(sorted(
        row.case.id for row in rows
        if not row.runtime_verified or not row.repeatability_verified
    ))
    phase_counts = Counter(
        phase.value for item in executions for phase in item.program_io_phases
    )
    stage_counts = {
        stage.value: {
            "success": sum(
                code == 0 for item in executions
                for observed, code in item.stage_exit_codes if observed is stage
            ),
            "failure": sum(
                code != 0 for item in executions
                for observed, code in item.stage_exit_codes if observed is stage
            ),
        }
        for stage, _code in executions[0].stage_exit_codes
    }
    residual_fields = tuple(executions[0].residual.__dataclass_fields__)
    residual_max = {
        field: max(getattr(item.residual, field) for item in executions)
        for field in residual_fields
    }
    residual_nonzero_counts = {
        field: sum(getattr(item.residual, field) != 0 for item in executions)
        for field in residual_fields
    }
    marker_counts = Counter(
        marker.value for item in executions for marker in item.completion_markers
    )
    return {
        "schema_version": "flexible_mesh_release_summary/v1",
        "capacity_policy_version": FLEXIBLE_MESH_RELEASE_CAPACITY_POLICY_VERSION,
        "completion_id": completion.id,
        "binding_id": completion.evidence_matrix.release_binding.id,
        "binding_digest": completion.evidence_matrix.release_binding.digest,
        "case_count": len(rows),
        "execution_count": len(executions),
        "family_case_counts": {
            family.value: family_counts[family.value]
            for family in FlexibleMeshReleaseFamily
        },
        "runtime_verified_cases": sum(row.runtime_verified for row in rows),
        "repeatability_verified_cases": sum(
            row.repeatability_verified for row in rows
        ),
        "runtime_verified_cases_by_family": {
            family.value: sum(
                row.runtime_verified for row in rows if row.case.family is family
            )
            for family in FlexibleMeshReleaseFamily
        },
        "repeatability_verified_cases_by_family": {
            family.value: sum(
                row.repeatability_verified for row in rows if row.case.family is family
            )
            for family in FlexibleMeshReleaseFamily
        },
        "failed_case_ids": failed_case_ids,
        "failure_count": stage_failure_count + residual_nonzero,
        "stage_failure_count": stage_failure_count,
        "residual_nonzero_execution_count": residual_nonzero,
        "runtime_stage_counts": stage_counts,
        "program_io_phase_counts": dict(sorted(phase_counts.items())),
        "residual_max": residual_max,
        "residual_nonzero_counts": residual_nonzero_counts,
        "completion_marker_execution_counts": dict(sorted(marker_counts.items())),
        "capacity_limits": capacity_limits,
        "capacity_max": capacity_max,
        "capacity_headroom": {
            name: limit - capacity_max[name]
            for name, limit in capacity_limits.items()
        },
        "capacity_max_by_family": capacity_by_family,
        "hardware_config_sha256_count": len({
            item.hardware_config_sha256 for item in executions
        }),
        "simulation_config_sha256": tuple(sorted({
            item.simulation_config_sha256 for item in executions
        })),
        "mapping_config_sha256": tuple(sorted({
            item.mapping_config_sha256 for item in executions
        })),
        "artifact_sha256_count": len({item.artifact_sha256 for item in executions}),
        "marker_digest_count": len({item.marker_digest for item in executions}),
        "tool_sha256": {
            tool.kind.value: tool.sha256
            for tool in completion.evidence_matrix.release_binding.tools
        },
        "timing_execution_only": all(
            item.timing_execution and not item.functional_execution
            for item in executions
        ),
        **({
            "validation_scope": completion.validation_scope.value,
            "exhaustive_runtime": completion.exhaustive_runtime,
            "tested_meshes": tuple(
                f"{mesh.rows}x{mesh.columns}" for mesh in completion.tested_meshes
            ),
            "exhaustive_rect_runtime_complete": (
                completion.exhaustive_rect_runtime_complete
            ),
            "dense_train_rect_complete": completion.dense_train_rect_complete,
            "moe_infer_rect_complete": completion.moe_infer_rect_complete,
            "moe_train_rect_complete": completion.moe_train_rect_complete,
            "meshslice_all_rect_complete": completion.meshslice_all_rect_complete,
            "workload_contract_complete": completion.workload_contract_complete,
            "flexible_mesh_workloads_complete": (
                completion.flexible_mesh_workloads_complete
            ),
        } if hasattr(completion, "validation_scope") else {}),
    }


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    binding = load_json_dataclass(
        FlexibleMeshReleaseBinding, args.binding, path="release_binding",
    )
    cases = generate_flexible_mesh_release_cases(
        trace_model_digests=release_trace_model_digests(),
        runtime_profile_version=binding.runtime_profile_version,
    )
    validation_scope = getattr(args, "validation_scope", "exhaustive")
    if validation_scope == FlexibleMeshValidationScope.REPRESENTATIVE.value:
        cases = select_flexible_mesh_representative_cases(cases)
    roots = tuple(path.resolve() for path in args.runtime_root)
    audit, _ordered = audit_flexible_mesh_release_roots(
        binding=binding, cases=cases, runtime_roots=roots,
    )
    audit["validation_scope"] = validation_scope
    audit["exhaustive_runtime"] = validation_scope == "exhaustive"
    if not audit["clean"]:
        audit_path = args.output_dir / "missing_audit.json"
        audit_path.write_text(canonical_json(audit) + "\n", encoding="utf-8")
        print(
            f"incomplete: valid={audit['valid_unique_cases']}/{len(cases)} "
            f"missing={audit['missing_case_count']} duplicate={audit['duplicate_count']} "
            f"drifted={audit['drifted_count']} invalid={audit['invalid_count']} "
            f"audit={audit_path}",
            flush=True,
        )
        return 2
    representative = validation_scope == FlexibleMeshValidationScope.REPRESENTATIVE.value
    completion = (
        aggregate_flexible_mesh_representative_roots(
            binding=binding, cases=cases, runtime_roots=roots,
        )
        if representative
        else aggregate_flexible_mesh_release_roots(
            binding=binding, cases=cases, runtime_roots=roots,
        )
    )
    report_path = args.output_dir / "completion.json"
    state_path = args.output_dir / "completion.toml"
    writer = (
        write_flexible_mesh_representative_completion_report
        if representative
        else write_flexible_mesh_completion_report
    )
    writer(
        completion, report_path=report_path, state_path=state_path,
    )
    summary_path = args.output_dir / "release_summary.json"
    summary_path.write_text(canonical_json(_summary(completion)) + "\n", encoding="utf-8")
    print(
        f"complete: scope={validation_scope} cases={len(cases)} "
        f"executions={2 * len(cases)} report={report_path} state={state_path} "
        f"summary={summary_path}",
        flush=True,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument(
        "--runtime-root", required=True, action="append", type=Path,
        help="repeat exactly three times: Dense, MoE, and MeshSlice roots",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--validation-scope",
        choices=("exhaustive", FlexibleMeshValidationScope.REPRESENTATIVE.value),
        default="exhaustive",
    )
    args = parser.parse_args()
    if len(args.runtime_root) != 3 or len(set(args.runtime_root)) != 3:
        parser.error("--runtime-root must name exactly three distinct roots")
    if not args.binding.is_file():
        parser.error("--binding is not a file")
    return args


if __name__ == "__main__":
    raise SystemExit(run(_parse_args()))
