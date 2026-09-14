"""Aggregate strict release evidence and emit derived completion states."""

from __future__ import annotations

import hashlib
from pathlib import Path

from llm.frontend.wafer_frontend.schema.flexible_mesh_completion import (
    FlexibleMeshCompletion,
    FlexibleMeshCompletionEvidenceMatrix,
    FlexibleMeshContractCheck,
    FlexibleMeshContractEvidence,
    derive_flexible_mesh_completion,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_representative_completion import (
    FlexibleMeshRepresentativeCompletion,
    FlexibleMeshRepresentativeEvidenceMatrix,
    derive_flexible_mesh_representative_completion,
    derive_flexible_mesh_representative_contract_evidence,
    validate_flexible_mesh_representative_cases,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseCaseEvidence,
    validate_flexible_mesh_release_cases,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    load_json_dataclass,
)


_CONTRACT_VERIFIER_VERSION = "flexible_mesh_contract_derivation/v1"


def derive_flexible_mesh_contract_evidence(
    *,
    binding: FlexibleMeshReleaseBinding,
    cases: tuple[FlexibleMeshReleaseCase, ...],
    case_evidence: tuple[FlexibleMeshReleaseCaseEvidence, ...],
) -> tuple[FlexibleMeshContractEvidence, ...]:
    """Derive all contract receipts from the same strict runtime matrix."""

    binding.validate("binding")
    validate_flexible_mesh_release_cases(
        cases, binding.runtime_profile_version, "cases"
    )
    if (
        len(case_evidence) != len(cases)
        or tuple(row.case.id for row in case_evidence)
        != tuple(case.id for case in cases)
    ):
        raise ValueError("case evidence does not exactly match release cases")
    for row in case_evidence:
        row.validate_against(binding)
    payloads = {
        FlexibleMeshContractCheck.SCHEMA: tuple(case.digest for case in cases),
        FlexibleMeshContractCheck.AXIS: tuple(
            (
                row.case.mesh.digest,
                row.executions[0].spec_digest,
                row.executions[1].spec_digest,
            )
            for row in case_evidence
        ),
        FlexibleMeshContractCheck.GROUP: tuple(
            (
                row.case.mesh.rows,
                row.case.mesh.columns,
                row.executions[0].plan_digest,
                row.executions[1].plan_digest,
            )
            for row in case_evidence
        ),
        FlexibleMeshContractCheck.CAPACITY: tuple(
            execution.capacity
            for row in case_evidence
            for execution in row.executions
        ),
        FlexibleMeshContractCheck.FALLBACK: tuple(
            (case.family, case.operation, case.selected_baseline)
            for case in cases
        ),
        FlexibleMeshContractCheck.RUNTIME_EVIDENCE: tuple(
            row.id for row in case_evidence
        ),
        FlexibleMeshContractCheck.STABLE_IDS: tuple(
            (
                row.executions[0].spec_digest,
                row.executions[1].spec_digest,
                row.executions[0].plan_digest,
                row.executions[1].plan_digest,
                row.executions[0].manifest_digest,
                row.executions[1].manifest_digest,
                row.executions[0].hardware_config_sha256,
                row.executions[1].hardware_config_sha256,
                row.executions[0].simulation_config_sha256,
                row.executions[1].simulation_config_sha256,
                row.executions[0].mapping_config_sha256,
                row.executions[1].mapping_config_sha256,
                row.executions[0].artifact_sha256,
                row.executions[1].artifact_sha256,
            )
            for row in case_evidence
        ),
    }
    return tuple(
        FlexibleMeshContractEvidence.create(
            check=check,
            evidence_digest=canonical_digest(payloads[check]),
            verifier_digest=hashlib.sha256(
                f"{_CONTRACT_VERIFIER_VERSION}:{check.value}".encode("utf-8")
            ).hexdigest(),
            binding=binding,
        )
        for check in FlexibleMeshContractCheck
    )

def aggregate_flexible_mesh_release(
    *,
    binding: FlexibleMeshReleaseBinding,
    cases: tuple[FlexibleMeshReleaseCase, ...],
    case_evidence: tuple[FlexibleMeshReleaseCaseEvidence, ...],
    contract_evidence: tuple[FlexibleMeshContractEvidence, ...] | None = None,
) -> FlexibleMeshCompletion:
    """Fail closed unless the exact 600-case/1200-execution matrix is present."""

    receipts = (
        derive_flexible_mesh_contract_evidence(
            binding=binding,
            cases=cases,
            case_evidence=case_evidence,
        )
        if contract_evidence is None
        else contract_evidence
    )

    matrix = FlexibleMeshCompletionEvidenceMatrix.create(
        release_binding=binding,
        release_cases=cases,
        case_evidence=case_evidence,
        contract_evidence=receipts,
    )
    return derive_flexible_mesh_completion(matrix)


def aggregate_flexible_mesh_release_shards(
    *,
    binding: FlexibleMeshReleaseBinding,
    cases: tuple[FlexibleMeshReleaseCase, ...],
    runtime_root: Path,
    shard_count: int,
) -> FlexibleMeshCompletion:
    """Load strict shard files, reject gaps/duplicates, and aggregate canonically."""

    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("shard_count must be positive")
    expected_ids = tuple(case.id for case in cases)
    expected_id_set = set(expected_ids)
    evidence_by_case: dict[str, FlexibleMeshReleaseCaseEvidence] = {}
    for shard_index in range(shard_count):
        shard_path = (
            runtime_root
            / f"release_shard_{shard_index}_of_{shard_count}.json"
        )
        if not shard_path.is_file():
            raise ValueError(f"missing release shard: {shard_path}")
        rows = load_json_dataclass(
            tuple[FlexibleMeshReleaseCaseEvidence, ...],
            shard_path,
            path=f"release_shard[{shard_index}]",
        )
        for row in rows:
            row.validate_against(binding)
            if row.case.id not in expected_id_set:
                raise ValueError(f"unexpected release case: {row.case.id}")
            if row.case.id in evidence_by_case:
                raise ValueError(f"duplicate release case: {row.case.id}")
            evidence_by_case[row.case.id] = row
    missing = tuple(case_id for case_id in expected_ids if case_id not in evidence_by_case)
    if missing:
        raise ValueError(f"missing {len(missing)} release cases; first={missing[0]}")
    ordered = tuple(evidence_by_case[case_id] for case_id in expected_ids)
    return aggregate_flexible_mesh_release(
        binding=binding, cases=cases, case_evidence=ordered
    )


def aggregate_flexible_mesh_release_roots(
    *,
    binding: FlexibleMeshReleaseBinding,
    cases: tuple[FlexibleMeshReleaseCase, ...],
    runtime_roots: tuple[Path, ...],
) -> FlexibleMeshCompletion:
    """Merge complete per-case checkpoints from workload-specific roots."""

    if type(runtime_roots) is not tuple or not runtime_roots:
        raise ValueError("runtime_roots must be a non-empty tuple")
    case_by_id = {case.id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("release cases contain duplicate IDs")
    evidence_by_case: dict[str, FlexibleMeshReleaseCaseEvidence] = {}
    for runtime_root in runtime_roots:
        if not runtime_root.is_dir():
            raise ValueError(f"missing runtime root: {runtime_root}")
        for evidence_path in sorted(runtime_root.glob("*/case_evidence.json")):
            row = load_json_dataclass(
                FlexibleMeshReleaseCaseEvidence,
                evidence_path,
                path=f"case_evidence[{evidence_path.parent.name}]",
            )
            expected_case = case_by_id.get(row.case.id)
            if expected_case is None or row.case != expected_case:
                raise ValueError(f"unexpected/drifted release case: {row.case.id}")
            row.validate_against(binding)
            if row.case.id in evidence_by_case:
                raise ValueError(f"duplicate release case: {row.case.id}")
            evidence_by_case[row.case.id] = row
    missing = tuple(case.id for case in cases if case.id not in evidence_by_case)
    if missing:
        raise ValueError(f"missing {len(missing)} release cases; first={missing[0]}")
    ordered = tuple(evidence_by_case[case.id] for case in cases)
    return aggregate_flexible_mesh_release(
        binding=binding, cases=cases, case_evidence=ordered
    )


def aggregate_flexible_mesh_representative_release(
    *,
    binding: FlexibleMeshReleaseBinding,
    cases: tuple[FlexibleMeshReleaseCase, ...],
    case_evidence: tuple[FlexibleMeshReleaseCaseEvidence, ...],
) -> FlexibleMeshRepresentativeCompletion:
    """Derive completion only for the exact, explicitly non-exhaustive scope."""

    validate_flexible_mesh_representative_cases(
        cases, binding.runtime_profile_version, "cases"
    )
    receipts = derive_flexible_mesh_representative_contract_evidence(
        binding=binding,
        cases=cases,
        case_evidence=case_evidence,
    )
    matrix = FlexibleMeshRepresentativeEvidenceMatrix.create(
        release_binding=binding,
        release_cases=cases,
        case_evidence=case_evidence,
        contract_evidence=receipts,
    )
    return derive_flexible_mesh_representative_completion(matrix)


def aggregate_flexible_mesh_representative_roots(
    *,
    binding: FlexibleMeshReleaseBinding,
    cases: tuple[FlexibleMeshReleaseCase, ...],
    runtime_roots: tuple[Path, ...],
) -> FlexibleMeshRepresentativeCompletion:
    """Merge the representative case set while retaining strict row validation."""

    case_by_id = {case.id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("representative cases contain duplicate IDs")
    evidence_by_case: dict[str, FlexibleMeshReleaseCaseEvidence] = {}
    for runtime_root in runtime_roots:
        if not runtime_root.is_dir():
            raise ValueError(f"missing runtime root: {runtime_root}")
        for evidence_path in sorted(runtime_root.glob("*/case_evidence.json")):
            row = load_json_dataclass(
                FlexibleMeshReleaseCaseEvidence,
                evidence_path,
                path=f"case_evidence[{evidence_path.parent.name}]",
            )
            expected_case = case_by_id.get(row.case.id)
            if expected_case is None:
                continue
            if row.case != expected_case:
                raise ValueError(f"drifted release case: {row.case.id}")
            row.validate_against(binding)
            if row.case.id in evidence_by_case:
                raise ValueError(f"duplicate release case: {row.case.id}")
            evidence_by_case[row.case.id] = row
    missing = tuple(case.id for case in cases if case.id not in evidence_by_case)
    if missing:
        raise ValueError(f"missing {len(missing)} representative cases; first={missing[0]}")
    ordered = tuple(evidence_by_case[case.id] for case in cases)
    return aggregate_flexible_mesh_representative_release(
        binding=binding,
        cases=cases,
        case_evidence=ordered,
    )


def completion_state_toml(completion: FlexibleMeshCompletion) -> str:
    """Render only derived state; no caller can supply replacement booleans."""

    completion.validate("completion")
    values = (
        ("dense_train_rect_complete", completion.dense_train_rect_complete),
        ("moe_infer_rect_complete", completion.moe_infer_rect_complete),
        ("moe_train_rect_complete", completion.moe_train_rect_complete),
        (
            "meshslice_all_rect_complete",
            completion.meshslice_all_rect_complete,
        ),
        (
            "workload_contract_complete",
            completion.workload_contract_complete,
        ),
        (
            "flexible_mesh_workloads_complete",
            completion.flexible_mesh_workloads_complete,
        ),
    )
    return "".join(
        f"{name} = {'true' if value else 'false'}\n" for name, value in values
    )


def representative_completion_state_toml(
    completion: FlexibleMeshRepresentativeCompletion,
) -> str:
    completion.validate("completion")
    meshes = ", ".join(
        f'"{mesh.rows}x{mesh.columns}"' for mesh in completion.tested_meshes
    )
    values = (
        ("dense_train_rect_complete", completion.dense_train_rect_complete),
        ("moe_infer_rect_complete", completion.moe_infer_rect_complete),
        ("moe_train_rect_complete", completion.moe_train_rect_complete),
        ("meshslice_all_rect_complete", completion.meshslice_all_rect_complete),
        ("workload_contract_complete", completion.workload_contract_complete),
        ("flexible_mesh_workloads_complete", completion.flexible_mesh_workloads_complete),
        ("exhaustive_rect_runtime_complete", completion.exhaustive_rect_runtime_complete),
    )
    return (
        'validation_scope = "representative"\n'
        "exhaustive_runtime = false\n"
        f"tested_meshes = [{meshes}]\n"
        + "".join(
            f"{name} = {'true' if value else 'false'}\n"
            for name, value in values
        )
    )


def write_flexible_mesh_representative_completion_report(
    completion: FlexibleMeshRepresentativeCompletion,
    *,
    report_path: Path,
    state_path: Path,
) -> None:
    completion.validate("completion")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(canonical_json(completion) + "\n", encoding="utf-8")
    state_path.write_text(
        representative_completion_state_toml(completion), encoding="utf-8"
    )


def write_flexible_mesh_completion_report(
    completion: FlexibleMeshCompletion,
    *,
    report_path: Path,
    state_path: Path,
) -> None:
    completion.validate("completion")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(canonical_json(completion) + "\n", encoding="utf-8")
    state_path.write_text(completion_state_toml(completion), encoding="utf-8")


__all__ = [
    "aggregate_flexible_mesh_release",
    "aggregate_flexible_mesh_release_roots",
    "aggregate_flexible_mesh_release_shards",
    "completion_state_toml",
    "derive_flexible_mesh_contract_evidence",
    "write_flexible_mesh_completion_report",
]
