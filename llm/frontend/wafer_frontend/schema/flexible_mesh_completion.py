"""Fail-closed completion derivation for the flexible-Mesh release matrix."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id
from .flexible_mesh_release import (
    FLEXIBLE_MESH_RELEASE_CASE_COUNT,
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseFamily,
    validate_flexible_mesh_release_cases,
)
from .serde import canonical_digest


FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_completion/v1alpha1"
)


def _validate_sha256(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


class FlexibleMeshContractCheck(str, Enum):
    SCHEMA = "schema"
    AXIS = "axis"
    GROUP = "group"
    CAPACITY = "capacity"
    FALLBACK = "fallback"
    RUNTIME_EVIDENCE = "runtime_evidence"
    STABLE_IDS = "stable_ids"


@dataclass(frozen=True, slots=True)
class FlexibleMeshContractEvidence:
    schema_version: str
    id: str
    check: FlexibleMeshContractCheck
    evidence_digest: str
    verifier_digest: str
    binding_id: str
    binding_digest: str

    @classmethod
    def create(
        cls,
        *,
        check: FlexibleMeshContractCheck,
        evidence_digest: str,
        verifier_digest: str,
        binding: FlexibleMeshReleaseBinding,
    ) -> "FlexibleMeshContractEvidence":
        semantic = {
            "check": check,
            "evidence_digest": evidence_digest,
            "verifier_digest": verifier_digest,
            "binding_id": binding.id,
            "binding_digest": binding.digest,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_contract_evidence",
                semantic,
                schema_version=FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate_against(binding)
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate_against(
        self,
        binding: FlexibleMeshReleaseBinding,
        path: str = "flexible_mesh_contract_evidence",
    ) -> None:
        if self.schema_version != FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.check) is not FlexibleMeshContractCheck:
            raise SchemaError("must be a contract check", path=f"{path}.check")
        _validate_sha256(self.evidence_digest, f"{path}.evidence_digest")
        _validate_sha256(self.verifier_digest, f"{path}.verifier_digest")
        binding.validate(f"{path}.binding")
        if self.binding_id != binding.id or self.binding_digest != binding.digest:
            raise SchemaError("tool/config binding drifted", path=path)
        expected_id = stable_artifact_id(
            "flexible_mesh_contract_evidence",
            self._semantic_key(),
            schema_version=FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable contract evidence id", path=f"{path}.id")


_CONTRACT_VERIFIER_VERSION = "flexible_mesh_contract_derivation/v1"


def derive_flexible_mesh_contract_evidence(
    *,
    binding: FlexibleMeshReleaseBinding,
    cases: tuple[FlexibleMeshReleaseCase, ...],
    case_evidence: tuple[FlexibleMeshReleaseCaseEvidence, ...],
) -> tuple[FlexibleMeshContractEvidence, ...]:
    """Recompute contract receipts from the exact runtime matrix."""

    binding.validate("binding")
    validate_flexible_mesh_release_cases(
        cases, binding.runtime_profile_version, "cases"
    )
    if (
        len(case_evidence) != len(cases)
        or tuple(row.case.id for row in case_evidence)
        != tuple(case.id for case in cases)
    ):
        raise SchemaError(
            "case evidence does not exactly match release cases",
            path="case_evidence",
        )
    for index, row in enumerate(case_evidence):
        row.validate_against(binding, f"case_evidence[{index}]")
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


@dataclass(frozen=True, slots=True)
class FlexibleMeshCompletionEvidenceMatrix:
    schema_version: str
    id: str
    release_binding: FlexibleMeshReleaseBinding
    release_cases: tuple[FlexibleMeshReleaseCase, ...]
    case_evidence: tuple[FlexibleMeshReleaseCaseEvidence, ...]
    contract_evidence: tuple[FlexibleMeshContractEvidence, ...]

    @classmethod
    def create(
        cls,
        *,
        release_binding: FlexibleMeshReleaseBinding,
        release_cases: tuple[FlexibleMeshReleaseCase, ...],
        case_evidence: tuple[FlexibleMeshReleaseCaseEvidence, ...],
        contract_evidence: tuple[FlexibleMeshContractEvidence, ...],
    ) -> "FlexibleMeshCompletionEvidenceMatrix":
        semantic = {
            "release_binding": release_binding,
            "release_cases": release_cases,
            "case_evidence": case_evidence,
            "contract_evidence": contract_evidence,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_completion_evidence_matrix",
                semantic,
                schema_version=FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "flexible_mesh_completion_evidence_matrix") -> None:
        if self.schema_version != FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.release_binding) is not FlexibleMeshReleaseBinding:
            raise SchemaError("must be a release binding", path=f"{path}.release_binding")
        self.release_binding.validate(f"{path}.release_binding")
        validate_flexible_mesh_release_cases(
            self.release_cases,
            self.release_binding.runtime_profile_version,
            f"{path}.release_cases",
        )
        if type(self.case_evidence) is not tuple or len(self.case_evidence) != FLEXIBLE_MESH_RELEASE_CASE_COUNT:
            raise SchemaError("must contain exactly 600 case evidence rows", path=f"{path}.case_evidence")
        if tuple(evidence.case.id for evidence in self.case_evidence) != tuple(
            case.id for case in self.release_cases
        ):
            raise SchemaError(
                "case evidence must exactly match release cases in canonical order",
                path=f"{path}.case_evidence",
            )
        evidence_ids: set[str] = set()
        for index, evidence in enumerate(self.case_evidence):
            if type(evidence) is not FlexibleMeshReleaseCaseEvidence:
                raise SchemaError("must be case evidence", path=f"{path}.case_evidence[{index}]")
            if evidence.id in evidence_ids:
                raise SchemaError("duplicate case evidence id", path=f"{path}.case_evidence[{index}].id")
            evidence_ids.add(evidence.id)
            evidence.validate_against(
                self.release_binding,
                f"{path}.case_evidence[{index}]",
            )
            if not evidence.runtime_verified or not evidence.repeatability_verified:
                raise SchemaError("case is not runtime/repeatability verified", path=f"{path}.case_evidence[{index}]")
        if (
            type(self.contract_evidence) is not tuple
            or tuple(receipt.check for receipt in self.contract_evidence)
            != tuple(FlexibleMeshContractCheck)
        ):
            raise SchemaError(
                "contract evidence must contain all checks once in canonical order",
                path=f"{path}.contract_evidence",
            )
        for index, receipt in enumerate(self.contract_evidence):
            if type(receipt) is not FlexibleMeshContractEvidence:
                raise SchemaError("must be contract evidence", path=f"{path}.contract_evidence[{index}]")
            receipt.validate_against(
                self.release_binding,
                f"{path}.contract_evidence[{index}]",
            )
        expected_contract_evidence = derive_flexible_mesh_contract_evidence(
            binding=self.release_binding,
            cases=self.release_cases,
            case_evidence=self.case_evidence,
        )
        if self.contract_evidence != expected_contract_evidence:
            raise SchemaError(
                "contract evidence is not reproducible from the runtime matrix",
                path=f"{path}.contract_evidence",
            )
        expected_id = stable_artifact_id(
            "flexible_mesh_completion_evidence_matrix",
            self._semantic_key(),
            schema_version=FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable evidence matrix id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMeshCompletion:
    """Read-only completion view; target states are derived properties only."""

    schema_version: str
    id: str
    evidence_matrix: FlexibleMeshCompletionEvidenceMatrix

    def _semantic_key(self) -> dict[str, object]:
        return {"evidence_matrix": self.evidence_matrix}

    def _matrix_valid(self) -> bool:
        try:
            if self.schema_version != FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION:
                return False
            self.evidence_matrix.validate()
            expected_id = stable_artifact_id(
                "flexible_mesh_completion",
                self._semantic_key(),
                schema_version=FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION,
            )
            return self.id == expected_id
        except (SchemaError, AttributeError, TypeError, ValueError):
            return False

    def _family_complete(self, family: FlexibleMeshReleaseFamily) -> bool:
        if not self._matrix_valid():
            return False
        rows = tuple(
            evidence
            for evidence in self.evidence_matrix.case_evidence
            if evidence.case.family is family
        )
        return (
            len(rows) == 100
            and {(row.case.mesh.rows, row.case.mesh.columns) for row in rows}
            == {(rows, columns) for rows in range(1, 11) for columns in range(1, 11)}
            and all(row.runtime_verified and row.repeatability_verified for row in rows)
        )

    @property
    def dense_train_rect_complete(self) -> bool:
        return self._family_complete(FlexibleMeshReleaseFamily.DENSE_TRAIN)

    @property
    def moe_infer_rect_complete(self) -> bool:
        return self._family_complete(FlexibleMeshReleaseFamily.MOE_INFERENCE)

    @property
    def moe_train_rect_complete(self) -> bool:
        return self._family_complete(FlexibleMeshReleaseFamily.MOE_TRAIN)

    @property
    def meshslice_all_rect_complete(self) -> bool:
        return all(
            self._family_complete(family)
            for family in (
                FlexibleMeshReleaseFamily.MESHSLICE_AG,
                FlexibleMeshReleaseFamily.MESHSLICE_RS_FALLBACK,
                FlexibleMeshReleaseFamily.MESHSLICE_AR_FALLBACK,
            )
        )

    @property
    def workload_contract_complete(self) -> bool:
        return self._matrix_valid() and tuple(
            receipt.check for receipt in self.evidence_matrix.contract_evidence
        ) == tuple(FlexibleMeshContractCheck)

    @property
    def flexible_mesh_workloads_complete(self) -> bool:
        return (
            self.workload_contract_complete
            and self.meshslice_all_rect_complete
            and self.dense_train_rect_complete
            and self.moe_infer_rect_complete
            and self.moe_train_rect_complete
        )

    def validate(self, path: str = "flexible_mesh_completion") -> None:
        if not self._matrix_valid():
            raise SchemaError("completion report is forged or incomplete", path=path)
        if not all(
            (
                self.dense_train_rect_complete,
                self.moe_infer_rect_complete,
                self.moe_train_rect_complete,
                self.flexible_mesh_workloads_complete,
            )
        ):
            raise SchemaError("release matrix is incomplete", path=path)


def derive_flexible_mesh_completion(
    evidence_matrix: FlexibleMeshCompletionEvidenceMatrix,
) -> FlexibleMeshCompletion:
    if type(evidence_matrix) is not FlexibleMeshCompletionEvidenceMatrix:
        raise SchemaError("must be a completion evidence matrix", path="evidence_matrix")
    evidence_matrix.validate()
    semantic = {"evidence_matrix": evidence_matrix}
    result = FlexibleMeshCompletion(
        schema_version=FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION,
        id=stable_artifact_id(
            "flexible_mesh_completion",
            semantic,
            schema_version=FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION,
        ),
        evidence_matrix=evidence_matrix,
    )
    result.validate()
    return result


__all__ = [
    "FLEXIBLE_MESH_COMPLETION_SCHEMA_VERSION",
    "FlexibleMeshCompletion",
    "FlexibleMeshCompletionEvidenceMatrix",
    "FlexibleMeshContractCheck",
    "FlexibleMeshContractEvidence",
    "derive_flexible_mesh_contract_evidence",
    "derive_flexible_mesh_completion",
]
