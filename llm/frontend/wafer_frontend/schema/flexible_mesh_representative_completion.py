"""Fail-closed completion derivation for the representative flexible-Mesh scope."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id
from .flexible_mesh_completion import (
    FlexibleMeshContractCheck,
    FlexibleMeshContractEvidence,
)
from .flexible_mesh_release import (
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseCase,
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseFamily,
)
from .rect_mesh import RectMeshSpec
from .serde import canonical_digest


FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_representative_completion/v1alpha1"
)
FLEXIBLE_MESH_REPRESENTATIVE_SHAPES = (
    (1, 1),
    (1, 4),
    (4, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (10, 10),
)
FLEXIBLE_MESH_REPRESENTATIVE_CASE_COUNT = (
    len(FLEXIBLE_MESH_REPRESENTATIVE_SHAPES) * len(FlexibleMeshReleaseFamily)
)


class FlexibleMeshValidationScope(str, Enum):
    REPRESENTATIVE = "representative"


def representative_tested_meshes() -> tuple[RectMeshSpec, ...]:
    return tuple(
        RectMeshSpec(rows=rows, columns=columns)
        for rows, columns in FLEXIBLE_MESH_REPRESENTATIVE_SHAPES
    )


def select_flexible_mesh_representative_cases(
    cases: tuple[FlexibleMeshReleaseCase, ...],
) -> tuple[FlexibleMeshReleaseCase, ...]:
    """Select the exact representative matrix without changing case identities."""

    if type(cases) is not tuple:
        raise SchemaError("must be a tuple", path="release_cases")
    selected_shapes = set(FLEXIBLE_MESH_REPRESENTATIVE_SHAPES)
    selected = tuple(
        case for case in cases
        if type(case) is FlexibleMeshReleaseCase
        and (case.mesh.rows, case.mesh.columns) in selected_shapes
    )
    family_order = {
        family: index
        for index, family in enumerate(FlexibleMeshReleaseFamily)
    }
    shape_order = {
        shape: index
        for index, shape in enumerate(FLEXIBLE_MESH_REPRESENTATIVE_SHAPES)
    }
    result = tuple(
        sorted(
            selected,
            key=lambda case: (
                family_order[case.family],
                shape_order[(case.mesh.rows, case.mesh.columns)],
            ),
        )
    )
    validate_flexible_mesh_representative_cases(
        result,
        result[0].runtime_profile_version if result else "",
    )
    return result


def validate_flexible_mesh_representative_cases(
    cases: tuple[FlexibleMeshReleaseCase, ...],
    runtime_profile_version: str,
    path: str = "release_cases",
) -> None:
    if (
        type(cases) is not tuple
        or len(cases) != FLEXIBLE_MESH_REPRESENTATIVE_CASE_COUNT
    ):
        raise SchemaError("must contain exactly 48 representative cases", path=path)
    expected_order = tuple(
        (family, rows, columns)
        for family in FlexibleMeshReleaseFamily
        for rows, columns in FLEXIBLE_MESH_REPRESENTATIVE_SHAPES
    )
    actual_order: list[tuple[FlexibleMeshReleaseFamily, int, int]] = []
    ids: set[str] = set()
    family_digests: dict[FlexibleMeshReleaseFamily, str] = {}
    for index, case in enumerate(cases):
        item_path = f"{path}[{index}]"
        if type(case) is not FlexibleMeshReleaseCase:
            raise SchemaError("must be a release case", path=item_path)
        case.validate(item_path)
        if case.runtime_profile_version != runtime_profile_version:
            raise SchemaError("runtime profile drifted", path=item_path)
        if case.id in ids:
            raise SchemaError("duplicate release case id", path=f"{item_path}.id")
        ids.add(case.id)
        actual_order.append((case.family, case.mesh.rows, case.mesh.columns))
        previous = family_digests.setdefault(case.family, case.trace_model_digest)
        if previous != case.trace_model_digest:
            raise SchemaError("family trace/model digest drifted", path=item_path)
    if tuple(actual_order) != expected_order:
        raise SchemaError(
            "cases do not exactly cover the canonical representative matrix",
            path=path,
        )


_CONTRACT_VERIFIER_VERSION = "flexible_mesh_representative_contract_derivation/v1"


def derive_flexible_mesh_representative_contract_evidence(
    *,
    binding: FlexibleMeshReleaseBinding,
    cases: tuple[FlexibleMeshReleaseCase, ...],
    case_evidence: tuple[FlexibleMeshReleaseCaseEvidence, ...],
) -> tuple[FlexibleMeshContractEvidence, ...]:
    binding.validate("binding")
    validate_flexible_mesh_representative_cases(
        cases, binding.runtime_profile_version, "cases"
    )
    if type(case_evidence) is not tuple or len(case_evidence) != len(cases):
        raise SchemaError(
            "case evidence does not exactly match representative cases",
            path="case_evidence",
        )
    for index, row in enumerate(case_evidence):
        if type(row) is not FlexibleMeshReleaseCaseEvidence:
            raise SchemaError("must be case evidence", path=f"case_evidence[{index}]")
    if tuple(row.case.id for row in case_evidence) != tuple(case.id for case in cases):
        raise SchemaError(
            "case evidence does not exactly match representative cases",
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
            (case.family, case.operation, case.selected_baseline) for case in cases
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
class FlexibleMeshRepresentativeEvidenceMatrix:
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
    ) -> "FlexibleMeshRepresentativeEvidenceMatrix":
        semantic = {
            "release_binding": release_binding,
            "release_cases": release_cases,
            "case_evidence": case_evidence,
            "contract_evidence": contract_evidence,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_representative_evidence_matrix",
                semantic,
                schema_version=FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION,
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

    def validate(
        self, path: str = "flexible_mesh_representative_evidence_matrix"
    ) -> None:
        if self.schema_version != FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.release_binding) is not FlexibleMeshReleaseBinding:
            raise SchemaError("must be a release binding", path=f"{path}.release_binding")
        self.release_binding.validate(f"{path}.release_binding")
        validate_flexible_mesh_representative_cases(
            self.release_cases,
            self.release_binding.runtime_profile_version,
            f"{path}.release_cases",
        )
        if (
            type(self.case_evidence) is not tuple
            or len(self.case_evidence) != FLEXIBLE_MESH_REPRESENTATIVE_CASE_COUNT
        ):
            raise SchemaError(
                "case evidence must exactly match representative cases",
                path=f"{path}.case_evidence",
            )
        if any(
            type(row) is not FlexibleMeshReleaseCaseEvidence
            for row in self.case_evidence
        ):
            raise SchemaError(
                "must contain only case evidence",
                path=f"{path}.case_evidence",
            )
        if tuple(row.case.id for row in self.case_evidence) != tuple(
            case.id for case in self.release_cases
        ):
            raise SchemaError(
                "case evidence must exactly match representative cases",
                path=f"{path}.case_evidence",
            )
        ids: set[str] = set()
        for index, row in enumerate(self.case_evidence):
            row_path = f"{path}.case_evidence[{index}]"
            if row.id in ids:
                raise SchemaError("duplicate case evidence id", path=f"{row_path}.id")
            ids.add(row.id)
            row.validate_against(self.release_binding, row_path)
            if not row.runtime_verified or not row.repeatability_verified:
                raise SchemaError("case is not runtime/repeatability verified", path=row_path)
        if type(self.contract_evidence) is not tuple or any(
            type(receipt) is not FlexibleMeshContractEvidence
            for receipt in self.contract_evidence
        ):
            raise SchemaError(
                "must contain only contract evidence",
                path=f"{path}.contract_evidence",
            )
        if tuple(receipt.check for receipt in self.contract_evidence) != tuple(
            FlexibleMeshContractCheck
        ):
            raise SchemaError(
                "contract evidence must contain all checks once in canonical order",
                path=f"{path}.contract_evidence",
            )
        expected_contracts = derive_flexible_mesh_representative_contract_evidence(
            binding=self.release_binding,
            cases=self.release_cases,
            case_evidence=self.case_evidence,
        )
        if self.contract_evidence != expected_contracts:
            raise SchemaError(
                "contract evidence is not reproducible from representative evidence",
                path=f"{path}.contract_evidence",
            )
        expected_id = stable_artifact_id(
            "flexible_mesh_representative_evidence_matrix",
            self._semantic_key(),
            schema_version=FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable evidence matrix id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMeshRepresentativeCompletion:
    """Representative completion; every target state is a derived property."""

    schema_version: str
    id: str
    validation_scope: FlexibleMeshValidationScope
    exhaustive_runtime: bool
    tested_meshes: tuple[RectMeshSpec, ...]
    evidence_matrix: FlexibleMeshRepresentativeEvidenceMatrix

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def _matrix_valid(self) -> bool:
        try:
            if (
                self.schema_version
                != FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION
                or self.validation_scope is not FlexibleMeshValidationScope.REPRESENTATIVE
                or type(self.exhaustive_runtime) is not bool
                or self.exhaustive_runtime
                or self.tested_meshes != representative_tested_meshes()
            ):
                return False
            self.evidence_matrix.validate()
            return self.id == stable_artifact_id(
                "flexible_mesh_representative_completion",
                self._semantic_key(),
                schema_version=FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION,
            )
        except (SchemaError, AttributeError, TypeError, ValueError):
            return False

    def _family_complete(self, family: FlexibleMeshReleaseFamily) -> bool:
        if not self._matrix_valid():
            return False
        rows = tuple(
            row
            for row in self.evidence_matrix.case_evidence
            if row.case.family is family
        )
        return (
            len(rows) == len(FLEXIBLE_MESH_REPRESENTATIVE_SHAPES)
            and tuple((row.case.mesh.rows, row.case.mesh.columns) for row in rows)
            == FLEXIBLE_MESH_REPRESENTATIVE_SHAPES
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
            and self.dense_train_rect_complete
            and self.moe_infer_rect_complete
            and self.moe_train_rect_complete
            and self.meshslice_all_rect_complete
        )

    @property
    def exhaustive_rect_runtime_complete(self) -> bool:
        return False

    def validate(self, path: str = "flexible_mesh_representative_completion") -> None:
        if not self._matrix_valid() or not self.flexible_mesh_workloads_complete:
            raise SchemaError("representative completion is forged or incomplete", path=path)
        if self.exhaustive_rect_runtime_complete:
            raise SchemaError("representative scope cannot claim exhaustive runtime", path=path)


def derive_flexible_mesh_representative_completion(
    evidence_matrix: FlexibleMeshRepresentativeEvidenceMatrix,
) -> FlexibleMeshRepresentativeCompletion:
    if type(evidence_matrix) is not FlexibleMeshRepresentativeEvidenceMatrix:
        raise SchemaError("must be a representative evidence matrix", path="evidence_matrix")
    evidence_matrix.validate()
    semantic = {
        "validation_scope": FlexibleMeshValidationScope.REPRESENTATIVE,
        "exhaustive_runtime": False,
        "tested_meshes": representative_tested_meshes(),
        "evidence_matrix": evidence_matrix,
    }
    result = FlexibleMeshRepresentativeCompletion(
        schema_version=FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION,
        id=stable_artifact_id(
            "flexible_mesh_representative_completion",
            semantic,
            schema_version=FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION,
        ),
        **semantic,
    )
    result.validate()
    return result


__all__ = [
    "FLEXIBLE_MESH_REPRESENTATIVE_CASE_COUNT",
    "FLEXIBLE_MESH_REPRESENTATIVE_COMPLETION_SCHEMA_VERSION",
    "FLEXIBLE_MESH_REPRESENTATIVE_SHAPES",
    "FlexibleMeshRepresentativeCompletion",
    "FlexibleMeshRepresentativeEvidenceMatrix",
    "FlexibleMeshValidationScope",
    "derive_flexible_mesh_representative_completion",
    "derive_flexible_mesh_representative_contract_evidence",
    "representative_tested_meshes",
    "select_flexible_mesh_representative_cases",
    "validate_flexible_mesh_representative_cases",
]
