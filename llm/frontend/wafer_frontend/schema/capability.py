"""Machine-readable capability claims and their exact evidence closure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty
from .serde import canonical_digest


CASE_MATRIX_SCHEMA_VERSION = "wafer_frontend.case_matrix/v1alpha1"
CAPABILITY_MANIFEST_SCHEMA_VERSION = (
    "wafer_frontend.capability_manifest/v1alpha1"
)


class CapabilityStage(str, Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


class CapabilityStatus(str, Enum):
    UNSUPPORTED = "unsupported"
    SCHEMA_ONLY = "schema-only"
    UNIT_ONLY = "unit-only"
    E2E_TIMING = "e2e-timing"
    E2E_FUNCTIONAL = "e2e-functional"


_STATUS_RANK = {
    CapabilityStatus.UNSUPPORTED: 0,
    CapabilityStatus.SCHEMA_ONLY: 1,
    CapabilityStatus.UNIT_ONLY: 2,
    CapabilityStatus.E2E_TIMING: 3,
    CapabilityStatus.E2E_FUNCTIONAL: 4,
}


def capability_status_rank(status: CapabilityStatus) -> int:
    if type(status) is not CapabilityStatus:
        raise SchemaError("must be a CapabilityStatus", path="capability_status")
    return _STATUS_RANK[status]


def _validate_digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _validate_sorted_unique_strings(values: tuple[str, ...], path: str) -> None:
    if type(values) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    previous: str | None = None
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        validate_nonempty(value, item_path)
        if previous is not None and value <= previous:
            raise SchemaError(
                "values must be unique and strictly increasing", path=item_path
            )
        previous = value


def _validate_sorted_unique_by_key(
    values: tuple[object, ...], *, attribute: str, path: str
) -> None:
    if type(values) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    previous: str | None = None
    for index, value in enumerate(values):
        key = getattr(value, attribute, None)
        key_path = f"{path}[{index}].{attribute}"
        validate_nonempty(key, key_path)
        if previous is not None and key <= previous:
            raise SchemaError(
                f"{attribute} values must be unique and strictly increasing",
                path=key_path,
            )
        previous = key


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    key: str
    stage: CapabilityStage
    required_case_ids: tuple[str, ...]
    score_numerator: int
    score_denominator: int
    notes: tuple[str, ...]

    def validate(self, path: str = "capability_definition") -> None:
        validate_nonempty(self.key, f"{path}.key")
        if type(self.stage) is not CapabilityStage:
            raise SchemaError("must be a CapabilityStage", path=f"{path}.stage")
        _validate_sorted_unique_strings(
            self.required_case_ids, f"{path}.required_case_ids"
        )
        if not self.required_case_ids:
            raise SchemaError(
                "must contain at least one case", path=f"{path}.required_case_ids"
            )
        if type(self.score_numerator) is not int or self.score_numerator < 0:
            raise SchemaError(
                "must be a non-negative integer", path=f"{path}.score_numerator"
            )
        if type(self.score_denominator) is not int or self.score_denominator <= 0:
            raise SchemaError(
                "must be a positive integer", path=f"{path}.score_denominator"
            )
        if self.score_numerator > self.score_denominator:
            raise SchemaError(
                "must not exceed score_denominator",
                path=f"{path}.score_numerator",
            )
        _validate_sorted_unique_strings(self.notes, f"{path}.notes")


@dataclass(frozen=True, slots=True)
class AcceptanceDefinition:
    key: str
    stage: CapabilityStage
    required_case_ids: tuple[str, ...]
    blocking_capability_ids: tuple[str, ...]
    notes: tuple[str, ...]

    def validate(self, path: str = "acceptance_definition") -> None:
        validate_nonempty(self.key, f"{path}.key")
        if type(self.stage) is not CapabilityStage:
            raise SchemaError("must be a CapabilityStage", path=f"{path}.stage")
        _validate_sorted_unique_strings(
            self.required_case_ids, f"{path}.required_case_ids"
        )
        _validate_sorted_unique_strings(
            self.blocking_capability_ids, f"{path}.blocking_capability_ids"
        )
        if not self.required_case_ids and not self.blocking_capability_ids:
            raise SchemaError(
                "must require at least one case or capability", path=path
            )
        _validate_sorted_unique_strings(self.notes, f"{path}.notes")


@dataclass(frozen=True, slots=True)
class CaseDefinition:
    id: str
    stage: CapabilityStage
    maximum_status: CapabilityStatus
    capability_ids: tuple[str, ...]
    input_refs: tuple[str, ...]
    description: str

    def validate(self, path: str = "case_definition") -> None:
        validate_nonempty(self.id, f"{path}.id")
        if type(self.stage) is not CapabilityStage:
            raise SchemaError("must be a CapabilityStage", path=f"{path}.stage")
        if type(self.maximum_status) is not CapabilityStatus:
            raise SchemaError(
                "must be a CapabilityStatus", path=f"{path}.maximum_status"
            )
        if self.maximum_status is CapabilityStatus.UNSUPPORTED:
            raise SchemaError(
                "a case must be capable of producing evidence",
                path=f"{path}.maximum_status",
            )
        _validate_sorted_unique_strings(self.capability_ids, f"{path}.capability_ids")
        if not self.capability_ids:
            raise SchemaError(
                "must reference at least one capability",
                path=f"{path}.capability_ids",
            )
        _validate_sorted_unique_strings(self.input_refs, f"{path}.input_refs")
        if not self.input_refs:
            raise SchemaError(
                "must reference at least one checked-in input",
                path=f"{path}.input_refs",
            )
        validate_nonempty(self.description, f"{path}.description")


@dataclass(frozen=True, slots=True)
class CaseMatrix:
    schema_version: str
    producer_pass: str
    id: str
    capabilities: tuple[CapabilityDefinition, ...]
    acceptance: tuple[AcceptanceDefinition, ...]
    cases: tuple[CaseDefinition, ...]

    @classmethod
    def create(
        cls,
        *,
        capabilities: tuple[CapabilityDefinition, ...],
        acceptance: tuple[AcceptanceDefinition, ...],
        cases: tuple[CaseDefinition, ...],
    ) -> "CaseMatrix":
        semantic_key = {
            "capabilities": capabilities,
            "acceptance": acceptance,
            "cases": cases,
        }
        result = cls(
            schema_version=CASE_MATRIX_SCHEMA_VERSION,
            producer_pass="case_matrix_builder",
            id=stable_artifact_id(
                "case_matrix",
                semantic_key,
                schema_version=CASE_MATRIX_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def validate(self, path: str = "case_matrix") -> None:
        if self.schema_version != CASE_MATRIX_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "case_matrix_builder":
            raise SchemaError(
                "must be 'case_matrix_builder'", path=f"{path}.producer_pass"
            )
        _validate_sorted_unique_by_key(
            self.capabilities, attribute="key", path=f"{path}.capabilities"
        )
        _validate_sorted_unique_by_key(
            self.acceptance, attribute="key", path=f"{path}.acceptance"
        )
        _validate_sorted_unique_by_key(self.cases, attribute="id", path=f"{path}.cases")
        capability_by_id: dict[str, CapabilityDefinition] = {}
        for index, definition in enumerate(self.capabilities):
            definition.validate(f"{path}.capabilities[{index}]")
            capability_by_id[definition.key] = definition
        acceptance_by_id: dict[str, AcceptanceDefinition] = {}
        for index, definition in enumerate(self.acceptance):
            definition.validate(f"{path}.acceptance[{index}]")
            acceptance_by_id[definition.key] = definition
        case_by_id: dict[str, CaseDefinition] = {}
        for index, case in enumerate(self.cases):
            case.validate(f"{path}.cases[{index}]")
            case_by_id[case.id] = case
            for capability_index, capability_id in enumerate(case.capability_ids):
                capability = capability_by_id.get(capability_id)
                if capability is None:
                    raise SchemaError(
                        "references an unknown capability",
                        path=(
                            f"{path}.cases[{index}].capability_ids"
                            f"[{capability_index}]"
                        ),
                    )
                if capability.stage is not case.stage:
                    raise SchemaError(
                        "case and capability stages must match",
                        path=f"{path}.cases[{index}].stage",
                    )
        for index, capability in enumerate(self.capabilities):
            for case_index, case_id in enumerate(capability.required_case_ids):
                case = case_by_id.get(case_id)
                if case is None:
                    raise SchemaError(
                        "references an unknown case",
                        path=(
                            f"{path}.capabilities[{index}].required_case_ids"
                            f"[{case_index}]"
                        ),
                    )
                if capability.key not in case.capability_ids:
                    raise SchemaError(
                        "case does not reciprocally reference the capability",
                        path=f"{path}.cases[{self.cases.index(case)}].capability_ids",
                    )
                if case.stage is not capability.stage:
                    raise SchemaError(
                        "case and capability stages must match",
                        path=f"{path}.capabilities[{index}].stage",
                    )
        for case_index, case in enumerate(self.cases):
            for capability_id in case.capability_ids:
                capability = capability_by_id[capability_id]
                if case.id not in capability.required_case_ids:
                    raise SchemaError(
                        "capability does not reciprocally require the case",
                        path=f"{path}.cases[{case_index}].capability_ids",
                    )
        for index, definition in enumerate(self.acceptance):
            for case_index, case_id in enumerate(definition.required_case_ids):
                case = case_by_id.get(case_id)
                if case is None:
                    raise SchemaError(
                        "references an unknown case",
                        path=(
                            f"{path}.acceptance[{index}].required_case_ids"
                            f"[{case_index}]"
                        ),
                    )
                if case.stage is not definition.stage:
                    raise SchemaError(
                        "case and acceptance stages must match",
                        path=f"{path}.acceptance[{index}].stage",
                    )
            for capability_index, capability_id in enumerate(
                definition.blocking_capability_ids
            ):
                capability = capability_by_id.get(capability_id)
                if capability is None:
                    raise SchemaError(
                        "references an unknown capability",
                        path=(
                            f"{path}.acceptance[{index}].blocking_capability_ids"
                            f"[{capability_index}]"
                        ),
                    )
                if capability.stage is not definition.stage:
                    raise SchemaError(
                        "capability and acceptance stages must match",
                        path=f"{path}.acceptance[{index}].stage",
                    )
        expected_id = stable_artifact_id(
            "case_matrix",
            {
                "capabilities": self.capabilities,
                "acceptance": self.acceptance,
                "cases": self.cases,
            },
            schema_version=CASE_MATRIX_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class CaseEvidence:
    case_id: str
    status: CapabilityStatus
    evidence_digest: str
    artifact_digest: str | None
    report_digest: str | None

    def validate(self, path: str = "case_evidence") -> None:
        validate_nonempty(self.case_id, f"{path}.case_id")
        if type(self.status) is not CapabilityStatus:
            raise SchemaError("must be a CapabilityStatus", path=f"{path}.status")
        if self.status is CapabilityStatus.UNSUPPORTED:
            raise SchemaError(
                "unsupported is absence of evidence, not evidence",
                path=f"{path}.status",
            )
        _validate_digest(self.evidence_digest, f"{path}.evidence_digest")
        if self.artifact_digest is not None:
            _validate_digest(self.artifact_digest, f"{path}.artifact_digest")
        if self.report_digest is not None:
            _validate_digest(self.report_digest, f"{path}.report_digest")
        if capability_status_rank(self.status) >= capability_status_rank(
            CapabilityStatus.E2E_TIMING
        ):
            if self.artifact_digest is None or self.report_digest is None:
                raise SchemaError(
                    "end-to-end evidence requires artifact and report digests",
                    path=path,
                )


@dataclass(frozen=True, slots=True)
class CapabilityClaim:
    key: str
    stage: CapabilityStage
    status: CapabilityStatus
    evidence_case_ids: tuple[str, ...]
    score_numerator: int
    score_denominator: int
    notes: tuple[str, ...]

    def validate(self, path: str = "capability_claim") -> None:
        CapabilityDefinition(
            key=self.key,
            stage=self.stage,
            required_case_ids=("placeholder",),
            score_numerator=self.score_numerator,
            score_denominator=self.score_denominator,
            notes=self.notes,
        ).validate(path)
        if type(self.status) is not CapabilityStatus:
            raise SchemaError("must be a CapabilityStatus", path=f"{path}.status")
        _validate_sorted_unique_strings(
            self.evidence_case_ids, f"{path}.evidence_case_ids"
        )
        if self.status is CapabilityStatus.UNSUPPORTED and self.evidence_case_ids:
            raise SchemaError(
                "unsupported claims cannot cite evidence",
                path=f"{path}.evidence_case_ids",
            )


@dataclass(frozen=True, slots=True)
class AcceptanceClaim:
    key: str
    stage: CapabilityStage
    passed: bool
    evidence_case_ids: tuple[str, ...]
    blocking_capability_ids: tuple[str, ...]
    notes: tuple[str, ...]

    def validate(self, path: str = "acceptance_claim") -> None:
        validate_nonempty(self.key, f"{path}.key")
        if type(self.stage) is not CapabilityStage:
            raise SchemaError("must be a CapabilityStage", path=f"{path}.stage")
        if type(self.passed) is not bool:
            raise SchemaError("must be a bool", path=f"{path}.passed")
        _validate_sorted_unique_strings(
            self.evidence_case_ids, f"{path}.evidence_case_ids"
        )
        _validate_sorted_unique_strings(
            self.blocking_capability_ids, f"{path}.blocking_capability_ids"
        )
        _validate_sorted_unique_strings(self.notes, f"{path}.notes")


@dataclass(frozen=True, slots=True)
class CapabilityManifest:
    schema_version: str
    producer_pass: str
    id: str
    baseline_epoch: str
    case_matrix_id: str
    case_matrix_digest: str
    evidence: tuple[CaseEvidence, ...]
    claims: tuple[CapabilityClaim, ...]
    acceptance_claims: tuple[AcceptanceClaim, ...]

    def validate(self, path: str = "capability_manifest") -> None:
        if self.schema_version != CAPABILITY_MANIFEST_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "capability_manifest_builder":
            raise SchemaError(
                "must be 'capability_manifest_builder'",
                path=f"{path}.producer_pass",
            )
        validate_nonempty(self.baseline_epoch, f"{path}.baseline_epoch")
        validate_nonempty(self.case_matrix_id, f"{path}.case_matrix_id")
        _validate_digest(self.case_matrix_digest, f"{path}.case_matrix_digest")
        _validate_sorted_unique_by_key(
            self.evidence, attribute="case_id", path=f"{path}.evidence"
        )
        _validate_sorted_unique_by_key(
            self.claims, attribute="key", path=f"{path}.claims"
        )
        _validate_sorted_unique_by_key(
            self.acceptance_claims,
            attribute="key",
            path=f"{path}.acceptance_claims",
        )
        for index, evidence in enumerate(self.evidence):
            evidence.validate(f"{path}.evidence[{index}]")
        for index, claim in enumerate(self.claims):
            claim.validate(f"{path}.claims[{index}]")
        for index, claim in enumerate(self.acceptance_claims):
            claim.validate(f"{path}.acceptance_claims[{index}]")
        expected_id = stable_artifact_id(
            "capability_manifest",
            self._semantic_key(),
            schema_version=CAPABILITY_MANIFEST_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )

    def validate_against(
        self, case_matrix: CaseMatrix, path: str = "capability_manifest"
    ) -> None:
        self.validate(path)
        case_matrix.validate(f"{path}.case_matrix")
        if self.case_matrix_id != case_matrix.id:
            raise SchemaError(
                "does not identify the supplied case matrix",
                path=f"{path}.case_matrix_id",
            )
        expected_digest = canonical_digest(case_matrix)
        if self.case_matrix_digest != expected_digest:
            raise SchemaError(
                "does not digest the supplied case matrix",
                path=f"{path}.case_matrix_digest",
            )
        from ..passes.capability_manifest import build_capability_manifest

        expected = build_capability_manifest(
            case_matrix,
            self.evidence,
            baseline_epoch=self.baseline_epoch,
        )
        if self != expected:
            raise SchemaError(
                "claims are not the exact result of the supplied evidence",
                path=path,
            )

    def coverage_score(self, stage: CapabilityStage) -> Fraction:
        if type(stage) is not CapabilityStage:
            raise SchemaError("must be a CapabilityStage", path="stage")
        return sum(
            (
                Fraction(claim.score_numerator, claim.score_denominator)
                for claim in self.claims
                if claim.stage is stage
            ),
            start=Fraction(0, 1),
        )

    def acceptance_score(self, stage: CapabilityStage) -> tuple[int, int]:
        if type(stage) is not CapabilityStage:
            raise SchemaError("must be a CapabilityStage", path="stage")
        claims = tuple(
            claim for claim in self.acceptance_claims if claim.stage is stage
        )
        return sum(claim.passed for claim in claims), len(claims)

    def _semantic_key(self) -> dict[str, object]:
        return {
            "baseline_epoch": self.baseline_epoch,
            "case_matrix_id": self.case_matrix_id,
            "case_matrix_digest": self.case_matrix_digest,
            "evidence": self.evidence,
            "claims": self.claims,
            "acceptance_claims": self.acceptance_claims,
        }
