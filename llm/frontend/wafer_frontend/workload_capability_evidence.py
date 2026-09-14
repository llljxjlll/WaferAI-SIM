"""Evidence-derived workload capabilities and exact-case readiness."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SchemaError
from .schema.common import stable_artifact_id
from .schema.serde import canonical_digest
from .schema.workload_run import (
    WorkloadCapabilityLevel,
    WorkloadExecutionStrategy,
    WorkloadFamily,
    WorkloadFamilyCapability,
    WorkloadMemoryMode,
    WorkloadOptimizerKind,
    WorkloadRunCapability,
    WorkloadRunRequest,
)


WORKLOAD_CAPABILITY_ARTIFACT_SCHEMA_VERSION = (
    "wafer_frontend.workload_capability_artifact/v1alpha1"
)
WORKLOAD_CAPABILITY_EVIDENCE_SCHEMA_VERSION = (
    "wafer_frontend.workload_capability_evidence/v1alpha1"
)
WORKLOAD_CAPABILITY_DERIVATION_SCHEMA_VERSION = (
    "wafer_frontend.workload_capability_derivation/v1alpha1"
)


class WorkloadEvidenceScope(str, Enum):
    FULL_MODEL = "full_model"
    MOTIF = "motif"


class WorkloadCapabilityArtifactKind(str, Enum):
    LOWERING = "lowering"
    RUNTIME = "runtime"
    FUNCTIONAL = "functional"
    CAPACITY = "capacity"


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


@dataclass(frozen=True, slots=True)
class WorkloadCapabilityArtifact:
    schema_version: str
    id: str
    kind: WorkloadCapabilityArtifactKind
    artifact_digest: str

    @classmethod
    def create(
        cls,
        *,
        kind: WorkloadCapabilityArtifactKind,
        artifact_digest: str,
    ) -> "WorkloadCapabilityArtifact":
        key = {"kind": kind, "artifact_digest": artifact_digest}
        result = cls(
            schema_version=WORKLOAD_CAPABILITY_ARTIFACT_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_capability_artifact",
                key,
                schema_version=WORKLOAD_CAPABILITY_ARTIFACT_SCHEMA_VERSION,
            ),
            **key,
        )
        result.validate()
        return result

    def validate(self, path: str = "workload_capability_artifact") -> None:
        if self.schema_version != WORKLOAD_CAPABILITY_ARTIFACT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.kind) is not WorkloadCapabilityArtifactKind:
            raise SchemaError("must be an artifact kind", path=f"{path}.kind")
        _digest(self.artifact_digest, f"{path}.artifact_digest")
        expected = stable_artifact_id(
            "workload_capability_artifact",
            {"kind": self.kind, "artifact_digest": self.artifact_digest},
            schema_version=WORKLOAD_CAPABILITY_ARTIFACT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable artifact id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class WorkloadCapabilityEvidence:
    schema_version: str
    id: str
    request: WorkloadRunRequest
    request_digest: str
    family: WorkloadFamily
    case_id: str
    scope: WorkloadEvidenceScope
    source_digest: str
    binary_digest: str
    toolchain_digest: str
    artifacts: tuple[WorkloadCapabilityArtifact, ...]
    independent_execution_digests: tuple[str, ...]

    @classmethod
    def create(
        cls,
        *,
        request: WorkloadRunRequest,
        scope: WorkloadEvidenceScope,
        source_digest: str,
        binary_digest: str,
        toolchain_digest: str,
        artifacts: tuple[WorkloadCapabilityArtifact, ...],
        independent_execution_digests: tuple[str, ...],
    ) -> "WorkloadCapabilityEvidence":
        artifacts = tuple(sorted(artifacts, key=lambda item: item.kind.value))
        key = {
            "request": request,
            "request_digest": request.digest,
            "family": request.family,
            "case_id": request.case_id,
            "scope": scope,
            "source_digest": source_digest,
            "binary_digest": binary_digest,
            "toolchain_digest": toolchain_digest,
            "artifacts": artifacts,
            "independent_execution_digests": independent_execution_digests,
        }
        result = cls(
            schema_version=WORKLOAD_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
            id=stable_artifact_id(
                "workload_capability_evidence",
                key,
                schema_version=WORKLOAD_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
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
    def artifact_kinds(self) -> frozenset[WorkloadCapabilityArtifactKind]:
        return frozenset(item.kind for item in self.artifacts)

    def validate(self, path: str = "workload_capability_evidence") -> None:
        if self.schema_version != WORKLOAD_CAPABILITY_EVIDENCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.request) is not WorkloadRunRequest:
            raise SchemaError("must be a WorkloadRunRequest", path=f"{path}.request")
        self.request.validate(f"{path}.request")
        if self.request_digest != self.request.digest:
            raise SchemaError("does not match request", path=f"{path}.request_digest")
        if self.family is not self.request.family:
            raise SchemaError("does not match request", path=f"{path}.family")
        if self.case_id != self.request.case_id:
            raise SchemaError("does not match request", path=f"{path}.case_id")
        if type(self.scope) is not WorkloadEvidenceScope:
            raise SchemaError("must be an evidence scope", path=f"{path}.scope")
        for name in ("source_digest", "binary_digest", "toolchain_digest"):
            _digest(getattr(self, name), f"{path}.{name}")
        if type(self.artifacts) is not tuple:
            raise SchemaError("must be a tuple", path=f"{path}.artifacts")
        for index, artifact in enumerate(self.artifacts):
            if type(artifact) is not WorkloadCapabilityArtifact:
                raise SchemaError(
                    "must be a WorkloadCapabilityArtifact",
                    path=f"{path}.artifacts[{index}]",
                )
            artifact.validate(f"{path}.artifacts[{index}]")
        if self.artifacts != tuple(sorted(self.artifacts, key=lambda item: item.kind.value)):
            raise SchemaError("must use canonical kind order", path=f"{path}.artifacts")
        kinds = tuple(item.kind for item in self.artifacts)
        if len(set(kinds)) != len(kinds):
            raise SchemaError("must not repeat artifact kinds", path=f"{path}.artifacts")
        if type(self.independent_execution_digests) is not tuple:
            raise SchemaError(
                "must be a tuple", path=f"{path}.independent_execution_digests"
            )
        for index, digest in enumerate(self.independent_execution_digests):
            _digest(digest, f"{path}.independent_execution_digests[{index}]")
        expected = stable_artifact_id(
            "workload_capability_evidence",
            self._key(),
            schema_version=WORKLOAD_CAPABILITY_EVIDENCE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable evidence id", path=f"{path}.id")


def _has(
    evidence: WorkloadCapabilityEvidence,
    *kinds: WorkloadCapabilityArtifactKind,
) -> bool:
    return all(kind in evidence.artifact_kinds for kind in kinds)


def _is_multistep(request: WorkloadRunRequest) -> bool:
    if request.family.is_training:
        assert request.steps.training is not None
        return request.steps.training.step_count > 1
    assert request.steps.inference is not None
    return request.steps.inference.decode_steps > 0


def _derive_family(
    family: WorkloadFamily,
    evidence: tuple[WorkloadCapabilityEvidence, ...],
) -> WorkloadFamilyCapability:
    supported = WorkloadCapabilityLevel.SUPPORTED
    missing = WorkloadCapabilityLevel.NOT_MEASURED
    items = tuple(item for item in evidence if item.family is family)
    full_lowered = tuple(
        item
        for item in items
        if item.scope is WorkloadEvidenceScope.FULL_MODEL
        and _has(item, WorkloadCapabilityArtifactKind.LOWERING)
    )
    runtime = tuple(
        item
        for item in full_lowered
        if _has(item, WorkloadCapabilityArtifactKind.RUNTIME)
    )
    repeatable = tuple(
        item
        for item in runtime
        if len(item.independent_execution_digests) >= 2
        and len(set(item.independent_execution_digests)) == 1
    )
    baseline = any(
        item.request.execution.strategy is WorkloadExecutionStrategy.BASELINE
        for item in full_lowered
    )
    optimized = any(
        item.request.execution.strategy is WorkloadExecutionStrategy.OPTIMIZED
        for item in full_lowered
    )
    return WorkloadFamilyCapability(
        family=family,
        full_model=supported if full_lowered else missing,
        motif=supported if items else missing,
        baseline=supported if baseline or optimized else missing,
        optimized=supported if optimized else missing,
        lowering=supported if full_lowered else missing,
        runtime=supported if runtime else missing,
        timing=supported if runtime else missing,
        functional=(
            supported
            if any(_has(item, WorkloadCapabilityArtifactKind.FUNCTIONAL) for item in runtime)
            else missing
        ),
        capacity=(
            supported
            if any(_has(item, WorkloadCapabilityArtifactKind.CAPACITY) for item in items)
            else missing
        ),
        multi_step=(
            supported if any(_is_multistep(item.request) for item in runtime) else missing
        ),
        remote_hbm=(
            supported
            if any(
                item.request.memory.mode is WorkloadMemoryMode.REMOTE_HBM
                and _has(item, WorkloadCapabilityArtifactKind.CAPACITY)
                for item in runtime
            )
            else missing
        ),
        external_offload=(
            supported
            if any(
                item.request.memory.mode is WorkloadMemoryMode.EXTERNAL_OFFLOAD
                and _has(item, WorkloadCapabilityArtifactKind.CAPACITY)
                for item in runtime
            )
            else missing
        ),
        sgd_optimizer=(
            supported
            if any(
                item.request.optimizer is not None
                and item.request.optimizer.kind is WorkloadOptimizerKind.SGD
                for item in runtime
            )
            else missing
        ),
        adamw_optimizer=(
            supported
            if any(
                item.request.optimizer is not None
                and item.request.optimizer.kind is WorkloadOptimizerKind.ADAMW
                for item in runtime
            )
            else missing
        ),
        repeatability=supported if repeatable else missing,
    )


def _derive_capability(
    evidence: tuple[WorkloadCapabilityEvidence, ...],
    *,
    max_mesh_rows: int,
    max_mesh_columns: int,
    max_mesh_ranks: int,
) -> WorkloadRunCapability:
    return WorkloadRunCapability.create(
        max_mesh_rows=max_mesh_rows,
        max_mesh_columns=max_mesh_columns,
        max_mesh_ranks=max_mesh_ranks,
        families=tuple(_derive_family(family, evidence) for family in WorkloadFamily),
    )


@dataclass(frozen=True, slots=True)
class WorkloadCapabilityDerivation:
    schema_version: str
    id: str
    evidence: tuple[WorkloadCapabilityEvidence, ...]
    capability: WorkloadRunCapability

    def _key(self) -> dict[str, object]:
        return {
            "evidence": self.evidence,
            "capability": self.capability,
        }

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)

    def validate(self, path: str = "workload_capability_derivation") -> None:
        if self.schema_version != WORKLOAD_CAPABILITY_DERIVATION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.evidence) is not tuple or not self.evidence:
            raise SchemaError("must be a non-empty tuple", path=f"{path}.evidence")
        for index, item in enumerate(self.evidence):
            if type(item) is not WorkloadCapabilityEvidence:
                raise SchemaError(
                    "must be WorkloadCapabilityEvidence",
                    path=f"{path}.evidence[{index}]",
                )
            item.validate(f"{path}.evidence[{index}]")
        canonical = tuple(
            sorted(self.evidence, key=lambda item: (item.family.value, item.case_id, item.id))
        )
        if self.evidence != canonical or len({item.id for item in self.evidence}) != len(self.evidence):
            raise SchemaError("must be canonical and unique", path=f"{path}.evidence")
        self.capability.validate(f"{path}.capability")
        expected_capability = _derive_capability(
            self.evidence,
            max_mesh_rows=self.capability.max_mesh_rows,
            max_mesh_columns=self.capability.max_mesh_columns,
            max_mesh_ranks=self.capability.max_mesh_ranks,
        )
        if self.capability != expected_capability:
            raise SchemaError("capability is not evidence-derived", path=f"{path}.capability")
        expected_id = stable_artifact_id(
            "workload_capability_derivation",
            self._key(),
            schema_version=WORKLOAD_CAPABILITY_DERIVATION_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable derivation id", path=f"{path}.id")

    def readiness_reasons(
        self,
        request: WorkloadRunRequest,
        *,
        source_digest: str,
        binary_digest: str,
        toolchain_digest: str,
    ) -> tuple[str, ...]:
        self.validate()
        request.validate()
        exact = tuple(
            item
            for item in self.evidence
            if item.family is request.family
            and item.case_id == request.case_id
            and item.request_digest == request.digest
            and item.request == request
        )
        if not exact:
            return ("exact_case_evidence",)
        all_reasons: set[str] = set()
        for item in exact:
            reasons: list[str] = []
            if item.scope is not WorkloadEvidenceScope.FULL_MODEL:
                reasons.append("full_model_evidence")
            if item.source_digest != source_digest:
                reasons.append("source_digest")
            if item.binary_digest != binary_digest:
                reasons.append("binary_digest")
            if item.toolchain_digest != toolchain_digest:
                reasons.append("toolchain_digest")
            for kind in (
                WorkloadCapabilityArtifactKind.LOWERING,
                WorkloadCapabilityArtifactKind.RUNTIME,
                WorkloadCapabilityArtifactKind.CAPACITY,
            ):
                if kind not in item.artifact_kinds:
                    reasons.append(f"{kind.value}_artifact")
            if (
                request.execution.functional
                and WorkloadCapabilityArtifactKind.FUNCTIONAL not in item.artifact_kinds
            ):
                reasons.append("functional_artifact")
            required_executions = max(2, request.execution.independent_repeats)
            if len(item.independent_execution_digests) < required_executions:
                reasons.append("independent_executions")
            elif len(set(item.independent_execution_digests)) != 1:
                reasons.append("repeatability")
            if not reasons:
                return ()
            all_reasons.update(reasons)
        return tuple(sorted(all_reasons))


def build_workload_capability_from_evidence(
    evidence: tuple[WorkloadCapabilityEvidence, ...],
    *,
    max_mesh_rows: int,
    max_mesh_columns: int,
    max_mesh_ranks: int,
) -> WorkloadCapabilityDerivation:
    """The sole production constructor for evidence-backed capabilities."""

    if type(evidence) is not tuple or not evidence:
        raise SchemaError("must be a non-empty tuple", path="evidence")
    canonical = tuple(
        sorted(evidence, key=lambda item: (item.family.value, item.case_id, item.id))
    )
    for index, item in enumerate(canonical):
        if type(item) is not WorkloadCapabilityEvidence:
            raise SchemaError(
                "must be WorkloadCapabilityEvidence", path=f"evidence[{index}]"
            )
        item.validate(f"evidence[{index}]")
    capability = _derive_capability(
        canonical,
        max_mesh_rows=max_mesh_rows,
        max_mesh_columns=max_mesh_columns,
        max_mesh_ranks=max_mesh_ranks,
    )
    key = {"evidence": canonical, "capability": capability}
    result = WorkloadCapabilityDerivation(
        schema_version=WORKLOAD_CAPABILITY_DERIVATION_SCHEMA_VERSION,
        id=stable_artifact_id(
            "workload_capability_derivation",
            key,
            schema_version=WORKLOAD_CAPABILITY_DERIVATION_SCHEMA_VERSION,
        ),
        **key,
    )
    result.validate()
    return result


__all__ = [
    "WORKLOAD_CAPABILITY_ARTIFACT_SCHEMA_VERSION",
    "WORKLOAD_CAPABILITY_DERIVATION_SCHEMA_VERSION",
    "WORKLOAD_CAPABILITY_EVIDENCE_SCHEMA_VERSION",
    "WorkloadCapabilityArtifact",
    "WorkloadCapabilityArtifactKind",
    "WorkloadCapabilityDerivation",
    "WorkloadCapabilityEvidence",
    "WorkloadEvidenceScope",
    "build_workload_capability_from_evidence",
]
