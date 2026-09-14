"""Typed compiler dispatch and capacity report for rectangular Die meshes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .rect_mesh import RectMeshSpec


RECT_MESH_COMPILE_CAPABILITY_SCHEMA_VERSION = (
    "wafer_frontend.rect_mesh_compile_capability/v1alpha1"
)
RECT_MESH_ARTIFACT_RECORD_LIMIT = 1_048_576
RECT_MESH_ARTIFACT_FILE_LIMIT_BYTES = 64 * 1024 * 1024


class RectMeshCompileMode(str, Enum):
    NAIVE = "naive"
    AUTO = "auto"
    STANDARD = "standard"


class RectMeshCompileChain(str, Enum):
    NAIVE_FIXED_V1 = "naive_fixed/v1"
    RECT_MESH_STANDARD_V1 = "rect_mesh_standard/v1"


class RectMeshArtifactFilePreflight(str, Enum):
    DEFERRED_TO_FINALIZER = "deferred_to_finalizer"
    PASS = "pass"


class RectMeshFallbackReason(str, Enum):
    INVALID_MESH = "invalid_mesh"
    INVALID_PLACEMENT = "invalid_placement"
    INCOMPATIBLE_SHARDING = "incompatible_sharding"
    NO_CYCLE_USE_SNAKE = "no_cycle_use_snake"
    CANDIDATE_ACTION_BUDGET = "candidate_action_budget"
    CANDIDATE_BUFFER_BUDGET = "candidate_buffer_budget"
    CANDIDATE_SRAM_BUDGET = "candidate_sram_budget"
    RUNTIME_SESSION_BUDGET = "runtime_session_budget"
    PLANNER_DERIVED_BUDGET = "planner_derived_budget"
    ARTIFACT_RECORD_BUDGET = "artifact_record_budget"
    ARTIFACT_FILE_BUDGET = "artifact_file_budget"
    NO_ECONOMIC_BENEFIT = "no_economic_benefit"
    UNSUPPORTED_FUNCTIONAL_EXECUTION = "unsupported_functional_execution"
    UNSUPPORTED_MOE_MESH = "unsupported_moe_mesh"
    STANDARD_CHAIN_UNAVAILABLE = "standard_chain_unavailable"


_FALLBACK_ORDER = {
    reason: index for index, reason in enumerate(RectMeshFallbackReason)
}


@dataclass(frozen=True, slots=True)
class RectMeshCompileCapabilityReport:
    """One closed whole-workload dispatch and symbolic-capacity report.

    The record demand is exact for the linked manifests. Encoded bytes remain
    owned by the C++ finalizer, so Python reports that 64-MiB check as deferred
    rather than publishing an unsafe estimate.
    """

    schema_version: str
    producer_pass: str
    id: str
    mesh: RectMeshSpec
    requested_mode: RectMeshCompileMode
    selected_chain: RectMeshCompileChain
    fallback_reasons: tuple[RectMeshFallbackReason, ...]
    executable_baseline: bool
    standard_chain_complete: bool
    dense_workload_complete: bool
    profile_count: int
    manifest_count: int
    fragment_count: int
    symbolic_record_count: int
    artifact_record_limit: int
    artifact_file_limit_bytes: int
    artifact_file_preflight: RectMeshArtifactFilePreflight
    timing_execution: bool
    functional_execution: bool

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        mesh: RectMeshSpec,
        requested_mode: RectMeshCompileMode,
        selected_chain: RectMeshCompileChain,
        fallback_reasons: tuple[RectMeshFallbackReason, ...],
        profile_count: int,
        manifest_count: int,
        fragment_count: int,
        symbolic_record_count: int,
    ) -> "RectMeshCompileCapabilityReport":
        standard_complete = (
            selected_chain is RectMeshCompileChain.RECT_MESH_STANDARD_V1
        )
        semantic = {
            "mesh": mesh,
            "requested_mode": requested_mode,
            "selected_chain": selected_chain,
            "fallback_reasons": fallback_reasons,
            "executable_baseline": True,
            "standard_chain_complete": standard_complete,
            "dense_workload_complete": standard_complete,
            "profile_count": profile_count,
            "manifest_count": manifest_count,
            "fragment_count": fragment_count,
            "symbolic_record_count": symbolic_record_count,
            "artifact_record_limit": RECT_MESH_ARTIFACT_RECORD_LIMIT,
            "artifact_file_limit_bytes": RECT_MESH_ARTIFACT_FILE_LIMIT_BYTES,
            "artifact_file_preflight": (
                RectMeshArtifactFilePreflight.DEFERRED_TO_FINALIZER
            ),
            "timing_execution": mesh.timing_execution,
            "functional_execution": mesh.functional_execution,
        }
        result = cls(
            schema_version=RECT_MESH_COMPILE_CAPABILITY_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "rect_mesh_compile_capability",
                semantic,
                schema_version=RECT_MESH_COMPILE_CAPABILITY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "rect_mesh_compile_capability") -> None:
        if self.schema_version != RECT_MESH_COMPILE_CAPABILITY_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version", path=f"{path}.schema_version"
            )
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.mesh) is not RectMeshSpec:
            raise SchemaError("must be a RectMeshSpec", path=f"{path}.mesh")
        self.mesh.validate(f"{path}.mesh")
        if type(self.requested_mode) is not RectMeshCompileMode:
            raise SchemaError(
                "must be a RectMeshCompileMode", path=f"{path}.requested_mode"
            )
        if type(self.selected_chain) is not RectMeshCompileChain:
            raise SchemaError(
                "must be a RectMeshCompileChain", path=f"{path}.selected_chain"
            )
        if type(self.fallback_reasons) is not tuple or any(
            type(reason) is not RectMeshFallbackReason
            for reason in self.fallback_reasons
        ):
            raise SchemaError(
                "must be an immutable tuple of RectMeshFallbackReason",
                path=f"{path}.fallback_reasons",
            )
        if self.fallback_reasons != tuple(
            sorted(set(self.fallback_reasons), key=_FALLBACK_ORDER.__getitem__)
        ):
            raise SchemaError(
                "must be unique and canonical",
                path=f"{path}.fallback_reasons",
            )
        naive = self.selected_chain is RectMeshCompileChain.NAIVE_FIXED_V1
        if self.requested_mode is RectMeshCompileMode.NAIVE and (
            not naive or self.fallback_reasons
        ):
            raise SchemaError(
                "naive mode requires the naive chain without fallback", path=path
            )
        if self.requested_mode is RectMeshCompileMode.AUTO and (
            (naive and not self.fallback_reasons)
            or (not naive and self.fallback_reasons)
        ):
            raise SchemaError(
                "AUTO fallback reasons must agree with the selected chain",
                path=f"{path}.fallback_reasons",
            )
        if self.requested_mode is RectMeshCompileMode.STANDARD and naive:
            raise SchemaError(
                "forced standard mode requires the standard chain",
                path=f"{path}.selected_chain",
            )
        if not naive:
            raise SchemaError(
                "RectMesh standard whole-workload chain is not implemented",
                path=f"{path}.selected_chain",
                code=RectMeshFallbackReason.STANDARD_CHAIN_UNAVAILABLE.value,
            )
        for field_name in (
            "executable_baseline",
            "standard_chain_complete",
            "dense_workload_complete",
            "timing_execution",
            "functional_execution",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise SchemaError("must be a bool", path=f"{path}.{field_name}")
        if not self.executable_baseline:
            raise SchemaError(
                "requires an executable whole-workload baseline",
                path=f"{path}.executable_baseline",
            )
        standard = not naive
        if (
            self.standard_chain_complete != standard
            or self.dense_workload_complete != standard
        ):
            raise SchemaError(
                "completion flags must agree with selected_chain", path=path
            )
        for field_name in (
            "profile_count",
            "manifest_count",
            "fragment_count",
            "symbolic_record_count",
            "artifact_record_limit",
            "artifact_file_limit_bytes",
        ):
            validate_uint64(getattr(self, field_name), f"{path}.{field_name}")
        if self.profile_count != 1 or self.manifest_count != 1:
            raise SchemaError(
                "v1 must contain one whole-workload linked manifest",
                path=f"{path}.manifest_count",
            )
        if self.fragment_count == 0:
            raise SchemaError("must be positive", path=f"{path}.fragment_count")
        if self.artifact_record_limit != RECT_MESH_ARTIFACT_RECORD_LIMIT:
            raise SchemaError(
                "must preserve the production record limit",
                path=f"{path}.artifact_record_limit",
            )
        if self.symbolic_record_count > self.artifact_record_limit:
            raise SchemaError(
                "symbolic records exceed the production artifact limit",
                path=f"{path}.symbolic_record_count",
                code=RectMeshFallbackReason.ARTIFACT_RECORD_BUDGET.value,
            )
        if self.artifact_file_limit_bytes != RECT_MESH_ARTIFACT_FILE_LIMIT_BYTES:
            raise SchemaError(
                "must preserve the production 64-MiB file limit",
                path=f"{path}.artifact_file_limit_bytes",
            )
        if type(self.artifact_file_preflight) is not RectMeshArtifactFilePreflight:
            raise SchemaError(
                "must be a RectMeshArtifactFilePreflight",
                path=f"{path}.artifact_file_preflight",
            )
        if (
            self.artifact_file_preflight
            is not RectMeshArtifactFilePreflight.DEFERRED_TO_FINALIZER
        ):
            raise SchemaError(
                "encoded file-size proof belongs to the C++ finalizer",
                path=f"{path}.artifact_file_preflight",
            )
        if (
            self.timing_execution != self.mesh.timing_execution
            or self.functional_execution != self.mesh.functional_execution
        ):
            raise SchemaError(
                "execution capability must agree with RectMeshSpec", path=path
            )
        expected = stable_artifact_id(
            "rect_mesh_compile_capability",
            self._semantic_key(),
            schema_version=RECT_MESH_COMPILE_CAPABILITY_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )


__all__ = [
    "RECT_MESH_ARTIFACT_FILE_LIMIT_BYTES",
    "RECT_MESH_ARTIFACT_RECORD_LIMIT",
    "RECT_MESH_COMPILE_CAPABILITY_SCHEMA_VERSION",
    "RectMeshArtifactFilePreflight",
    "RectMeshCompileCapabilityReport",
    "RectMeshCompileChain",
    "RectMeshCompileMode",
    "RectMeshFallbackReason",
]
