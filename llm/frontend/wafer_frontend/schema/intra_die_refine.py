"""Versioned, exact-provenance intra-die graph-refinement artifacts.

The identity contract preserves an IR2 graph exactly.  ``local_transport/v1``
adds a separately versioned local-flow annotation, which is closed against the
final schedule before any future backend lowering consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty
from .ir2 import IR2ProjectionResult
from .intra_die_optimization import IntraDieOptimizationMode
from .intra_die_v2_search import IntraDieV2CandidateKind
from .local_transport import LocalTransportPlan
from .split_k_refine import SplitKRefinedProjection
from .n5 import ProjectedIR2Bundle, ProjectedProfileIR2
from .policy import PolicySelection, RegistryKind


INTRA_DIE_REFINE_CONTEXT_SCHEMA_VERSION = (
    "wafer_frontend.intra_die_refine_context/v1alpha1"
)
REFINED_PROFILE_IR2_SCHEMA_VERSION = (
    "wafer_frontend.refined_profile_ir2/v1alpha1"
)
REFINED_IR2_BUNDLE_SCHEMA_VERSION = (
    "wafer_frontend.refined_ir2_bundle/v1alpha1"
)
INTRA_DIE_OPTIMIZATION_OPTIONS_SCHEMA_VERSION = (
    "wafer_frontend.intra_die_optimization_options/v1alpha4"
)


class IntraDieRefineContract(str, Enum):
    """The exact graph rewrite semantics selected for a refine pass."""

    IDENTITY_V1 = "identity/v1"
    LOCAL_TRANSPORT_V1 = "local_transport/v1"
    SPLIT_K_REDUCE_DOUBLE_BUFFER_V2 = "split_k_reduce_double_buffer/v2"

_INTRA_DIE_CANDIDATE_NAMES = frozenset(
    {
        "identity",
        "split_k",
        "split_k_barrier",
        "split_k_double_buffer",
        "split_k_tree_direct_dma",
    }
)


@dataclass(frozen=True, slots=True)
class IntraDieOptimizationOptions:
    """Versioned, bounded product options used by the AUTO selector."""

    schema_version: str = INTRA_DIE_OPTIMIZATION_OPTIONS_SCHEMA_VERSION
    mode: IntraDieOptimizationMode = IntraDieOptimizationMode.AUTO
    allowed_candidates: tuple[str, ...] = ("identity", "split_k")
    max_candidates: int = 8
    split_k_parts: tuple[int, ...] = (2, 4)
    temporal_chunks: tuple[int, ...] = (1,)
    compute_groups_per_die: int = 2
    require_full_compute_groups: bool = False
    force_candidate: str | None = None
    timing_hardware_digest: str | None = None
    timing_simulation_digest: str | None = None

    def validate(self, path: str = "intra_die_optimization_options") -> None:
        if self.schema_version != INTRA_DIE_OPTIMIZATION_OPTIONS_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.mode) is not IntraDieOptimizationMode:
            raise SchemaError("must be an IntraDieOptimizationMode", path=f"{path}.mode")
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 8:
            raise SchemaError("max_candidates must be in [1, 8]", path=f"{path}.max_candidates")
        if (
            type(self.compute_groups_per_die) is not int
            or not 1 <= self.compute_groups_per_die <= 16
        ):
            raise SchemaError(
                "compute_groups_per_die must be in [1, 16]",
                path=f"{path}.compute_groups_per_die",
            )
        if type(self.require_full_compute_groups) is not bool:
            raise SchemaError(
                "require_full_compute_groups must be bool",
                path=f"{path}.require_full_compute_groups",
            )
        if (
            self.require_full_compute_groups
            and self.mode is not IntraDieOptimizationMode.AUTO
        ):
            raise SchemaError(
                "require_full_compute_groups is legal only in AUTO mode",
                path=f"{path}.require_full_compute_groups",
            )
        if (
            type(self.allowed_candidates) is not tuple
            or not self.allowed_candidates
            or any(
                type(candidate) is not str
                or candidate not in _INTRA_DIE_CANDIDATE_NAMES
                for candidate in self.allowed_candidates
            )
            or len(set(self.allowed_candidates)) != len(self.allowed_candidates)
        ):
            raise SchemaError(
                "allowed_candidates must be unique supported candidate names",
                path=f"{path}.allowed_candidates",
            )
        if self.allowed_candidates[0] != "identity":
            raise SchemaError(
                "identity must be the first allowed candidate",
                path=f"{path}.allowed_candidates",
            )
        for field_name, values, minimum in (
            ("split_k_parts", self.split_k_parts, 2),
            ("temporal_chunks", self.temporal_chunks, 1),
        ):
            if (
                type(values) is not tuple
                or not values
                or any(type(value) is not int or value < minimum for value in values)
                or tuple(sorted(set(values))) != values
            ):
                raise SchemaError(
                    f"{field_name} must be a strictly increasing tuple of integers >= {minimum}",
                    path=f"{path}.{field_name}",
                )
        generated_bound = 1 + len(self.split_k_parts) * sum(
            candidate != "identity" for candidate in self.allowed_candidates
        )
        if self.mode is not IntraDieOptimizationMode.OFF and generated_bound > self.max_candidates:
            raise SchemaError(
                "configured candidate product exceeds max_candidates",
                path=f"{path}.max_candidates",
            )
        if self.mode is IntraDieOptimizationMode.FORCE:
            if self.force_candidate not in self.allowed_candidates:
                raise SchemaError(
                    "FORCE requires force_candidate in allowed_candidates",
                    path=f"{path}.force_candidate",
                )
            if self.force_candidate != "identity" and len(self.split_k_parts) != 1:
                raise SchemaError(
                    "FORCE split-K requires exactly one split_k_parts value",
                    path=f"{path}.split_k_parts",
                )
        elif self.force_candidate is not None:
            raise SchemaError(
                "force_candidate is only legal in FORCE mode",
                path=f"{path}.force_candidate",
            )
        timing_digests = (
            self.timing_hardware_digest, self.timing_simulation_digest
        )
        if (timing_digests[0] is None) != (timing_digests[1] is None):
            raise SchemaError(
                "timing hardware and simulation digests must be provided together",
                path=f"{path}.timing_hardware_digest",
            )
        for field_name, digest in zip(
            ("timing_hardware_digest", "timing_simulation_digest"),
            timing_digests, strict=True,
        ):
            if digest is not None and (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SchemaError(
                    "must be a lowercase SHA256 digest",
                    path=f"{path}.{field_name}",
                )



@dataclass(frozen=True, slots=True)
class SplitKRefineOptions:
    """Explicit v2 knobs; disabled defaults preserve identity behavior."""

    split_k_parts: int = 1
    enable_reduce: bool = False
    enable_double_buffer: bool = False
    compute_groups_per_die: int = 2
    enable_streaming_reduce: bool = True
    enable_tree_reduce: bool = False
    enable_direct_dma: bool = False

    def validate(self, path: str = "split_k_refine_options") -> None:
        if type(self.split_k_parts) is not int or self.split_k_parts < 1:
            raise SchemaError("split_k_parts must be a positive integer", path=f"{path}.split_k_parts")
        if (
            type(self.enable_reduce) is not bool
            or type(self.enable_double_buffer) is not bool
            or type(self.enable_streaming_reduce) is not bool
            or type(self.enable_tree_reduce) is not bool
            or type(self.enable_direct_dma) is not bool
        ):
            raise SchemaError("flags must be bool", path=path)
        if (
            type(self.compute_groups_per_die) is not int
            or not 1 <= self.compute_groups_per_die <= 16
        ):
            raise SchemaError(
                "compute_groups_per_die must be in [1, 16]",
                path=f"{path}.compute_groups_per_die",
            )
        if self.split_k_parts == 1 and (self.enable_reduce or self.enable_double_buffer):
            raise SchemaError("split-k reduce/double-buffer require split_k_parts > 1", path=path)
        if self.enable_tree_reduce and not self.enable_reduce:
            raise SchemaError("tree reduce requires enable_reduce", path=path)
        if self.enable_direct_dma and not self.enable_reduce:
            raise SchemaError("direct DMA requires enable_reduce", path=path)


@dataclass(frozen=True, slots=True)
class IntraDieRefineContext:
    schema_version: str
    producer_pass: str
    id: str
    policy: PolicySelection
    contract: IntraDieRefineContract
    options: SplitKRefineOptions | IntraDieOptimizationOptions = SplitKRefineOptions()

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        policy: PolicySelection,
        contract: IntraDieRefineContract = IntraDieRefineContract.IDENTITY_V1,
        options: SplitKRefineOptions | IntraDieOptimizationOptions = SplitKRefineOptions(),
    ) -> "IntraDieRefineContext":
        semantic_key = {"policy": policy, "contract": contract, "options": options}
        return cls(
            schema_version=INTRA_DIE_REFINE_CONTEXT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "intra_die_refine_context",
                semantic_key,
                schema_version=INTRA_DIE_REFINE_CONTEXT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str = "intra_die_refine_context") -> None:
        if self.schema_version != INTRA_DIE_REFINE_CONTEXT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.policy) is not PolicySelection:
            raise SchemaError("must be a PolicySelection", path=f"{path}.policy")
        self.policy.validate(f"{path}.policy")
        if self.policy.kind is not RegistryKind.INTRA_DIE:
            raise SchemaError("must select an intra-die policy", path=f"{path}.policy.kind")
        if type(self.contract) is not IntraDieRefineContract:
            raise SchemaError("must be an IntraDieRefineContract", path=f"{path}.contract")
        if self.contract not in (
            IntraDieRefineContract.IDENTITY_V1,
            IntraDieRefineContract.LOCAL_TRANSPORT_V1,
            IntraDieRefineContract.SPLIT_K_REDUCE_DOUBLE_BUFFER_V2,
        ):
            raise SchemaError("unsupported refine contract", path=f"{path}.contract")
        if type(self.options) not in (SplitKRefineOptions, IntraDieOptimizationOptions):
            raise SchemaError(
                "must be SplitKRefineOptions or IntraDieOptimizationOptions",
                path=f"{path}.options",
            )
        self.options.validate(f"{path}.options")
        if self.contract is not IntraDieRefineContract.SPLIT_K_REDUCE_DOUBLE_BUFFER_V2 and self.options != SplitKRefineOptions():
            raise SchemaError("options require split_k_reduce_double_buffer/v2", path=f"{path}.options")
        expected_id = stable_artifact_id(
            "intra_die_refine_context",
            {"policy": self.policy, "contract": self.contract, "options": self.options},
            schema_version=INTRA_DIE_REFINE_CONTEXT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class RefinedProfileIR2:
    """One refined profile plus an optional local-transport annotation."""

    schema_version: str
    producer_pass: str
    id: str
    source: ProjectedProfileIR2
    context: IntraDieRefineContext
    projection: IR2ProjectionResult
    split_k_refinement: SplitKRefinedProjection | None = None
    local_transport_plan: LocalTransportPlan | None = None

    @classmethod
    def create(
        cls,
        *,
        source: ProjectedProfileIR2,
        context: IntraDieRefineContext,
        projection: IR2ProjectionResult,
        local_transport_plan: LocalTransportPlan | None = None,
        split_k_refinement: SplitKRefinedProjection | None = None,
    ) -> "RefinedProfileIR2":
        if type(source) is not ProjectedProfileIR2:
            raise SchemaError("must be a ProjectedProfileIR2", path="source")
        if type(context) is not IntraDieRefineContext:
            raise SchemaError("must be an IntraDieRefineContext", path="intra_die_refine_context")
        if type(projection) is not IR2ProjectionResult:
            raise SchemaError("must be an IR2ProjectionResult", path="projection")
        semantic_key = {
            "source_projected_profile_id": source.id,
            "refine_context_id": context.id,
            "projection_id": projection.id,
            "local_transport_plan": local_transport_plan,
            "split_k_refinement": split_k_refinement,
        }
        return cls(
            schema_version=REFINED_PROFILE_IR2_SCHEMA_VERSION,
            producer_pass="intra_die_refine",
            id=stable_artifact_id(
                "refined_profile_ir2",
                semantic_key,
                schema_version=REFINED_PROFILE_IR2_SCHEMA_VERSION,
            ),
            source=source,
            context=context,
            projection=projection,
            local_transport_plan=local_transport_plan,
            split_k_refinement=split_k_refinement,
        )

    def validate(self, path: str = "refined_profile_ir2") -> None:
        if self.schema_version != REFINED_PROFILE_IR2_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "intra_die_refine":
            raise SchemaError("must be 'intra_die_refine'", path=f"{path}.producer_pass")
        if type(self.source) is not ProjectedProfileIR2:
            raise SchemaError("must be a ProjectedProfileIR2", path=f"{path}.source")
        if type(self.context) is not IntraDieRefineContext:
            raise SchemaError("must be an IntraDieRefineContext", path=f"{path}.context")
        if type(self.projection) is not IR2ProjectionResult:
            raise SchemaError("must be an IR2ProjectionResult", path=f"{path}.projection")
        if self.split_k_refinement is not None:
            if type(self.split_k_refinement) is not SplitKRefinedProjection:
                raise SchemaError(
                    "must be a SplitKRefinedProjection",
                    path=f"{path}.split_k_refinement",
                )
            if self.context.contract is not IntraDieRefineContract.SPLIT_K_REDUCE_DOUBLE_BUFFER_V2:
                raise SchemaError(
                    "requires split_k_reduce_double_buffer/v2 contract",
                    path=f"{path}.split_k_refinement",
                )
            decision = self.split_k_refinement.search_decision
            selected = next(
                candidate for candidate in decision.candidates
                if candidate.id == decision.selected_candidate_ref
            )
            identity = selected.kind is IntraDieV2CandidateKind.IDENTITY
            self.split_k_refinement.validate_against(
                self.projection, self.source.graph,
                split_k_parts=1 if identity else selected.split_k_parts,
                enable_reduce=False if identity else selected.enable_reduce,
                enable_double_buffer=(
                    False if identity else selected.enable_double_buffer
                ),
                enable_streaming_reduce=(
                    False if identity else selected.enable_streaming_reduce
                ),
                enable_tree_reduce=(
                    False if identity else selected.enable_tree_reduce
                ),
                enable_direct_dma=(
                    False if identity else selected.enable_direct_dma
                ),
                path=f"{path}.split_k_refinement",
            )
        elif (
            self.context.contract
            is IntraDieRefineContract.SPLIT_K_REDUCE_DOUBLE_BUFFER_V2
            and (
                type(self.context.options) is IntraDieOptimizationOptions
                or self.context.options.split_k_parts > 1
            )
        ):
            raise SchemaError(
                "enabled v2 refinement requires a search-decision carrier",
                path=f"{path}.split_k_refinement",
            )
        if self.local_transport_plan is not None:
            if type(self.local_transport_plan) is not LocalTransportPlan:
                raise SchemaError("must be a LocalTransportPlan", path=f"{path}.local_transport_plan")
            if self.context.contract is not IntraDieRefineContract.LOCAL_TRANSPORT_V1:
                raise SchemaError("requires local_transport/v1 contract", path=f"{path}.local_transport_plan")
            self.local_transport_plan.validate(f"{path}.local_transport_plan")
            if self.local_transport_plan.source_dag_id not in {dag.id for dag in self.projection.dags}:
                raise SchemaError("must reference a refined projection DAG", path=f"{path}.local_transport_plan.source_dag_id")
        elif self.context.contract is IntraDieRefineContract.LOCAL_TRANSPORT_V1:
            raise SchemaError("requires a local transport plan", path=f"{path}.local_transport_plan")
        self.source.validate(f"{path}.source")
        self.context.validate(f"{path}.context")
        if self.projection != self.source.projection:
            raise SchemaError("refine must preserve the exact projection until IR2 local tasks are introduced", path=f"{path}.projection")
        expected_id = stable_artifact_id(
            "refined_profile_ir2",
            {
                "source_projected_profile_id": self.source.id,
                "refine_context_id": self.context.id,
                "projection_id": self.projection.id,
                "local_transport_plan": self.local_transport_plan,
                "split_k_refinement": self.split_k_refinement,
            },
            schema_version=REFINED_PROFILE_IR2_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: ProjectedProfileIR2,
        context: IntraDieRefineContext,
        path: str = "refined_profile_ir2",
    ) -> None:
        if type(source) is not ProjectedProfileIR2:
            raise SchemaError("must be a ProjectedProfileIR2", path="source")
        if type(context) is not IntraDieRefineContext:
            raise SchemaError("must be an IntraDieRefineContext", path="intra_die_refine_context")
        source.validate("source")
        context.validate("intra_die_refine_context")
        self.validate(path)
        if self.source != source or self.context != context:
            raise SchemaError("must preserve exact source and context provenance", path=path)


@dataclass(frozen=True, slots=True)
class RefinedIR2Bundle:
    """Canonical identity-refined wrapper for every projected profile."""

    schema_version: str
    producer_pass: str
    id: str
    source: ProjectedIR2Bundle
    context: IntraDieRefineContext
    entries: tuple[RefinedProfileIR2, ...]

    @classmethod
    def create(
        cls,
        *,
        source: ProjectedIR2Bundle,
        context: IntraDieRefineContext,
        entries: tuple[RefinedProfileIR2, ...],
    ) -> "RefinedIR2Bundle":
        semantic_key = {
            "source_projected_bundle_id": source.id,
            "refine_context_id": context.id,
            "entries": entries,
        }
        return cls(
            schema_version=REFINED_IR2_BUNDLE_SCHEMA_VERSION,
            producer_pass="intra_die_refine",
            id=stable_artifact_id(
                "refined_ir2_bundle",
                semantic_key,
                schema_version=REFINED_IR2_BUNDLE_SCHEMA_VERSION,
            ),
            source=source,
            context=context,
            entries=entries,
        )

    def validate(self, path: str = "refined_ir2_bundle") -> None:
        if self.schema_version != REFINED_IR2_BUNDLE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "intra_die_refine":
            raise SchemaError("must be 'intra_die_refine'", path=f"{path}.producer_pass")
        if type(self.source) is not ProjectedIR2Bundle:
            raise SchemaError("must be a ProjectedIR2Bundle", path=f"{path}.source")
        if type(self.context) is not IntraDieRefineContext:
            raise SchemaError("must be an IntraDieRefineContext", path=f"{path}.context")
        self.source.validate(f"{path}.source")
        self.context.validate(f"{path}.context")
        if type(self.entries) is not tuple or len(self.entries) != len(self.source.entries):
            raise SchemaError("must contain one immutable entry per source profile", path=f"{path}.entries")
        for index, (entry, source_entry) in enumerate(zip(self.entries, self.source.entries)):
            if type(entry) is not RefinedProfileIR2:
                raise SchemaError("must be a RefinedProfileIR2", path=f"{path}.entries[{index}]")
            entry.validate_against(source_entry, self.context, f"{path}.entries[{index}]")
        expected_id = stable_artifact_id(
            "refined_ir2_bundle",
            {
                "source_projected_bundle_id": self.source.id,
                "refine_context_id": self.context.id,
                "entries": self.entries,
            },
            schema_version=REFINED_IR2_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: ProjectedIR2Bundle,
        context: IntraDieRefineContext,
        path: str = "refined_ir2_bundle",
    ) -> None:
        if type(source) is not ProjectedIR2Bundle:
            raise SchemaError("must be a ProjectedIR2Bundle", path="source")
        if type(context) is not IntraDieRefineContext:
            raise SchemaError("must be an IntraDieRefineContext", path="intra_die_refine_context")
        source.validate("source")
        context.validate("intra_die_refine_context")
        self.validate(path)
        if self.source != source or self.context != context:
            raise SchemaError("must preserve exact source and context provenance", path=path)


__all__ = [
    "INTRA_DIE_OPTIMIZATION_OPTIONS_SCHEMA_VERSION",
    "INTRA_DIE_REFINE_CONTEXT_SCHEMA_VERSION",
    "REFINED_IR2_BUNDLE_SCHEMA_VERSION",
    "REFINED_PROFILE_IR2_SCHEMA_VERSION",
    "IntraDieRefineContext",
    "IntraDieRefineContract",
    "IntraDieOptimizationMode",
    "IntraDieOptimizationOptions",
    "SplitKRefineOptions",
    "RefinedIR2Bundle",
    "RefinedProfileIR2",
]
