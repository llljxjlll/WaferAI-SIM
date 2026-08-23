"""Versioned evidence for the bounded intra-die v2 fallback search.

The first v2 search deliberately contains only the always-legal identity
candidate and one explicitly requested split-K fallback.  It records hard
budgets and deterministic analytic estimates; the compiler never invokes the
simulator while enumerating candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .intra_die_optimization import IntraDieOptimizationMode
from .intra_die_timing_model import INTRA_DIE_TIMING_MODEL_VERSION


INTRA_DIE_V2_SEARCH_BUDGET_SCHEMA_VERSION = (
    "wafer_frontend.intra_die_v2_search_budget/v1alpha1"
)
INTRA_DIE_V2_CANDIDATE_SCHEMA_VERSION = (
    "wafer_frontend.intra_die_v2_candidate/v1alpha5"
)
INTRA_DIE_V2_SEARCH_DECISION_SCHEMA_VERSION = (
    "wafer_frontend.intra_die_v2_search_decision/v1alpha5"
)
INTRA_DIE_V2_ANALYTIC_MODEL_VERSION = INTRA_DIE_TIMING_MODEL_VERSION
INTRA_DIE_V2_CANDIDATE_REJECTION_SCHEMA_VERSION = (
    "wafer_frontend.intra_die_v2_candidate_rejection/v1alpha1"
)


class IntraDieV2CandidateKind(str, Enum):
    IDENTITY = "identity"
    SPLIT_K_FALLBACK = "split_k_fallback"


@dataclass(frozen=True, slots=True)
class IntraDieV2SearchBudget:
    """Hard product-flow caps; simulator calls are reserved for final evidence."""

    schema_version: str = INTRA_DIE_V2_SEARCH_BUDGET_SCHEMA_VERSION
    max_candidates: int = 8
    simulator_call_budget: int = 10
    analytic_model_version: str = INTRA_DIE_V2_ANALYTIC_MODEL_VERSION

    def validate(self, path: str = "intra_die_v2_search_budget") -> None:
        if self.schema_version != INTRA_DIE_V2_SEARCH_BUDGET_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 8:
            raise SchemaError("max_candidates must be in [1, 8]", path=f"{path}.max_candidates")
        if type(self.simulator_call_budget) is not int or not 0 <= self.simulator_call_budget <= 10:
            raise SchemaError(
                "simulator_call_budget must be in [0, 10]",
                path=f"{path}.simulator_call_budget",
            )
        if self.analytic_model_version != INTRA_DIE_V2_ANALYTIC_MODEL_VERSION:
            raise SchemaError(
                "unsupported analytic model",
                path=f"{path}.analytic_model_version",
            )


@dataclass(frozen=True, slots=True)
class IntraDieV2AnalyticCost:
    compute_cycles: int
    memory_cycles: int
    compute_memory_overlap_cycles: int
    transport_cycles: int
    reduction_cycles: int
    sync_cycles: int
    fixed_pipeline_cycles: int
    predicted_makespan_cycles: int

    def validate(self, path: str) -> None:
        for name in (
            "compute_cycles",
            "memory_cycles",
            "compute_memory_overlap_cycles",
            "transport_cycles",
            "reduction_cycles",
            "sync_cycles",
            "fixed_pipeline_cycles",
            "predicted_makespan_cycles",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.compute_memory_overlap_cycles > min(
            self.compute_cycles, self.memory_cycles
        ):
            raise SchemaError(
                "compute/memory overlap exceeds either resource timeline",
                path=f"{path}.compute_memory_overlap_cycles",
            )
        if self.predicted_makespan_cycles != (
            self.compute_cycles
            + self.memory_cycles
            - self.compute_memory_overlap_cycles
            + self.transport_cycles
            + self.reduction_cycles
            + self.sync_cycles
            + self.fixed_pipeline_cycles
        ):
            raise SchemaError(
                "predicted makespan must equal its exact analytic components",
                path=f"{path}.predicted_makespan_cycles",
            )


@dataclass(frozen=True, slots=True)
class IntraDieV2Candidate:
    schema_version: str
    id: str
    kind: IntraDieV2CandidateKind
    split_k_parts: int
    enable_reduce: bool
    enable_double_buffer: bool
    eligible_gemm_count: int
    analytic_cost: IntraDieV2AnalyticCost
    enable_streaming_reduce: bool = True
    enable_tree_reduce: bool = False
    enable_direct_dma: bool = False

    @classmethod
    def create(cls, **semantic: object) -> "IntraDieV2Candidate":
        return cls(
            schema_version=INTRA_DIE_V2_CANDIDATE_SCHEMA_VERSION,
            id=stable_artifact_id(
                "intra_die_v2_candidate",
                semantic,
                schema_version=INTRA_DIE_V2_CANDIDATE_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "kind",
                "split_k_parts",
                "enable_reduce",
                "enable_double_buffer",
                "eligible_gemm_count",
                "analytic_cost",
                "enable_streaming_reduce",
                "enable_tree_reduce",
                "enable_direct_dma",
            )
        }

    def validate(self, path: str = "intra_die_v2_candidate") -> None:
        if self.schema_version != INTRA_DIE_V2_CANDIDATE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.kind) is not IntraDieV2CandidateKind:
            raise SchemaError("must be an IntraDieV2CandidateKind", path=f"{path}.kind")
        for name in ("split_k_parts", "eligible_gemm_count"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            type(self.enable_reduce) is not bool
            or type(self.enable_double_buffer) is not bool
            or type(self.enable_streaming_reduce) is not bool
            or type(self.enable_tree_reduce) is not bool
            or type(self.enable_direct_dma) is not bool
        ):
            raise SchemaError("candidate flags must be bool", path=path)
        if self.kind is IntraDieV2CandidateKind.IDENTITY:
            if (
                self.split_k_parts != 1
                or self.enable_reduce
                or self.enable_double_buffer
                or self.enable_streaming_reduce
                or self.enable_tree_reduce
                or self.enable_direct_dma
            ):
                raise SchemaError("identity candidate must preserve the graph", path=path)
        elif self.split_k_parts < 2 or self.eligible_gemm_count == 0:
            raise SchemaError(
                "split-K fallback requires parts >= 2 and eligible GEMMs",
                path=path,
            )
        elif (self.enable_tree_reduce or self.enable_direct_dma) and not self.enable_reduce:
            raise SchemaError("tree reduce/direct DMA require reduction", path=path)
        if type(self.analytic_cost) is not IntraDieV2AnalyticCost:
            raise SchemaError("must be IntraDieV2AnalyticCost", path=f"{path}.analytic_cost")
        self.analytic_cost.validate(f"{path}.analytic_cost")
        expected = stable_artifact_id(
            "intra_die_v2_candidate",
            self._semantic_key(),
            schema_version=INTRA_DIE_V2_CANDIDATE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class IntraDieV2CandidateRejection:
    """Fail-closed AUTO rejection with exact break-even evidence."""

    schema_version: str
    id: str
    candidate_name: str
    split_k_parts: int
    enable_double_buffer: bool
    reason: str
    identity_predicted_makespan_cycles: int
    candidate_predicted_makespan_cycles: int | None
    compute_savings_cycles: int
    overhead_cycles: int

    @classmethod
    def create(cls, **semantic: object) -> "IntraDieV2CandidateRejection":
        return cls(
            schema_version=INTRA_DIE_V2_CANDIDATE_REJECTION_SCHEMA_VERSION,
            id=stable_artifact_id(
                "intra_die_v2_candidate_rejection",
                semantic,
                schema_version=INTRA_DIE_V2_CANDIDATE_REJECTION_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "candidate_name",
                "split_k_parts",
                "enable_double_buffer",
                "reason",
                "identity_predicted_makespan_cycles",
                "candidate_predicted_makespan_cycles",
                "compute_savings_cycles",
                "overhead_cycles",
            )
        }

    def validate(self, path: str = "intra_die_v2_candidate_rejection") -> None:
        if self.schema_version != INTRA_DIE_V2_CANDIDATE_REJECTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.candidate_name not in {
            "split_k", "split_k_barrier", "split_k_double_buffer",
            "split_k_tree_direct_dma",
        }:
            raise SchemaError("unsupported rejected candidate", path=f"{path}.candidate_name")
        if type(self.split_k_parts) is not int or self.split_k_parts < 2:
            raise SchemaError("split_k_parts must be >= 2", path=f"{path}.split_k_parts")
        if type(self.enable_double_buffer) is not bool:
            raise SchemaError("must be bool", path=f"{path}.enable_double_buffer")
        validate_nonempty(self.reason, f"{path}.reason")
        if self.reason not in {
            "no_eligible_gemm", "k_not_divisible", "break_even_not_met",
        }:
            raise SchemaError("unsupported rejection reason", path=f"{path}.reason")
        for name in (
            "identity_predicted_makespan_cycles",
            "compute_savings_cycles",
            "overhead_cycles",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.candidate_predicted_makespan_cycles is not None:
            validate_uint64(
                self.candidate_predicted_makespan_cycles,
                f"{path}.candidate_predicted_makespan_cycles",
            )
        if self.reason == "break_even_not_met":
            if (
                self.candidate_predicted_makespan_cycles is None
                or (
                    self.compute_savings_cycles > self.overhead_cycles
                    and self.candidate_predicted_makespan_cycles
                    < self.identity_predicted_makespan_cycles
                )
            ):
                raise SchemaError("break-even rejection evidence is inconsistent", path=path)
        expected = stable_artifact_id(
            "intra_die_v2_candidate_rejection",
            self._semantic_key(),
            schema_version=INTRA_DIE_V2_CANDIDATE_REJECTION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class IntraDieV2SearchDecision:
    schema_version: str
    producer_pass: str
    id: str
    source_projection_id: str
    source_ir1_id: str
    budget: IntraDieV2SearchBudget
    timing_model_ref: str
    hardware_digest: str
    simulation_digest: str
    mode: IntraDieOptimizationMode
    candidates: tuple[IntraDieV2Candidate, ...]
    rejected_candidates: tuple[IntraDieV2CandidateRejection, ...]
    selected_candidate_ref: str
    selection_reason: str
    generated_candidate_count: int
    full_analytic_evaluation_count: int
    simulator_calls_during_search: int
    reserved_simulator_calls_for_final_evidence: int
    require_full_compute_groups: bool = False

    @classmethod
    def create(cls, **semantic: object) -> "IntraDieV2SearchDecision":
        return cls(
            schema_version=INTRA_DIE_V2_SEARCH_DECISION_SCHEMA_VERSION,
            producer_pass="intra_die_refine",
            id=stable_artifact_id(
                "intra_die_v2_search_decision",
                semantic,
                schema_version=INTRA_DIE_V2_SEARCH_DECISION_SCHEMA_VERSION,
            ),
            **semantic,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_projection_id",
                "source_ir1_id",
                "budget",
                "timing_model_ref",
                "hardware_digest",
                "simulation_digest",
                "mode",
                "candidates",
                "rejected_candidates",
                "selected_candidate_ref",
                "selection_reason",
                "generated_candidate_count",
                "full_analytic_evaluation_count",
                "simulator_calls_during_search",
                "reserved_simulator_calls_for_final_evidence",
                "require_full_compute_groups",
            )
        }

    def validate(self, path: str = "intra_die_v2_search_decision") -> None:
        if self.schema_version != INTRA_DIE_V2_SEARCH_DECISION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "intra_die_refine":
            raise SchemaError("must be intra_die_refine", path=f"{path}.producer_pass")
        validate_nonempty(self.source_projection_id, f"{path}.source_projection_id")
        validate_nonempty(self.source_ir1_id, f"{path}.source_ir1_id")
        if type(self.budget) is not IntraDieV2SearchBudget:
            raise SchemaError("must be IntraDieV2SearchBudget", path=f"{path}.budget")
        self.budget.validate(f"{path}.budget")
        validate_nonempty(self.timing_model_ref, f"{path}.timing_model_ref")
        for name in ("hardware_digest", "simulation_digest"):
            digest = getattr(self, name)
            if (
                type(digest) is not str or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SchemaError("must be a lowercase SHA256 digest", path=f"{path}.{name}")
        if type(self.mode) is not IntraDieOptimizationMode:
            raise SchemaError("must be an IntraDieOptimizationMode", path=f"{path}.mode")
        if type(self.require_full_compute_groups) is not bool:
            raise SchemaError(
                "must be bool", path=f"{path}.require_full_compute_groups"
            )
        if not 1 <= len(self.candidates) <= self.budget.max_candidates:
            raise SchemaError("candidate count exceeds the frozen search bounds", path=f"{path}.candidates")
        ids: set[str] = set()
        for index, candidate in enumerate(self.candidates):
            if type(candidate) is not IntraDieV2Candidate:
                raise SchemaError("must be IntraDieV2Candidate", path=f"{path}.candidates[{index}]")
            candidate.validate(f"{path}.candidates[{index}]")
            if candidate.id in ids:
                raise SchemaError("duplicate candidate", path=f"{path}.candidates[{index}].id")
            ids.add(candidate.id)
        identity = tuple(
            item for item in self.candidates
            if item.kind is IntraDieV2CandidateKind.IDENTITY
        )
        if len(identity) != 1:
            raise SchemaError("identity candidate must be retained exactly once", path=f"{path}.candidates")
        expected_order = tuple(
            sorted(
                self.candidates,
                key=lambda item: (
                    item.analytic_cost.predicted_makespan_cycles,
                    0 if item.kind is IntraDieV2CandidateKind.IDENTITY else 1,
                    item.split_k_parts,
                    item.enable_double_buffer,
                    item.enable_streaming_reduce,
                    item.enable_tree_reduce,
                    item.enable_direct_dma,
                    item.id,
                ),
            )
        )
        if self.candidates != expected_order:
            raise SchemaError("candidates must use canonical analytic rank order", path=path)
        if type(self.rejected_candidates) is not tuple:
            raise SchemaError("must be a tuple", path=f"{path}.rejected_candidates")
        rejection_ids: set[str] = set()
        for index, rejection in enumerate(self.rejected_candidates):
            if type(rejection) is not IntraDieV2CandidateRejection:
                raise SchemaError("must be IntraDieV2CandidateRejection", path=f"{path}.rejected_candidates[{index}]")
            rejection.validate(f"{path}.rejected_candidates[{index}]")
            if rejection.id in rejection_ids:
                raise SchemaError("duplicate rejection", path=f"{path}.rejected_candidates[{index}].id")
            rejection_ids.add(rejection.id)
        expected_rejections = tuple(sorted(
            self.rejected_candidates,
            key=lambda item: (
                item.candidate_name, item.split_k_parts,
                item.enable_double_buffer, item.id,
            ),
        ))
        if self.rejected_candidates != expected_rejections:
            raise SchemaError("rejections must use canonical order", path=f"{path}.rejected_candidates")
        selected = next((item for item in self.candidates if item.id == self.selected_candidate_ref), None)
        if selected is None:
            raise SchemaError("selected candidate must be retained", path=f"{path}.selected_candidate_ref")
        if self.mode is IntraDieOptimizationMode.OFF:
            if (
                len(self.candidates) != 1
                or selected.kind is not IntraDieV2CandidateKind.IDENTITY
                or self.rejected_candidates
                or self.selection_reason != "off_identity_only"
            ):
                raise SchemaError("OFF must select only identity", path=path)
        elif self.mode is IntraDieOptimizationMode.FORCE:
            if self.rejected_candidates:
                raise SchemaError("FORCE cannot retain rejected candidates", path=path)
            expected_reason = (
                "explicit_identity_request"
                if selected.kind is IntraDieV2CandidateKind.IDENTITY
                else "explicit_split_k_request"
            )
            if self.selection_reason != expected_reason:
                raise SchemaError("selection must preserve the explicit split-K/identity request", path=path)
        else:
            eligible_auto = (
                tuple(
                    item for item in self.candidates
                    if item.kind is not IntraDieV2CandidateKind.IDENTITY
                )
                if self.require_full_compute_groups
                else self.candidates
            )
            if not eligible_auto or selected is not eligible_auto[0]:
                raise SchemaError(
                    "AUTO must select minimum feasible predicted makespan",
                    path=path,
                )
            expected_reason = (
                "auto_minimum_required_compute_groups"
                if self.require_full_compute_groups
                else (
                    "auto_identity_no_profitable_candidate"
                    if selected.kind is IntraDieV2CandidateKind.IDENTITY
                    else "auto_minimum_predicted_makespan"
                )
            )
            if self.selection_reason != expected_reason:
                raise SchemaError("AUTO selection reason is inconsistent", path=path)
            identity_cycles = identity[0].analytic_cost.predicted_makespan_cycles
            if not self.require_full_compute_groups and any(
                item.kind is not IntraDieV2CandidateKind.IDENTITY
                and item.analytic_cost.predicted_makespan_cycles >= identity_cycles
                for item in self.candidates
            ):
                raise SchemaError("AUTO must eliminate non-profitable split-K", path=path)
        for name in (
            "generated_candidate_count",
            "full_analytic_evaluation_count",
            "simulator_calls_during_search",
            "reserved_simulator_calls_for_final_evidence",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        closed_count = len(self.candidates) + len(self.rejected_candidates)
        if (
            self.generated_candidate_count != closed_count
            or self.full_analytic_evaluation_count != closed_count
            or closed_count > self.budget.max_candidates
            or self.simulator_calls_during_search != 0
            or self.simulator_calls_during_search
            + self.reserved_simulator_calls_for_final_evidence
            > self.budget.simulator_call_budget
        ):
            raise SchemaError("search counters do not close the frozen budget", path=path)
        expected = stable_artifact_id(
            "intra_die_v2_search_decision",
            self._semantic_key(),
            schema_version=INTRA_DIE_V2_SEARCH_DECISION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

__all__ = [
    "INTRA_DIE_V2_ANALYTIC_MODEL_VERSION",
    "INTRA_DIE_V2_CANDIDATE_SCHEMA_VERSION",
    "INTRA_DIE_V2_CANDIDATE_REJECTION_SCHEMA_VERSION",
    "INTRA_DIE_V2_SEARCH_BUDGET_SCHEMA_VERSION",
    "INTRA_DIE_V2_SEARCH_DECISION_SCHEMA_VERSION",
    "IntraDieV2AnalyticCost",
    "IntraDieV2Candidate",
    "IntraDieV2CandidateRejection",
    "IntraDieV2CandidateKind",
    "IntraDieV2SearchBudget",
    "IntraDieV2SearchDecision",
]
