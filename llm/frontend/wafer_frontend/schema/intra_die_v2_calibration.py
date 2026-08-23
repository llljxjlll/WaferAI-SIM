"""Measured final-run evidence for the intra-die v2 analytic model.

Candidate enumeration is analytic-only.  This artifact binds the selected
candidate's prediction to the two simulator calls already required by the
final timing stability gate; it never authorizes extra search-time calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ..errors import SchemaError
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .intra_die_v2_search import (
    IntraDieV2SearchDecision,
)


INTRA_DIE_V2_CALIBRATION_EVIDENCE_SCHEMA_VERSION = (
    "wafer_frontend.intra_die_v2_calibration_evidence/v1alpha1"
)
INTRA_DIE_V2_MAX_RELATIVE_ERROR = 0.20
INTRA_DIE_V2_FINAL_EVIDENCE_SIMULATOR_CALLS = (2, 3)


@dataclass(frozen=True, slots=True)
class IntraDieV2CalibrationEvidence:
    schema_version: str
    producer_pass: str
    id: str
    source_search_decision_ref: str
    source_projection_id: str
    source_ir1_id: str
    selected_candidate_ref: str
    analytic_model_version: str
    predicted_makespan_cycles: int
    simulator_measured_makespan_cycles: int
    relative_error: float
    relative_error_threshold: float
    calibrated: bool
    simulator_calls_used: int
    reserved_simulator_calls_for_final_evidence: int
    repeat_signature_stable: bool

    @classmethod
    def create(
        cls,
        decision: IntraDieV2SearchDecision,
        *,
        simulator_measured_makespan_cycles: int,
        simulator_calls_used: int,
        repeat_signature_stable: bool,
    ) -> "IntraDieV2CalibrationEvidence":
        if type(decision) is not IntraDieV2SearchDecision:
            raise SchemaError(
                "must be IntraDieV2SearchDecision", path="decision"
            )
        decision.validate("decision")
        validate_uint64(
            simulator_measured_makespan_cycles,
            "simulator_measured_makespan_cycles",
        )
        if simulator_measured_makespan_cycles == 0:
            raise SchemaError(
                "must be positive", path="simulator_measured_makespan_cycles"
            )
        selected = next(
            candidate
            for candidate in decision.candidates
            if candidate.id == decision.selected_candidate_ref
        )
        predicted = selected.analytic_cost.predicted_makespan_cycles
        relative_error = (
            abs(predicted - simulator_measured_makespan_cycles)
            / simulator_measured_makespan_cycles
        )
        semantic = {
            "source_search_decision_ref": decision.id,
            "source_projection_id": decision.source_projection_id,
            "source_ir1_id": decision.source_ir1_id,
            "selected_candidate_ref": selected.id,
            "analytic_model_version": decision.budget.analytic_model_version,
            "predicted_makespan_cycles": predicted,
            "simulator_measured_makespan_cycles": simulator_measured_makespan_cycles,
            "relative_error": relative_error,
            "relative_error_threshold": INTRA_DIE_V2_MAX_RELATIVE_ERROR,
            "calibrated": relative_error <= INTRA_DIE_V2_MAX_RELATIVE_ERROR,
            "simulator_calls_used": simulator_calls_used,
            "reserved_simulator_calls_for_final_evidence": (
                decision.reserved_simulator_calls_for_final_evidence
            ),
            "repeat_signature_stable": repeat_signature_stable,
        }
        evidence = cls(
            schema_version=INTRA_DIE_V2_CALIBRATION_EVIDENCE_SCHEMA_VERSION,
            producer_pass="naive_runner",
            id=stable_artifact_id(
                "intra_die_v2_calibration_evidence",
                semantic,
                schema_version=INTRA_DIE_V2_CALIBRATION_EVIDENCE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        evidence.validate_against(decision)
        return evidence

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "source_search_decision_ref",
                "source_projection_id",
                "source_ir1_id",
                "selected_candidate_ref",
                "analytic_model_version",
                "predicted_makespan_cycles",
                "simulator_measured_makespan_cycles",
                "relative_error",
                "relative_error_threshold",
                "calibrated",
                "simulator_calls_used",
                "reserved_simulator_calls_for_final_evidence",
                "repeat_signature_stable",
            )
        }

    def validate(
        self, path: str = "intra_die_v2_calibration_evidence"
    ) -> None:
        if self.schema_version != INTRA_DIE_V2_CALIBRATION_EVIDENCE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "naive_runner":
            raise SchemaError("must be 'naive_runner'", path=f"{path}.producer_pass")
        for name in (
            "source_search_decision_ref",
            "source_projection_id",
            "source_ir1_id",
            "selected_candidate_ref",
            "analytic_model_version",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        for name in (
            "predicted_makespan_cycles",
            "simulator_measured_makespan_cycles",
            "simulator_calls_used",
            "reserved_simulator_calls_for_final_evidence",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.simulator_measured_makespan_cycles == 0:
            raise SchemaError("must be positive", path=f"{path}.simulator_measured_makespan_cycles")
        if type(self.relative_error) is not float or not math.isfinite(self.relative_error):
            raise SchemaError("must be a finite float", path=f"{path}.relative_error")
        if (
            type(self.relative_error_threshold) is not float
            or self.relative_error_threshold != INTRA_DIE_V2_MAX_RELATIVE_ERROR
        ):
            raise SchemaError(
                "must equal the frozen 0.20 threshold",
                path=f"{path}.relative_error_threshold",
            )
        expected_error = (
            abs(
                self.predicted_makespan_cycles
                - self.simulator_measured_makespan_cycles
            )
            / self.simulator_measured_makespan_cycles
        )
        if self.relative_error != expected_error:
            raise SchemaError(
                "does not equal the exact prediction error",
                path=f"{path}.relative_error",
            )
        if type(self.calibrated) is not bool or self.calibrated != (
            self.relative_error <= self.relative_error_threshold
        ):
            raise SchemaError(
                "does not match the frozen error threshold",
                path=f"{path}.calibrated",
            )
        if type(self.repeat_signature_stable) is not bool or not self.repeat_signature_stable:
            raise SchemaError(
                "final repeat signature must be stable",
                path=f"{path}.repeat_signature_stable",
            )
        if (
            self.simulator_calls_used
            not in INTRA_DIE_V2_FINAL_EVIDENCE_SIMULATOR_CALLS
            or self.reserved_simulator_calls_for_final_evidence
            not in INTRA_DIE_V2_FINAL_EVIDENCE_SIMULATOR_CALLS
            or self.simulator_calls_used
            != self.reserved_simulator_calls_for_final_evidence
        ):
            raise SchemaError(
                "final evidence calls must exactly close a reserved two/three-call budget",
                path=path,
            )
        expected_id = stable_artifact_id(
            "intra_die_v2_calibration_evidence",
            self._semantic_key(),
            schema_version=INTRA_DIE_V2_CALIBRATION_EVIDENCE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id"
            )

    def validate_against(
        self,
        decision: IntraDieV2SearchDecision,
        path: str = "intra_die_v2_calibration_evidence",
    ) -> None:
        self.validate(path)
        if type(decision) is not IntraDieV2SearchDecision:
            raise SchemaError("must be IntraDieV2SearchDecision", path="decision")
        decision.validate("decision")
        selected = next(
            candidate
            for candidate in decision.candidates
            if candidate.id == decision.selected_candidate_ref
        )
        expected = (
            decision.id,
            decision.source_projection_id,
            decision.source_ir1_id,
            decision.selected_candidate_ref,
            decision.budget.analytic_model_version,
            selected.analytic_cost.predicted_makespan_cycles,
            decision.reserved_simulator_calls_for_final_evidence,
        )
        actual = (
            self.source_search_decision_ref,
            self.source_projection_id,
            self.source_ir1_id,
            self.selected_candidate_ref,
            self.analytic_model_version,
            self.predicted_makespan_cycles,
            self.reserved_simulator_calls_for_final_evidence,
        )
        if actual != expected:
            raise SchemaError(
                "does not close against the source search decision", path=path
            )


__all__ = [
    "INTRA_DIE_V2_CALIBRATION_EVIDENCE_SCHEMA_VERSION",
    "INTRA_DIE_V2_FINAL_EVIDENCE_SIMULATOR_CALLS",
    "INTRA_DIE_V2_MAX_RELATIVE_ERROR",
    "IntraDieV2CalibrationEvidence",
]
