from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from llm.frontend.wafer_frontend.policies.intra_die_v2_search import (
    evaluate_intra_die_v2_candidates,
)
from llm.frontend.wafer_frontend.runner import (
    _intra_die_v2_calibration_notes,
    _write_intra_die_v2_calibration_evidence,
)
from llm.frontend.wafer_frontend.schema.intra_die_refine import (
    SplitKRefineOptions,
)
from llm.frontend.wafer_frontend.schema.intra_die_v2_calibration import (
    INTRA_DIE_V2_MAX_RELATIVE_ERROR,
    IntraDieV2CalibrationEvidence,
)
from llm.frontend.wafer_frontend.schema.serde import (
    from_data,
    load_json_value,
    to_primitive,
)

from test_split_k_intra_die_refine import _source_projection


def _decision():
    graph, projection = _source_projection()
    return evaluate_intra_die_v2_candidates(
        projection,
        graph,
        SplitKRefineOptions(
            split_k_parts=2,
            enable_reduce=True,
            enable_double_buffer=True,
        ),
    )


def _predicted(decision) -> int:
    return next(
        item.analytic_cost.predicted_makespan_cycles
        for item in decision.candidates
        if item.id == decision.selected_candidate_ref
    )


class IntraDieV2CalibrationEvidenceTest(unittest.TestCase):
    def test_at_or_below_twenty_percent_is_calibrated_and_round_trips(self) -> None:
        decision = _decision()
        predicted = _predicted(decision)
        evidence = IntraDieV2CalibrationEvidence.create(
            decision,
            simulator_measured_makespan_cycles=predicted,
            simulator_calls_used=2,
            repeat_signature_stable=True,
        )

        self.assertTrue(evidence.calibrated)
        self.assertEqual(evidence.relative_error, 0.0)
        decoded = from_data(
            IntraDieV2CalibrationEvidence,
            to_primitive(evidence),
            path="evidence",
        )
        decoded.validate_against(decision)
        self.assertEqual(decoded, evidence)

        # Exact boundary: predicted=4, measured=5 gives 20% measured-relative
        # error.  Rebind the measured value only when this decision admits an
        # integral 20% point; otherwise the exact-zero case above covers <=.
        if predicted % 4 == 0:
            boundary = IntraDieV2CalibrationEvidence.create(
                decision,
                simulator_measured_makespan_cycles=predicted * 5 // 4,
                simulator_calls_used=2,
                repeat_signature_stable=True,
            )
            self.assertEqual(boundary.relative_error, INTRA_DIE_V2_MAX_RELATIVE_ERROR)
            self.assertTrue(boundary.calibrated)

    def test_above_twenty_percent_is_recorded_without_failing(self) -> None:
        decision = _decision()
        predicted = _predicted(decision)
        evidence = IntraDieV2CalibrationEvidence.create(
            decision,
            simulator_measured_makespan_cycles=predicted * 2,
            simulator_calls_used=2,
            repeat_signature_stable=True,
        )

        evidence.validate_against(decision)
        self.assertEqual(evidence.relative_error, 0.5)
        self.assertFalse(evidence.calibrated)

    def test_tampering_and_call_budget_drift_fail_closed(self) -> None:
        decision = _decision()
        predicted = _predicted(decision)
        evidence = IntraDieV2CalibrationEvidence.create(
            decision,
            simulator_measured_makespan_cycles=predicted,
            simulator_calls_used=2,
            repeat_signature_stable=True,
        )

        with self.assertRaisesRegex(Exception, "exact prediction error"):
            replace(evidence, relative_error=0.1).validate()
        with self.assertRaisesRegex(Exception, "error threshold"):
            replace(evidence, calibrated=False).validate()
        with self.assertRaisesRegex(Exception, "unstable artifact id"):
            replace(evidence, id="forged").validate()
        with self.assertRaisesRegex(Exception, "two/three-call budget"):
            replace(evidence, simulator_calls_used=1).validate()
        with self.assertRaisesRegex(Exception, "two/three-call budget"):
            IntraDieV2CalibrationEvidence.create(
                decision,
                simulator_measured_makespan_cycles=predicted,
                simulator_calls_used=3,
                repeat_signature_stable=True,
            )

    def test_runner_writer_emits_versioned_json_only_for_search_decisions(self) -> None:
        decision = _decision()
        predicted = _predicted(decision)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "intra_die_v2_calibration_evidence.json"
            emitted = _write_intra_die_v2_calibration_evidence(
                output,
                (decision,),
                simulator_measured_makespan_cycles=predicted * 2,
                simulator_calls_used=2,
                repeat_signature_stable=True,
            )
            self.assertTrue(output.is_file())
            self.assertFalse(emitted[0].calibrated)
            notes = _intra_die_v2_calibration_notes(emitted)
            self.assertTrue(any("calibrated=false" in note for note in notes))
            self.assertTrue(any("paper-grade" in note and "forbids" in note for note in notes))
            raw_rows = load_json_value(output, path="runner_output")
            self.assertIsInstance(raw_rows, list)
            assert isinstance(raw_rows, list)
            decoded = from_data(
                IntraDieV2CalibrationEvidence,
                raw_rows[0],
                path="runner_output[0]",
            )
            decoded.validate_against(decision)
            self.assertEqual(decoded, emitted[0])

            absent = root / "absent.json"
            self.assertEqual(
                _write_intra_die_v2_calibration_evidence(
                    absent,
                    (),
                    simulator_measured_makespan_cycles=predicted,
                    simulator_calls_used=2,
                    repeat_signature_stable=True,
                ),
                (),
            )
            self.assertFalse(absent.exists())


if __name__ == "__main__":
    unittest.main()
