from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.swizzle import (
    SWIZZLE_CANDIDATE_SCHEMA_VERSION,
    SWIZZLE_COST_SCHEMA_VERSION,
    SWIZZLE_DECISION_SCHEMA_VERSION,
    SWIZZLE_HARDWARE_PROFILE_SCHEMA_VERSION,
    SwizzleCandidate,
    SwizzleDecision,
    SwizzleDecisionReason,
)

from test_swizzle_schema import SwizzleSchemaTest


def _old(version: str) -> str:
    return version.rsplit("/", 1)[0] + "/v0"


class SwizzleVersionBoundaryTest(unittest.TestCase):
    def _baseline(self) -> tuple[object, SwizzleCandidate]:
        fixture = SwizzleSchemaTest()
        problem, witness = fixture._problem()
        from llm.frontend.wafer_frontend.policies.swizzle.cost import (
            build_unfused_baseline,
        )

        return problem, build_unfused_baseline(problem, witness)

    def test_every_top_level_carrier_rejects_immediate_old_version(self) -> None:
        problem, baseline = self._baseline()
        decision = SwizzleDecision.create(
            problem=problem,
            baseline=baseline,
            ranked_candidates=(baseline,),
            selected_candidate_ref=baseline.id,
            decision_reason=SwizzleDecisionReason.BASELINE_ONLY,
        )
        cases = (
            replace(
                problem.hardware_profile,
                schema_version=_old(SWIZZLE_HARDWARE_PROFILE_SCHEMA_VERSION),
            ),
            replace(
                baseline.cost,
                schema_version=_old(SWIZZLE_COST_SCHEMA_VERSION),
            ),
            replace(
                baseline,
                schema_version=_old(SWIZZLE_CANDIDATE_SCHEMA_VERSION),
            ),
            replace(
                decision,
                schema_version=_old(SWIZZLE_DECISION_SCHEMA_VERSION),
            ),
        )
        for artifact in cases:
            with self.subTest(artifact=type(artifact).__name__):
                with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
                    artifact.validate()

    def test_nested_profile_and_cost_tamper_do_not_restable(self) -> None:
        problem, baseline = self._baseline()
        with self.assertRaisesRegex(SchemaError, "profile digest"):
            replace(
                problem.hardware_profile,
                peak_flops_per_cycle=2048.0,
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(
                baseline.cost,
                upper_cycles=baseline.cost.upper_cycles + 1.0,
            ).validate()


if __name__ == "__main__":
    unittest.main()
