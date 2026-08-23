from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.policies.swizzle.cost import (
    build_unfused_baseline,
)
from llm.frontend.wafer_frontend.policies.swizzle.enumerate import (
    materialize_drafts,
)
from llm.frontend.wafer_frontend.policies.swizzle.project import (
    SwizzleProjectionCheckStatus,
    SwizzleProjectionGate,
    preflight_current_ir2_projection,
)
from llm.frontend.wafer_frontend.policies.swizzle.wang_1d import (
    generate_wang_1d_drafts,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind

from test_swizzle_wang_1d import _ag_problem, _post_problem


def _wang_candidate(problem, witness):
    draft = generate_wang_1d_drafts(problem, witness)[0]
    return materialize_drafts(problem, (draft,))[0]


class SwizzleProjectPreflightTest(unittest.TestCase):
    def test_all_patterns_stop_at_executable_action_carrier(self) -> None:
        fixtures = (
            _ag_problem(),
            _post_problem(CollectiveKind.REDUCE_SCATTER),
            _post_problem(CollectiveKind.ALL_REDUCE),
        )
        for problem, witness in fixtures:
            with self.subTest(pattern=problem.pattern.value):
                report = preflight_current_ir2_projection(
                    problem,
                    _wang_candidate(problem, witness),
                )
                report.validate()
                self.assertFalse(report.ready)
                self.assertIsNotNone(report.first_blocker)
                assert report.first_blocker is not None
                self.assertIs(
                    report.first_blocker.gate,
                    SwizzleProjectionGate.FUSION_ACTION_CARRIER,
                )
                self.assertEqual(
                    tuple(check.status for check in report.checks[:4]),
                    (SwizzleProjectionCheckStatus.PASSED,) * 4,
                )
                self.assertTrue(
                    all(
                        check.status is SwizzleProjectionCheckStatus.DEFERRED
                        for check in report.checks[5:]
                    )
                )

    def test_unfused_baseline_is_not_a_projection_input(self) -> None:
        problem, witness = _ag_problem()
        baseline = build_unfused_baseline(problem, witness)
        report = preflight_current_ir2_projection(problem, baseline)

        self.assertIsNotNone(report.first_blocker)
        assert report.first_blocker is not None
        self.assertIs(
            report.first_blocker.gate,
            SwizzleProjectionGate.FUSED_EXECUTION,
        )

    def test_cross_problem_candidate_fails_before_structural_audit(self) -> None:
        ag_problem, ag_witness = _ag_problem()
        rs_problem, _ = _post_problem(CollectiveKind.REDUCE_SCATTER)
        candidate = _wang_candidate(ag_problem, ag_witness)

        report = preflight_current_ir2_projection(rs_problem, candidate)

        self.assertIsNotNone(report.first_blocker)
        assert report.first_blocker is not None
        self.assertIs(
            report.first_blocker.gate,
            SwizzleProjectionGate.CANDIDATE_PROVENANCE,
        )
        self.assertTrue(
            all(
                check.status is SwizzleProjectionCheckStatus.DEFERRED
                for check in report.checks[1:]
            )
        )


if __name__ == "__main__":
    unittest.main()
