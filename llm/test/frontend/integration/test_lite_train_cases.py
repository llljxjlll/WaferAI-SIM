from __future__ import annotations

import unittest

from lite_train_cases import (
    build_s2_lite_production_case,
    build_s2_lite_source_case,
)


class S2LiteSourceCaseTest(unittest.TestCase):
    def test_self_contained_production_source_and_determinism(self) -> None:
        first = build_s2_lite_source_case()
        second = build_s2_lite_source_case()
        self.assertEqual(first, second)
        self.assertEqual(first.contract.case_id, "case.s2_lite.lm_head_train")
        self.assertEqual(
            (
                len(first.logical.graph.nodes),
                len(first.logical.graph.values),
                len(first.logical.graph.edges),
                len(first.logical.graph.persistent_states),
                len(first.logical.graph.state_accesses),
            ),
            (29, 47, 34, 15, 16),
        )
        self.assertEqual(first.oracle.total_floating_point_ops, 9_216)
        self.assertIn('"die": {"x": 2, "y": 1}', first.runtime_inputs.hardware_json)
        self.assertTrue(first.runtime_inputs.mapping_text.strip())

    def test_production_chain_reaches_exact_global_action(self) -> None:
        case = build_s2_lite_production_case()
        self.assertEqual(
            (
                len(case.placed.replicas[0].graph.nodes),
                sum(
                    len(dag.tasks)
                    for dag in case.projected.replicas[0].projection.dags
                ),
                len(case.scheduled.replicas[0].schedule_set.schedules[0].placements),
                len(case.global_action.global_dags[0].actions),
            ),
            (29, 46, 46, 46),
        )


if __name__ == "__main__":
    unittest.main()
