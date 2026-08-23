from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.policies.swizzle.meshslice_2d import (
    generate_meshslice_2d_drafts,
)

from test_swizzle_meshslice_cost import _problem


class SwizzleMeshSliceScaleTest(unittest.TestCase):
    def test_four_by_four_rectangle_generates_bounded_candidates(self) -> None:
        problem, witness = _problem(4, 4, k=64)
        drafts = generate_meshslice_2d_drafts(problem, witness)

        self.assertTrue(drafts)
        self.assertTrue(
            all(
                len(draft.rank_programs) == 16
                and draft.chunk_count <= problem.constraints.max_chunk_count
                and sum(len(program.actions) for program in draft.rank_programs)
                <= problem.constraints.max_actions
                for draft in drafts
            )
        )
        self.assertTrue(
            all(
                len(draft.topology_witness.row_orders) == 4
                and len(draft.topology_witness.column_orders) == 4
                for draft in drafts
            )
        )

    def test_rectangular_orientation_transposes_row_and_column_orders(self) -> None:
        wide_problem, wide_witness = _problem(2, 4, k=64)
        tall_problem, tall_witness = _problem(4, 2, k=64)
        wide = generate_meshslice_2d_drafts(wide_problem, wide_witness)[0]
        tall = generate_meshslice_2d_drafts(tall_problem, tall_witness)[0]

        self.assertEqual(
            (
                len(wide.topology_witness.row_orders),
                len(wide.topology_witness.row_orders[0]),
                len(wide.topology_witness.column_orders),
                len(wide.topology_witness.column_orders[0]),
            ),
            (2, 4, 4, 2),
        )
        self.assertEqual(
            (
                len(tall.topology_witness.row_orders),
                len(tall.topology_witness.row_orders[0]),
                len(tall.topology_witness.column_orders),
                len(tall.topology_witness.column_orders[0]),
            ),
            (4, 2, 2, 4),
        )


if __name__ == "__main__":
    unittest.main()
