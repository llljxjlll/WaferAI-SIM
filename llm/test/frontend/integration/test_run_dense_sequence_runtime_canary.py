"""Check the rectangular four-die fixture before expensive production linking."""

from __future__ import annotations

import unittest

from llm.test.frontend.integration.run_dense_sequence_runtime_canary import (
    _four_die_rect_case,
)


class RectangularDenseSequenceFixtureTest(unittest.TestCase):
    def test_four_active_dies_preserve_graph_and_physical_mesh(self) -> None:
        for rows, columns in ((1, 4), (4, 1)):
            with self.subTest(rows=rows, columns=columns):
                manifest, template, fabric = _four_die_rect_case(rows, columns)
                manifest.validate()
                template.validate()
                fabric.validate("rect_fabric")
                self.assertEqual(manifest.request.mesh.rank_count, 4)
                self.assertEqual(manifest.request.parallel.tp, 4)
                self.assertEqual(
                    tuple(rank.die_id for rank in manifest.placement.rank_placements),
                    (0, 1, 2, 3),
                )
                self.assertEqual(manifest.placement.mesh.rows, rows)
                self.assertEqual(manifest.placement.mesh.columns, columns)
                self.assertEqual(fabric.die_grid, (columns, rows))
                self.assertEqual(len(fabric.dies), 4)
                self.assertEqual(len(fabric.links), 6)
                forward = next(
                    link for link in fabric.links
                    if link.source_die == 0 and link.destination_die == 1
                )
                self.assertEqual(
                    forward.source_port_ref,
                    "c2c_port_d0_p3" if rows == 1 else "c2c_port_d0_p0",
                )
                self.assertEqual(manifest.logical_graph.request, manifest.request)
                inference = manifest.request.steps.inference
                self.assertIsNotNone(inference)
                self.assertEqual(inference.decode_steps, 2)


if __name__ == "__main__":
    unittest.main()
