"""The paged TP6 audit must identify the same real Dies as production source."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

from .run_dense_inference_tp6_paged_offload_full_fresh import _active_for_mesh, run
from .run_dense_sequence_runtime_canary import _six_die_fixed_model_case


class DenseInferenceTp6PagedMeshTest(unittest.TestCase):
    def test_representative_rectangles_bind_exact_six_physical_dies(self) -> None:
        for rows, columns in ((2, 3), (3, 2), (3, 3), (1, 6), (6, 1),
                              (4, 7), (10, 10)):
            with self.subTest(mesh=(rows, columns)):
                actual = _active_for_mesh(f"{rows}x{columns}")
                source, _template, _fabric = _six_die_fixed_model_case(
                    rows, columns,
                )
                self.assertEqual(actual, (
                    rows, columns, list(source.placement.active_die_ids),
                ))
                self.assertEqual(len(actual[2]), 6)
                self.assertEqual(len(set(actual[2])), 6)

    def test_compact_receipt_must_outlive_managed_scratch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scratch" / "full"
            with self.assertRaisesRegex(ValueError, "outlive"):
                run(argparse.Namespace(
                    mesh_size="1x6", output_root=root,
                    receipt_output=root / "receipt.json",
                ))
            self.assertFalse(root.exists())

    def test_noncanonical_or_too_small_mesh_is_rejected(self) -> None:
        for mesh in ("01x6", "1x05", "1X6", "1x2x3", "0x6", "1x5",
                     "11x1", "1x11", "1x0", "1x-6"):
            with self.subTest(mesh=mesh), self.assertRaises(ValueError):
                _active_for_mesh(mesh)


if __name__ == "__main__":
    unittest.main()
