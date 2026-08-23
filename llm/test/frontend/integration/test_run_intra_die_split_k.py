from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import yaml

from llm.frontend.wafer_frontend import NaiveRunCase, NaiveRunValidation
from run_intra_die_split_k import build_parser, build_request, require_optimized_intra_die


class IntraDieSplitKCliTest(unittest.TestCase):
    def _args(self, root: Path, *, spec: Path) -> Namespace:
        return build_parser().parse_args(
            [
                "--spec", str(spec),
                "--hardware", str(root / "hardware.json"),
                "--simulation", str(root / "simulation.json"),
                "--mapping", str(root / "mapping.spec"),
                "--npusim", str(root / "npusim"),
                "--finalizer", str(root / "finalizer"),
                "--output", str(root / "result"),
                "--case", "E2",
                "--profile", "decode",
                "--split-k-parts", "4",
                "--enable-reduce",
                "--enable-double-buffer",
                "--repeat", "2",
            ]
        )

    def test_options_are_wired_into_naive_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            spec = root / "spec.yaml"
            spec.write_text(
                yaml.safe_dump({"policy": {"intra_die": "optimized"}}),
                encoding="utf-8",
            )
            args = self._args(root, spec=spec)
            require_optimized_intra_die(spec)
            request = build_request(args)
            self.assertIs(request.case, NaiveRunCase.E2)
            self.assertIs(request.validation, NaiveRunValidation.TIMING)
            self.assertEqual(request.profile_id, "decode")
            self.assertEqual(request.repeat, 2)
            options = request.intra_die_refine_options
            self.assertIsNotNone(options)
            assert options is not None
            self.assertEqual(options.split_k_parts, 4)
            self.assertTrue(options.enable_reduce)
            self.assertTrue(options.enable_double_buffer)

    def test_nonoptimized_spec_fails_before_runner(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            spec = Path(raw) / "spec.yaml"
            spec.write_text(
                yaml.safe_dump({"policy": {"intra_die": "naive"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "intra_die=optimized"):
                require_optimized_intra_die(spec)


if __name__ == "__main__":
    unittest.main()
