"""Fixed global model placement across physically different rectangles."""

import contextlib
import io
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.compiler import _validate_rect_mesh_compile_inputs
from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    _segment_spec, _validate_inputs,
)
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.integration.run_dense_sequence_runtime_canary import (
    _parse_args, _six_die_fixed_model_case,
)
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces


class DenseFixedGlobalTp6SourceTest(unittest.TestCase):
    def test_same_model_spans_sparse_3x3_and_10x10(self) -> None:
        signatures = set()
        for rows, columns, expected in (
            (3, 3, (0, 1, 2, 6, 7, 8)),
            (10, 10, (0, 19, 39, 59, 79, 99)),
        ):
            with self.subTest(mesh=(rows, columns)):
                manifest, template, fabric = _six_die_fixed_model_case(rows, columns)
                model = manifest.request.model
                signatures.add((model.num_layers, model.hidden_size,
                                model.intermediate_size, model.num_attention_heads,
                                model.num_kv_heads, model.head_dim,
                                manifest.request.steps.inference.prefill_tokens,
                                manifest.request.steps.inference.decode_steps,
                                manifest.request.steps.inference.request_count))
                self.assertEqual(manifest.request.parallel.tp, 6)
                self.assertEqual(manifest.placement.active_die_ids, expected)
                self.assertEqual(len(fabric.dies), rows * columns)
                self.assertEqual(len(manifest.placement.idle_die_ids),
                                 rows * columns - 6)
                _validate_inputs(manifest, template, fabric,
                                 valid_hbm_address_spaces(fabric))
                for index in range(3):
                    spec = _segment_spec(template, manifest, index)
                    _validate_rect_mesh_compile_inputs(
                        spec, fabric, RectMeshSpec(rows, columns),
                    )
                    expanded = logical_expand(build_ir0(spec))
                    self.assertGreater(len(expanded.entries[0].graph.nodes), 0)
        self.assertEqual(len(signatures), 1)

    def test_every_release_rectangle_with_six_dies_has_a_stable_subset(self) -> None:
        case_ids = set()
        signatures = set()
        accepted = 0
        for rows in range(1, 11):
            for columns in range(1, 11):
                if rows * columns < 6:
                    continue
                with self.subTest(mesh=(rows, columns)):
                    manifest, template, fabric = _six_die_fixed_model_case(
                        rows, columns,
                    )
                    active = manifest.placement.active_die_ids
                    self.assertEqual(len(active), 6)
                    self.assertEqual(tuple(sorted(set(active))), active)
                    self.assertTrue(all(0 <= die_id < rows * columns
                                        for die_id in active))
                    self.assertEqual(len(fabric.dies), rows * columns)
                    _validate_inputs(manifest, template, fabric,
                                     valid_hbm_address_spaces(fabric))
                    case_ids.add(manifest.request.case_id)
                    model = manifest.request.model
                    signatures.add((model.num_layers, model.hidden_size,
                                    model.intermediate_size, model.head_dim,
                                    model.num_attention_heads, model.num_kv_heads))
                    accepted += 1
        self.assertEqual(len(case_ids), accepted)
        self.assertEqual(len(signatures), 1)

    def test_cli_accepts_fixed_10x10_and_rejects_conflicting_or_small_mode(self) -> None:
        with patch("sys.argv", ["runner", "--fixed-global-tp6",
                                "--mesh-size", "10x10"]):
            args = _parse_args()
        self.assertEqual(args.mesh_size, "10x10")
        self.assertTrue(args.fixed_global_tp6)
        self.assertFalse(args.scaled_all_dies)
        for extra in (("--mesh-size", "2x2"),
                      ("--mesh-size", "10x10", "--scaled-all-dies")):
            with (
                self.subTest(extra=extra),
                patch("sys.argv", ["runner", "--fixed-global-tp6", *extra]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                _parse_args()
            self.assertEqual(raised.exception.code, 2)

    def test_small_mesh_is_rejected_without_changing_the_model(self) -> None:
        for dimensions in ((1, 1), (1, 5), (2, 2), (11, 1)):
            with self.subTest(mesh=dimensions), self.assertRaises(ValueError):
                _six_die_fixed_model_case(*dimensions)

    def test_six_die_source_is_unchanged(self) -> None:
        for rows, columns in ((1, 6), (6, 1), (2, 3), (3, 2)):
            with self.subTest(mesh=(rows, columns)):
                manifest, _, _ = _six_die_fixed_model_case(rows, columns)
                self.assertEqual(manifest.placement.active_die_ids,
                                 tuple(range(6)))


if __name__ == "__main__":
    unittest.main()
