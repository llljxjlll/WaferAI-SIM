from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.compiler import _validate_rect_mesh_compile_inputs
from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    _segment_spec,
    _validate_inputs,
)
from llm.frontend.wafer_frontend.schema.experiment import PlacementStrategy
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.test.frontend.integration.run_dense_sequence_runtime_canary import (
    _all_die_scaled_model_case,
)
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces


class DenseAllDieScaledModelTest(unittest.TestCase):
    def test_nine_die_main_source_uses_every_rank_and_three_sequence_phases(self) -> None:
        manifest, template, fabric = _all_die_scaled_model_case(3, 3)
        request = manifest.request
        self.assertEqual((request.model.num_layers, request.parallel.tp), (2, 9))
        self.assertEqual((request.model.hidden_size,
                          request.model.intermediate_size,
                          request.model.num_attention_heads,
                          request.model.num_kv_heads,
                          request.model.head_dim), (9, 18, 9, 9, 1))
        self.assertEqual((request.steps.inference.prefill_tokens,
                          request.steps.inference.decode_steps,
                          request.steps.inference.request_count), (1, 2, 9))
        self.assertEqual(manifest.placement.active_die_ids, tuple(range(9)))
        self.assertEqual(manifest.placement.idle_die_ids, ())
        self.assertEqual({item.step for item in manifest.logical_graph.operations},
                         {0, 1, 2})
        self.assertEqual({item.layer for item in manifest.logical_graph.operations
                          if item.layer is not None}, {0, 1})
        self.assertEqual(len(fabric.dies), 9)
        self.assertEqual(len(fabric.links), 24)

        _validate_inputs(manifest, template, fabric,
                         valid_hbm_address_spaces(fabric))
        for index, profile_tokens in enumerate(((9, 0), (0, 9), (0, 9))):
            with self.subTest(index=index):
                spec = _segment_spec(template, manifest, index)
                self.assertIs(spec.placement.strategy, PlacementStrategy.COMPACT)
                profile = spec.workload.infer.profile
                self.assertEqual((profile.prefill_tokens, profile.decode_tokens),
                                 profile_tokens)
                _validate_rect_mesh_compile_inputs(spec, fabric,
                                                   RectMeshSpec(3, 3))

    def test_hundred_die_candidate_has_full_fabric_but_no_runtime_claim(self) -> None:
        manifest, template, fabric = _all_die_scaled_model_case(10, 10)
        self.assertEqual(manifest.request.parallel.tp, 100)
        self.assertEqual(manifest.placement.active_die_ids, tuple(range(100)))
        self.assertEqual(manifest.placement.idle_die_ids, ())
        self.assertEqual(len(fabric.dies), 100)
        self.assertEqual(len(fabric.links), 360)
        self.assertEqual(len(manifest.logical_graph.operations), 96)
        _validate_inputs(manifest, template, fabric,
                         valid_hbm_address_spaces(fabric))
        for index in range(3):
            _validate_rect_mesh_compile_inputs(
                _segment_spec(template, manifest, index),
                fabric,
                RectMeshSpec(10, 10),
            )
        self.assertNotEqual(manifest.request.case_id,
                            _all_die_scaled_model_case(3, 3)[0].request.case_id)

    def test_scaled_source_rejects_shapes_outside_frozen_release_envelope(self) -> None:
        for dimensions in ((0, 1), (11, 1), (1, 11)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                _all_die_scaled_model_case(*dimensions)

    def test_hundred_rectangles_have_distinct_full_die_source_cases(self) -> None:
        case_ids = set()
        for rows in range(1, 11):
            for columns in range(1, 11):
                with self.subTest(rows=rows, columns=columns):
                    manifest, template, fabric = _all_die_scaled_model_case(
                        rows, columns
                    )
                    die_count = rows * columns
                    self.assertEqual(manifest.request.parallel.tp, die_count)
                    self.assertEqual(manifest.placement.active_die_ids,
                                     tuple(range(die_count)))
                    self.assertEqual(manifest.placement.idle_die_ids, ())
                    self.assertEqual(len(fabric.dies), die_count)
                    self.assertEqual(len(fabric.links),
                                     2 * ((rows - 1) * columns + rows * (columns - 1)))
                    _validate_inputs(manifest, template, fabric,
                                     valid_hbm_address_spaces(fabric))
                    for index in range(3):
                        _validate_rect_mesh_compile_inputs(
                            _segment_spec(template, manifest, index), fabric,
                            RectMeshSpec(rows, columns)
                        )
                    case_ids.add(manifest.request.case_id)
        self.assertEqual(len(case_ids), 100)


if __name__ == "__main__":
    unittest.main()
