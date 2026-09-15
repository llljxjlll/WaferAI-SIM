from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.compiler import _validate_rect_mesh_compile_inputs
from llm.frontend.wafer_frontend.errors import SchemaError, UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    _segment_spec,
    _validate_inputs,
)
from llm.frontend.wafer_frontend.passes.group_registry import build_group_registry
from llm.frontend.wafer_frontend.schema.experiment import (
    ExplicitGroupPlacement,
    PlacementSpec,
    PlacementStrategy,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.test.frontend.integration.run_dense_sequence_runtime_canary import (
    _six_die_fixed_model_case,
)
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_group_registry import (
    context,
    mesh_fabric,
    mesh_graph,
)


class SixDieFixedDenseModelTest(unittest.TestCase):
    def test_horizontal_and_vertical_mesh_keep_one_full_model_and_work(self) -> None:
        horizontal = _six_die_fixed_model_case(2, 3)
        vertical = _six_die_fixed_model_case(3, 2)
        first, last = horizontal[0], vertical[0]
        self.assertNotEqual(first.id, last.id)
        self.assertEqual(first.request.model, last.request.model)
        self.assertEqual(canonical_digest(horizontal[1].model),
                         canonical_digest(vertical[1].model))
        self.assertEqual((first.request.model.num_layers,
                          first.request.model.hidden_size,
                          first.request.model.intermediate_size), (2, 48, 96))
        self.assertEqual((first.request.parallel.tp, last.request.parallel.tp), (6, 6))
        self.assertEqual(
            Counter((op.kind, op.step, op.layer)
                    for op in first.logical_graph.operations),
            Counter((op.kind, op.step, op.layer)
                    for op in last.logical_graph.operations),
        )
        self.assertEqual({op.layer for op in first.logical_graph.operations
                          if op.layer is not None}, {0, 1})
        self.assertEqual({op.step for op in first.logical_graph.operations},
                         {0, 1, 2})
        self.assertEqual((horizontal[2].die_grid, vertical[2].die_grid),
                         ((3, 2), (2, 3)))
        for manifest, template, fabric in (horizontal, vertical):
            self.assertEqual(manifest.placement.active_die_ids, tuple(range(6)))
            self.assertEqual(
                tuple(profile.allocation_alignment_bytes
                      for profile in fabric.sram_profiles), (32,)
            )
            self.assertEqual(
                tuple(profile.capacity_bytes for profile in fabric.sram_profiles),
                (128 * 1024,),
            )
            self.assertEqual(tuple(space.size_bytes for space in
                                   valid_hbm_address_spaces(fabric)),
                             (1 << 30,) * 6)
            self.assertEqual(template.model.V,
                             manifest.request.model.vocabulary_size)
            self.assertEqual(template.model.L,
                             manifest.request.model.num_layers)
            self.assertEqual(template.model.H,
                             manifest.request.model.hidden_size)
            self.assertEqual(template.model.NH,
                             manifest.request.model.num_attention_heads)

    def test_kv_boundaries_are_computed_from_source_heads_and_requests(self) -> None:
        for rows, columns in ((2, 3), (3, 2)):
            manifest, _, _ = _six_die_fixed_model_case(rows, columns)
            model = manifest.request.model
            steps = manifest.request.steps.inference
            assert steps is not None
            per_token = (2 * model.num_layers * model.num_kv_heads
                         * model.head_dim * 2)
            boundary_bytes = tuple(
                (steps.prefill_tokens * steps.request_count
                 + decode * steps.request_count) * per_token
                for decode in range(steps.decode_steps + 1)
            )
            self.assertEqual(boundary_bytes, (13824, 16128, 18432))

    def test_nine_die_materialization_maps_six_ranks_across_idle_row(self) -> None:
        compact, template, _ = _six_die_fixed_model_case(2, 3)
        expanded, expanded_template, fabric = _six_die_fixed_model_case(3, 3)
        self.assertEqual(expanded.request.model, compact.request.model)
        self.assertEqual(canonical_digest(expanded_template.model),
                         canonical_digest(template.model))
        self.assertEqual(expanded.placement.active_die_ids,
                         (0, 1, 2, 6, 7, 8))
        self.assertEqual(expanded.request.parallel.tp, 6)
        self.assertEqual(fabric.die_grid, (3, 3))
        self.assertEqual(tuple(space.size_bytes for space in
                               valid_hbm_address_spaces(fabric)),
                         (1 << 30,) * 9)
        self.assertEqual(
            Counter((op.kind, op.step, op.layer)
                    for op in expanded.logical_graph.operations),
            Counter((op.kind, op.step, op.layer)
                    for op in compact.logical_graph.operations),
        )

    def test_nine_die_production_segments_and_physical_routes_preserve_idle_row(self) -> None:
        manifest, template, fabric = _six_die_fixed_model_case(3, 3)
        active = manifest.placement.active_die_ids
        _validate_inputs(manifest, template, fabric,
                         valid_hbm_address_spaces(fabric))
        for index in range(3):
            with self.subTest(segment=index):
                spec = _segment_spec(template, manifest, index)
                self.assertIs(spec.placement.strategy, PlacementStrategy.EXPLICIT)
                self.assertEqual(spec.placement.groups[0].die_ids, active)
                _validate_rect_mesh_compile_inputs(spec, fabric, RectMeshSpec(3, 3))

        physical_group = build_group_registry(
            mesh_graph(6),
            context(mesh_fabric(3, 3), PlacementSpec(
                PlacementStrategy.EXPLICIT,
                (ExplicitGroupPlacement("P0", "mesh_tp", active),),
            )),
        )[0]
        self.assertEqual(tuple(item.die_id for item in physical_group.placements),
                         active)
        routes = {(route.source_rank, route.destination_rank): route.die_path
                  for route in physical_group.embedding.routes}
        self.assertEqual(routes[(0, 3)], (0, 3, 6))
        self.assertEqual(routes[(2, 5)], (2, 5, 8))
        self.assertTrue({3, 4, 5}.isdisjoint(active))
        self.assertEqual(len(routes), 6 * 5)

    def test_nine_die_partial_group_rejects_compact_or_out_of_mesh_die(self) -> None:
        manifest, template, fabric = _six_die_fixed_model_case(3, 3)
        spec = _segment_spec(template, manifest, 0)
        mesh = RectMeshSpec(3, 3)
        with self.assertRaises(SchemaError):
            _validate_rect_mesh_compile_inputs(replace(spec, placement=PlacementSpec(
                PlacementStrategy.COMPACT, (),
            )), fabric, mesh)
        invalid = replace(spec, placement=PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (ExplicitGroupPlacement(
                spec.parallel.instances[0].id,
                f"{spec.parallel.instances[0].id}.mesh.tp",
                (0, 1, 2, 6, 7, 9)),),
        ))
        with self.assertRaises(SchemaError):
            _validate_rect_mesh_compile_inputs(invalid, fabric, mesh)
        mismatched_template = replace(template, placement=PlacementSpec(
            PlacementStrategy.EXPLICIT,
            (ExplicitGroupPlacement(
                spec.parallel.instances[0].id,
                f"{spec.parallel.instances[0].id}.mesh.tp",
                tuple(range(6))),),
        ))
        with self.assertRaises(UnsupportedFeatureError):
            _validate_inputs(manifest, mismatched_template, fabric,
                             valid_hbm_address_spaces(fabric))

    def test_long_rectangles_preserve_six_active_dies_and_the_same_model(self) -> None:
        reference, template, _ = _six_die_fixed_model_case(2, 3)
        expected_ops = Counter((op.kind, op.step, op.layer)
                               for op in reference.logical_graph.operations)
        for rows, columns in ((1, 6), (6, 1)):
            with self.subTest(mesh=(rows, columns)):
                materialized, extended_template, fabric = (
                    _six_die_fixed_model_case(rows, columns)
                )
                self.assertEqual(materialized.request.model,
                                 reference.request.model)
                self.assertEqual(canonical_digest(extended_template.model),
                                 canonical_digest(template.model))
                self.assertEqual(materialized.placement.active_die_ids,
                                 tuple(range(6)))
                self.assertEqual(fabric.die_grid, (columns, rows))
                self.assertEqual(
                    tuple(space.size_bytes for space in
                          valid_hbm_address_spaces(fabric)),
                    (1 << 30,) * 6,
                )
                self.assertEqual(Counter((op.kind, op.step, op.layer)
                                         for op in materialized.logical_graph.operations),
                                 expected_ops)


if __name__ == "__main__":
    unittest.main()
