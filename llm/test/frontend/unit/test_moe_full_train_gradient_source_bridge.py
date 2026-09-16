"""Fault-inject exact E2E↔IR0 expert0x25 producer/optimizer versions."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_gradient_source_bridge import (
    build_moe_full_train_gradient_source_bridge,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_named_wgrad_tiles import (
    build_moe_full_train_named_wgrad_tiles,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


class MoeExpertGradientSourceBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.tiles = build_moe_full_train_named_wgrad_tiles(
            Fixture.phase, Fixture.sequence, Fixture.placement,
            original_dense=Fixture.dense, dense_manifest=Fixture.manifest,
            context=Fixture.context,
        )
        cls.bridge = build_moe_full_train_gradient_source_bridge(
            Fixture.phase, cls.tiles, Fixture.sequence, Fixture.placement,
            original_dense=Fixture.dense, dense_manifest=Fixture.manifest,
            context=Fixture.context,
        )

    def _check(self, value):
        value.validate_against(
            Fixture.phase, self.tiles, Fixture.sequence,
            Fixture.placement, original_dense=Fixture.dense,
            dense_manifest=Fixture.manifest, context=Fixture.context,
        )

    def test_single_die_six_expert_tiles_and_two_step_sgd_lineage(self):
        phase, sequence, placement, context = (
            build_single_die_moe_train_physical_source(Fixture)
        )
        tiles = build_moe_full_train_named_wgrad_tiles(
            phase, sequence, placement, original_dense=Fixture.dense,
            dense_manifest=Fixture.manifest, context=context,
        )
        self.assertEqual(len(tiles.tiles), 6)
        self.assertEqual(tiles.router_local_gradient_flops, ((32,), (32,)))
        self.assertEqual({tile.native_workload.k for tile in tiles.tiles}, {4})
        self.assertEqual(sum(tile.logical_flops for tile in tiles.tiles), 1536)
        bridge = build_moe_full_train_gradient_source_bridge(
            phase, tiles, sequence, placement,
            original_dense=Fixture.dense,
            dense_manifest=Fixture.manifest, context=context,
        )
        self.assertEqual(len(bridge.paths), 6)
        self.assertEqual({path.owner_physical_die for path in bridge.paths}, {0})
        self.assertTrue(all(path.source_parameter_v0_ref != path.source_parameter_v1_ref
                            and path.source_parameter_v1_ref != path.source_parameter_v2_ref
                            for path in bridge.paths))
        with self.assertRaisesRegex(SchemaError, "two-step gradient-to-SGD"):
            replace(bridge, paths=bridge.paths[:-1]).validate_against(
                phase, tiles, sequence, placement,
                original_dense=Fixture.dense,
                dense_manifest=Fixture.manifest, context=context,
            )

    def test_twelve_named_native_forward_nodes_and_true_v0_v1_v2(self):
        self._check(self.bridge)
        self.assertEqual(len(self.bridge.paths), 12)
        for path in self.bridge.paths:
            with self.subTest(layer=path.layer, expert=path.expert,
                              projection=path.projection):
                self.assertEqual(path.source_ir0_forward_node_ref,
                    f"T0.layer{path.layer}.moe.expert{path.expert}")
                self.assertEqual(path.native_wgrad.source_forward_op_ref,
                                 path.source_ir0_forward_node_ref)
                self.assertNotEqual(path.source_e2e_forward_op_ref,
                                    path.source_ir0_forward_node_ref)
                self.assertEqual(path.owner_ep_rank, path.owner_physical_die)
                self.assertEqual(path.native_wgrad.gradient_bytes, 128)
                self.assertEqual(path.source_parameter_v1_ref,
                    next(binding.input_parameter_state_refs[
                        binding.parameter_refs.index(
                            f"layer.{path.layer}.expert.{path.expert}."
                            f"{path.projection}.weight")]
                        for unit in Fixture.sequence.units
                        if (unit.step, unit.layer) == (1, path.layer)
                        for binding in unit.parameter_bindings
                        if binding.expert == path.expert))

    def test_drop_gate_or_final_down_refuses_partial_gradient(self):
        for position in (0, 11):
            with self.subTest(position=position), self.assertRaisesRegex(
                    SchemaError, "each EP projection"):
                self._check(replace(self.bridge,
                    paths=self.bridge.paths[:position] +
                    self.bridge.paths[position + 1:]))

    def test_fake_ir0_forward_ref_or_e2e_backward_ref_is_rejected(self):
        path = self.bridge.paths[-1]
        for field in ("source_ir0_forward_node_ref",
                      "source_e2e_forward_op_ref",
                      "source_e2e_backward_op_ref"):
            with self.subTest(field=field), self.assertRaisesRegex(
                    SchemaError, "each EP projection"):
                self._check(replace(self.bridge,
                    paths=(*self.bridge.paths[:-1],
                           replace(path, **{field: "forged"}))))

    def test_each_raw_gradient_sync_sgd_store_is_source_bound(self):
        path = self.bridge.paths[0]
        for field in ("source_raw_gradient_v0_ref",
                      "source_synced_gradient_v0_ref",
                      "source_e2e_sgd_op_ref",
                      "source_e2e_store_op_ref",
                      "source_e2e_gradient_v1_op_ref",
                      "source_e2e_sync_v1_op_ref",
                      "source_e2e_sgd_v1_op_ref",
                      "source_e2e_store_v1_op_ref",
                      "source_parameter_v2_ref"):
            with self.subTest(field=field), self.assertRaisesRegex(
                    SchemaError, "two-step gradient-to-SGD"):
                self._check(replace(self.bridge,
                    paths=(replace(path, **{field: "forged"}),
                           *self.bridge.paths[1:])))

    def test_native_workload_cannot_keep_wrong_e2e_forward_label(self):
        path = self.bridge.paths[0]
        fake = replace(path.native_wgrad,
                       source_forward_op_ref=path.source_e2e_forward_op_ref)
        with self.assertRaisesRegex(SchemaError, "each EP projection"):
            self._check(replace(self.bridge,
                paths=(replace(path, native_wgrad=fake),
                       *self.bridge.paths[1:])))


if __name__ == "__main__":
    unittest.main()
