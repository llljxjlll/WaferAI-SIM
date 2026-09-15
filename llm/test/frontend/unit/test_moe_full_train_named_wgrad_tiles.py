"""Native0x25 source tile dimensions versus real E2E/P2 expert gradients."""

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_named_wgrad_tiles import (
    build_moe_full_train_named_wgrad_tiles,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
)


class MoeFullTrainNamedWgradTilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Fixture.setUpClass()
        cls.source=build_moe_full_train_named_wgrad_tiles(
            Fixture.phase,Fixture.sequence,Fixture.placement,
            original_dense=Fixture.dense,dense_manifest=Fixture.manifest,
            context=Fixture.context,
        )

    def _check(self,source):
        source.validate_against(
            Fixture.phase,Fixture.sequence,Fixture.placement,
            original_dense=Fixture.dense,dense_manifest=Fixture.manifest,
            context=Fixture.context,
        )

    def test_four_experts_three_physical_named_projection_tiles(self):
        self._check(self.source)
        self.assertEqual(len(self.source.tiles),12)
        self.assertEqual(self.source.router_local_gradient_flops,
                         ((64,0),(64,0)))
        self.assertEqual(sum(tile.logical_flops for tile in self.source.tiles),
                         4*384)
        self.assertEqual(sum(tile.gradient_bytes for tile in self.source.tiles),
                         4*384)
        for layer in (0,1):
            for expert in (0,1):
                tiles=[tile for tile in self.source.tiles
                       if (tile.layer,tile.expert)==(layer,expert)]
                with self.subTest(layer=layer,expert=expert):
                    self.assertEqual([tile.projection for tile in tiles],
                                     ["gate","up","down"])
                    self.assertEqual([tile.source_aggregate_slice_offset
                                      for tile in tiles],[0,64,128])
                    self.assertEqual([(tile.native_workload.m,
                                       tile.native_workload.n,
                                       tile.native_workload.k)
                                      for tile in tiles],
                                     [(4,8,2),(4,8,2),(8,4,2)])
                    self.assertEqual([tile.gradient_bytes for tile in tiles],
                                     [128,128,128])
                    self.assertEqual({tile.owner_ep_rank for tile in tiles},
                                     {expert})
                    self.assertEqual({tile.source_p2_expert_wgrad_action_ref
                                      for tile in tiles},
                                     {next(action.id for action in
                                           Fixture.sequence.units[layer].plan.actions
                                           if action.kind.value=="expert_wgrad"
                                           and action.rank==expert)})

    def test_missing_any_projection_or_last_layer_is_rejected(self):
        for position in (0,1,2,11):
            removed=self.source.tiles[:position]+self.source.tiles[position+1:]
            with self.subTest(position=position),self.assertRaisesRegex(
                    SchemaError,"every MoE expert projection"):
                self._check(replace(self.source,tiles=removed))

    def test_original_expert_aggregate_slice_and_ep_die_are_exact(self):
        for position,changes in ((2,{"source_aggregate_slice_offset":0}),
                                 (3,{"owner_ep_rank":0})):
            tiles=list(self.source.tiles)
            tiles[position]=replace(tiles[position],**changes)
            with self.subTest(position=position),self.assertRaisesRegex(
                    SchemaError,"every MoE expert projection"):
                self._check(replace(self.source,tiles=tuple(tiles)))

    def test_source_action_and_route_digest_cannot_be_forged(self):
        for field in ("source_p2_expert_wgrad_action_ref",
                      "source_route_trace_digest",
                      "source_e2e_backward_op_ref"):
            tiles=list(self.source.tiles)
            tiles[-1]=replace(tiles[-1],**{field:"0"*64})
            with self.subTest(field=field),self.assertRaisesRegex(
                    SchemaError,"original E2E/P2"):
                self._check(replace(self.source,tiles=tuple(tiles)))

    def test_k_rank_rows_or_wgrad_fp32_dtype_cannot_change(self):
        original=self.source.tiles[0]
        wrong_k=replace(original.native_workload,k=3)
        wrong_k.validate()
        tiles=(replace(original,native_workload=wrong_k),*self.source.tiles[1:])
        with self.assertRaisesRegex(SchemaError,"every MoE expert projection"):
            self._check(replace(self.source,tiles=tiles))
        with self.assertRaisesRegex(SchemaError,"FP16 X/dY to FP32 dW"):
            replace(original.native_workload,
                    gradient_dtype=DType.FP16).validate()

    def test_physical_case_and_original_parameter_state_decl_are_bound(self):
        changed=replace(self.source,source_physical_case_ref="other-profile")
        with self.assertRaisesRegex(SchemaError,"original E2E/P2"):
            self._check(changed)
        tiles=list(self.source.tiles)
        native=replace(tiles[0].native_workload,
                       source_parameter_state_ref="missing_expert_weight")
        tiles[0]=replace(tiles[0],native_workload=native)
        with self.assertRaisesRegex(SchemaError,"original E2E/P2"):
            self._check(replace(self.source,tiles=tuple(tiles)))


if __name__=="__main__":
    unittest.main()
