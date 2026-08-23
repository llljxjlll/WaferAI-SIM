from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.passes.build_ir0 import build_ir0
from llm.frontend.wafer_frontend.passes.fusion_partition import (
    partition_bundle,
    partition_ir1,
)
from llm.frontend.wafer_frontend.passes.logical_expand import logical_expand
from llm.frontend.wafer_frontend.passes.placement import place_bundle
from llm.frontend.wafer_frontend.schema.experiment import ExperimentSpec
from llm.frontend.wafer_frontend.schema.ir0 import FusionImpl, FusionPattern
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.serde import from_data

from _fixtures import valid_hbm_address_spaces, valid_ir1, valid_spec


def _source_bundle(*, tp: int = 2):
    raw = valid_spec()
    instance = raw["parallel"]["instances"][0]
    instance["tp"] = tp
    instance["sp"] = tp > 1
    spec = from_data(ExperimentSpec, raw, path="spec")
    expanded = logical_expand(build_ir0(spec))
    fabric = valid_ir1().fabric
    placement = PlacementContext.create(
        producer_pass="unit",
        fabric=fabric,
        placement=spec.placement,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    return place_bundle(expanded, placement)


class FusionPartitionPassTest(unittest.TestCase):
    def test_tp2_selects_all_candidates_without_planning(self) -> None:
        source = _source_bundle()
        context = FusionPartitionContext.create(producer_pass="unit")
        result = partition_bundle(source, context)
        result.validate_against(source, context)

        self.assertEqual(len(result.entries), 1)
        graph = result.entries[0].graph
        self.assertEqual(graph.producer_pass, "fusion_partition")
        self.assertEqual(len(graph.fusion_candidates), 4)
        self.assertEqual(len(graph.fused_op_skeletons), 2)
        self.assertEqual(
            tuple(item.fusion_ref for item in graph.fused_op_skeletons),
            tuple(
                item.id
                for item in graph.fusion_candidates
                if item.semantic_contract.pattern is FusionPattern.GEMM_RS
            ),
        )
        self.assertTrue(
            all(item.impl is FusionImpl.NONE for item in graph.fused_op_skeletons)
        )
        self.assertEqual(graph.cross_routes, ())
        self.assertEqual(partition_bundle(source, context), result)

    def test_tp1_has_no_candidate_or_skeleton(self) -> None:
        source = _source_bundle(tp=1)
        context = FusionPartitionContext.create(producer_pass="unit")
        result = partition_bundle(source, context)
        self.assertEqual(result.entries[0].graph.fusion_candidates, ())
        self.assertEqual(result.entries[0].graph.fused_op_skeletons, ())

    def test_single_graph_rejects_nonplacement_input(self) -> None:
        source = _source_bundle().entries[0].graph
        partitioned = partition_ir1(source)
        with self.assertRaisesRegex(Exception, "must be produced by placement"):
            partition_ir1(partitioned)


if __name__ == "__main__":
    unittest.main()
