from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleTopologyKind,
)
from llm.frontend.wafer_frontend.schema.swizzle_scale import SwizzleScalePoint
from llm.frontend.wafer_frontend.policies.swizzle.chunking import wang_tile_shape

from swizzle_scale_cases import (
    build_first_green_swizzle_scale_cases,
    build_swizzle_scale_case,
    build_swizzle_scale_points,
)


class SwizzleScaleCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.points = build_swizzle_scale_points()
        cls.cases = build_first_green_swizzle_scale_cases()

    def test_s0_s4_matrix_is_typed_deterministic_and_fail_closed(self) -> None:
        self.assertEqual(
            tuple(
                (
                    item.name,
                    item.tokens,
                    item.hidden_size,
                    item.intermediate_size,
                    item.tp,
                    item.mesh_rows,
                    item.mesh_columns,
                )
                for item in self.points
            ),
            (
                ("S0", 8, 16, 32, 2, 1, 2),
                ("S1", 32, 64, 256, 4, 2, 2),
                ("S2", 64, 64, 256, 4, 2, 2),
                ("S3", 128, 64, 256, 4, 2, 2),
                ("S4", 256, 512, 2048, 4, 2, 2),
            ),
        )
        self.assertEqual(build_swizzle_scale_points(), self.points)
        with self.assertRaisesRegex(SchemaError, "tp must equal"):
            SwizzleScalePoint.create(
                name="bad_mesh",
                tokens=8,
                hidden_size=16,
                intermediate_size=32,
                tp=4,
                mesh_rows=1,
                mesh_columns=2,
                dtype=DType.FP16,
            )
        with self.assertRaisesRegex(SchemaError, "tokens must divide exactly"):
            SwizzleScalePoint.create(
                name="bad_divisor",
                tokens=6,
                hidden_size=16,
                intermediate_size=32,
                tp=4,
                mesh_rows=2,
                mesh_columns=2,
                dtype=DType.FP16,
            )
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(self.points[0], tokens=10).validate()

    def test_s0_freezes_official_geometry_and_unfused_decision(self) -> None:
        case = self.cases[0]
        self.assertEqual(case.point.name, "S0")
        self.assertEqual(case.placed_graph.fabric.die_grid, (2, 1))
        self.assertEqual(
            tuple(
                (placement.rank, placement.die_id)
                for placement in case.placed_graph.groups[0].placements
            ),
            ((0, 0), (1, 1)),
        )
        expected = {
            "ag_gemm": (
                (8, 48, 16),
                (8, 24, 16),
                12288,
                (256, 128, 256),
                2,
            ),
            "gemm_rs": (
                (8, 16, 16),
                (8, 16, 8),
                4096,
                (256, 256, 128),
                2,
            ),
        }
        expected_max_chunks = {"S1": 8, "S2": 16, "S3": 32}
        for decision in case.decisions:
            with self.subTest(pattern=decision.problem.pattern.value):
                node = next(
                    item
                    for item in case.partitioned_graph.nodes
                    if item.id == decision.problem.gemm.node_ref
                )
                fused = tuple(
                    item
                    for item in decision.ranked_candidates
                    if item.algorithm is not SwizzleAlgorithm.UNFUSED
                )
                base_fused = tuple(
                    item
                    for item in fused
                    if item.chunk_count == case.point.tp
                )
                self.assertEqual(
                    (
                        node.workload.logical_shape,
                        node.workload.rank_shape,
                        decision.problem.gemm.flops,
                        (
                            decision.problem.collective.logical_bytes,
                            decision.problem.collective.rank_input_bytes,
                            decision.problem.collective.rank_output_bytes,
                        ),
                        len(base_fused),
                    ),
                    expected[decision.problem.pattern.value],
                )
                self.assertIs(
                    decision.ranked_candidates[0].algorithm,
                    SwizzleAlgorithm.UNFUSED,
                )
                self.assertEqual(
                    decision.problem.constraints.max_chunk_count,
                    case.point.tp,
                )

    def test_s1_s3_are_exact_2x2_wang_first_green_cases(self) -> None:
        expected = {
            "S1": {
                "ag_gemm": ((32, 192, 64), (32, 48, 64), 786432, 4096),
                "gemm_rs": ((32, 64, 64), (32, 64, 16), 262144, 4096),
            },
            "S2": {
                "ag_gemm": ((64, 192, 64), (64, 48, 64), 1572864, 8192),
                "gemm_rs": ((64, 64, 64), (64, 64, 16), 524288, 8192),
            },
            "S3": {
                "ag_gemm": ((128, 192, 64), (128, 48, 64), 3145728, 16384),
                "gemm_rs": ((128, 64, 64), (128, 64, 16), 1048576, 16384),
            },
        }
        for case in self.cases[1:]:
            with self.subTest(scale=case.point.name):
                group = case.placed_graph.groups[0]
                self.assertEqual(case.placed_graph.fabric.die_grid, (2, 2))
                self.assertEqual(
                    tuple(
                        (item.rank, item.die_id, item.logical_coord)
                        for item in group.placements
                    ),
                    (
                        (0, 0, (0,)),
                        (1, 1, (1,)),
                        (2, 2, (2,)),
                        (3, 3, (3,)),
                    ),
                )
                self.assertEqual(len(group.embedding.routes), 12)
                for decision in case.decisions:
                    node = next(
                        item
                        for item in case.partitioned_graph.nodes
                        if item.id == decision.problem.gemm.node_ref
                    )
                    fused = tuple(
                        item
                        for item in decision.ranked_candidates
                        if item.algorithm is not SwizzleAlgorithm.UNFUSED
                    )
                    base_fused = tuple(
                        item
                        for item in fused
                        if item.chunk_count == case.point.tp
                    )
                    golden = expected[case.point.name][
                        decision.problem.pattern.value
                    ]
                    self.assertEqual(
                        (
                            node.workload.logical_shape,
                            node.workload.rank_shape,
                            decision.problem.gemm.flops,
                            decision.problem.collective.logical_bytes,
                        ),
                        golden,
                    )
                    self.assertEqual(len(base_fused), 4)
                    self.assertEqual(
                        max(item.chunk_count for item in fused),
                        expected_max_chunks[case.point.name],
                    )
                    self.assertEqual(
                        {
                            (item.topology_witness.kind, item.unroll_degree)
                            for item in base_fused
                        },
                        {
                            (SwizzleTopologyKind.BIDIRECTIONAL_LINE, 1),
                            (SwizzleTopologyKind.BIDIRECTIONAL_LINE, 2),
                            (SwizzleTopologyKind.HAMILTONIAN_RING, 1),
                            (SwizzleTopologyKind.HAMILTONIAN_RING, 2),
                        },
                    )


                    self.assertEqual(
                        {item.algorithm for item in fused},
                        {SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL},
                    )
                    self.assertEqual(
                        {item.topology_witness.kind for item in base_fused},
                        {
                            SwizzleTopologyKind.BIDIRECTIONAL_LINE,
                            SwizzleTopologyKind.HAMILTONIAN_RING,
                        },
                    )
                    self.assertEqual(
                        {
                            item.topology_witness.rank_order
                            for item in base_fused
                            if item.topology_witness.kind
                            is SwizzleTopologyKind.HAMILTONIAN_RING
                        },
                        {(0, 1, 3, 2)},
                    )
                    self.assertTrue(
                        any(item.unroll_degree == 2 for item in base_fused)
                    )
                    self.assertIs(
                        decision.ranked_candidates[0].algorithm,
                        SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL,
                    )
                    self.assertEqual(
                        decision.problem.constraints.max_chunk_count,
                        32,
                    )
                    floor = decision.problem.hardware_profile.efficient_tile_floor
                    self.assertEqual(floor, (4, 16, 16))
                    self.assertTrue(
                        all(
                            all(actual >= required for actual, required in zip(
                                wang_tile_shape(
                                    decision.problem,
                                    item.semantic_witness,
                                    item.chunk_count,
                                ),
                                floor,
                            ))
                            for item in fused
                        )
                    )
                    self.assertIn(
                        SwizzleAlgorithm.MESHSLICE_2D_OS,
                        decision.problem.constraints.allowed_algorithms,
                    )
                    self.assertNotIn(
                        SwizzleAlgorithm.MESHSLICE_2D_OS,
                        {item.algorithm for item in decision.ranked_candidates},
                    )

    def test_s4_and_nonmatching_rectangle_fail_at_exact_production_gates(self) -> None:
        with self.assertRaisesRegex(
            SchemaError, "persistent state exceeds its home HBM capacity"
        ):
            build_swizzle_scale_case(self.points[4])

        wrong_rectangle = SwizzleScalePoint.create(
            name="S1_1x4",
            tokens=32,
            hidden_size=64,
            intermediate_size=256,
            tp=4,
            mesh_rows=1,
            mesh_columns=4,
            dtype=DType.FP16,
        )
        with self.assertRaisesRegex(
            SchemaError, "physical fabric must be exact rectangle"
        ):
            build_swizzle_scale_case(wrong_rectangle)


if __name__ == "__main__":
    unittest.main()
