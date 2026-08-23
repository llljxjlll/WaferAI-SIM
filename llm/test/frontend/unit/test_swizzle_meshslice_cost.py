from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.policies.swizzle.cost import (
    build_unfused_baseline,
    interpolate_efficiency,
)
from llm.frontend.wafer_frontend.policies.swizzle.decide import decide_swizzle
from llm.frontend.wafer_frontend.policies.swizzle.enumerate import materialize_drafts
from llm.frontend.wafer_frontend.policies.swizzle.meshslice_2d import (
    generate_meshslice_2d_drafts,
)
from llm.frontend.wafer_frontend.schema.common import DType, MeshAxisName
from llm.frontend.wafer_frontend.schema.ir0 import (
    CollectiveKind,
    FusionPattern,
    GemmPartition,
)
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleCollectiveDescriptor,
    SwizzleCollectivePosition,
    SwizzleConstraints,
    SwizzleCost,
    SwizzleDecisionReason,
    SwizzleEfficiencyPoint,
    SwizzleGemmDescriptor,
    SwizzleGroupView,
    SwizzleHardwareProfile,
    SwizzleOperand,
    SwizzleProblem,
    SwizzleRankPlacement,
    SwizzleRouteView,
    SwizzleSemanticWitness,
    SwizzleTensorAxis,
    SwizzleTensorAxisRole,
    SwizzleTensorView,
    SwizzleUpdateKind,
)


def _view(ref, shape, roles, dim_map):
    return SwizzleTensorView(ref, shape, "row_major", roles, dim_map, ())


def _problem(
    rows: int = 2,
    columns: int = 2,
    *,
    k: int = 64,
    exact_2d: bool = True,
    launch_cycles: int = 8,
):
    m = n = 64
    rank_count = rows * columns
    batch = ()
    roles_lhs = (SwizzleTensorAxisRole.FREE_LHS, SwizzleTensorAxisRole.CONTRACT)
    roles_rhs = (SwizzleTensorAxisRole.CONTRACT, SwizzleTensorAxisRole.FREE_RHS)
    roles_output = (SwizzleTensorAxisRole.FREE_LHS, SwizzleTensorAxisRole.FREE_RHS)
    two_d = (MeshAxisName.DP, MeshAxisName.TP)
    one_d = (None, MeshAxisName.TP)
    dim_map = two_d if exact_2d else one_d
    lhs = _view("lhs", (m, k), roles_lhs, dim_map)
    rhs = _view("rhs", (k, n), roles_rhs, dim_map)
    output = _view("gemm.out", (m, n), roles_output, dim_map)
    gemm = SwizzleGemmDescriptor(
        "gemm",
        GemmPartition.COLUMN_PARALLEL,
        m,
        n,
        k,
        batch,
        lhs,
        rhs,
        output,
        (lhs.value_ref, rhs.value_ref),
        (),
        DType.FP16,
        DType.FP32,
        2 * m * n * k,
    )
    gathered_input = _view(
        "ag.in",
        (m, k // rank_count),
        roles_lhs,
        dim_map,
    )
    logical_bytes = m * k * 2
    collective = SwizzleCollectiveDescriptor(
        "ag",
        CollectiveKind.ALL_GATHER,
        None,
        SwizzleCollectivePosition.BEFORE_GEMM,
        (MeshAxisName.TP,),
        tuple(range(rank_count)),
        1,
        None,
        logical_bytes,
        logical_bytes // rank_count,
        logical_bytes,
        gathered_input,
        lhs,
    )
    placements = tuple(
        SwizzleRankPlacement(row * columns + column, column, row)
        for row in range(rows)
        for column in range(columns)
    )
    routes = tuple(
        SwizzleRouteView(
            f"route.{source}.{destination}",
            source,
            destination,
            (source, destination),
            (f"link.{source}.{destination}",),
        )
        for source in range(rank_count)
        for destination in range(rank_count)
        if source != destination
    )
    group = SwizzleGroupView("group", (rows, columns), placements, routes)
    profile = SwizzleHardwareProfile.create(
        peak_flops_per_cycle=1024.0,
        confidence_fraction=0.1,
        efficiency_points=(
            SwizzleEfficiencyPoint(16, 16, 8, 0.5),
            SwizzleEfficiencyPoint(32, 32, 32, 0.9),
            SwizzleEfficiencyPoint(64, 64, 64, 0.95),
        ),
        dte_launch_cycles=launch_cycles,
        dte_sync_cycles=2,
        hop_latency_cycles=1,
        lane_bytes_per_cycle=32.0,
        max_inflight_dte=2,
        min_transfer_bytes=16,
        efficient_tile_floor=(8, 8, 8),
        sram_budget_bytes=1 << 20,
        double_buffer_supported=True,
    )
    constraints = SwizzleConstraints(
        (
            SwizzleAlgorithm.MESHSLICE_2D_OS,
            SwizzleAlgorithm.UNFUSED,
        ),
        32,
        4096,
        256,
        32,
        True,
    )
    problem = SwizzleProblem.create(
        source_ir1_id="ir1",
        fused_op_id="fusion",
        pattern=FusionPattern.AG_GEMM,
        gemm=gemm,
        collective=collective,
        group=group,
        hardware_profile=profile,
        constraints=constraints,
    )
    witness = SwizzleSemanticWitness(
        FusionPattern.AG_GEMM,
        ("ag", "gemm"),
        ("ag.in", "rhs"),
        ("gemm.out",),
        "lhs",
        SwizzleOperand.LHS,
        SwizzleTensorAxis("lhs", 1, "K", k, SwizzleTensorAxisRole.CONTRACT),
        SwizzleUpdateKind.PARTIAL_ACCUMULATION,
        1,
        None,
        False,
        False,
        True,
        True,
        True,
    )
    problem.validate()
    witness.validate()
    return problem, witness


def _candidate_with_cost(candidate: SwizzleCandidate, cost: SwizzleCost) -> SwizzleCandidate:
    return SwizzleCandidate.create(
        problem_ref=candidate.problem_ref,
        pattern=candidate.pattern,
        algorithm=candidate.algorithm,
        split_axis=candidate.split_axis,
        chunk_count=candidate.chunk_count,
        unroll_degree=candidate.unroll_degree,
        rank_programs=candidate.rank_programs,
        buffer_requirements=candidate.buffer_requirements,
        topology_witness=candidate.topology_witness,
        semantic_witness=candidate.semantic_witness,
        feasibility_witness=candidate.feasibility_witness,
        cost=cost,
    )


def _cost(cycles: float, *, utilization: float = 0.5, control: int = 4, sram: int = 64):
    return SwizzleCost.create(
        estimated_cycles=cycles,
        lower_cycles=cycles * 0.9,
        upper_cycles=cycles * 1.1,
        prologue_cycles=cycles * 0.2,
        steady_cycles=cycles * 0.6,
        epilogue_cycles=cycles * 0.2,
        logical_bytes=64,
        byte_hops=64,
        message_count=1,
        direction_port_utilization=utilization,
        control_action_count=control,
        max_inflight=1,
        sram_high_water_bytes=sram,
        bottleneck_resources=(),
    )


class SwizzleMeshSliceCostTest(unittest.TestCase):
    def test_two_by_two_blocked_slices_materialize_typed_action_dags(self) -> None:
        problem, witness = _problem()
        drafts = generate_meshslice_2d_drafts(problem, witness)

        self.assertEqual(tuple(item.chunk_count for item in drafts), (1, 2, 4))
        self.assertTrue(all(item.topology_witness.row_orders == ((0, 1), (2, 3)) for item in drafts))
        self.assertTrue(all(item.topology_witness.column_orders == ((0, 2), (1, 3)) for item in drafts))
        candidates = materialize_drafts(problem, drafts)
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            candidate.validate()
            self.assertGreater(candidate.cost.logical_bytes, 0)
            self.assertGreater(candidate.cost.byte_hops, 0)
            self.assertLessEqual(candidate.cost.max_inflight, 2)
            self.assertLessEqual(
                candidate.cost.sram_high_water_bytes,
                problem.hardware_profile.sram_budget_bytes,
            )

    def test_rectangular_two_by_four_keeps_asymmetric_lines(self) -> None:
        problem, witness = _problem(2, 4, k=64)
        drafts = generate_meshslice_2d_drafts(problem, witness)
        self.assertTrue(drafts)
        topology = drafts[0].topology_witness
        self.assertEqual(len(topology.row_orders), 2)
        self.assertEqual(len(topology.row_orders[0]), 4)
        self.assertEqual(len(topology.column_orders), 4)
        self.assertEqual(len(topology.column_orders[0]), 2)

    def test_placement_rectangle_alone_never_enables_meshslice(self) -> None:
        problem, witness = _problem(exact_2d=False)
        self.assertEqual(generate_meshslice_2d_drafts(problem, witness), ())
        self.assertTrue(generate_meshslice_2d_drafts(problem, witness, allow_boundary_reshard=True))

    def test_invalid_block_divisor_fails_closed(self) -> None:
        problem, witness = _problem(k=60)
        self.assertEqual(generate_meshslice_2d_drafts(problem, witness), ())

    def test_eta_exact_hit_and_interpolation_are_deterministic(self) -> None:
        problem, _ = _problem()
        profile = problem.hardware_profile
        self.assertEqual(interpolate_efficiency(profile, (32, 32, 32)), 0.9)
        interpolated = interpolate_efficiency(profile, (32, 32, 16))
        self.assertEqual(interpolated, interpolate_efficiency(profile, (32, 32, 16)))
        self.assertGreaterEqual(interpolated, 0.5)
        self.assertLessEqual(interpolated, 0.95)

    def test_unprofitable_candidate_falls_back_to_unfused(self) -> None:
        problem, witness = _problem()
        baseline = build_unfused_baseline(problem, witness)
        candidate = materialize_drafts(problem, generate_meshslice_2d_drafts(problem, witness)[:1])[0]
        expensive = _candidate_with_cost(candidate, _cost(baseline.cost.estimated_cycles * 2.0))
        decision = decide_swizzle(problem, baseline, (expensive,))
        self.assertEqual(decision.selected_candidate_ref, baseline.id)
        self.assertIs(decision.decision_reason, SwizzleDecisionReason.NO_PROFITABLE_FUSION)

    def test_interval_tie_break_is_input_order_independent(self) -> None:
        problem, witness = _problem()
        baseline = build_unfused_baseline(problem, witness)
        drafts = generate_meshslice_2d_drafts(problem, witness)
        candidates = materialize_drafts(problem, drafts[:2])
        fast = _candidate_with_cost(candidates[0], _cost(10.0, utilization=0.9, control=8))
        tied = _candidate_with_cost(candidates[1], _cost(10.5, utilization=0.5, control=1))
        first = decide_swizzle(problem, baseline, (fast, tied))
        second = decide_swizzle(problem, baseline, (tied, fast))
        self.assertEqual(first.selected_candidate_ref, fast.id)
        self.assertEqual(second.selected_candidate_ref, fast.id)
        self.assertIs(first.decision_reason, SwizzleDecisionReason.INTERVAL_TIE_BREAK)


if __name__ == "__main__":
    unittest.main()
