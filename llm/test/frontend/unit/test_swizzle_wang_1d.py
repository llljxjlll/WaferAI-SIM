from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.policies.swizzle.semantics import (
    analyze_ag_gemm,
    analyze_gemm_ar,
    analyze_gemm_rs,
)
from llm.frontend.wafer_frontend.policies.swizzle.wang_1d import (
    generate_wang_1d_drafts,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind, FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleConstraints,
    SwizzleHardwareProfile,
    SwizzleGroupView,
    SwizzleProblem,
    SwizzleRankPlacement,
    SwizzleRouteView,
    SwizzleTensorAxisRole,
    SwizzleTopologyKind,
)

from test_swizzle_schema import _ag_case, _post_collective, _profile


_COORDS = ((0, 0), (1, 0), (0, 1), (1, 1))


def _xy_path(source: int, destination: int) -> tuple[int, ...]:
    x, y = _COORDS[source]
    destination_x, destination_y = _COORDS[destination]
    result = [source]
    while x != destination_x:
        x += 1 if destination_x > x else -1
        result.append(y * 2 + x)
    while y != destination_y:
        y += 1 if destination_y > y else -1
        result.append(y * 2 + x)
    return tuple(result)


def _group4() -> SwizzleGroupView:
    routes = []
    for source in range(4):
        for destination in range(4):
            if source == destination:
                continue
            path = _xy_path(source, destination)
            routes.append(
                SwizzleRouteView(
                    f"route.{source}.{destination}",
                    source,
                    destination,
                    path,
                    tuple(
                        f"resource.{left}.{right}"
                        for left, right in zip(path, path[1:])
                    ),
                )
            )
    return SwizzleGroupView(
        group_ref="group4",
        logical_shape=(2, 2),
        placements=tuple(
            SwizzleRankPlacement(rank, *_COORDS[rank]) for rank in range(4)
        ),
        routes=tuple(routes),
    )


def _constraints() -> SwizzleConstraints:
    return SwizzleConstraints(
        allowed_algorithms=(
            SwizzleAlgorithm.UNFUSED,
            SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL,
        ),
        max_candidates=32,
        max_actions=1024,
        max_buffers=64,
        max_chunk_count=32,
        allow_unroll_two=True,
    )


def _ag_problem():
    gemm, collective, boundaries = _ag_case(SwizzleTensorAxisRole.CONTRACT)
    axis = collective.gather_tensor_axis
    assert axis is not None
    local_shape = list(collective.output.shape)
    local_shape[axis] //= 4
    collective = replace(
        collective,
        participant_ranks=(0, 1, 2, 3),
        rank_input_bytes=16,
        input=replace(collective.input, shape=tuple(local_shape)),
    )
    witness = analyze_ag_gemm(
        gemm,
        collective,
        boundary_input_refs=boundaries,
        boundary_output_refs=(gemm.output.value_ref,),
    )
    problem = SwizzleProblem.create(
        source_ir1_id="ir1",
        fused_op_id="fused.ag",
        pattern=FusionPattern.AG_GEMM,
        gemm=gemm,
        collective=collective,
        group=_group4(),
        hardware_profile=_profile(),
        constraints=_constraints(),
    )
    return problem, witness


def _post_problem(kind: CollectiveKind):
    gemm, collective = _post_collective(kind)
    if kind is CollectiveKind.REDUCE_SCATTER:
        collective = replace(
            collective,
            participant_ranks=(0, 1, 2, 3),
            rank_output_bytes=16,
            output=replace(collective.output, shape=(1, 8)),
        )
        pattern = FusionPattern.GEMM_RS
        witness = analyze_gemm_rs(
            gemm,
            collective,
            boundary_input_refs=(gemm.lhs.value_ref, gemm.rhs.value_ref),
            boundary_output_refs=(collective.output.value_ref,),
        )
    else:
        collective = replace(
            collective,
            participant_ranks=(0, 1, 2, 3),
        )
        pattern = FusionPattern.GEMM_AR
        witness = analyze_gemm_ar(
            gemm,
            collective,
            boundary_input_refs=(gemm.lhs.value_ref, gemm.rhs.value_ref),
            boundary_output_refs=(collective.output.value_ref,),
        )
    problem = SwizzleProblem.create(
        source_ir1_id="ir1",
        fused_op_id=f"fused.{pattern.value}",
        pattern=pattern,
        gemm=gemm,
        collective=collective,
        group=_group4(),
        hardware_profile=_profile(),
        constraints=_constraints(),
    )
    return problem, witness


def _actions(draft):
    return tuple(action for program in draft.rank_programs for action in program.actions)


class Wang1DTest(unittest.TestCase):
    def test_four_way_ag_has_line_and_only_a_proven_ring(self) -> None:
        problem, witness = _ag_problem()
        drafts = generate_wang_1d_drafts(problem, witness)

        self.assertEqual(
            tuple((item.topology_witness.kind, item.unroll_degree) for item in drafts),
            (
                (SwizzleTopologyKind.BIDIRECTIONAL_LINE, 1),
                (SwizzleTopologyKind.BIDIRECTIONAL_LINE, 2),
                (SwizzleTopologyKind.HAMILTONIAN_RING, 1),
                (SwizzleTopologyKind.HAMILTONIAN_RING, 2),
            ),
        )
        line = drafts[0]
        self.assertEqual(line.topology_witness.rank_order, (0, 1, 3, 2))
        self.assertTrue(line.topology_witness.has_hamiltonian_cycle)
        comp = [action for action in _actions(line) if action.kind is SwizzleActionKind.COMP]
        self.assertEqual(len(comp), 16)
        self.assertEqual(
            {(action.rank, action.chunk_index) for action in comp},
            {(rank, chunk) for rank in range(4) for chunk in range(4)},
        )
        for action in _actions(line):
            if action.route_ref is not None:
                route = next(item for item in problem.group.routes if item.id == action.route_ref)
                self.assertEqual(len(route.die_path), 2)

    def test_unroll_two_is_typed_as_double_buffering(self) -> None:
        problem, witness = _ag_problem()
        drafts = generate_wang_1d_drafts(problem, witness)
        unrolled = next(
            item
            for item in drafts
            if item.topology_witness.kind is SwizzleTopologyKind.BIDIRECTIONAL_LINE
            and item.unroll_degree == 2
        )
        self.assertTrue(unrolled.buffer_requirements)
        self.assertTrue(
            all(item.double_buffered for item in unrolled.buffer_requirements)
        )
        unrolled.validate_against(problem)

    def test_four_way_rs_has_loop_carried_reduce_and_owner_alignment(self) -> None:
        problem, witness = _post_problem(CollectiveKind.REDUCE_SCATTER)
        draft = generate_wang_1d_drafts(problem, witness)[0]
        actions = _actions(draft)

        self.assertEqual(
            len([item for item in actions if item.kind is SwizzleActionKind.COMP]),
            16,
        )
        owner_reduces = [
            item
            for item in actions
            if item.kind is SwizzleActionKind.REDUCE
            and item.output_refs
            and "::owner" in item.output_refs[0]
        ]
        self.assertEqual(len(owner_reduces), 6)
        self.assertEqual(
            {(item.rank, item.chunk_index) for item in owner_reduces},
            {(rank, rank) for rank in range(4)},
        )
        reductions = tuple(
            item for item in actions if item.kind is SwizzleActionKind.REDUCE
        )
        self.assertTrue(
            all(
                len(item.input_refs) == 2 and len(item.deps) == 2
                for item in reductions
            )
        )
        producer_by_output = {
            output: item for item in reductions for output in item.output_refs
        }
        loop_edges = tuple(
            (producer_by_output[input_ref], item)
            for item in reductions
            for input_ref in item.input_refs
            if input_ref in producer_by_output
        )
        self.assertEqual(len(loop_edges), 2)
        self.assertTrue(
            all(producer.id in consumer.deps for producer, consumer in loop_edges)
        )
        terminal_reduces = tuple(
            item
            for item in reductions
            if item.output_refs[0].startswith(
                f"{problem.collective.output.value_ref}::owner"
            )
        )
        self.assertEqual(len(terminal_reduces), 4)
        self.assertEqual(
            len([item for item in actions if item.kind is SwizzleActionKind.LOCAL_COPY]),
            4,
        )

    def test_four_way_ar_has_explicit_replication_epilogue(self) -> None:
        problem, witness = _post_problem(CollectiveKind.ALL_REDUCE)
        draft = generate_wang_1d_drafts(problem, witness)[0]
        actions = _actions(draft)
        epilogue_sends = [
            item
            for item in actions
            if item.kind is SwizzleActionKind.SEND
            and item.phase.value == "epilogue"
        ]
        epilogue_barriers = [
            item
            for item in actions
            if item.kind is SwizzleActionKind.BARRIER
            and item.phase.value == "epilogue"
        ]

        self.assertEqual(len(epilogue_sends), 12)
        self.assertEqual(len(epilogue_barriers), 4)
        self.assertTrue(all(len(item.deps) == 4 for item in epilogue_barriers))
        self.assertTrue(draft.semantic_witness.has_reduction_phase)
        self.assertTrue(draft.semantic_witness.has_replication_phase)

    def test_missing_route_and_insufficient_sram_fail_closed(self) -> None:
        problem, witness = _ag_problem()
        broken = SwizzleProblem.create(
            source_ir1_id=problem.source_ir1_id,
            fused_op_id=problem.fused_op_id,
            pattern=problem.pattern,
            gemm=problem.gemm,
            collective=problem.collective,
            group=replace(
                problem.group,
                routes=problem.group.routes[:-1],
            ),
            hardware_profile=problem.hardware_profile,
            constraints=problem.constraints,
        )
        with self.assertRaisesRegex(Exception, "every ordered rank-pair"):
            generate_wang_1d_drafts(broken, witness)
        profile = problem.hardware_profile
        tiny_profile = SwizzleHardwareProfile.create(
            peak_flops_per_cycle=profile.peak_flops_per_cycle,
            confidence_fraction=profile.confidence_fraction,
            efficiency_points=profile.efficiency_points,
            dte_launch_cycles=profile.dte_launch_cycles,
            dte_sync_cycles=profile.dte_sync_cycles,
            hop_latency_cycles=profile.hop_latency_cycles,
            lane_bytes_per_cycle=profile.lane_bytes_per_cycle,
            max_inflight_dte=profile.max_inflight_dte,
            min_transfer_bytes=profile.min_transfer_bytes,
            efficient_tile_floor=profile.efficient_tile_floor,
            sram_budget_bytes=1,
            double_buffer_supported=profile.double_buffer_supported,
        )
        tiny = SwizzleProblem.create(
            source_ir1_id=problem.source_ir1_id,
            fused_op_id=problem.fused_op_id,
            pattern=problem.pattern,
            gemm=problem.gemm,
            collective=problem.collective,
            group=problem.group,
            hardware_profile=tiny_profile,
            constraints=problem.constraints,
        )
        self.assertEqual(generate_wang_1d_drafts(tiny, witness), ())


if __name__ == "__main__":
    unittest.main()
