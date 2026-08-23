from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.policies.swizzle.chunking import (
    legal_wang_chunk_specs,
)
from llm.frontend.wafer_frontend.policies.swizzle.cost import (
    build_unfused_baseline,
)
from llm.frontend.wafer_frontend.policies.swizzle.decide import decide_swizzle
from llm.frontend.wafer_frontend.policies.swizzle.enumerate import (
    materialize_drafts,
)
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
    SwizzleDecisionReason,
    SwizzleHardwareProfile,
    SwizzleProblem,
    SwizzleTopologyKind,
)

from test_swizzle_wang_1d import _ag_problem, _post_problem


def _actions(draft):
    return tuple(
        action for program in draft.rank_programs for action in program.actions
    )


def _with_profile(problem, **changes):
    profile = problem.hardware_profile
    fields = {
        "peak_flops_per_cycle": profile.peak_flops_per_cycle,
        "confidence_fraction": profile.confidence_fraction,
        "efficiency_points": profile.efficiency_points,
        "dte_launch_cycles": profile.dte_launch_cycles,
        "dte_sync_cycles": profile.dte_sync_cycles,
        "hop_latency_cycles": profile.hop_latency_cycles,
        "lane_bytes_per_cycle": profile.lane_bytes_per_cycle,
        "max_inflight_dte": profile.max_inflight_dte,
        "min_transfer_bytes": profile.min_transfer_bytes,
        "efficient_tile_floor": profile.efficient_tile_floor,
        "sram_budget_bytes": profile.sram_budget_bytes,
        "double_buffer_supported": profile.double_buffer_supported,
    }
    fields.update(changes)
    return SwizzleProblem.create(
        source_ir1_id=problem.source_ir1_id,
        fused_op_id=problem.fused_op_id,
        pattern=problem.pattern,
        gemm=problem.gemm,
        collective=problem.collective,
        group=problem.group,
        hardware_profile=SwizzleHardwareProfile.create(**fields),
        constraints=problem.constraints,
    )


def _scaled(fixture):
    problem, witness = fixture
    if problem.pattern is FusionPattern.AG_GEMM:
        gemm = replace(
            problem.gemm,
            m=8,
            n=8,
            k=16,
            lhs=replace(problem.gemm.lhs, shape=(8, 16)),
            rhs=replace(problem.gemm.rhs, shape=(16, 8)),
            output=replace(problem.gemm.output, shape=(8, 8)),
            flops=2048,
        )
        collective = replace(
            problem.collective,
            logical_bytes=256,
            rank_input_bytes=64,
            rank_output_bytes=256,
            input=replace(problem.collective.input, shape=(8, 4)),
            output=gemm.lhs,
        )
        witness = analyze_ag_gemm(
            gemm,
            collective,
            boundary_input_refs=witness.boundary_input_refs,
            boundary_output_refs=witness.boundary_output_refs,
        )
    else:
        gemm = replace(
            problem.gemm,
            m=16,
            n=8,
            k=4,
            lhs=replace(problem.gemm.lhs, shape=(16, 4)),
            output=replace(problem.gemm.output, shape=(16, 8)),
            flops=1024,
        )
        rank_output_bytes = (
            64 if problem.pattern is FusionPattern.GEMM_RS else 256
        )
        output_shape = (
            (4, 8)
            if problem.pattern is FusionPattern.GEMM_RS
            else (16, 8)
        )
        collective = replace(
            problem.collective,
            logical_bytes=256,
            rank_input_bytes=256,
            rank_output_bytes=rank_output_bytes,
            input=gemm.output,
            output=replace(problem.collective.output, shape=output_shape),
        )
        analyze = (
            analyze_gemm_rs
            if problem.pattern is FusionPattern.GEMM_RS
            else analyze_gemm_ar
        )
        witness = analyze(
            gemm,
            collective,
            boundary_input_refs=witness.boundary_input_refs,
            boundary_output_refs=witness.boundary_output_refs,
        )
    scaled = SwizzleProblem.create(
        source_ir1_id=problem.source_ir1_id,
        fused_op_id=problem.fused_op_id,
        pattern=problem.pattern,
        gemm=gemm,
        collective=collective,
        group=problem.group,
        hardware_profile=problem.hardware_profile,
        constraints=problem.constraints,
    )
    return scaled, witness


class WangChunkingTest(unittest.TestCase):
    def test_rank_independent_chunk_family_and_work_conservation(self) -> None:
        fixtures = (
            _scaled(_ag_problem()),
            _scaled(_post_problem(CollectiveKind.REDUCE_SCATTER)),
            _scaled(_post_problem(CollectiveKind.ALL_REDUCE)),
        )
        for problem, witness in fixtures:
            with self.subTest(pattern=problem.pattern.value):
                specs = legal_wang_chunk_specs(problem, witness)
                self.assertEqual(
                    tuple(item.chunk_count for item in specs), (4, 8, 16)
                )
                drafts = generate_wang_1d_drafts(problem, witness)
                self.assertEqual(
                    tuple(dict.fromkeys(item.chunk_count for item in drafts)),
                    (4, 8, 16),
                )
                ranks = problem.collective.participant_ranks
                for draft in drafts:
                    actions = _actions(draft)
                    chunked = tuple(
                        item for item in actions if item.chunk_index is not None
                    )
                    self.assertEqual(
                        {item.chunk_index for item in chunked},
                        set(range(draft.chunk_count)),
                    )
                    comps = tuple(
                        item for item in actions
                        if item.kind is SwizzleActionKind.COMP
                    )
                    self.assertEqual(len(comps), len(ranks) * draft.chunk_count)
                    self.assertEqual(
                        {(item.rank, item.chunk_index) for item in comps},
                        {
                            (rank, chunk)
                            for rank in ranks
                            for chunk in range(draft.chunk_count)
                        },
                    )
                    self.assertEqual(
                        sum(item.flops for item in comps), problem.gemm.flops
                    )
                    sends = tuple(
                        item for item in actions
                        if item.kind is SwizzleActionKind.SEND
                    )
                    phase_multiplier = (
                        2 if problem.pattern is FusionPattern.GEMM_AR else 1
                    )
                    self.assertEqual(
                        sum(item.logical_bytes for item in sends),
                        phase_multiplier
                        * (len(ranks) - 1)
                        * problem.collective.logical_bytes,
                    )

                    if problem.pattern is FusionPattern.AG_GEMM:
                        for chunk in range(draft.chunk_count):
                            local = tuple(
                                item
                                for item in comps
                                if item.chunk_index == chunk
                                and item.input_refs[0].endswith(f"::chunk{chunk}")
                            )
                            self.assertEqual(len(local), 1)
                            self.assertEqual(local[0].rank, ranks[chunk % len(ranks)])
                    else:
                        finals = tuple(
                            item
                            for item in actions
                            if item.kind is SwizzleActionKind.REDUCE
                            and item.output_refs
                            and (
                                item.output_refs[0].startswith(
                                    f"{problem.collective.output.value_ref}::owner"
                                )
                                if problem.pattern is FusionPattern.GEMM_RS
                                else "::reduced_chunk" in item.output_refs[0]
                            )
                        )
                        self.assertEqual(len(finals), draft.chunk_count)
                        self.assertTrue(
                            all(
                                item.rank == ranks[item.chunk_index % len(ranks)]
                                for item in finals
                                if item.chunk_index is not None
                            )
                        )
                        reductions = tuple(
                            item
                            for item in actions
                            if item.kind is SwizzleActionKind.REDUCE
                        )
                        self.assertTrue(
                            all(
                                len(item.input_refs) == 2
                                and len(item.deps) == 2
                                for item in reductions
                            )
                        )
                        producer_by_output = {
                            output: item
                            for item in reductions
                            for output in item.output_refs
                        }
                        self.assertTrue(
                            all(
                                producer_by_output[input_ref].id in item.deps
                                for item in reductions
                                for input_ref in item.input_refs
                                if input_ref in producer_by_output
                            )
                        )

    def test_unroll_two_has_only_same_slot_reuse_edges_and_real_inflight(self) -> None:
        problem, witness = _scaled(_ag_problem())
        draft = next(
            item
            for item in generate_wang_1d_drafts(problem, witness)
            if item.chunk_count == 8
            and item.unroll_degree == 2
            and item.topology_witness.kind is SwizzleTopologyKind.BIDIRECTIONAL_LINE
        )
        actions = _actions(draft)
        sends = tuple(
            item for item in actions if item.kind is SwizzleActionKind.SEND
        )
        recvs = tuple(
            item for item in actions if item.kind is SwizzleActionKind.RECV
        )
        comps = tuple(
            item for item in actions if item.kind is SwizzleActionKind.COMP
        )
        for rank in problem.collective.participant_ranks:
            by_chunk = {
                chunk: tuple(
                    item
                    for item in sends
                    if item.rank == rank and item.chunk_index == chunk
                )
                for chunk in range(draft.chunk_count)
            }
            for chunk in range(2, draft.chunk_count):
                previous = {item.id for item in by_chunk[chunk - 2]}
                if previous and by_chunk[chunk]:
                    self.assertTrue(
                        all(previous.issubset(item.deps) for item in by_chunk[chunk])
                    )
            for chunk in range(1, draft.chunk_count):
                other_slot = {item.id for item in by_chunk[chunk - 1]}
                if other_slot and by_chunk[chunk]:
                    self.assertTrue(
                        all(other_slot.isdisjoint(item.deps) for item in by_chunk[chunk])
                    )
            rank_comps = {
                item.chunk_index: item for item in comps if item.rank == rank
            }
            rank_recvs = {
                chunk: tuple(
                    item
                    for item in recvs
                    if item.rank == rank and item.chunk_index == chunk
                )
                for chunk in range(draft.chunk_count)
            }
            for chunk in range(2, draft.chunk_count):
                previous_comp = rank_comps[chunk - 2]
                self.assertIn(previous_comp.id, rank_comps[chunk].deps)
                self.assertTrue(
                    all(
                        previous_comp.id in item.deps
                        for item in rank_recvs[chunk]
                    )
                )
        candidate = materialize_drafts(problem, (draft,))[0]
        self.assertGreaterEqual(candidate.cost.max_inflight, 2)
        self.assertLessEqual(
            candidate.cost.max_inflight,
            problem.hardware_profile.max_inflight_dte,
        )
        self.assertTrue(
            all(item.double_buffered for item in draft.buffer_requirements)
        )

        single_dte = _with_profile(problem, max_inflight_dte=1)
        self.assertTrue(
            all(
                item.unroll_degree == 1
                for item in generate_wang_1d_drafts(single_dte, witness)
            )
        )

    def test_chunk_limit_and_candidate_cap_are_canonical(self) -> None:
        problem, witness = _scaled(_ag_problem())
        constraints = replace(
            problem.constraints, max_chunk_count=8, max_candidates=5
        )
        limited = SwizzleProblem.create(
            source_ir1_id=problem.source_ir1_id,
            fused_op_id=problem.fused_op_id,
            pattern=problem.pattern,
            gemm=problem.gemm,
            collective=problem.collective,
            group=problem.group,
            hardware_profile=problem.hardware_profile,
            constraints=constraints,
        )
        first = generate_wang_1d_drafts(limited, witness)
        second = generate_wang_1d_drafts(limited, witness)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(
            tuple(item.chunk_count for item in first), (4, 4, 4, 4, 8)
        )

    def test_official_rank_unroll_one_s0_has_no_id_or_decision_drift(self) -> None:
        expected = {
            FusionPattern.AG_GEMM: (
                (
                    "swizzle_candidate_f7ca0279522246f3",
                    "swizzle_candidate_9bc8aaceccbb2875",
                ),
                "swizzle_decision_4bc438e3d04e3e7e",
                (256, 384, 192),
            ),
            FusionPattern.GEMM_RS: (
                (
                    "swizzle_candidate_b54d0f1b427982cf",
                    "swizzle_candidate_4f5ea37a205556f0",
                ),
                "swizzle_decision_c53afd892154c395",
                (256, 384, 192),
            ),
            FusionPattern.GEMM_AR: (
                (
                    "swizzle_candidate_a8d9b73ae1712ea0",
                    "swizzle_candidate_0371e9a9811755d6",
                ),
                "swizzle_decision_087fc2d6b9cc2652",
                (256, 768, 384),
            ),
        }
        fixtures = (
            _ag_problem(),
            _post_problem(CollectiveKind.REDUCE_SCATTER),
            _post_problem(CollectiveKind.ALL_REDUCE),
        )
        for problem, witness in fixtures:
            with self.subTest(pattern=problem.pattern.value):
                s0 = tuple(
                    item
                    for item in generate_wang_1d_drafts(problem, witness)
                    if item.chunk_count == len(problem.collective.participant_ranks)
                    and item.unroll_degree == 1
                )
                candidates = materialize_drafts(problem, s0)
                self.assertEqual(
                    tuple(item.id for item in candidates), expected[problem.pattern][0]
                )
                baseline = build_unfused_baseline(problem, witness)
                decision = decide_swizzle(
                    problem,
                    baseline,
                    candidates,
                )
                self.assertEqual(decision.id, expected[problem.pattern][1])
                selected = next(
                    item
                    for item in decision.ranked_candidates
                    if item.id == decision.selected_candidate_ref
                )
                self.assertIs(baseline.algorithm, SwizzleAlgorithm.UNFUSED)
                self.assertEqual(decision.selected_candidate_ref, baseline.id)
                self.assertIs(selected.algorithm, SwizzleAlgorithm.UNFUSED)
                self.assertIs(
                    decision.decision_reason,
                    SwizzleDecisionReason.NO_PROFITABLE_FUSION,
                )
                expected_flops, expected_bytes, expected_baseline_bytes = (
                    expected[problem.pattern][2]
                )
                for candidate in candidates:
                    actions = _actions(candidate)
                    self.assertEqual(
                        sum(item.flops for item in actions),
                        expected_flops,
                    )
                    self.assertEqual(
                        sum(item.logical_bytes for item in actions),
                        expected_bytes,
                    )
                self.assertEqual(
                    baseline.cost.logical_bytes,
                    expected_baseline_bytes,
                )


if __name__ == "__main__":
    unittest.main()
