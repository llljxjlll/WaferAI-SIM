from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.swizzle_unfused import (
    allocate_unfused_comparison_core_abi,
)
from llm.frontend.wafer_frontend.passes.project_unfused_comparison import (
    build_unfused_comparison_plan,
    project_unfused_comparison,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleProblem,
)
from llm.frontend.wafer_frontend.schema.swizzle_unfused import (
    UnfusedComparisonAction,
    UnfusedComparisonPlan,
    UnfusedComparisonProjection,
)
from llm.frontend.wafer_frontend.schema.swizzle_unfused_abi import (
    UnfusedComparisonCoreABI,
)

from swizzle_cases import build_swizzle_integration_cases
from swizzle_scale_cases import build_swizzle_scale_case, build_swizzle_scale_points


class SwizzleUnfusedComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_swizzle_integration_cases()

    def test_three_exact_unfused_rank_dags_are_deterministic(self) -> None:
        expected = {
            FusionPattern.AG_GEMM: ((4, 4), 14, 2),
            FusionPattern.GEMM_RS: ((6, 6), 22, 2),
            FusionPattern.GEMM_AR: ((10, 10), 30, 4),
        }
        for case in self.cases:
            with self.subTest(pattern=case.pattern.value):
                baseline = case.decision.baseline
                decision_before = case.decision
                self.assertIs(baseline.algorithm, SwizzleAlgorithm.UNFUSED)
                plan = build_unfused_comparison_plan(
                    case.partitioned_graph,
                    case.decision.problem,
                    baseline,
                )
                projection = project_unfused_comparison(
                    case.partitioned_graph,
                    plan,
                )
                plan.validate_against(case.partitioned_graph)
                projection.validate_against(case.partitioned_graph, plan)
                self.assertEqual(
                    (
                        tuple(len(item.actions) for item in plan.rank_programs),
                        len(projection.operands),
                        len(projection.flows),
                    ),
                    expected[case.pattern],
                )
                self.assertEqual(
                    build_unfused_comparison_plan(
                        case.partitioned_graph,
                        case.decision.problem,
                        baseline,
                    ),
                    plan,
                )
                self.assertEqual(
                    project_unfused_comparison(case.partitioned_graph, plan),
                    projection,
                )
                self.assertEqual(case.decision, decision_before)
                selected = next(
                    item
                    for item in case.decision.ranked_candidates
                    if item.id == case.decision.selected_candidate_ref
                )
                self.assertIs(selected.algorithm, SwizzleAlgorithm.UNFUSED)
                self.assertEqual(
                    sum(
                        action.flops
                        for program in plan.rank_programs
                        for action in program.actions
                    ),
                    plan.problem.gemm.flops,
                )
                self.assertTrue(all(
                    operand.byte_offset + operand.byte_extent <= operand.storage_bytes
                    for operand in projection.operands
                ))
                if case.pattern in (FusionPattern.AG_GEMM, FusionPattern.GEMM_AR):
                    shared = {}
                    for operand in projection.operands:
                        shared.setdefault(operand.storage_ref, []).append(operand)
                    self.assertTrue(any(
                        len({item.byte_offset for item in views}) >= 2
                        and any(item.byte_offset == 0 and item.byte_extent == item.storage_bytes for item in views)
                        for views in shared.values()
                    ))

    def test_fused_candidate_cannot_replace_unfused_baseline(self) -> None:
        case = self.cases[0]
        fused = next(
            item
            for item in case.decision.ranked_candidates
            if item.algorithm is not SwizzleAlgorithm.UNFUSED
        )
        with self.assertRaisesRegex(SchemaError, "UNFUSED baseline"):
            UnfusedComparisonPlan.create(
                source_ir1_id=case.partitioned_graph.id,
                problem=case.decision.problem,
                baseline=fused,
                pattern=case.pattern,
                rank_programs=(),
            )

    def test_restabled_plan_and_projection_tamper_fail_exact_rebuild(self) -> None:
        case = self.cases[2]
        plan = build_unfused_comparison_plan(
            case.partitioned_graph,
            case.decision.problem,
            case.decision.baseline,
        )
        projection = project_unfused_comparison(case.partitioned_graph, plan)

        first_program = plan.rank_programs[0]
        terminal_action = first_program.actions[-1]
        action_semantic = terminal_action._semantic_key()
        action_semantic["member_ref"] = "forged_member"
        forged_action = UnfusedComparisonAction.create(**action_semantic)
        forged_plan = UnfusedComparisonPlan.create(
            source_ir1_id=plan.source_ir1_id,
            problem=plan.problem,
            baseline=plan.baseline,
            pattern=plan.pattern,
            rank_programs=(
                replace(
                    first_program,
                    actions=(*first_program.actions[:-1], forged_action),
                ),
                plan.rank_programs[1],
            ),
        )
        with self.assertRaisesRegex(SchemaError, "exact deterministic"):
            forged_plan.validate_against(case.partitioned_graph)

        first_operand = projection.operands[0]
        forged_operand = replace(
            first_operand,
            source_tensor_ref="forged_tensor",
        )
        forged_projection = UnfusedComparisonProjection.create(
            source_ir1_id=projection.source_ir1_id,
            source_plan_ref=projection.source_plan_ref,
            problem_ref=projection.problem_ref,
            baseline_ref=projection.baseline_ref,
            pattern=projection.pattern,
            ranks=projection.ranks,
            operands=(forged_operand, *projection.operands[1:]),
            flows=projection.flows,
        )
        with self.assertRaisesRegex(SchemaError, "exact deterministic"):
            forged_projection.validate_against(case.partitioned_graph, plan)

        with self.assertRaisesRegex(SchemaError, "exceeds its explicit storage"):
            UnfusedComparisonProjection.create(
                source_ir1_id=projection.source_ir1_id,
                source_plan_ref=projection.source_plan_ref,
                problem_ref=projection.problem_ref,
                baseline_ref=projection.baseline_ref,
                pattern=projection.pattern,
                ranks=projection.ranks,
                operands=(
                    replace(first_operand, storage_bytes=first_operand.byte_extent - 1),
                    *projection.operands[1:],
                ),
                flows=projection.flows,
            )

        with self.assertRaisesRegex(SchemaError, "tensor offset"):
            replace(first_operand, tensor_offset=()).validate("operand")
        forged_offset = replace(
            first_operand,
            tensor_offset=tuple(item + 1 for item in first_operand.tensor_offset),
        )
        forged_projection = UnfusedComparisonProjection.create(
            source_ir1_id=projection.source_ir1_id,
            source_plan_ref=projection.source_plan_ref,
            problem_ref=projection.problem_ref,
            baseline_ref=projection.baseline_ref,
            pattern=projection.pattern,
            ranks=projection.ranks,
            operands=(forged_offset, *projection.operands[1:]),
            flows=projection.flows,
        )
        with self.assertRaisesRegex(SchemaError, "exact deterministic"):
            forged_projection.validate_against(case.partitioned_graph, plan)

    def test_four_rank_ag_rs_serde_work_actions_and_routes_are_exact(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[1])
        expected = {
            FusionPattern.AG_GEMM: (
                (10, 10, 10, 10),
                48,
                Counter(send=12, recv=12, wait=12, comp=4),
            ),
            FusionPattern.GEMM_RS: (
                (14, 14, 14, 14),
                92,
                Counter(
                    send=12,
                    recv=12,
                    wait=12,
                    reduce=12,
                    comp=4,
                    local_copy=4,
                ),
            ),
        }
        for decision in case.decisions:
            with self.subTest(pattern=decision.problem.pattern.value):
                plan = build_unfused_comparison_plan(
                    case.partitioned_graph, decision.problem, decision.baseline,
                )
                projection = project_unfused_comparison(
                    case.partitioned_graph, plan,
                )
                round_trip_plan = loads_dataclass(
                    UnfusedComparisonPlan,
                    canonical_json(plan),
                    path="plan",
                )
                round_trip_projection = loads_dataclass(
                    UnfusedComparisonProjection,
                    canonical_json(projection),
                    path="projection",
                )
                self.assertEqual((round_trip_plan, round_trip_projection), (plan, projection))

                actions = tuple(
                    action
                    for program in plan.rank_programs
                    for action in program.actions
                )
                rank_actions, operand_count, kinds = expected[plan.pattern]
                self.assertEqual(
                    (
                        tuple(len(program.actions) for program in plan.rank_programs),
                        len(projection.operands),
                        len(projection.flows),
                    ),
                    (rank_actions, operand_count, 12),
                )
                self.assertEqual(Counter(action.kind.value for action in actions), kinds)
                payload = (
                    plan.problem.collective.rank_input_bytes
                    if plan.pattern is FusionPattern.AG_GEMM
                    else plan.problem.collective.rank_output_bytes
                )
                for kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
                    self.assertEqual(
                        sum(
                            action.logical_bytes
                            for action in actions
                            if action.kind is kind
                        ),
                        4 * 3 * payload,
                    )
                self.assertEqual(
                    sum(action.flops for action in actions),
                    plan.problem.gemm.flops,
                )
                output_views = tuple(
                    operand
                    for operand in projection.operands
                    if operand.source_tensor_ref
                    == plan.problem.gemm.output.value_ref
                    and next(
                        action
                        for action in actions
                        if action.id == operand.task_ref
                    ).kind is SwizzleActionKind.COMP
                    and operand.use.value == "write"
                )
                self.assertEqual(len(output_views), 4)
                if plan.pattern is FusionPattern.AG_GEMM:
                    self.assertEqual(
                        tuple(
                            (item.shape, item.tensor_offset)
                            for item in output_views
                        ),
                        tuple(
                            ((32, 48), (0, rank * 48))
                            for rank in range(4)
                        ),
                    )
                routes = {
                    (route.source_rank, route.destination_rank): route
                    for route in plan.problem.group.routes
                }
                self.assertEqual(
                    {(flow.source_rank, flow.destination_rank) for flow in projection.flows},
                    set(routes),
                )
                for flow in projection.flows:
                    route = routes[(flow.source_rank, flow.destination_rank)]
                    self.assertEqual((flow.route_ref, flow.die_path), (route.id, route.die_path))

                core_abi = allocate_unfused_comparison_core_abi(
                    case.partitioned_graph, plan, projection
                )
                runtime_by_task = {
                    item.task_ref: item for item in core_abi.runtime_bindings
                }
                for program in plan.rank_programs:
                    sends = tuple(
                        action
                        for action in program.actions
                        if action.kind is SwizzleActionKind.SEND
                    )
                    recvs = tuple(
                        action
                        for action in program.actions
                        if action.kind is SwizzleActionKind.RECV
                    )
                    waits = tuple(
                        action
                        for action in program.actions
                        if action.kind is SwizzleActionKind.WAIT
                    )
                    self.assertEqual(
                        tuple(action.peer_rank for action in sends),
                        tuple(action.peer_rank for action in recvs),
                    )
                    for index in (1, 2):
                        predecessor = (
                            waits[index - 1]
                            if plan.pattern is FusionPattern.AG_GEMM
                            else next(
                                action
                                for action in program.actions
                                if action.kind is SwizzleActionKind.REDUCE
                                and action.id in sends[index].deps
                            )
                        )
                        self.assertIn(predecessor.id, sends[index].deps)
                        self.assertIn(predecessor.id, recvs[index].deps)
                    tokens = tuple(
                        runtime_by_task[action.id].token_symbol_ref
                        for action in recvs
                    )
                    self.assertEqual(len(set(tokens)), 3)
                    self.assertEqual(
                        tokens,
                        tuple(
                            runtime_by_task[action.id].token_symbol_ref
                            for action in waits
                        ),
                    )

                if plan.pattern is FusionPattern.AG_GEMM:
                    third_wait = tuple(
                        action
                        for action in plan.rank_programs[0].actions
                        if action.kind is SwizzleActionKind.WAIT
                    )[2]
                    binding_index = next(
                        index
                        for index, item in enumerate(core_abi.runtime_bindings)
                        if item.task_ref == third_wait.id
                    )
                    forged_bindings = list(core_abi.runtime_bindings)
                    forged_bindings[binding_index] = replace(
                        forged_bindings[binding_index],
                        token_symbol_ref="forged_non_reused_token",
                    )
                    forged_semantic = core_abi._semantic_key()
                    forged_semantic["runtime_bindings"] = tuple(forged_bindings)
                    forged = UnfusedComparisonCoreABI.create(**forged_semantic)
                    with self.assertRaisesRegex(SchemaError, "exact deterministic"):
                        forged.validate_against(
                            case.partitioned_graph, plan, projection
                        )

                if plan.pattern is FusionPattern.GEMM_RS:
                    for program in plan.rank_programs:
                        recvs = tuple(
                            action for action in program.actions
                            if action.kind is SwizzleActionKind.RECV
                        )
                        reductions = tuple(
                            action for action in program.actions
                            if action.kind is SwizzleActionKind.REDUCE
                        )
                        self.assertEqual(
                            len({action.write_value_refs for action in recvs}),
                            1,
                        )
                        for recv, previous in zip(
                            recvs[1:], reductions[:-1], strict=True
                        ):
                            self.assertIn(previous.id, recv.deps)

    def test_four_rank_missing_exact_route_fails_closed(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[1])
        decision = case.decisions[0]
        problem_semantic = decision.problem._semantic_key()
        problem_semantic["group"] = replace(
            decision.problem.group,
            routes=decision.problem.group.routes[:-1],
        )
        problem = SwizzleProblem.create(**problem_semantic)
        baseline_semantic = decision.baseline._semantic_key()
        baseline_semantic["problem_ref"] = problem.id
        baseline = SwizzleCandidate.create(**baseline_semantic)
        with self.assertRaisesRegex(SchemaError, "one exact directed route"):
            build_unfused_comparison_plan(
                case.partitioned_graph, problem, baseline,
            )


if __name__ == "__main__":
    unittest.main()
