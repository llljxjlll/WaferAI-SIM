from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.swizzle_unfused import (
    allocate_unfused_comparison_core_abi,
)
from llm.frontend.wafer_frontend.passes.project_unfused_comparison import (
    _multi_rank_peer_waves,
    build_unfused_comparison_plan,
    project_unfused_comparison,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleGroupView,
    SwizzleProblem,
    SwizzleRankPlacement,
    SwizzleRouteView,
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


def _flexible_rank_problem(decision, rank_count: int):
    """Retype one production scale problem for focused cyclic-wave tests."""

    problem = decision.problem
    ranks = tuple(range(rank_count))
    group = SwizzleGroupView(
        group_ref=f"flexible_mesh_tp_{rank_count}",
        logical_shape=(1, rank_count),
        placements=tuple(
            SwizzleRankPlacement(rank=rank, x=rank, y=0) for rank in ranks
        ),
        routes=tuple(
            SwizzleRouteView(
                id=f"flexible_route_r{source}_r{destination}",
                source_rank=source,
                destination_rank=destination,
                die_path=(source, destination),
                resource_ids=(f"flexible_link_r{source}_r{destination}",),
            )
            for source in ranks
            for destination in ranks
            if source != destination
        ),
    )
    if problem.pattern is FusionPattern.AG_GEMM:
        m, n, k = 8 * rank_count, 48 * rank_count, 64
        gemm = replace(
            problem.gemm,
            m=m,
            n=n,
            k=k,
            lhs=replace(problem.gemm.lhs, shape=(m, k)),
            rhs=replace(problem.gemm.rhs, shape=(k, n)),
            output=replace(problem.gemm.output, shape=(m, n)),
            flops=2 * m * n * k,
        )
        rank_input_bytes = 8 * k * 2
        collective = replace(
            problem.collective,
            participant_ranks=ranks,
            logical_bytes=rank_count * rank_input_bytes,
            rank_input_bytes=rank_input_bytes,
            rank_output_bytes=rank_count * rank_input_bytes,
            input=replace(problem.collective.input, shape=(8, k)),
            output=replace(problem.collective.output, shape=(m, k)),
        )
    elif problem.pattern is FusionPattern.GEMM_RS:
        m, n, k = 8 * rank_count, 64, 16 * rank_count
        gemm = replace(
            problem.gemm,
            m=m,
            n=n,
            k=k,
            lhs=replace(problem.gemm.lhs, shape=(m, k)),
            rhs=replace(problem.gemm.rhs, shape=(k, n)),
            output=replace(problem.gemm.output, shape=(m, n)),
            flops=2 * m * n * k,
        )
        rank_output_bytes = 8 * n * 2
        collective = replace(
            problem.collective,
            participant_ranks=ranks,
            logical_bytes=rank_count * rank_output_bytes,
            rank_input_bytes=rank_count * rank_output_bytes,
            rank_output_bytes=rank_output_bytes,
            input=replace(problem.collective.input, shape=(m, n)),
            output=replace(problem.collective.output, shape=(8, n)),
        )
    else:
        m, n, k = 4 * rank_count, 8, 4 * rank_count
        gemm = replace(
            problem.gemm,
            m=m,
            n=n,
            k=k,
            lhs=replace(problem.gemm.lhs, shape=(m, k)),
            rhs=replace(problem.gemm.rhs, shape=(k, n)),
            output=replace(problem.gemm.output, shape=(m, n)),
            flops=2 * m * n * k,
        )
        rank_bytes = m * n * 2
        collective = replace(
            problem.collective,
            participant_ranks=ranks,
            logical_bytes=rank_bytes,
            rank_input_bytes=rank_bytes,
            rank_output_bytes=rank_bytes,
            input=replace(problem.collective.input, shape=(m, n)),
            output=replace(problem.collective.output, shape=(m, n)),
        )
    problem_semantic = problem._semantic_key()
    problem_semantic.update(gemm=gemm, collective=collective, group=group)
    flexible_problem = SwizzleProblem.create(**problem_semantic)
    baseline_semantic = decision.baseline._semantic_key()
    baseline_semantic["problem_ref"] = flexible_problem.id
    return flexible_problem, SwizzleCandidate.create(**baseline_semantic)


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

    def test_arbitrary_rank_ag_rs_use_complete_circle_waves(self) -> None:
        case = build_swizzle_scale_case(build_swizzle_scale_points()[1])
        decisions = {item.problem.pattern: item for item in case.decisions}
        for rank_count in (3, 6, 10):
            for pattern in (FusionPattern.AG_GEMM, FusionPattern.GEMM_RS):
                with self.subTest(rank_count=rank_count, pattern=pattern.value):
                    problem, baseline = _flexible_rank_problem(
                        decisions[pattern], rank_count
                    )
                    plan = build_unfused_comparison_plan(
                        case.partitioned_graph, problem, baseline
                    )
                    projection = project_unfused_comparison(
                        case.partitioned_graph, plan
                    )
                    plan.validate_against(case.partitioned_graph)
                    projection.validate_against(case.partitioned_graph, plan)

                    payload = (
                        problem.collective.rank_input_bytes
                        if pattern is FusionPattern.AG_GEMM
                        else problem.collective.rank_output_bytes
                    )
                    all_actions = tuple(
                        action
                        for program in plan.rank_programs
                        for action in program.actions
                    )
                    expected_pairs = {
                        (source, destination)
                        for source in range(rank_count)
                        for destination in range(rank_count)
                        if source != destination
                    }
                    self.assertEqual(
                        {
                            (flow.source_rank, flow.destination_rank)
                            for flow in projection.flows
                        },
                        expected_pairs,
                    )
                    self.assertEqual(len(projection.flows), rank_count * (rank_count - 1))
                    for kind in (SwizzleActionKind.SEND, SwizzleActionKind.RECV):
                        self.assertEqual(
                            sum(
                                action.logical_bytes
                                for action in all_actions
                                if action.kind is kind
                            ),
                            rank_count * (rank_count - 1) * payload,
                        )

                    action_index = {action.id: action for action in all_actions}
                    waves = _multi_rank_peer_waves(tuple(range(rank_count)))
                    for program in plan.rank_programs:
                        expected_peers = tuple(
                            dict(wave)[program.rank]
                            for wave in waves
                            if program.rank in dict(wave)
                        )
                        sends = tuple(
                            action for action in program.actions
                            if action.kind is SwizzleActionKind.SEND
                        )
                        recvs = tuple(
                            action for action in program.actions
                            if action.kind is SwizzleActionKind.RECV
                        )
                        waits = tuple(
                            action for action in program.actions
                            if action.kind is SwizzleActionKind.WAIT
                        )
                        self.assertEqual(
                            tuple(action.peer_rank for action in sends),
                            expected_peers,
                        )
                        self.assertEqual(
                            tuple(action.peer_rank for action in recvs),
                            expected_peers,
                        )
                        positions = {
                            action.id: ordinal
                            for ordinal, action in enumerate(program.actions)
                        }
                        for wave, (send, recv, wait) in enumerate(
                            zip(sends, recvs, waits, strict=True)
                        ):
                            matching_send = next(
                                action
                                for action in all_actions
                                if action.kind is SwizzleActionKind.SEND
                                and action.rank == recv.peer_rank
                                and action.peer_rank == recv.rank
                            )
                            self.assertIn(matching_send.id, recv.deps)
                            self.assertEqual(wait.deps, (recv.id,))
                            if program.rank < send.peer_rank:
                                self.assertLess(
                                    positions[send.id], positions[recv.id]
                                )
                                self.assertLess(
                                    positions[recv.id], positions[wait.id]
                                )
                            else:
                                self.assertLess(
                                    positions[recv.id], positions[wait.id]
                                )
                                self.assertLess(
                                    positions[wait.id], positions[send.id]
                                )
                            if wave:
                                predecessor_kind = (
                                    SwizzleActionKind.WAIT
                                    if pattern is FusionPattern.AG_GEMM
                                    else SwizzleActionKind.REDUCE
                                )
                                self.assertTrue(any(
                                    action_index[dep].kind is predecessor_kind
                                    for dep in send.deps
                                ))
                                self.assertTrue(any(
                                    action_index[dep].kind is predecessor_kind
                                    for dep in recv.deps
                                ))

    def test_hundred_rank_circle_wave_planning_is_bounded_and_complete(self) -> None:
        ranks = tuple(range(100))
        waves = _multi_rank_peer_waves(ranks)
        self.assertEqual((len(waves), {len(wave) for wave in waves}), (99, {100}))
        pairs = tuple(pair for wave in waves for pair in wave)
        self.assertEqual(len(pairs), 9_900)
        self.assertEqual(len(set(pairs)), 9_900)
        for wave in waves:
            self.assertEqual(len({source for source, _ in wave}), 100)
            self.assertEqual(len({destination for _, destination in wave}), 100)
            peer = dict(wave)
            self.assertTrue(all(peer[peer[rank]] == rank for rank in ranks))
        cyclic = tuple(
            tuple((rank, (rank + offset) % 100) for rank in ranks)
            for offset in range(1, 100)
        )
        self.assertNotEqual(waves, cyclic)

    def test_three_and_six_rank_ar_is_exact_typed_rs_then_ag(self) -> None:
        decision = self.cases[2].decision
        for rank_count in (3, 6):
            with self.subTest(rank_count=rank_count):
                problem, baseline = _flexible_rank_problem(decision, rank_count)
                plan = build_unfused_comparison_plan(
                    self.cases[2].partitioned_graph, problem, baseline
                )
                projection = project_unfused_comparison(
                    self.cases[2].partitioned_graph, plan
                )
                plan.validate_against(self.cases[2].partitioned_graph)
                projection.validate_against(
                    self.cases[2].partitioned_graph, plan
                )

                peers = rank_count - 1
                waves = _multi_rank_peer_waves(tuple(range(rank_count)))
                actions = {
                    action.id: action
                    for program in plan.rank_programs
                    for action in program.actions
                }
                for program in plan.rank_programs:
                    peer_order = tuple(
                        dict(wave)[program.rank]
                        for wave in waves
                        if program.rank in dict(wave)
                    )

                    def pair(peer: int, reduce: bool):
                        sequence = (
                            (
                                SwizzleActionKind.SEND,
                                SwizzleActionKind.RECV,
                                SwizzleActionKind.WAIT,
                            )
                            if program.rank < peer
                            else (
                                SwizzleActionKind.RECV,
                                SwizzleActionKind.WAIT,
                                SwizzleActionKind.SEND,
                            )
                        )
                        return sequence + ((SwizzleActionKind.REDUCE,) if reduce else ())

                    expected_kinds = (
                        SwizzleActionKind.COMP,
                        SwizzleActionKind.LOCAL_COPY,
                    ) + tuple(kind for peer in peer_order for kind in pair(peer, True)) + tuple(kind for peer in peer_order for kind in pair(peer, False)) + (SwizzleActionKind.BARRIER,)
                    self.assertEqual(
                        tuple(action.kind for action in program.actions),
                        expected_kinds,
                    )
                    self.assertEqual(
                        Counter(
                            action.stage.value
                            for action in program.actions
                            if action.kind is SwizzleActionKind.SEND
                        ),
                        Counter(reduction=peers, replication=peers),
                    )

                self.assertEqual(len(projection.flows), 2 * rank_count * peers)
                self.assertEqual(
                    {
                        (
                            actions[flow.send_task_ref].stage,
                            flow.source_rank,
                            flow.destination_rank,
                        )
                        for flow in projection.flows
                    },
                    {
                        (stage, source, destination)
                        for stage in (
                            actions[next(
                                action.id
                                for program in plan.rank_programs
                                for action in program.actions
                                if action.stage.value == stage_name
                            )].stage
                            for stage_name in ("reduction", "replication")
                        )
                        for source in range(rank_count)
                        for destination in range(rank_count)
                        if source != destination
                    },
                )

                chunk_bytes = problem.collective.rank_output_bytes // rank_count
                operands_by_task = {}
                for operand in projection.operands:
                    operands_by_task.setdefault(operand.task_ref, []).append(operand)
                for program in plan.rank_programs:
                    rank = program.rank
                    barrier = program.actions[-1]
                    output = operands_by_task[barrier.id][0]
                    self.assertEqual(
                        (
                            output.shape,
                            output.byte_extent,
                            output.storage_bytes,
                            output.byte_offset,
                        ),
                        (
                            problem.collective.output.shape,
                            problem.collective.rank_output_bytes,
                            (rank_count + 1) * chunk_bytes,
                            chunk_bytes,
                        ),
                    )
                    reductions = tuple(
                        action
                        for action in program.actions
                        if action.kind is SwizzleActionKind.REDUCE
                    )
                    self.assertEqual(len(reductions), peers)
                    for reduction in reductions:
                        source, accumulator, result = sorted(
                            operands_by_task[reduction.id],
                            key=lambda item: item.ordinal,
                        )
                        self.assertEqual(
                            (
                                source.storage_ref,
                                accumulator.storage_ref,
                                result.storage_ref,
                            ),
                            (output.storage_ref,) * 3,
                        )
                        self.assertEqual(source.byte_offset, rank * chunk_bytes)
                        self.assertEqual(
                            (accumulator.byte_offset, result.byte_offset),
                            ((rank + 1) * chunk_bytes,) * 2,
                        )


if __name__ == "__main__":
    unittest.main()
