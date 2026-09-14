from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_moe import (
    build_round_robin_flexible_moe_spec,
    compile_flexible_moe_baseline,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.flexible_moe import (
    FlexibleMoeLimits,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectActionKind,
    MoeRectFlowStage,
    MoeRectStateRole,
    MoeRectStaticTrace,
    MoeRectTraceAssignment,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec


def _hot_spec(mode: FlexibleMoeMode) -> FlexibleMoeSpec:
    mesh = RectMeshSpec(2, 3)
    assignments = tuple(
        MoeRectTraceAssignment(token, token, 0, 0, token)
        for token in range(mesh.rank_count)
    )
    trace = MoeRectStaticTrace.create(
        token_count=mesh.rank_count,
        expert_count=mesh.rank_count,
        capacity_per_expert=mesh.rank_count,
        assignments=assignments,
    )
    return FlexibleMoeSpec.create(
        mesh=mesh,
        mode=mode,
        hidden_size=16,
        intermediate_size=32,
        expert_count=mesh.rank_count,
        expert_parallel_degree=mesh.rank_count,
        top_k=1,
        trace_mode="static",
        trace=trace,
        limits=FlexibleMoeLimits(),
        expert_dtype=DType.FP16,
        combine_dtype=DType.FP32,
        token_drop=False,
    )


class FlexibleMoeTest(unittest.TestCase):
    def test_all_100_rectangles_compile_inference_and_train_baselines(self) -> None:
        seen = set()
        for rows in range(1, 11):
            for columns in range(1, 11):
                mesh = RectMeshSpec(rows, columns)
                ranks = mesh.rank_count
                for mode in FlexibleMoeMode:
                    spec = build_round_robin_flexible_moe_spec(
                        mesh,
                        mode,
                        routing_shift=0 if ranks == 1 else 1,
                    )
                    plan = compile_flexible_moe_baseline(spec)
                    plan.validate_against(spec)
                    expected_flows = 0 if ranks == 1 else (
                        2 * ranks if mode is FlexibleMoeMode.INFERENCE
                        else 4 * ranks + 2 * (ranks - 1)
                    )
                    self.assertEqual(len(plan.flows), expected_flows)
                    self.assertEqual(
                        len(plan.state_bindings),
                        ranks * (2 if mode is FlexibleMoeMode.INFERENCE else 4),
                    )
                    self.assertEqual(
                        len(plan.terminal_action_refs),
                        ranks if mode is FlexibleMoeMode.INFERENCE else 2 * ranks,
                    )
                    self.assertLessEqual(plan.symbolic_record_count, spec.limits.max_records)
                    seen.add((rows, columns, mode))
        self.assertEqual(len(seen), 200)

    def test_pair_bucket_waves_routes_bytes_and_repeatability(self) -> None:
        spec = build_round_robin_flexible_moe_spec(
            RectMeshSpec(3, 2),
            FlexibleMoeMode.INFERENCE,
            tokens_per_rank=2,
            routing_shift=1,
        )
        first = compile_flexible_moe_baseline(spec)
        second = compile_flexible_moe_baseline(spec)
        self.assertEqual(first, second)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(first.flows), 2 * spec.mesh.rank_count)
        self.assertTrue(all(item.wave_index == 1 or item.wave_index == 5 for item in first.flows))
        self.assertEqual(
            sum(item.logical_bytes for item in first.flows),
            2 * spec.trace.token_count * spec.hidden_size * 2,
        )
        sessions = {}
        for flow in first.flows:
            for rank in (flow.source_rank, flow.destination_rank):
                key = (flow.stage, flow.wave_index, rank)
                sessions[key] = sessions.get(key, 0) + 1
        self.assertEqual(max(sessions.values()), 2)
        for flow in first.flows:
            source_x, source_y = spec.mesh.coordinate(flow.source_rank)
            destination_x, destination_y = spec.mesh.coordinate(flow.destination_rank)
            horizontal_hops = abs(destination_x - source_x)
            self.assertEqual(
                tuple(spec.mesh.coordinate(rank)[1] for rank in flow.die_path[: horizontal_hops + 1]),
                (source_y,) * (horizontal_hops + 1),
            )

    def test_all_local_and_hot_empty_expert_traces_are_typed(self) -> None:
        local_spec = build_round_robin_flexible_moe_spec(
            RectMeshSpec(2, 3), FlexibleMoeMode.INFERENCE, routing_shift=0
        )
        local = compile_flexible_moe_baseline(local_spec)
        self.assertEqual(local.flows, ())
        self.assertFalse(any(
            item.kind in (MoeRectActionKind.SEND, MoeRectActionKind.RECV, MoeRectActionKind.WAIT)
            for item in local.actions
        ))

        hot_spec = _hot_spec(FlexibleMoeMode.INFERENCE)
        hot = compile_flexible_moe_baseline(hot_spec)
        self.assertEqual(hot_spec.trace.expert_histogram, (6, 0, 0, 0, 0, 0))
        self.assertEqual(len(hot.flows), 10)
        self.assertEqual(
            sum(item.logical_bytes for item in hot.flows if item.stage is MoeRectFlowStage.DISPATCH),
            5 * hot_spec.hidden_size * 2,
        )
        expert_actions = tuple(
            item for item in hot.actions
            if item.kind is MoeRectActionKind.EXPERT_FORWARD
        )
        self.assertEqual(len(expert_actions), 6)
        self.assertEqual(
            tuple(len(item.assignment_refs) for item in expert_actions),
            (6, 0, 0, 0, 0, 0),
        )
        flow_by_id = {item.id: item for item in hot.flows}
        root_dispatch_receives = tuple(
            flow_by_id[item.flow_ref].source_rank
            for item in hot.actions
            if item.kind is MoeRectActionKind.RECV
            and item.flow_ref is not None
            and flow_by_id[item.flow_ref].stage is MoeRectFlowStage.DISPATCH
            and item.rank == 0
        )
        self.assertEqual(root_dispatch_receives, (5, 4, 3, 2, 1))
        root_dispatch = tuple(
            flow for flow in hot.flows
            if flow.stage is MoeRectFlowStage.DISPATCH
            and flow.destination_rank == 0
        )
        self.assertEqual(
            tuple(flow.wave_index for flow in root_dispatch),
            (1, 2, 3, 4, 5),
        )
        groups = tuple(
            tuple(action for action in hot.actions if action.flow_ref == flow.id)
            for flow in root_dispatch
        )
        start = min(hot.actions.index(action) for group in groups for action in group)
        stop = max(hot.actions.index(action) for group in groups for action in group) + 1
        self.assertEqual(stop - start, 3 * len(groups))
        reversed_fan_in = (
            hot.actions[:start]
            + tuple(action for group in reversed(groups) for action in group)
            + hot.actions[stop:]
        )
        with self.assertRaisesRegex(SchemaError, "cyclic-wave order"):
            replace(hot, actions=reversed_fan_in).validate_against(hot_spec)
        second_send = groups[1][0]
        self.assertEqual(second_send.kind, MoeRectActionKind.SEND)
        missing_fence = tuple(
            replace(
                action,
                deps=action.deps[:-1],
            )
            if action.id == second_send.id else action
            for action in hot.actions
        )
        with self.assertRaisesRegex(SchemaError, "previous wave WAIT"):
            replace(hot, actions=missing_fence).validate_against(hot_spec)

        root_combine_sends = tuple(
            action for action in hot.actions
            if action.kind is MoeRectActionKind.SEND
            and action.flow_ref is not None
            and flow_by_id[action.flow_ref].stage is MoeRectFlowStage.COMBINE
            and action.rank == 0
        )
        self.assertEqual(len(root_combine_sends), 5)
        second_combine_send = root_combine_sends[1]
        self.assertTrue(any(
            next(item for item in hot.actions if item.id == ref).kind
            is MoeRectActionKind.WAIT
            for ref in second_combine_send.deps
        ))
        missing_fan_out_fence = tuple(
            replace(action, deps=action.deps[:-1])
            if action.id == second_combine_send.id else action
            for action in hot.actions
        )
        with self.assertRaisesRegex(SchemaError, "fan-out SEND requires"):
            replace(hot, actions=missing_fan_out_fence).validate_against(hot_spec)

    def test_balanced_2x3_canonical_plan_is_unchanged_by_fan_in_guard(self) -> None:
        expected = {
            FlexibleMoeMode.INFERENCE: (
                "flexible_moe_plan_7aff7a970ba10a38",
                "5eb37ce6c3fa29a2cb918f91acd3a46a1e177cd2c9240e2f7974ecba7a8156ca",
            ),
            FlexibleMoeMode.TRAIN: (
                "flexible_moe_plan_db4a5ebb24b89ee3",
                "0462c62b02a5b3c488117a866cee55a1c96c35fce473a7eb40da99d396f3dc1f",
            ),
        }
        from llm.frontend.wafer_frontend.schema.serde import canonical_digest

        for mode in FlexibleMoeMode:
            spec = build_round_robin_flexible_moe_spec(
                RectMeshSpec(2, 3), mode, routing_shift=1,
            )
            plan = compile_flexible_moe_baseline(spec)
            self.assertEqual((plan.id, canonical_digest(plan)), expected[mode])

    def test_train_has_four_way_routes_wgrad_ar_sgd_and_state_closure(self) -> None:
        spec = _hot_spec(FlexibleMoeMode.TRAIN)
        plan = compile_flexible_moe_baseline(spec)
        self.assertEqual(set(item.stage for item in plan.flows), set(MoeRectFlowStage))
        by_assignment = {}
        for flow in plan.flows:
            for ref in flow.assignment_refs:
                by_assignment.setdefault(ref, {})[flow.stage] = flow
        for assignment in spec.trace.assignments:
            if assignment.source_rank == assignment.expert_home_rank:
                continue
            stages = by_assignment[f"assignment.{assignment.token_index}"]
            self.assertEqual(set(stages), {
                MoeRectFlowStage.DISPATCH,
                MoeRectFlowStage.COMBINE,
                MoeRectFlowStage.BACKWARD_GRADIENT,
                MoeRectFlowStage.BACKWARD_DX,
            })
            self.assertEqual(
                (stages[MoeRectFlowStage.DISPATCH].source_rank, stages[MoeRectFlowStage.DISPATCH].destination_rank),
                (stages[MoeRectFlowStage.BACKWARD_GRADIENT].source_rank, stages[MoeRectFlowStage.BACKWARD_GRADIENT].destination_rank),
            )
        by_id = {item.id: item for item in plan.actions}
        flow_by_id = {flow.id: flow for flow in plan.flows}
        reduce_sends = tuple(
            action for action in plan.actions
            if action.kind is MoeRectActionKind.SEND
            and action.flow_ref is not None
            and flow_by_id[action.flow_ref].stage is MoeRectFlowStage.GATE_ALL_REDUCE
            and flow_by_id[action.flow_ref].source_rank > flow_by_id[action.flow_ref].destination_rank
        )
        self.assertEqual(len(reduce_sends), spec.mesh.rank_count - 1)
        self.assertTrue(all(
            {by_id[ref].kind for ref in action.deps}
            == {MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE}
            for action in reduce_sends
        ))
        for action in plan.actions:
            dep_kinds = {by_id[ref].kind for ref in action.deps}
            if action.kind is MoeRectActionKind.EXPERT_SGD:
                self.assertEqual(dep_kinds, {MoeRectActionKind.EXPERT_WGRAD})
            elif action.kind is MoeRectActionKind.GATE_SGD:
                self.assertEqual(dep_kinds, {MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE})
            elif action.kind is MoeRectActionKind.GATE_GRADIENT_ALL_REDUCE:
                self.assertEqual(
                    dep_kinds,
                    {MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE}
                    if action.rank == 0 else {
                        MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE,
                        MoeRectActionKind.WAIT,
                    },
                )
            elif action.kind is MoeRectActionKind.STATE_STORE:
                self.assertEqual(len(action.state_refs), 1)
                role = next(
                    state.role for state in plan.state_bindings
                    if state.id == action.state_refs[0]
                )
                self.assertEqual(
                    dep_kinds,
                    {{
                        MoeRectStateRole.EXPERT_PARAMETER: MoeRectActionKind.EXPERT_SGD,
                        MoeRectStateRole.GATE_PARAMETER: MoeRectActionKind.GATE_SGD,
                    }[role]},
                )
        self.assertIsNotNone(plan.gate_all_reduce)
        self.assertEqual(plan.gate_all_reduce.wave_count, spec.mesh.rank_count - 1)
        self.assertEqual(
            {item.role for item in plan.state_bindings},
            set(MoeRectStateRole),
        )
        for state in plan.state_bindings:
            if state.expert_index is not None:
                self.assertEqual(state.owner_rank, state.expert_index)
        with self.assertRaisesRegex(SchemaError, "binary-tree reduce/broadcast pairs"):
            replace(plan, flows=plan.flows[:-1]).validate_against(spec)
        local_reduce = next(
            action for action in plan.actions
            if action.kind is MoeRectActionKind.GATE_GRADIENT_LOCAL_REDUCE
            and len(action.deps) > 1
        )
        drifted_actions = tuple(
            replace(action, deps=action.deps[:-1])
            if action.id == local_reduce.id else action
            for action in plan.actions
        )
        with self.assertRaisesRegex(SchemaError, "exact child tree edges"):
            replace(plan, actions=drifted_actions).validate_against(spec)
        reduce_send = next(
            action for action in plan.actions
            if action.kind is MoeRectActionKind.SEND
            and action.flow_ref is not None
            and flow_by_id[action.flow_ref].stage is MoeRectFlowStage.GATE_ALL_REDUCE
            and flow_by_id[action.flow_ref].source_rank > flow_by_id[action.flow_ref].destination_rank
        )
        drifted_actions = tuple(
            replace(action, deps=(gate_wgrad.id,))
            if action.id == reduce_send.id else action
            for action in plan.actions
            for gate_wgrad in (
                next(item for item in plan.actions if item.kind is MoeRectActionKind.GATE_WGRAD),
            )
        )
        with self.assertRaisesRegex(SchemaError, "SEND predecessor"):
            replace(plan, actions=drifted_actions).validate_against(spec)

    def test_invalid_trace_and_capacity_fail_before_materialization(self) -> None:
        with self.assertRaisesRegex(SchemaError, "slots"):
            MoeRectStaticTrace.create(
                token_count=2,
                expert_count=1,
                capacity_per_expert=2,
                assignments=(
                    MoeRectTraceAssignment(0, 0, 0, 0, 0),
                    MoeRectTraceAssignment(1, 0, 0, 0, 0),
                ),
            )
        spec = build_round_robin_flexible_moe_spec(
            RectMeshSpec(2, 3), FlexibleMoeMode.TRAIN
        )
        constrained = FlexibleMoeSpec.create(
            **(
                spec._semantic()
                | {"limits": replace(spec.limits, max_actions=1)}
            )
        )
        with patch(
            "llm.frontend.wafer_frontend.passes.flexible_moe._states",
            side_effect=AssertionError("state/action materialization must not run"),
        ):
            with self.assertRaisesRegex(SchemaError, "capacity preflight"):
                compile_flexible_moe_baseline(constrained)


if __name__ == "__main__":
    unittest.main()
