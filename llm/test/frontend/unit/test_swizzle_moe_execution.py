from __future__ import annotations

from collections import Counter
from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
    validate_moe_swizzle_c0_execution,
    validate_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionActionKind,
    MoeScaleExecution,
    MoeScaleExecutionCapacityResult,
    MoeScaleExecutionMode,
    MoeScaleExecutionFlowRole,
    MoeScaleExecutionTerminalKind,
)
from llm.test.frontend.integration.lite_moe_dp4_cases import (
    LiteMoeDp4Mode,
    build_lite_moe_dp4_case,
)
from llm.test.frontend.integration.moe_swizzle_scale_cases import (
    build_moe_swizzle_scale_cases,
)


class MoeScaleExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = build_moe_swizzle_scale_cases()
        cls.executions = tuple(
            (
                build_moe_swizzle_execution(
                    case.spec,
                    case.oracle,
                    MoeScaleExecutionMode.INFER_FORWARD,
                ),
                build_moe_swizzle_execution(
                    case.spec,
                    case.oracle,
                    MoeScaleExecutionMode.TRAIN_FORWARD,
                ),
            )
            for case in cls.cases
        )

    def test_c0_c4_action_flow_terminal_and_capacity_formulas(self) -> None:
        expected = (
            ("C0", 92, 100, 12, 8, 16, 24576, True, (2, 2, 2, 2)),
            ("C1", 368, 400, 48, 32, 64, 98304, True, (8, 8, 8, 8)),
            ("C2", 736, 800, 96, 64, 128, 196608, True, (16, 16, 16, 16)),
            ("C3", 1472, 1600, 192, 128, 256, 393216, True, (32, 32, 32, 32)),
            ("C4", 736, 800, 96, 64, 128, 196608, True, (32, 16, 8, 8)),
        )
        actual = tuple(
            (
                case.spec.name,
                len(infer.actions),
                len(train.actions),
                len(infer.flows),
                len(infer.terminals),
                len(train.terminals),
                sum(item.flops for item in infer.actions),
                infer.execution_ready,
                infer.capacity.required_slots_by_expert,
            )
            for case, (infer, train) in zip(self.cases, self.executions, strict=True)
        )
        self.assertEqual(actual, expected)
        for case, (infer, train) in zip(self.cases, self.executions, strict=True):
            remote = len(case.oracle.remote_token_indices)
            self.assertEqual(
                Counter(item.role for item in infer.flows),
                Counter({MoeScaleExecutionFlowRole.DISPATCH: remote, MoeScaleExecutionFlowRole.COMBINE: remote}),
            )
            self.assertEqual(
                sum(item.bytes for item in infer.flows),
                case.oracle.logical_p2p_bytes,
            )
            self.assertEqual(
                sum(item.bytes for item in train.terminals if item.kind is MoeScaleExecutionTerminalKind.TAPE),
                case.oracle.train_tape_terminal_bytes,
            )
            self.assertTrue(infer.capacity.admitted)

    def test_balanced_and_skewed_capacity_admission_is_typed(self) -> None:
        for index in (3, 4):
            infer = self.executions[index][0]
            self.assertTrue(infer.execution_ready)
            self.assertTrue(infer.capacity.admitted)
            required = infer.capacity.required_slots_by_expert
            configured = list(infer.capacity.configured_slots_by_expert)
            configured[0] = required[0] - 1
            rejected = MoeScaleExecutionCapacityResult.create(
                required_slots_by_expert=required,
                configured_slots_by_expert=tuple(configured),
                admitted=False,
            )
            self.assertFalse(rejected.admitted)
            with self.assertRaisesRegex(SchemaError, "capacity admission"):
                replace(rejected, admitted=True).validate()

    def test_c0_cross_validates_frozen_legacy_execution(self) -> None:
        legacy = build_lite_moe_dp4_case(LiteMoeDp4Mode.TRAIN_FORWARD)
        c0 = self.cases[0]
        infer, train = self.executions[0]
        assert legacy.train_forward is not None
        validate_moe_swizzle_c0_execution(
            infer,
            train,
            legacy.forward,
            legacy.train_forward,
            c0.spec,
            c0.oracle,
        )
        self.assertEqual(
            Counter(item.kind for item in infer.actions),
            Counter({
                MoeScaleExecutionActionKind.DMA_IN: 24,
                MoeScaleExecutionActionKind.GEMM: 24,
                MoeScaleExecutionActionKind.SWIGLU: 8,
                MoeScaleExecutionActionKind.SEND: 12,
                MoeScaleExecutionActionKind.RECV: 12,
                MoeScaleExecutionActionKind.WAIT: 12,
            }),
        )

    def test_strict_serde_and_deterministic_ids(self) -> None:
        for execution in self.executions[0]:
            self.assertEqual(
                loads_dataclass(
                    MoeScaleExecution,
                    canonical_json(execution),
                    path="execution",
                ),
                execution,
            )
        rebuilt = build_moe_swizzle_execution(
            self.cases[2].spec,
            self.cases[2].oracle,
            MoeScaleExecutionMode.INFER_FORWARD,
        )
        self.assertEqual(rebuilt.id, self.executions[2][0].id)

    def test_dependency_route_terminal_and_capacity_tampers_fail_closed(self) -> None:
        case = self.cases[1]
        infer = self.executions[1][0]
        action = next(item for item in infer.actions if item.deps)
        forged_action = replace(action, deps=())
        forged_actions = tuple(
            forged_action if item.id == action.id else item for item in infer.actions
        )
        with self.assertRaisesRegex(SchemaError, "topological order|unstable artifact id"):
            replace(infer, actions=forged_actions).validate()

        flow = next(item for item in infer.flows if len(item.die_path) == 3)
        wrong_route = replace(flow, die_path=(flow.source_die_id, flow.destination_die_id))
        semantic = infer._semantic_key() if hasattr(infer, "_semantic_key") else {
            name: getattr(infer, name)
            for name in infer.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }
        semantic["flows"] = tuple(
            wrong_route if item.id == flow.id else item for item in infer.flows
        )
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            MoeScaleExecution.create(**semantic)

        terminal = infer.terminals[0]
        with self.assertRaisesRegex(SchemaError, "terminal shape/bytes"):
            replace(terminal, bytes=terminal.bytes + 2).validate()
        capacity = replace(infer.capacity, admitted=False)
        with self.assertRaisesRegex(SchemaError, "capacity admission"):
            capacity.validate()


if __name__ == "__main__":
    unittest.main()
