from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    FUSION_ACTION_EXTENSION_FIELDS,
    FUSION_PLAN_EXTENSION_FIELDS,
    force_swizzle_deployment,
    materialize_swizzle_decision,
)
from llm.frontend.wafer_frontend.policies.swizzle.materialize_ir1 import (
    _index_temporary_producers,
)
from llm.frontend.wafer_frontend.schema.action import FusionActionKind
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleActionKind,
    SwizzleActionWitness,
    SwizzleAlgorithm,
    SwizzleCandidate,
    SwizzleDecision,
    SwizzleDecisionReason,
    SwizzleFeasibilityCheck,
    SwizzleFeasibilityWitness,
    SwizzleTopologyKind,
    SwizzleTopologyWitness,
    SwizzlePhase,
)
from llm.frontend.wafer_frontend.schema.swizzle_plan import (
    SwizzleDeploymentReason,
)

import test_swizzle_action_schema as action_fixture
import test_swizzle_schema as schema_fixture


def _decision(*, fused: bool) -> SwizzleDecision:
    candidate = action_fixture.SwizzleActionSchemaTest()._candidate()
    problem = action_fixture.SwizzleActionSchemaTest()._candidate().problem_ref
    # Recover the exact problem embedded by the canonical fixture through the
    # candidate's builder.  The builder is deterministic, so the id must agree.
    typed_problem, witness = schema_fixture.SwizzleSchemaTest()._problem()
    assert problem == typed_problem.id
    baseline = SwizzleCandidate.create(
        problem_ref=typed_problem.id,
        pattern=typed_problem.pattern,
        algorithm=SwizzleAlgorithm.UNFUSED,
        split_axis=None,
        chunk_count=0,
        unroll_degree=0,
        rank_programs=(),
        buffer_requirements=(),
        topology_witness=SwizzleTopologyWitness(
            kind=SwizzleTopologyKind.UNFUSED,
            rank_order=(),
            row_orders=(),
            column_orders=(),
            route_refs=(),
            is_complete_rectangle=False,
            has_hamiltonian_cycle=False,
        ),
        semantic_witness=witness,
        feasibility_witness=SwizzleFeasibilityWitness(
            (SwizzleFeasibilityCheck("baseline", True, "always retained"),)
        ),
        cost=schema_fixture.SwizzleSchemaTest()._cost(),
    )
    if fused:
        ranked = (candidate, baseline)
        reason = SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES
    else:
        ranked = (baseline, candidate)
        reason = SwizzleDecisionReason.NO_PROFITABLE_FUSION
    return SwizzleDecision.create(
        problem=typed_problem,
        baseline=baseline,
        ranked_candidates=ranked,
        selected_candidate_ref=ranked[0].id,
        decision_reason=reason,
    )


class SwizzleMaterializeTest(unittest.TestCase):
    def test_selected_candidate_is_retained_and_routes_are_closed(self) -> None:
        decision = _decision(fused=True)
        adapter = materialize_swizzle_decision(decision)

        self.assertEqual(adapter.decision, decision)
        self.assertEqual(adapter.candidate.id, decision.selected_candidate_ref)
        self.assertEqual(adapter.source_ir1_id, decision.problem.source_ir1_id)
        self.assertEqual(adapter.fused_op_id, decision.problem.fused_op_id)
        self.assertEqual(adapter.group_ref, decision.problem.group.group_ref)
        self.assertEqual(adapter.buffer_requirements, adapter.candidate.buffer_requirements)
        self.assertEqual(adapter.economic_decision_ref, decision.id)
        self.assertIs(
            adapter.deployment_selection.reason,
            SwizzleDeploymentReason.ECONOMIC_DECISION,
        )
        self.assertEqual(
            tuple(
                action.source_action
                for program in adapter.rank_programs
                for action in program.actions
            ),
            tuple(
                action
                for program in adapter.candidate.rank_programs
                for action in program.actions
            ),
        )
        send = adapter.rank_programs[0].actions[0]
        recv = adapter.rank_programs[1].actions[0]
        self.assertIs(send.fusion_kind, FusionActionKind.SEND)
        self.assertIs(recv.fusion_kind, FusionActionKind.RECV)
        self.assertEqual(send.expected_route, (10, 11))
        self.assertEqual(recv.expected_route, (10, 11))
        adapter.validate()

    def test_tampered_route_and_unfused_selection_fail_closed(self) -> None:
        adapter = materialize_swizzle_decision(_decision(fused=True))
        first_program = adapter.rank_programs[0]
        forged = replace(
            adapter,
            rank_programs=(
                replace(
                    first_program,
                    actions=(
                        replace(first_program.actions[0], expected_route=(10, 99)),
                    ),
                ),
                adapter.rank_programs[1],
            ),
        )
        with self.assertRaisesRegex(SchemaError, "frozen IR-1 route"):
            forged.validate()
        with self.assertRaisesRegex(SchemaError, "unfused pipeline"):
            materialize_swizzle_decision(_decision(fused=False))

    def test_forced_deployment_preserves_unfused_economic_decision(self) -> None:
        economic = _decision(fused=False)
        adapter = force_swizzle_deployment(economic)

        self.assertEqual(adapter.decision, economic)
        self.assertEqual(adapter.economic_decision_ref, economic.id)
        self.assertEqual(
            economic.selected_candidate_ref,
            economic.baseline.id,
        )
        self.assertIs(
            adapter.deployment_selection.reason,
            SwizzleDeploymentReason.FORCED_BY_POLICY,
        )
        self.assertIsNot(adapter.candidate.algorithm, SwizzleAlgorithm.UNFUSED)
        adapter.validate()

    def test_forced_deployment_rejects_unfused_or_economic_candidate(self) -> None:
        economic = _decision(fused=False)
        with self.assertRaisesRegex(SchemaError, "deployment candidate must be fused"):
            force_swizzle_deployment(
                economic,
                candidate_ref=economic.selected_candidate_ref,
            )
        already_fused = _decision(fused=True)
        with self.assertRaisesRegex(SchemaError, "FORCED_BY_POLICY"):
            force_swizzle_deployment(
                already_fused,
                candidate_ref=already_fused.selected_candidate_ref,
            )

    def test_shared_schema_delta_is_explicit_and_minimal(self) -> None:
        self.assertEqual(
            FUSION_PLAN_EXTENSION_FIELDS,
            (
                "pattern",
                "algorithm",
                "decision_ref",
                "candidate_ref",
                "swizzle_rank_programs",
                "buffer_requirements",
            ),
        )
        self.assertEqual(
            FUSION_ACTION_EXTENSION_FIELDS,
            ("phase", "source_action_ref", "flops"),
        )

    def test_temporary_producers_are_scoped_by_chunk(self) -> None:
        def producer(chunk: int, flops: int) -> SwizzleActionWitness:
            return SwizzleActionWitness.create(
                rank=0,
                kind=SwizzleActionKind.COMP,
                deps=(),
                chunk_index=chunk,
                phase=SwizzlePhase.STEADY,
                peer_rank=None,
                route_ref=None,
                input_refs=(),
                output_refs=("loop_buffer",),
                logical_bytes=0,
                flops=flops,
            )

        first = producer(0, 2)
        reused = producer(1, 4)
        self.assertEqual(
            _index_temporary_producers((first, reused)),
            {
                ("loop_buffer", 0): first.id,
                ("loop_buffer", 1): reused.id,
            },
        )
        duplicate = producer(0, 4)
        with self.assertRaisesRegex(SchemaError, "multiple data producers in one chunk"):
            _index_temporary_producers((first, duplicate))


if __name__ == "__main__":
    unittest.main()
