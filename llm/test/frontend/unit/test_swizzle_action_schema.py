from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.policies.swizzle.semantics import analyze_ag_gemm
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle import (
    SWIZZLE_ACTION_SCHEMA_VERSION,
    SwizzleActionKind,
    SwizzleActionWitness,
    SwizzleAlgorithm,
    SwizzleBufferRequirement,
    SwizzleCandidate,
    SwizzleFeasibilityCheck,
    SwizzleFeasibilityWitness,
    SwizzlePhase,
    SwizzleRankProgramWitness,
    SwizzleTopologyKind,
    SwizzleTopologyWitness,
)

from test_swizzle_schema import SwizzleSchemaTest, _ag_case


class SwizzleActionSchemaTest(unittest.TestCase):
    def _candidate(self) -> SwizzleCandidate:
        problem, _ = SwizzleSchemaTest()._problem()
        gemm, collective, boundaries = _ag_case(problem.gemm.lhs.axis_roles[-1])
        witness = analyze_ag_gemm(
            gemm,
            collective,
            boundary_input_refs=boundaries,
            boundary_output_refs=(gemm.output.value_ref,),
        )
        # Use the witness embedded in the problem descriptors, not merely a
        # structurally similar synthetic witness.
        witness = analyze_ag_gemm(
            problem.gemm,
            problem.collective,
            boundary_input_refs=witness.boundary_input_refs,
            boundary_output_refs=witness.boundary_output_refs,
        )
        send = SwizzleActionWitness.create(
            rank=0,
            kind=SwizzleActionKind.SEND,
            deps=(),
            chunk_index=0,
            phase=SwizzlePhase.PROLOGUE,
            peer_rank=1,
            route_ref="route.0.1",
            input_refs=(problem.collective.input.value_ref,),
            output_refs=(),
            logical_bytes=32,
            flops=0,
        )
        recv = SwizzleActionWitness.create(
            rank=1,
            kind=SwizzleActionKind.RECV,
            deps=(send.id,),
            chunk_index=0,
            phase=SwizzlePhase.PROLOGUE,
            peer_rank=0,
            route_ref="route.0.1",
            input_refs=(),
            output_refs=(problem.collective.output.value_ref,),
            logical_bytes=32,
            flops=0,
        )
        cost = SwizzleSchemaTest()._cost()
        return SwizzleCandidate.create(
            problem_ref=problem.id,
            pattern=FusionPattern.AG_GEMM,
            algorithm=SwizzleAlgorithm.WANG_1D_BIDIRECTIONAL,
            split_axis=witness.split_axis,
            chunk_count=1,
            unroll_degree=1,
            rank_programs=(
                SwizzleRankProgramWitness(0, (send,)),
                SwizzleRankProgramWitness(1, (recv,)),
            ),
            buffer_requirements=(
                SwizzleBufferRequirement(0, "send_buffer", 32, False, (send.id,)),
                SwizzleBufferRequirement(1, "recv_buffer", 32, False, (recv.id,)),
            ),
            topology_witness=SwizzleTopologyWitness(
                kind=SwizzleTopologyKind.BIDIRECTIONAL_LINE,
                rank_order=(0, 1),
                row_orders=((0, 1),),
                column_orders=((0,), (1,)),
                route_refs=("route.0.1",),
                is_complete_rectangle=True,
                has_hamiltonian_cycle=False,
            ),
            semantic_witness=witness,
            feasibility_witness=SwizzleFeasibilityWitness(
                (SwizzleFeasibilityCheck("semantic", True, "closed"),)
            ),
            cost=cost,
        )

    def test_action_candidate_strict_roundtrip_and_stable_identity(self) -> None:
        candidate = self._candidate()
        self.assertEqual(
            loads_dataclass(
                SwizzleCandidate,
                canonical_json(candidate),
                path="candidate",
            ),
            candidate,
        )
        send = candidate.rank_programs[0].actions[0]
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                send,
                schema_version=SWIZZLE_ACTION_SCHEMA_VERSION.replace("v1alpha1", "v0"),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(send, id="forged").validate()
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(candidate, cost=replace(candidate.cost, logical_bytes=65)).validate()

    def test_action_dag_route_and_buffer_lifetime_are_closed(self) -> None:
        candidate = self._candidate()
        recv = candidate.rank_programs[1].actions[0]
        bad_recv = replace(recv, deps=("missing",))
        with self.assertRaisesRegex(SchemaError, "unstable artifact id"):
            replace(
                candidate,
                rank_programs=(
                    candidate.rank_programs[0],
                    SwizzleRankProgramWitness(1, (bad_recv,)),
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "outside topology witness"):
            # Recreate the action so its own stable identity is valid; the
            # candidate must then reject the unknown route provenance.
            bad_route = SwizzleActionWitness.create(
                rank=1,
                kind=SwizzleActionKind.RECV,
                deps=(candidate.rank_programs[0].actions[0].id,),
                chunk_index=0,
                phase=SwizzlePhase.PROLOGUE,
                peer_rank=0,
                route_ref="route.unknown",
                input_refs=(),
                output_refs=recv.output_refs,
                logical_bytes=recv.logical_bytes,
                flops=0,
            )
            replace(
                candidate,
                rank_programs=(
                    candidate.rank_programs[0],
                    SwizzleRankProgramWitness(1, (bad_route,)),
                ),
                buffer_requirements=(
                    candidate.buffer_requirements[0],
                    replace(candidate.buffer_requirements[1], lifetime_action_refs=(bad_route.id,)),
                ),
            ).validate()
        with self.assertRaisesRegex(SchemaError, "unknown action"):
            replace(
                candidate,
                buffer_requirements=(
                    replace(candidate.buffer_requirements[0], lifetime_action_refs=("missing",)),
                    candidate.buffer_requirements[1],
                ),
            ).validate()


if __name__ == "__main__":
    unittest.main()
