from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.project_swizzle_ir2 import (
    project_swizzle_adapter,
    validate_swizzle_projection_against_adapter,
)
from llm.frontend.wafer_frontend.policies.swizzle.cost import build_unfused_baseline
from llm.frontend.wafer_frontend.policies.swizzle.enumerate import materialize_drafts
from llm.frontend.wafer_frontend.policies.swizzle.materialize import (
    materialize_swizzle_decision,
)
from llm.frontend.wafer_frontend.policies.swizzle.meshslice_2d import (
    generate_meshslice_2d_drafts,
)
from llm.frontend.wafer_frontend.policies.swizzle.wang_1d import (
    generate_wang_1d_drafts,
)
from llm.frontend.wafer_frontend.schema.ir0 import CollectiveKind
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.frontend.wafer_frontend.schema.swizzle import (
    SwizzleDecision,
    SwizzleDecisionReason,
)
from llm.frontend.wafer_frontend.schema.swizzle_ir2 import (
    SWIZZLE_IR2_SCHEMA_VERSION,
    SwizzleIr2ArStage,
    SwizzleIr2BufferRole,
    SwizzleIr2ConsumerContract,
    SwizzleIr2Projection,
    SwizzleIr2ValueOriginKind,
)

import test_swizzle_materialize as materialize_fixture
import test_swizzle_meshslice_cost as meshslice_fixture
import test_swizzle_wang_1d as wang_fixture


def _forced_adapter(problem, witness, candidate):
    baseline = build_unfused_baseline(problem, witness)
    decision = SwizzleDecision.create(
        problem=problem,
        baseline=baseline,
        ranked_candidates=(candidate, baseline),
        selected_candidate_ref=candidate.id,
        decision_reason=SwizzleDecisionReason.LOWEST_ESTIMATED_CYCLES,
    )
    return materialize_swizzle_decision(decision)


def _restable_projection(projection, **updates):
    semantic = projection._semantic_key()
    semantic.update(updates)
    return SwizzleIr2Projection.create(**semantic)


class SwizzleIr2ProjectionTest(unittest.TestCase):
    def test_w7_adapter_projects_one_task_per_action_and_exact_flow(self) -> None:
        adapter = materialize_swizzle_decision(
            materialize_fixture._decision(fused=True)
        )
        projection = project_swizzle_adapter(adapter)

        self.assertEqual(projection.source_adapter_ref, adapter.id)
        self.assertEqual(projection.source_decision_ref, adapter.decision.id)
        self.assertEqual(projection.source_candidate_ref, adapter.candidate.id)
        self.assertEqual(projection.split_axis, adapter.candidate.split_axis)
        self.assertEqual(projection.chunk_count, adapter.candidate.chunk_count)
        self.assertEqual(projection.unroll_degree, adapter.candidate.unroll_degree)
        self.assertEqual(projection.output_ownership[0].terminal_task_refs, ())
        self.assertEqual(
            len(projection.output_ownership[1].terminal_task_refs),
            1,
        )
        self.assertEqual(
            sum(len(dag.tasks) for dag in projection.rank_dags),
            sum(len(program.actions) for program in adapter.rank_programs),
        )
        self.assertEqual(len(projection.flows), 1)
        flow = projection.flows[0]
        self.assertEqual(flow.die_path, (10, 11))
        self.assertEqual((flow.source_rank, flow.destination_rank), (0, 1))
        self.assertTrue(
            any(
                value.origin_kind is SwizzleIr2ValueOriginKind.BOUNDARY_INPUT
                for value in projection.rank_dags[0].values
            )
        )
        self.assertTrue(
            any(
                value.origin_kind is SwizzleIr2ValueOriginKind.RECEIVED_PAYLOAD
                for value in projection.rank_dags[1].values
            )
        )
        validate_swizzle_projection_against_adapter(projection, adapter)

    def test_meshslice_retains_loop_accumulator_and_double_buffer_slots(self) -> None:
        problem, witness = meshslice_fixture._problem()
        draft = next(
            item
            for item in generate_meshslice_2d_drafts(problem, witness)
            if item.chunk_count == 2
        )
        candidate = materialize_drafts(problem, (draft,))[0]
        projection = project_swizzle_adapter(
            _forced_adapter(problem, witness, candidate)
        )

        roles = {
            buffer.role for dag in projection.rank_dags for buffer in dag.buffers
        }
        self.assertIn(SwizzleIr2BufferRole.DOUBLE_BUFFER, roles)
        self.assertIn(SwizzleIr2BufferRole.LOOP_ACCUMULATOR, roles)
        loop_values = tuple(
            value
            for dag in projection.rank_dags
            for value in dag.values
            if value.loop_carried
        )
        self.assertTrue(loop_values)
        self.assertTrue(
            all(value.buffer_ref is not None for value in loop_values)
        )
        double_uses = tuple(
            use
            for dag in projection.rank_dags
            for task in dag.tasks
            for use in task.buffer_uses
            if next(
                buffer
                for buffer in dag.buffers
                if buffer.buffer_ref == use.buffer_ref
            ).slot_count
            == 2
        )
        self.assertEqual({use.slot for use in double_uses}, {0, 1})

    def test_all_reduce_retains_two_stages_and_identity_replicated_outputs(self) -> None:
        problem, witness = wang_fixture._post_problem(CollectiveKind.ALL_REDUCE)
        draft = next(
            item
            for item in generate_wang_1d_drafts(problem, witness)
            if item.unroll_degree == 1
        )
        candidate = materialize_drafts(problem, (draft,))[0]
        projection = project_swizzle_adapter(
            _forced_adapter(problem, witness, candidate)
        )

        stages = {
            task.ar_stage for dag in projection.rank_dags for task in dag.tasks
        }
        self.assertIn(SwizzleIr2ArStage.REDUCTION, stages)
        self.assertIn(SwizzleIr2ArStage.REPLICATION, stages)
        self.assertTrue(all(item.replicated for item in projection.output_ownership))
        self.assertTrue(
            all(
                item.rank == item.logical_owner_rank == item.physical_owner_rank
                for item in projection.output_ownership
            )
        )
        self.assertTrue(
            any(
                buffer.role in (
                    SwizzleIr2BufferRole.LOOP_ACCUMULATOR,
                    SwizzleIr2BufferRole.REDUCTION,
                )
                for dag in projection.rank_dags
                for buffer in dag.buffers
            )
        )

    def test_isolated_carrier_roundtrip_and_exact_downstream_gate(self) -> None:
        adapter = materialize_swizzle_decision(
            materialize_fixture._decision(fused=True)
        )
        projection = project_swizzle_adapter(adapter)
        self.assertEqual(
            loads_dataclass(
                SwizzleIr2Projection,
                canonical_json(projection),
                path="projection",
            ),
            projection,
        )
        projection.require_consumer(
            SwizzleIr2ConsumerContract.STRICT_SWIZZLE_TIMING_V1
        )
        with self.assertRaisesRegex(SchemaError, "strict_swizzle_timing_v1"):
            projection.require_consumer(
                SwizzleIr2ConsumerContract.CURRENT_NAIVE_IR2
            )
        with self.assertRaisesRegex(SchemaError, "unsupported schema version"):
            replace(
                projection,
                schema_version=SWIZZLE_IR2_SCHEMA_VERSION.replace("v1alpha1", "v0"),
            ).validate()

    def test_tampered_temp_origin_and_output_owner_fail_closed(self) -> None:
        projection = project_swizzle_adapter(
            materialize_swizzle_decision(materialize_fixture._decision(fused=True))
        )
        value = projection.rank_dags[1].values[0]
        with self.assertRaisesRegex(SchemaError, "requires a producer"):
            replace(
                value,
                origin_kind=SwizzleIr2ValueOriginKind.RECEIVED_PAYLOAD,
                producer_task_refs=(),
            ).validate()
        ownership = projection.output_ownership[0]
        with self.assertRaisesRegex(SchemaError, "identity output ownership"):
            replace(ownership, physical_owner_rank=1).validate()

    def test_restable_provenance_and_incidence_tamper_fail_closed(self) -> None:
        adapter = materialize_swizzle_decision(
            materialize_fixture._decision(fused=True)
        )
        projection = project_swizzle_adapter(adapter)
        forged = _restable_projection(
            projection,
            source_decision_ref="forged_decision",
        )
        with self.assertRaisesRegex(SchemaError, "provenance disagrees"):
            validate_swizzle_projection_against_adapter(forged, adapter)
        with self.assertRaisesRegex(SchemaError, "SEND/RECV"):
            _restable_projection(projection, flows=())

        dag = next(
            item for item in projection.rank_dags
            if any(value.consumer_task_refs for value in item.values)
        )
        value = next(item for item in dag.values if item.consumer_task_refs)
        forged_value_semantic = value._semantic_key()
        forged_value_semantic["consumer_task_refs"] = ()
        forged_value = type(value).create(**forged_value_semantic)
        forged_dag = replace(
            dag,
            values=tuple(
                forged_value if item.id == value.id else item
                for item in dag.values
            ),
        )
        with self.assertRaisesRegex(SchemaError, "incidence is not exact"):
            _restable_projection(
                projection,
                rank_dags=tuple(
                    forged_dag if item.rank == dag.rank else item
                    for item in projection.rank_dags
                ),
            )

        forged_owner = replace(
            projection.output_ownership[0],
            terminal_task_refs=(projection.rank_dags[0].tasks[0].id,),
        )
        with self.assertRaisesRegex(SchemaError, "globally exact"):
            _restable_projection(
                projection,
                output_ownership=(forged_owner, projection.output_ownership[1]),
            )


if __name__ == "__main__":
    unittest.main()
