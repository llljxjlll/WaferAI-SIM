from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from llm.test.frontend.integration.lite_train_rooted_ar_cases import (
    S2_LITE_DP2_ROOTED_AR_CASE_ID,
    build_s2_lite_dp2_rooted_ar_case,
)

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind
from llm.frontend.wafer_frontend.schema.lite_train_dp2 import RootedArStepKind


class S2LiteDp2RootedArCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_s2_lite_dp2_rooted_ar_case()

    def test_self_contained_exact_chain_and_counts(self) -> None:
        case = self.case
        self.assertEqual(
            case.case_id,
            "case.s2_lite.dp2_tp1.rooted_allreduce",
        )
        self.assertEqual(case.case_id, S2_LITE_DP2_ROOTED_AR_CASE_ID)
        self.assertEqual(
            (
                len(case.rooted_source.graph.nodes),
                len(case.rooted_source.graph.values),
                len(case.rooted_source.graph.edges),
                len(case.rooted_source.graph.persistent_states),
                len(case.rooted_source.graph.state_accesses),
                len(case.placed.replicas),
            ),
            (29, 47, 34, 15, 16, 2),
        )
        self.assertEqual(
            tuple(
                replica.graph.groups[0].placements[0].die_id
                for replica in case.placed.replicas
            ),
            (0, 1),
        )

        tasks = tuple(
            task
            for replica in case.projected.replicas
            for dag in replica.projection.dags
            for task in dag.tasks
        )
        self.assertEqual(
            (
                len(tasks),
                sum(task.kind is SemanticTaskKind.COMP for task in tasks),
                sum(task.kind is SemanticTaskKind.DMA_IN for task in tasks),
                sum(task.kind is SemanticTaskKind.DMA_OUT for task in tasks),
            ),
            (92, 58, 32, 2),
        )
        schedules = tuple(
            schedule
            for replica in case.scheduled.replicas
            for schedule in replica.schedule_set.schedules
        )
        self.assertEqual(
            (
                sum(len(schedule.placements) for schedule in schedules),
                sum(len(schedule.buffer_bindings) for schedule in schedules),
                sum(len(schedule.task_buffer_uses) for schedule in schedules),
                sum(len(schedule.task_state_uses) for schedule in schedules),
            ),
            (92, 96, 198, 34),
        )
        self.assertEqual(
            (
                sum(len(dag.actions) for dag in case.global_action.local_dags),
                tuple(step.kind for step in case.global_action.ar_steps),
                len(case.n6_intent.lowering_contexts),
                len(case.n6_intent.gradient_buffer_abis),
                len(case.n6_intent.scratch_buffer_abis),
                len(case.n6_intent.units),
            ),
            (
                92,
                (
                    RootedArStepKind.UPLOAD,
                    RootedArStepKind.ROOT_REDUCE,
                    RootedArStepKind.DOWNLOAD,
                ),
                2,
                2,
                2,
                8,
            ),
        )

    def test_determinism_and_exact_runtime_inputs(self) -> None:
        rebuilt = build_s2_lite_dp2_rooted_ar_case()
        self.assertEqual(rebuilt, self.case)
        inputs = self.case.source.runtime_inputs
        self.assertEqual(
            hashlib.sha256(inputs.hardware_json.encode("utf-8")).hexdigest(),
            "0696e5805d9b6a94b2652db341a6e3b5ccef387212a55b8e537a0e71a3a7c322",
        )
        self.assertEqual(
            hashlib.sha256(inputs.mapping_text.encode("utf-8")).hexdigest(),
            "99a357b646bc6d0d81ac188c8bfffcbf6ab8f8f72a5d262fe81624f6f9a9a66c",
        )
        self.assertEqual(inputs.source_hardware_path.name, "hardware_2x1.json")
        self.assertEqual(inputs.source_mapping_path.name, "mapping.spec")

    def test_case_and_provenance_tamper_fail_closed(self) -> None:
        with self.subTest("case-id"):
            with self.assertRaisesRegex(ValueError, "case id"):
                replace(self.case, case_id="case.s2_lite.lm_head_train").validate()
        with self.subTest("hardware"):
            bad_inputs = replace(
                self.case.source.runtime_inputs,
                hardware_json="{}",
            )
            with self.assertRaisesRegex(ValueError, "hardware/mapping"):
                replace(
                    self.case,
                    source=replace(self.case.source, runtime_inputs=bad_inputs),
                ).validate()
        with self.subTest("projection-provenance"):
            forged = replace(
                self.case.projected,
                source_planned_carrier_id="train_interdie_planned_ir1_forged",
            )
            with self.assertRaises(SchemaError):
                replace(self.case, projected=forged).validate()


if __name__ == "__main__":
    unittest.main()
