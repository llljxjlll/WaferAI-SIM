from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from llm.test.frontend.integration.lite_train_dp4_cases import (
    build_s2_lite_dp4_tree_ar_case,
)

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.program_io import (
    _resolved_state_abis,
    _validate_dp4_updated_weight_probes,
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.program_io import (
    ProgramHbmTarget,
    ProgramSramTarget,
)
from llm.frontend.wafer_frontend.schema.ir2 import SemanticTaskKind
from llm.frontend.wafer_frontend.schema.lite_train_dp4 import (
    S2_LITE_DP4_TREE_AR_CASE_ID,
)


class S2LiteDp4TreeArCaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = build_s2_lite_dp4_tree_ar_case()

    def test_self_contained_exact_chain_and_counts(self) -> None:
        case = self.case
        self.assertEqual(case.case_id, S2_LITE_DP4_TREE_AR_CASE_ID)
        self.assertEqual(
            (
                len(case.dp4_source.graph.nodes),
                len(case.dp4_source.graph.values),
                len(case.dp4_source.graph.edges),
                len(case.dp4_source.graph.persistent_states),
                len(case.dp4_source.graph.state_accesses),
                len(case.placed.replicas),
            ),
            (29, 47, 34, 15, 16, 4),
        )
        self.assertEqual(
            tuple(replica.graph.groups[0].placements[0].die_id for replica in case.placed.replicas),
            (0, 1, 2, 3),
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
            (184, 116, 64, 4),
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
            (184, 192, 396, 68),
        )
        self.assertEqual(
            (
                sum(len(dag.actions) for dag in case.global_action.local_dags),
                len(case.global_action.tree_flows),
                len(case.global_action.tree_reduces),
                len(case.global_action.sgd_dependencies),
                sum(flow.gradient_bytes for flow in case.global_action.tree_flows),
            ),
            (184, 6, 3, 4, 12288),
        )
        manifest = case.linked.manifest
        self.assertEqual(
            (
                len(manifest.fragments),
                sum(
                    len(stream.records)
                    for fragment in manifest.fragments
                    for stream in fragment.core_streams
                ),
                len(manifest.input_digests),
                len(manifest.core_streams),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
            ),
            (201, 709, 218, 4, 1262, 68, 30, 448),
        )

    def test_determinism_and_exact_inputs(self) -> None:
        rebuilt = build_s2_lite_dp4_tree_ar_case()
        self.assertEqual(rebuilt, self.case)
        inputs = self.case.source.runtime_inputs
        self.assertEqual(
            hashlib.sha256(inputs.hardware_json.encode("utf-8")).hexdigest(),
            "c4f8eadcf84963a7cf2ba337ae1147a72ce920dad9d7bf1c0c5eec415aa372fe",
        )
        self.assertEqual(
            hashlib.sha256(inputs.mapping_text.encode("utf-8")).hexdigest(),
            "99a357b646bc6d0d81ac188c8bfffcbf6ab8f8f72a5d262fe81624f6f9a9a66c",
        )
        self.assertEqual(self.case.hardware_path.name, "hardware_2x2.json")

    def test_actual_sha_program_io_exact_and_fail_closed(self) -> None:
        seeds, expected = build_deterministic_timing_state_overrides(
            self.case.linked
        )
        self.assertEqual((len(seeds), len(expected)), (15, 0))
        artifact_sha = hashlib.sha256(b"dp4-tree-ar-program").hexdigest()
        first = build_timing_program_io(
            self.case.linked,
            artifact_sha,
            state_seed_overrides=seeds,
            state_expected_overrides=expected,
        )
        second = build_timing_program_io(
            self.case.linked,
            artifact_sha,
            state_seed_overrides=seeds,
            state_expected_overrides=expected,
        )
        self.assertEqual(first, second)
        first.validate_against(self.case.linked.manifest)
        self.assertEqual(
            (len(first.blobs), len(first.initializations), len(first.output_probes)),
            (26, 252, 4),
        )
        self.assertTrue(
            all(type(probe.target) is ProgramHbmTarget for probe in first.output_probes)
        )
        self.assertFalse(
            any(type(probe.target) is ProgramSramTarget for probe in first.output_probes)
        )
        self.assertEqual(sum(probe.length_bytes for probe in first.output_probes), 4096)
        resolved_state = _resolved_state_abis(self.case.linked)
        with self.assertRaisesRegex(SchemaError, "exactly cover four"):
            _validate_dp4_updated_weight_probes(
                self.case.linked,
                replace(first, output_probes=first.output_probes[:-1]),
                resolved_state,
            )
        duplicate_target = replace(
            first.output_probes[1].target,
            state_abi_id=first.output_probes[0].target.state_abi_id,
        )
        with self.assertRaisesRegex(SchemaError, "exactly cover four"):
            _validate_dp4_updated_weight_probes(
                self.case.linked,
                replace(
                    first,
                    output_probes=(
                        first.output_probes[0],
                        replace(first.output_probes[1], target=duplicate_target),
                        *first.output_probes[2:],
                    ),
                ),
                resolved_state,
            )
        missing = dict(seeds)
        missing.pop(next(iter(missing)))
        with self.assertRaisesRegex(SchemaError, "exactly cover"):
            build_timing_program_io(
                self.case.linked,
                artifact_sha,
                state_seed_overrides=missing,
            )

    def test_case_and_provenance_tamper_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "case id"):
            replace(self.case, case_id="case.forged").validate()
        bad_inputs = replace(self.case.source.runtime_inputs, hardware_json="{}")
        with self.assertRaisesRegex(ValueError, "hardware/mapping"):
            replace(self.case, source=replace(self.case.source, runtime_inputs=bad_inputs)).validate()
        forged = replace(self.case.projected, source_planned_carrier_id="forged")
        with self.assertRaises(SchemaError):
            replace(self.case, projected=forged).validate()


if __name__ == "__main__":
    unittest.main()
