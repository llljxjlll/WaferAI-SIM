from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_representative_completion import (
    FLEXIBLE_MESH_REPRESENTATIVE_CASE_COUNT,
    FLEXIBLE_MESH_REPRESENTATIVE_SHAPES,
    FlexibleMeshRepresentativeCompletion,
    FlexibleMeshRepresentativeEvidenceMatrix,
    FlexibleMeshValidationScope,
    derive_flexible_mesh_representative_completion,
    derive_flexible_mesh_representative_contract_evidence,
    representative_tested_meshes,
    select_flexible_mesh_representative_cases,
    validate_flexible_mesh_representative_cases,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseFamily,
    generate_flexible_mesh_release_cases,
    validate_flexible_mesh_release_cases,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_json, loads_dataclass
from llm.test.frontend.unit.test_flexible_mesh_completion import (
    _binding,
    _execution,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _representative_matrix() -> FlexibleMeshRepresentativeEvidenceMatrix:
    binding = _binding()
    all_cases = generate_flexible_mesh_release_cases(
        trace_model_digests=tuple(
            (family, _sha(f"representative:{family.value}"))
            for family in FlexibleMeshReleaseFamily
        ),
        runtime_profile_version=binding.runtime_profile_version,
    )
    cases = select_flexible_mesh_representative_cases(all_cases)
    evidence = tuple(
        FlexibleMeshReleaseCaseEvidence.create(
            case=case,
            binding=binding,
            executions=(
                _execution(case, binding, 0),
                _execution(case, binding, 1),
            ),
        )
        for case in cases
    )
    contracts = derive_flexible_mesh_representative_contract_evidence(
        binding=binding,
        cases=cases,
        case_evidence=evidence,
    )
    return FlexibleMeshRepresentativeEvidenceMatrix.create(
        release_binding=binding,
        release_cases=cases,
        case_evidence=evidence,
        contract_evidence=contracts,
    )


class FlexibleMeshRepresentativeCompletionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = _representative_matrix()
        cls.completion = derive_flexible_mesh_representative_completion(cls.matrix)

    def test_exact_scope_and_all_target_flags_are_derived(self) -> None:
        self.assertEqual(len(self.matrix.release_cases), FLEXIBLE_MESH_REPRESENTATIVE_CASE_COUNT)
        self.assertEqual(
            tuple(
                (case.mesh.rows, case.mesh.columns)
                for case in self.matrix.release_cases[:8]
            ),
            FLEXIBLE_MESH_REPRESENTATIVE_SHAPES,
        )
        self.assertIs(
            self.completion.validation_scope,
            FlexibleMeshValidationScope.REPRESENTATIVE,
        )
        self.assertFalse(self.completion.exhaustive_runtime)
        self.assertEqual(self.completion.tested_meshes, representative_tested_meshes())
        self.assertTrue(self.completion.dense_train_rect_complete)
        self.assertTrue(self.completion.moe_infer_rect_complete)
        self.assertTrue(self.completion.moe_train_rect_complete)
        self.assertTrue(self.completion.meshslice_all_rect_complete)
        self.assertTrue(self.completion.workload_contract_complete)
        self.assertTrue(self.completion.flexible_mesh_workloads_complete)
        self.assertFalse(self.completion.exhaustive_rect_runtime_complete)

        field_names = {field.name for field in fields(FlexibleMeshRepresentativeCompletion)}
        self.assertTrue({
            "dense_train_rect_complete",
            "moe_infer_rect_complete",
            "moe_train_rect_complete",
            "meshslice_all_rect_complete",
            "workload_contract_complete",
            "flexible_mesh_workloads_complete",
            "exhaustive_rect_runtime_complete",
        }.isdisjoint(field_names))

    def test_serialized_report_states_scope_without_claiming_exhaustive(self) -> None:
        serialized = canonical_json(self.completion)
        self.assertIn('"validation_scope":"representative"', serialized)
        self.assertIn('"exhaustive_runtime":false', serialized)
        self.assertIn('"tested_meshes"', serialized)
        self.assertNotIn('"exhaustive_rect_runtime_complete":true', serialized)
        round_tripped = loads_dataclass(
            FlexibleMeshRepresentativeCompletion,
            serialized,
            path="representative_completion",
        )
        round_tripped.validate()
        self.assertEqual(round_tripped, self.completion)

    def test_missing_duplicate_extra_or_reordered_shape_fails_closed(self) -> None:
        cases = self.matrix.release_cases
        profile = self.matrix.release_binding.runtime_profile_version
        invalid = (
            cases[:-1],
            cases[:-1] + (cases[0],),
            (cases[1], cases[0]) + cases[2:],
        )
        for candidate in invalid:
            with self.subTest(length=len(candidate)):
                with self.assertRaises(SchemaError):
                    validate_flexible_mesh_representative_cases(candidate, profile)

        all_cases = generate_flexible_mesh_release_cases(
            trace_model_digests=tuple(
                (family, _sha(f"representative:{family.value}"))
                for family in FlexibleMeshReleaseFamily
            ),
            runtime_profile_version=profile,
        )
        extra = cases + (all_cases[1],)
        with self.assertRaises(SchemaError):
            validate_flexible_mesh_representative_cases(extra, profile)

    def test_forged_scope_meshes_exhaustive_and_id_fail_closed(self) -> None:
        forged = (
            replace(self.completion, id="forged"),
            replace(self.completion, validation_scope="representative"),
            replace(self.completion, exhaustive_runtime=True),
            replace(
                self.completion,
                tested_meshes=self.completion.tested_meshes[:-1],
            ),
        )
        for candidate in forged:
            with self.subTest(candidate=candidate.id):
                self.assertFalse(candidate.flexible_mesh_workloads_complete)
                self.assertFalse(candidate.dense_train_rect_complete)
                with self.assertRaises(SchemaError):
                    candidate.validate()

    def test_missing_execution_or_repeat_drift_fails_closed(self) -> None:
        first = self.matrix.case_evidence[0]
        invalid_rows = (
            replace(first, executions=(first.executions[0],)),
            replace(
                first,
                executions=(
                    first.executions[0],
                    replace(first.executions[1], marker_digest=_sha("repeat-drift")),
                ),
            ),
        )
        for invalid in invalid_rows:
            forged_matrix = replace(
                self.matrix,
                case_evidence=(invalid,) + self.matrix.case_evidence[1:],
            )
            forged_completion = replace(
                self.completion,
                evidence_matrix=forged_matrix,
            )
            self.assertFalse(forged_completion.flexible_mesh_workloads_complete)
            with self.assertRaises(SchemaError):
                forged_matrix.validate()

    def test_strict_600_validator_remains_strict(self) -> None:
        with self.assertRaisesRegex(SchemaError, "exactly 600"):
            validate_flexible_mesh_release_cases(
                self.matrix.release_cases,
                self.matrix.release_binding.runtime_profile_version,
            )


if __name__ == "__main__":
    unittest.main()
