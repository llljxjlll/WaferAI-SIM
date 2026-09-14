from __future__ import annotations

import hashlib
import unittest
from dataclasses import fields, replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_completion import (
    FlexibleMeshCompletion,
    FlexibleMeshCompletionEvidenceMatrix,
    FlexibleMeshContractCheck,
    FlexibleMeshContractEvidence,
    derive_flexible_mesh_contract_evidence,
    derive_flexible_mesh_completion,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshIndependentRun,
    FlexibleMeshProgramIOPhase,
    FlexibleMeshReleaseBinding,
    FlexibleMeshReleaseCapacityEvidence,
    FlexibleMeshReleaseCaseEvidence,
    FlexibleMeshReleaseExecutionEvidence,
    FlexibleMeshReleaseFamily,
    FlexibleMeshReleaseResidual,
    FlexibleMeshReleaseRuntimeStage,
    FlexibleMeshReleaseTool,
    FlexibleMeshReleaseToolKind,
    expected_completion_markers,
    generate_flexible_mesh_release_cases,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding() -> FlexibleMeshReleaseBinding:
    return FlexibleMeshReleaseBinding.create(
        runtime_profile_version="timing-v1",
        environment_profile_version="env-v1",
        tools=tuple(
            FlexibleMeshReleaseTool(
                kind=kind,
                binary_path=f"/release/bin/{kind.value}",
                version="release-v1",
                sha256=_sha(f"tool:{kind.value}"),
                allowlisted_sha256=(_sha(f"tool:{kind.value}"),),
            )
            for kind in FlexibleMeshReleaseToolKind
        ),
        hardware_config_sha256=_sha("hardware"),
        simulation_config_sha256=_sha("simulation"),
        mapping_config_sha256=_sha("mapping"),
    )


def _execution(case, binding, execution_index: int):
    rank_count = case.mesh.rank_count
    return FlexibleMeshReleaseExecutionEvidence.create(
        case=case,
        binding=binding,
        execution_index=execution_index,
        run=FlexibleMeshIndependentRun(
            materialization_id=f"{case.id}:materialize:{execution_index}",
            finalizer_run_id=f"{case.id}:finalizer:{execution_index}",
            resolver_run_id=f"{case.id}:resolver:{execution_index}",
            npusim_run_id=f"{case.id}:npusim:{execution_index}",
        ),
        spec_digest=_sha(f"{case.id}:spec"),
        plan_digest=_sha(f"{case.id}:plan"),
        manifest_digest=_sha(f"{case.id}:manifest"),
        hardware_config_sha256=_sha(f"{case.id}:hardware"),
        simulation_config_sha256=binding.simulation_config_sha256,
        mapping_config_sha256=binding.mapping_config_sha256,
        artifact_sha256=_sha(f"{case.id}:artifact"),
        artifact_file_bytes=64,
        program_io_artifact_sha256=_sha(f"{case.id}:artifact"),
        program_io_digest=_sha(f"{case.id}:program-io"),
        resolver_digest=_sha(f"{case.id}:resolver"),
        makespan_cycles=1000 + rank_count,
        marker_digest=_sha(f"{case.id}:marker"),
        capacity=FlexibleMeshReleaseCapacityEvidence(
            rank_count=rank_count,
            peak_sessions_per_core_per_wave=3,
            symbolic_record_count=2,
            exact_record_count=1,
            linked_manifest_file_bytes=128,
            artifact_file_bytes=64,
            max_runtime_core_id=rank_count - 1,
            transport_tag_count=rank_count - 1,
        ),
        residual=FlexibleMeshReleaseResidual(),
        rank_coverage=tuple(range(rank_count)),
        core_coverage=tuple(range(rank_count)),
        completion_markers=expected_completion_markers(case.family),
        stage_exit_codes=tuple(
            (stage, 0) for stage in FlexibleMeshReleaseRuntimeStage
        ),
        program_io_phases=tuple(FlexibleMeshProgramIOPhase),
    )


def _complete_matrix() -> FlexibleMeshCompletionEvidenceMatrix:
    binding = _binding()
    cases = generate_flexible_mesh_release_cases(
        trace_model_digests=tuple(
            (family, _sha(f"trace:{family.value}"))
            for family in FlexibleMeshReleaseFamily
        ),
        runtime_profile_version=binding.runtime_profile_version,
    )
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
    contracts = derive_flexible_mesh_contract_evidence(
        binding=binding, cases=cases, case_evidence=evidence
    )
    return FlexibleMeshCompletionEvidenceMatrix.create(
        release_binding=binding,
        release_cases=cases,
        case_evidence=evidence,
        contract_evidence=contracts,
    )


class FlexibleMeshCompletionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = _complete_matrix()
        cls.completion = derive_flexible_mesh_completion(cls.matrix)

    def test_four_target_states_are_derived_from_exact_matrix(self) -> None:
        field_names = {field.name for field in fields(FlexibleMeshCompletion)}
        self.assertTrue(
            {
                "dense_train_rect_complete",
                "moe_infer_rect_complete",
                "moe_train_rect_complete",
                "flexible_mesh_workloads_complete",
            }.isdisjoint(field_names)
        )
        self.assertTrue(self.completion.dense_train_rect_complete)
        self.assertTrue(self.completion.moe_infer_rect_complete)
        self.assertTrue(self.completion.moe_train_rect_complete)
        self.assertTrue(self.completion.meshslice_all_rect_complete)
        self.assertTrue(self.completion.workload_contract_complete)
        self.assertTrue(self.completion.flexible_mesh_workloads_complete)

    def test_forged_report_and_missing_or_duplicate_rows_fail_closed(self) -> None:
        forged = replace(self.completion, id="forged")
        self.assertFalse(forged.dense_train_rect_complete)
        self.assertFalse(forged.flexible_mesh_workloads_complete)
        forged_receipt = FlexibleMeshContractEvidence.create(
            check=FlexibleMeshContractCheck.SCHEMA,
            evidence_digest=_sha("forged-contract-evidence"),
            verifier_digest=_sha("forged-contract-verifier"),
            binding=self.matrix.release_binding,
        )
        with self.assertRaises(SchemaError):
            FlexibleMeshCompletionEvidenceMatrix.create(
                release_binding=self.matrix.release_binding,
                release_cases=self.matrix.release_cases,
                case_evidence=self.matrix.case_evidence,
                contract_evidence=(forged_receipt,)
                + self.matrix.contract_evidence[1:],
            )


        missing = replace(
            self.matrix,
            case_evidence=self.matrix.case_evidence[:-1],
        )
        self.assertFalse(
            replace(self.completion, evidence_matrix=missing).flexible_mesh_workloads_complete
        )
        with self.assertRaises(SchemaError):
            derive_flexible_mesh_completion(missing)

        duplicate = replace(
            self.matrix,
            case_evidence=self.matrix.case_evidence[:-1]
            + (self.matrix.case_evidence[0],),
        )
        self.assertFalse(
            replace(self.completion, evidence_matrix=duplicate).flexible_mesh_workloads_complete
        )

    def test_missing_second_execution_and_repeat_drift_fail_closed(self) -> None:
        first = self.matrix.case_evidence[0]
        single_execution = replace(first, executions=(first.executions[0],))
        single_matrix = replace(
            self.matrix,
            case_evidence=(single_execution,) + self.matrix.case_evidence[1:],
        )
        self.assertFalse(
            replace(self.completion, evidence_matrix=single_matrix).dense_train_rect_complete
        )

        drifted_execution = replace(first.executions[1], marker_digest=_sha("drift"))
        drifted_evidence = replace(
            first,
            executions=(first.executions[0], drifted_execution),
        )
        drifted_matrix = replace(
            self.matrix,
            case_evidence=(drifted_evidence,) + self.matrix.case_evidence[1:],
        )
        self.assertFalse(
            replace(self.completion, evidence_matrix=drifted_matrix).dense_train_rect_complete
        )
        self.assertFalse(
            replace(self.completion, evidence_matrix=drifted_matrix).flexible_mesh_workloads_complete
        )


if __name__ == "__main__":
    unittest.main()
