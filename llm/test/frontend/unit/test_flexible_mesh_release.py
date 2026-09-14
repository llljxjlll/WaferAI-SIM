from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FLEXIBLE_MESH_RELEASE_CASE_COUNT,
    FLEXIBLE_MESH_RELEASE_EXECUTION_COUNT,
    FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES,
    FLEXIBLE_MESH_RELEASE_SHAPE_COUNT,
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
    tools = tuple(
        FlexibleMeshReleaseTool(
            kind=kind,
            binary_path=f"/release/bin/{kind.value}",
            version="release-v1",
            sha256=_sha(f"tool:{kind.value}"),
            allowlisted_sha256=(_sha(f"tool:{kind.value}"),),
        )
        for kind in FlexibleMeshReleaseToolKind
    )
    return FlexibleMeshReleaseBinding.create(
        runtime_profile_version="timing-v1",
        environment_profile_version="env-v1",
        tools=tools,
        hardware_config_sha256=_sha("hardware"),
        simulation_config_sha256=_sha("simulation"),
        mapping_config_sha256=_sha("mapping"),
    )


def _cases():
    return generate_flexible_mesh_release_cases(
        trace_model_digests=tuple(
            (family, _sha(f"trace:{family.value}"))
            for family in FlexibleMeshReleaseFamily
        ),
        runtime_profile_version="timing-v1",
    )


def _execution(case, binding, index: int):
    rank_count = case.mesh.rank_count
    shared = case.id
    return FlexibleMeshReleaseExecutionEvidence.create(
        case=case,
        binding=binding,
        execution_index=index,
        run=FlexibleMeshIndependentRun(
            materialization_id=f"{case.id}:materialize:{index}",
            finalizer_run_id=f"{case.id}:finalizer:{index}",
            resolver_run_id=f"{case.id}:resolver:{index}",
            npusim_run_id=f"{case.id}:npusim:{index}",
        ),
        spec_digest=_sha(f"{shared}:spec"),
        plan_digest=_sha(f"{shared}:plan"),
        manifest_digest=_sha(f"{shared}:manifest"),
        hardware_config_sha256=_sha(f"{shared}:hardware"),
        simulation_config_sha256=binding.simulation_config_sha256,
        mapping_config_sha256=binding.mapping_config_sha256,
        artifact_sha256=_sha(f"{shared}:artifact"),
        artifact_file_bytes=64,
        program_io_artifact_sha256=_sha(f"{shared}:artifact"),
        program_io_digest=_sha(f"{shared}:program-io"),
        resolver_digest=_sha(f"{shared}:resolver"),
        makespan_cycles=123,
        marker_digest=_sha(f"{shared}:marker"),
        capacity=FlexibleMeshReleaseCapacityEvidence(
            rank_count=rank_count,
            peak_sessions_per_core_per_wave=3,
            symbolic_record_count=2,
            exact_record_count=1,
            linked_manifest_file_bytes=128,
            artifact_file_bytes=64,
            max_runtime_core_id=rank_count - 1,
            transport_tag_count=0,
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


class FlexibleMeshReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = _binding()
        cls.cases = _cases()

    def test_generator_is_exact_six_by_one_hundred(self) -> None:
        self.assertEqual(FLEXIBLE_MESH_RELEASE_SHAPE_COUNT, 100)
        self.assertEqual(FLEXIBLE_MESH_RELEASE_CASE_COUNT, 600)
        self.assertEqual(FLEXIBLE_MESH_RELEASE_EXECUTION_COUNT, 1_200)
        self.assertEqual(len(self.cases), 600)
        self.assertEqual(len({case.id for case in self.cases}), 600)
        for family in FlexibleMeshReleaseFamily:
            meshes = {
                (case.mesh.rows, case.mesh.columns)
                for case in self.cases
                if case.family is family
            }
            self.assertEqual(
                meshes,
                {(rows, columns) for rows in range(1, 11) for columns in range(1, 11)},
            )

    def test_two_independent_executions_are_runtime_and_repeat_verified(self) -> None:
        case = self.cases[0]
        evidence = FlexibleMeshReleaseCaseEvidence.create(
            case=case,
            binding=self.binding,
            executions=(
                _execution(case, self.binding, 0),
                _execution(case, self.binding, 1),
            ),
        )
        self.assertTrue(evidence.runtime_verified)
        self.assertTrue(evidence.repeatability_verified)

        drifted = replace(
            evidence,
            executions=(
                evidence.executions[0],
                replace(evidence.executions[1], makespan_cycles=124),
            ),
        )
        self.assertFalse(drifted.runtime_verified)
        self.assertFalse(drifted.repeatability_verified)

    def test_tool_capacity_and_residual_drift_fail_closed(self) -> None:
        bad_tool = replace(
            self.binding.tools[0],
            sha256=_sha("unapproved-finalizer"),
        )
        with self.assertRaises(SchemaError):
            replace(self.binding, tools=(bad_tool,) + self.binding.tools[1:]).validate()

        case = self.cases[0]
        capacity = FlexibleMeshReleaseCapacityEvidence(
            rank_count=1,
            peak_sessions_per_core_per_wave=4,
            symbolic_record_count=1,
            exact_record_count=1,
            linked_manifest_file_bytes=1,
            artifact_file_bytes=1,
            max_runtime_core_id=0,
            transport_tag_count=0,
        )
        with self.assertRaises(SchemaError):
            capacity.validate(case)
        replace(
            capacity,
            peak_sessions_per_core_per_wave=3,
            linked_manifest_file_bytes=(
                FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES
            ),
        ).validate(case)
        with self.assertRaises(SchemaError):
            replace(
                capacity,
                peak_sessions_per_core_per_wave=3,
                linked_manifest_file_bytes=(
                    FLEXIBLE_MESH_RELEASE_MAX_LINKED_MANIFEST_BYTES + 1
                ),
            ).validate(case)
        with self.assertRaises(SchemaError):
            FlexibleMeshReleaseResidual(active_sessions=1).validate()


if __name__ == "__main__":
    unittest.main()
