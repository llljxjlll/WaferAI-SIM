from __future__ import annotations

from dataclasses import replace
import unittest

from llm.frontend.wafer_frontend.schema.flexible_mesh_runtime import (
    FLEXIBLE_MESH_RUNTIME_MARKER_SCHEMA_VERSION,
    FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION,
    FlexibleMeshArtifactCapacityEvidence,
    FlexibleMeshRuntimeCase,
    FlexibleMeshRuntimeEvidence,
    FlexibleMeshRuntimeMarker,
    FlexibleMeshRuntimeResidual,
    FlexibleMeshRuntimeStage,
    FlexibleMeshRuntimeStageStatus,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_workload import (
    FlexibleMeshSliceOperation,
    FlexibleMeshWorkloadSpec,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec


_DIGEST = "0" * 64


def _case(repeat_count: int = 1) -> FlexibleMeshRuntimeCase:
    return FlexibleMeshRuntimeCase.create(
        FlexibleMeshWorkloadSpec.dense_infer(RectMeshSpec(1, 1)),
        FlexibleMeshSliceOperation.AG_GEMM,
        repeat_count=repeat_count,
    )


def _marker(case: FlexibleMeshRuntimeCase) -> FlexibleMeshRuntimeMarker:
    return FlexibleMeshRuntimeMarker(
        schema_version=FLEXIBLE_MESH_RUNTIME_MARKER_SCHEMA_VERSION,
        mesh_digest=case.workload.mesh.digest,
        workload_digest=case.workload.digest,
        manifest_digest=_DIGEST,
        program_io_digest=_DIGEST,
        makespan_cycles=1,
        rank_coverage=(0,),
        core_coverage=(0,),
        active_routes=(),
        state_completion=(),
        residual=FlexibleMeshRuntimeResidual(0, 0, 0, 0, 0, 0, 0),
        marker_digest=_DIGEST,
    )


def _semantic(case: FlexibleMeshRuntimeCase) -> dict[str, object]:
    marker = _marker(case)
    return {
        "case": case,
        "compilation_id": "compilation",
        "linked_source_ref": "linked",
        "manifest_id": "manifest",
        "manifest_digest": _DIGEST,
        "artifact_sha256": _DIGEST,
        "program_io_digest": _DIGEST,
        "resolver_digest": _DIGEST,
        "npusim_exit_code": 0,
        "capacity": FlexibleMeshArtifactCapacityEvidence(1, 0, 0, 1, 1),
        "markers": (marker,) * case.repeat_count,
        "stages": tuple(
            (
                stage,
                (
                    FlexibleMeshRuntimeStageStatus.VERIFIED
                    if stage is not FlexibleMeshRuntimeStage.REPEATABILITY
                    or case.repeat_count == 2
                    else FlexibleMeshRuntimeStageStatus.NOT_MEASURED
                ),
            )
            for stage in FlexibleMeshRuntimeStage
        ),
        "timing_execution": True,
        "functional_execution": False,
    }


class FlexibleMeshRuntimeEvidenceTest(unittest.TestCase):
    def test_verified_properties_require_the_entire_validated_payload(self) -> None:
        evidence = FlexibleMeshRuntimeEvidence.create(**_semantic(_case()))
        self.assertTrue(evidence.runtime_verified)
        self.assertFalse(evidence.repeatability_verified)

        self.assertFalse(replace(evidence, id="forged").runtime_verified)
        self.assertFalse(replace(evidence, stages=evidence.stages[-1:]).runtime_verified)
        invalid_marker = replace(evidence.markers[0], rank_coverage=())
        self.assertFalse(
            replace(evidence, markers=(invalid_marker,)).runtime_verified
        )

    def test_repeatability_cannot_bypass_runtime_validation(self) -> None:
        case = _case(2)
        semantic = _semantic(case)
        forged = FlexibleMeshRuntimeEvidence(
            schema_version=FLEXIBLE_MESH_RUNTIME_SCHEMA_VERSION,
            id="forged",
            **semantic,
        )
        self.assertFalse(forged.runtime_verified)
        self.assertFalse(forged.repeatability_verified)


if __name__ == "__main__":
    unittest.main()
