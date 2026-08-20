"""Production rebuild of the approved S2/S3 Lite timing reports."""

from __future__ import annotations

from dataclasses import dataclass

from llm.frontend.wafer_frontend.passes import link_lite_moe_n6, lower_lite_moe_n6
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema import (
    CapabilityStatus,
    LiteRuntimeCapabilityMatrix,
    S2LiteRuntimeReport,
    S3LiteRuntimeReport,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from lite_moe_cases import build_lite_moe_execution_case
from run_lite_train_runtime import build_production_pre_runtime


_S2_SHA = "e5dcd18e9de62400c2ea0eb706cc2f2e491f685d57712577aa11ab9427be46d9"
_S3_SHA = "c49a8202eac0fbbe89aa1b96ba6cba782136b6e8bdf09a6b5870817a27650361"
_S2_MARKERS = "9b15908e449e9fc6d1e5515fc6c69bd78701082c3ba384fffa6160f74b491fe4"
_S3_MARKERS = "63eb904a18cb52f2ba31ffb4922b501c485127b4eeb5de4263c7268d5a749f75"


@dataclass(frozen=True, slots=True)
class LiteRuntimeEvidenceBundle:
    s2_source: object
    s2_program_io: object
    s2_report: S2LiteRuntimeReport
    s3_source: object
    s3_program_io: object
    s3_report: S3LiteRuntimeReport
    matrix: LiteRuntimeCapabilityMatrix

    def validate(self) -> None:
        self.s2_report.validate_against(self.s2_source, self.s2_program_io)
        self.s3_report.validate_against(self.s3_source, self.s3_program_io)
        self.matrix.validate_against(self.s2_report, self.s3_report)


def build_lite_runtime_evidence() -> LiteRuntimeEvidenceBundle:
    s2 = build_production_pre_runtime(artifact_sha256=_S2_SHA)
    if s2.program_io is None:
        raise AssertionError("actual-SHA S2 ProgramIo was not built")
    s2_manifest = s2.linked.manifest
    s2_report = S2LiteRuntimeReport.create(
        case_id="case.s2_lite.lm_head_train",
        baseline_epoch="s2-lite-v1",
        linked_program_id=s2.linked.id,
        manifest_id=s2_manifest.id,
        manifest_digest=canonical_digest(s2_manifest),
        program_io_id=s2.program_io.id,
        program_io_digest=canonical_digest(s2.program_io),
        artifact_sha256=_S2_SHA,
        artifact_bytes=23616,
        fragment_count=46,
        record_count=169,
        address_relocation_count=324,
        blob_count=23,
        initialization_count=62,
        probe_count=1,
        hbm_read_bytes=13472,
        hbm_write_bytes=1024,
        ack_total=2,
        done_total=1,
        makespan_cycles=6164,
        repeat_count=2,
        runtime_marker_digest=_S2_MARKERS,
        timing_execution=True,
        compute_functional=False,
        model_functional=False,
    )

    case = build_lite_moe_execution_case()
    lowered = lower_lite_moe_n6(
        case.n6_intent,
        case.global_dag,
        case.schedule,
        case.projection,
        case.n4,
    )
    s3_linked = link_lite_moe_n6(lowered)
    seeds, expected = build_deterministic_timing_state_overrides(s3_linked)
    s3_program_io = build_timing_program_io(
        s3_linked,
        _S3_SHA,
        state_seed_overrides=seeds,
        state_expected_overrides=expected,
    )
    manifest = s3_linked.manifest
    s3_report = S3LiteRuntimeReport.create(
        case_id="case.s3_lite.static_moe_infer",
        baseline_epoch="s3-lite-v1",
        linked_program_id=s3_linked.id,
        manifest_id=manifest.id,
        manifest_digest=canonical_digest(manifest),
        program_io_id=s3_program_io.id,
        program_io_digest=canonical_digest(s3_program_io),
        artifact_sha256=_S3_SHA,
        artifact_bytes=30368,
        fragment_count=72,
        record_count=240,
        address_relocation_count=408,
        input_digest_count=77,
        runtime_definition_count=34,
        program_definition_count=141,
        address_binding_count=384,
        state_binding_count=24,
        blob_count=16,
        initialization_count=76,
        probe_count=8,
        hbm_read_bytes=24576,
        hbm_write_bytes=0,
        d2d_logical_bytes=256,
        d2d_data_packets=16,
        ack_total=4,
        done_total=2,
        makespan_cycles=8342,
        repeat_count=2,
        runtime_marker_digest=_S3_MARKERS,
        timing_execution=True,
        compute_functional=False,
        routing_functional=False,
        model_functional=False,
    )
    matrix = LiteRuntimeCapabilityMatrix.create(
        s2_case_id="case.s2_lite.lm_head_train",
        s2_status=CapabilityStatus.E2E_TIMING,
        s2_report_id=s2_report.id,
        s2_report_digest=canonical_digest(s2_report),
        s3_case_id="case.s3_lite.static_moe_infer",
        s3_status=CapabilityStatus.E2E_TIMING,
        s3_report_id=s3_report.id,
        s3_report_digest=canonical_digest(s3_report),
        full_training_status=CapabilityStatus.UNSUPPORTED,
        dynamic_moe_status=CapabilityStatus.UNSUPPORTED,
    )
    result = LiteRuntimeEvidenceBundle(
        s2.linked,
        s2.program_io,
        s2_report,
        s3_linked,
        s3_program_io,
        s3_report,
        matrix,
    )
    result.validate()
    return result


__all__ = ["LiteRuntimeEvidenceBundle", "build_lite_runtime_evidence"]
