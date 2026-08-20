"""Typed, scope-limited runtime evidence for the two-day S2/S3 slices."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SchemaError
from .common import stable_artifact_id
from .capability import CapabilityStatus
from .lite_moe import S3_LITE_BASELINE_EPOCH
from .lite_moe_n6 import LiteMoeLinkedProgram
from .lite_train import S2_LITE_BASELINE_EPOCH
from .lite_train_n6 import S2LiteTrainLinkedProgram
from .program_io import ProgramIoContract
from .serde import canonical_digest


S2_LITE_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.s2_lite_runtime_report/v1alpha1"
)
S3_LITE_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.s3_lite_runtime_report/v1alpha1"
)
LITE_RUNTIME_CAPABILITY_MATRIX_SCHEMA_VERSION = (
    "wafer_frontend.lite_runtime_capability_matrix/v1alpha1"
)

_S2_CASE = "case.s2_lite.lm_head_train"
_S3_CASE = "case.s3_lite.static_moe_infer"
_S2_ARTIFACT_SHA = "e5dcd18e9de62400c2ea0eb706cc2f2e491f685d57712577aa11ab9427be46d9"
_S3_ARTIFACT_SHA = "c49a8202eac0fbbe89aa1b96ba6cba782136b6e8bdf09a6b5870817a27650361"


def _sha256(value: str, path: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise SchemaError("must be a canonical lowercase SHA-256", path=path)


def _manifest_counts(source: object) -> tuple[int, int, int]:
    manifest = source.manifest
    return (
        len(manifest.fragments),
        sum(
            len(stream.records)
            for fragment in manifest.fragments
            for stream in fragment.core_streams
        ),
        sum(
            len(stream.address_relocations)
            for fragment in manifest.fragments
            for stream in fragment.core_streams
        ),
    )


@dataclass(frozen=True, slots=True)
class S2LiteRuntimeReport:
    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    baseline_epoch: str
    linked_program_id: str
    manifest_id: str
    manifest_digest: str
    program_io_id: str
    program_io_digest: str
    artifact_sha256: str
    artifact_bytes: int
    fragment_count: int
    record_count: int
    address_relocation_count: int
    blob_count: int
    initialization_count: int
    probe_count: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    ack_total: int
    done_total: int
    makespan_cycles: int
    repeat_count: int
    runtime_marker_digest: str
    timing_execution: bool
    compute_functional: bool
    model_functional: bool

    @classmethod
    def create(cls, **semantic_key: object) -> "S2LiteRuntimeReport":
        result = cls(
            schema_version=S2_LITE_RUNTIME_REPORT_SCHEMA_VERSION,
            producer_pass="s2_lite_runtime_evidence",
            id=stable_artifact_id(
                "s2_lite_runtime_report",
                semantic_key,
                schema_version=S2_LITE_RUNTIME_REPORT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "s2_lite_runtime_report") -> None:
        if (
            self.schema_version != S2_LITE_RUNTIME_REPORT_SCHEMA_VERSION
            or self.producer_pass != "s2_lite_runtime_evidence"
        ):
            raise SchemaError("unsupported report schema/producer", path=path)
        if self.case_id != _S2_CASE or self.baseline_epoch != S2_LITE_BASELINE_EPOCH:
            raise SchemaError("S2-Lite case/epoch changed", path=path)
        for name in ("manifest_digest", "program_io_digest", "artifact_sha256", "runtime_marker_digest"):
            _sha256(getattr(self, name), f"{path}.{name}")
        if (
            self.artifact_sha256 != _S2_ARTIFACT_SHA
            or (
                self.artifact_bytes,
                self.fragment_count,
                self.record_count,
                self.address_relocation_count,
                self.blob_count,
                self.initialization_count,
                self.probe_count,
                self.hbm_read_bytes,
                self.hbm_write_bytes,
                self.ack_total,
                self.done_total,
                self.makespan_cycles,
                self.repeat_count,
            ) != (23616, 46, 169, 324, 23, 62, 1, 13472, 1024, 2, 1, 6164, 2)
        ):
            raise SchemaError("S2-Lite observed evidence changed", path=path)
        if not self.timing_execution or self.compute_functional or self.model_functional:
            raise SchemaError("S2-Lite only proves non-functional timing execution", path=path)
        expected_id = stable_artifact_id(
            "s2_lite_runtime_report",
            self._semantic_key(),
            schema_version=S2_LITE_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable artifact id", path=f"{path}.id")

    def validate_against(
        self,
        source: S2LiteTrainLinkedProgram,
        program_io: ProgramIoContract,
        path: str = "s2_lite_runtime_report",
    ) -> None:
        self.validate(path)
        if type(source) is not S2LiteTrainLinkedProgram:
            raise SchemaError("requires S2LiteTrainLinkedProgram", path=f"{path}.source")
        source.validate(f"{path}.source")
        program_io.validate_against(source.manifest, f"{path}.program_io")
        if (
            self.linked_program_id != source.id
            or self.manifest_id != source.manifest.id
            or self.manifest_digest != canonical_digest(source.manifest)
            or self.program_io_id != program_io.id
            or self.program_io_digest != canonical_digest(program_io)
            or _manifest_counts(source) != (46, 169, 324)
            or (
                len(program_io.blobs),
                len(program_io.initializations),
                len(program_io.output_probes),
            ) != (23, 62, 1)
        ):
            raise SchemaError("S2-Lite production closure changed", path=path)


@dataclass(frozen=True, slots=True)
class S3LiteRuntimeReport:
    schema_version: str
    producer_pass: str
    id: str
    case_id: str
    baseline_epoch: str
    linked_program_id: str
    manifest_id: str
    manifest_digest: str
    program_io_id: str
    program_io_digest: str
    artifact_sha256: str
    artifact_bytes: int
    fragment_count: int
    record_count: int
    address_relocation_count: int
    input_digest_count: int
    runtime_definition_count: int
    program_definition_count: int
    address_binding_count: int
    state_binding_count: int
    blob_count: int
    initialization_count: int
    probe_count: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    d2d_logical_bytes: int
    d2d_data_packets: int
    ack_total: int
    done_total: int
    makespan_cycles: int
    repeat_count: int
    runtime_marker_digest: str
    timing_execution: bool
    compute_functional: bool
    routing_functional: bool
    model_functional: bool

    @classmethod
    def create(cls, **semantic_key: object) -> "S3LiteRuntimeReport":
        result = cls(
            schema_version=S3_LITE_RUNTIME_REPORT_SCHEMA_VERSION,
            producer_pass="s3_lite_runtime_evidence",
            id=stable_artifact_id(
                "s3_lite_runtime_report",
                semantic_key,
                schema_version=S3_LITE_RUNTIME_REPORT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "s3_lite_runtime_report") -> None:
        if (
            self.schema_version != S3_LITE_RUNTIME_REPORT_SCHEMA_VERSION
            or self.producer_pass != "s3_lite_runtime_evidence"
        ):
            raise SchemaError("unsupported report schema/producer", path=path)
        if self.case_id != _S3_CASE or self.baseline_epoch != S3_LITE_BASELINE_EPOCH:
            raise SchemaError("S3-Lite case/epoch changed", path=path)
        for name in ("manifest_digest", "program_io_digest", "artifact_sha256", "runtime_marker_digest"):
            _sha256(getattr(self, name), f"{path}.{name}")
        if (
            self.artifact_sha256 != _S3_ARTIFACT_SHA
            or (
                self.artifact_bytes,
                self.fragment_count,
                self.record_count,
                self.address_relocation_count,
                self.input_digest_count,
                self.runtime_definition_count,
                self.program_definition_count,
                self.address_binding_count,
                self.state_binding_count,
                self.blob_count,
                self.initialization_count,
                self.probe_count,
                self.hbm_read_bytes,
                self.hbm_write_bytes,
                self.d2d_logical_bytes,
                self.d2d_data_packets,
                self.ack_total,
                self.done_total,
                self.makespan_cycles,
                self.repeat_count,
            ) != (30368, 72, 240, 408, 77, 34, 141, 384, 24, 16, 76, 8, 24576, 0, 256, 16, 4, 2, 8342, 2)
        ):
            raise SchemaError("S3-Lite observed evidence changed", path=path)
        if (
            not self.timing_execution
            or self.compute_functional
            or self.routing_functional
            or self.model_functional
        ):
            raise SchemaError("S3-Lite only proves non-functional timing execution", path=path)
        expected_id = stable_artifact_id(
            "s3_lite_runtime_report",
            self._semantic_key(),
            schema_version=S3_LITE_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable artifact id", path=f"{path}.id")

    def validate_against(
        self,
        source: LiteMoeLinkedProgram,
        program_io: ProgramIoContract,
        path: str = "s3_lite_runtime_report",
    ) -> None:
        self.validate(path)
        if type(source) is not LiteMoeLinkedProgram:
            raise SchemaError("requires LiteMoeLinkedProgram", path=f"{path}.source")
        source.validate(f"{path}.source")
        program_io.validate_against(source.manifest, f"{path}.program_io")
        manifest = source.manifest
        if (
            self.linked_program_id != source.id
            or self.manifest_id != manifest.id
            or self.manifest_digest != canonical_digest(manifest)
            or self.program_io_id != program_io.id
            or self.program_io_digest != canonical_digest(program_io)
            or _manifest_counts(source) != (72, 240, 408)
            or (
                len(manifest.input_digests),
                len(manifest.runtime_symbol_definitions),
                len(manifest.program_symbol_definitions),
                len(manifest.address_operand_bindings),
                len(manifest.state_operand_bindings),
            ) != (77, 34, 141, 384, 24)
            or (
                len(program_io.blobs),
                len(program_io.initializations),
                len(program_io.output_probes),
            ) != (16, 76, 8)
            or len(source.source.intent.dte_units) != 8
            or sum(unit.bytes for unit in source.source.intent.dte_units) != 256
        ):
            raise SchemaError("S3-Lite production closure changed", path=path)


@dataclass(frozen=True, slots=True)
class LiteRuntimeCapabilityMatrix:
    schema_version: str
    producer_pass: str
    id: str
    s2_case_id: str
    s2_status: CapabilityStatus
    s2_report_id: str
    s2_report_digest: str
    s3_case_id: str
    s3_status: CapabilityStatus
    s3_report_id: str
    s3_report_digest: str
    full_training_status: CapabilityStatus
    dynamic_moe_status: CapabilityStatus

    @classmethod
    def create(cls, **semantic_key: object) -> "LiteRuntimeCapabilityMatrix":
        result = cls(
            schema_version=LITE_RUNTIME_CAPABILITY_MATRIX_SCHEMA_VERSION,
            producer_pass="lite_runtime_capability_matrix",
            id=stable_artifact_id(
                "lite_runtime_capability_matrix",
                semantic_key,
                schema_version=LITE_RUNTIME_CAPABILITY_MATRIX_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "lite_runtime_capability_matrix") -> None:
        if (
            self.schema_version != LITE_RUNTIME_CAPABILITY_MATRIX_SCHEMA_VERSION
            or self.producer_pass != "lite_runtime_capability_matrix"
        ):
            raise SchemaError("unsupported matrix schema/producer", path=path)
        if (
            self.s2_case_id != _S2_CASE
            or self.s3_case_id != _S3_CASE
            or self.s2_status is not CapabilityStatus.E2E_TIMING
            or self.s3_status is not CapabilityStatus.E2E_TIMING
            or self.full_training_status is not CapabilityStatus.UNSUPPORTED
            or self.dynamic_moe_status is not CapabilityStatus.UNSUPPORTED
        ):
            raise SchemaError("Lite capability scope changed", path=path)
        _sha256(self.s2_report_digest, f"{path}.s2_report_digest")
        _sha256(self.s3_report_digest, f"{path}.s3_report_digest")
        expected_id = stable_artifact_id(
            "lite_runtime_capability_matrix",
            self._semantic_key(),
            schema_version=LITE_RUNTIME_CAPABILITY_MATRIX_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError("unstable artifact id", path=f"{path}.id")

    def validate_against(
        self,
        s2_report: S2LiteRuntimeReport,
        s3_report: S3LiteRuntimeReport,
        path: str = "lite_runtime_capability_matrix",
    ) -> None:
        self.validate(path)
        s2_report.validate(f"{path}.s2_report")
        s3_report.validate(f"{path}.s3_report")
        if (
            self.s2_report_id != s2_report.id
            or self.s2_report_digest != canonical_digest(s2_report)
            or self.s3_report_id != s3_report.id
            or self.s3_report_digest != canonical_digest(s3_report)
        ):
            raise SchemaError("capability evidence link changed", path=path)


__all__ = [
    "S2_LITE_RUNTIME_REPORT_SCHEMA_VERSION",
    "S3_LITE_RUNTIME_REPORT_SCHEMA_VERSION",
    "LITE_RUNTIME_CAPABILITY_MATRIX_SCHEMA_VERSION",
    "LiteRuntimeCapabilityMatrix",
    "S2LiteRuntimeReport",
    "S3LiteRuntimeReport",
]
