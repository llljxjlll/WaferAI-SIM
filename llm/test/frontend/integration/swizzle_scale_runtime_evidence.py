"""Strict evidence for an honest Swizzle scale no-benefit conclusion."""

from __future__ import annotations

from dataclasses import dataclass
import math

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import (
    stable_artifact_id,
    validate_nonempty,
    validate_uint64,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleAlgorithm
from llm.frontend.wafer_frontend.schema.swizzle_performance_evidence import (
    SWIZZLE_MINIMUM_QUALIFYING_SPEEDUP,
    SwizzleBenefitBranch,
)


SWIZZLE_SCALE_RUNTIME_REPORT_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_scale_runtime_report/v1alpha1"
)

_MISSING_RUNTIME_MARKERS = (
    "compute_dte_overlap_cycles",
    "directional_port_utilization_over_time",
    "observed_max_inflight_recv",
    "observed_max_inflight_send",
)


def _digest(value: str, path: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _positive_float(value: float, path: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise SchemaError("must be a finite positive float", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleScaleDirectionalLinkEvidence:
    link_index: int
    source_die: int
    destination_die: int
    direction: str
    request_count: int
    ack_count: int
    packet_count: int

    def validate(self, path: str) -> None:
        for name in (
            "link_index",
            "source_die",
            "destination_die",
            "request_count",
            "ack_count",
            "packet_count",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        validate_nonempty(self.direction, f"{path}.direction")
        if self.source_die == self.destination_die:
            raise SchemaError("link endpoints must differ", path=path)


@dataclass(frozen=True, slots=True)
class SwizzleScaleRuntimeBranchEvidence:
    id: str
    scale_name: str
    scale_ordinal: int
    pattern: FusionPattern
    branch: SwizzleBenefitBranch
    algorithm: SwizzleAlgorithm
    same_work_digest: str
    problem_shape: tuple[int, int, int]
    rank_output_slices: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    topology: str
    chunk_count: int
    unroll_degree: int
    tile_shape: tuple[int, int, int] | None
    tile_efficiency: float
    logical_bytes: int
    byte_hops: int
    gemm_flops: int
    predicted_cycles: tuple[float, float, float]
    predicted_phases: tuple[float, float, float]
    repeat_makespans: tuple[int, int]
    repeat_marker_digests: tuple[str, str]
    packet_count: int
    directional_links: tuple[SwizzleScaleDirectionalLinkEvidence, ...]
    observed_max_inflight_send: int | None
    observed_max_inflight_recv: int | None
    directional_port_utilization_over_time: float | None
    compute_dte_overlap_cycles: int | None
    record_count: int
    opcode_counts: tuple[tuple[str, int], ...]
    owned_buffer_count: int
    borrowed_buffer_count: int
    aliased_buffer_count: int
    alloc_record_count: int
    free_record_count: int
    event_record_count: int
    sram_high_water_bytes: int
    artifact_size_bytes: int
    artifact_sha256: str
    manifest_id: str
    program_io_id: str
    initialization_count: int
    probe_count: int
    economic_auto_selected: bool
    forced_deployment: bool
    decision_reason: str
    deployment_reason: str
    finalizer_sha256: str
    resolver_sha256: str
    npusim_sha256: str
    measurement_complete: bool
    missing_runtime_markers: tuple[str, ...]

    @classmethod
    def create(cls, **semantic: object) -> "SwizzleScaleRuntimeBranchEvidence":
        result = cls(
            stable_artifact_id(
                "swizzle_scale_runtime_branch",
                semantic,
                schema_version=SWIZZLE_SCALE_RUNTIME_REPORT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "id"
        }

    def validate(self, path: str = "swizzle_scale_runtime_branch") -> None:
        if self.scale_name != f"S{self.scale_ordinal}" or self.scale_ordinal not in (1, 2):
            raise SchemaError("official scale must be S1 or S2", path=f"{path}.scale_name")
        if type(self.pattern) is not FusionPattern:
            raise SchemaError("requires typed fusion pattern", path=f"{path}.pattern")
        if type(self.branch) is not SwizzleBenefitBranch or type(self.algorithm) is not SwizzleAlgorithm:
            raise SchemaError("requires typed branch/algorithm", path=path)
        _digest(self.same_work_digest, f"{path}.same_work_digest")
        if (
            type(self.problem_shape) is not tuple
            or len(self.problem_shape) != 3
            or any(type(value) is not int or value <= 0 for value in self.problem_shape)
        ):
            raise SchemaError("problem shape must be one positive M/N/K triple", path=f"{path}.problem_shape")
        if type(self.rank_output_slices) is not tuple or len(self.rank_output_slices) != 4:
            raise SchemaError("official 2x2 run requires four rank output slices", path=f"{path}.rank_output_slices")
        for index, item in enumerate(self.rank_output_slices):
            if (
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not tuple
                or type(item[1]) is not tuple
                or len(item[0]) != len(item[1])
                or not item[0]
                or any(type(value) is not int or value < 0 for value in item[0])
                or any(type(value) is not int or value <= 0 for value in item[1])
            ):
                raise SchemaError("rank output slice must be exact offset/shape tuples", path=f"{path}.rank_output_slices[{index}]")
        validate_nonempty(self.topology, f"{path}.topology")
        for name in (
            "chunk_count", "unroll_degree", "logical_bytes", "byte_hops",
            "gemm_flops", "packet_count", "record_count", "owned_buffer_count",
            "borrowed_buffer_count", "aliased_buffer_count", "alloc_record_count",
            "free_record_count", "event_record_count", "sram_high_water_bytes",
            "artifact_size_bytes", "initialization_count", "probe_count",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.logical_bytes == 0 or self.gemm_flops == 0 or self.record_count == 0:
            raise SchemaError("work and record counts must be positive", path=path)
        _positive_float(self.tile_efficiency, f"{path}.tile_efficiency")
        if self.tile_efficiency > 1.0:
            raise SchemaError("tile efficiency cannot exceed one", path=f"{path}.tile_efficiency")
        if self.branch is SwizzleBenefitBranch.NAIVE:
            if (
                self.algorithm is not SwizzleAlgorithm.UNFUSED
                or self.chunk_count != 0
                or self.unroll_degree != 0
                or self.tile_shape is not None
                or self.economic_auto_selected
                or self.forced_deployment
            ):
                raise SchemaError("NAIVE facts drifted", path=path)
        elif self.branch is SwizzleBenefitBranch.SWIZZLE_AUTO:
            if (
                self.algorithm is SwizzleAlgorithm.UNFUSED
                or self.chunk_count == 0
                or self.unroll_degree not in (1, 2)
                or type(self.tile_shape) is not tuple
                or len(self.tile_shape) != 3
                or not self.economic_auto_selected
                or self.forced_deployment
            ):
                raise SchemaError("economic AUTO facts drifted", path=path)
        else:
            raise SchemaError("forced diagnostics cannot enter official evidence", path=f"{path}.branch")
        if (
            type(self.predicted_cycles) is not tuple
            or len(self.predicted_cycles) != 3
            or any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in self.predicted_cycles)
            or not self.predicted_cycles[0] <= self.predicted_cycles[1] <= self.predicted_cycles[2]
        ):
            raise SchemaError("predicted lower/estimate/upper interval is invalid", path=f"{path}.predicted_cycles")
        if (
            type(self.predicted_phases) is not tuple
            or len(self.predicted_phases) != 3
            or any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in self.predicted_phases)
        ):
            raise SchemaError("predicted phase tuple is invalid", path=f"{path}.predicted_phases")
        if (
            type(self.repeat_makespans) is not tuple
            or len(self.repeat_makespans) != 2
            or self.repeat_makespans[0] <= 0
            or self.repeat_makespans[0] != self.repeat_makespans[1]
        ):
            raise SchemaError("runtime makespans must repeat exactly", path=f"{path}.repeat_makespans")
        if type(self.repeat_marker_digests) is not tuple or len(self.repeat_marker_digests) != 2:
            raise SchemaError("requires two marker digests", path=f"{path}.repeat_marker_digests")
        for index, digest in enumerate(self.repeat_marker_digests):
            _digest(digest, f"{path}.repeat_marker_digests[{index}]")
        if self.repeat_marker_digests[0] != self.repeat_marker_digests[1]:
            raise SchemaError("marker digests must repeat exactly", path=f"{path}.repeat_marker_digests")
        if type(self.directional_links) is not tuple or len(self.directional_links) != 8:
            raise SchemaError("2x2 runtime requires eight directed link rows", path=f"{path}.directional_links")
        for index, link in enumerate(self.directional_links):
            link.validate(f"{path}.directional_links[{index}]")
        if tuple(link.link_index for link in self.directional_links) != tuple(range(8)):
            raise SchemaError("directional links must be canonically indexed", path=f"{path}.directional_links")
        if sum(link.packet_count for link in self.directional_links) != self.packet_count:
            raise SchemaError("directional packets do not close global packet count", path=f"{path}.packet_count")
        if (
            self.measurement_complete
            or self.observed_max_inflight_send is not None
            or self.observed_max_inflight_recv is not None
            or self.directional_port_utilization_over_time is not None
            or self.compute_dte_overlap_cycles is not None
            or self.missing_runtime_markers != _MISSING_RUNTIME_MARKERS
        ):
            raise SchemaError("missing runtime markers must remain explicit and fail-closed", path=path)
        if (
            type(self.opcode_counts) is not tuple
            or not self.opcode_counts
            or self.opcode_counts != tuple(sorted(self.opcode_counts))
            or sum(count for _, count in self.opcode_counts) != self.record_count
        ):
            raise SchemaError("opcode counts must be canonical and close record count", path=f"{path}.opcode_counts")
        for name, count in self.opcode_counts:
            validate_nonempty(name, f"{path}.opcode_counts.name")
            validate_uint64(count, f"{path}.opcode_counts[{name}]")
            if count == 0:
                raise SchemaError("opcode counts must be positive", path=f"{path}.opcode_counts[{name}]")
        if self.alloc_record_count != self.free_record_count:
            raise SchemaError("ALLOC/FREE lifecycle must close", path=path)
        for name in (
            "artifact_sha256", "finalizer_sha256", "resolver_sha256", "npusim_sha256",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        for name in ("manifest_id", "program_io_id", "decision_reason", "deployment_reason"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.economic_auto_selected) is not bool or type(self.forced_deployment) is not bool:
            raise SchemaError("selection flags must be bool", path=path)
        expected = stable_artifact_id(
            "swizzle_scale_runtime_branch",
            self._semantic_key(),
            schema_version=SWIZZLE_SCALE_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class SwizzleScaleRuntimePairEvidence:
    scale_name: str
    pattern: FusionPattern
    same_work_digest: str
    naive_ref: str
    swizzle_auto_ref: str
    speedup: float
    threshold: float
    qualifies: bool
    bottlenecks: tuple[str, ...]

    def validate(
        self,
        path: str = "swizzle_scale_runtime_pair",
        branches: dict[str, SwizzleScaleRuntimeBranchEvidence] | None = None,
    ) -> None:
        validate_nonempty(self.scale_name, f"{path}.scale_name")
        if type(self.pattern) is not FusionPattern:
            raise SchemaError("requires typed fusion pattern", path=f"{path}.pattern")
        _digest(self.same_work_digest, f"{path}.same_work_digest")
        validate_nonempty(self.naive_ref, f"{path}.naive_ref")
        validate_nonempty(self.swizzle_auto_ref, f"{path}.swizzle_auto_ref")
        _positive_float(self.speedup, f"{path}.speedup")
        if self.threshold != SWIZZLE_MINIMUM_QUALIFYING_SPEEDUP:
            raise SchemaError("qualification threshold drifted", path=f"{path}.threshold")
        if type(self.qualifies) is not bool:
            raise SchemaError("qualifies must be bool", path=f"{path}.qualifies")
        if self.bottlenecks != ("control_setup_overhead", "matmul_setup_overhead"):
            raise SchemaError("no-benefit bottleneck classification drifted", path=f"{path}.bottlenecks")
        if branches is None:
            return
        naive = branches.get(self.naive_ref)
        auto = branches.get(self.swizzle_auto_ref)
        if naive is None or auto is None:
            raise SchemaError("pair references unknown branches", path=path)
        if (
            naive.scale_name != self.scale_name
            or auto.scale_name != self.scale_name
            or naive.pattern is not self.pattern
            or auto.pattern is not self.pattern
            or naive.same_work_digest != self.same_work_digest
            or auto.same_work_digest != self.same_work_digest
            or naive.branch is not SwizzleBenefitBranch.NAIVE
            or auto.branch is not SwizzleBenefitBranch.SWIZZLE_AUTO
            or (
                naive.problem_shape,
                naive.rank_output_slices,
                naive.logical_bytes,
                naive.gemm_flops,
            )
            != (
                auto.problem_shape,
                auto.rank_output_slices,
                auto.logical_bytes,
                auto.gemm_flops,
            )
        ):
            raise SchemaError("pair same-work identity drifted", path=path)
        expected_speedup = naive.repeat_makespans[0] / auto.repeat_makespans[0]
        if type(self.speedup) is not float or not math.isclose(self.speedup, expected_speedup):
            raise SchemaError("speedup does not match actual makespans", path=f"{path}.speedup")
        if self.qualifies or self.speedup >= self.threshold:
            raise SchemaError("this report is an exact no-benefit conclusion", path=f"{path}.qualifies")
        naive_opcodes = dict(naive.opcode_counts)
        auto_opcodes = dict(auto.opcode_counts)
        if (
            auto.record_count <= naive.record_count
            or auto_opcodes.get("MATMUL", 0) <= naive_opcodes.get("MATMUL", 0)
        ):
            raise SchemaError(
                "MATMUL/control setup bottlenecks require extra MATMUL and total records",
                path=f"{path}.bottlenecks",
            )


@dataclass(frozen=True, slots=True)
class SwizzleScaleRuntimeReport:
    schema_version: str
    id: str
    branches: tuple[SwizzleScaleRuntimeBranchEvidence, ...]
    pairs: tuple[SwizzleScaleRuntimePairEvidence, ...]
    measurement_complete: bool
    performance_benefit: bool
    capability_flags: tuple[tuple[str, bool], ...]

    @classmethod
    def create(
        cls,
        *,
        branches: tuple[SwizzleScaleRuntimeBranchEvidence, ...],
        pairs: tuple[SwizzleScaleRuntimePairEvidence, ...],
        capability_flags: tuple[tuple[str, bool], ...],
    ) -> "SwizzleScaleRuntimeReport":
        semantic = {
            "branches": branches,
            "pairs": pairs,
            "measurement_complete": False,
            "performance_benefit": False,
            "capability_flags": capability_flags,
        }
        result = cls(
            SWIZZLE_SCALE_RUNTIME_REPORT_SCHEMA_VERSION,
            stable_artifact_id(
                "swizzle_scale_runtime_report",
                semantic,
                schema_version=SWIZZLE_SCALE_RUNTIME_REPORT_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "branches": self.branches,
            "pairs": self.pairs,
            "measurement_complete": self.measurement_complete,
            "performance_benefit": self.performance_benefit,
            "capability_flags": self.capability_flags,
        }

    def validate(self, path: str = "swizzle_scale_runtime_report") -> None:
        if self.schema_version != SWIZZLE_SCALE_RUNTIME_REPORT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if type(self.branches) is not tuple or len(self.branches) != 6:
            raise SchemaError("official report requires all six branches", path=f"{path}.branches")
        for index, branch in enumerate(self.branches):
            branch.validate(f"{path}.branches[{index}]")
        keys = tuple((item.scale_name, item.pattern, item.branch) for item in self.branches)
        expected_keys = (
            ("S1", FusionPattern.AG_GEMM, SwizzleBenefitBranch.NAIVE),
            ("S1", FusionPattern.AG_GEMM, SwizzleBenefitBranch.SWIZZLE_AUTO),
            ("S2", FusionPattern.AG_GEMM, SwizzleBenefitBranch.NAIVE),
            ("S2", FusionPattern.AG_GEMM, SwizzleBenefitBranch.SWIZZLE_AUTO),
            ("S2", FusionPattern.GEMM_RS, SwizzleBenefitBranch.NAIVE),
            ("S2", FusionPattern.GEMM_RS, SwizzleBenefitBranch.SWIZZLE_AUTO),
        )
        if keys != expected_keys or len({item.id for item in self.branches}) != 6:
            raise SchemaError("official branch coverage/order drifted", path=f"{path}.branches")
        if type(self.pairs) is not tuple or len(self.pairs) != 3:
            raise SchemaError("official report requires three NAIVE/AUTO pairs", path=f"{path}.pairs")
        branch_index = {item.id: item for item in self.branches}
        for index, pair in enumerate(self.pairs):
            pair.validate(f"{path}.pairs[{index}]", branch_index)
        if self.measurement_complete or self.performance_benefit:
            raise SchemaError("missing markers and sub-threshold speedups cannot raise capability", path=path)
        expected_flags = (
            ("performance_benefit", False),
            ("supports_calibrated_tile_efficiency", False),
            ("supports_control_lifecycle_compaction", True),
            ("supports_economic_swizzle_auto_selection", True),
            ("supports_production_meshslice_2d", False),
            ("supports_rank_independent_chunk_search", True),
            ("supports_reproducible_swizzle_speedup", False),
            ("supports_runtime_verified_double_buffer", False),
            ("supports_runtime_verified_multi_inflight", False),
        )
        if self.capability_flags != expected_flags:
            raise SchemaError("capability flags overclaim or drift", path=f"{path}.capability_flags")
        expected = stable_artifact_id(
            "swizzle_scale_runtime_report",
            self._semantic_key(),
            schema_version=SWIZZLE_SCALE_RUNTIME_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


__all__ = [
    "SWIZZLE_SCALE_RUNTIME_REPORT_SCHEMA_VERSION",
    "SwizzleScaleDirectionalLinkEvidence",
    "SwizzleScaleRuntimeBranchEvidence",
    "SwizzleScaleRuntimePairEvidence",
    "SwizzleScaleRuntimeReport",
]
