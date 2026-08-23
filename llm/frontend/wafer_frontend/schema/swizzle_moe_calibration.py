"""Strict calibration and runtime-marker evidence for MoE Swizzle V2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from ..errors import SchemaError
from .common import DType, stable_artifact_id, validate_nonempty, validate_uint64


MOE_SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_calibration_profile/v1alpha1"
)
MOE_SWIZZLE_RUNTIME_MARKERS_SCHEMA_VERSION = (
    "wafer_frontend.moe_swizzle_runtime_markers/v1alpha1"
)
MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES = tuple(
    (m, n, k)
    for n, k in ((8, 32), (16, 32), (32, 16))
    for m in (1, 2, 4, 8)
)
MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES = tuple(
    (m, 32, m * 32)
    for m in (1, 2, 4, 8)
)


class MoeCalibrationStatus(str, Enum):
    PROVISIONAL = "provisional"
    MEASURED = "measured"


class MoeCalibrationKind(str, Enum):
    GROUP_GEMM = "group_gemm"
    SWIGLU_GROUP = "swiglu_group"
    DTE_LAUNCH = "dte_launch"
    DTE_SYNC = "dte_sync"
    DTE_HOP = "dte_hop"
    SESSION_OPEN = "session_open"
    SESSION_RETIRE = "session_retire"
    LOCAL_COPY = "local_copy"
    SRAM_ALLOC = "sram_alloc"
    SRAM_BIND = "sram_bind"
    SRAM_FREE = "sram_free"
    EVENT_SET = "event_set"
    EVENT_WAIT = "event_wait"
    TERMINAL_DONE = "terminal_done"


_SHAPED_KINDS = (
    MoeCalibrationKind.GROUP_GEMM,
    MoeCalibrationKind.SWIGLU_GROUP,
)
_FIXED_KINDS = tuple(item for item in MoeCalibrationKind if item not in _SHAPED_KINDS)
_RUNTIME_MISSING = (
    "compute_dte_overlap_cycles",
    "die_compute_dte_overlap_over_time",
    "directional_port_utilization_over_time",
    "die_session_capacity",
    "dte_launch_count",
    "event_record_count",
    "group_gemm_primitive_count",
    "group_gemm_setup_cycles",
    "matmul_total_cycles",
    "observed_max_inflight_recv",
    "observed_max_inflight_send",
    "physical_root_count",
    "sram_lifecycle_cycles",
    "bind_cycles",
    "event_control_cycles",
)

_MESH_2X2_DIRECTIONAL_EDGES = {
    (0, 1, "x+"),
    (1, 0, "x-"),
    (2, 3, "x+"),
    (3, 2, "x-"),
    (0, 2, "y+"),
    (2, 0, "y-"),
    (1, 3, "y+"),
    (3, 1, "y-"),
}


def _digest(value: str, path: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _positive(value: int, path: str) -> None:
    validate_uint64(value, path)
    if value == 0:
        raise SchemaError("must be positive", path=path)


@dataclass(frozen=True, slots=True)
class MoeCalibrationSample:
    kind: MoeCalibrationKind
    sample_index: int
    repeat_index: int
    cycles: int
    shape: tuple[int, int, int] | None
    dtype: DType | None
    tool_sha256: str
    hardware_sha256: str
    simulation_sha256: str
    mapping_sha256: str

    def validate(self, path: str = "moe_calibration_sample") -> None:
        if type(self.kind) is not MoeCalibrationKind:
            raise SchemaError("must use a typed calibration kind", path=f"{path}.kind")
        validate_uint64(self.sample_index, f"{path}.sample_index")
        validate_uint64(self.repeat_index, f"{path}.repeat_index")
        if self.repeat_index not in (0, 1):
            raise SchemaError("repeat must be exactly 0 or 1", path=f"{path}.repeat_index")
        _positive(self.cycles, f"{path}.cycles")
        if self.kind in _SHAPED_KINDS:
            if (
                type(self.shape) is not tuple
                or len(self.shape) != 3
                or any(type(item) is not int or item <= 0 for item in self.shape)
                or self.dtype is not DType.FP16
            ):
                raise SchemaError("compute calibration requires an exact positive FP16 shape", path=path)
            allowed = (
                MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES
                if self.kind is MoeCalibrationKind.GROUP_GEMM
                else MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES
            )
            if self.shape not in allowed:
                raise SchemaError(
                    "compute calibration shape is outside exact production coverage",
                    path=f"{path}.shape",
                )
        elif self.shape is not None or self.dtype is not None:
            raise SchemaError("fixed-cost sample forbids shape/dtype", path=path)
        for name in (
            "tool_sha256", "hardware_sha256", "simulation_sha256", "mapping_sha256",
        ):
            _digest(getattr(self, name), f"{path}.{name}")


def calibration_missing(
    samples: tuple[MoeCalibrationSample, ...],
) -> tuple[str, ...]:
    missing: set[str] = set()
    shaped = (
        (
            MoeCalibrationKind.GROUP_GEMM,
            MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES,
            "group_gemm_shape",
        ),
        (
            MoeCalibrationKind.SWIGLU_GROUP,
            MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES,
            "swiglu_group_shape",
        ),
    )
    for kind, required_shapes, label in shaped:
        for shape in required_shapes:
            keys = {
                (item.sample_index, item.repeat_index)
                for item in samples
                if item.kind is kind and item.shape == shape
            }
            if keys != {(sample, repeat) for sample in range(3) for repeat in range(2)}:
                missing.add(f"{label}:{shape}")
    for kind in _FIXED_KINDS:
        keys = {
            (item.sample_index, item.repeat_index)
            for item in samples if item.kind is kind
        }
        if keys != {(sample, repeat) for sample in range(3) for repeat in range(2)}:
            missing.add(kind.value)
    return tuple(sorted(missing))


@dataclass(frozen=True, slots=True)
class MoeSwizzleCalibrationProfile:
    schema_version: str
    producer_pass: str
    id: str
    status: MoeCalibrationStatus
    samples: tuple[MoeCalibrationSample, ...]
    missing_measurements: tuple[str, ...]
    tool_sha256: str
    hardware_sha256: str
    simulation_sha256: str
    mapping_sha256: str

    @classmethod
    def create(
        cls,
        *,
        samples: tuple[MoeCalibrationSample, ...],
        tool_sha256: str,
        hardware_sha256: str,
        simulation_sha256: str,
        mapping_sha256: str,
    ) -> "MoeSwizzleCalibrationProfile":
        missing = calibration_missing(samples)
        semantic = {
            "status": (
                MoeCalibrationStatus.MEASURED
                if not missing else MoeCalibrationStatus.PROVISIONAL
            ),
            "samples": samples,
            "missing_measurements": missing,
            "tool_sha256": tool_sha256,
            "hardware_sha256": hardware_sha256,
            "simulation_sha256": simulation_sha256,
            "mapping_sha256": mapping_sha256,
        }
        result = cls(
            MOE_SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION,
            "moe_swizzle_calibration_parser",
            stable_artifact_id(
                "moe_swizzle_calibration_profile",
                semantic,
                schema_version=MOE_SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "moe_swizzle_calibration_profile") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION
            or self.producer_pass != "moe_swizzle_calibration_parser"
        ):
            raise SchemaError("unsupported calibration schema/producer", path=path)
        for name in (
            "tool_sha256", "hardware_sha256", "simulation_sha256", "mapping_sha256",
        ):
            _digest(getattr(self, name), f"{path}.{name}")
        for index, item in enumerate(self.samples):
            item.validate(f"{path}.samples[{index}]")
            if tuple(
                getattr(item, name) for name in (
                    "tool_sha256", "hardware_sha256", "simulation_sha256", "mapping_sha256",
                )
            ) != tuple(
                getattr(self, name) for name in (
                    "tool_sha256", "hardware_sha256", "simulation_sha256", "mapping_sha256",
                )
            ):
                raise SchemaError("sample/config SHA drifted", path=f"{path}.samples[{index}]")
        keys = tuple(
            (item.kind, item.shape, item.sample_index, item.repeat_index)
            for item in self.samples
        )
        if len(keys) != len(set(keys)):
            raise SchemaError("duplicate calibration sample", path=f"{path}.samples")
        unexpected_group_gemm = {
            item.shape for item in self.samples
            if item.kind is MoeCalibrationKind.GROUP_GEMM
        } - set(MOE_SWIZZLE_PRODUCTION_GROUP_GEMM_SHAPES)
        unexpected_swiglu = {
            item.shape for item in self.samples
            if item.kind is MoeCalibrationKind.SWIGLU_GROUP
        } - set(MOE_SWIZZLE_PRODUCTION_SWIGLU_GROUP_SHAPES)
        if unexpected_group_gemm or unexpected_swiglu:
            raise SchemaError(
                "compute samples must use exact production candidate shapes",
                path=f"{path}.samples",
            )
        expected_missing = calibration_missing(self.samples)
        if self.missing_measurements != expected_missing:
            raise SchemaError("missing-measurement list drifted", path=f"{path}.missing_measurements")
        measured = self.status is MoeCalibrationStatus.MEASURED
        if measured != (not expected_missing):
            raise SchemaError("MEASURED requires complete independent coverage", path=f"{path}.status")
        expected = stable_artifact_id(
            "moe_swizzle_calibration_profile",
            self._semantic_key(),
            schema_version=MOE_SWIZZLE_CALIBRATION_PROFILE_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class MoeDirectionalPortTime:
    source_die: int
    destination_die: int
    direction: str
    busy_cycles: int
    window_cycles: int

    def validate(self, path: str = "moe_directional_port_time") -> None:
        for name in ("source_die", "destination_die", "busy_cycles", "window_cycles"):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        validate_nonempty(self.direction, f"{path}.direction")
        if self.source_die == self.destination_die or self.window_cycles == 0:
            raise SchemaError("port-time edge/window is invalid", path=path)
        if self.busy_cycles > self.window_cycles:
            raise SchemaError("busy cycles exceed observation window", path=f"{path}.busy_cycles")


@dataclass(frozen=True, slots=True)
class MoeDieComputeDteOverlap:
    die: int
    compute_cycles: int
    dte_cycles: int
    compute_dte_cycles: int
    window_cycles: int

    def validate(self, path: str = "moe_die_compute_dte_overlap") -> None:
        for name in (
            "die", "compute_cycles", "dte_cycles", "compute_dte_cycles",
            "window_cycles",
        ):
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if self.die > 3 or self.window_cycles == 0:
            raise SchemaError("die/window is outside the exact 2x2 capture", path=path)
        if (
            self.compute_cycles > self.window_cycles
            or self.dte_cycles > self.window_cycles
            or self.compute_dte_cycles > min(self.compute_cycles, self.dte_cycles)
        ):
            raise SchemaError("overlap interval totals are inconsistent", path=path)


@dataclass(frozen=True, slots=True)
class MoeDieSessionCapacity:
    die: int
    capacity_per_core: int
    active_core_count: int
    aggregate_capacity: int
    send_peak: int
    recv_peak: int

    def validate(self, path: str = "moe_die_session_capacity") -> None:
        for name in self.__dataclass_fields__:
            validate_uint64(getattr(self, name), f"{path}.{name}")
        if (
            self.die > 3
            or self.capacity_per_core != 3
            or self.active_core_count == 0
            or self.aggregate_capacity
            != self.capacity_per_core * self.active_core_count
            or self.send_peak > self.aggregate_capacity
            or self.recv_peak > self.aggregate_capacity
        ):
            raise SchemaError("session capacity/core/peak closure drifted", path=path)


@dataclass(frozen=True, slots=True)
class MoeSwizzleRuntimeMarkers:
    schema_version: str
    producer_pass: str
    id: str
    marker_digest: str
    measurement_complete: bool
    missing_measurements: tuple[str, ...]
    observed_max_inflight_send: int | None
    observed_max_inflight_recv: int | None
    die_session_capacities: tuple[MoeDieSessionCapacity, ...]
    compute_dte_overlap_cycles: int | None
    die_compute_dte_overlaps: tuple[MoeDieComputeDteOverlap, ...]
    directional_port_times: tuple[MoeDirectionalPortTime, ...]
    group_gemm_primitive_count: int | None
    group_gemm_setup_cycles: int | None
    matmul_total_cycles: int | None
    dte_launch_count: int | None
    physical_root_count: int | None
    event_record_count: int | None
    sram_lifecycle_cycles: int | None
    bind_cycles: int | None
    event_control_cycles: int | None

    @classmethod
    def create(cls, **semantic: object) -> "MoeSwizzleRuntimeMarkers":
        result = cls(
            MOE_SWIZZLE_RUNTIME_MARKERS_SCHEMA_VERSION,
            "moe_swizzle_runtime_marker_parser",
            stable_artifact_id(
                "moe_swizzle_runtime_markers",
                semantic,
                schema_version=MOE_SWIZZLE_RUNTIME_MARKERS_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "producer_pass", "id")
        }

    def validate(self, path: str = "moe_swizzle_runtime_markers") -> None:
        if (
            self.schema_version != MOE_SWIZZLE_RUNTIME_MARKERS_SCHEMA_VERSION
            or self.producer_pass != "moe_swizzle_runtime_marker_parser"
        ):
            raise SchemaError("unsupported runtime marker schema/producer", path=path)
        _digest(self.marker_digest, f"{path}.marker_digest")
        if type(self.measurement_complete) is not bool:
            raise SchemaError("must be bool", path=f"{path}.measurement_complete")
        for name in (
            "observed_max_inflight_send", "observed_max_inflight_recv",
            "compute_dte_overlap_cycles", "group_gemm_primitive_count",
            "group_gemm_setup_cycles", "matmul_total_cycles",
            "dte_launch_count", "physical_root_count", "event_record_count",
            "sram_lifecycle_cycles", "bind_cycles", "event_control_cycles",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_uint64(value, f"{path}.{name}")
        for index, item in enumerate(self.directional_port_times):
            item.validate(f"{path}.directional_port_times[{index}]")
        for index, item in enumerate(self.die_compute_dte_overlaps):
            item.validate(f"{path}.die_compute_dte_overlaps[{index}]")
        overlap_dies = tuple(item.die for item in self.die_compute_dte_overlaps)
        if len(overlap_dies) != len(set(overlap_dies)):
            raise SchemaError("duplicate die overlap marker", path=f"{path}.die_compute_dte_overlaps")
        for index, item in enumerate(self.die_session_capacities):
            item.validate(f"{path}.die_session_capacities[{index}]")
        session_dies = tuple(item.die for item in self.die_session_capacities)
        if len(session_dies) != len(set(session_dies)):
            raise SchemaError("duplicate die session capacity", path=f"{path}.die_session_capacities")
        edges = tuple((item.source_die, item.destination_die) for item in self.directional_port_times)
        if len(edges) != len(set(edges)):
            raise SchemaError("duplicate directional port edge", path=f"{path}.directional_port_times")
        typed_edges = {
            (item.source_die, item.destination_die, item.direction)
            for item in self.directional_port_times
        }
        if not typed_edges.issubset(_MESH_2X2_DIRECTIONAL_EDGES):
            raise SchemaError(
                "port-time markers must belong to the exact directed 2x2 mesh",
                path=f"{path}.directional_port_times",
            )
        missing = []
        for name in _RUNTIME_MISSING:
            if name == "directional_port_utilization_over_time":
                if len(self.directional_port_times) != 8:
                    missing.append(name)
            elif name == "die_compute_dte_overlap_over_time":
                if set(overlap_dies) != {0, 1, 2, 3}:
                    missing.append(name)
            elif name == "die_session_capacity":
                if set(session_dies) != {0, 1, 2, 3}:
                    missing.append(name)
            elif getattr(self, name) is None:
                missing.append(name)
        expected_missing = tuple(sorted(missing))
        if self.missing_measurements != expected_missing:
            raise SchemaError("runtime missing-measurement list drifted", path=f"{path}.missing_measurements")
        if self.measurement_complete != (not expected_missing):
            raise SchemaError("measurement completeness drifted", path=f"{path}.measurement_complete")
        expected = stable_artifact_id(
            "moe_swizzle_runtime_markers",
            self._semantic_key(),
            schema_version=MOE_SWIZZLE_RUNTIME_MARKERS_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")


__all__ = [name for name in globals() if name.startswith("Moe") or name.startswith("MOE_SWIZZLE")]
