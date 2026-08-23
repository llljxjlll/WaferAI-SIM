"""Strict text-to-typed-evidence boundary for MoE Swizzle V2.

Generic simulator output is deliberately ignored.  Only dedicated markers
with an exact field set can populate calibration or runtime observations.
"""

from __future__ import annotations

import hashlib
import shlex

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MoeCalibrationKind,
    MoeCalibrationSample,
    MoeDieComputeDteOverlap,
    MoeDieSessionCapacity,
    MoeDirectionalPortTime,
    MoeSwizzleCalibrationProfile,
    MoeSwizzleRuntimeMarkers,
)


_CALIBRATION = "[MOE_SWIZZLE_CALIBRATION] "
_SESSION = "[MOE_SWIZZLE_SESSION] "
_OVERLAP = "[MOE_SWIZZLE_OVERLAP] "
_PORT = "[MOE_SWIZZLE_PORT_TIME] "
_SETUP = "[MOE_SWIZZLE_SETUP] "
_DEDICATED_STEM = "[MOE_SWIZZLE_"
_CALIBRATION_FIELDS = {
    "kind",
    "sample",
    "repeat",
    "cycles",
    "shape",
    "dtype",
    "tool_sha256",
    "hardware_sha256",
    "simulation_sha256",
    "mapping_sha256",
}
_SESSION_FIELDS = {
    "die", "capacity_per_core", "active_core_count", "aggregate_capacity",
    "send_peak", "recv_peak", "opens", "retires",
}
_OVERLAP_FIELDS = {
    "scope", "die", "compute_cycles", "dte_cycles",
    "compute_dte_cycles", "window_cycles",
}
_PORT_FIELDS = {
    "source_die",
    "destination_die",
    "direction",
    "busy_cycles",
    "window_cycles",
}
_SETUP_FIELDS = {
    "group_gemm_primitives",
    "group_gemm_setup_cycles",
    "matmul_total_cycles",
    "dte_launch_count",
    "physical_root_count",
    "event_record_count",
    "sram_lifecycle_cycles",
    "bind_cycles",
    "event_control_cycles",
}


def _digest(value: str, path: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SchemaError("must be a lowercase SHA-256 digest", path=path)


def _fields(line: str, prefix: str, line_number: int) -> dict[str, str]:
    try:
        words = shlex.split(line[len(prefix) :], posix=True)
    except ValueError as error:
        raise SchemaError(
            f"malformed dedicated marker: {error}",
            path=f"runtime_output.lines[{line_number}]",
        ) from error
    result: dict[str, str] = {}
    for word in words:
        if word.count("=") != 1:
            raise SchemaError(
                "dedicated marker fields must be key=value",
                path=f"runtime_output.lines[{line_number}]",
            )
        key, value = word.split("=", 1)
        if not key or not value or key in result:
            raise SchemaError(
                "dedicated marker fields must be nonempty and unique",
                path=f"runtime_output.lines[{line_number}]",
            )
        result[key] = value
    return result


def _exact(fields: dict[str, str], expected: set[str], line_number: int) -> None:
    if set(fields) != expected:
        raise SchemaError(
            "dedicated marker has missing or unknown fields; "
            f"expected={sorted(expected)}, actual={sorted(fields)}",
            path=f"runtime_output.lines[{line_number}]",
        )


def _uint(value: str, field: str, line_number: int) -> int:
    if not value.isascii() or not value.isdecimal():
        raise SchemaError(
            "must be an unsigned decimal integer",
            path=f"runtime_output.lines[{line_number}].{field}",
        )
    result = int(value)
    if result > (1 << 64) - 1:
        raise SchemaError("must fit uint64", path=f"runtime_output.lines[{line_number}].{field}")
    return result


def parse_moe_swizzle_calibration(
    output: str,
    *,
    tool_sha256: str,
    hardware_sha256: str,
    simulation_sha256: str,
    mapping_sha256: str,
) -> MoeSwizzleCalibrationProfile:
    """Parse dedicated isolated samples; absence yields PROVISIONAL, never zeros."""

    if type(output) is not str:
        raise SchemaError("must be text", path="runtime_output")
    expected_digests = {
        "tool_sha256": tool_sha256,
        "hardware_sha256": hardware_sha256,
        "simulation_sha256": simulation_sha256,
        "mapping_sha256": mapping_sha256,
    }
    for name, value in expected_digests.items():
        _digest(value, name)
    samples = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        if not line.startswith(_CALIBRATION):
            continue
        fields = _fields(line, _CALIBRATION, line_number)
        _exact(fields, _CALIBRATION_FIELDS, line_number)
        for name, expected in expected_digests.items():
            if fields[name] != expected:
                raise SchemaError(
                    "marker/config SHA drifted",
                    path=f"runtime_output.lines[{line_number}].{name}",
                )
        try:
            kind = MoeCalibrationKind(fields["kind"])
        except ValueError as error:
            raise SchemaError(
                "unknown calibration kind",
                path=f"runtime_output.lines[{line_number}].kind",
            ) from error
        if fields["shape"] == "none":
            shape = None
        else:
            pieces = fields["shape"].split("x")
            if len(pieces) != 3:
                raise SchemaError(
                    "shape must be none or MxNxK",
                    path=f"runtime_output.lines[{line_number}].shape",
                )
            shape = tuple(_uint(item, "shape", line_number) for item in pieces)
        if fields["dtype"] == "none":
            dtype = None
        else:
            try:
                dtype = DType(fields["dtype"])
            except ValueError as error:
                raise SchemaError(
                    "unknown dtype",
                    path=f"runtime_output.lines[{line_number}].dtype",
                ) from error
        sample = MoeCalibrationSample(
            kind,
            _uint(fields["sample"], "sample", line_number),
            _uint(fields["repeat"], "repeat", line_number),
            _uint(fields["cycles"], "cycles", line_number),
            shape,  # type: ignore[arg-type]
            dtype,
            tool_sha256,
            hardware_sha256,
            simulation_sha256,
            mapping_sha256,
        )
        sample.validate(f"runtime_output.lines[{line_number}]")
        samples.append(sample)
    ordered = tuple(
        sorted(
            samples,
            key=lambda item: (
                item.kind.value,
                item.shape or (),
                item.sample_index,
                item.repeat_index,
            ),
        )
    )
    return MoeSwizzleCalibrationProfile.create(samples=ordered, **expected_digests)


def _runtime_missing(semantic: dict[str, object]) -> tuple[str, ...]:
    missing = []
    if semantic["compute_dte_overlap_cycles"] is None:
        missing.append("compute_dte_overlap_cycles")
    if len(semantic["die_compute_dte_overlaps"]) != 4:  # type: ignore[arg-type]
        missing.append("die_compute_dte_overlap_over_time")
    if len(semantic["die_session_capacities"]) != 4:  # type: ignore[arg-type]
        missing.append("die_session_capacity")
    if len(semantic["directional_port_times"]) != 8:  # type: ignore[arg-type]
        missing.append("directional_port_utilization_over_time")
    for name in (
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
    ):
        if semantic[name] is None:
            missing.append(name)
    return tuple(sorted(missing))


def parse_moe_swizzle_runtime_markers(output: str) -> MoeSwizzleRuntimeMarkers:
    """Parse strict session/overlap/port/setup evidence from one runtime output."""

    if type(output) is not str:
        raise SchemaError("must be text", path="runtime_output")
    sessions: dict[int, MoeDieSessionCapacity] = {}
    overlap: int | None = None
    die_overlaps: dict[int, MoeDieComputeDteOverlap] = {}
    ports: list[MoeDirectionalPortTime] = []
    setup: tuple[int, int, int, int, int, int, int, int, int] | None = None
    marker_lines = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        prefix = next(
            (
                item
                for item in (_SESSION, _OVERLAP, _PORT, _SETUP)
                if line.startswith(item)
            ),
            None,
        )
        if prefix is None:
            if line.startswith(_DEDICATED_STEM):
                raise SchemaError(
                    "unknown dedicated runtime marker",
                    path=f"runtime_output.lines[{line_number}]",
                )
            continue
        marker_lines.append(line)
        fields = _fields(line, prefix, line_number)
        if prefix == _SESSION:
            _exact(fields, _SESSION_FIELDS, line_number)
            die = _uint(fields["die"], "die", line_number)
            if die > 3 or die in sessions:
                raise SchemaError(
                    "session markers require unique 2x2-mesh dies",
                    path=f"runtime_output.lines[{line_number}].die",
                )
            opens = _uint(fields["opens"], "opens", line_number)
            retires = _uint(fields["retires"], "retires", line_number)
            if opens != retires:
                raise SchemaError(
                    "all sessions must retire",
                    path=f"runtime_output.lines[{line_number}]",
                )
            sessions[die] = (
                MoeDieSessionCapacity(
                    die,
                    _uint(fields["capacity_per_core"], "capacity_per_core", line_number),
                    _uint(fields["active_core_count"], "active_core_count", line_number),
                    _uint(fields["aggregate_capacity"], "aggregate_capacity", line_number),
                    _uint(fields["send_peak"], "send_peak", line_number),
                    _uint(fields["recv_peak"], "recv_peak", line_number),
                )
            )
            sessions[die].validate(f"runtime_output.lines[{line_number}]")
        elif prefix == _OVERLAP:
            _exact(fields, _OVERLAP_FIELDS, line_number)
            scope = fields["scope"]
            values = tuple(
                _uint(fields[name], name, line_number)
                for name in (
                    "compute_cycles", "dte_cycles", "compute_dte_cycles",
                    "window_cycles",
                )
            )
            if scope == "global":
                if fields["die"] != "all" or overlap is not None:
                    raise SchemaError("global overlap marker is malformed/duplicate", path="runtime_output")
                if values[2] > min(values[0], values[1]) or values[0] > values[3] or values[1] > values[3]:
                    raise SchemaError("global overlap totals are inconsistent", path="runtime_output")
                overlap = values[2]
            elif scope == "die":
                die = _uint(fields["die"], "die", line_number)
                item = MoeDieComputeDteOverlap(die, *values)
                item.validate(f"runtime_output.lines[{line_number}]")
                if die in die_overlaps:
                    raise SchemaError("duplicate die overlap marker", path="runtime_output")
                die_overlaps[die] = item
            else:
                raise SchemaError("overlap scope must be die or global", path=f"runtime_output.lines[{line_number}].scope")
        elif prefix == _PORT:
            _exact(fields, _PORT_FIELDS, line_number)
            port = MoeDirectionalPortTime(
                _uint(fields["source_die"], "source_die", line_number),
                _uint(fields["destination_die"], "destination_die", line_number),
                fields["direction"],
                _uint(fields["busy_cycles"], "busy_cycles", line_number),
                _uint(fields["window_cycles"], "window_cycles", line_number),
            )
            port.validate(f"runtime_output.lines[{line_number}]")
            ports.append(port)
        else:
            _exact(fields, _SETUP_FIELDS, line_number)
            if setup is not None:
                raise SchemaError("duplicate setup marker", path="runtime_output")
            setup = tuple(
                _uint(fields[name], name, line_number)
                for name in (
                    "group_gemm_primitives",
                    "group_gemm_setup_cycles",
                    "matmul_total_cycles",
                    "dte_launch_count",
                    "physical_root_count",
                    "event_record_count",
                    "sram_lifecycle_cycles",
                    "bind_cycles",
                    "event_control_cycles",
                )
            )
    ordered_ports = tuple(
        sorted(ports, key=lambda item: (item.source_die, item.destination_die))
    )
    complete_sessions = set(sessions) == {0, 1, 2, 3}
    semantic: dict[str, object] = {
        "marker_digest": hashlib.sha256("\n".join(marker_lines).encode("utf-8")).hexdigest(),
        "observed_max_inflight_send": (
            max(value.send_peak for value in sessions.values()) if complete_sessions else None
        ),
        "observed_max_inflight_recv": (
            max(value.recv_peak for value in sessions.values()) if complete_sessions else None
        ),
        "die_session_capacities": tuple(sessions[index] for index in sorted(sessions)),
        "compute_dte_overlap_cycles": overlap,
        "die_compute_dte_overlaps": tuple(die_overlaps[index] for index in sorted(die_overlaps)),
        "directional_port_times": ordered_ports,
        "group_gemm_primitive_count": (setup[0] if setup is not None else None),
        "group_gemm_setup_cycles": (setup[1] if setup is not None else None),
        "matmul_total_cycles": (setup[2] if setup is not None else None),
        "dte_launch_count": (setup[3] if setup is not None else None),
        "physical_root_count": (setup[4] if setup is not None else None),
        "event_record_count": (setup[5] if setup is not None else None),
        "sram_lifecycle_cycles": (setup[6] if setup is not None else None),
        "bind_cycles": (setup[7] if setup is not None else None),
        "event_control_cycles": (setup[8] if setup is not None else None),
    }
    missing = _runtime_missing(semantic)
    return MoeSwizzleRuntimeMarkers.create(
        measurement_complete=not missing,
        missing_measurements=missing,
        **semantic,
    )


__all__ = [
    "parse_moe_swizzle_calibration",
    "parse_moe_swizzle_runtime_markers",
]
