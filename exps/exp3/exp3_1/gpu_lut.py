#!/usr/bin/env python3
"""Strict Exp3.1 GPU GEMM LUT loader with an explicit placeholder mode."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence
import warnings

import yaml


Shape = tuple[int, int, int]

# Deterministic stand-in for functional testing only.  These values describe
# a 360-TFLOP/s single-card surrogate (36 nominal 10-TFLOP/s compute units)
# and its aggregate 256-GB/s bandwidth.  They are not GPU measurements.
PLACEHOLDER_ROOFLINE = MappingProxyType(
    {
        "peak_flops_per_second": 360.0e12,
        "memory_bandwidth_bytes_per_second": 256.0e9,
        "launch_overhead_ns": 2_000.0,
        "element_bytes": 2,
    }
)
PLACEHOLDER_EVIDENCE = "deterministic_roofline_placeholder"
MEASURED_EVIDENCE = "gpu_p50_measurement"
PLACEHOLDER_DATA_STATUS = "placeholder_analytical"


class GpuLutError(ValueError):
    """Raised when the measurement document violates the LUT contract."""


@dataclass(frozen=True, slots=True)
class GpuLatency:
    shape: Shape
    latency_ns: float
    evidence: str


@dataclass(frozen=True, slots=True)
class GpuLut:
    latencies_ns: Mapping[Shape, float]
    evidence_by_shape: Mapping[Shape, str]
    groups: Mapping[str, tuple[Shape, ...]]
    measurement_sha256: str
    required_sha256: str
    normalized_sha256: str
    warnings: tuple[str, ...]
    placeholder_shapes: frozenset[Shape]
    metadata: Mapping[str, object]

    @property
    def latencies(self) -> Mapping[Shape, float]:
        return self.latencies_ns

    @property
    def uses_placeholders(self) -> bool:
        return bool(self.placeholder_shapes)

    @property
    def evidence(self) -> str:
        if not self.placeholder_shapes:
            return MEASURED_EVIDENCE
        if len(self.placeholder_shapes) == len(self.latencies_ns):
            return PLACEHOLDER_EVIDENCE
        return "mixed_gpu_measurement_and_deterministic_placeholder"

    def lookup(self, shape: Sequence[int]) -> float:
        key = _shape(shape, "lookup.shape")
        try:
            return self.latencies_ns[key]
        except KeyError as error:
            raise KeyError(f"GPU LUT has no exact shape {key}") from error

    def lookup_entry(self, shape: Sequence[int]) -> GpuLatency:
        key = _shape(shape, "lookup.shape")
        return GpuLatency(key, self.lookup(key), self.evidence_by_shape[key])

    def group_latency_ns(
        self, group: str, shape: Sequence[int], execution_count: int = 1
    ) -> float:
        if group not in self.groups:
            raise KeyError(f"unknown required GPU LUT group: {group}")
        key = _shape(shape, "group_latency.shape")
        if key not in self.groups[group]:
            raise KeyError(f"shape {key} is not required by group {group}")
        if type(execution_count) is not int or execution_count <= 0:
            raise ValueError("execution_count must be a positive integer")
        return self.lookup(key) * execution_count

    def __getitem__(self, shape: Sequence[int]) -> float:
        return self.lookup(shape)


def placeholder_latency_ns(shape: Sequence[int]) -> float:
    """Return a deterministic roofline estimate, never a claimed measurement."""

    m, n, k = _shape(shape, "placeholder.shape")
    flops = 2 * m * n * k
    element_bytes = int(PLACEHOLDER_ROOFLINE["element_bytes"])
    traffic_bytes = element_bytes * (m * k + k * n + m * n)
    compute_ns = (
        flops / float(PLACEHOLDER_ROOFLINE["peak_flops_per_second"]) * 1.0e9
    )
    memory_ns = (
        traffic_bytes
        / float(PLACEHOLDER_ROOFLINE["memory_bandwidth_bytes_per_second"])
        * 1.0e9
    )
    latency = float(PLACEHOLDER_ROOFLINE["launch_overhead_ns"]) + max(
        compute_ns, memory_ns
    )
    # Stable text/JSON representations across repeated runs.
    return round(latency, 6)


def _load_document(source: str | Path | Mapping[str, object], name: str) -> tuple[dict[str, Any], bytes]:
    if isinstance(source, Mapping):
        document = dict(source)
        raw = _canonical_json(document)
    else:
        path = Path(source)
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise GpuLutError(f"cannot read {name} YAML {path}: {error}") from error
        try:
            document = yaml.safe_load(raw)
        except yaml.YAMLError as error:
            raise GpuLutError(f"invalid {name} YAML: {error}") from error
    if not isinstance(document, dict):
        raise GpuLutError(f"{name} YAML root must be a mapping")
    return document, raw


def _canonicalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _shape(value: object, path: str) -> Shape:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise GpuLutError(f"{path} must be a three-element [M,N,K] sequence")
    result: list[int] = []
    for index, dimension in enumerate(value):
        if type(dimension) is not int or dimension <= 0:
            raise GpuLutError(f"{path}[{index}] must be a positive integer")
        result.append(dimension)
    return result[0], result[1], result[2]


def _lookup_entries(
    document: Mapping[str, object],
    *,
    name: str,
    allow_null: bool,
) -> tuple[dict[str, tuple[Shape, ...]], dict[Shape, float | None]]:
    lookup = document.get("lookup")
    if not isinstance(lookup, Mapping) or not lookup:
        raise GpuLutError(f"{name}.lookup must be a non-empty mapping")
    grouped: dict[str, tuple[Shape, ...]] = {}
    values: dict[Shape, float | None] = {}
    for raw_group, raw_entries in lookup.items():
        if not isinstance(raw_group, str) or not raw_group:
            raise GpuLutError(f"{name}.lookup group names must be non-empty strings")
        if not isinstance(raw_entries, list):
            raise GpuLutError(f"{name}.lookup.{raw_group} must be a list")
        group_shapes: list[Shape] = []
        for index, raw_entry in enumerate(raw_entries):
            path = f"{name}.lookup.{raw_group}[{index}]"
            if not isinstance(raw_entry, (list, tuple)) or len(raw_entry) != 2:
                raise GpuLutError(f"{path} must be [[M,N,K], latency_ns]")
            shape = _shape(raw_entry[0], f"{path}[0]")
            latency = raw_entry[1]
            if latency is None:
                if not allow_null:
                    raise GpuLutError(f"{path}[1] cannot be null")
                parsed_latency = None
            else:
                if type(latency) not in (int, float):
                    raise GpuLutError(f"{path}[1] must be a positive finite number")
                parsed_latency = float(latency)
                if not math.isfinite(parsed_latency) or parsed_latency <= 0:
                    raise GpuLutError(f"{path}[1] must be a positive finite number")
            if shape in values:
                prior = values[shape]
                if (
                    prior is not None
                    and parsed_latency is not None
                    and prior != parsed_latency
                ):
                    raise GpuLutError(
                        f"duplicate shape {shape} has conflicting latencies "
                        f"{prior} and {parsed_latency}"
                    )
                if prior is None and parsed_latency is not None:
                    values[shape] = parsed_latency
            else:
                values[shape] = parsed_latency
            group_shapes.append(shape)
        grouped[raw_group] = tuple(group_shapes)
    return grouped, values


def _require_mapping(document: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise GpuLutError(f"measurements.{key} must be a mapping")
    return value


def _require_nonempty(mapping: Mapping[str, object], key: str, path: str) -> None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GpuLutError(f"{path}.{key} must be a non-empty string")


def _validate_metadata(document: Mapping[str, object], *, allow_placeholder: bool) -> None:
    data_status = document.get("data_status")
    if data_status == PLACEHOLDER_DATA_STATUS and not allow_placeholder:
        raise GpuLutError(
            "measurements.data_status is placeholder_analytical; "
            "pass allow_placeholder=True to acknowledge non-measured evidence"
        )
    if data_status not in (None, "measured", PLACEHOLDER_DATA_STATUS):
        raise GpuLutError(
            "measurements.data_status must be 'measured' or "
            f"'{PLACEHOLDER_DATA_STATUS}'"
        )
    if document.get("schema_version") != 1:
        raise GpuLutError("measurements.schema_version must equal 1")
    if document.get("units") != "ns":
        raise GpuLutError("measurements.units must be 'ns'")
    if "shape_order" in document and document["shape_order"] != ["M", "N", "K"]:
        raise GpuLutError("measurements.shape_order must be [M, N, K]")

    gpu = _require_mapping(document, "gpu")
    count = gpu.get("count")
    if type(count) is not int or count <= 0:
        raise GpuLutError("measurements.gpu.count must be a positive integer")
    software = _require_mapping(document, "software")
    if not allow_placeholder:
        _require_nonempty(gpu, "name", "measurements.gpu")
        _require_nonempty(gpu, "clock_policy", "measurements.gpu")
        for key in ("driver", "cuda", "cublas_or_backend"):
            _require_nonempty(software, key, "measurements.software")

    gemm = _require_mapping(document, "gemm")
    expected_gemm = {
        "input_dtype": "bf16",
        "output_dtype": "bf16",
        "accumulation_dtype": "fp32",
        "transpose_a": False,
        "transpose_b": False,
    }
    for key, expected in expected_gemm.items():
        if gemm.get(key) != expected:
            raise GpuLutError(f"measurements.gemm.{key} must equal {expected!r}")

    measurement = _require_mapping(document, "measurement")
    fixed_measurement = {
        "statistic": "p50",
        "synchronization": "per_iteration",
        "operands_resident_on_device": True,
        "includes_host_to_device": False,
        "includes_device_to_host": False,
    }
    for key, expected in fixed_measurement.items():
        if measurement.get(key) != expected:
            raise GpuLutError(
                f"measurements.measurement.{key} must equal {expected!r}"
            )
    for key, minimum in (("warmup_iterations", 0), ("measured_iterations", 1)):
        value = measurement.get(key)
        if allow_placeholder and value is None:
            continue
        if type(value) is not int or value < minimum:
            raise GpuLutError(
                f"measurements.measurement.{key} must be an integer >= {minimum}"
            )


def _missing_message(
    missing: set[Shape], required_document: Mapping[str, object]
) -> str:
    references: dict[Shape, list[str]] = {}
    raw_refs = required_document.get("shape_references", [])
    if isinstance(raw_refs, list):
        for index, raw_reference in enumerate(raw_refs):
            if not isinstance(raw_reference, Mapping) or "shape" not in raw_reference:
                continue
            try:
                shape = _shape(raw_reference["shape"], f"shape_references[{index}].shape")
            except GpuLutError:
                continue
            references.setdefault(shape, []).append(
                str(raw_reference.get("case_id", "unknown"))
            )
    details = []
    for shape in sorted(missing):
        source = references.get(shape, [])
        details.append(f"{shape} (cases: {', '.join(source[:3]) or 'unknown'})")
    return "GPU LUT is missing required exact shapes: " + "; ".join(details)


def load_gpu_lut(
    required: str | Path | Mapping[str, object],
    measurements: str | Path | Mapping[str, object] | None = None,
    *,
    allow_placeholder: bool = False,
    strict_extra_shapes: bool = False,
) -> GpuLut:
    """Load and exactly cover the required shapes.

    When ``measurements`` is omitted, ``required`` is also used as the
    measurement document.  This is the intended smoke-test path for a freshly
    generated template with ``null`` latency values and
    ``allow_placeholder=True``.
    """

    required_document, required_raw = _load_document(required, "required")
    if measurements is None:
        measurement_document, measurement_raw = required_document, required_raw
    else:
        measurement_document, measurement_raw = _load_document(
            measurements, "measurements"
        )
    _validate_metadata(measurement_document, allow_placeholder=allow_placeholder)
    required_groups, required_values = _lookup_entries(
        required_document, name="required", allow_null=True
    )
    _, measurement_values = _lookup_entries(
        measurement_document,
        name="measurements",
        allow_null=allow_placeholder,
    )

    required_shapes = set(required_values)
    measured_shapes = set(measurement_values)
    missing = required_shapes - measured_shapes
    if missing:
        raise GpuLutError(_missing_message(missing, required_document))
    extra = measured_shapes - required_shapes
    messages: list[str] = []
    if extra:
        message = "GPU LUT contains extra shapes: " + ", ".join(
            str(shape) for shape in sorted(extra)
        )
        if strict_extra_shapes:
            raise GpuLutError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        messages.append(message)

    materialized_placeholder = (
        measurement_document.get("data_status") == PLACEHOLDER_DATA_STATUS
    )
    latencies: dict[Shape, float] = {}
    evidence: dict[Shape, str] = {}
    placeholders: set[Shape] = set()
    for shape, value in measurement_values.items():
        if value is None:
            # Nulls only survive parsing when the explicit opt-in is active.
            value = placeholder_latency_ns(shape)
            evidence[shape] = PLACEHOLDER_EVIDENCE
            placeholders.add(shape)
        elif materialized_placeholder:
            evidence[shape] = PLACEHOLDER_EVIDENCE
            placeholders.add(shape)
        else:
            evidence[shape] = MEASURED_EVIDENCE
        latencies[shape] = value

    normalized = {
        "schema_version": 1,
        "units": "ns",
        "lookup": [
            {
                "shape": list(shape),
                "latency_ns": latencies[shape],
                "evidence": evidence[shape],
            }
            for shape in sorted(latencies)
        ],
        "placeholder_model": (
            dict(PLACEHOLDER_ROOFLINE) if placeholders else None
        ),
    }
    metadata: dict[str, object] = {
        "gpu": dict(_require_mapping(measurement_document, "gpu")),
        "software": dict(_require_mapping(measurement_document, "software")),
        "gemm": dict(_require_mapping(measurement_document, "gemm")),
        "measurement": dict(
            _require_mapping(measurement_document, "measurement")
        ),
        "evidence": (
            PLACEHOLDER_EVIDENCE if placeholders else MEASURED_EVIDENCE
        ),
        "placeholder_model": dict(PLACEHOLDER_ROOFLINE) if placeholders else None,
    }
    return GpuLut(
        latencies_ns=MappingProxyType(latencies),
        evidence_by_shape=MappingProxyType(evidence),
        groups=MappingProxyType(required_groups),
        measurement_sha256=_digest(measurement_raw),
        required_sha256=_digest(required_raw),
        normalized_sha256=_digest(_canonical_json(normalized)),
        warnings=tuple(messages),
        placeholder_shapes=frozenset(placeholders),
        metadata=MappingProxyType(metadata),
    )


def materialize_placeholder_document(
    required: str | Path | Mapping[str, object],
) -> dict[str, object]:
    """Build a numeric, explicitly non-measured LUT from a required template.

    Numeric placeholder files are convenient for end-to-end smoke tests, but
    ``data_status=placeholder_analytical`` makes their evidence boundary
    machine-checkable.  The strict loader refuses such a file unless the caller
    explicitly opts in with ``allow_placeholder=True``.
    """

    document, raw = _load_document(required, "required")
    _validate_metadata(document, allow_placeholder=True)
    groups, _ = _lookup_entries(document, name="required", allow_null=True)
    result = yaml.safe_load(yaml.safe_dump(document, sort_keys=False))
    if not isinstance(result, dict):
        raise AssertionError("placeholder copy unexpectedly changed YAML root type")
    result["data_status"] = PLACEHOLDER_DATA_STATUS
    result["evidence"] = {
        "kind": PLACEHOLDER_EVIDENCE,
        "is_gpu_measurement": False,
        "purpose": "functional_pipeline_validation_only",
    }
    result["placeholder_model"] = dict(PLACEHOLDER_ROOFLINE)
    result["materialized_from_required_sha256"] = _digest(raw)
    lookup = result.get("lookup")
    if not isinstance(lookup, dict):
        raise AssertionError("validated lookup unexpectedly changed type")
    for group, shapes in groups.items():
        entries = lookup[group]
        if not isinstance(entries, list) or len(entries) != len(shapes):
            raise AssertionError("placeholder lookup copy changed entry count")
        for entry, shape in zip(entries, shapes):
            entry[1] = placeholder_latency_ns(shape)
    return result


def write_placeholder_yaml(
    required: str | Path | Mapping[str, object], output: str | Path
) -> Path:
    """Materialize and write a deterministic analytical placeholder YAML."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(
            materialize_placeholder_document(required),
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        ),
        encoding="utf-8",
    )
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--required", type=Path, required=True)
    validate.add_argument("--measurements", type=Path)
    validate.add_argument("--allow-placeholder", action="store_true")
    validate.add_argument("--strict-extra-shapes", action="store_true")
    materialize = subparsers.add_parser("materialize-placeholder")
    materialize.add_argument("--required", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "materialize-placeholder":
        output = write_placeholder_yaml(args.required, args.output)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "data_status": PLACEHOLDER_DATA_STATUS,
                    "evidence": PLACEHOLDER_EVIDENCE,
                    "output": str(output),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    lut = load_gpu_lut(
        args.required,
        args.measurements,
        allow_placeholder=args.allow_placeholder,
        strict_extra_shapes=args.strict_extra_shapes,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "shape_count": len(lut.latencies_ns),
                "placeholder_count": len(lut.placeholder_shapes),
                "evidence": lut.evidence,
                "measurement_sha256": lut.measurement_sha256,
                "required_sha256": lut.required_sha256,
                "normalized_sha256": lut.normalized_sha256,
                "warnings": list(lut.warnings),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
