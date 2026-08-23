"""Transactional N7 runner for the frozen naive Dense timing path.

The runner is deliberately stricter than a convenience shell script.  It
binds one validated frontend compilation to the exact finalizer output and
ProgramIo sidecar, runs the simulator twice, validates its machine-readable
protocol markers, and publishes the complete result directory atomically.

E1/E2 are timing gates.  They prove compilation, address/lifecycle closure,
transport/control completion and deterministic makespan; they do not claim
that the current timing-only compute primitives produce Dense numerical data.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Mapping

from .compiler import NaiveCompilation, compile_naive
from .errors import FrontendError, SchemaError, UnsupportedFeatureError
from .passes.pass_manager import PASS_SPECS, PassReceipt
from .policies.registry import PolicyRegistry
from .passes import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
    load_physical_fabric_and_hbm_address_spaces,
)
from .passes.load_fabric import SIMULATOR_PACKET_PAYLOAD_BYTES
from .schema.artifact_manifest import RegionManifest
from .schema.common import stable_artifact_id, validate_nonempty
from .schema.ir0 import AttentionWorkload, GemmWorkload
from .schema.n6 import LinkedProgramProfile
from .schema.policy import PolicySelection, RegistryKind
from .schema.intra_die_refine import (
    IntraDieOptimizationOptions, RefinedIR2Bundle, SplitKRefineOptions,
)
from .schema.intra_die_v2_calibration import IntraDieV2CalibrationEvidence
from .schema.intra_die_v2_search import IntraDieV2SearchDecision
from .schema.serde import (
    canonical_digest,
    canonical_json,
    from_data,
    load_json_value,
    to_primitive,
)


NAIVE_RUN_REPORT_SCHEMA_VERSION = "wafer_frontend.naive_run_report/v1alpha3"

_PROGRAM_IO_PREFIX = "[PROGRAM_IO] "
_PROGRAM_IO_PROBE_PREFIX = "[PROGRAM_IO_PROBE] "
_SIM_RESULT_PREFIX = "[SIM_RESULT] "
_HOST_PREFIX = "[HOSTLANE] "
_HOSTSIG_PREFIX = "[HOSTSIG] "
_P5_DRAIN_PREFIX = "[P5 P2P DRAIN] "
_P5_TIMING_PREFIX = "[P5 P2P TIMING DRAIN] "
_COLL_DRAIN_PREFIX = "[COLL_DRAIN] "
_GLOBAL_DRAIN_PREFIX = "[DRAIN] "
_D2D_TYPE_PREFIX = "[D2D_TYPE] "
_D2D_LINK_PREFIX = "[D2D_LINK] "
_MARKER_PREFIXES = (
    _PROGRAM_IO_PREFIX,
    _PROGRAM_IO_PROBE_PREFIX,
    _SIM_RESULT_PREFIX,
    _HOST_PREFIX,
    _HOSTSIG_PREFIX,
    _P5_DRAIN_PREFIX,
    _P5_TIMING_PREFIX,
    _COLL_DRAIN_PREFIX,
    _GLOBAL_DRAIN_PREFIX,
    _D2D_TYPE_PREFIX,
    _D2D_LINK_PREFIX,
)


class NaiveRunError(FrontendError):
    """A production run failed before its output directory was committed."""

    default_code = "naive_run_error"


class NaiveRunCase(str, Enum):
    E0 = "E0"
    E1 = "E1"
    E2 = "E2"


class NaiveRunValidation(str, Enum):
    TIMING = "timing"
    REDUCTION_U3F = "reduction-u3f"


@dataclass(frozen=True, slots=True)
class NaiveRunRequest:
    case: NaiveRunCase
    validation: NaiveRunValidation
    spec_path: Path
    hardware_config_path: Path
    simulation_config_path: Path
    mapping_config_path: Path
    output_dir: Path
    npusim_path: Path
    finalizer_path: Path
    profile_id: str | None = None
    trace_window: int = 1_000_000
    timeout_seconds: int = 300
    repeat: int = 2
    keep_failed: bool = False
    intra_die_refine_options: SplitKRefineOptions | IntraDieOptimizationOptions | None = None

    def validate(self, path: str = "request") -> None:
        if type(self.case) is not NaiveRunCase:
            raise SchemaError("must be a NaiveRunCase", path=f"{path}.case")
        if type(self.validation) is not NaiveRunValidation:
            raise SchemaError(
                "must be a NaiveRunValidation", path=f"{path}.validation"
            )
        if self.case is NaiveRunCase.E0:
            raise UnsupportedFeatureError(
                "E0 uses the independent non-zero reduction-u3f backend gate; "
                "it cannot reuse the timing ProgramIo runner",
                path=f"{path}.case",
            )
        if self.validation is not NaiveRunValidation.TIMING:
            raise UnsupportedFeatureError(
                "E1/E2 support timing validation only",
                path=f"{path}.validation",
            )
        for field_name in (
            "spec_path",
            "hardware_config_path",
            "simulation_config_path",
            "mapping_config_path",
            "output_dir",
            "npusim_path",
            "finalizer_path",
        ):
            if not isinstance(getattr(self, field_name), Path):
                raise SchemaError("must be a pathlib.Path", path=f"{path}.{field_name}")
        for field_name in (
            "spec_path",
            "hardware_config_path",
            "simulation_config_path",
            "mapping_config_path",
        ):
            source = getattr(self, field_name)
            if not source.is_file():
                raise SchemaError("must identify an existing file", path=f"{path}.{field_name}")
        for field_name in ("npusim_path", "finalizer_path"):
            executable = getattr(self, field_name)
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise SchemaError(
                    "must identify an executable file", path=f"{path}.{field_name}"
                )
        if self.output_dir.exists() or self.output_dir.is_symlink():
            raise SchemaError(
                "output directory must not already exist", path=f"{path}.output_dir"
            )
        if self.profile_id is not None:
            validate_nonempty(self.profile_id, f"{path}.profile_id")
        for field_name in ("trace_window", "timeout_seconds"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise SchemaError("must be a positive integer", path=f"{path}.{field_name}")
        if self.repeat not in (2, 3):
            raise SchemaError(
                "the timing gate requires exactly two or three runs",
                path=f"{path}.repeat",
            )
        if type(self.keep_failed) is not bool:
            raise SchemaError("must be bool", path=f"{path}.keep_failed")
        if self.intra_die_refine_options is not None:
            if type(self.intra_die_refine_options) not in (
                SplitKRefineOptions, IntraDieOptimizationOptions,
            ):
                raise SchemaError(
                    "must be SplitKRefineOptions, IntraDieOptimizationOptions, or None",
                    path=f"{path}.intra_die_refine_options",
                )
            self.intra_die_refine_options.validate(
                f"{path}.intra_die_refine_options"
            )


@dataclass(frozen=True, slots=True)
class NaiveRunReport:
    schema_version: str
    producer_pass: str
    id: str
    status: str
    case: NaiveRunCase
    validation_mode: NaiveRunValidation
    # ``object`` lets the shared strict decoder materialize the nested JSON;
    # validate() below then closes every nested field set explicitly.
    inputs: object
    tools: object
    provenance: object
    artifact: object
    static_metrics: object
    runtime: object
    validation: object

    @classmethod
    def create(
        cls,
        *,
        case: NaiveRunCase,
        validation_mode: NaiveRunValidation,
        inputs: dict[str, object],
        tools: dict[str, object],
        provenance: dict[str, object],
        artifact: dict[str, object],
        static_metrics: dict[str, object],
        runtime: dict[str, object],
        validation: dict[str, object],
    ) -> "NaiveRunReport":
        semantic_key = {
            "status": "pass",
            "case": case,
            "validation_mode": validation_mode,
            "inputs": inputs,
            "tools": tools,
            "provenance": provenance,
            "artifact": artifact,
            "static_metrics": static_metrics,
            "runtime": runtime,
            "validation": validation,
        }
        return cls(
            schema_version=NAIVE_RUN_REPORT_SCHEMA_VERSION,
            producer_pass="naive_runner",
            id=stable_artifact_id(
                "naive_run_report",
                semantic_key,
                schema_version=NAIVE_RUN_REPORT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "status",
                "case",
                "validation_mode",
                "inputs",
                "tools",
                "provenance",
                "artifact",
                "static_metrics",
                "runtime",
                "validation",
            )
        }

    def validate(self, path: str = "naive_run_report") -> None:
        if self.schema_version != NAIVE_RUN_REPORT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "naive_runner" or self.status != "pass":
            raise SchemaError("must be a successful naive_runner report", path=path)
        if type(self.case) is not NaiveRunCase:
            raise SchemaError("must be a NaiveRunCase", path=f"{path}.case")
        if type(self.validation_mode) is not NaiveRunValidation:
            raise SchemaError(
                "must be a NaiveRunValidation", path=f"{path}.validation_mode"
            )
        expected_keys = {
            "inputs": {
                "spec_digest",
                "fabric_digest",
                "hardware_sha256",
                "simulation_sha256",
                "mapping_sha256",
            },
            "tools": {
                "finalizer_sha256",
                "npusim_sha256",
                "linked_manifest_schema",
                "program_io_schema",
            },
            "provenance": {
                "profile_id",
                "profile_weight",
                "pass_receipts",
                "stage_digests",
                "context_ids",
                "policy_selections",
                "linked_bundle_id",
                "linked_profile_id",
                "linked_manifest_id",
                "linked_manifest_digest",
            },
            "artifact": {
                "artifact_sha256",
                "artifact_bytes",
                "core_count",
                "record_count",
                "relocation_count",
                "finalizer_report_digest",
                "program_io_id",
                "program_io_digest",
            },
            "static_metrics": {
                "op_counts",
                "task_counts",
                "action_counts",
                "unique_flow_count",
                "opcode_counts",
                "fragment_count",
                "record_count",
                "rank_gemm_flops",
                "rank_attention_matmul_flops",
                "analytic_transfer_bytes",
                "scheduled_binding_count",
                "per_core_sram_max_end",
            },
            "runtime": {
                "repeat",
                "makespan_cycles",
                "ack_total",
                "done_total",
                "ack_by_core",
                "done_by_core",
                "drain_residuals",
                "credit_balanced",
                "program_io_phases",
                "program_io_initializations",
                "program_io_probes",
                "repeat_signature_stable",
                "observed_transfer_bytes",
                "d2d_link_packets",
            },
            "validation": {
                "timing",
                "address_lifecycle",
                "transport_control",
                "compute_functional",
                "reduction_u3f",
                "end_to_end_functional",
                "capability_notes",
            },
        }
        for field_name, keys in expected_keys.items():
            value = getattr(self, field_name)
            if type(value) is not dict or set(value) != keys:
                raise SchemaError(
                    "must contain the exact report field set",
                    path=f"{path}.{field_name}",
                )
        assert isinstance(self.provenance, dict)
        policy_selections = _decode_policy_selection_rows(
            self.provenance["policy_selections"],
            f"{path}.provenance.policy_selections",
        )
        if tuple(selection.kind for selection in policy_selections) != (
            RegistryKind.INTER_DIE,
            RegistryKind.STANDALONE_COLLECTIVE,
            RegistryKind.INTRA_DIE,
        ):
            raise SchemaError(
                "must contain the exact inter/standalone/intra policy order",
                path=f"{path}.provenance.policy_selections",
            )
        raw_receipts = self.provenance["pass_receipts"]
        if not isinstance(raw_receipts, (tuple, list)):
            raise SchemaError(
                "must be an ordered array",
                path=f"{path}.provenance.pass_receipts",
            )
        receipts = tuple(
            from_data(
                PassReceipt,
                raw_receipt,
                path=f"{path}.provenance.pass_receipts[{index}]",
            )
            for index, raw_receipt in enumerate(raw_receipts)
        )
        if (
            len(receipts) != 11
            or tuple(receipt.pass_name for receipt in receipts)
            != tuple(spec.name for spec in PASS_SPECS[:11])
        ):
            raise SchemaError(
                "must contain the eleven fixed compile receipts in order",
                path=f"{path}.provenance.pass_receipts",
            )
        raw_stage_digests = self.provenance["stage_digests"]
        if not isinstance(raw_stage_digests, (tuple, list)):
            raise SchemaError(
                "must be an ordered array",
                path=f"{path}.provenance.stage_digests",
            )
        stage_digests = tuple(raw_stage_digests)
        if len(stage_digests) != 12:
            raise SchemaError(
                "must contain twelve adjacent artifact digests",
                path=f"{path}.provenance.stage_digests",
            )
        for index, digest in enumerate(stage_digests):
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise SchemaError(
                    "must be a lowercase SHA-256 digest",
                    path=f"{path}.provenance.stage_digests[{index}]",
                )
        for index, receipt in enumerate(receipts):
            if (
                receipt.input_digest != stage_digests[index]
                or receipt.output_digest != stage_digests[index + 1]
            ):
                raise SchemaError(
                    "receipt does not identify adjacent stage digests",
                    path=f"{path}.provenance.pass_receipts[{index}]",
                )
        receipt_selections = (
            *receipts[4].policy_selections,
            *receipts[7].policy_selections,
        )
        if receipt_selections != policy_selections:
            raise SchemaError(
                "policy summary disagrees with compile receipts",
                path=f"{path}.provenance.policy_selections",
            )
        context_ids = self.provenance["context_ids"]
        if (
            not isinstance(context_ids, (tuple, list))
            or len(context_ids) != 6
            or any(type(value) is not str or not value for value in context_ids)
        ):
            raise SchemaError(
                "must contain the six non-empty context ids",
                path=f"{path}.provenance.context_ids",
            )
        assert isinstance(self.runtime, dict)
        if self.runtime.get("makespan_cycles", 0) <= 0:
            raise SchemaError(
                "must contain a positive stable makespan",
                path=f"{path}.runtime.makespan_cycles",
            )
        expected_id = stable_artifact_id(
            "naive_run_report",
            self._semantic_key(),
            schema_version=NAIVE_RUN_REPORT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable report id; expected {expected_id!r}", path=f"{path}.id"
            )


@dataclass(frozen=True, slots=True)
class NaiveRunResult:
    report_id: str
    output_dir: Path
    report_path: Path
    report: NaiveRunReport


@dataclass(frozen=True, slots=True)
class _RuntimeEvidence:
    makespan_cycles: int
    ack_total: int
    done_total: int
    ack_by_core: tuple[tuple[int, int], ...]
    done_by_core: tuple[tuple[int, int], ...]
    observed_transfer_bytes: int
    d2d_link_packets: tuple[tuple[int, int], ...]
    marker_signature: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise NaiveRunError(str(error), path=str(path)) from error


def _write_json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _build_intra_die_v2_calibration_evidence(
    decisions: tuple[IntraDieV2SearchDecision, ...],
    *,
    simulator_measured_makespan_cycles: int,
    simulator_calls_used: int,
    repeat_signature_stable: bool,
) -> tuple[IntraDieV2CalibrationEvidence, ...]:
    """Bind analytic predictions to the already-reserved final timing runs."""

    evidence = tuple(
        IntraDieV2CalibrationEvidence.create(
            decision,
            simulator_measured_makespan_cycles=simulator_measured_makespan_cycles,
            simulator_calls_used=simulator_calls_used,
            repeat_signature_stable=repeat_signature_stable,
        )
        for decision in decisions
    )
    for index, (item, decision) in enumerate(zip(evidence, decisions)):
        item.validate_against(
            decision, path=f"intra_die_v2_calibration_evidence[{index}]"
        )
    return evidence


def _write_intra_die_v2_calibration_evidence(
    path: Path,
    decisions: tuple[IntraDieV2SearchDecision, ...],
    *,
    simulator_measured_makespan_cycles: int,
    simulator_calls_used: int,
    repeat_signature_stable: bool,
) -> tuple[IntraDieV2CalibrationEvidence, ...]:
    evidence = _build_intra_die_v2_calibration_evidence(
        decisions,
        simulator_measured_makespan_cycles=simulator_measured_makespan_cycles,
        simulator_calls_used=simulator_calls_used,
        repeat_signature_stable=repeat_signature_stable,
    )
    if evidence:
        _write_json(path, evidence)
    return evidence


def _intra_die_v2_calibration_notes(
    evidence: tuple[IntraDieV2CalibrationEvidence, ...],
) -> tuple[str, ...]:
    if not evidence:
        return ()
    notes = [
        "intra-die v2 calibration evidence: "
        + ", ".join(
            f"{item.id}:calibrated={str(item.calibrated).lower()}:"
            f"relative_error={item.relative_error:.6f}"
            for item in evidence
        )
    ]
    if any(not item.calibrated for item in evidence):
        notes.append(
            "calibrated=false: this timing-only evidence explicitly forbids "
            "paper-grade performance claims"
        )
    return tuple(notes)


def _copy_input(source: Path, destination: Path) -> str:
    try:
        payload = source.read_bytes()
        destination.write_bytes(payload)
    except OSError as error:
        raise NaiveRunError(str(error), path=str(source)) from error
    return _sha256_bytes(payload)


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        raise NaiveRunError(
            f"command timed out after {timeout}s: {command[0]}", path="runtime"
        ) from error
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        diagnostic = (completed.stdout + "\n" + completed.stderr)[-4000:]
        raise NaiveRunError(
            f"command failed with exit {completed.returncode}: {command[0]}\n{diagnostic}",
            path="runtime",
        )
    return completed


def _row(line: str, prefix: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line[len(prefix) :].strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def _rows(output: str, prefix: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position < 0:
            continue
        normalized = line[position:].split(" | ", 1)[0].rstrip()
        if normalized.endswith("."):
            normalized = normalized[:-1]
        result.append(_row(normalized, prefix))
    return result


def _number(row: Mapping[str, str], key: str, path: str) -> int:
    try:
        return int(row[key], 10)
    except (KeyError, ValueError) as error:
        raise NaiveRunError(
            f"missing or non-decimal {key!r} in {dict(row)!r}", path=path
        ) from error


def _signature_counts(value: str, arity: int, path: str) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    try:
        for item in value.rstrip(",").split(","):
            if item:
                parts = tuple(int(part, 10) for part in item.split(":"))
                if len(parts) != arity:
                    raise ValueError(item)
                result.append(parts)
    except ValueError as error:
        raise NaiveRunError(f"malformed runtime signature {value!r}", path=path) from error
    return sorted(result)


def _marker_signature(output: str) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    result: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for prefix in _MARKER_PREFIXES:
        result.extend((prefix, tuple(sorted(row.items()))) for row in _rows(output, prefix))
    return tuple(result)


def _expected_runtime_cores(
    source: LinkedProgramProfile,
    field_name: str,
) -> tuple[int, ...]:
    runtime = {
        binding.logical_core: binding.runtime_core_id
        for binding in source.manifest.core_bindings
    }
    try:
        return tuple(runtime[core] for core in getattr(source.manifest.envelope, field_name))
    except KeyError as error:
        raise NaiveRunError(
            "control envelope references a core without runtime binding",
            path=f"manifest.envelope.{field_name}",
        ) from error


def _validate_runtime(
    output: str,
    *,
    source: LinkedProgramProfile,
    artifact_sha256: str,
    initialization_count: int,
    probe_count: int,
) -> _RuntimeEvidence:
    if "[PROTO_WAIT]" in output or "End DONE reception" not in output:
        raise NaiveRunError("runtime did not close the DONE path", path="runtime")
    statuses = _rows(output, _PROGRAM_IO_PREFIX)
    if len(statuses) != 3 or [row.get("phase") for row in statuses] != [
        "resolved",
        "applied",
        "verify",
    ]:
        raise NaiveRunError(
            f"wrong ProgramIo phase closure: {statuses!r}", path="runtime.program_io"
        )
    for row in statuses:
        if (
            row.get("mode") != "timing"
            or _number(row, "initializations", "runtime.program_io") != initialization_count
            or _number(row, "probes", "runtime.program_io") != probe_count
            or row.get("pass") != "1"
        ):
            raise NaiveRunError(
                f"failed ProgramIo status: {row!r}", path="runtime.program_io"
            )
    if statuses[0].get("checksum") != artifact_sha256 or statuses[1].get(
        "checksum"
    ) != artifact_sha256:
        raise NaiveRunError(
            "ProgramIo checksum is not the actual artifact SHA-256",
            path="runtime.program_io.checksum",
        )
    probes = _rows(output, _PROGRAM_IO_PROBE_PREFIX)
    if len(probes) != probe_count:
        raise NaiveRunError(
            f"expected {probe_count} probes, got {len(probes)}",
            path="runtime.program_io.probes",
        )
    for probe in probes:
        if (
            probe.get("valid") != "1"
            or probe.get("exact") != "1"
            or probe.get("pass") != "1"
            or probe.get("checksum") != probe.get("expected_checksum")
        ):
            raise NaiveRunError(
                f"failed output probe: {probe!r}", path="runtime.program_io.probes"
            )

    simulation = _rows(output, _SIM_RESULT_PREFIX)
    if len(simulation) != 1:
        raise NaiveRunError(
            "SIM_RESULT marker must appear exactly once", path="runtime.makespan"
        )
    makespan = _number(simulation[0], "makespan_cycles", "runtime.makespan")
    if makespan <= 0:
        raise NaiveRunError("makespan must be positive", path="runtime.makespan")

    host = _rows(output, _HOST_PREFIX)
    signatures = _rows(output, _HOSTSIG_PREFIX)
    if len(host) != 1 or len(signatures) != 1:
        raise NaiveRunError(
            "HOSTLANE/HOSTSIG marker missing or duplicated", path="runtime.control"
        )
    ack_total = _number(host[0], "ack_total", "runtime.control")
    done_total = _number(host[0], "done_total", "runtime.control")
    if _number(host[0], "mismatch", "runtime.control") != 0:
        raise NaiveRunError("host observed a control mismatch", path="runtime.control")
    done_items = _signature_counts(signatures[0].get("done", ""), 2, "runtime.control.done")
    ack_items = _signature_counts(signatures[0].get("ack", ""), 3, "runtime.control.ack")
    done_by_core: Counter[int] = Counter()
    ack_by_core: Counter[int] = Counter()
    for core, count in done_items:
        done_by_core[core] += count
    for core, _lane, count in ack_items:
        ack_by_core[core] += count
    expected_done = _expected_runtime_cores(source, "expected_done_cores")
    expected_ack = _expected_runtime_cores(source, "expected_ack_cores")
    if done_total != len(expected_done) or dict(done_by_core) != {
        core: 1 for core in expected_done
    }:
        raise NaiveRunError(
            f"wrong DONE closure: {dict(done_by_core)!r}", path="runtime.control.done"
        )
    # Backend-v1 reports one ACK on each of its two host lanes per expected
    # core.  This is a frozen runtime protocol fact, not a count guessed from
    # logs after the run.
    if ack_total != 2 * len(expected_ack) or dict(ack_by_core) != {
        core: 2 for core in expected_ack
    }:
        raise NaiveRunError(
            f"wrong ACK closure: {dict(ack_by_core)!r}", path="runtime.control.ack"
        )

    p5 = _rows(output, _P5_DRAIN_PREFIX)
    active_runtime = set(_expected_runtime_cores(source, "active_cores"))
    if (
        len(p5) != len(active_runtime)
        or {_number(row, "core", "runtime.drain.p2p") for row in p5} != active_runtime
        or any(_number(row, "residual", "runtime.drain.p2p") != 0 for row in p5)
    ):
        raise NaiveRunError(f"P2P endpoint drain is incomplete: {p5!r}", path="runtime.drain")
    timing = _rows(output, _P5_TIMING_PREFIX)
    if len(timing) != 1 or _number(timing[0], "residual", "runtime.drain.timing") != 0:
        raise NaiveRunError("P2P timing sideband did not drain", path="runtime.drain")
    collective = _rows(output, _COLL_DRAIN_PREFIX)
    if len(collective) != 1:
        raise NaiveRunError("collective drain marker missing or duplicated", path="runtime.drain")
    for key in (
        "tree_entries",
        "reduce_nodes",
        "barriers",
        "gather",
        "reduce_rx",
        "endpoints",
        "dte_tokens",
        "event",
    ):
        if _number(collective[0], key, "runtime.drain.collective") != 0:
            raise NaiveRunError(f"collective drain {key} is non-zero", path="runtime.drain")
    global_rows = _rows(output, _GLOBAL_DRAIN_PREFIX)
    residuals = {
        key: _number(row, key, "runtime.drain.global")
        for row in global_rows
        for key in ("router_residual", "d2d_link_residual")
        if key in row
    }
    if residuals != {"router_residual": 0, "d2d_link_residual": 0}:
        raise NaiveRunError(
            f"router/D2D drains are incomplete: {residuals!r}", path="runtime.drain"
        )
    d2d = _rows(output, _D2D_TYPE_PREFIX)
    if len(d2d) != 1:
        raise NaiveRunError(
            "D2D_TYPE marker missing or duplicated", path="runtime.d2d"
        )
    data_in_packets = _number(d2d[0], "data_in", "runtime.d2d")
    data_out_packets = _number(d2d[0], "data_out", "runtime.d2d")
    if data_in_packets != data_out_packets:
        raise NaiveRunError("D2D packet accounting is unbalanced", path="runtime.d2d")
    link_rows = _rows(output, _D2D_LINK_PREFIX)
    expected_link_count = len(source.lowering_context.ir1.fabric.links)
    link_packets = tuple(
        sorted(
            (
                _number(row, "idx", "runtime.d2d.links"),
                _number(row, "data_out", "runtime.d2d.links"),
            )
            for row in link_rows
        )
    )
    if (
        len(link_packets) != expected_link_count
        or tuple(index for index, _packets in link_packets)
        != tuple(range(expected_link_count))
        or any(packets <= 0 for _index, packets in link_packets)
        or sum(packets for _index, packets in link_packets) != data_out_packets
    ):
        raise NaiveRunError(
            f"directed D2D link accounting is incomplete: {link_packets!r}",
            path="runtime.d2d.links",
        )
    return _RuntimeEvidence(
        makespan_cycles=makespan,
        ack_total=ack_total,
        done_total=done_total,
        ack_by_core=tuple(sorted(ack_by_core.items())),
        done_by_core=tuple(sorted(done_by_core.items())),
        observed_transfer_bytes=(
            data_out_packets * SIMULATOR_PACKET_PAYLOAD_BYTES
        ),
        d2d_link_packets=link_packets,
        marker_signature=_marker_signature(output),
    )


def _select_profile(
    compilation: NaiveCompilation,
    profile_id: str | None,
) -> LinkedProgramProfile:
    entries = compilation.linked.entries
    if profile_id is None:
        if len(entries) != 1:
            raise SchemaError(
                "multi-profile compilation requires explicit profile_id",
                path="request.profile_id",
            )
        return entries[0]
    matches = tuple(entry for entry in entries if entry.profile_id == profile_id)
    if len(matches) != 1:
        raise SchemaError(
            "does not select exactly one linked profile", path="request.profile_id"
        )
    return matches[0]


def _validate_case_identity(
    request: NaiveRunRequest,
    compilation: NaiveCompilation,
) -> None:
    instance = compilation.spec.parallel.instances[0]
    expected = {
        NaiveRunCase.E1: (2, 1, (2, 1)),
        NaiveRunCase.E2: (4, 2, (2, 2)),
    }[request.case]
    actual = (instance.tp, compilation.spec.model.L, compilation.fabric.die_grid)
    if actual != expected:
        raise SchemaError(
            f"{request.case.value} requires (tp,layers,die_grid)={expected!r}, got {actual!r}",
            path="request.case",
        )


def _static_metrics(source: LinkedProgramProfile) -> dict[str, object]:
    context = source.lowering_context
    task_counts: Counter[str] = Counter()
    flow_by_id: dict[str, tuple[object, ...]] = {}
    for dag in context.projection.dags:
        task_counts.update(task.kind.value for task in dag.tasks)
        for flow in dag.flows:
            # task_ids are deliberately local to each die replica.  Every
            # other field is the canonical cross-die flow identity/payload.
            semantic = (
                flow.logical_channel,
                flow.pair_route_ref,
                flow.source_rank,
                flow.destination_rank,
                flow.source_die,
                flow.destination_die,
                flow.die_path,
                flow.tensor_slice,
                flow.bytes,
                flow.dtype,
            )
            previous = flow_by_id.setdefault(flow.id, semantic)
            if previous != semantic:
                raise NaiveRunError("conflicting replicated SemanticFlow", path="projection.dags")
    action_counts = Counter(action.task_kind.value for action in context.global_dag.actions)
    op_counts = Counter(node.kind.value for node in context.ir1.nodes)
    rank_gemm_flops = 0
    rank_attention_matmul_flops = 0
    for node in context.ir1.nodes:
        workload = node.workload
        if type(workload) is GemmWorkload:
            m, n, k = workload.rank_shape
            rank_gemm_flops += 2 * m * n * k
        elif type(workload) is AttentionWorkload:
            rank_attention_matmul_flops += (
                4
                * workload.rank_num_heads
                * workload.head_dim
                * workload.query_key_pairs
            )
    transfer_bytes = sum(
        semantic[8] * (len(semantic[6]) - 1)
        for semantic in flow_by_id.values()
    )
    binding_count = 0
    per_core_end: dict[str, int] = {}
    for schedule in context.schedule_set.schedules:
        binding_count += len(schedule.buffer_bindings)
        for binding in schedule.buffer_bindings:
            key = str(binding.core_id)
            per_core_end[key] = max(
                per_core_end.get(key, 0),
                binding.region_offset_bytes + binding.size_bytes,
            )
    opcode_counts: Counter[str] = Counter()
    record_count = 0
    for outer in source.manifest.fragments:
        leaf = outer.fragment if type(outer) is RegionManifest else outer
        for stream in leaf.core_streams:
            record_count += len(stream.records)
            opcode_counts.update(record.opcode.name for record in stream.records)
    return {
        "op_counts": dict(sorted(op_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "unique_flow_count": len(flow_by_id),
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "fragment_count": len(source.manifest.fragments),
        "record_count": record_count,
        "rank_gemm_flops": rank_gemm_flops,
        "rank_attention_matmul_flops": rank_attention_matmul_flops,
        "analytic_transfer_bytes": transfer_bytes,
        "scheduled_binding_count": binding_count,
        "per_core_sram_max_end": dict(sorted(per_core_end.items(), key=lambda item: int(item[0]))),
    }


def _policy_selection_rows(
    selections: tuple[PolicySelection, ...],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for index, selection in enumerate(selections):
        selection.validate(f"policy_selections[{index}]")
        row = to_primitive(selection, path=f"policy_selections[{index}]")
        if type(row) is not dict:
            raise SchemaError(
                "selection must encode as an object",
                path=f"policy_selections[{index}]",
            )
        rows.append(row)
    return tuple(rows)


def _decode_policy_selection_rows(
    value: object,
    path: str,
) -> tuple[PolicySelection, ...]:
    if not isinstance(value, (tuple, list)):
        raise SchemaError("must be an ordered array", path=path)
    return tuple(
        from_data(PolicySelection, row, path=f"{path}[{index}]")
        for index, row in enumerate(value)
    )


def _receipt_rows(compilation: NaiveCompilation) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for index, receipt in enumerate(compilation.snapshot.receipts):
        row = to_primitive(receipt, path=f"pass_receipts[{index}]")
        if type(row) is not dict:
            raise SchemaError(
                "receipt must encode as an object",
                path=f"pass_receipts[{index}]",
            )
        rows.append(row)
    return tuple(rows)


def _load_finalizer_report(path: Path, source: LinkedProgramProfile, artifact: bytes) -> dict[str, object]:
    raw = load_json_value(path, path="finalizer_report")
    if not isinstance(raw, dict):
        raise NaiveRunError("must be a JSON object", path="finalizer_report")
    manifest_digest = canonical_digest(source.manifest)
    artifact_sha = _sha256_bytes(artifact)
    expected = {
        "linked_manifest_id": source.manifest.id,
        "linked_manifest_digest": manifest_digest,
        "artifact_sha256": artifact_sha,
        "artifact_bytes": len(artifact),
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise NaiveRunError(
                f"{key} does not match finalized inputs", path=f"finalizer_report.{key}"
            )
    for key in ("core_count", "record_count", "relocation_count"):
        if type(raw.get(key)) is not int or raw[key] <= 0:
            raise NaiveRunError("must be a positive integer", path=f"finalizer_report.{key}")
    return raw


def run_naive(
    request: NaiveRunRequest,
    *,
    registry: PolicyRegistry | None = None,
) -> NaiveRunResult:
    """Compile, finalize and run one E1/E2 timing case transactionally."""

    if type(request) is not NaiveRunRequest:
        raise SchemaError("must be a NaiveRunRequest", path="request")
    request.validate()
    if registry is not None and type(registry) is not PolicyRegistry:
        raise SchemaError("must be a PolicyRegistry", path="registry")
    target = request.output_dir.absolute()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{target.name}.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise NaiveRunError("another runner owns this output target", path="request.output_dir") from error
    os.close(lock_fd)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp.", dir=parent))
    try:
        inputs_dir = temporary / "inputs"
        compile_dir = temporary / "compile"
        program_dir = temporary / "program"
        run_dir = temporary / "run"
        for directory in (inputs_dir, compile_dir, program_dir, run_dir):
            directory.mkdir()

        # Import lazily so cli.py can expose run_naive without a module cycle.
        from .cli import load_experiment_spec

        spec = load_experiment_spec(request.spec_path)
        simulation = load_json_value(
            request.simulation_config_path, path="simulation_config"
        )
        if not isinstance(simulation, dict):
            raise SchemaError("must be a JSON object", path="simulation_config")
        if not request.mapping_config_path.read_text(encoding="utf-8").strip():
            raise SchemaError("must not be empty", path="mapping_config")
        fabric, hbm_address_spaces = load_physical_fabric_and_hbm_address_spaces(
            request.hardware_config_path, request.mapping_config_path
        )
        compilation = compile_naive(
            spec,
            fabric,
            hbm_address_spaces=hbm_address_spaces,
            registry=registry,
            intra_die_refine_options=request.intra_die_refine_options,
        )
        _validate_case_identity(request, compilation)
        source = _select_profile(compilation, request.profile_id)

        _write_json(inputs_dir / "experiment.json", spec)
        hardware_sha = _copy_input(
            request.hardware_config_path, inputs_dir / "hardware.json"
        )
        simulation_sha = _copy_input(
            request.simulation_config_path, inputs_dir / "simulation.json"
        )
        mapping_sha = _copy_input(
            request.mapping_config_path, inputs_dir / "mapping.spec"
        )
        manifest_path = compile_dir / "linked_manifest.json"
        receipts_path = compile_dir / "pass_receipts.json"
        stage_path = compile_dir / "stage_digests.json"
        _write_json(manifest_path, source.manifest)
        receipts = _receipt_rows(compilation)
        _write_json(receipts_path, receipts)
        stage_digests = tuple(canonical_digest(value) for value in compilation.artifacts)
        _write_json(stage_path, stage_digests)
        refined_bundle = next(
            (
                artifact
                for artifact in compilation.artifacts
                if type(artifact) is RefinedIR2Bundle
            ),
            None,
        )
        v2_search_decisions = (
            tuple(
                entry.split_k_refinement.search_decision
                for entry in refined_bundle.entries
                if entry.split_k_refinement is not None
            )
            if refined_bundle is not None
            else ()
        )
        for decision_index, decision in enumerate(v2_search_decisions):
            if decision.hardware_digest != hardware_sha:
                raise NaiveRunError(
                    "timing model hardware digest does not match the run input",
                    path=(
                        "compile.intra_die_v2_search_decisions"
                        f"[{decision_index}].hardware_digest"
                    ),
                )
            if decision.simulation_digest != simulation_sha:
                raise NaiveRunError(
                    "timing model simulation digest does not match the run input",
                    path=(
                        "compile.intra_die_v2_search_decisions"
                        f"[{decision_index}].simulation_digest"
                    ),
                )
        _write_json(
            compile_dir / "intra_die_v2_search_decisions.json",
            v2_search_decisions,
        )

        artifact_path = program_dir / "program.npup"
        finalizer_report_path = program_dir / "finalization_report.json"
        _run_process(
            [
                str(request.finalizer_path.absolute()),
                "--input",
                str(manifest_path.absolute()),
                "--output",
                str(artifact_path.absolute()),
                "--report",
                str(finalizer_report_path.absolute()),
            ],
            cwd=request.finalizer_path.absolute().parent,
            timeout=request.timeout_seconds,
            stdout_path=program_dir / "finalizer.stdout.log",
            stderr_path=program_dir / "finalizer.stderr.log",
        )
        artifact_bytes = artifact_path.read_bytes()
        finalizer_report = _load_finalizer_report(
            finalizer_report_path, source, artifact_bytes
        )
        artifact_sha = _sha256_bytes(artifact_bytes)
        state_seeds, state_expected = (
            build_deterministic_timing_state_overrides(source)
        )
        program_io = build_timing_program_io(
            source,
            artifact_sha,
            state_seed_overrides=state_seeds,
            state_expected_overrides=state_expected,
        )
        program_io.validate_against(source.manifest)
        program_io_path = program_dir / "program_io.json"
        _write_json(program_io_path, program_io)

        runtime_evidence: list[_RuntimeEvidence] = []
        parsed_runs: list[dict[str, object]] = []
        for repeat_index in range(request.repeat):
            completed = _run_process(
                [
                    str(request.npusim_path.absolute()),
                    "--program",
                    str(artifact_path.absolute()),
                    "--linked-manifest",
                    str(manifest_path.absolute()),
                    "--program-io",
                    str(program_io_path.absolute()),
                    "--hardware-config",
                    str((inputs_dir / "hardware.json").absolute()),
                    "--simulation-config",
                    str((inputs_dir / "simulation.json").absolute()),
                    "--mapping-config",
                    str((inputs_dir / "mapping.spec").absolute()),
                    "--trace-window",
                    str(request.trace_window),
                ],
                cwd=request.npusim_path.absolute().parent,
                timeout=request.timeout_seconds,
                stdout_path=run_dir / f"stdout.{repeat_index}.log",
                stderr_path=run_dir / f"stderr.{repeat_index}.log",
            )
            output = completed.stdout + "\n" + completed.stderr
            evidence = _validate_runtime(
                output,
                source=source,
                artifact_sha256=artifact_sha,
                initialization_count=len(program_io.initializations),
                probe_count=len(program_io.output_probes),
            )
            runtime_evidence.append(evidence)
            parsed_runs.append(
                {
                    "repeat": repeat_index,
                    "makespan_cycles": evidence.makespan_cycles,
                    "ack_total": evidence.ack_total,
                    "done_total": evidence.done_total,
                    "ack_by_core": evidence.ack_by_core,
                    "done_by_core": evidence.done_by_core,
                    "observed_transfer_bytes": evidence.observed_transfer_bytes,
                    "d2d_link_packets": evidence.d2d_link_packets,
                }
            )
        if any(
            evidence != runtime_evidence[0]
            for evidence in runtime_evidence[1:]
        ):
            raise NaiveRunError(
                "repeat runtime marker signature changed", path="runtime.repeat"
            )
        intra_die_v2_calibration_evidence = (
            _write_intra_die_v2_calibration_evidence(
                compile_dir / "intra_die_v2_calibration_evidence.json",
                v2_search_decisions,
                simulator_measured_makespan_cycles=(
                    runtime_evidence[0].makespan_cycles
                ),
                simulator_calls_used=request.repeat,
                repeat_signature_stable=True,
            )
        )
        _write_json(run_dir / "parsed_markers.json", tuple(parsed_runs))

        static_metrics = _static_metrics(source)
        if (
            runtime_evidence[0].observed_transfer_bytes
            != static_metrics["analytic_transfer_bytes"]
        ):
            raise NaiveRunError(
                "observed D2D bytes do not equal analytic routed bytes",
                path="runtime.observed_transfer_bytes",
            )
        runtime = {
            "repeat": request.repeat,
            "makespan_cycles": runtime_evidence[0].makespan_cycles,
            "ack_total": runtime_evidence[0].ack_total,
            "done_total": runtime_evidence[0].done_total,
            "ack_by_core": runtime_evidence[0].ack_by_core,
            "done_by_core": runtime_evidence[0].done_by_core,
            "drain_residuals": 0,
            "credit_balanced": True,
            "program_io_phases": ("resolved", "applied", "verify"),
            "program_io_initializations": len(program_io.initializations),
            "program_io_probes": len(program_io.output_probes),
            "repeat_signature_stable": True,
            "observed_transfer_bytes": runtime_evidence[0].observed_transfer_bytes,
            "d2d_link_packets": runtime_evidence[0].d2d_link_packets,
        }
        context_ids = tuple(context.id for context in compilation.contexts)
        calibration_notes = _intra_die_v2_calibration_notes(
            intra_die_v2_calibration_evidence
        )
        report = NaiveRunReport.create(
            case=request.case,
            validation_mode=request.validation,
            inputs={
                "spec_digest": canonical_digest(spec),
                "fabric_digest": canonical_digest(fabric),
                "hardware_sha256": hardware_sha,
                "simulation_sha256": simulation_sha,
                "mapping_sha256": mapping_sha,
            },
            tools={
                "finalizer_sha256": _sha256_file(request.finalizer_path),
                "npusim_sha256": _sha256_file(request.npusim_path),
                "linked_manifest_schema": source.manifest.schema_version,
                "program_io_schema": program_io.schema_version,
            },
            provenance={
                "profile_id": source.profile_id,
                "profile_weight": source.weight,
                "pass_receipts": receipts,
                "stage_digests": stage_digests,
                "context_ids": context_ids,
                "policy_selections": _policy_selection_rows(
                    compilation.policy_selections
                ),
                "linked_bundle_id": compilation.linked.id,
                "linked_profile_id": source.id,
                "linked_manifest_id": source.manifest.id,
                "linked_manifest_digest": canonical_digest(source.manifest),
            },
            artifact={
                "artifact_sha256": artifact_sha,
                "artifact_bytes": len(artifact_bytes),
                "core_count": finalizer_report["core_count"],
                "record_count": finalizer_report["record_count"],
                "relocation_count": finalizer_report["relocation_count"],
                "finalizer_report_digest": canonical_digest(finalizer_report),
                "program_io_id": program_io.id,
                "program_io_digest": canonical_digest(program_io),
            },
            static_metrics=static_metrics,
            runtime=runtime,
            validation={
                "timing": "pass",
                "address_lifecycle": "pass",
                "transport_control": "pass",
                "compute_functional": "unsupported",
                "reduction_u3f": "unsupported",
                "end_to_end_functional": "unsupported",
                "capability_notes": (
                    "timing-only compute does not write Dense numerical output",
                    "zero timing probes prove address/lifecycle/transport/control closure only",
                    "reduction-u3f requires the independent E0 non-zero oracle",
                    *calibration_notes,
                ),
            },
        )
        report.validate()
        report_path = temporary / "run_report.json"
        _write_json(report_path, report)
        (temporary / "SUCCESS").write_text(report.id + "\n", encoding="utf-8")

        if target.exists() or target.is_symlink():
            raise NaiveRunError(
                "output target appeared during the run", path="request.output_dir"
            )
        os.rename(temporary, target)
        return NaiveRunResult(
            report_id=report.id,
            output_dir=target,
            report_path=target / "run_report.json",
            report=report,
        )
    except Exception:
        if temporary.exists():
            if request.keep_failed:
                failed = parent / f"{target.name}.failed.{temporary.name.rsplit('.', 1)[-1]}"
                if not failed.exists():
                    os.rename(temporary, failed)
                else:
                    shutil.rmtree(temporary)
            else:
                shutil.rmtree(temporary)
        raise
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "NAIVE_RUN_REPORT_SCHEMA_VERSION",
    "NaiveRunCase",
    "NaiveRunError",
    "NaiveRunReport",
    "NaiveRunRequest",
    "NaiveRunResult",
    "NaiveRunValidation",
    "run_naive",
]
