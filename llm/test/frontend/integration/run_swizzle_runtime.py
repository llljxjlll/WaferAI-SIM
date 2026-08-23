#!/usr/bin/env python3
"""Run the W10/W11 Swizzle timing comparison through official binaries.

The production provider is constructed from explicit checked-in hardware and
mapping inputs.  A Swizzle-only mode exists for bringing up the runtime chain
without fabricating the still-separate naive comparison branch.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Protocol

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from llm.frontend.wafer_frontend.errors import (  # noqa: E402
    SchemaError,
    StageNotImplementedError,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (  # noqa: E402
    LinkedProgramManifest,
)
from llm.frontend.wafer_frontend.schema.program_io import (  # noqa: E402
    ProgramIoContract,
    ProgramIoMode,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
)
from llm.frontend.wafer_frontend.schema.swizzle_evidence import (  # noqa: E402
    SWIZZLE_RUNTIME_MARKER_SCHEMA_VERSION,
    SwizzleBranchRuntimeEvidence,
    SwizzleCaseRuntimeEvidence,
    SwizzleComparisonBranch,
    SwizzleComparisonCasePlan,
    SwizzleComparisonSuitePlan,
    SwizzleCoreCount,
    SwizzleNamedCount,
    SwizzleRuntimeArtifactEvidence,
    SwizzleRuntimeComparisonReport,
    SwizzleRuntimeControlEvidence,
    SwizzleRuntimeMetrics,
    SwizzleRuntimeProgramIoEvidence,
    SwizzleRuntimeRepeatEvidence,
    SwizzleRuntimeToolEvidence,
)
from swizzle_comparison import build_swizzle_comparison_suite  # noqa: E402


_STATUS = "[PROGRAM_IO] "
_PROBE = "[PROGRAM_IO_PROBE] "
_SIM = "[SIM_RESULT] "
_HOST = "[HOSTLANE] "
_HOSTSIG = "[HOSTSIG] "
_P5 = "[P5 P2P DRAIN] "
_P5_TIMING = "[P5 P2P TIMING DRAIN] "
_COLL = "[COLL_DRAIN] "
_DRAIN = "[DRAIN] "
_D2D_TYPE = "[D2D_TYPE] "
_D2D_LINK = "[D2D_LINK] "


@dataclass(frozen=True, slots=True)
class SwizzleRuntimeExecutable:
    case_plan_ref: str
    branch: SwizzleComparisonBranch
    linked_source_ref: str
    manifest: LinkedProgramManifest
    hardware_json: str
    mapping_text: str

    def validate(self, path: str = "swizzle_runtime_executable") -> None:
        for name in ("case_plan_ref", "linked_source_ref", "hardware_json", "mapping_text"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise SchemaError("must be a non-empty string", path=f"{path}.{name}")
        if type(self.branch) is not SwizzleComparisonBranch:
            raise SchemaError("must be a comparison branch", path=f"{path}.branch")
        if type(self.manifest) is not LinkedProgramManifest:
            raise SchemaError("must be a LinkedProgramManifest", path=f"{path}.manifest")
        self.manifest.validate(f"{path}.manifest")


class SwizzleLowerLinkProvider(Protocol):
    """The exact API W9 must implement before W10 runtime can activate."""

    def lower_link(
        self,
        case: SwizzleComparisonCasePlan,
        branch: SwizzleComparisonBranch,
    ) -> SwizzleRuntimeExecutable: ...

    def build_actual_sha_program_io(
        self,
        executable: SwizzleRuntimeExecutable,
        program_artifact_sha256: str,
    ) -> ProgramIoContract: ...


@dataclass(frozen=True, slots=True)
class _Observation:
    makespan_cycles: int
    marker_digest: str
    control: SwizzleRuntimeControlEvidence
    packet_count: int


class SwizzleRuntimeStageFailure(RuntimeError):
    """The first failing external stage, with its complete captured output."""

    def __init__(
        self,
        *,
        stage: str,
        command: tuple[str, ...],
        returncode: int,
        output: str,
    ) -> None:
        self.stage = stage
        self.command = command
        self.returncode = returncode
        self.output = output
        super().__init__(
            f"Swizzle runtime first failure stage={stage!r} "
            f"returncode={returncode}: {' '.join(command)}\n{output}"
        )


def require_swizzle_lower_link_provider(
    provider: SwizzleLowerLinkProvider | None,
) -> SwizzleLowerLinkProvider:
    if provider is None:
        raise StageNotImplementedError(
            "production Swizzle lower/link and ProgramIo APIs are not published; "
            "runtime comparison refuses to fabricate a manifest",
            path="swizzle_runtime.lower_link_provider",
            hint=(
                "implement SwizzleLowerLinkProvider.lower_link and "
                "build_actual_sha_program_io from the typed N5 projection"
            ),
        )
    return provider


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    stage: str,
    stdout_path: Path,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise SwizzleRuntimeStageFailure(
            stage=stage,
            command=tuple(command),
            returncode=completed.returncode,
            output=completed.stdout,
        )
    return completed


def _row(line: str, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line[len(prefix):].strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value.rstrip(",.")
    return result


def _rows(output: str, prefix: str) -> tuple[dict[str, str], ...]:
    result = []
    for line in output.splitlines():
        position = line.find(prefix)
        if position >= 0:
            normalized = line[position:].split(" | ", 1)[0].rstrip(". ")
            result.append(_row(normalized, prefix))
    return tuple(result)


def _number(row: dict[str, str], key: str) -> int:
    try:
        return int(row[key], 10)
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"missing/non-decimal {key!r} in {row}") from error


def _parse_signature(value: str, arity: int) -> tuple[tuple[int, ...], ...]:
    result = []
    for item in value.rstrip(",").split(","):
        if not item:
            continue
        try:
            parsed = tuple(int(part, 10) for part in item.split(":"))
        except ValueError as error:
            raise RuntimeError(f"non-decimal HOSTSIG item {item!r}") from error
        if len(parsed) != arity:
            raise RuntimeError(f"bad HOSTSIG item {item!r}")
        result.append(parsed)
    return tuple(sorted(result))


def _validate_program_io(
    output: str,
    artifact_sha256: str,
    contract: ProgramIoContract,
) -> None:
    status = _rows(output, _STATUS)
    if tuple(row.get("phase") for row in status) != ("resolved", "applied", "verify"):
        raise RuntimeError(f"ProgramIo phase closure changed: {status}")
    for row in status:
        if (
            row.get("mode") != "timing"
            or _number(row, "initializations") != len(contract.initializations)
            or _number(row, "probes") != len(contract.output_probes)
            or row.get("pass") != "1"
        ):
            raise RuntimeError(f"ProgramIo status failed: {row}")
    if any(row.get("checksum") != artifact_sha256 for row in status[:2]):
        raise RuntimeError("ProgramIo checksum is not the actual artifact SHA")
    probes = {item.id: item for item in contract.output_probes}
    blobs = {item.id: item for item in contract.blobs}
    rows = _rows(output, _PROBE)
    if len(rows) != len(probes) or not probes:
        raise RuntimeError("ProgramIo probe coverage is not exact")
    seen: set[str] = set()
    for row in rows:
        probe = probes.get(row.get("id", ""))
        if probe is None:
            raise RuntimeError(f"unknown ProgramIo probe: {row}")
        expected = blobs[probe.blob_ref].sha256
        if (
            row.get("expected_checksum") != expected
            or row.get("checksum") != expected
            or row.get("valid") != "1"
            or row.get("exact") != "1"
            or row.get("pass") != "1"
        ):
            raise RuntimeError(f"ProgramIo probe failed: {row}")
        seen.add(probe.id)
    if seen != set(probes):
        raise RuntimeError("ProgramIo probe IDs are not exact")


def _parse_control(output: str) -> tuple[int, SwizzleRuntimeControlEvidence]:
    simulation = _rows(output, _SIM)
    if len(simulation) != 1:
        raise RuntimeError("SIM_RESULT must appear exactly once")
    makespan = _number(simulation[0], "makespan_cycles")
    if makespan == 0 or "[PROTO_WAIT]" in output or "End DONE reception" not in output:
        raise RuntimeError("simulation/DONE boundary did not close")
    host = _rows(output, _HOST)
    signatures = _rows(output, _HOSTSIG)
    if len(host) != 1 or len(signatures) != 1 or _number(host[0], "mismatch") != 0:
        raise RuntimeError("HOSTLANE/HOSTSIG closure changed")
    done = _parse_signature(signatures[0].get("done", ""), 2)
    ack = _parse_signature(signatures[0].get("ack", ""), 3)
    done_counts = tuple(SwizzleCoreCount(core, count) for core, count in done)
    ack_by_core: Counter[int] = Counter()
    for core, _lane, count in ack:
        ack_by_core[core] += count
    ack_counts = tuple(SwizzleCoreCount(core, count) for core, count in sorted(ack_by_core.items()))
    if (
        sum(item.count for item in ack_counts) != _number(host[0], "ack_total")
        or sum(item.count for item in done_counts) != _number(host[0], "done_total")
    ):
        raise RuntimeError("ACK/DONE totals disagree with per-core signatures")
    p5 = _rows(output, _P5)
    p5_timing = _rows(output, _P5_TIMING)
    collective = _rows(output, _COLL)
    drains = _rows(output, _DRAIN)
    p2p_residual = sum(_number(row, "residual") for row in p5 + p5_timing)
    collective_residual = sum(
        _number(row, key)
        for row in collective
        for key in (
            "tree_entries", "reduce_nodes", "barriers", "gather",
            "reduce_rx", "endpoints", "dte_tokens", "event",
        )
        if key in row
    )
    global_residual = sum(
        _number(row, key)
        for row in drains
        for key in ("router_residual", "d2d_link_residual")
        if key in row
    )
    control = SwizzleRuntimeControlEvidence(
        ack_counts=ack_counts,
        done_counts=done_counts,
        drain_residuals=(
            SwizzleNamedCount("collective", collective_residual),
            SwizzleNamedCount("global", global_residual),
            SwizzleNamedCount("p2p", p2p_residual),
            SwizzleNamedCount("timing", 0),
        ),
        proto_wait_count=0,
        all_done_boundary_reached=True,
    )
    control.validate()
    return makespan, control


def _observe(output: str, artifact_sha256: str, contract: ProgramIoContract) -> _Observation:
    _validate_program_io(output, artifact_sha256, contract)
    makespan, control = _parse_control(output)
    typed = _rows(output, _D2D_TYPE)
    links = _rows(output, _D2D_LINK)
    if len(typed) != 1 or not links:
        raise RuntimeError("typed D2D packet markers are required")
    packet_count = sum(_number(row, "data_out") for row in links)
    marker_prefixes = (
        _STATUS, _PROBE, _SIM, _HOST, _HOSTSIG, _P5,
        _P5_TIMING, _COLL, _DRAIN, _D2D_TYPE, _D2D_LINK,
    )
    lines = tuple(
        line[line.find(prefix):].split(" | ", 1)[0].rstrip(". ")
        for line in output.splitlines()
        for prefix in marker_prefixes
        if line.find(prefix) >= 0
    )
    return _Observation(
        makespan,
        hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        control,
        packet_count,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _branch_cost(case: SwizzleComparisonCasePlan, branch: SwizzleComparisonBranch):
    return case.baseline_cost if branch is SwizzleComparisonBranch.NAIVE else case.selected_cost


def _run_branch(
    case: SwizzleComparisonCasePlan,
    branch: SwizzleComparisonBranch,
    provider: SwizzleLowerLinkProvider,
    *,
    finalizer: Path,
    resolver: Path,
    npusim: Path,
    simulation: Path,
    runtime_root: Path,
    timeout: int,
    evidence_root: Path | None = None,
) -> SwizzleBranchRuntimeEvidence:
    executable = provider.lower_link(case, branch)
    executable.validate()
    if executable.case_plan_ref != case.id or executable.branch is not branch:
        raise SchemaError("lower/link result provenance drifted", path="swizzle_runtime.executable")
    context = (
        nullcontext(str(evidence_root / case.case_id / branch.value))
        if evidence_root is not None
        else tempfile.TemporaryDirectory(
            prefix=f"swizzle-{case.case_id}-{branch.value}-",
            dir=runtime_root,
        )
    )
    with context as raw:
        directory = Path(raw)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / "linked.json"
        hardware_path = directory / "hardware.json"
        mapping_path = directory / "mapping.spec"
        sidecar_path = directory / "program_io.json"
        manifest_path.write_text(canonical_json(executable.manifest), encoding="utf-8")
        hardware_path.write_text(executable.hardware_json, encoding="utf-8")
        mapping_path.write_text(executable.mapping_text, encoding="utf-8")
        artifact_paths = (directory / "program.0.npup", directory / "program.1.npup")
        report_paths = (directory / "finalizer.0.json", directory / "finalizer.1.json")
        artifact_bytes = []
        finalizer_reports = []
        for index, (artifact_path, report_path) in enumerate(
            zip(artifact_paths, report_paths, strict=True)
        ):
            _run(
                [str(finalizer), "--input", str(manifest_path), "--output", str(artifact_path), "--report", str(report_path)],
                cwd=runtime_root,
                timeout=120,
                stage=f"{case.case_id}.{branch.value}.finalizer.{index}",
                stdout_path=directory / f"finalizer.{index}.stdout.txt",
            )
            artifact_bytes.append(artifact_path.read_bytes())
            finalizer_reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        if artifact_bytes[0] != artifact_bytes[1] or finalizer_reports[0] != finalizer_reports[1]:
            raise RuntimeError("finalizer byte/report repeat changed")
        artifact_sha = hashlib.sha256(artifact_bytes[0]).hexdigest()
        finalizer_report_digest = canonical_digest(finalizer_reports[0])
        report = finalizer_reports[0]
        if (
            report.get("artifact_sha256") != artifact_sha
            or report.get("artifact_bytes") != len(artifact_bytes[0])
            or report.get("linked_manifest_id") != executable.manifest.id
            or report.get("linked_manifest_digest") != canonical_digest(executable.manifest)
        ):
            raise RuntimeError("finalizer report does not close actual artifact/manifest")
        contract = provider.build_actual_sha_program_io(executable, artifact_sha)
        if type(contract) is not ProgramIoContract or contract.mode is not ProgramIoMode.TIMING:
            raise SchemaError("provider must return timing ProgramIoContract", path="swizzle_runtime.program_io")
        contract.validate_against(executable.manifest)
        if contract.program_artifact_sha256 != artifact_sha:
            raise SchemaError("ProgramIo is not bound to actual artifact SHA", path="swizzle_runtime.program_io")
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        resolved = _run(
            [str(resolver), "--resolve", str(manifest_path), str(artifact_paths[0]), str(sidecar_path)],
            cwd=runtime_root,
            timeout=120,
            stage=f"{case.case_id}.{branch.value}.resolver",
            stdout_path=directory / "resolver.stdout.txt",
        )
        if (
            f"initializations={len(contract.initializations)}" not in resolved.stdout
            or f"probes={len(contract.output_probes)}" not in resolved.stdout
        ):
            raise RuntimeError(
                "resolver output lost exact ProgramIo initialization/probe counts"
            )
        observations = []
        for index in range(2):
            execution = _run(
                [
                    str(npusim), "--program", str(artifact_paths[0]),
                    "--linked-manifest", str(manifest_path), "--program-io", str(sidecar_path),
                    "--hardware-config", str(hardware_path), "--simulation-config", str(simulation),
                    "--mapping-config", str(mapping_path), "--trace-window", "1000000",
                ],
                cwd=runtime_root,
                timeout=timeout,
                stage=f"{case.case_id}.{branch.value}.npusim.{index}",
                stdout_path=directory / f"npusim.{index}.stdout.txt",
            )
            observations.append(_observe(execution.stdout, artifact_sha, contract))
        if observations[0] != observations[1]:
            raise RuntimeError("npusim marker/makespan repeat changed")
        first = observations[0]
        cost = _branch_cost(case, branch)
        metrics = SwizzleRuntimeMetrics(
            logical_bytes=cost.logical_bytes,
            byte_hops=cost.byte_hops,
            packet_count=first.packet_count,
            direction_port_utilization=cost.direction_port_utilization,
            sram_high_water_bytes=cost.sram_high_water_bytes,
            control_action_count=cost.control_action_count,
        )
        control_digest = canonical_digest(first.control)
        metrics_digest = canonical_digest(metrics)
        result = SwizzleBranchRuntimeEvidence(
            branch=branch,
            tools=SwizzleRuntimeToolEvidence(_sha(finalizer), _sha(resolver), _sha(npusim)),
            artifact=SwizzleRuntimeArtifactEvidence(
                linked_manifest_id=executable.manifest.id,
                linked_manifest_digest=canonical_digest(executable.manifest),
                program_artifact_sha256=artifact_sha,
                artifact_size_bytes=len(artifact_bytes[0]),
                finalizer_artifact_sha256s=(artifact_sha, artifact_sha),
                finalizer_report_digests=(finalizer_report_digest, finalizer_report_digest),
            ),
            program_io=SwizzleRuntimeProgramIoEvidence(
                contract_id=contract.id,
                contract_digest=canonical_digest(contract),
                program_artifact_sha256=artifact_sha,
                mode=contract.mode,
                initialization_count=len(contract.initializations),
                probe_count=len(contract.output_probes),
                all_probes_passed=True,
            ),
            control=first.control,
            metrics=metrics,
            marker_schema_version=SWIZZLE_RUNTIME_MARKER_SCHEMA_VERSION,
            repeats=tuple(
                SwizzleRuntimeRepeatEvidence(
                    index,
                    observation.makespan_cycles,
                    observation.marker_digest,
                    control_digest,
                    metrics_digest,
                )
                for index, observation in enumerate(observations)
            ),
        )
        result.validate()
        (directory / "branch_evidence.json").write_text(
            canonical_json(result), encoding="utf-8"
        )
        return result


def run_swizzle_branch_preflight(
    *,
    provider: SwizzleLowerLinkProvider | None,
    finalizer: Path,
    resolver: Path,
    npusim: Path,
    simulation: Path,
    runtime_root: Path,
    case_ids: tuple[str, ...] = (),
    timeout: int = 600,
) -> tuple[SwizzleBranchRuntimeEvidence, ...]:
    """Run only real SWIZZLE branches; never synthesize naive evidence."""

    suite = build_swizzle_comparison_suite()
    active = require_swizzle_lower_link_provider(provider)
    selected = tuple(
        case for case in suite.cases if not case_ids or case.case_id in case_ids
    )
    if not selected or (
        case_ids and {item.case_id for item in selected} != set(case_ids)
    ):
        raise SchemaError(
            "preflight case ids must exactly select canonical comparison cases",
            path="swizzle_runtime.case_ids",
        )
    return tuple(
        _run_branch(
            case,
            SwizzleComparisonBranch.SWIZZLE,
            active,
            finalizer=finalizer,
            resolver=resolver,
            npusim=npusim,
            simulation=simulation,
            runtime_root=runtime_root,
            timeout=timeout,
            evidence_root=runtime_root / "swizzle-runtime-evidence",
        )
        for case in selected
    )


def run_naive_branch_preflight(
    *,
    provider: SwizzleLowerLinkProvider | None,
    finalizer: Path,
    resolver: Path,
    npusim: Path,
    simulation: Path,
    runtime_root: Path,
    case_ids: tuple[str, ...] = (),
    timeout: int = 600,
) -> tuple[SwizzleBranchRuntimeEvidence, ...]:
    """Run only real production UNFUSED/NAIVE branches twice."""

    suite = build_swizzle_comparison_suite()
    active = require_swizzle_lower_link_provider(provider)
    selected = tuple(
        case for case in suite.cases if not case_ids or case.case_id in case_ids
    )
    if not selected or (
        case_ids and {item.case_id for item in selected} != set(case_ids)
    ):
        raise SchemaError(
            "preflight case ids must exactly select canonical comparison cases",
            path="swizzle_runtime.case_ids",
        )
    return tuple(
        _run_branch(
            case,
            SwizzleComparisonBranch.NAIVE,
            active,
            finalizer=finalizer,
            resolver=resolver,
            npusim=npusim,
            simulation=simulation,
            runtime_root=runtime_root,
            timeout=timeout,
            evidence_root=runtime_root / "swizzle-runtime-evidence",
        )
        for case in selected
    )


def run_swizzle_runtime_suite(
    *,
    provider: SwizzleLowerLinkProvider | None,
    finalizer: Path,
    resolver: Path,
    npusim: Path,
    simulation: Path,
    runtime_root: Path,
    timeout: int = 600,
    evidence_root: Path | None = None,
) -> SwizzleRuntimeComparisonReport:
    suite = build_swizzle_comparison_suite()
    active = require_swizzle_lower_link_provider(provider)
    cases = tuple(
        SwizzleCaseRuntimeEvidence(
            case_plan_ref=case.id,
            naive=_run_branch(
                case, SwizzleComparisonBranch.NAIVE, active,
                finalizer=finalizer, resolver=resolver, npusim=npusim,
                simulation=simulation, runtime_root=runtime_root, timeout=timeout,
                evidence_root=evidence_root,
            ),
            swizzle=_run_branch(
                case, SwizzleComparisonBranch.SWIZZLE, active,
                finalizer=finalizer, resolver=resolver, npusim=npusim,
                simulation=simulation, runtime_root=runtime_root, timeout=timeout,
                evidence_root=evidence_root,
            ),
        )
        for case in suite.cases
    )
    report = SwizzleRuntimeComparisonReport.create(
        suite=suite,
        cases=cases,
        timing_execution=True,
        functional_execution=False,
    )
    report.validate()
    return report


def validate_runtime_config_paths(
    *,
    hardware_json: str,
    simulation_json: str,
    runtime_root: Path,
) -> tuple[Path, ...]:
    """Resolve simulator-owned relative config paths against its real cwd."""

    try:
        hardware = json.loads(hardware_json)
        simulation = json.loads(simulation_json)
    except (TypeError, json.JSONDecodeError) as error:
        raise SchemaError(
            "hardware/simulation inputs must be JSON objects",
            path="swizzle_runtime.runtime_inputs",
        ) from error
    if type(hardware) is not dict or type(simulation) is not dict:
        raise SchemaError(
            "hardware/simulation inputs must be JSON objects",
            path="swizzle_runtime.runtime_inputs",
        )
    references: list[tuple[str, object]] = [
        (
            "simulation.gpu.dram_config_file",
            simulation.get("gpu", {}).get("dram_config_file")
            if type(simulation.get("gpu")) is dict
            else None,
        )
    ]
    memory_system = hardware.get("memory_system")
    if type(memory_system) is dict:
        stacks = memory_system.get("hbm_stacks", [])
        if type(stacks) is not list:
            raise SchemaError(
                "must be a list",
                path="swizzle_runtime.hardware.memory_system.hbm_stacks",
            )
        references.extend(
            (
                f"hardware.memory_system.hbm_stacks[{index}].channel_dram_config",
                stack.get("channel_dram_config") if type(stack) is dict else None,
            )
            for index, stack in enumerate(stacks)
        )
    resolved = []
    for path, raw in references:
        if type(raw) is not str or not raw:
            raise SchemaError("must be a non-empty path", path=path)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = runtime_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise SchemaError(
                f"runtime cwd does not resolve config path {raw!r} to a file; "
                f"resolved {candidate}",
                path=path,
            )
        resolved.append(candidate)
    return tuple(resolved)


def _existing_file(
    parser: argparse.ArgumentParser, path: Path, option: str
) -> Path:
    result = path.resolve()
    if not result.is_file():
        parser.error(f"{option} is not a file: {result}")
    return result


def main(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run official Swizzle timing preflight/comparison"
    )
    parser.add_argument("--npusim", required=True, type=Path)
    parser.add_argument("--finalizer", required=True, type=Path)
    parser.add_argument("--resolver", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("naive-preflight", "swizzle-preflight", "suite"), default="suite"
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="canonical case id; repeat in a preflight mode",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    for option in (
        "npusim", "finalizer", "resolver", "hardware", "mapping", "simulation"
    ):
        setattr(
            args,
            option,
            _existing_file(parser, getattr(args, option), f"--{option}"),
        )
    args.runtime_root = args.runtime_root.resolve()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    if args.report is not None:
        args.report = args.report.resolve()
    if args.mode == "suite" and args.case:
        parser.error("--case is only valid with a preflight mode")

    sys.modules.setdefault("run_swizzle_runtime", sys.modules[__name__])
    from swizzle_runtime_provider import ProductionSwizzleLowerLinkProvider

    hardware_json = args.hardware.read_text(encoding="utf-8")
    mapping_text = args.mapping.read_text(encoding="utf-8")
    simulation_json = args.simulation.read_text(encoding="utf-8")
    validate_runtime_config_paths(
        hardware_json=hardware_json,
        simulation_json=simulation_json,
        runtime_root=args.runtime_root,
    )
    provider = ProductionSwizzleLowerLinkProvider(
        hardware_json=hardware_json,
        mapping_text=mapping_text,
    )
    evidence_root = args.runtime_root / "swizzle-runtime-evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "input_sources.json").write_text(
        canonical_json({
            "hardware_path": str(args.hardware),
            "hardware_sha256": _sha(args.hardware),
            "mapping_path": str(args.mapping),
            "mapping_sha256": _sha(args.mapping),
            "simulation_path": str(args.simulation),
            "simulation_sha256": _sha(args.simulation),
        }),
        encoding="utf-8",
    )
    if args.mode in ("naive-preflight", "swizzle-preflight"):
        if args.report is not None:
            parser.error("--report requires the complete naive/swizzle suite")
        branch = "naive" if args.mode == "naive-preflight" else "swizzle"
        run_preflight = (
            run_naive_branch_preflight if branch == "naive" else run_swizzle_branch_preflight
        )
        evidence = run_preflight(
            provider=provider,
            finalizer=args.finalizer,
            resolver=args.resolver,
            npusim=args.npusim,
            simulation=args.simulation,
            runtime_root=args.runtime_root,
            case_ids=tuple(args.case),
            timeout=args.timeout,
        )
        print(
            "[SWIZZLE RUNTIME PREFLIGHT] PASS: timing_execution=1 "
            f"functional_execution=0 branch={branch} repeat=2 "
            f"cases={len(evidence)} hardware={args.hardware} mapping={args.mapping}"
        )
        return 0

    report = run_swizzle_runtime_suite(
        provider=provider,
        finalizer=args.finalizer,
        resolver=args.resolver,
        npusim=args.npusim,
        simulation=args.simulation,
        runtime_root=args.runtime_root,
        timeout=args.timeout,
        evidence_root=evidence_root,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(canonical_json(report), encoding="utf-8")
    print(
        "[SWIZZLE RUNTIME COMPARISON] PASS: timing_execution=1 "
        "functional_execution=0 cases=3 branches=6 repeat=2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SwizzleLowerLinkProvider",
    "SwizzleRuntimeExecutable",
    "SwizzleRuntimeStageFailure",
    "require_swizzle_lower_link_provider",
    "run_swizzle_branch_preflight",
    "run_swizzle_runtime_suite",
    "run_naive_branch_preflight",
    "validate_runtime_config_paths",
]
