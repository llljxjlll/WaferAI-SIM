#!/usr/bin/env python3
"""Run the 4-Die Swizzle scale matrix and report an honest no-benefit result."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT))

from llm.frontend.wafer_frontend.errors import SchemaError  # noqa: E402
from llm.frontend.wafer_frontend.policies.swizzle.cost import (  # noqa: E402
    interpolate_efficiency,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import (  # noqa: E402
    BufferOwnership,
    RecordOpcode,
)
from llm.frontend.wafer_frontend.schema.serde import (  # noqa: E402
    canonical_digest,
    canonical_json,
)
from llm.frontend.wafer_frontend.schema.swizzle_performance_evidence import (  # noqa: E402
    SWIZZLE_MINIMUM_QUALIFYING_SPEEDUP,
    SwizzleBenefitBranch,
)
from run_swizzle_runtime import (  # noqa: E402
    _D2D_LINK,
    _observe,
    _run,
    _sha,
    validate_runtime_config_paths,
)
from run_swizzle_scale_comparison import build_scale_official_matrix  # noqa: E402
from swizzle_scale_cases import build_first_green_swizzle_scale_cases  # noqa: E402
from swizzle_scale_comparison import build_swizzle_scale_comparison_suite  # noqa: E402
from swizzle_scale_runtime_evidence import (  # noqa: E402
    SwizzleScaleDirectionalLinkEvidence,
    SwizzleScaleRuntimeBranchEvidence,
    SwizzleScaleRuntimePairEvidence,
    SwizzleScaleRuntimeReport,
)
from swizzle_scale_runtime_provider import (  # noqa: E402
    PreparedSwizzleScaleBranch,
    ProductionSwizzleScaleProvider,
    build_actual_scale_program_io,
)


_D2D_LINK_PATTERN = re.compile(
    r"\[D2D_LINK\]\s+idx=(\d+)\s+die(\d+)->die(\d+)\s+dir=([^\s]+)\s+"
    r"req_in=(\d+)\s+req_out=(\d+)\s+ack_in=(\d+)\s+ack_out=(\d+)\s+"
    r"data_in=(\d+)\s+data_out=(\d+)"
)
_UNAVAILABLE_MARKERS = (
    "[SWIZZLE_INFLIGHT]",
    "[SWIZZLE_PORT_UTILIZATION]",
    "[SWIZZLE_COMPUTE_DTE_OVERLAP]",
)
_CAPABILITY_FLAGS = (
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


def _fragment(value: object) -> object:
    return getattr(value, "fragment", value)


def _manifest_counts(prepared: PreparedSwizzleScaleBranch) -> dict[str, object]:
    manifest = prepared.source.manifest
    record_count = sum(len(stream.records) for stream in manifest.core_streams)
    opcodes: Counter[RecordOpcode] = Counter()
    buffers: dict[str, object] = {}
    fragment_record_count = 0
    for linked in manifest.fragments:
        fragment = _fragment(linked)
        for stream in fragment.core_streams:
            fragment_record_count += len(stream.records)
            opcodes.update(record.opcode for record in stream.records)
        for abi in fragment.buffer_abi:
            previous = buffers.setdefault(abi.id, abi)
            if previous != abi:
                raise RuntimeError("shared BufferABI definition drifted")
    if fragment_record_count != record_count:
        raise RuntimeError("fragment records do not close linked core streams")
    ownership = Counter(abi.ownership for abi in buffers.values())
    return {
        "record_count": record_count,
        "opcode_counts": tuple(sorted((opcode.name, count) for opcode, count in opcodes.items())),
        "owned_buffer_count": ownership[BufferOwnership.OWNED],
        "borrowed_buffer_count": ownership[BufferOwnership.BORROWED],
        "aliased_buffer_count": ownership[BufferOwnership.ALIASED],
        "alloc_record_count": opcodes[RecordOpcode.SRAM_ALLOC_AT],
        "free_record_count": opcodes[RecordOpcode.SRAM_FREE],
        "event_record_count": opcodes[RecordOpcode.EVENT_SET] + opcodes[RecordOpcode.EVENT_WAIT],
    }


def _directional_links(output: str) -> tuple[SwizzleScaleDirectionalLinkEvidence, ...]:
    result = []
    for line in output.splitlines():
        match = _D2D_LINK_PATTERN.search(line)
        if match is None:
            continue
        (
            index,
            source,
            destination,
            direction,
            _request_in,
            request_out,
            _ack_in,
            ack_out,
            _data_in,
            data_out,
        ) = match.groups()
        result.append(
            SwizzleScaleDirectionalLinkEvidence(
                int(index),
                int(source),
                int(destination),
                direction,
                int(request_out),
                int(ack_out),
                int(data_out),
            )
        )
    links = tuple(sorted(result, key=lambda item: item.link_index))
    if len(links) != 8:
        raise RuntimeError("runtime did not emit the exact eight 2x2 directed links")
    return links


def _rank_output_slices(prepared: PreparedSwizzleScaleBranch, contract: object):
    rows = []
    for probe in contract.output_probes:
        target = probe.target
        rows.append(
            (
                target.runtime_core_id,
                (
                    tuple(target.tensor_slice.offset),
                    tuple(target.tensor_slice.shape),
                ),
            )
        )
    rows.sort()
    if len(rows) != 4 or len({core for core, _ in rows}) != 4:
        raise RuntimeError("official ProgramIo must probe one rank-local output per core")
    del prepared
    return tuple(item for _, item in rows)


def _cost_and_candidate(cases: tuple[object, ...], prepared: PreparedSwizzleScaleBranch):
    case = next(item for item in cases if item.point.name == prepared.case_plan.scale_name)
    decision = next(
        item for item in case.decisions if item.problem.pattern is prepared.case_plan.pattern
    )
    if prepared.branch_plan.branch is SwizzleBenefitBranch.NAIVE:
        candidate = decision.baseline
    else:
        candidate = next(
            item
            for item in decision.ranked_candidates
            if item.id == prepared.branch_plan.candidate_ref
        )
    return case, decision, candidate, candidate.cost


def _tile_efficiency(decision: object, prepared: PreparedSwizzleScaleBranch) -> float:
    tile = prepared.branch_plan.tile_shape
    if tile is None:
        rows, columns = decision.problem.group.logical_shape
        tile = (
            max(1, decision.problem.gemm.m // rows),
            max(1, decision.problem.gemm.n // columns),
            decision.problem.gemm.k,
        )
    return float(interpolate_efficiency(decision.problem.hardware_profile, tile))


def _run_external_branch(
    prepared: PreparedSwizzleScaleBranch,
    *,
    cases: tuple[object, ...],
    finalizer: Path,
    resolver: Path,
    npusim: Path,
    hardware: Path,
    simulation: Path,
    mapping: Path,
    runtime_root: Path,
    evidence_root: Path,
    timeout: int,
    formal: bool,
) -> SwizzleScaleRuntimeBranchEvidence | dict[str, object]:
    branch = prepared.branch_plan.branch
    label = (
        f"{prepared.case_plan.scale_name}-{prepared.case_plan.pattern.value}-"
        f"{branch.value}"
    )
    directory = evidence_root / label
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "linked.json"
    manifest_path.write_text(canonical_json(prepared.source.manifest), encoding="utf-8")
    artifacts = []
    reports = []
    for repeat in range(2):
        artifact = directory / f"program.{repeat}.npup"
        report = directory / f"finalizer.{repeat}.json"
        _run(
            [
                str(finalizer), "--input", str(manifest_path), "--output", str(artifact),
                "--report", str(report),
            ],
            cwd=runtime_root,
            timeout=120,
            stage=f"{label}.finalizer.{repeat}",
            stdout_path=directory / f"finalizer.{repeat}.stdout.txt",
        )
        artifacts.append(artifact.read_bytes())
        reports.append(json.loads(report.read_text(encoding="utf-8")))
    if artifacts[0] != artifacts[1] or reports[0] != reports[1]:
        raise RuntimeError(f"{label}: finalizer repeat drifted")
    artifact_sha = hashlib.sha256(artifacts[0]).hexdigest()
    if (
        reports[0].get("artifact_sha256") != artifact_sha
        or reports[0].get("linked_manifest_id") != prepared.source.manifest.id
        or reports[0].get("linked_manifest_digest")
        != canonical_digest(prepared.source.manifest)
    ):
        raise RuntimeError(f"{label}: finalizer report did not close manifest/artifact")
    contract = build_actual_scale_program_io(prepared, artifact_sha)
    sidecar_path = directory / "program_io.json"
    sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
    resolved = _run(
        [
            str(resolver), "--resolve", str(manifest_path),
            str(directory / "program.0.npup"), str(sidecar_path),
        ],
        cwd=runtime_root,
        timeout=120,
        stage=f"{label}.resolver",
        stdout_path=directory / "resolver.stdout.txt",
    )
    if (
        f"initializations={len(contract.initializations)}" not in resolved.stdout
        or f"probes={len(contract.output_probes)}" not in resolved.stdout
    ):
        raise RuntimeError(f"{label}: resolver counts drifted")
    observations = []
    outputs = []
    for repeat in range(2):
        execution = _run(
            [
                str(npusim), "--program", str(directory / "program.0.npup"),
                "--linked-manifest", str(manifest_path), "--program-io", str(sidecar_path),
                "--hardware-config", str(hardware), "--simulation-config", str(simulation),
                "--mapping-config", str(mapping), "--trace-window", "1000000",
            ],
            cwd=runtime_root,
            timeout=timeout,
            stage=f"{label}.npusim.{repeat}",
            stdout_path=directory / f"npusim.{repeat}.stdout.txt",
        )
        outputs.append(execution.stdout)
        observations.append(_observe(execution.stdout, artifact_sha, contract))
    if observations[0] != observations[1]:
        raise RuntimeError(f"{label}: npusim repeat drifted")
    links = _directional_links(outputs[0])
    if links != _directional_links(outputs[1]):
        raise RuntimeError(f"{label}: directional link repeat drifted")
    if any(marker in output for marker in _UNAVAILABLE_MARKERS for output in outputs):
        raise RuntimeError(
            f"{label}: new measurement markers require a reviewed schema update"
        )
    summary = {
        "label": label,
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": len(artifacts[0]),
        "repeat_makespans": tuple(item.makespan_cycles for item in observations),
        "repeat_marker_digests": tuple(item.marker_digest for item in observations),
        "packet_count": observations[0].packet_count,
        "initialization_count": len(contract.initializations),
        "probe_count": len(contract.output_probes),
    }
    if not formal:
        return summary
    case, decision, candidate, cost = _cost_and_candidate(cases, prepared)
    counts = _manifest_counts(prepared)
    result = SwizzleScaleRuntimeBranchEvidence.create(
        scale_name=prepared.case_plan.scale_name,
        scale_ordinal=prepared.case_plan.scale_ordinal,
        pattern=prepared.case_plan.pattern,
        branch=branch,
        algorithm=prepared.branch_plan.algorithm,
        same_work_digest=prepared.branch_plan.same_work_digest,
        problem_shape=(
            decision.problem.gemm.m,
            decision.problem.gemm.n,
            decision.problem.gemm.k,
        ),
        rank_output_slices=_rank_output_slices(prepared, contract),
        topology=candidate.topology_witness.kind.value,
        chunk_count=prepared.branch_plan.chunk_count,
        unroll_degree=prepared.branch_plan.unroll_degree,
        tile_shape=prepared.branch_plan.tile_shape,
        tile_efficiency=_tile_efficiency(decision, prepared),
        logical_bytes=cost.logical_bytes,
        byte_hops=cost.byte_hops,
        gemm_flops=decision.problem.gemm.flops,
        predicted_cycles=(cost.lower_cycles, cost.estimated_cycles, cost.upper_cycles),
        predicted_phases=(cost.prologue_cycles, cost.steady_cycles, cost.epilogue_cycles),
        repeat_makespans=summary["repeat_makespans"],
        repeat_marker_digests=summary["repeat_marker_digests"],
        packet_count=summary["packet_count"],
        directional_links=links,
        observed_max_inflight_send=None,
        observed_max_inflight_recv=None,
        directional_port_utilization_over_time=None,
        compute_dte_overlap_cycles=None,
        **counts,
        sram_high_water_bytes=cost.sram_high_water_bytes,
        artifact_size_bytes=summary["artifact_size_bytes"],
        artifact_sha256=artifact_sha,
        manifest_id=prepared.source.manifest.id,
        program_io_id=contract.id,
        initialization_count=summary["initialization_count"],
        probe_count=summary["probe_count"],
        economic_auto_selected=prepared.branch_plan.economic_auto_selected,
        forced_deployment=prepared.branch_plan.forced_deployment,
        decision_reason=decision.decision_reason.value,
        deployment_reason=("economic_auto" if branch is SwizzleBenefitBranch.SWIZZLE_AUTO else "unfused_baseline"),
        finalizer_sha256=_sha(finalizer),
        resolver_sha256=_sha(resolver),
        npusim_sha256=_sha(npusim),
        measurement_complete=False,
        missing_runtime_markers=(
            "compute_dte_overlap_cycles",
            "directional_port_utilization_over_time",
            "observed_max_inflight_recv",
            "observed_max_inflight_send",
        ),
    )
    (directory / "branch_evidence.json").write_text(
        canonical_json(result), encoding="utf-8"
    )
    del case
    return result


def _pairs(branches: tuple[SwizzleScaleRuntimeBranchEvidence, ...]):
    result = []
    for naive, auto in zip(branches[::2], branches[1::2], strict=True):
        speedup = float(naive.repeat_makespans[0] / auto.repeat_makespans[0])
        result.append(
            SwizzleScaleRuntimePairEvidence(
                scale_name=naive.scale_name,
                pattern=naive.pattern,
                same_work_digest=naive.same_work_digest,
                naive_ref=naive.id,
                swizzle_auto_ref=auto.id,
                speedup=speedup,
                threshold=SWIZZLE_MINIMUM_QUALIFYING_SPEEDUP,
                qualifies=False,
                bottlenecks=("control_setup_overhead", "matmul_setup_overhead"),
            )
        )
    return tuple(result)


def run(args: argparse.Namespace) -> SwizzleScaleRuntimeReport:
    for name in ("finalizer", "resolver", "npusim", "hardware", "simulation", "mapping"):
        path = getattr(args, name)
        if not path.is_file():
            raise RuntimeError(f"{name} is not a file: {path}")
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    validate_runtime_config_paths(
        hardware_json=args.hardware.read_text(encoding="utf-8"),
        simulation_json=args.simulation.read_text(encoding="utf-8"),
        runtime_root=args.runtime_root,
    )
    cases = build_first_green_swizzle_scale_cases()
    suite = build_swizzle_scale_comparison_suite(cases)
    matrix = build_scale_official_matrix(suite)
    provider = ProductionSwizzleScaleProvider(cases=cases, suite=suite)

    forced = []
    for target in matrix.forced_preflight:
        prepared = provider.prepare(target.case_plan, target.branch_plan.branch)
        forced.append(
            _run_external_branch(
                prepared,
                cases=cases,
                finalizer=args.finalizer,
                resolver=args.resolver,
                npusim=args.npusim,
                hardware=args.hardware,
                simulation=args.simulation,
                mapping=args.mapping,
                runtime_root=args.runtime_root,
                evidence_root=args.evidence_root / "forced-diagnostic",
                timeout=args.timeout,
                formal=False,
            )
        )
    (args.evidence_root / "forced-diagnostic-summary.json").write_text(
        canonical_json(tuple(forced)), encoding="utf-8"
    )

    branches = []
    for target in matrix.official:
        prepared = provider.prepare(target.case_plan, target.branch_plan.branch)
        branch = _run_external_branch(
            prepared,
            cases=cases,
            finalizer=args.finalizer,
            resolver=args.resolver,
            npusim=args.npusim,
            hardware=args.hardware,
            simulation=args.simulation,
            mapping=args.mapping,
            runtime_root=args.runtime_root,
            evidence_root=args.evidence_root / "official",
            timeout=args.timeout,
            formal=True,
        )
        if type(branch) is not SwizzleScaleRuntimeBranchEvidence:
            raise RuntimeError("official branch did not produce typed evidence")
        branches.append(branch)
        print(
            f"[SWIZZLE SCALE CASE] {branch.scale_name}/{branch.pattern.value}/"
            f"{branch.branch.value} makespan={branch.repeat_makespans[0]} "
            f"artifact_sha256={branch.artifact_sha256}",
            flush=True,
        )
    report = SwizzleScaleRuntimeReport.create(
        branches=tuple(branches),
        pairs=_pairs(tuple(branches)),
        capability_flags=_CAPABILITY_FLAGS,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(canonical_json(report), encoding="utf-8")
    speedups = ",".join(f"{item.speedup:.6f}" for item in report.pairs)
    print(
        "[SWIZZLE SCALE RUNTIME] PASS: performance_benefit=0 "
        "measurement_complete=0 cases=3 branches=6 repeat=2 "
        f"speedups={speedups}",
        flush=True,
    )
    return report


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    for name in (
        "finalizer", "resolver", "npusim", "hardware", "simulation", "mapping",
        "runtime_root", "evidence_root", "report",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        run(_parse_args(sys.argv[1:] if argv is None else argv))
        return 0
    except Exception as error:
        print(f"Swizzle scale runtime first failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
