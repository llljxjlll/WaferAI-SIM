"""Bounded production C1/C2 measured-profile MoE Swizzle regression.

This entry intentionally runs only the selected AUTO workload for the four
required scale/mode scenarios.  It consumes one frozen MEASURED calibration
profile, rebuilds the exact joint-selection frontier, lowers one whole linked
program, finalizes it, derives actual-SHA ProgramIo, and executes matching
npusim.  It is not the broad W11/W12 planner matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.lowering.moe_swizzle_workload_linker import (
    build_moe_swizzle_standard_linked_program,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_pair_feasibility import (
    build_moe_swizzle_pair_feasibility_witness,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_execution import (
    build_moe_swizzle_execution,
)
from llm.frontend.wafer_frontend.passes.discover_moe_swizzle import (
    discover_moe_swizzle_regions,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_program_io import (
    build_moe_swizzle_program_io,
)
from llm.frontend.wafer_frontend.passes.build_moe_swizzle_workload_state_abi import (
    build_moe_swizzle_workload_state_abi,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_cost import (
    build_moe_swizzle_pair_cost_context,
    decide_moe_swizzle,
    estimate_moe_swizzle_materialized_pair_cycles,
    select_moe_swizzle_workload_deployment,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_comet_mesh import (
    build_comet_mesh_moe_candidate_grid,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_direct_xy import (
    build_direct_xy_moe_candidates,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_problem import (
    build_moe_swizzle_problem,
)
from llm.frontend.wafer_frontend.policies.swizzle.moe_unfused import (
    build_executable_moe_unfused_baseline,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
    load_json_dataclass,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_calibration import (
    MoeCalibrationStatus,
    MoeSwizzleCalibrationProfile,
)
from llm.frontend.wafer_frontend.schema.swizzle_moe_execution import (
    MoeScaleExecutionMode,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle_moe_standard import (
    MoeSwizzleStandardLinkedProgram,
)
from llm.test.frontend.integration.moe_swizzle_runtime_markers import (
    parse_moe_swizzle_runtime_markers,
)
from llm.test.frontend.integration.moe_swizzle_runtime_suite import (
    MoeSwizzleRuntimeScope,
    build_moe_swizzle_runtime_case_plan,
)
from llm.test.frontend.integration.moe_swizzle_scale_cases import (
    build_moe_swizzle_scale_cases,
)
from llm.test.frontend.integration.run_moe_swizzle_calibration import (
    prepare_moe_swizzle_calibration_dramsys_runtime,
)


_SCENARIOS = {
    "c1-infer": ("C1", MoeScaleExecutionMode.INFER_FORWARD),
    "c1-train": ("C1", MoeScaleExecutionMode.TRAIN_FORWARD),
    "c2-infer": ("C2", MoeScaleExecutionMode.INFER_FORWARD),
    "c2-train": ("C2", MoeScaleExecutionMode.TRAIN_FORWARD),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed exit={completed.returncode}: {' '.join(command)}\n"
            + completed.stdout[-4000:]
        )
    return completed.stdout


def _select(case, mode, profile, ir1):
    execution = build_moe_swizzle_execution(case.spec, case.oracle, mode)
    regions = discover_moe_swizzle_regions(
        case.spec, case.oracle, execution,
    )
    decisions = []
    for region in regions:
        problem = build_moe_swizzle_problem(
            region,
            case.spec,
            case.oracle,
            execution,
            hardware_facts=case.hardware_facts,
            endpoint_session_contract=case.endpoint_session_contract,
        )
        baseline = build_executable_moe_unfused_baseline(
            problem, case.spec, case.oracle, execution,
            calibration_profile=profile,
        )
        fused = build_direct_xy_moe_candidates(
            problem, case.spec, case.oracle, execution,
            calibration_profile=profile,
        ) + build_comet_mesh_moe_candidate_grid(
            problem, case.spec, case.oracle, execution,
            calibration_profile=profile,
        )
        decisions.append(decide_moe_swizzle(problem, baseline, fused))
    ordered = tuple(
        next(item for item in decisions if item.problem.region.pattern is pattern)
        for pattern in (
            FusionPattern.MOE_DISPATCH_GEMM,
            FusionPattern.MOE_GEMM_COMBINE,
        )
    )
    dispatch, combine = ordered
    context = build_moe_swizzle_pair_cost_context(dispatch, combine)
    argmin = lambda decision: min(
        decision.ranked_candidates,
        key=lambda candidate: (candidate.cost.estimated_cycles, candidate.id),
    ).id
    baseline_pair = (dispatch.baseline.id, combine.baseline.id)
    fused_pair = (argmin(dispatch), argmin(combine))
    required_modes = {
        (dispatch_ref, combine_ref)
        for dispatch_ref in (baseline_pair[0], fused_pair[0])
        for combine_ref in (baseline_pair[1], fused_pair[1])
    }
    state_abi = build_moe_swizzle_workload_state_abi(
        ir1, execution, case.spec, case.oracle,
    )
    witness_by_pair = {}
    frontier = []
    best = float("inf")
    for pair, lower_bound in context.bounds:
        if lower_bound >= best:
            break
        witness = build_moe_swizzle_pair_feasibility_witness(
            ir1,
            execution,
            case.spec,
            ordered,
            state_abi,
            pair,
        )
        witness_by_pair[pair] = witness
        frontier.append(pair)
        if witness.feasible:
            best = min(
                best,
                estimate_moe_swizzle_materialized_pair_cycles(
                    dispatch, combine, witness, context=context,
                ),
            )
    for pair in sorted(required_modes - set(witness_by_pair)):
        witness_by_pair[pair] = build_moe_swizzle_pair_feasibility_witness(
            ir1,
            execution,
            case.spec,
            ordered,
            state_abi,
            pair,
        )
    witnesses = tuple(witness_by_pair[pair] for pair in sorted(witness_by_pair))
    selection = select_moe_swizzle_workload_deployment(
        dispatch, combine, witnesses,
    )
    workload = build_moe_swizzle_runtime_case_plan(
        case,
        mode,
        profile,
        scope=MoeSwizzleRuntimeScope.WORKLOAD,
        workload_selections=(selection,),
    )
    return workload, selection, tuple(frontier)


def _scenario(
    *, name: str, profile, ir1, cases, output_root: Path, finalizer: Path,
    npusim: Path, hardware: Path, simulation: Path, mapping: Path,
    timeout: int,
) -> dict[str, object]:
    scale, mode = _SCENARIOS[name]
    case = next(item for item in cases if item.spec.name == scale)
    root = output_root / name
    root.mkdir(parents=False, exist_ok=False)
    timings = {}
    started = time.monotonic()
    workload, selection, frontier = _select(case, mode, profile, ir1)
    timings["selection_seconds"] = time.monotonic() - started
    _write(root / "selection.json", selection)

    started = time.monotonic()
    linked = build_moe_swizzle_standard_linked_program(
        ir1,
        case.spec,
        case.oracle,
        workload.execution,
        workload.economic_decisions,
        selection,
        case.hardware_facts,
    )
    timings["link_seconds"] = time.monotonic() - started
    _write(root / "linked_manifest.json", linked.manifest)
    _write(root / "prefinal.wrapper.json", linked)

    started = time.monotonic()
    finalizer_output = _run(
        (
            str(finalizer),
            "--input", str(root / "linked_manifest.json"),
            "--output", str(root / "program.npup"),
            "--report", str(root / "finalizer.json"),
        ),
        cwd=root,
        timeout=timeout,
    )
    (root / "finalizer.stdout").write_text(finalizer_output, encoding="utf-8")
    timings["finalizer_seconds"] = time.monotonic() - started
    artifact_sha = _sha256(root / "program.npup")
    program_io = build_moe_swizzle_program_io(linked, artifact_sha)
    exact_semantic = linked._semantic()
    exact_semantic["program_io"] = program_io
    exact = MoeSwizzleStandardLinkedProgram.create(**exact_semantic)
    _write(root / "program_io.json", program_io)
    _write(root / "finalized.wrapper.json", exact)

    started = time.monotonic()
    output = _run(
        (
            str(npusim),
            f"--program={root / 'program.npup'}",
            f"--linked-manifest={root / 'linked_manifest.json'}",
            f"--program-io={root / 'program_io.json'}",
            f"--hardware-config={hardware}",
            f"--simulation-config={simulation}",
            f"--mapping-config={mapping}",
            "--moe-swizzle-runtime-markers",
        ),
        cwd=root,
        timeout=timeout,
    )
    (root / "raw.stdout").write_text(output, encoding="utf-8")
    timings["npusim_seconds"] = time.monotonic() - started
    marker = parse_moe_swizzle_runtime_markers(output)
    if "[PROGRAM_IO] phase=verify" not in output or "pass=1" not in output:
        raise RuntimeError("npusim did not emit a passing ProgramIo verification")
    window_cycles = {
        item.window_cycles for item in marker.die_compute_dte_overlaps
    }
    if len(window_cycles) != 1:
        raise RuntimeError("runtime marker lacks one exact whole-program window")
    return {
        "scenario": name,
        "profile_id": profile.id,
        "case_id": workload.id,
        "selection_id": selection.id,
        "selected_candidate_refs": (
            selection.selected_dispatch_candidate_ref,
            selection.selected_combine_candidate_ref,
        ),
        "baseline_cycles": selection.baseline_estimated_cycles,
        "selected_cycles": selection.selected_estimated_cycles,
        "frontier_count": len(frontier),
        "linked_program_id": exact.id,
        "manifest_id": exact.manifest.id,
        "program_io_id": program_io.id,
        "program_sha256": artifact_sha,
        "raw_sha256": _sha256(root / "raw.stdout"),
        "makespan_cycles": next(iter(window_cycles)),
        "marker_digest": marker.marker_digest,
        "timings": timings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scenario", action="append", choices=tuple(_SCENARIOS))
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    paths = (
        args.profile, args.finalizer, args.npusim, args.hardware,
        args.simulation, args.mapping,
    )
    if any(not item.is_absolute() or not item.is_file() for item in paths):
        raise SchemaError("all input paths must be absolute regular files", path="measured_e2e")
    if _sha256(args.profile) != args.profile_sha256:
        raise SchemaError("measured profile SHA mismatch", path="measured_e2e.profile")
    profile = load_json_dataclass(MoeSwizzleCalibrationProfile, args.profile)
    profile.validate("measured_e2e.profile")
    if profile.status is not MoeCalibrationStatus.MEASURED:
        raise SchemaError("requires MEASURED profile", path="measured_e2e.profile")
    expected_hashes = (
        profile.tool_sha256, profile.hardware_sha256,
        profile.simulation_sha256, profile.mapping_sha256,
    )
    actual_hashes = tuple(_sha256(item) for item in (
        args.npusim, args.hardware, args.simulation, args.mapping,
    ))
    if actual_hashes != expected_hashes:
        raise SchemaError("runtime/config SHA differs from measured profile", path="measured_e2e")
    args.output_root.mkdir(parents=True, exist_ok=False)
    prepare_moe_swizzle_calibration_dramsys_runtime(
        npusim=args.npusim, runtime_root=args.output_root,
    )
    cases = build_moe_swizzle_scale_cases()
    c0 = cases[0].c0_production_case
    assert c0 is not None
    ir1 = c0.forward.n4.graph
    scenarios = tuple(args.scenario or _SCENARIOS)
    rows = tuple(
        _scenario(
            name=name, profile=profile, ir1=ir1, cases=cases,
            output_root=args.output_root, finalizer=args.finalizer,
            npusim=args.npusim, hardware=args.hardware,
            simulation=args.simulation, mapping=args.mapping,
            timeout=args.timeout,
        )
        for name in scenarios
    )
    summary = {
        "profile_id": profile.id,
        "profile_sha256": args.profile_sha256,
        "scenarios": rows,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
