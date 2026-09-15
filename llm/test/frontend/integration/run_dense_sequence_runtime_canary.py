"""Run a strict Dense Prefill -> Decode -> Decode sequence canary."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time

from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    compile_dense_e2e_sequence_runtime_profiles,
)
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RegionManifest
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadInferenceSteps,
    WorkloadMeshSpec,
    WorkloadParallelSpec,
    WorkloadRunCapability,
    WorkloadRunRequest,
    WorkloadStepSpec,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_dense_compile_sequence import (
    _one_die_case,
    _two_by_two_case,
)
from llm.test.frontend.unit.test_legacy_dense_backend import (
    _capability, _legacy_spec, _request,
)

from .flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)


_ROOT = Path(__file__).resolve().parents[4]
_SIX_DIE_SRAM_BYTES = 128 * 1024
_SIX_DIE_SRAM_ALIGNMENT_BYTES = 32


def _four_die_rect_case(rows: int, columns: int):
    if (rows, columns) not in ((1, 4), (4, 1)):
        raise ValueError("four-die rectangular canary requires 1x4 or 4x1")
    original, template, _ = _two_by_two_case()
    request = original.request
    changed = WorkloadRunRequest.create(
        family=request.family,
        model=request.model,
        steps=request.steps,
        mesh=WorkloadMeshSpec(rows, columns),
        parallel=request.parallel,
        memory=request.memory,
        optimizer=request.optimizer,
        execution=request.execution,
    )
    baseline = _capability()
    capability = WorkloadRunCapability.create(
        max_mesh_rows=4,
        max_mesh_columns=4,
        max_mesh_ranks=4,
        families=baseline.families,
    )
    capacities = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{die_id}",
            base_address=0,
            capacity_bytes=1 << 30,
            alignment_bytes=64,
        )
        for die_id in range(4)
    )
    manifest = materialize_workload_preflight(
        changed, capability, capacities=capacities
    )
    fabric = physical_fabric_from_data(
        minimal_hardware(columns, rows, sram_bytes=65536)
    )
    return manifest, template, fabric


def _six_die_fixed_model_case(rows: int, columns: int):
    """One unchanged two-layer TP6 model, including a 3x3 idle-middle row."""

    if (rows, columns) not in ((2, 3), (3, 2), (1, 6), (6, 1), (3, 3)):
        raise ValueError("fixed-model TP6 requires a six-die rectangle or 3x3")
    physical_dies = rows * columns
    ranks = 6
    active_dies = (
        (0, 1, 2, 6, 7, 8) if (rows, columns) == (3, 3)
        else tuple(range(ranks))
    )
    base = _request(layers=2, prefill=6, decode=2)
    model = replace(
        base.model, hidden_size=48, intermediate_size=96,
        num_attention_heads=6, num_kv_heads=6, head_dim=8,
    )
    steps = WorkloadStepSpec(inference=WorkloadInferenceSteps(
        prefill_tokens=6, decode_steps=2, request_count=6,
    ))
    request = WorkloadRunRequest.create(
        family=base.family, model=model, steps=steps,
        mesh=WorkloadMeshSpec(rows, columns),
        parallel=WorkloadParallelSpec(tp=ranks, active_die_ids=active_dies),
        memory=base.memory, execution=base.execution,
    )
    baseline = _capability()
    capability = WorkloadRunCapability.create(
        max_mesh_rows=max(rows, 3), max_mesh_columns=max(columns, 3),
        max_mesh_ranks=physical_dies,
        families=baseline.families,
    )
    capacities = tuple(MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref=f"die:{rank}",
        base_address=0, capacity_bytes=1 << 30, alignment_bytes=64,
    ) for rank in range(physical_dies))
    manifest = materialize_workload_preflight(
        request, capability, capacities=capacities,
    )
    template = _legacy_spec(layers=2, prefill=6, decode=0)
    template = replace(
        template,
        model=replace(template.model, H=48, I=96, NH=6, KVH=6, DH=8),
        parallel=replace(template.parallel, instances=(
            replace(template.parallel.instances[0], tp=ranks, sp=True),
        )),
    )
    template.validate()
    compiled_hardware = minimal_hardware(
        columns, rows, sram_bytes=_SIX_DIE_SRAM_BYTES
    )
    # Decode's 96B TP reduce inputs are tightly packed at 32B boundaries.
    # Fixed SRAM_ALLOC_AT uses the same physical alignment in NpuSim.
    compiled_hardware["memory"]["sram"][
        "allocation_alignment_bytes"
    ] = _SIX_DIE_SRAM_ALIGNMENT_BYTES
    fabric = physical_fabric_from_data(compiled_hardware)
    return manifest, template, fabric


def _all_die_scaled_model_case(rows: int, columns: int):
    """One two-layer Dense source with every physical Die doing TP work."""

    if not (1 <= rows <= 10 and 1 <= columns <= 10):
        raise ValueError("scaled all-die case requires the 1..10 release envelope")
    ranks = rows * columns
    base = _request(layers=2, prefill=2, decode=2)
    model = replace(
        base.model,
        vocabulary_size=max(128, ranks),
        hidden_size=16 * ranks,
        intermediate_size=16 * ranks,
        num_attention_heads=ranks,
        num_kv_heads=ranks,
        head_dim=16,
    )
    request = WorkloadRunRequest.create(
        family=base.family,
        model=model,
        steps=WorkloadStepSpec(inference=WorkloadInferenceSteps(
            prefill_tokens=1,
            decode_steps=2,
            request_count=ranks,
        )),
        mesh=WorkloadMeshSpec(rows, columns),
        parallel=WorkloadParallelSpec(
            tp=ranks,
            active_die_ids=tuple(range(ranks)),
        ),
        memory=base.memory,
        execution=base.execution,
    )
    baseline = _capability()
    capability = WorkloadRunCapability.create(
        max_mesh_rows=max(rows, 3),
        max_mesh_columns=max(columns, 3),
        max_mesh_ranks=ranks,
        families=baseline.families,
    )
    capacities = tuple(MemoryTierCapacity.create(
        tier=MemoryTier.HBM,
        location_ref=f"die:{die_id}",
        base_address=0,
        capacity_bytes=1 << 30,
        alignment_bytes=64,
    ) for die_id in range(ranks))
    manifest = materialize_workload_preflight(
        request, capability, capacities=capacities,
    )
    template = _legacy_spec(layers=2, prefill=2, decode=0)
    template = replace(
        template,
        model=replace(
            template.model,
            V=model.vocabulary_size,
            H=model.hidden_size,
            I=model.intermediate_size,
            NH=model.num_attention_heads,
            KVH=model.num_kv_heads,
            DH=model.head_dim,
            rotary_dim=16,
        ),
        parallel=replace(template.parallel, instances=(
            replace(template.parallel.instances[0], tp=ranks, sp=False),
        )),
    )
    template.validate()
    sram_bytes = _SIX_DIE_SRAM_BYTES
    hardware = minimal_hardware(columns, rows, sram_bytes=sram_bytes)
    hardware["memory"]["sram"]["allocation_alignment_bytes"] = 32
    fabric = physical_fabric_from_data(hardware)
    return manifest, template, fabric


def _run(command: tuple[str, ...], *, cwd: Path, timeout: int,
         failure_log: Path | None = None) -> str:
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
        if failure_log is not None:
            failure_log.write_text(completed.stdout, encoding="utf-8")
        raise RuntimeError(
            f"returncode={completed.returncode}: {' '.join(command)}\n"
            f"stdout_file={failure_log if failure_log is not None else 'none'}\n"
            f"stdout_tail={completed.stdout[-1600:]}"
        )
    return completed.stdout


def _source_tool_snapshot(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """Bind loaded repository Python and the exact executable bytes in use."""

    sources: dict[str, str] = {}
    tracked_roots = (
        _ROOT / "llm/frontend/wafer_frontend",
        _ROOT / "llm/test/frontend",
    )
    for module in tuple(sys.modules.values()):
        module_path = getattr(module, "__file__", None)
        if type(module_path) is not str or not module_path.endswith(".py"):
            continue
        path = Path(module_path).resolve()
        if not any(path.is_relative_to(root) for root in tracked_roots):
            continue
        sources[str(path.relative_to(_ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    sources[str(Path(__file__).resolve().relative_to(_ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    tools = {
        name: hashlib.sha256(getattr(args, name).resolve().read_bytes()).hexdigest()
        for name in ("finalizer", "resolver", "npusim", "simulation")
    }
    return {"imported_python_sha256": dict(sorted(sources.items())),
            "tool_sha256": dict(sorted(tools.items()))}


def run(args: argparse.Namespace) -> None:
    source_tool_at_entry = _source_tool_snapshot(args)
    rows, columns = (int(dimension) for dimension in args.mesh_size.split("x"))
    if args.scaled_all_dies:
        manifest, template, fabric = _all_die_scaled_model_case(rows, columns)
    elif args.mesh_size == "1x1":
        manifest, template, fabric = _one_die_case()
    elif args.mesh_size == "2x2":
        manifest, template, fabric = _two_by_two_case()
    elif args.mesh_size in ("2x3", "3x2", "1x6", "6x1", "3x3"):
        manifest, template, fabric = _six_die_fixed_model_case(rows, columns)
    else:
        manifest, template, fabric = _four_die_rect_case(rows, columns)
    hbm_address_spaces = valid_hbm_address_spaces(fabric)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def _compile_budget_expired(_signal: int, _frame: object) -> None:
        raise TimeoutError(f"Dense compile exceeded {args.compile_timeout}s")

    original_alarm = signal.signal(signal.SIGALRM, _compile_budget_expired)
    signal.alarm(args.compile_timeout)
    try:
        with builder_validation_session():
            sequence, linked_profiles = compile_dense_e2e_sequence_runtime_profiles(
                manifest, template, fabric,
                hbm_address_spaces=hbm_address_spaces,
                intra_die_wire_address_limit_bytes=(
                    65536 if args.scaled_all_dies
                    or manifest.request.parallel.tp == 6 else None
                ),
            )
    except Exception as error:
        (output / "compile_failure.json").write_text(json.dumps({
            "mesh": args.mesh_size,
            "workload_case_id": manifest.request.case_id,
            "source_request_sha256": canonical_digest(manifest.request),
            "phase": "production_compile",
            "runtime_status": "not_measured",
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_seconds": round(time.monotonic() - started, 3),
            "frontend_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "source_tool_at_entry": source_tool_at_entry,
        }, indent=2, sort_keys=True), encoding="utf-8")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_alarm)
    sequence.validate()
    active_dies = set(manifest.placement.active_die_ids)
    compiled_core_die_ids = tuple(
        tuple(sorted({stream.logical_core.die_id
                      for stream in segment.linked_manifest.core_streams}))
        for segment in sequence.segments
    )
    if any(set(die_ids) != active_dies for die_ids in compiled_core_die_ids):
        raise RuntimeError(
            "compiled core streams do not exactly cover active workload Dies: "
            f"active={sorted(active_dies)}, observed={compiled_core_die_ids}"
        )
    (output / "compiled_receipt.json").write_text(json.dumps({
        "mesh": args.mesh_size,
        "scaled_all_dies": args.scaled_all_dies,
        "active_die_ids": manifest.placement.active_die_ids,
        "idle_die_ids": manifest.placement.idle_die_ids,
        "compiled_core_die_ids": compiled_core_die_ids,
        "workload_case_id": manifest.request.case_id,
        "source_request_sha256": canonical_digest(manifest.request),
        "sequence_digest": sequence.digest,
        "linked_manifest_sha256": [canonical_digest(segment.linked_manifest)
                                   for segment in sequence.segments],
        "wall_seconds": round(time.monotonic() - started, 3),
        "frontend_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runtime_status": "not_measured",
        "intra_die_wire_address_limit_bytes": (
            65536 if args.scaled_all_dies
            or manifest.request.parallel.tp == 6 else None
        ),
    }, indent=2, sort_keys=True), encoding="utf-8")
    manifests: list[Path] = []
    programs: list[Path] = []
    sidecars: list[Path] = []
    artifact_digests: list[str] = []
    for index, segment in enumerate(sequence.segments):
        manifest_path = output / f"segment_{index}.linked.json"
        artifact_path = output / f"segment_{index}.npup"
        report_path = output / f"segment_{index}.finalizer.json"
        manifest_path.write_text(
            canonical_json(segment.linked_manifest), encoding="utf-8"
        )
        _run(
            (
                str(args.finalizer.resolve()),
                "--input",
                str(manifest_path),
                "--output",
                str(artifact_path),
                "--report",
                str(report_path),
            ),
            cwd=output,
            timeout=120,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if (
            report.get("artifact_sha256") != artifact_digest
            or report.get("linked_manifest_id") != segment.linked_manifest.id
            or report.get("linked_manifest_digest")
            != canonical_digest(segment.linked_manifest)
        ):
            raise RuntimeError(f"segment {index} finalizer closure failed")
        manifests.append(manifest_path)
        programs.append(artifact_path)
        artifact_digests.append(artifact_digest)
        abi_by_binding = {
            abi.hbm_binding_ref: abi
            for fragment in linked_profiles[index].manifest.fragments
            for abi in (
                fragment.fragment.state_abi
                if isinstance(fragment, RegionManifest)
                else fragment.state_abi
            )
        }
        first_access: dict[str, StateUseAccess] = {}
        for action in linked_profiles[index].lowering_context.global_dag.actions:
            for use in action.state_uses:
                first_access.setdefault(use.hbm_binding_ref, use.access)
        state_seeds = {
            abi_by_binding[binding].state_ref: bytes(
                abi_by_binding[binding].size_bytes
            )
            for binding, access in first_access.items()
            if access is StateUseAccess.READ
        }
        contract = build_timing_program_io(
            linked_profiles[index],
            artifact_digest,
            state_seed_overrides=state_seeds,
        )
        contract.validate_against(segment.linked_manifest)
        sidecar_path = output / f"segment_{index}.program_io.json"
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        resolved = _run(
            (
                str(args.resolver.resolve()), "--resolve",
                str(manifest_path), str(artifact_path), str(sidecar_path),
            ),
            cwd=args.resolver.resolve().parent,
            timeout=min(args.timeout, 900),
        )
        (output / f"segment_{index}.resolver.stdout.txt").write_text(
            resolved, encoding="utf-8",
        )
        if (f"initializations={len(contract.initializations)}" not in resolved
                or f"probes={len(contract.output_probes)}" not in resolved):
            raise RuntimeError(f"segment {index} native ProgramIO resolver closure failed")
        sidecars.append(sidecar_path)

    hardware_path = output / "hardware.json"
    mapping_path = output / "mapping.spec"
    hardware = json.loads(specialize_p5_large_release_hardware(rows, columns))
    sram_bytes = (
        _SIX_DIE_SRAM_BYTES if args.scaled_all_dies
        or manifest.request.parallel.tp == 6 else 65536
    )
    sram_alignment = (
        32 if args.scaled_all_dies else
        _SIX_DIE_SRAM_ALIGNMENT_BYTES
        if manifest.request.parallel.tp == 6 else 64
    )
    hardware["memory"]["sram_size"] = sram_bytes
    hardware["memory"]["sram"]["capacity_bytes"] = sram_bytes
    hardware["memory"]["sram"]["allocation_alignment_bytes"] = sram_alignment
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = sram_bytes
    address_spaces_by_die = {
        space.die_id: space for space in hbm_address_spaces
    }
    for stack in hardware["memory_system"]["hbm_stacks"]:
        space = address_spaces_by_die[stack["compute_die_id"]]
        stack["capacity_bytes"] = space.size_bytes
    hardware["memory_system"]["address_policy"]["home_ranges"] = [
        {
            "die_id": space.die_id,
            "base": space.base_address,
            "size_bytes": space.size_bytes,
        }
        for space in hbm_address_spaces
    ]
    hardware["memory_system"]["address_policy"]["stack_interleave_bytes"] = max(
        space.size_bytes for space in hbm_address_spaces
    )
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    preflight = _run(
        (str(args.resolver.resolve()), "--validate-hardware-sram",
         str(hardware_path), str(sram_bytes), str(sram_alignment)),
        cwd=args.resolver.resolve().parent, timeout=60,
    )
    (output / "hardware_sram_preflight.stdout.txt").write_text(
        preflight, encoding="utf-8",
    )
    if (f"region=sram capacity_bytes={sram_bytes} "
            f"alignment_bytes={sram_alignment} region_count=1"
            not in preflight):
        raise RuntimeError("native SRAM hardware profile differs from compiled fabric")
    mapping_path.write_text("0:0\n", encoding="utf-8")
    try:
        runtime_output = _run(
        (
            str(args.npusim.resolve()),
            "--program-sequence",
            ",".join(str(path) for path in programs),
            "--linked-manifest-sequence",
            ",".join(str(path) for path in manifests),
            "--program-io-sequence",
            ",".join(str(path) for path in sidecars),
            "--hardware-config",
            str(hardware_path),
            "--simulation-config",
            str(args.simulation.resolve()),
            "--mapping-config",
            str(mapping_path),
            "--trace-window",
            "1000000",
        ),
        cwd=output,
            timeout=args.timeout,
            failure_log=output / "npusim.failure.stdout.txt",
        )
    except Exception as error:
        (output / "runtime_failure.json").write_text(json.dumps({
            "mesh": args.mesh_size,
            "workload_case_id": manifest.request.case_id,
            "source_request_sha256": canonical_digest(manifest.request),
            "phase": "native_npusim",
            "runtime_status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_seconds": round(time.monotonic() - started, 3),
            "frontend_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "source_tool_at_entry": source_tool_at_entry,
            "npup_sha256": artifact_digests,
        }, indent=2, sort_keys=True), encoding="utf-8")
        raise
    (output / "npusim.stdout.txt").write_text(runtime_output, encoding="utf-8")

    segment_markers = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)",
        runtime_output,
    )
    kv_markers = re.findall(
        r"\[DENSE_SEQUENCE_KV\] index=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) pass=1",
        runtime_output,
    )
    drain_markers = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",
        runtime_output,
    )
    if segment_markers != [("0", "0"), ("1", "0"), ("2", "1")]:
        raise RuntimeError(f"segment marker closure failed: {segment_markers}")
    inference = manifest.request.steps.inference
    assert inference is not None
    model = manifest.request.model
    if model.dtype.value != "fp16":
        raise RuntimeError("KV byte oracle requires actual FP16 model")
    kv_bytes_per_token = (
        2 * model.num_layers * model.num_kv_heads * model.head_dim * 2
    )
    prefill_context = inference.prefill_tokens * inference.request_count
    expected_kv_bytes = [
        (str(index), str((prefill_context + index * inference.request_count)
                         * kv_bytes_per_token))
        for index in range(3)
    ]
    if [tuple(item[:2]) for item in kv_markers] != expected_kv_bytes:
        raise RuntimeError(f"KV boundary closure failed: {kv_markers}")
    if drain_markers != [("3", "1")]:
        raise RuntimeError(f"one-shot drain closure failed: {drain_markers}")
    if runtime_output.count("[SIM_RESULT]") != 1:
        raise RuntimeError("runtime did not emit exactly one SIM_RESULT")
    source_tool_at_exit = _source_tool_snapshot(args)
    drifted_sources = sorted(
        path for path, digest in
        source_tool_at_entry["imported_python_sha256"].items()
        if source_tool_at_exit["imported_python_sha256"].get(path) != digest
    )
    drifted_tools = sorted(
        name for name, digest in source_tool_at_entry["tool_sha256"].items()
        if source_tool_at_exit["tool_sha256"].get(name) != digest
    )
    if drifted_sources or drifted_tools:
        raise RuntimeError(
            f"loaded source/tool drifted while NpuSim ran: "
            f"python={drifted_sources}, tools={drifted_tools}"
        )
    (output / "source_tool_binding.json").write_text(json.dumps({
        "source_tool_at_entry": source_tool_at_entry,
        "additional_imported_python_sha256": {
            path: digest for path, digest in
            source_tool_at_exit["imported_python_sha256"].items()
            if path not in source_tool_at_entry["imported_python_sha256"]
        },
        "hardware_sha256": hashlib.sha256(hardware_path.read_bytes()).hexdigest(),
        "linked_manifest_sha256": [hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in manifests],
        "npup_sha256": artifact_digests,
        "program_io_sha256": [hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sidecars],
        "sequence_digest": sequence.digest,
        "workload_case_id": manifest.request.case_id,
        "kv_boundaries_bytes": [int(item[1]) for item in kv_markers],
        "runtime_status": "verified",
    }, indent=2, sort_keys=True), encoding="utf-8")
    compiled_receipt_path = output / "compiled_receipt.json"
    compiled_receipt = json.loads(compiled_receipt_path.read_text(encoding="utf-8"))
    compiled_receipt["runtime_status"] = "verified"
    compiled_receipt["total_wall_seconds"] = round(time.monotonic() - started, 3)
    compiled_receipt["source_tool_binding_sha256"] = hashlib.sha256(
        (output / "source_tool_binding.json").read_bytes()
    ).hexdigest()
    compiled_receipt_path.write_text(
        json.dumps(compiled_receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Dense sequence runtime canary PASS mesh={args.mesh_size} "
        f"sequence={sequence.digest} artifacts={','.join(artifact_digests)} "
        f"kv={','.join(item[2] for item in kv_markers)}"
    )


def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-debug-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=build / "dense-sequence-runtime-canary"
    )
    parser.add_argument(
        "--mesh-size",
        choices=tuple(f"{rows}x{columns}"
                      for rows in range(1, 11)
                      for columns in range(1, 11)),
        default="1x1",
        help="physical mesh within the 1..10 release envelope",
    )
    parser.add_argument(
        "--scaled-all-dies",
        action="store_true",
        help="two-layer Dense TP=all physical Dies; shape-scaled main case",
    )
    parser.add_argument(
        "--finalizer", type=Path, default=build / "npusim_program_finalizer"
    )
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument("--resolver", type=Path,
                        default=build / "npusim_program_io_selftest")
    parser.add_argument(
        "--simulation",
        type=Path,
        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--compile-timeout", type=int, default=2400)
    args = parser.parse_args()
    fixed_shapes = {"1x1", "2x2", "1x4", "4x1", "2x3", "3x2",
                    "1x6", "6x1", "3x3"}
    if not args.scaled_all_dies and args.mesh_size not in fixed_shapes:
        parser.error("this mesh size requires --scaled-all-dies")
    if args.timeout <= 0 or args.compile_timeout <= 0:
        parser.error("--timeout and --compile-timeout must be positive")
    for name in ("finalizer", "npusim", "resolver", "simulation"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name} must name an existing file")
    return args


if __name__ == "__main__":
    run(_parse_args())
