"""Two fresh 1x1 H16/L2 AdamW runs with real mid-program 83-state DMA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from llm.frontend.wafer_frontend.passes.dense_adamw_compile_sequence import (
    compile_dense_adamw_step,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_mid_program_residency import (
    assign_dense_adamw_bounded_slots, derive_dense_adamw_mid_program_residency,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_paged_compile_sequence import (
    compile_dense_adamw_paged_step,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_paged_runtime import (
    build_dense_adamw_paged_runtime,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

from .flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from .run_dense_adamw_dma_component_canary import build_source_dma_program
from .run_dense_adamw_training_runtime_canary import (
    _linked_oracle, _source_oracle,
)
from .run_dense_training_sequence_runtime_canary import _run


_ROOT = Path(__file__).resolve().parents[4]


def _observe(stdout: str, sidecar) -> dict[str, object]:
    events = re.findall(
        r"\[DENSE_ADAMW_PAGED_DMA_EVENT\] index=(\d+) step=(\d+) "
        r"linked_record=(\d+) direction=(\w+) state_ref=(\S+) "
        r"bytes=(\d+) issue_cycle=(\d+) completed_at_ticks=(\d+) "
        r"lsu_dependency_complete=(\d+) pass=(\d+)", stdout,
    )
    if len(events) != 332 or any(
        int(actual[0]) != index
        or int(actual[1]) != signed.step_index
        or int(actual[2]) != signed.linked_record_index
        or actual[3] != signed.kind
        or actual[4] != signed.state_ref
        or int(actual[5]) != signed.size_bytes
        or actual[8:] != ("1", "1")
        for index, (actual, signed) in enumerate(zip(events, sidecar.events))
    ):
        raise RuntimeError("332 actual DMA completions drifted from signed LSU/StateABI event order")
    if any(
        int(actual[7]) <= 0
        or (index and int(actual[7]) <= int(events[index - 1][7]))
        for index, actual in enumerate(events)
    ):
        raise RuntimeError("shared external link did not advance in real service time")
    for step in (0, 1):
        for direction in ("restore_before_lsu_load", "writeback_after_lsu_store"):
            actual = tuple(item for item in events
                           if item[1] == str(step) and item[3] == direction)
            if len(actual) != 83 or sum(int(item[5]) for item in actual) != 32100:
                raise RuntimeError("step/role real shared DMA did not cover 83 exact bytes")
    states = re.findall(
        r"\[DENSE_TRAINING_SEQUENCE_STATE\] version=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) content_changed=(\d+) "
        r"functional=(\d+) pass=(\d+) authority=external", stdout,
    )
    steps = re.findall(
        r"\[DENSE_ADAMW_SEQUENCE_STEP\] index=(\d+) "
        r"input_version=(\d+) output_version=(\d+) "
        r"trainable_states=(\d+) optimizer_states=(\d+) "
        r"adamw_records=(\d+) load_records=(\d+) store_records=(\d+) "
        r"functional=(\d+) pass=(\d+)", stdout,
    )
    probes = re.findall(
        r"\[DENSE_ADAMW_EXTERNAL_PROGRAM_IO\] index=(\d+) "
        r"probes=(\d+) physical_state_bytes=(\d+) pending=(\d+) "
        r"pass=(\d+) functional=(\d+)", stdout,
    )
    segments = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)",
        stdout,
    )
    drain = re.findall(
        r"\[DENSE_ADAMW_PAGED_DMA_DRAIN\] events=(\d+) "
        r"probes=(\d+) submitted=(\d+) completed=(\d+) "
        r"external_read_bytes=(\d+) external_write_bytes=(\d+) "
        r"hbm_read_bytes=(\d+) hbm_write_bytes=(\d+) "
        r"pending=(\d+) dirty=(\d+) pinned=(\d+) pass=(\d+)",
        stdout,
    )
    makespan = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout)
    final = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",
        stdout,
    )
    if (
        len(states) != 3
        or tuple(item[0] for item in states) != ("0", "1", "2")
        or any(item[1] != "32100" or item[3:] != ("0", "0", "1")
               for item in states)
        or len(steps) != 2
        or tuple(item[:3] for item in steps) != (
            ("0", "0", "1"), ("1", "1", "2"),
        )
        or any(item[3:] != ("15", "68", "17", "83", "83", "0", "1")
               for item in steps)
        or probes != [
            ("0", "83", "32100", "0", "1", "0"),
            ("1", "83", "32100", "0", "1", "0"),
        ]
        or segments != [("0", "0"), ("1", "1")]
        or final != [("2", "1")]
        or drain != [(
            "332", "166", "332", "332", "64200", "64200",
            "64200", "64200", "0", "0", "0", "1",
        )]
        or len(makespan) != 1
        or stdout.count("[TRAIN_ADAMW]") != 34
        or stdout.count("[DENSE_ADAMW_PAGED_BINDING]") != 1
    ):
        raise RuntimeError(
            "real bounded source/332 DMA events/83 external probes/two-step drain failed: "
            f"states={states} steps={steps} probes={probes} "
            f"segments={segments} drain={drain} ADAMW={stdout.count('[TRAIN_ADAMW]')} "
            f"makespan={makespan} final={final}"
        )
    return {
        "versions": [0, 1, 2],
        "state_bytes": 32100,
        "authority_state_digests": [item[2] for item in states],
        "adamw_updates": 34,
        "external_dma_events": 332,
        "external_state_probes": 166,
        "first_dma_issue_cycle": int(events[0][6]),
        "last_dma_completed_at_ticks": int(events[-1][7]),
        "makespan_cycles": int(makespan[0]),
        "functional": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source, physical, window, program, _action_graph = build_source_dma_program()
    _source_oracle(source, physical)
    if (
        source.request.model != window.materialization.request.model
        or source.request.steps != window.materialization.request.steps
        or source.request.mesh != window.materialization.request.mesh
        or source.request.parallel != window.materialization.request.parallel
        or source.request.optimizer != window.materialization.request.optimizer
    ):
        raise RuntimeError("paired low-HBM resident/offload exact source model drifted")
    model_digest = canonical_digest(source.request.model)
    paired_operations = tuple(
        (item.kind.value, item.step, item.layer,
         item.expert, item.parameter_ref)
        for item in source.logical_graph.operations
    )
    offload_operations = tuple(
        (item.kind.value, item.step, item.layer,
         item.expert, item.parameter_ref)
        for item in window.materialization.logical_graph.operations
    )
    if paired_operations != offload_operations:
        raise RuntimeError("same-model source logical operation digest changed offload source")
    operation_digest = canonical_digest(paired_operations)
    if (
        window.resident_rejection_code != "memory_capacity_exceeded"
        or window.startup_window_sufficient
        or window.external_state_bytes != 32100
        or window.resident_hbm_capacity_bytes != 36864
        or window.hbm_workspace_peak_bytes != 9248
    ):
        raise RuntimeError("same-model resident reject and P3 low-capacity source drifted")
    original = tuple(
        compile_dense_adamw_step(window.materialization, physical, step)
        for step in (0, 1)
    )
    schedule = derive_dense_adamw_mid_program_residency(window, original)
    slots = assign_dense_adamw_bounded_slots(window, schedule)
    if slots.highest_state_end_bytes != 23300:
        raise RuntimeError("strict workspace+timed HBM slots changed")
    linked = tuple(
        compile_dense_adamw_paged_step(window, physical, step, slots)
        for step in (0, 1)
    )
    for step, item in enumerate(linked):
        _linked_oracle(item, window.materialization, physical, step)
    sidecar = build_dense_adamw_paged_runtime(
        window, schedule, slots, linked, program,
    )
    output = args.output.resolve()
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "external_dma_program.json").write_text(
        canonical_json(program), encoding="utf-8",
    )
    sidecar_path = output / "dense_adamw_paged_runtime.json"
    sidecar_path.write_text(canonical_json(sidecar), encoding="utf-8")
    manifests = []
    programs = []
    for step, item in enumerate(linked):
        manifest_path = output / f"step_{step}.linked.json"
        npup_path = output / f"step_{step}.npup"
        finalizer_report = output / f"step_{step}.finalizer.json"
        manifest_path.write_text(canonical_json(item.manifest), encoding="utf-8")
        _run((
            str(args.finalizer.resolve()), "--input", str(manifest_path),
            "--output", str(npup_path), "--report", str(finalizer_report),
        ), cwd=output, timeout=120)
        report = json.loads(finalizer_report.read_text(encoding="utf-8"))
        if (
            report["artifact_sha256"] !=
                hashlib.sha256(npup_path.read_bytes()).hexdigest()
            or report["linked_manifest_id"] != item.manifest.id
            or report["linked_manifest_digest"] != canonical_digest(item.manifest)
        ):
            raise RuntimeError("actual paged production finalizer digest drifted")
        manifests.append(manifest_path)
        programs.append(npup_path)
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 1 << 20
    physical_hbm = hardware["memory_system"]
    if len(physical_hbm["hbm_stacks"]) != 1 or len(
        physical_hbm["address_policy"]["home_ranges"]
    ) != 1 or physical_hbm["hbm_stacks"][0]["backend"] != "behavioral":
        raise RuntimeError("real die0 behavioral HBM fixture identity drifted")
    physical_hbm["hbm_stacks"][0]["capacity_bytes"] = \
        window.resident_hbm_capacity_bytes
    physical_hbm["address_policy"]["home_ranges"][0]["size_bytes"] = \
        window.resident_hbm_capacity_bytes
    physical_hbm["address_policy"]["stack_interleave_bytes"] = \
        window.resident_hbm_capacity_bytes
    hardware_path = output / "hardware.json"
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    mapping = output / "mapping.spec"
    mapping.write_text("0:0\n", encoding="utf-8")
    tool_sha = hashlib.sha256(args.npusim.resolve().read_bytes()).hexdigest()
    finalizer_sha = hashlib.sha256(args.finalizer.resolve().read_bytes()).hexdigest()
    hardware_sha = hashlib.sha256(hardware_path.read_bytes()).hexdigest()
    simulation_sha = hashlib.sha256(args.simulation.resolve().read_bytes()).hexdigest()
    observations = []
    for fresh in (0, 1):
        cwd = output / f"fresh_{fresh}"
        cwd.mkdir(parents=True, exist_ok=True)
        stdout = _run((
            str(args.npusim.resolve()), "--program-sequence",
            ",".join(map(str, programs)),
            "--linked-manifest-sequence", ",".join(map(str, manifests)),
            "--dense-adamw-paged-runtime", str(sidecar_path),
            "--hardware-config", str(hardware_path),
            "--simulation-config", str(args.simulation.resolve()),
            "--mapping-config", str(mapping), "--trace-window", "1000000",
        ), cwd=args.npusim.resolve().parent, timeout=args.timeout)
        (cwd / "npusim.stdout.txt").write_text(stdout, encoding="utf-8")
        if hashlib.sha256(args.npusim.resolve().read_bytes()).hexdigest() != tool_sha:
            raise RuntimeError("two fresh NpuSim processes were bound to different binaries")
        observation = _observe(stdout, sidecar)
        observations.append(observation)
    if observations[0] != observations[1]:
        raise RuntimeError(f"two fresh paged NpuSim processes drifted: {observations}")
    report = {
        **observations[0], "paired_fresh_runs": 2,
        "resident_rejection_code": window.resident_rejection_code,
        "bounded_hbm_capacity_bytes": window.resident_hbm_capacity_bytes,
        "hbm_workspace_peak_bytes": window.hbm_workspace_peak_bytes,
        "highest_paged_state_end_bytes": slots.highest_state_end_bytes,
        "external_state_bytes": window.external_state_bytes,
        "source_dma_program_id": program.id,
        "paged_runtime_contract_id": sidecar.id,
        "model_digest": model_digest,
        "paired_logical_operation_digest": operation_digest,
        "npusim_sha256": tool_sha,
        "finalizer_sha256": finalizer_sha,
        "hardware_sha256": hardware_sha,
        "simulation_sha256": simulation_sha,
        "paged_manifest_digests": list(sidecar.linked_manifest_digests),
    }
    (output / "adamw-paged-runtime-evidence.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    print(
        "Dense AdamW bounded external offload two-layer two-step PASS "
        "resident=memory_capacity_exceeded HBM=36864 workspace=9248 "
        "state=32100B pager_peak_end=23300B DMA=332 probes=166 "
        f"fresh_runs=2 makespan={report['makespan_cycles']} functional=0"
    )
    return report


def _args() -> argparse.Namespace:
    build = _ROOT / "build-debug-adamw-pager"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=build / "dense-adamw-paged-offload-canary")
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument("--finalizer", type=Path,
                        default=_ROOT / "build-debug-final/npusim_program_finalizer")
    parser.add_argument("--simulation", type=Path,
                        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.timeout <= 0 or not all(
        getattr(args, field).is_file()
        for field in ("npusim", "finalizer", "simulation")
    ):
        parser.error("simulation/finalizer/npusim must exist and timeout >0")
    return args


if __name__ == "__main__":
    run(_args())
