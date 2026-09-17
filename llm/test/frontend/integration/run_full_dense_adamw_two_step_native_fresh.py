"""One independent full-source Dense AdamW two-step native Fresh.

Invoke twice in separate empty roots to establish full materialization repeatability.
The native optimizer is timing-only; ProgramIO checks physical StateABI payloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.lowering.full_dense_adamw_physical_gate import (
    build_full_dense_adamw_physical_dag,
)
from llm.frontend.wafer_frontend.passes.flexible_dense_train import build_flexible_dense_train_plan
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_adamw_ir0 import (
    build_full_dense_training_two_step_adamw_ir0,
)
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_train_forward
from llm.frontend.wafer_frontend.passes.inter_die_plan import plan_train_forward
from llm.frontend.wafer_frontend.passes.intra_die_schedule import schedule_train_forward
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data, physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.placement import place_train_forward_ir0
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides, build_timing_program_io,
)
from llm.frontend.wafer_frontend.passes.project_to_ir2 import project_train_forward
from llm.frontend.wafer_frontend.passes.train_global_action import build_train_global_action
from llm.frontend.wafer_frontend.passes.train_link_program import link_train
from llm.frontend.wafer_frontend.passes.train_lower_program import lower_train
from llm.frontend.wafer_frontend.policies.registry import RegistryKind, production_registry
from llm.frontend.wafer_frontend.schema._validation_session import builder_validation_session
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.n4 import FusionPartitionContext, InterDiePlanningContext
from llm.frontend.wafer_frontend.schema.n5 import ProjectToIR2Context, IntraDieSchedulingContext
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragments
from llm.frontend.wafer_frontend.schema.placement import PlacementContext
from llm.frontend.wafer_frontend.schema.program_io import ProgramHbmTarget
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec

from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], cwd: Path, timeout: int, log: Path) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout, check=False)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{command[0]} exit={result.returncode}; log={log}; tail={result.stdout[-2500:]}")
    return result.stdout


@builder_validation_session()
def _compile():
    producer = "full_dense_training_two_step_adamw_native"
    plan = build_flexible_dense_train_plan(_spec(1, 1), RectMeshSpec(1, 1))
    graph = build_full_dense_training_two_step_adamw_ir0(plan)
    hardware = _hardware(1, 1)
    placed = place_train_forward_ir0(graph, PlacementContext.create(
        producer_pass=producer, fabric=physical_fabric_from_data(hardware),
        placement=plan.source_experiment.placement,
        hbm_address_spaces=hbm_address_spaces_from_data(hardware),
    ))
    partitioned = partition_train_forward(placed, FusionPartitionContext.create(
        producer_pass=producer))
    registry = production_registry()
    planned = plan_train_forward(partitioned, InterDiePlanningContext.create(
        producer_pass=producer,
        fused_policy=registry.instantiate(RegistryKind.INTER_DIE, "naive").selection,
        standalone_policy=registry.instantiate(
            RegistryKind.STANDALONE_COLLECTIVE, "direct_all_gather").selection,
    ))
    projected = project_train_forward(planned, ProjectToIR2Context.create(
        producer_pass=producer, state_transfers=()))
    scheduled = schedule_train_forward(projected, IntraDieSchedulingContext.create(
        producer_pass=producer,
        policy=registry.instantiate(RegistryKind.INTRA_DIE, "naive").selection))
    global_actions = build_train_global_action(scheduled)
    native = lower_train(global_actions)
    linked = link_train(native)
    physical = build_full_dense_adamw_physical_dag(linked, plan)
    return linked, physical


@builder_validation_session()
def run(*, output: Path, npusim: Path, finalizer: Path, resolver: Path,
        simulation: Path, timeout: int = 600) -> dict:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("AdamW Fresh requires a new empty output root")
    output.mkdir(parents=True, exist_ok=True)
    tools = {name: path.resolve() for name, path in (
        ("npusim", npusim), ("finalizer", finalizer),
        ("resolver", resolver), ("simulation", simulation))}
    if any(not path.is_file() for path in tools.values()):
        raise RuntimeError("one frozen native tool or simulation config is missing")
    linked, physical = _compile()
    manifest = linked.manifest
    if (len(manifest.fragments) != 520 or len(physical.actions) != 520
            or len(physical.state_version_edges) != 75):
        raise RuntimeError("AdamW full-source physical inventory changed")
    manifest_path = output / "full_adamw_two_step.linked.json"
    physical_path = output / "full_adamw_two_step.physical_dag.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    physical_path.write_text(canonical_json(physical), encoding="utf-8")
    npup = output / "full_adamw_two_step.npup"
    report_path = output / "full_adamw_two_step.finalizer.json"
    _run([str(tools["finalizer"]), "--input", str(manifest_path),
          "--output", str(npup), "--report", str(report_path)],
         output, 180, output / "finalizer.stdout.txt")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("artifact_sha256") != _sha(npup)
            or report.get("linked_manifest_id") != manifest.id
            or report.get("linked_manifest_digest") != canonical_digest(manifest)):
        raise RuntimeError("AdamW finalizer manifest and NPUP do not close")
    leaves = _leaf_fragments(manifest.fragments)
    graph = linked.source.replicas[0].lowering_context.ir1
    ce_values = {node.inputs[2] for node in graph.nodes if node.kind is OpKind.CE_BACKWARD}
    ce_seeds = {abi.id: b"\x00\x00\x80\x3f" * (abi.size_bytes // 4)
                for fragment in leaves for abi in fragment.buffer_abi
                if abi.value_id in ce_values}
    states, expected = build_deterministic_timing_state_overrides(linked)
    if len(ce_seeds) != 2 or len(states) != 75 or len(expected) != 60:
        raise RuntimeError("AdamW needs two CE seeds and all 75 StateABI seeds")
    program_io = build_timing_program_io(
        linked, _sha(npup), sram_seed_overrides=ce_seeds,
        state_seed_overrides=states, state_expected_overrides=expected)
    physical_state_abis = {abi.id for fragment in leaves for abi in fragment.state_abi
                           if abi.state_ref in states}
    hbm_seeds = tuple(item.target for item in program_io.initializations
                      if isinstance(item.target, ProgramHbmTarget))
    if (len(physical_state_abis) != 75 or len(hbm_seeds) != 75
            or {item.state_abi_id for item in hbm_seeds} != physical_state_abis):
        raise RuntimeError("all 75 physical HBM StateABI homes need one initialization")
    io_path = output / "full_adamw_two_step.program_io.json"
    io_path.write_text(canonical_json(program_io), encoding="utf-8")
    _run([str(tools["resolver"]), "--resolve", str(manifest_path),
          str(npup), str(io_path)], output, 180,
         output / "program_io_resolver.stdout.txt")
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 1 << 20
    dram_source = Path(__file__).resolve().parents[4] / "DRAMSys"
    dram_config = dram_source / "configs/hbm2-example.json"
    if not dram_config.is_file():
        raise RuntimeError("frozen DRAMSys configuration is missing")
    for stack in hardware["memory_system"]["hbm_stacks"]:
        stack["channel_dram_config"] = str(dram_config)
    hw_path = output / "hardware.json"
    hw_path.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")),
                       encoding="utf-8")
    (output / "DRAMSys").symlink_to(dram_source, target_is_directory=True)
    mapping = output / "mapping.spec"
    mapping.write_text("0:0\n", encoding="utf-8")
    command = [str(tools["npusim"]), "--program", str(npup),
               "--linked-manifest", str(manifest_path), "--program-io", str(io_path),
               "--hardware-config", str(hw_path), "--simulation-config",
               str(tools["simulation"]), "--mapping-config", str(mapping),
               "--trace-window", "1000000"]
    (output / "npusim_command.json").write_text(json.dumps(command, indent=2) + "\n",
                                                  encoding="utf-8")
    native_cwd = output / "native_run"
    native_cwd.mkdir()
    stdout = _run(command, native_cwd, timeout, output / "npusim.stdout.txt")
    markers = tuple(re.findall(
        r"\[PROGRAM_IO\] phase=(resolved|applied|verify) mode=timing "
        r"initializations=(\d+) probes=(\d+) checksum=[0-9a-f]{64} pass=(\d+)", stdout))
    cycles = re.search(r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout)
    if (tuple(row[0] for row in markers) != ("resolved", "applied", "verify")
            or any(row[1:] != (str(len(program_io.initializations)),
                                  str(len(program_io.output_probes)), "1") for row in markers)
            or stdout.count("[TRAIN_CE] core=") != 2
            or stdout.count("[TRAIN_CE_BACKWARD] core=") != 2
            or stdout.count("[TRAIN_ADAMW] core=") != 30
            or cycles is None or stdout.count("[SIM_RESULT] makespan_cycles=") != 1
            or "[DRAIN] router_residual=0" not in stdout
            or "[DRAIN] d2d_link_residual=0" not in stdout):
        raise RuntimeError("AdamW native timing/ProgramIO/gradient/optimizer/drain closure failed")
    receipt = {
        "schema_version": "dense_adamw_two_step_native_fresh/v1", "mesh": [1, 1],
        "steps": 2, "layers": 2, "states": 75, "parameters": 15,
        "wgrad_paths": 30, "adamw_updates": 30,
        "state_version_edges": len(physical.state_version_edges),
        "physical_actions": len(physical.actions),
        "native_fragments": len(manifest.fragments),
        "native_adamw_records": sum(record.opcode is RecordOpcode.ADAMW_UPDATE
                                     for fragment in leaves for stream in fragment.core_streams
                                     for record in stream.records),
        "program_io_initializations": len(program_io.initializations),
        "program_io_probes": len(program_io.output_probes),
        "physical_hbm_initializations": len(hbm_seeds),
        "makespan_cycles": int(cycles.group(1)),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=Path(__file__).resolve().parents[4],
                                                 text=True).strip(),
        "files_sha256": {name: _sha(path) for name, path in {
            "linked": manifest_path, "physical_dag": physical_path,
            "npup": npup, "program_io": io_path, "hardware": hw_path,
            "native_stdout": output / "npusim.stdout.txt",
            **tools,
        }.items()},
        "numeric_mode": "timing", "offload": False,
    }
    (output / "evidence.json").write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "npusim", "finalizer", "resolver", "simulation"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    run(output=args.output, npusim=args.npusim, finalizer=args.finalizer,
        resolver=args.resolver, simulation=args.simulation, timeout=args.timeout)


if __name__ == "__main__":
    main()
