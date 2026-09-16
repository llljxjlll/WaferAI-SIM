"""Physical EP1 two-step MoE forward sequence; not a full SGD trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from llm.frontend.wafer_frontend.lowering.context import LoweringContext
from llm.frontend.wafer_frontend.lowering.linker import NaiveManifestLinker
from llm.frontend.wafer_frontend.passes.fusion_partition import partition_ir1
from llm.frontend.wafer_frontend.passes.global_action import build_global_action_dag
from llm.frontend.wafer_frontend.passes.lower_program import (
    _lower_fragments, _resolve_dependencies,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_ep_ir1_source import (
    build_moe_ep_placed_ir1_candidate,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_route_program_io import (
    bind_full_moe_route_state_program_io,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_route_table_source import (
    build_moe_full_train_route_table_source,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    MoeFullTrainForwardLinkedSource, _resolved_state_abis,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.policies.naive_intra_die import NaiveIntraDiePolicy
from llm.frontend.wafer_frontend.policies.naive_project_to_ir2 import NaiveProjectToIR2
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.ir1 import IR1
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import canonical_json
from llm.test.frontend.integration.flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from llm.test.frontend.integration.run_moe_full_model_sequence_runtime_canary import (
    _bind_native_hardware_to_fabric,
)
from llm.test.frontend.unit.test_moe_full_train_ep_placement import (
    MoeFullTrainEpPlacementTest as Fixture,
    build_single_die_moe_train_physical_source,
)


def _run(args: list[str], *, cwd: Path, log: Path) -> None:
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=1200)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited {result.returncode}; see {log}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject(args: list[str], *, cwd: Path, log: Path,
            contains: str) -> None:
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=120)
    log.write_text(result.stdout)
    if result.returncode != 2 or contains not in result.stdout:
        raise RuntimeError(f"negative preflight did not reject exactly: {log}")


def _forward(step: int, output: Path, finalizer: Path, resolver: Path) -> dict:
    phase, sequence, placement, physical_context = (
        build_single_die_moe_train_physical_source(Fixture, step=step)
    )
    candidate = build_moe_ep_placed_ir1_candidate(
        phase, original_dense=Fixture.dense, sequence=sequence,
        placement=placement, context=physical_context,
        dense_manifest=Fixture.manifest,
    )
    graph = partition_ir1(IR1.create(
        producer_pass="placement", **candidate.physical_ir1._semantic_key(),
    ))
    projection = NaiveProjectToIR2().run(graph, (), (), state_transfers=())
    schedules = NaiveIntraDiePolicy().schedule(projection, graph)
    dag = build_global_action_dag(graph, projection, schedules)
    context = LoweringContext(graph, (), (), projection, schedules, dag)
    leaves = _lower_fragments(
        context, _resolve_dependencies(None, None, None, None, None),
    )
    if len(leaves) != 51:
        raise RuntimeError(f"step{step} changed full forward leaf count")
    manifest = NaiveManifestLinker().link(context, leaves)
    manifest.validate_against(
        context.ir1, context.fusion_plans, context.standalone_plans,
        context.projection, context.schedule_set, context.global_dag,
        manifest.fragments,
    )
    source = MoeFullTrainForwardLinkedSource(manifest, context)
    source.validate()
    path = output / f"step{step}.linked.json"
    artifact = output / f"step{step}.npup"
    report = output / f"step{step}.finalizer.json"
    sidecar = output / f"step{step}.program_io.json"
    path.write_text(canonical_json(manifest))
    _run([str(finalizer), "--input", str(path), "--output",
          str(artifact), "--report", str(report)],
         cwd=output, log=output / f"step{step}.finalizer.log")
    route = build_moe_full_train_route_table_source(phase, sequence)
    route_by_state = {phase.route_state_refs[seed.layer]: seed.payload
                      for seed in route.seeds}
    seeds = {}
    for item in _resolved_state_abis(source):
        if item.first_access is not StateUseAccess.READ:
            continue
        abi = item.abi
        if abi.kind is StateKind.MOE_STATIC_ROUTE:
            seeds[abi.state_ref] = route_by_state[abi.state_ref]
        elif abi.dtype is DType.FP16 and abi.size_bytes % 2 == 0:
            seeds[abi.state_ref] = struct.pack("<e", 0.0625) * (abi.size_bytes // 2)
        else:
            raise RuntimeError(f"step{step} has unseeded state {abi.state_ref}")
    base = build_timing_program_io(
        source, _sha(artifact), state_seed_overrides=seeds,
    )
    contract = bind_full_moe_route_state_program_io(
        manifest, base, route, phase, sequence, placement,
        original_dense=Fixture.dense, dense_manifest=Fixture.manifest,
        context=physical_context,
    )
    if len(contract.initializations) != 76 or len(contract.output_probes) != 1:
        raise RuntimeError(f"step{step} forward ProgramIO changed extent")
    sidecar.write_text(canonical_json(contract))
    _run([str(resolver), "--resolve", str(path), str(artifact), str(sidecar)],
         cwd=output, log=output / f"step{step}.resolver.log")
    if "ProgramIo resolved" not in (output / f"step{step}.resolver.log").read_text():
        raise RuntimeError(f"step{step} resolver lacked success evidence")
    return dict(step=step, ir0_id=phase.graph.id, route_source_id=route.id,
                input_parameter_version=step,
                e2e_parameter_state_refs=tuple(sorted(
                    owner.source_e2e_state_ref
                    for owner in phase.ep_state_owners)),
                linked_id=manifest.id, linked_sha256=_sha(path),
                artifact_sha256=_sha(artifact), program_io_sha256=_sha(sidecar),
                leaves=len(leaves), records=sum(len(stream.records)
                     for stream in manifest.core_streams),
                initializations=len(contract.initializations),
                probes=len(contract.output_probes),
                route_state_refs=phase.route_state_refs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--finalizer", required=True, type=Path)
    parser.add_argument("--resolver", required=True, type=Path)
    parser.add_argument("--npusim", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    finalizer, resolver, npusim = (args.finalizer.resolve(),
                                   args.resolver.resolve(), args.npusim.resolve())
    Fixture.setUpClass()
    rows = [_forward(step, output, finalizer, resolver) for step in (0, 1)]
    if (rows[0]["ir0_id"] == rows[1]["ir0_id"]
            or rows[0]["artifact_sha256"] == rows[1]["artifact_sha256"]
            or rows[0]["route_source_id"] == rows[1]["route_source_id"]):
        raise RuntimeError("step1 repeated step0 source or native program")
    _, _, _, physical_context = build_single_die_moe_train_physical_source(
        Fixture, step=0,
    )
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    _bind_native_hardware_to_fabric(hardware, physical_context.fabric)
    hardware["memory"]["sram_size"] = 131072
    hardware["memory"]["sram"]["capacity_bytes"] = 131072
    hardware["memory"]["sram"]["regions"] = [dict(
        name="sram", base_bytes=0, size_bytes=131072, allocator="block",
        spillable=False, access=["compute", "dte", "lsu", "legacy", "noc_rx"],
    )]
    space = physical_context.hbm_address_spaces[0]
    for stack in hardware["memory_system"]["hbm_stacks"]:
        stack["capacity_bytes"] = space.size_bytes
    hardware["memory_system"]["address_policy"]["home_ranges"] = [dict(
        die_id=0, base=space.base_address, size_bytes=space.size_bytes,
    )]
    hardware["memory_system"]["address_policy"]["stack_interleave_bytes"] = space.size_bytes
    hardware_path = output / "hardware.json"
    hardware_path.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")))
    mapping = output / "mapping.spec"
    mapping.write_text("0:0\n")
    simulation = Path(__file__).resolve().parents[3] / "test/program/p5_behavioral_simulation.json"
    if not simulation.is_file():
        raise RuntimeError(f"missing simulation config: {simulation}")
    native_logs = []
    for step in (0, 1):
        native_log = output / f"step{step}.npusim.log"
        _run([str(npusim), "--program-one-shot",
              "--program", str(output / f"step{step}.npup"),
              "--linked-manifest", str(output / f"step{step}.linked.json"),
              "--program-io", str(output / f"step{step}.program_io.json"),
              "--hardware-config", str(hardware_path),
              "--simulation-config", str(simulation),
              "--mapping-config", str(mapping), "--trace-window", "1000000"],
             cwd=npusim.parent, log=native_log)
        log = native_log.read_text()
        if (log.count("[PROGRAM_IO] phase=verify") != 1
                or log.count("[MOE_SIGNED_ROUTER] stage=forward") != 2
                or log.count("[PROGRAM_MEMORY] core=0") != 1
                or "[CREDIT] data_balanced=1 ctrl_balanced=1" not in log
                or "[DRAIN] d2d_link_residual=0" not in log):
            raise RuntimeError(f"step{step} forward native audit failed; see {native_log}")
        native_logs.append(native_log)
    sequence_log = output / "sequence.npusim.log"
    sequence_args = [str(npusim), "--moe-forward-sequence",
          "--program-sequence",
          ",".join(str(output / f"step{step}.npup") for step in (0, 1)),
          "--linked-manifest-sequence",
          ",".join(str(output / f"step{step}.linked.json") for step in (0, 1)),
          "--program-io-sequence",
          ",".join(str(output / f"step{step}.program_io.json") for step in (0, 1)),
          "--hardware-config", str(hardware_path),
          "--simulation-config", str(simulation),
          "--mapping-config", str(mapping), "--trace-window", "1000000"]
    _run(sequence_args, cwd=npusim.parent, log=sequence_log)
    sequence_text = sequence_log.read_text()
    if (sequence_text.count("[MOE_FORWARD_SEQUENCE_STEP]") != 2
            or sequence_text.count("[DENSE_SEQUENCE_PROGRAM_IO]") != 2
            or "[DENSE_TRAINING_SEQUENCE_STEP]" in sequence_text
            or "[DENSE_ADAMW_SEQUENCE_STEP]" in sequence_text
            or "[CREDIT] data_balanced=1 ctrl_balanced=1" not in sequence_text
            or "[DRAIN] d2d_link_residual=0" not in sequence_text):
        raise RuntimeError("MoE forward-only sequence native audit failed")
    no_flag = [arg for arg in sequence_args if arg != "--moe-forward-sequence"]
    _reject(no_flag, cwd=npusim.parent,
            log=output / "reject_dense_training_gate.log",
            contains="Dense training sequence lacks exact WGRAD/optimizer")
    forged_io = output / "step1.program_io.unsigned_route.json"
    forged = json.loads((output / "step1.program_io.json").read_text())
    forged["producer_pass"] = "program_io"
    forged_io.write_text(json.dumps(forged, sort_keys=True, separators=(",", ":")))
    _reject([arg.replace(str(output / "step1.program_io.json"), str(forged_io))
             for arg in sequence_args], cwd=npusim.parent,
            log=output / "reject_unsigned_route.log",
            contains="requires source-bound route ProgramIO")
    replay = [arg.replace(str(output / "step1.npup"),
                          str(output / "step0.npup"))
                  .replace(str(output / "step1.linked.json"),
                           str(output / "step0.linked.json"))
                  .replace(str(output / "step1.program_io.json"),
                           str(output / "step0.program_io.json"))
              for arg in sequence_args]
    _reject(replay, cwd=npusim.parent,
            log=output / "reject_step0_replay.log",
            contains="replayed the first source/route/program")
    repo = Path(__file__).resolve().parents[4]
    source_paths = (
        "llm/frontend/wafer_frontend/passes/moe_full_train_forward_ir0.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_hbm_layout.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_ep_placement.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_ep_ir1_source.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_forward_validator.py",
        "llm/frontend/wafer_frontend/passes/moe_full_train_named_wgrad_tiles.py",
        "llm/frontend/wafer_frontend/passes/program_io.py",
        "llm/test/frontend/unit/test_moe_full_train_ep_placement.py",
        "llm/test/frontend/integration/run_moe_full_train_two_step_forward_canary.py",
        "llm/unittest/npusim.cpp",
    )
    receipt = dict(status="two_step_forward_sequence_native_pass",
                   full_training_gate="closed", steps=rows,
                   source_file_sha256={path: _sha(repo / path)
                                       for path in source_paths},
                   finalizer_sha256=_sha(finalizer),
                   resolver_sha256=_sha(resolver),
                   hardware_sha256=_sha(hardware_path),
                   simulation_sha256=_sha(simulation),
                   npusim_sha256=_sha(npusim),
                   npusim_log_sha256=tuple(_sha(log) for log in native_logs),
                   sequence_log_sha256=_sha(sequence_log))
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
