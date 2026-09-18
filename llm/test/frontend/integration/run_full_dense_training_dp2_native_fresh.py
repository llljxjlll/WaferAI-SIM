"""Independently compile and execute one real TP×DP2 two-step Dense TRAIN invocation.

Run twice with distinct empty output roots and the same frozen tools for full
materialize→finalize→resolve→simulate repeatability verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides, build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragments
from llm.frontend.wafer_frontend.schema.program_io import ProgramHbmTarget
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware
from .run_full_dense_training_dp2_native_pipeline import compile_dp2


def _run(command: list[str], cwd: Path, timeout: int, log: Path) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"native tool failed exit={result.returncode}, command={command!r}: "
            f"{result.stdout[-4500:]}"
        )
    return result.stdout


def _execute(linked, plan, output: Path, receipt: dict, *,
             npusim: Path, finalizer: Path, resolver: Path,
             simulation: Path, timeout: int) -> None:
    manifest = linked.manifest
    linked.validate()
    columns = plan.spec.tp_degree
    logical_states = 15 * columns
    physical_states = 2 * logical_states
    leaves = _leaf_fragments(manifest.fragments)
    ce_values = {
        node.inputs[2]
        for replica in linked.source.replicas
        for node in replica.lowering_context.ir1.nodes
        if node.kind is OpKind.CE_BACKWARD
    }
    if len(ce_values) != 2:
        raise RuntimeError("DP2 source requires two distinct step-local CE backward values across replicated ranks")
    ce_seeds = {
        abi.id: b"\x00\x00\x80\x3f" * (abi.size_bytes // 4)
        for fragment in leaves for abi in fragment.buffer_abi
        if abi.value_id in ce_values
    }
    if len(ce_seeds) != 4 * columns or any(len(seed) < 4 or len(seed) % 4 for seed in ce_seeds.values()):
        raise RuntimeError("two DP replicas need one independent FP32 CE dLoss seed per TP shard and step")
    states, expected = build_deterministic_timing_state_overrides(linked)
    if len(states) != logical_states or expected:
        raise RuntimeError("DP2 ProgramIO requires one logical seed pattern per TP parameter shard")
    active_dies = {binding.logical_core.die_id for binding in manifest.core_bindings}
    if active_dies != set(range(2 * columns)) or len(manifest.core_bindings) != 5 * columns:
        raise RuntimeError("DP2 native ProgramArtifact must bind each actual core across all physical dies")
    repository = Path(__file__).resolve().parents[4]
    dram_source = repository / "DRAMSys"
    dram_link = output / "DRAMSys"
    if dram_link.exists() or dram_link.is_symlink():
        if not dram_link.is_dir() or dram_link.resolve() != dram_source.resolve():
            raise RuntimeError("native DRAMSys source differs from the project configuration")
    else:
        dram_link.symlink_to(dram_source, target_is_directory=True)
    manifest_path = output / "full_dp2_two_step.linked.json"
    artifact_path = output / "full_dp2_two_step.npup"
    finalizer_path = output / "finalizer.json"
    finalize = [str(finalizer.resolve()), "--input", str(manifest_path),
                "--output", str(artifact_path), "--report", str(finalizer_path)]
    (output / "finalizer_command.json").write_text(
        json.dumps(finalize, indent=2) + "\n", encoding="utf-8",
    )
    # The finalizer validates every physical fragment. Large TP manifests
    # can exceed the small-shape fixed deadline while making normal progress.
    _run(finalize, output, max(240, 90 * columns),
         output / "finalizer.stdout.txt")
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    report = json.loads(finalizer_path.read_text(encoding="utf-8"))
    if (report.get("artifact_sha256") != artifact_sha
            or report.get("linked_manifest_id") != manifest.id
            or report.get("linked_manifest_digest") != canonical_digest(manifest)):
        raise RuntimeError("finalizer output is not bound to the exact DP2 linked manifest")
    io = build_timing_program_io(
        linked, artifact_sha,
        sram_seed_overrides=ce_seeds, state_seed_overrides=states,
    )
    physical_state_abis = {
        abi.id for fragment in leaves for abi in fragment.state_abi
        if abi.state_ref in states
    }
    hbm_initializations = tuple(
        item.target for item in io.initializations
        if isinstance(item.target, ProgramHbmTarget)
    )
    if (len(physical_state_abis) != physical_states
            or len(hbm_initializations) != physical_states
            or {item.state_abi_id for item in hbm_initializations}
               != physical_state_abis):
        raise RuntimeError("each real DP2 StateABI home needs its own physical HBM initialization")
    io_path = output / "full_dp2_two_step.program_io.json"
    io_path.write_text(canonical_json(io), encoding="utf-8")
    receipt.update({
        "artifact_sha256": artifact_sha,
        "program_io_digest": canonical_digest(io),
        "independent_ce_seed_abi": len(ce_seeds),
        "logical_state_seed_patterns": len(states),
        "physical_hbm_state_initializations": len(hbm_initializations),
        "program_io_initializations": len(io.initializations),
        "program_io_probes": len(io.output_probes),
    })
    (output / "compile_evidence.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    resolve = [str(resolver.resolve()), "--resolve", str(manifest_path),
               str(artifact_path), str(io_path)]
    (output / "program_io_resolve_command.json").write_text(
        json.dumps(resolve, indent=2) + "\n", encoding="utf-8",
    )
    _run(resolve, output, max(180, 60 * columns),
         output / "program_io_resolver.stdout.txt")
    hardware = json.loads(specialize_p5_large_release_hardware(2, columns))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 1 << 20
    dram_config = dram_source / "configs/hbm2-example.json"
    if not dram_config.is_file():
        raise RuntimeError("source-bound DRAMSys configuration is missing")
    for stack in hardware["memory_system"]["hbm_stacks"]:
        stack["channel_dram_config"] = str(dram_config)
    hardware_path = output / "hardware.json"
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")), encoding="utf-8",
    )
    mapping_path = output / "mapping.spec"
    mapping_path.write_text("0:0\n", encoding="utf-8")
    command = [str(npusim.resolve()), "--program", str(artifact_path),
               "--linked-manifest", str(manifest_path),
               "--program-io", str(io_path),
               "--hardware-config", str(hardware_path),
               "--simulation-config", str(simulation.resolve()),
               "--mapping-config", str(mapping_path), "--trace-window", "1000000"]
    (output / "npusim_command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8",
    )
    expected_sgd_cores = Counter()
    binding_core = {binding.logical_core: binding.runtime_core_id
                    for binding in manifest.core_bindings}
    leaf_by_id = {fragment.id: fragment for fragment in leaves}
    for stream in manifest.core_streams:
        for ref in stream.records:
            fragment = leaf_by_id[ref.fragment_id]
            core_stream = next(item for item in fragment.core_streams
                               if item.logical_core == stream.logical_core)
            if core_stream.records[ref.fragment_record_index].opcode is RecordOpcode.SGD_UPDATE:
                expected_sgd_cores[binding_core[stream.logical_core]] += 1
    if sum(expected_sgd_cores.values()) != 4 * logical_states:
        raise RuntimeError("both DP replicas must execute both steps of every parameter update")
    native_sha = hashlib.sha256(npusim.read_bytes()).hexdigest()
    native_cwd = output / "native_run"
    native_cwd.mkdir(exist_ok=False)
    stdout = _run(command, native_cwd, timeout, output / "npusim.stdout.txt")
    sgd_cores = Counter(int(core) for core in re.findall(r"\[TRAIN_SGD\] core=(\d+)", stdout))
    markers = tuple(re.findall(
        r"\[PROGRAM_IO\] phase=(resolved|applied|verify) "
        r"mode=timing initializations=(\d+) probes=(\d+) "
        r"checksum=[0-9a-f]{64} pass=(\d+)", stdout,
    ))
    d2d = re.search(r"\[D2D\] link_sites=(\d+) link_units=(\d+)", stdout)
    cycles = re.search(r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout)
    if (tuple(row[0] for row in markers) != ("resolved", "applied", "verify")
            or any(row[1:] != (str(len(io.initializations)),
                                  str(len(io.output_probes)), "1") for row in markers)
            or stdout.count("[TRAIN_CE] core=") != 4 * columns
            or stdout.count("[TRAIN_CE_BACKWARD] core=") != 4 * columns
            or sgd_cores != expected_sgd_cores
            or d2d is None or int(d2d.group(1)) <= 0 or int(d2d.group(2)) <= 0
            or cycles is None or stdout.count("[SIM_RESULT] makespan_cycles=") != 1
            or "[DRAIN] router_residual=0" not in stdout
            or "[DRAIN] d2d_link_residual=0" not in stdout
            or hashlib.sha256(npusim.read_bytes()).hexdigest() != native_sha):
        raise RuntimeError("single-invocation TP×DP2 native CE/SGD/DTE/ProgramIO/drain proof failed")
    receipt.update({
        "program_invocations": 1, "native_executed": True,
        "npusim_sha256": native_sha, "makespan_cycles": int(cycles.group(1)),
        "actual_sgd_by_runtime_core": dict(sorted(sgd_cores.items())),
        "physical_core_count": len(manifest.core_bindings),
    })
    (output / "evidence.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    print("DP2_NATIVE_FRESH_PASS", receipt["makespan_cycles"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-columns", type=int, default=2)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    compile_dp2(args.output, columns=args.tp_columns, on_linked=lambda linked, plan, output, receipt: _execute(
        linked, plan, output, receipt,
        npusim=args.npusim, finalizer=args.finalizer,
        resolver=args.resolver, simulation=args.simulation,
        timeout=args.timeout,
    ))


if __name__ == "__main__":
    main()
