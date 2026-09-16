"""Execute two independent 1x4 two-step Dense TRAIN Fresh runs from one linked source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
    build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.n6 import _leaf_fragments
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json

from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware


def _execute(command: list[str], *, cwd: Path, timeout: int,
             stdout_path: Path | None = None) -> str:
    completed = subprocess.run(
        command, cwd=cwd, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )
    if stdout_path is not None:
        stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"native command failed (exit={completed.returncode}, argv={command!r}): "
            f"{completed.stdout[-3500:]}"
        )
    return completed.stdout


def run_linked_tp4_fresh(
    linked, physical, *, npusim: Path, finalizer: Path,
    simulation: Path, output: Path, timeout: int = 900,
) -> dict:
    """Use the production linked manifest, ProgramIO, and actual NpuSim receipts."""
    linked.validate()
    physical.physical_dag.validate_against(
        _leaf_fragments(linked.manifest.fragments), linked.manifest.core_streams,
        required_operation_ids=tuple(sorted({
            action.operation_ref for action in physical.physical_dag.actions
        })),
    )
    if len(physical.requirements.paths) != 120 or len(
        physical.physical_dag.state_version_edges
    ) != 60 or len(physical.physical_dag.transport_edges) < 96:
        raise RuntimeError("1x4 physical TP TRAIN closure is incomplete")
    active_dies = {binding.logical_core.die_id
                   for binding in linked.manifest.core_bindings}
    if active_dies != set(range(4)):
        raise RuntimeError("TP4 native manifest does not use physical dies 0..3")
    source = linked.source.replicas[0].lowering_context
    ce_values = {node.inputs[2] for node in source.ir1.nodes
                 if node.kind is OpKind.CE_BACKWARD}
    ce_seeds = {
        abi.id: b"\x00\x00\x80\x3f" * (abi.size_bytes // 4)
        for fragment in _leaf_fragments(linked.manifest.fragments) for abi in fragment.buffer_abi
        if abi.value_id in ce_values
    }
    expected_seeds = {(step, rank) for step in (0, 1) for rank in range(4)}
    if (len(ce_seeds) != 8
            or set(physical.loss_gradient_seed_abi_by_step) != expected_seeds
            or len(set(physical.loss_gradient_seed_abi_by_step.values())) != 8
            or set(ce_seeds) != set(
                physical.loss_gradient_seed_abi_by_step.values()
            ) or any(not seed or len(seed) % 4 for seed in ce_seeds.values())):
        raise RuntimeError("each TP4 step/rank requires an independent physical FP32 dLoss seed")
    states, expected = build_deterministic_timing_state_overrides(linked)
    if len(states) != 60 or expected:
        raise RuntimeError("all sixty TP4 HBM parameter shard seeds are required")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[4]
    dram_source = repository / "DRAMSys"
    dram_link = output / "DRAMSys"
    if dram_link.exists() or dram_link.is_symlink():
        if not dram_link.is_dir() or dram_link.resolve() != dram_source.resolve():
            raise RuntimeError("Fresh DRAMSys path differs from frozen repository source")
    else:
        dram_link.symlink_to(dram_source, target_is_directory=True)
    manifest_path = output / "full_tp4_two_step.linked.json"
    manifest_path.write_text(canonical_json(linked.manifest), encoding="utf-8")
    artifact_path = output / "full_tp4_two_step.npup"
    finalizer_report = output / "full_tp4_two_step.finalizer.json"
    finalizer_command = [
        str(finalizer.resolve()), "--input", str(manifest_path),
        "--output", str(artifact_path), "--report", str(finalizer_report),
    ]
    (output / "finalizer_command.json").write_text(
        json.dumps(finalizer_command, indent=2) + "\n", encoding="utf-8",
    )
    _execute(finalizer_command, cwd=output, timeout=180)
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    report = json.loads(finalizer_report.read_text(encoding="utf-8"))
    if (report.get("artifact_sha256") != artifact_hash
            or report.get("linked_manifest_id") != linked.manifest.id
            or report.get("linked_manifest_digest") != canonical_digest(linked.manifest)):
        raise RuntimeError("finalizer output does not bind exact linked manifest bytes")
    io = build_timing_program_io(
        linked, artifact_hash,
        sram_seed_overrides=ce_seeds, state_seed_overrides=states,
    )
    io_path = output / "full_tp4_two_step.program_io.json"
    io_path.write_text(canonical_json(io), encoding="utf-8")
    hardware = json.loads(specialize_p5_large_release_hardware(1, 4))
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
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    mapping_path = output / "mapping.spec"
    mapping_path.write_text("0:0\n", encoding="utf-8")
    command = [
        str(npusim.resolve()), "--program", str(artifact_path),
        "--linked-manifest", str(manifest_path), "--program-io", str(io_path),
        "--hardware-config", str(hardware_path),
        "--simulation-config", str(simulation.resolve()),
        "--mapping-config", str(mapping_path), "--trace-window", "1000000",
    ]
    (output / "npusim_command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8",
    )
    native_hash = hashlib.sha256(npusim.read_bytes()).hexdigest()
    counts = {
        op.name: sum(record.opcode is op
                     for fragment in _leaf_fragments(linked.manifest.fragments)
                     for stream in fragment.core_streams for record in stream.records)
        for op in (RecordOpcode.CROSS_ENTROPY_FORWARD,
                   RecordOpcode.CROSS_ENTROPY_BACKWARD,
                   RecordOpcode.SGD_UPDATE, RecordOpcode.DTE_SEND,
                   RecordOpcode.DTE_RECV, RecordOpcode.LOCAL_REDUCE)
    }
    if counts["CROSS_ENTROPY_FORWARD"] != 8 or counts[
        "CROSS_ENTROPY_BACKWARD"
    ] != 8 or counts["SGD_UPDATE"] != 120:
        raise RuntimeError("TP4 native CE or per-shard SGD inventory differs")
    cycles = []
    for fresh in range(2):
        folder = output / f"fresh_{fresh}"
        folder.mkdir(exist_ok=False)
        stdout = _execute(command, cwd=folder, timeout=timeout,
                          stdout_path=folder / "npusim.stdout.txt")
        sgd_cores = [int(core) for core in re.findall(
            r"\[TRAIN_SGD\] core=(\d+)", stdout,
        )]
        expected_cores = {
            binding.runtime_core_id for binding in linked.manifest.core_bindings
        }
        d2d = re.search(r"\[D2D\] link_sites=(\d+) link_units=(\d+)", stdout)
        markers = tuple(re.findall(
            r"\[PROGRAM_IO\] phase=(resolved|applied|verify) "
            r"mode=timing initializations=(\d+) probes=(\d+) "
            r"checksum=[0-9a-f]{64} pass=(\d+)", stdout,
        ))
        if (tuple(row[0] for row in markers) != ("resolved", "applied", "verify")
                or any(row[1:] != (
                    str(len(io.initializations)), str(len(io.output_probes)), "1"
                ) for row in markers)
                or stdout.count("[TRAIN_CE] core=") != 8
                or stdout.count("[TRAIN_CE_BACKWARD] core=") != 8
                or len(sgd_cores) != 120
                or {core: sgd_cores.count(core) for core in expected_cores}
                   != {core: 30 for core in expected_cores}
                or set(sgd_cores) != expected_cores
                or d2d is None or int(d2d.group(1)) <= 0
                or int(d2d.group(2)) <= 0
                or stdout.count("[SIM_RESULT] makespan_cycles=") != 1
                or "[DRAIN] router_residual=0" not in stdout
                or "[DRAIN] d2d_link_residual=0" not in stdout
                or hashlib.sha256(npusim.read_bytes()).hexdigest() != native_hash):
            raise RuntimeError(f"Fresh {fresh} lacks signed TP4 native/ProgramIO/drain closure")
        cycles.append(int(re.search(
            r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout,
        ).group(1)))
    if cycles[0] != cycles[1]:
        raise RuntimeError("two independent TP4 Fresh makespans disagree")
    result = {
        "schema_version": "dense_full_tp4_two_step_native_fresh/v1",
        "mesh": [1, 4], "steps": 2, "layers": 2,
        "parameter_state_shards": len(states),
        "state_version_edges": len(physical.physical_dag.state_version_edges),
        "transport_edges": len(physical.physical_dag.transport_edges),
        "independent_loss_seeds": len(ce_seeds),
        "program_io_initializations": len(io.initializations),
        "program_io_probes": len(io.output_probes),
        "native_opcode_counts": counts,
        "linked_manifest_digest": canonical_digest(linked.manifest),
        "artifact_sha256": artifact_hash, "npusim_sha256": native_hash,
        "program_invocations_per_fresh": 1, "fresh_count": 2,
        "makespan_cycles": cycles,
    }
    (output / "evidence.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return result
