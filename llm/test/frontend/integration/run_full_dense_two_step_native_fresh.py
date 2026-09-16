"""Two independent Fresh NpuSim runs of one source-complete Dense TRAIN program."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.passes.flexible_dense_train import (
    build_flexible_dense_train_plan,
)
from llm.frontend.wafer_frontend.passes.full_dense_training_two_step_runtime import (
    compile_full_dense_two_step_native,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides, build_timing_program_io,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RecordOpcode
from llm.frontend.wafer_frontend.schema.ir0 import OpKind
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.test.frontend.unit.test_flexible_dense_train import _hardware, _spec

from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware


def _execute(command: list[str], *, cwd: Path, timeout: int) -> str:
    result = subprocess.run(command, cwd=cwd, check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"command exited {result.returncode}: {command[0]}\n"
            f"{result.stdout[-3200:]}")
    return result.stdout


def _verify_fresh(output: str, *, expected_updates: int) -> int:
    markers = tuple(re.findall(
        r"\[PROGRAM_IO\] phase=(resolved|applied|verify) "
        r"mode=timing initializations=(\d+) probes=(\d+) "
        r"checksum=[0-9a-f]{64} pass=(\d+)", output,
    ))
    if (tuple(row[0] for row in markers) != ("resolved", "applied", "verify")
            or any(row[1:] != ("253", "2", "1") for row in markers)
            or output.count("[TRAIN_CE] core=") != 2
            or output.count("[TRAIN_CE_BACKWARD] core=") != 2
            or output.count("[TRAIN_SGD] core=") != expected_updates
            or output.count("[SIM_RESULT] makespan_cycles=") != 1
            or "[START_DATA] enqueued=1 injected=1 accepted=1 delivered=1 consumed=1 completed=1" not in output
            or "[DRAIN] router_residual=0" not in output
            or "[DRAIN] d2d_link_residual=0" not in output):
        raise RuntimeError("two-step Dense native Fresh lacked real CE/backward/SGD, ProgramIO or drain closure")
    return int(re.search(r"\[SIM_RESULT\] makespan_cycles=(\d+)", output).group(1))


def run(*, npusim: Path, finalizer: Path, simulation: Path,
        output: Path, timeout: int = 300, mesh: tuple[int, int] = (1, 1)) -> dict:
    rows, columns = mesh
    plan = build_flexible_dense_train_plan(_spec(rows, columns),
                                           RectMeshSpec(rows, columns))
    physical = compile_full_dense_two_step_native(plan, _hardware(rows, columns))
    linked = physical.program
    manifest = linked.manifest
    if (len(physical.physical_dag.actions) != 280
            or len(physical.physical_dag.state_version_edges) != 15
            or len(manifest.fragments) != 280):
        raise RuntimeError("full Dense two-step physical coverage changed")
    source = linked.source.replicas[0].lowering_context
    ce_values = {node.inputs[2] for node in source.ir1.nodes
                 if node.kind is OpKind.CE_BACKWARD}
    ce_seeds = {
        abi.id: b"\x00\x00\x80\x3f" * (abi.size_bytes // 4)
        for fragment in manifest.fragments for abi in fragment.buffer_abi
        if abi.value_id in ce_values
    }
    if (len(ce_seeds) != 2 or any(not payload or len(payload) % 4
                                  for payload in ce_seeds.values())
            or set(ce_seeds) != set(physical.loss_gradient_seed_abi_by_step.values())):
        raise RuntimeError("two independent physical FP32 dLoss seeds are required")
    states, expected = build_deterministic_timing_state_overrides(linked)
    if len(states) != 15 or expected:
        raise RuntimeError("all fifteen physical HBM parameter seeds are required")
    output.mkdir(parents=True, exist_ok=True)
    dram_root = Path(__file__).resolve().parents[4] / "DRAMSys"
    dram_link = output / "DRAMSys"
    if dram_link.exists() or dram_link.is_symlink():
        if not dram_link.is_dir() or dram_link.resolve() != dram_root.resolve():
            raise RuntimeError("Fresh DRAMSys path differs from repository source")
    else:
        dram_link.symlink_to(dram_root, target_is_directory=True)
    manifest_path = output / "full_two_step.linked.json"
    artifact_path = output / "full_two_step.npup"
    finalizer_path = output / "full_two_step.finalizer.json"
    manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
    _execute([str(finalizer.resolve()), "--input", str(manifest_path),
              "--output", str(artifact_path), "--report", str(finalizer_path)],
             cwd=output, timeout=120)
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    report = json.loads(finalizer_path.read_text(encoding="utf-8"))
    if (report.get("artifact_sha256") != artifact_hash
            or report.get("linked_manifest_id") != manifest.id
            or report.get("linked_manifest_digest") != canonical_digest(manifest)):
        raise RuntimeError("native finalizer bytes/digest/manifest closure failed")
    program_io = build_timing_program_io(
        linked, artifact_hash,
        sram_seed_overrides=ce_seeds,
        state_seed_overrides=states,
    )
    io_path = output / "full_two_step.program_io.json"
    io_path.write_text(canonical_json(program_io), encoding="utf-8")
    hardware = json.loads(specialize_p5_large_release_hardware(rows, columns))
    hardware["memory"]["sram_size"] = 1 << 20
    hardware["memory"]["sram"]["capacity_bytes"] = 1 << 20
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 1 << 20
    dram_config = (Path(__file__).resolve().parents[4] /
                   "DRAMSys/configs/hbm2-example.json")
    if not dram_config.is_file():
        raise RuntimeError("source-bound DRAMSys configuration is missing")
    for stack in hardware["memory_system"]["hbm_stacks"]:
        stack["channel_dram_config"] = str(dram_config)
    hw_path = output / "hardware.json"
    hw_path.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")),
                       encoding="utf-8")
    mapping = output / "mapping.spec"
    mapping.write_text("0:0\n", encoding="utf-8")
    command = [str(npusim.resolve()), "--program", str(artifact_path),
               "--linked-manifest", str(manifest_path),
               "--program-io", str(io_path),
               "--hardware-config", str(hw_path),
               "--simulation-config", str(simulation.resolve()),
               "--mapping-config", str(mapping),
               "--trace-window", "1000000"]
    cycles = []
    for fresh in range(2):
        folder = output / f"fresh_{fresh}"
        folder.mkdir(exist_ok=True)
        stdout = _execute(command, cwd=folder, timeout=timeout)
        (folder / "npusim.stdout.txt").write_text(stdout, encoding="utf-8")
        cycles.append(_verify_fresh(stdout, expected_updates=30))
    if cycles[0] != cycles[1]:
        raise RuntimeError("independent Fresh two-step makespan drifted")
    receipt = {
        "schema_version": "dense_full_two_step_native_fresh/v1",
        "mesh": [rows, columns], "steps": 2, "layers": 2,
        "parameter_states": 15, "backbone_reverse_per_step": 24,
        "wgrad_per_step": 15, "sgd_per_step": 15,
        "state_version_edges": len(physical.physical_dag.state_version_edges),
        "program_invocations_per_fresh": 1,
        "fresh_count": 2, "program_io_initializations": 253,
        "program_io_loss_probes": 2,
        "native_opcode_counts": {
            opcode.name: sum(record.opcode is opcode
                             for fragment in manifest.fragments
                             for stream in fragment.core_streams
                             for record in stream.records)
            for opcode in (RecordOpcode.CROSS_ENTROPY_FORWARD,
                           RecordOpcode.CROSS_ENTROPY_BACKWARD,
                           RecordOpcode.SGD_UPDATE, RecordOpcode.LSU_LOAD,
                           RecordOpcode.LSU_STORE)
        },
        "native_artifact_sha256": artifact_hash,
        "npusim_sha256": hashlib.sha256(npusim.read_bytes()).hexdigest(),
        "makespan_cycles": cycles,
        "numeric_mode": "timing",
    }
    (output / "evidence.json").write_text(json.dumps(receipt, ensure_ascii=False,
                               sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"full Dense two-step native double Fresh PASS {receipt}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", default="1x1")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    dims = args.mesh.split("x")
    if len(dims) != 2:
        parser.error("mesh must be ROWSxCOLUMNS")
    run(npusim=args.npusim, finalizer=args.finalizer,
        simulation=args.simulation, output=args.output,
        timeout=args.timeout, mesh=(int(dims[0]), int(dims[1])))


if __name__ == "__main__":
    main()
