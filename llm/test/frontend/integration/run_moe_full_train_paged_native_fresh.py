"""One independent source-to-NpuSim low-HBM MoE SGD offload Fresh run.

The full two-layer reverse is timing-only; this proves physical state paging,
not numerical gradient or optimizer correctness. Run under frontend_tmp.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.moe_full_train_paged_compile_sequence import (
    relink_moe_full_train_paged_step,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_paged_program_io import (
    retarget_moe_full_train_paged_program_io,
)
from llm.frontend.wafer_frontend.passes.moe_full_train_paged_runtime import (
    build_moe_full_train_paged_runtime,
)
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import LinkedProgramManifest
from llm.frontend.wafer_frontend.schema.external_memory import (
    ExternalMemoryConnection, ExternalMemoryFabric, ExternalMemoryLink,
)
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.program_io import ProgramIoContract
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest, canonical_json, load_json_dataclass,
)
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily, WorkloadMemoryMode, WorkloadMemoryPolicy, WorkloadRunRequest,
)
from llm.test.frontend.unit.test_moe_compile_sequence import _request
from llm.test.frontend.unit.test_workload_materialization import _capability


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path, log: Path, timeout: int,
         expected: int = 0) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout, check=False)
    log.write_text(result.stdout, encoding="utf-8")
    if (result.returncode == 0) != (expected == 0):
        raise RuntimeError(f"unexpected exit {result.returncode}: {command}; "
                           f"tail={result.stdout[-1800:]}")
    return result.stdout


def _source_contract():
    request = _request(WorkloadFamily.MOE_TRAINING, rows=1, columns=1)
    hbm = MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref="die:0", base_address=0,
        capacity_bytes=2560, alignment_bytes=16,
    )
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL, location_ref="host:0", base_address=0,
        capacity_bytes=2048, alignment_bytes=16,
    )
    try:
        materialize_workload_preflight(
            request, _capability(supported=True), capacities=(hbm,))
    except SchemaError as error:
        if error.code != "memory_capacity_exceeded":
            raise
    else:
        raise RuntimeError("same-model 2560B resident-only source unexpectedly fits")
    offload_request = WorkloadRunRequest.create(
        family=request.family, model=request.model, steps=request.steps,
        mesh=request.mesh, parallel=request.parallel,
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD, external_tier_ref="host:0"
        ), optimizer=request.optimizer, execution=request.execution,
    )
    offload = materialize_workload_preflight(
        offload_request, _capability(supported=True), capacities=(hbm, external),
    )
    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id, ingress_die_id=0,
        bytes_per_cycle=256, latency_cycles=2, queue_depth=2,
        max_outstanding=2,
    )
    connection = ExternalMemoryConnection.create(
        link_ref=link.id, hbm_capacity_ref=hbm.id, target_die_id=0,
        route_die_ids=(0,), route_latency_cycles=0,
        route_bytes_per_cycle=None,
    )
    fabric = ExternalMemoryFabric.create(
        external_capacities=(external,), hbm_capacities=(hbm,),
        links=(link,), connections=(connection,),
    )
    return offload, fabric


def _low_hbm_hardware(source: Path, target: Path) -> None:
    hardware = json.loads(source.read_text(encoding="utf-8"))
    system = hardware["memory_system"]
    stacks = system["hbm_stacks"]
    policy = system["address_policy"]
    if (len(stacks) != 1 or stacks[0]["backend"] != "behavioral" or
            stacks[0]["capacity_bytes"] != 33554432 or
            policy["home_ranges"] != [
                {"die_id": 0, "base": 0, "size_bytes": 33554432}
            ] or policy["stack_interleave_bytes"] != 33554432):
        raise RuntimeError("source hardware differs from audited MoE HBM profile")
    stacks[0]["capacity_bytes"] = 2560
    policy["home_ranges"] = [{"die_id": 0, "base": 0, "size_bytes": 2560}]
    policy["stack_interleave_bytes"] = 256
    target.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")))


def run(*, output: Path, npusim: Path, finalizer: Path, resolver: Path,
        simulation: Path, timeout: int) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Fresh output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    tools = (npusim.resolve(), finalizer.resolve(), resolver.resolve(),
             simulation.resolve())
    if any(not path.is_file() for path in tools):
        raise RuntimeError("native tool or simulation configuration missing")
    offload, fabric = _source_contract()
    source_dir = output / "source"
    _run([
        sys.executable, "-m",
        "llm.test.frontend.integration.run_moe_full_train_ce_backward_canary",
        "--output", str(source_dir), "--finalizer", str(finalizer.resolve()),
        "--resolver", str(resolver.resolve()), "--npusim", str(npusim.resolve()),
        "--all-parameter-sgd",
    ], cwd=Path(__file__).resolve().parents[4],
        log=output / "source.stdout.txt", timeout=timeout)
    original_receipt = json.loads((source_dir / "receipt.json").read_text())
    if (original_receipt["status"] != "all_parameter_sgd_physical_partial" or
            original_receipt["full_training_gate"] != "closed" or
            not original_receipt["source_tree_clean_at_entry"]):
        raise RuntimeError("full 19-parameter source program changed")
    source = tuple(load_json_dataclass(
        LinkedProgramManifest, source_dir / f"step{step}.linked.json"
    ) for step in range(2))
    source_io = tuple(load_json_dataclass(
        ProgramIoContract, source_dir / f"step{step}.program_io.json"
    ) for step in range(2))
    paged = tuple(relink_moe_full_train_paged_step(item, step)
                  for step, item in enumerate(source))
    prefix = output / "paged_step"
    for step, (original, linked, io) in enumerate(zip(source, paged, source_io)):
        base = output / f"paged_step{step}"
        manifest = Path(str(base) + ".linked.json")
        artifact = Path(str(base) + ".npup")
        program_io = Path(str(base) + ".program_io.json")
        manifest.write_text(canonical_json(linked), encoding="utf-8")
        _run([str(finalizer.resolve()), "--input", str(manifest),
              "--output", str(artifact), "--report",
              str(output / f"paged_step{step}.finalizer.json")],
             cwd=output, log=output / f"paged_step{step}.finalizer.log",
             timeout=timeout)
        rebound = retarget_moe_full_train_paged_program_io(
            io, original, linked, _sha(artifact))
        program_io.write_text(canonical_json(rebound), encoding="utf-8")
        _run([str(resolver.resolve()), "--resolve", str(manifest),
              str(artifact), str(program_io)], cwd=output,
             log=output / f"paged_step{step}.resolver.log", timeout=timeout)
    sidecar = build_moe_full_train_paged_runtime(
        offload, fabric, source, paged, source_io)
    if len(sidecar["events"]) != 130:
        raise RuntimeError("MoE offload physical event count changed")
    sidecar_path = output / "paged_runtime.json"
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True, separators=(",", ":")))
    hardware = output / "low_hbm_hardware.json"
    _low_hbm_hardware(source_dir / "hardware.json", hardware)
    mapping = source_dir / "mapping.spec"
    if mapping.read_text() != "0:0\n":
        raise RuntimeError("one-Die MoE physical mapping changed")
    native_args = [
        "--program-sequence", ",".join(str(source_dir / f"step{i}.npup")
                                       for i in range(2)),
        "--linked-manifest-sequence", ",".join(
            str(source_dir / f"step{i}.linked.json") for i in range(2)),
        "--program-io-sequence", ",".join(
            str(source_dir / f"step{i}.program_io.json") for i in range(2)),
        "--moe-all-sgd-partial-sequence", "--hardware-config", str(hardware),
        "--simulation-config", str(simulation.resolve()),
        "--mapping-config", str(mapping), "--trace-window", "1000000",
    ]
    rejected = _run([str(npusim.resolve()), *native_args], cwd=npusim.parent,
                    log=output / "resident_rejection.npusim.log",
                    timeout=timeout, expected=1)
    if ("[SIM_RESULT]" in rejected or
            "DecodeAddress: address not covered by any home range" not in rejected):
        raise RuntimeError("resident-only low-HBM run lacked the expected physical capacity rejection")
    canary = output / "native_canary"
    _run([sys.executable, "-m",
          "llm.test.frontend.integration.run_moe_full_train_paged_runtime_canary",
          "--npusim", str(npusim.resolve()), "--artifact-prefix", str(prefix),
          "--sidecar", str(sidecar_path), "--hardware-config", str(hardware),
          "--simulation-config", str(simulation.resolve()),
          "--mapping-config", str(mapping), "--output", str(canary)],
         cwd=Path(__file__).resolve().parents[4],
         log=output / "canary.stdout.txt", timeout=timeout)
    observation = json.loads((canary / "receipt.json").read_text())
    if observation["status"] != "pass" or observation["dma_events"] != 130:
        raise RuntimeError("native external DMA and negative gates did not pass")
    receipt = {
        "schema_version": "moe_full_train_paged_native_fresh/v1",
        "source_commit": original_receipt["source_commit"],
        "source_tree_clean_at_entry": original_receipt["source_tree_clean_at_entry"],
        "source_manifest_digests": [canonical_digest(item) for item in source],
        "paged_manifest_digests": [canonical_digest(item) for item in paged],
        "paged_artifact_sha256": [_sha(Path(str(output / f"paged_step{i}") + ".npup"))
                                  for i in range(2)],
        "sidecar_sha256": _sha(sidecar_path),
        "hardware_sha256": _sha(hardware),
        "npusim_sha256": _sha(npusim.resolve()),
        "resident_only_rejected": True,
        "resident_rejection_log_sha256": _sha(output / "resident_rejection.npusim.log"),
        "native": observation,
        "full_training_gate": "closed",
        "model_functional_verified": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("MOE_PAGED_FRESH_PASS", observation["makespan_cycles"], flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--resolver", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    run(output=args.output, npusim=args.npusim, finalizer=args.finalizer,
        resolver=args.resolver, simulation=args.simulation, timeout=args.timeout)


if __name__ == "__main__":
    main()
