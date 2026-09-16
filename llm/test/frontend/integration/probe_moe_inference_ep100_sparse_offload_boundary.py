"""Fail-closed EP100 low-HBM sparse physical mesh capacity/native boundary probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.workload_materialization import materialize_workload_preflight
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily, WorkloadMemoryMode, WorkloadMemoryPolicy,
    WorkloadMeshSpec, WorkloadParallelSpec, WorkloadRunCapability,
    WorkloadRunRequest,
)
from llm.test.frontend.integration import run_moe_full_model_native_mesh_matrix as canonical
from llm.test.frontend.unit.test_moe_compile_sequence import _request
from llm.test.frontend.unit.test_workload_materialization import _capability


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"JSON object expected: {path}")
    return value


def _case(rows: int, columns: int, baseline, binding: dict) -> dict:
    physical_count = rows * columns
    active = tuple(range(100))
    request = WorkloadRunRequest.create(
        family=baseline.family, model=baseline.model, steps=baseline.steps,
        mesh=WorkloadMeshSpec(rows, columns),
        parallel=WorkloadParallelSpec(ep=100, active_die_ids=active),
        memory=baseline.memory, optimizer=baseline.optimizer,
        execution=baseline.execution,
    )
    if canonical_digest(request.model) != canonical_digest(baseline.model):
        raise ValueError("fixed hundred-expert source model drifted")
    capability = WorkloadRunCapability.create(
        max_mesh_rows=rows, max_mesh_columns=columns,
        max_mesh_ranks=physical_count,
        families=_capability(supported=True).families,
    )
    hbm = tuple(MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref=f"die:{die}",
        base_address=die << 30, capacity_bytes=2048, alignment_bytes=16,
    ) for die in range(physical_count))
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL, location_ref="host:0",
        base_address=0, capacity_bytes=262144, alignment_bytes=16,
    )
    try:
        materialize_workload_preflight(request, capability, capacities=hbm)
    except SchemaError as error:
        if error.code != "memory_capacity_exceeded":
            raise
        rejection = str(error)
    else:
        raise ValueError("fixed hundred-expert resident unexpectedly fits low HBM")
    offload_request = WorkloadRunRequest.create(
        family=request.family, model=request.model, steps=request.steps,
        mesh=request.mesh, parallel=request.parallel,
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
            external_tier_ref="host:0",
        ), optimizer=request.optimizer, execution=request.execution,
    )
    offload = materialize_workload_preflight(
        offload_request, capability, capacities=(external, *hbm),
    )
    if tuple(offload.placement.active_die_ids) != active:
        raise ValueError("100 active and physical idle Die placement drifted")
    requests = {item.id: item for item in offload.memory_plan.requests}
    external_end = max(
        item.address + requests[item.request_ref].size_bytes
        for item in offload.memory_plan.allocations
        if requests[item.request_ref].tier is MemoryTier.EXTERNAL
    )
    if external_end != 254392:
        raise ValueError("fixed hundred-expert P3 external reservation drifted")
    links, endpoints = canonical._recompute_flow_evidence(
        binding["expected_remote_flows"], rows=rows, columns=columns,
    )
    if (links != binding["expected_d2d_links"] or len(links) != 6
            or sum(item["packet_hops"] for item in links) != 32
            or not set(endpoints).issubset(active)):
        raise ValueError("signed production D2D X-first route changes on sparse mesh")
    return {
        "mesh": f"{rows}x{columns}",
        "model_sha256": canonical_digest(request.model),
        "physical_die_count": physical_count,
        "active_die_count": 100,
        "idle_die_count": physical_count - 100,
        "resident_rejection": rejection,
        "offload_manifest_id": offload.id,
        "hbm_capacity_bytes_per_die": 2048,
        "external_capacity_bytes": 262144,
        "external_allocation_end_bytes": external_end,
        "expected_d2d_links": links,
        "expected_d2d_packet_hops": 32,
    }


def run(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    if not Path(sys.modules[_request.__module__].__file__).resolve().is_relative_to(source_root):
        raise ValueError("probe did not import frozen MoE source")
    root = args.output.resolve()
    if root.exists():
        raise ValueError("sparse boundary probe requires a new empty root")
    root.mkdir(parents=True)
    baseline = _request(WorkloadFamily.MOE_INFERENCE, rows=10, columns=10)
    binding = _json(args.m1_binding.resolve())
    ep100 = args.ep100_root.resolve()
    case = _json(ep100 / "full_fresh_evidence.json")
    signed = _json(ep100 / "binding.json")
    if (case["status"] != "verified" or case["binding"] != signed
            or _sha(args.npusim.resolve()) != signed["paths_sha256"]["npusim"]):
        raise ValueError("EP100 double-Fresh artifact/tool closure drifted")
    cases = [_case(rows, columns, baseline, binding)
             for rows, columns in ((11, 11), (12, 12))]
    fresh = ep100 / "full_fresh_0"
    if (_sha(fresh / "compiled/compile_canary_receipt.json")
            != case["fresh"][0]["compile_receipt_sha256"]
            or _sha(fresh / "moe_inference_paged_runtime.json")
            != case["fresh"][0]["sidecar_sha256"]
            or _sha(fresh / "hardware.json")
            != case["fresh"][0]["hardware_sha256"]):
        raise ValueError("signed EP100 source/sidecar/hardware changed")
    hardware = _json(fresh / "hardware.json")
    if hardware["die"] != {"x": 10, "y": 10}:
        raise ValueError("source hardware is not physical 10x10")
    stacks = hardware["memory_system"]["hbm_stacks"]
    homes = hardware["memory_system"]["address_policy"]["home_ranges"]
    if len(stacks) != 100 or len(homes) != 100:
        raise ValueError("signed EP100 hardware lacks 100 active HBM homes")
    for die in range(100, 121):
        stacks.append({**stacks[0], "stack_id": die, "compute_die_id": die})
        homes.append({"die_id": die, "base": die << 30, "size_bytes": 2048})
    hardware["die"] = {"x": 11, "y": 11}
    hardware_path = root / "physical_11x11_hardware.json"
    hardware_path.write_text(json.dumps(hardware, sort_keys=True, separators=(",", ":")))
    mapping_path = root / "mapping.spec"
    mapping_path.write_text("0:0\n")
    compiled = fresh / "compiled"
    files = lambda suffix: ",".join(
        str(compiled / f"segment_{step}.{suffix}") for step in range(3)
    )
    command = [
        str(args.npusim.resolve()),
        "--program-sequence", files("paged.npup"),
        "--linked-manifest-sequence", files("paged.linked.json"),
        "--program-io-sequence", files("program_io.json"),
        "--moe-inference-paged-runtime", str(fresh / "moe_inference_paged_runtime.json"),
        "--hardware-config", str(hardware_path),
        "--simulation-config", str(args.simulation.resolve()),
        "--mapping-config", str(mapping_path),
        "--trace-window", "1000000",
    ]
    completed = subprocess.run(
        command, cwd=args.npusim.resolve().parent,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=120, check=False,
    )
    log_path = root / "npusim.stdout.txt"
    log_path.write_text(completed.stdout)
    message = "paged MoE inference physical Die mesh differs from EP"
    if (completed.returncode == 0 or message not in completed.stdout
            or "[MOE_INFERENCE_PAGED_DMA_EVENT]" in completed.stdout
            or "[SIM_RESULT]" in completed.stdout):
        raise ValueError("expected exact 0ns sparse native fail-closed gate changed")
    result = {
        "schema_version": "moe-ep100-sparse-offload-boundary-v1",
        "status": "physical_native_blocked_before_dma",
        "cases": cases,
        "native_failure": {
            "mesh": "11x11", "exit_code": completed.returncode,
            "diagnostic": message,
            "stdout_sha256": _sha(log_path),
            "physical_hardware_sha256": _sha(hardware_path),
        },
        "source_ep100_receipt_sha256": _sha(ep100 / "full_fresh_evidence.json"),
        "source_m1_binding_sha256": _sha(args.m1_binding.resolve()),
        "npusim_sha256": _sha(args.npusim.resolve()),
        "probe_sha256": _sha(Path(__file__).resolve()),
    }
    (root / "boundary_receipt.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print("sparse offload capacity/route PASS; physical native BLOCKED before DMA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--m1-binding", type=Path, required=True)
    parser.add_argument("--ep100-root", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
