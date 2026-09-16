"""Versioned fixed-model MoE inference on a larger physical mesh.

The production full-model and MoE-block compilers operate on their unchanged
logical full-EP mesh. This adapter admits only an identity-prefix placement
into a larger physical mesh and only if every signed remote flow has the same
X-first physical hop path. Both source requests, artifacts, hardware, and
native observations are bound and reopened for each independent Fresh run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.workload_materialization import materialize_workload_preflight
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily, WorkloadMeshSpec, WorkloadParallelSpec,
    WorkloadRunCapability, WorkloadRunRequest,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit.test_moe_compile_sequence import _request
from llm.test.frontend.unit.test_workload_materialization import _capability

from llm.test.frontend.integration import run_moe_full_model_native_mesh_matrix as canonical


_PAIRS = {"1x6": "1x4", "11x11": "10x10"}
_SCHEMA = "moe-full-model-sparse-physical-mesh-v2"
_MEMORY = canonical._MEMORY
_LINK = canonical._LINK
_KV = canonical._KV
_MAKESPAN = canonical._MAKESPAN
_D2D = canonical._D2D
_D2D_TYPE = canonical._D2D_TYPE


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"expected object: {path}")
    return value


def _size(shape: str) -> tuple[int, int]:
    rows, columns = (int(value) for value in shape.split("x"))
    if rows < 1 or columns < 1:
        raise ValueError("mesh dimensions must be positive")
    return rows, columns


def _source_snapshot(source_root: Path) -> dict[str, str]:
    sources = {"v2_driver": _sha(Path(__file__).resolve())}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if type(name) is not str or not name.endswith(".py"):
            continue
        path = Path(name).resolve()
        if path.is_relative_to(source_root):
            sources[str(path.relative_to(source_root))] = _sha(path)
    return dict(sorted(sources.items()))


def _physical_source(logical_shape: str, physical_shape: str) -> dict[str, object]:
    logical_rows, logical_columns = _size(logical_shape)
    physical_rows, physical_columns = _size(physical_shape)
    active_count = logical_rows * logical_columns
    physical_count = physical_rows * physical_columns
    active = tuple(range(active_count))
    baseline = _request(
        WorkloadFamily.MOE_INFERENCE,
        rows=logical_rows,
        columns=logical_columns,
    )
    request = WorkloadRunRequest.create(
        family=baseline.family,
        model=baseline.model,
        steps=baseline.steps,
        mesh=WorkloadMeshSpec(physical_rows, physical_columns),
        parallel=WorkloadParallelSpec(ep=active_count, active_die_ids=active),
        memory=baseline.memory,
        optimizer=baseline.optimizer,
        execution=baseline.execution,
    )
    capability = WorkloadRunCapability.create(
        max_mesh_rows=physical_rows,
        max_mesh_columns=physical_columns,
        max_mesh_ranks=physical_count,
        families=_capability(supported=True).families,
    )
    capacities = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{die}",
            base_address=0,
            capacity_bytes=1 << 24,
            alignment_bytes=16,
        )
        for die in range(physical_count)
    )
    manifest = materialize_workload_preflight(
        request, capability, capacities=capacities,
    )
    fabric = physical_fabric_from_data(
        minimal_hardware(physical_columns, physical_rows, sram_bytes=65536)
    )
    if (tuple(manifest.placement.active_die_ids) != active
            or len(fabric.dies) != physical_count
            or len(fabric.links) != 2 * (
                physical_rows * (physical_columns - 1)
                + physical_columns * (physical_rows - 1)
            )):
        raise ValueError("physical source materialization is incomplete")
    if (canonical_digest(request.model) != canonical_digest(baseline.model)
            or canonical_digest(request.steps) != canonical_digest(baseline.steps)):
        raise ValueError("fixed MoE model or Prefill/Decode/Decode steps drifted")
    return {
        "physical_request_case_id": request.case_id,
        "physical_request_sha256": canonical_digest(request),
        "physical_manifest_id": manifest.id,
        "physical_manifest_sha256": canonical_digest(manifest),
        "logical_request_case_id": baseline.case_id,
        "logical_request_sha256": canonical_digest(baseline),
        "model_sha256": canonical_digest(request.model),
        "steps_sha256": canonical_digest(request.steps),
        "expert_count": request.model.num_experts,
        "ep_rank_count": request.parallel.ep,
        "active_die_ids": list(active),
        "idle_die_ids": list(range(active_count, physical_count)),
        "physical_die_count": physical_count,
        "physical_directed_link_count": len(fabric.links),
        "logical_mesh": logical_shape,
        "physical_mesh": physical_shape,
        "placement_kind": "identity_prefix_route_equivalent",
    }


def _extend_hardware(logical: dict, physical_shape: str, active_count: int) -> dict:
    rows, columns = _size(physical_shape)
    physical_count = rows * columns
    if logical.get("die") != {
        "x": _size(_PAIRS[physical_shape])[1],
        "y": _size(_PAIRS[physical_shape])[0],
    }:
        raise ValueError("logical hardware has the wrong source mesh")
    hardware = json.loads(json.dumps(logical))
    hardware["die"] = {"x": columns, "y": rows}
    system = hardware["memory_system"]
    stacks = system["hbm_stacks"]
    homes = system["address_policy"]["home_ranges"]
    if (len(stacks) != active_count or len(homes) != active_count
            or [item["compute_die_id"] for item in stacks] != list(range(active_count))
            or [item["die_id"] for item in homes] != list(range(active_count))):
        raise ValueError("logical HBM home layout does not match active Die")
    stride = homes[0]["size_bytes"]
    if any(item["base"] != die * stride or item["size_bytes"] != stride
           for die, item in enumerate(homes)):
        raise ValueError("logical HBM home layout is not uniform and contiguous")
    for die in range(active_count, physical_count):
        stacks.append({**stacks[0], "stack_id": die, "compute_die_id": die})
        homes.append({"die_id": die, "base": die * stride, "size_bytes": stride})
    if ([item["compute_die_id"] for item in stacks] != list(range(physical_count))
            or [item["die_id"] for item in homes] != list(range(physical_count))):
        raise ValueError("physical HBM does not cover every Die")
    return hardware


def _flow_route_binding(binding: dict, logical_shape: str, physical_shape: str) -> list[dict]:
    logical_rows, logical_columns = _size(logical_shape)
    physical_rows, physical_columns = _size(physical_shape)
    flows = binding["expected_remote_flows"]
    logical_links, _ = canonical._recompute_flow_evidence(
        flows, rows=logical_rows, columns=logical_columns,
    )
    physical_links, endpoints = canonical._recompute_flow_evidence(
        flows, rows=physical_rows, columns=physical_columns,
    )
    if (logical_links != binding["expected_d2d_links"]
            or physical_links != logical_links
            or any(rank >= _size(logical_shape)[0] * _size(logical_shape)[1]
                   for rank in endpoints)):
        raise ValueError("signed remote flows are not route-equivalent on physical mesh")
    return physical_links


def _physical_observation(
    directory: Path, compiled: Path, source: dict[str, object],
    tool_sha: dict[str, str],
) -> dict[str, object]:
    logical = str(source["logical_mesh"])
    physical = str(source["physical_mesh"])
    rows, columns = _size(physical)
    active = source["active_die_ids"]
    physical_count = rows * columns
    canonical.audit_fresh(compiled, logical)
    receipt = _json(compiled / "compiled_receipt.json")
    binding = _json(compiled / "source_tool_binding.json")
    hardware_path = directory / "hardware.json"
    hardware = _json(hardware_path)
    if (receipt["source_request_sha256"] != source["logical_request_sha256"]
            or receipt["workload_case_id"] != source["logical_request_case_id"]
            or receipt["active_die_ids"] != active
            or receipt["compiled_core_die_ids"] != [active, active, active]
            or hardware["die"] != {"x": columns, "y": rows}
            or len(hardware["memory_system"]["hbm_stacks"]) != physical_count
            or len(hardware["memory_system"]["address_policy"]["home_ranges"]) != physical_count):
        raise ValueError("physical source, linked programs, or hardware drifted")
    if _extend_hardware(_json(compiled / "hardware.json"), physical, len(active)) != hardware:
        raise ValueError("physical hardware differs from exact signed extension")
    expected_links = _flow_route_binding(binding, logical, physical)
    runtime_path = directory / "npusim.stdout.txt"
    runtime = runtime_path.read_text(encoding="utf-8")
    memories = tuple(tuple(int(part) for part in match.groups())
                     for match in _MEMORY.finditer(runtime))
    cores = {rank * 4 for rank in active}
    if (len(memories) != len(active)
            or {item[0] for item in memories} != cores
            or any(item[1] <= 0 or item[2] != item[1] or item[3] or item[4]
                   for item in memories)):
        raise ValueError("physical runtime memory does not cover exactly active Die")
    p2p = tuple((int(core), int(residual)) for core, residual in re.findall(
        r"\[P5 P2P DRAIN\] core=(\d+) residual=(\d+)", runtime
    ))
    if (len(p2p) != len(binding["expected_p2p_core_ids"])
            or sorted(core for core, _ in p2p) != binding["expected_p2p_core_ids"]
            or any(residual for _, residual in p2p)):
        raise ValueError("physical P2P endpoints did not drain")
    kv = tuple((int(index), int(size), digest, int(passed))
               for index, size, digest, passed in _KV.findall(runtime))
    if (len(kv) != 3 or [item[0] for item in kv] != [0, 1, 2]
            or [item[1] for item in kv] != [128, 192, 256]
            or any(item[3] != 1 for item in kv)):
        raise ValueError("physical Prefill/Decode/Decode KV did not close")
    compiled_runtime = (compiled / "npusim.stdout.txt").read_text(encoding="utf-8")
    original_kv = tuple((int(index), int(size), digest, int(passed))
                        for index, size, digest, passed in _KV.findall(compiled_runtime))
    if kv != original_kv:
        raise ValueError("physical KV differs from source-compiled baseline")
    segments = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)", runtime
    )
    probes = re.findall(
        r"\[DENSE_SEQUENCE_PROGRAM_IO\] index=(\d+) probes=(\d+) pass=(\d+)", runtime
    )
    drains = re.findall(r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)", runtime)
    cycles = tuple(int(item) for item in _MAKESPAN.findall(runtime))
    if (segments != [("0", "0"), ("1", "0"), ("2", "1")]
            or probes != [("0", "1", "1"), ("1", "1", "1"), ("2", "1", "1")]
            or drains != [("3", "1")]
            or len(cycles) != 1 or cycles[0] <= 0):
        raise ValueError("physical segment, ProgramIO, drain, or makespan is incomplete")
    links = []
    for match in _LINK.finditer(runtime):
        (_, source_die, destination_die, direction,
         req_in, req_out, ack_in, ack_out, data_in, data_out) = match.groups()
        source_die, destination_die = int(source_die), int(destination_die)
        req_in, req_out, ack_in, ack_out, data_in, data_out = map(
            int, (req_in, req_out, ack_in, ack_out, data_in, data_out)
        )
        if (req_in != req_out or ack_in != ack_out or ack_in != 2 * req_in
                or data_in != data_out):
            raise ValueError("physical link counters did not close")
        links.append({
            "source_die": source_die,
            "destination_die": destination_die,
            "direction": direction,
            "request_hops": req_in,
            "packet_hops": data_in,
        })
    links.sort(key=lambda item: (item["source_die"], item["destination_die"], item["direction"]))
    if links != expected_links:
        raise ValueError("physical D2D links differ from signed source flows")
    requests = sum(item["request_hops"] for item in links)
    packets = sum(item["packet_hops"] for item in links)
    if (_D2D.findall(runtime) != [(str(packets), str(packets))]
            or _D2D_TYPE.findall(runtime) != [(
                str(requests), str(requests), str(2 * requests), str(2 * requests),
                str(packets), str(packets),
            )]):
        raise ValueError("physical aggregate D2D counters drifted")
    return {
        "schema_version": _SCHEMA,
        "source": source,
        "tool_sha256": tool_sha,
        "logical_sequence_digest": receipt["sequence_digest"],
        "logical_source_binding_sha256": _sha(compiled / "source_tool_binding.json"),
        "logical_artifact_files_sha256": binding["artifact_files_sha256"],
        "physical_hardware_sha256": _sha(hardware_path),
        "physical_runtime_log_sha256": _sha(runtime_path),
        "physical_die_count": physical_count,
        "active_die_ids": active,
        "idle_die_ids": source["idle_die_ids"],
        "memory": memories,
        "p2p_cores": p2p,
        "kv": kv,
        "d2d_links": links,
        "makespan_cycles": cycles[0],
    }


def _stable(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items()
            if key not in {"physical_runtime_log_sha256"}}


def _run_command(command: list[str], cwd: Path, timeout: int, log: Path) -> None:
    try:
        finished = subprocess.run(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as error:
        log.write_text(str(error), encoding="utf-8")
        raise
    log.write_text(finished.stdout, encoding="utf-8")
    if finished.returncode:
        raise RuntimeError(f"command failed exit={finished.returncode}: {command[0]}; see {log}")


def run(args: argparse.Namespace) -> dict[str, object]:
    physical = args.physical_mesh
    if physical not in _PAIRS:
        raise ValueError("v2 supports only measured 1x6 and 11x11 physical meshes")
    logical = _PAIRS[physical]
    source_root = args.source_root.resolve()
    if canonical._ROOT != source_root:
        raise ValueError("imported canonical source differs from --source-root")
    tools = {
        name: getattr(args, name).resolve()
        for name in ("finalizer", "resolver", "npusim", "simulation")
    }
    if any(not path.is_file() for path in tools.values()):
        raise ValueError("all native tools and simulation must exist")
    source_sha = _source_snapshot(source_root)
    tool_sha = {name: _sha(path) for name, path in tools.items()}
    binding = {
        "schema_version": _SCHEMA,
        "driver_sha256": _sha(Path(__file__).resolve()),
        "canonical_runner_sha256": _sha(source_root / "llm/test/frontend/integration/run_moe_full_model_sequence_runtime_canary.py"),
        "logical_mesh": logical,
        "physical_mesh": physical,
        "source_root": str(source_root),
        "tool_sha256": tool_sha,
        "source_sha256": source_sha,
    }
    root = args.output_root.resolve()
    binding_path = root / "v2_binding.json"
    if root.exists():
        if not args.resume or _json(binding_path) != binding:
            raise ValueError("v2 output exists or source/tool binding drifted")
    else:
        root.mkdir(parents=True)
        binding_path.write_text(json.dumps(binding, indent=2, sort_keys=True), encoding="utf-8")
    observations = []
    for index in range(2):
        fresh = root / f"fresh{index}"
        compiled = fresh / "compiled"
        physical_dir = fresh / "physical"
        evidence_path = fresh / "v2_evidence.json"
        if fresh.exists():
            if not args.resume or not evidence_path.is_file():
                raise ValueError("partial v2 Fresh is not resumable")
            source = _json(fresh / "physical_source.json")
            observed = _physical_observation(physical_dir, compiled, source, tool_sha)
            if _json(evidence_path) != json.loads(json.dumps(observed)):
                raise ValueError("v2 Fresh evidence drifted")
            observations.append(observed)
            continue
        fresh.mkdir()
        source = _physical_source(logical, physical)
        (fresh / "physical_source.json").write_text(
            json.dumps(source, indent=2, sort_keys=True), encoding="utf-8"
        )
        command = [
            sys.executable, "-m",
            "llm.test.frontend.integration.run_moe_full_model_sequence_runtime_canary",
            "--mesh-size", logical,
            "--output", str(compiled),
            "--finalizer", str(tools["finalizer"]),
            "--resolver", str(tools["resolver"]),
            "--npusim", str(tools["npusim"]),
            "--simulation", str(tools["simulation"]),
            "--timeout", str(args.native_timeout),
        ]
        _run_command(command, source_root, args.process_timeout,
                     fresh / "logical_runner.stdout.txt")
        canonical.audit_fresh(compiled, logical)
        binding_data = _json(compiled / "source_tool_binding.json")
        _flow_route_binding(binding_data, logical, physical)
        physical_dir.mkdir()
        hardware = _extend_hardware(
            _json(compiled / "hardware.json"), physical, len(source["active_die_ids"])
        )
        (physical_dir / "hardware.json").write_text(
            json.dumps(hardware, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        (physical_dir / "mapping.spec").write_text("0:0\n", encoding="utf-8")
        files = lambda suffix: ",".join(str(compiled / f"segment_{step}.{suffix}") for step in range(3))
        native = [
            str(tools["npusim"]),
            "--program-sequence", files("npup"),
            "--linked-manifest-sequence", files("linked.json"),
            "--program-io-sequence", files("program_io.json"),
            "--hardware-config", str(physical_dir / "hardware.json"),
            "--simulation-config", str(tools["simulation"]),
            "--mapping-config", str(physical_dir / "mapping.spec"),
            "--trace-window", "1000000",
        ]
        _run_command(native, tools["npusim"].parent, args.native_timeout,
                     physical_dir / "npusim.stdout.txt")
        observed = _physical_observation(physical_dir, compiled, source, tool_sha)
        evidence_path.write_text(json.dumps(observed, indent=2, sort_keys=True), encoding="utf-8")
        observations.append(observed)
        print(f"MoE sparse v2 PASS {physical} fresh{index}", flush=True)
    if _stable(observations[0]) != _stable(observations[1]):
        raise ValueError("v2 physical Fresh executions differ")
    case = {"schema_version": _SCHEMA, "status": "verified", "fresh": observations}
    case_path = root / "v2_case_evidence.json"
    if args.resume:
        if _json(case_path) != json.loads(json.dumps(case)):
            raise ValueError("v2 case receipt drifted")
    else:
        case_path.write_text(json.dumps(case, indent=2, sort_keys=True), encoding="utf-8")
    if _source_snapshot(source_root) != source_sha:
        raise ValueError("v2 imported source drifted during Fresh executions")
    print(f"MoE sparse v2 VERIFIED physical={physical} logical={logical} two fresh", flush=True)
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-mesh", choices=tuple(_PAIRS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    for name in ("finalizer", "resolver", "npusim", "simulation"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--native-timeout", type=int, default=1200)
    parser.add_argument("--process-timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
