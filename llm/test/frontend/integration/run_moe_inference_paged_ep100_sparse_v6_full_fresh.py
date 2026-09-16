"""Two independent fixed-model hundred-Die low-HBM MoE offload runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.workload_materialization import materialize_workload_preflight
from llm.frontend.wafer_frontend.schema.artifact_manifest import LinkedProgramManifest
from llm.frontend.wafer_frontend.schema.external_memory import (
    ExternalMemoryConnection, ExternalMemoryFabric, ExternalMemoryLink,
)
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, load_json_dataclass
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadMeshSpec, WorkloadParallelSpec, WorkloadRunCapability,
    WorkloadFamily, WorkloadMemoryMode, WorkloadMemoryPolicy, WorkloadRunRequest,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.integration.flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from llm.test.frontend.integration.run_dense_sequence_runtime_canary import (
    _bind_native_hardware_to_fabric,
)
from llm.test.frontend.integration.run_dense_inference_paged_offload_runtime_canary import (
    _useful_graph_digest,
)
from llm.test.frontend.integration import run_moe_full_model_native_mesh_matrix as canonical
from llm.test.frontend.unit.test_moe_compile_sequence import _request
from llm.test.frontend.unit.test_workload_materialization import _capability


_ROOT = Path(__file__).resolve().parents[4]
_COMPILE = Path(__file__).with_name("run_moe_inference_paged_ep100_compile_canary.py")
_RELINK = _ROOT / "llm/frontend/wafer_frontend/passes/moe_inference_paged_compile_sequence_ep100.py"
_SIDECAR = _ROOT / "llm/frontend/wafer_frontend/passes/moe_inference_paged_runtime_ep100_sparse_v6.py"
_SCHEMA = "moe-ep100-sparse-v6-low-hbm-two-full-fresh-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if type(result) is not dict:
        raise ValueError(f"JSON object expected: {path}")
    return result


def _sidecar_module():
    for name, path in (
        ("llm.frontend.wafer_frontend.passes.moe_inference_paged_compile_sequence_ep100", _RELINK),
        ("llm.frontend.wafer_frontend.passes.moe_inference_paged_runtime_ep100_sparse_v6", _SIDECAR),
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[
        "llm.frontend.wafer_frontend.passes.moe_inference_paged_runtime_ep100_sparse_v6"
    ]


def _source_contract(rows: int, columns: int):
    if (rows, columns) not in ((11, 11), (12, 12)):
        raise ValueError("v6 supports only audited 11x11 and 12x12 sparse meshes")
    baseline = _request(WorkloadFamily.MOE_INFERENCE, rows=10, columns=10)
    active = tuple(range(100))
    request = WorkloadRunRequest.create(
        family=baseline.family, model=baseline.model, steps=baseline.steps,
        mesh=WorkloadMeshSpec(rows, columns),
        parallel=WorkloadParallelSpec(ep=100, active_die_ids=active),
        memory=baseline.memory, optimizer=baseline.optimizer,
        execution=baseline.execution,
    )
    capability = WorkloadRunCapability.create(
        max_mesh_rows=rows, max_mesh_columns=columns,
        max_mesh_ranks=rows * columns,
        families=_capability(supported=True).families,
    )
    hbm = tuple(MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref=f"die:{die}",
        base_address=die << 30, capacity_bytes=2048,
        alignment_bytes=16,
    ) for die in range(rows * columns))
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
        raise ValueError("fixed100 sparse resident fits low HBM unexpectedly")
    reference_hbm = tuple(MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref=f"die:{die}",
        base_address=die << 30, capacity_bytes=1 << 24,
        alignment_bytes=16,
    ) for die in range(rows * columns))
    resident = materialize_workload_preflight(
        request, capability, capacities=reference_hbm,
    )
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
    if (resident.request.model != offload.request.model
            or resident.request.steps != offload.request.steps
            or tuple(offload.placement.active_die_ids) != active
            or _useful_graph_digest(resident) != _useful_graph_digest(offload)):
        raise ValueError("sparse offload changed fixed100 model, graph or active placement")
    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id, ingress_die_id=0,
        bytes_per_cycle=256, latency_cycles=2, queue_depth=2,
        max_outstanding=2,
    )
    def x_first_route(die: int) -> tuple[int, ...]:
        x, y = die % columns, die // columns
        return tuple(range(x + 1)) + tuple(
            row * columns + x for row in range(1, y + 1)
        )
    connections = tuple(ExternalMemoryConnection.create(
        link_ref=link.id, hbm_capacity_ref=hbm[die].id,
        target_die_id=die, route_die_ids=x_first_route(die),
        route_latency_cycles=len(x_first_route(die)) - 1,
        route_bytes_per_cycle=(256 if die else None),
    ) for die in range(rows * columns))
    fabric = ExternalMemoryFabric.create(
        external_capacities=(external,), hbm_capacities=hbm,
        links=(link,), connections=connections,
    )
    return resident, offload, fabric, rejection

def _hardware(rows: int, columns: int) -> dict:
    source_fabric = physical_fabric_from_data(
        minimal_hardware(10, 10, sram_bytes=65536)
    )
    hardware = json.loads(specialize_p5_large_release_hardware(10, 10))
    if _bind_native_hardware_to_fabric(hardware, source_fabric) != (2, 2):
        raise ValueError("physical core grid differs from source Fabric")
    hardware["memory"]["sram_size"] = 131072
    hardware["memory"]["sram"]["capacity_bytes"] = 131072
    access = ["compute", "dte", "lsu", "legacy", "noc_rx"]
    hardware["memory"]["sram"]["regions"] = [
        {"name": name, "base_bytes": base, "size_bytes": size,
         "allocator": "block", "spillable": name == "input", "access": access}
        for name, base, size in (
            ("sram", 0, 4096), ("input", 4096, 36864),
            ("comm", 40960, 36864),
        )
    ]
    physical = hardware["memory_system"]
    if (len(physical["hbm_stacks"]) != 100
            or any(item["backend"] != "behavioral"
                   for item in physical["hbm_stacks"])):
        raise ValueError("physical hundred-Die behavioral HBM layout changed")
    for stack in physical["hbm_stacks"]:
        stack["capacity_bytes"] = 2048
    for die in range(100, rows * columns):
        physical["hbm_stacks"].append({**physical["hbm_stacks"][0],
                                       "stack_id": die, "compute_die_id": die})
    policy = physical["address_policy"]
    policy["home_ranges"] = [
        {"die_id": die, "base": die << 30, "size_bytes": 2048}
        for die in range(rows * columns)
    ]
    policy["stack_interleave_bytes"] = 2048
    policy["allow_gaps"] = True
    hardware["die"] = {"x": columns, "y": rows}
    if len(physical["hbm_stacks"]) != rows * columns:
        raise ValueError("physical sparse Die backend count changed")
    return hardware


def _native_observation(fresh: Path, contract: dict, rows: int, columns: int) -> dict:
    compiled = fresh / "compiled"
    paged_path = fresh / "moe_inference_paged_runtime.json"
    sidecar = _json(paged_path)
    if sidecar != contract or len(sidecar["events"]) != 1853:
        raise ValueError("physical EP100 sidecar differs from source-derived contract")
    receipt = _json(compiled / "compile_canary_receipt.json")
    if (receipt["status"] != "compile_finalizer_program_io_only"
            or receipt["mesh"] != "10x10" or receipt["ep"] != 100):
        raise ValueError("three production compile/finalizer/ProgramIO stages absent")
    for step, segment in enumerate(receipt["segments"]):
        if (step != segment["step"] or segment["runtime_core_ids"] != [die * 4 for die in range(100)]
                or segment["program_io_initializations"] != 837
                or segment["program_io_probes"] != 1):
            raise ValueError("EP100 compiled segment/source geometry drifted")
        for role in ("source", "paged"):
            for suffix, name in (("manifest_sha256", "linked.json"),
                                 ("artifact_sha256", "npup"),
                                 ("report_sha256", "finalizer.json")):
                path = compiled / f"segment_{step}.{role}.{name}"
                if _sha(path) != segment[role][suffix]:
                    raise ValueError("EP100 production artifact SHA drifted")
        if _sha(compiled / f"segment_{step}.program_io.json") != segment["program_io_sha256"]:
            raise ValueError("EP100 ProgramIO SHA drifted")
    if _json(fresh / "hardware.json") != _hardware(rows, columns):
        raise ValueError("actual physical 2048B/Die hardware drifted")
    text = (fresh / "npusim.stdout.txt").read_text(encoding="utf-8")
    if len(re.findall(
        r"\[MOE_INFERENCE_PAGED_BINDING\] .*?mesh=(\d+x\d+) ep=(\d+)",
        text,
    )) != 1 or re.findall(
        r"\[MOE_INFERENCE_PAGED_BINDING\] .*?mesh=(\d+x\d+) ep=(\d+)",
        text,
    )[0] != (f"{rows}x{columns}", "100"):
        raise ValueError("native paged MoE binding did not use actual physical mesh")
    dma = re.findall(
        r"\[MOE_INFERENCE_PAGED_DMA_EVENT\] index=(\d+) segment=(\d+) "
        r"core=(\d+) linked_record=(\d+) kind=(\w+) state_ref=(\S+) "
        r"lsu_bytes=(\d+) dma_bytes=(\d+) issue_cycle=(\d+) "
        r"completed_at_ticks=(\d+) lsu_dependency_complete=(\d+) pass=(\d+)",
        text,
    )
    if len(dma) != 1853 or {int(x[0]) for x in dma} != set(range(1853)):
        raise ValueError("1853 actual source-bound EP100 DMA events absent")
    by_index = {int(row[0]): row for row in dma}
    for index, event in enumerate(sidecar["events"]):
        row = by_index[index]
        if (int(row[1]) != event["segment_index"]
                or int(row[2]) != event["runtime_core_id"]
                or int(row[3]) != event["linked_record_index"]
                or row[4] != event["kind"] or row[5] != event["state_ref"]
                or int(row[6]) != event["lsu_size_bytes"]
                or int(row[7]) != event["dma_size_bytes"]
                or int(row[8]) <= 0 or int(row[9]) <= 0
                or row[10:] != ("1", "1")):
            raise ValueError(f"physical EP100 DMA gate {index} differs from linked StateABI")
    drain = re.findall(
        r"\[MOE_INFERENCE_PAGED_DMA_DRAIN\] events=(\d+) kv_probes=(\d+) "
        r"submitted=(\d+) completed=(\d+) external_read_bytes=(\d+) "
        r"external_write_bytes=(\d+) hbm_read_bytes=(\d+) "
        r"hbm_write_bytes=(\d+) pending=(\d+) dirty=(\d+) pinned=(\d+) pass=(\d+)",
        text,
    )
    if drain != [("1853", "12", "1853", "1853", "597304", "115776",
                  "115776", "597304", "0", "0", "0", "1")]:
        raise ValueError("physical EP100 external/HBM page traffic did not close")
    admission = re.findall(
        r"\[MOE_INFERENCE_PAGED_ADMISSION_DRAIN\] waited_events=(\d+) "
        r"wait_cycles=(\d+) capacity=(\d+) pass=(\d+)", text
    )
    if (len(admission) != 1 or int(admission[0][0]) <= 0
            or int(admission[0][1]) <= 0 or admission[0][2:] != ("2", "1")):
        raise ValueError("hundred EP cores did not queue behind fixed two-outstanding link")
    if re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", text) == []:
        raise ValueError("native makespan missing")
    cycles = int(re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", text)[0])
    if cycles <= 0:
        raise ValueError("native makespan invalid")
    memory = [(int(core), int(issued), int(completed), int(lsu), int(dte))
              for core, issued, completed, lsu, dte in re.findall(
                  r"\[PROGRAM_MEMORY\] core=(\d+) lsu_issued=(\d+) "
                  r"lsu_completed=(\d+) .*?lsu_residual=(\d+) dte_residual=(\d+)",
                  text,
              )]
    if (len(memory) != 100 or {item[0] for item in memory} != {die * 4 for die in range(100)}
            or any(issued <= 0 or issued != completed or lsu or dte
                   for _, issued, completed, lsu, dte in memory)):
        raise ValueError("physical hundred-core LSU/DTE memory did not close")
    idle = re.findall(
        r"\[MOE_INFERENCE_PAGED_IDLE_DIE\] die=(\d+) hbm_requests=(\d+) "
        r"reads=(\d+) writes=(\d+) bytes=(\d+) completed=(\d+) "
        r"failed=(\d+) pass=(\d+)", text,
    )
    if (len(idle) != rows * columns - 100
            or [int(item[0]) for item in idle] != list(range(100, rows * columns))
            or any(item[1:] != ("0", "0", "0", "0", "0", "0", "1")
                   for item in idle)):
        raise ValueError("each physical idle Die must have zero HBM transactions")
    if (re.findall(r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)", text)
            != [("0", "0"), ("1", "0"), ("2", "1")]
            or len(re.findall(r"\[MOE_INFERENCE_PAGED_EXTERNAL_PROGRAM_IO\] index=\d+", text)) != 3
            or "[DENSE_SEQUENCE_DRAIN] segments=3 one_shot=1" not in text):
        raise ValueError("physical Prefill/Decode/Decode or ProgramIO incomplete")
    kv = re.findall(
        r"\[MOE_INFERENCE_PAGED_KV\] version=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) authority=external functional=(\d+) pass=(\d+)",
        text,
    )
    if (len(kv) != 4 or [int(item[1]) for item in kv] != [0, 128, 192, 256]
            or any(item[3:] != ("0", "1") for item in kv)):
        raise ValueError("external KV authority did not advance through all versions")
    expected_links = receipt["expected_d2d_links"]
    expected = {
        (row["source_die"], row["destination_die"], row["direction"]):
            (row["request_hops"], row["request_hops"],
             2 * row["request_hops"], 2 * row["request_hops"],
             row["packet_hops"], row["packet_hops"])
        for row in expected_links
    }
    if len(expected) != len(expected_links):
        raise ValueError("production source remote-flow link identity repeated")
    actual = {}
    for match in re.finditer(
        r"\[D2D_LINK\] idx=\d+ die(\d+)->die(\d+) dir=([EWNS]) "
        r"req_in=(\d+) req_out=(\d+) ack_in=(\d+) ack_out=(\d+) "
        r"data_in=(\d+) data_out=(\d+)", text,
    ):
        source, destination, direction, *counts = match.groups()
        key = (int(source), int(destination), direction)
        if key in actual:
            raise ValueError("physical directed D2D link repeated")
        actual[key] = tuple(map(int, counts))
    if actual != expected:
        raise ValueError(f"physical MoE D2D links differ from signed flows: {actual!r}")
    request_hops = sum(item["request_hops"] for item in expected_links)
    packet_hops = sum(item["packet_hops"] for item in expected_links)
    if (re.findall(r"\[D2D_DATA\] in_pkts=(\d+) out_pkts=(\d+)", text)
            != [(str(packet_hops), str(packet_hops))]
            or re.findall(
                r"\[D2D_TYPE\] request_in=(\d+) request_out=(\d+) "
                r"ack_in=(\d+) ack_out=(\d+) data_in=(\d+) data_out=(\d+)",
                text,
            ) != [tuple(map(str, (request_hops, request_hops,
                                  2 * request_hops, 2 * request_hops,
                                  packet_hops, packet_hops)))]):
        raise ValueError("physical MoE D2D aggregate differs from signed flows")
    return {
        "schema_version": _SCHEMA,
        "sidecar_sha256": _sha(paged_path),
        "compile_receipt_sha256": _sha(compiled / "compile_canary_receipt.json"),
        "hardware_sha256": _sha(fresh / "hardware.json"),
        "native_stdout_sha256": _sha(fresh / "npusim.stdout.txt"),
        "resident_rejection_code": "memory_capacity_exceeded",
        "ep": 100, "mesh": f"{rows}x{columns}",
        "physical_die_count": rows * columns,
        "idle_die_ids": list(range(100, rows * columns)),
        "idle_hbm_zero_die_count": len(idle), "hbm_capacity_bytes_per_die": 2048,
        "external_capacity_bytes": 262144,
        "dma_events": len(dma), "external_read_bytes": 597304,
        "external_write_bytes": 115776,
        "admission_waited_events": int(admission[0][0]),
        "admission_wait_cycles": int(admission[0][1]),
        "native_cycles": cycles, "memory": memory, "kv": kv,
        "d2d_packets": packet_hops, "d2d_links": expected_links,
    }


def _binding(args: argparse.Namespace, source_root: Path) -> dict:
    paths = {
        "driver": Path(__file__), "compile_driver": _COMPILE,
        "relinker": _RELINK, "sidecar_builder": _SIDECAR,
        "npusim": args.npusim, "finalizer": args.finalizer,
        "simulation": args.simulation, "m1_binding": args.m1_binding,
    }
    imported = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if type(name) is not str or not name.endswith(".py"):
            continue
        path = Path(name).resolve()
        if path.is_relative_to(source_root):
            imported[str(path.relative_to(source_root))] = _sha(path)
    config_root = args.npusim.resolve().parent.parent / "DRAMSys" / "configs"
    config_files = sorted(path for path in config_root.rglob("*") if path.is_file())
    if not config_root.is_dir() or not config_files:
        raise ValueError("frozen NpuSim DRAMSys configuration bundle missing")
    config_hash = hashlib.sha256()
    for path in config_files:
        config_hash.update(str(path.relative_to(config_root)).encode("utf-8"))
        config_hash.update(b"\0")
        config_hash.update(path.read_bytes())
        config_hash.update(b"\0")
    return {
        "paths_sha256": {key: _sha(path.resolve()) for key, path in paths.items()},
        "dramsys_config_bundle_sha256": config_hash.hexdigest(),
        "dramsys_config_files": len(config_files),
        "frozen_imported_source_sha256": dict(sorted(imported.items())),
        "source_root": str(source_root),
        "mesh": args.mesh,
    }


def run(args: argparse.Namespace) -> None:
    if args.mesh not in ("11x11", "12x12"):
        raise ValueError("only audited hundred-Die meshes are supported")
    rows, columns = (int(item) for item in args.mesh.split("x"))
    source_root = args.source_root.resolve()
    if not Path(sys.modules[_request.__module__].__file__).resolve().is_relative_to(source_root):
        raise ValueError("MoE source was not imported from frozen root")
    root = args.output_root.resolve()
    resident, offload, fabric, rejection = _source_contract(rows, columns)
    signed_flow_binding = _json(args.m1_binding.resolve())
    physical_links, endpoints = canonical._recompute_flow_evidence(
        signed_flow_binding["expected_remote_flows"], rows=rows, columns=columns,
    )
    if (physical_links != signed_flow_binding["expected_d2d_links"]
            or len(physical_links) != 6 or sum(row["packet_hops"] for row in physical_links) != 32
            or not set(endpoints).issubset(range(100))):
        raise ValueError("fixed100 signed D2D X-first route changed on sparse mesh")
    builder = _sidecar_module()
    binding = _binding(args, source_root)
    if root.exists():
        if not args.resume or _json(root / "binding.json") != binding:
            raise ValueError("existing root or source/native tool drifted")
    else:
        root.mkdir(parents=True)
        (root / "binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True))
    observed = []
    for index in range(2):
        fresh = root / f"full_fresh_{index}"
        compiled = fresh / "compiled"
        if not fresh.exists():
            fresh.mkdir()
            command = [
                sys.executable, str(_COMPILE),
                "--source-root", str(source_root),
                "--finalizer", str(args.finalizer.resolve()),
                "--output", str(compiled),
                "--mesh", "10x10",
            ]
            env = dict(os.environ, PYTHONPATH=str(source_root))
            done = subprocess.run(command, cwd=source_root, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=args.process_timeout, check=False)
            (fresh / "compile.stdout.txt").write_text(done.stdout)
            if done.returncode:
                raise RuntimeError(f"independent EP100 compile failed Fresh{index}")
        elif not args.resume:
            raise ValueError("partial or existing Fresh cannot be overwritten")
        source = tuple(load_json_dataclass(
            LinkedProgramManifest, compiled / f"segment_{step}.source.linked.json"
        ) for step in range(3))
        paged = tuple(load_json_dataclass(
            LinkedProgramManifest, compiled / f"segment_{step}.paged.linked.json"
        ) for step in range(3))
        compile_receipt = _json(compiled / "compile_canary_receipt.json")
        if compile_receipt["expected_d2d_links"] != physical_links:
            raise ValueError("production source flow route differs on physical sparse mesh")
        contract = builder.build_moe_inference_paged_runtime_ep100_sparse_v6(
            resident=resident, offload=offload,
            source_manifests=source, paged_manifests=paged, fabric=fabric,
        )
        contract_path = fresh / "moe_inference_paged_runtime.json"
        hardware_path = fresh / "hardware.json"
        mapping_path = fresh / "mapping.spec"
        native_path = fresh / "npusim.stdout.txt"
        if args.resume:
            if (_json(contract_path) != contract
                    or _json(hardware_path) != _hardware(rows, columns)
                    or mapping_path.read_text() != "0:0\n"):
                raise ValueError("source-derived EP100 sidecar or hardware drifted")
        else:
            contract_path.write_text(json.dumps(contract, sort_keys=True, separators=(",", ":")))
            hardware_path.write_text(json.dumps(_hardware(rows, columns), sort_keys=True, separators=(",", ":")))
            mapping_path.write_text("0:0\n")
            files = lambda suffix: ",".join(
                str(compiled / f"segment_{step}.{suffix}") for step in range(3)
            )
            command = [
                str(args.npusim.resolve()),
                "--program-sequence", files("paged.npup"),
                "--linked-manifest-sequence", files("paged.linked.json"),
                "--program-io-sequence", files("program_io.json"),
                "--moe-inference-paged-runtime", str(contract_path),
                "--hardware-config", str(hardware_path),
                "--simulation-config", str(args.simulation.resolve()),
                "--mapping-config", str(mapping_path),
                "--trace-window", "1000000",
            ]
            done = subprocess.run(command, cwd=args.npusim.resolve().parent,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, timeout=args.native_timeout, check=False)
            native_path.write_text(done.stdout)
            if done.returncode:
                raise RuntimeError(f"physical EP100 native Fresh{index} failed exit={done.returncode}; see {native_path}")
        item = _native_observation(fresh, contract, rows, columns)
        evidence_path = fresh / "evidence.json"
        if args.resume:
            if _json(evidence_path) != json.loads(json.dumps(item)):
                raise ValueError("physical EP100 Fresh receipt drifted")
        else:
            evidence_path.write_text(json.dumps(item, indent=2, sort_keys=True))
            print(f"MoE EP100 low-HBM PASS full_fresh_{index}", flush=True)
        observed.append(item)
    stable = lambda item: {key: value for key, value in item.items()
                           if key != "native_stdout_sha256"}
    if stable(observed[0]) != stable(observed[1]):
        raise ValueError("two independent EP100 full materializations disagree")
    case = {
        "schema_version": _SCHEMA, "status": "verified",
        "binding": binding, "resident_rejection": rejection,
        "offload_manifest_id": offload.id,
        "model_sha256": canonical_digest(resident.request.model),
        "fresh": observed,
    }
    case_path = root / "full_fresh_evidence.json"
    if args.resume:
        if _json(case_path) != json.loads(json.dumps(case)):
            raise ValueError("full EP100 case receipt drifted")
    else:
        case_path.write_text(json.dumps(case, indent=2, sort_keys=True))
    if _binding(args, source_root) != binding:
        raise ValueError("EP100 source/native tool drifted during full Fresh")
    print("MoE EP100 low-HBM VERIFIED two independent full materializations", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--npusim", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--m1-binding", type=Path, required=True)
    parser.add_argument("--native-timeout", type=int, default=1200)
    parser.add_argument("--process-timeout", type=int, default=1800)
    parser.add_argument("--mesh", choices=("11x11", "12x12"), default="11x11")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
