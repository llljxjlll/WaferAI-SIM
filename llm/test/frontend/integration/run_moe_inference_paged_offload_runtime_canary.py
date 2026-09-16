"""True 1x2/EP2 full MoE infer low-HBM resident reject vs paged DMA pair."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import (
    compile_moe_full_model_inference_sequence,
)
from llm.frontend.wafer_frontend.passes.moe_inference_paged_compile_sequence import (
    relink_moe_inference_paged_segment,
)
from llm.frontend.wafer_frontend.passes.moe_inference_paged_runtime import (
    build_moe_inference_paged_runtime,
)
from llm.frontend.wafer_frontend.passes.dense_inference_paged_program_io import (
    retarget_dense_inference_paged_sram_program_io,
)
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.external_memory import (
    ExternalMemoryConnection, ExternalMemoryFabric, ExternalMemoryLink,
)
from llm.frontend.wafer_frontend.schema.flexible_moe import (
    MoeRectActionKind, MoeRectFlowStage,
)
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryTier, MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily, WorkloadMemoryMode, WorkloadMemoryPolicy,
    WorkloadRunRequest,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_moe_compile_sequence import _manifest, _request
from llm.test.frontend.unit.test_moe_full_model_compile_sequence import _legacy_template
from llm.test.frontend.unit.test_workload_materialization import _capability

from .flexible_mesh_release_hardware import specialize_p5_large_release_hardware
from .run_dense_sequence_runtime_canary import _bind_native_hardware_to_fabric
from .run_moe_full_model_sequence_runtime_canary import (
    build_full_model_program_io, prove_full_model_dataflow,
)
from .run_dense_inference_paged_offload_runtime_canary import _useful_graph_digest


_ROOT = Path(__file__).resolve().parents[4]


def _run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> str:
    result = subprocess.run(command, cwd=cwd, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"returncode={result.returncode}: {' '.join(command)}\n"
            f"{result.stdout}"
        )
    return result.stdout


def _observe(stdout: str, contract: dict[str, object],
             *, expected_flows: int, outward: int, inward: int,
             remote_expert_matmuls: int) -> dict[str, object]:
    pattern = (
        r"\[MOE_INFERENCE_PAGED_DMA_EVENT\] index=(\d+) segment=(\d+) "
        r"core=(\d+) linked_record=(\d+) kind=(\w+) state_ref=(\S+) "
        r"lsu_bytes=(\d+) dma_bytes=(\d+) issue_cycle=(\d+) "
        r"completed_at_ticks=(\d+) lsu_dependency_complete=(\d+) pass=(\d+)"
    )
    actual = re.findall(pattern, stdout)
    signed = contract["events"]
    by_index = {int(event[0]): event for event in actual}
    if len(actual) != 89 or len(by_index) != 89:
        raise RuntimeError("89 real Core0/Core4 DMA completions missing or duplicate")
    for index, item in enumerate(signed):
        event = by_index.get(index)
        if event is None or (
            int(event[1]) != item["segment_index"] or
            int(event[2]) != item["runtime_core_id"] or
            int(event[3]) != item["linked_record_index"] or
            event[4] != item["kind"] or event[5] != item["state_ref"] or
            int(event[6]) != item["lsu_size_bytes"] or
            int(event[7]) != item["dma_size_bytes"] or
            event[10:] != ("1", "1") or int(event[9]) <= 0
        ):
            raise RuntimeError(f"real DMA completion lost signed MoE LSU gate {index}")
    if min(int(event[8]) for event in actual) <= 0:
        raise RuntimeError("MoE weights/KV were restored only at startup")
    segment_counts = Counter(int(event[1]) for event in actual)
    if segment_counts != Counter({0: 27, 1: 31, 2: 31}):
        raise RuntimeError("actual 27/31/31 mid-program gates changed")
    source_bytes = tuple((
        sum(int(event[7]) for event in actual if event[1] == str(segment) and
            event[4].endswith("before_load")),
        sum(int(event[7]) for event in actual if event[1] == str(segment) and
            event[4].endswith("after_store")),
    ) for segment in range(3))
    if source_bytes != ((1384, 896), (1576, 960), (1640, 1024)):
        raise RuntimeError(f"physical expert/parameter/KV traffic changed: {source_bytes}")
    kv = re.findall(
        r"\[MOE_INFERENCE_PAGED_KV\] version=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) authority=external functional=(\d+) pass=(\d+)",
        stdout,
    )
    if (len(kv) != 4 or tuple(x[:2] for x in kv) !=
        (("0", "0"), ("1", "128"), ("2", "192"), ("3", "256")) or
        any(x[3:] != ("0", "1") for x in kv)):
        raise RuntimeError("rank0 physical KV authority 0→1→2→3 failed")
    parameter = re.findall(
        r"\[MOE_INFERENCE_PAGED_PARAMETER_AUTHORITY\] version=(\d+) "
        r"physical_bytes=(\d+) digest=([0-9a-f]{64}) "
        r"immutable=(\d+) functional=(\d+) pass=(\d+)", stdout,
    )
    if (len(parameter) != 3 or
        tuple(x[0] for x in parameter) != ("1", "2", "3") or
        any(x[1] != "1384" or x[3:] != ("1", "0", "1") for x in parameter) or
        len({x[2] for x in parameter}) != 1):
        raise RuntimeError("expert physical STORE changed immutable weight authority")
    io = re.findall(
        r"\[MOE_INFERENCE_PAGED_EXTERNAL_PROGRAM_IO\] index=(\d+) "
        r"logits_probes=(\d+) kv_probes=(\d+) kv_bytes=(\d+) "
        r"parameter_immutable=(\d+) pending=(\d+) functional=(\d+) pass=(\d+)",
        stdout,
    )
    if io != [
        ("0", "1", "4", "128", "1", "0", "0", "1"),
        ("1", "1", "4", "192", "1", "0", "0", "1"),
        ("2", "1", "4", "256", "1", "0", "0", "1"),
    ]:
        raise RuntimeError(f"actual MoE SRAM logits/external authority probes failed: {io}")
    drain = re.findall(
        r"\[MOE_INFERENCE_PAGED_DMA_DRAIN\] events=(\d+) kv_probes=(\d+) "
        r"submitted=(\d+) completed=(\d+) external_read_bytes=(\d+) "
        r"external_write_bytes=(\d+) hbm_read_bytes=(\d+) "
        r"hbm_write_bytes=(\d+) pending=(\d+) dirty=(\d+) pinned=(\d+) pass=(\d+)",
        stdout,
    )
    if drain != [("89", "12", "89", "89", "4600", "2880", "2880",
                  "4600", "0", "0", "0", "1")]:
        raise RuntimeError(f"actual single shared route DMA byte/drain oracle failed: {drain}")
    segments = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)", stdout,
    )
    one_shot = re.findall(r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",stdout)
    makespan = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout)
    if (segments != [("0", "0"), ("1", "0"), ("2", "1")] or
        one_shot != [("3", "1")] or len(makespan) != 1 or
        stdout.count("[MOE_INFERENCE_PAGED_BINDING]") != 1):
        raise RuntimeError("true 1x2 full MoE Prefill+two Decode one-shot timeline failed")
    d2d = re.search(r"\[D2D_DATA\] in_pkts=(\d+) out_pkts=(\d+)", stdout)
    east = re.search(r"\[D2D_LINK\] idx=0 die0->die1 .*data_in=(\d+) data_out=(\d+)",stdout)
    west = re.search(r"\[D2D_LINK\] idx=1 die1->die0 .*data_in=(\d+) data_out=(\d+)",stdout)
    remote = len(re.findall(r"Core 4 start compute primitive Matmul_f\.", stdout))
    remote_swiglu = len(re.findall(r"Core 4 start compute primitive swiglu_forward\.", stdout))
    if (d2d is None or tuple(map(int,d2d.groups())) != (expected_flows,expected_flows)
        or east is None or tuple(map(int,east.groups())) != (outward,outward)
        or west is None or tuple(map(int,west.groups())) != (inward,inward)
        or remote != remote_expert_matmuls or
        remote_swiglu != remote_expert_matmuls // 3):
        raise RuntimeError("full remote EP2 expert dispatch/compute/combine did not run")
    residuals = re.findall(
        r"\[PROGRAM_MEMORY\] core=(\d+) .*lsu_residual=(\d+) dte_residual=(\d+)", stdout,
    )
    if residuals != [("0", "0", "0"), ("4", "0", "0")] or not all(
        marker in stdout for marker in (
            "[P5 P2P DRAIN] core=0 residual=0",
            "[P5 P2P DRAIN] core=4 residual=0",
            "[P5 P2P TIMING DRAIN] residual=0",
            "[DRAIN] router_residual=0", "[DRAIN] d2d_link_residual=0",
        )
    ):
        raise RuntimeError("full MoE P2P/DTE/LSU/router residual did not drain")
    return {
        "makespan_cycles": int(makespan[0]),
        "first_dma_issue_cycle": min(int(event[8]) for event in actual),
        "last_dma_completed_at_ticks": max(int(event[9]) for event in actual),
        "physical_external_read_bytes": 4600,
        "physical_external_write_bytes": 2880,
        "actual_dma_events": 89,
        "kv_versions": [0, 1, 2, 3],
        "kv_bytes": [0, 128, 192, 256],
        "kv_authority_digests": [item[2] for item in kv],
        "physical_parameter_authority_digest": parameter[0][2],
        "expert_retention_stores": 12,
        "external_kv_probes": 12,
        "remote_expert_matmuls": remote,
        "d2d_packets": expected_flows,
        "functional": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    # Bind both production tools before writing artifacts or launching finalizers.
    binary_sha = hashlib.sha256(args.npusim.resolve().read_bytes()).hexdigest()
    finalizer_sha = hashlib.sha256(args.finalizer.resolve().read_bytes()).hexdigest()
    for tool, actual in (("npusim", binary_sha), ("finalizer", finalizer_sha)):
        expected = getattr(args, f"expected_{tool}_sha256", None)
        if expected is not None and actual != expected:
            raise RuntimeError(f"replay {tool} SHA differs from requested frozen tool")
    previous_report = args.output.resolve()/"moe-inference-paged-runtime-evidence.json"
    if previous_report.is_file():
        previous = json.loads(previous_report.read_text(encoding="utf-8"))
        if (previous.get("npusim_sha256") != binary_sha or
            previous.get("finalizer_sha256") != finalizer_sha):
            raise RuntimeError("existing full MoE evidence binds different production tools; use a fresh output")
    request = _request(WorkloadFamily.MOE_INFERENCE)
    resident = _manifest(WorkloadFamily.MOE_INFERENCE)
    model_digest = canonical_digest(request.model)
    hbm = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM, location_ref=f"die:{die}",
            base_address=die << 30, capacity_bytes=1024,
            alignment_bytes=16,
        ) for die in (0, 1)
    )
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL, location_ref="host:0",
        base_address=0, capacity_bytes=2304, alignment_bytes=16,
    )
    try:
        materialize_workload_preflight(
            request, _capability(supported=True), capacities=hbm,
        )
    except SchemaError as error:
        resident_rejection = error.code
        resident_rejection_detail = str(error)
    else:
        raise RuntimeError("full MoE resident-only unexpectedly fits same 1024B die homes")
    if resident_rejection != "memory_capacity_exceeded":
        raise RuntimeError(f"resident-only failed for wrong reason: {resident_rejection_detail}")
    offload_request = WorkloadRunRequest.create(
        family=request.family, model=request.model, steps=request.steps,
        mesh=request.mesh, parallel=request.parallel,
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
            external_tier_ref="host:0",
        ), optimizer=request.optimizer, execution=request.execution,
    )
    offload = materialize_workload_preflight(
        offload_request, _capability(supported=True),
        capacities=(external, *hbm),
    )
    useful_resident = _useful_graph_digest(resident)
    useful_offload = _useful_graph_digest(offload)
    if useful_resident != useful_offload:
        raise RuntimeError("memory policy changed useful full MoE inference graph")
    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id, ingress_die_id=0,
        bytes_per_cycle=256, latency_cycles=2, queue_depth=2,
        max_outstanding=2,
    )
    routes = (
        ExternalMemoryConnection.create(
            link_ref=link.id, hbm_capacity_ref=hbm[0].id,
            target_die_id=0, route_die_ids=(0,),
            route_latency_cycles=0, route_bytes_per_cycle=None,
        ),
        ExternalMemoryConnection.create(
            link_ref=link.id, hbm_capacity_ref=hbm[1].id,
            target_die_id=1, route_die_ids=(0, 1),
            route_latency_cycles=1, route_bytes_per_cycle=256,
        ),
    )
    bridge = ExternalMemoryFabric.create(
        external_capacities=(external,), hbm_capacities=hbm,
        links=(link,), connections=routes,
    )
    source_fabric = physical_fabric_from_data(
        minimal_hardware(2, 1, sram_bytes=65536),
    )
    source_sequence = compile_moe_full_model_inference_sequence(
        resident, _legacy_template(), source_fabric,
        hbm_address_spaces=valid_hbm_address_spaces(source_fabric),
    )
    units = {item.id: item for item in source_sequence.moe_blocks.units}
    source_manifests = tuple(item.executable_manifest
                             for item in source_sequence.segments)
    paged_manifests = tuple(relink_moe_inference_paged_segment(item,step)
                            for step,item in enumerate(source_manifests))
    sidecar = build_moe_inference_paged_runtime(
        resident=resident, offload=offload,
        source_manifests=source_manifests,
        paged_manifests=paged_manifests, fabric=bridge,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sidecar_path = output / "moe_inference_paged_runtime.json"
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    expected_flows = []
    remote_expert_matmuls = 0
    operator_counts: Counter[str] = Counter()
    manifest_paths: list[Path] = []
    artifacts: list[Path] = []
    program_io_paths: list[Path] = []
    source_digests = []
    for step, (segment, original, paged) in enumerate(zip(
        source_sequence.segments, source_manifests, paged_manifests,
    )):
        block_units = tuple(units[ref] for ref in segment.moe_unit_refs)
        prove_full_model_dataflow(segment, block_units)
        expected_flows.extend(
            flow for unit in block_units for flow in unit.plan.flows
            if flow.stage in (MoeRectFlowStage.DISPATCH, MoeRectFlowStage.COMBINE)
        )
        remote_expert_matmuls += sum(
            3 if action.kind is MoeRectActionKind.EXPERT_FORWARD and
            action.rank == 1 and action.assignment_refs else 0
            for unit in block_units for action in unit.plan.actions
        )
        fragments = {item.id: item for item in original.fragments}
        for core in original.core_streams:
            for ref in core.records:
                fragment = fragments[ref.fragment_id]
                stream = next(item for item in fragment.core_streams
                              if item.logical_core == core.logical_core)
                operator_counts.update((stream.records[ref.fragment_record_index].opcode.name,))
        source_path = output/f"segment_{step}.source.linked.json"
        source_artifact = output/f"segment_{step}.source.npup"
        paged_path = output/f"segment_{step}.linked.json"
        paged_artifact = output/f"segment_{step}.npup"
        source_path.write_text(canonical_json(original), encoding="utf-8")
        paged_path.write_text(canonical_json(paged), encoding="utf-8")
        for manifest_path, artifact_path, manifest in (
            (source_path, source_artifact, original),
            (paged_path, paged_artifact, paged),
        ):
            report_path = output/f"{artifact_path.stem}.finalizer.json"
            if hashlib.sha256(args.finalizer.resolve().read_bytes()).hexdigest() != finalizer_sha:
                raise RuntimeError("source/paged full MoE finalizers changed within one signed run")
            _run((str(args.finalizer.resolve()), "--input",str(manifest_path),
                  "--output",str(artifact_path),"--report",str(report_path)),
                 cwd=output, timeout=120)
            if hashlib.sha256(args.finalizer.resolve().read_bytes()).hexdigest() != finalizer_sha:
                raise RuntimeError("source/paged full MoE finalizers changed within one signed run")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report["artifact_sha256"] != hashlib.sha256(
                    artifact_path.read_bytes()).hexdigest() or
                report["linked_manifest_id"] != manifest.id or
                report["linked_manifest_digest"] != canonical_digest(manifest)):
                raise RuntimeError("source/paged full MoE production finalizer closure drifted")
        original_io = build_full_model_program_io(
            segment, hashlib.sha256(source_artifact.read_bytes()).hexdigest(),
        )
        original_io.validate_against(original)
        paged_io = retarget_dense_inference_paged_sram_program_io(
            original_io, paged,
            hashlib.sha256(paged_artifact.read_bytes()).hexdigest(),
        )
        if len(paged_io.initializations) != 53 or len(paged_io.output_probes) != 1:
            raise RuntimeError("full MoE 53 SRAM timing seeds/one logits probe drifted")
        paged_io_path = output/f"segment_{step}.program_io.json"
        paged_io_path.write_text(canonical_json(paged_io), encoding="utf-8")
        manifest_paths.append(paged_path)
        artifacts.append(paged_artifact)
        program_io_paths.append(paged_io_path)
        source_digests.append(canonical_digest(original))
    hardware = json.loads(specialize_p5_large_release_hardware(1, 2))
    native_core_grid = _bind_native_hardware_to_fabric(hardware, source_fabric)
    if native_core_grid != (2, 2):
        raise RuntimeError("MoE low-HBM native/Fabric Die core grid must be 2x2")
    hardware["memory"]["sram_size"] = 131072
    hardware["memory"]["sram"]["capacity_bytes"] = 131072
    access = ["compute", "dte", "lsu", "legacy", "noc_rx"]
    hardware["memory"]["sram"]["regions"] = [
        {"name":name,"base_bytes":base,"size_bytes":size,"allocator":"block",
         "spillable":name == "input","access":access}
        for name,base,size in (("sram",0,4096),("input",4096,36864),
                               ("comm",40960,36864))
    ]
    physical = hardware["memory_system"]
    if (len(physical["hbm_stacks"]) != 2 or
        any(stack["backend"] != "behavioral" for stack in physical["hbm_stacks"])):
        raise RuntimeError("true same-hardware two behavioral HBM homes changed")
    for stack in physical["hbm_stacks"]:
        stack["capacity_bytes"] = 1024
    policy = physical["address_policy"]
    policy["home_ranges"] = [
        {"die_id":die,"base":die<<30,"size_bytes":1024}
        for die in (0,1)
    ]
    policy["stack_interleave_bytes"] = 1024
    policy["allow_gaps"] = True
    hardware_path = output/"hardware.json"
    hardware_path.write_text(
        json.dumps(hardware,sort_keys=True,separators=(",",":")),encoding="utf-8",
    )
    for die,home in enumerate(policy["home_ranges"]):
        if (home["base"] != hbm[die].base_address or
            home["size_bytes"] != hbm[die].capacity_bytes):
            raise RuntimeError("resident rejection and successful runtime hardware HBM differ")
    mapping_path = output/"mapping.spec"
    mapping_path.write_text("0:0\n",encoding="utf-8")
    hardware_sha = hashlib.sha256(hardware_path.read_bytes()).hexdigest()
    simulation_sha = hashlib.sha256(args.simulation.resolve().read_bytes()).hexdigest()
    actual = []
    flow_count = len(expected_flows)
    outward = sum(flow.source_rank == 0 and flow.destination_rank == 1
                  for flow in expected_flows)
    inward = sum(flow.source_rank == 1 and flow.destination_rank == 0
                 for flow in expected_flows)
    for fresh in (0,1):
        fresh_folder = output/f"fresh_{fresh}"
        fresh_folder.mkdir(parents=True,exist_ok=True)
        command = (
            str(args.npusim.resolve()),
            "--program-sequence", ",".join(map(str,artifacts)),
            "--linked-manifest-sequence", ",".join(map(str,manifest_paths)),
            "--program-io-sequence", ",".join(map(str,program_io_paths)),
            "--moe-inference-paged-runtime", str(sidecar_path),
            "--hardware-config", str(hardware_path),
            "--simulation-config", str(args.simulation.resolve()),
            "--mapping-config", str(mapping_path),
            "--trace-window", "1000000",
        )
        try:
            stdout = _run(command,cwd=args.npusim.resolve().parent,
                          timeout=args.timeout)
        except RuntimeError as error:
            (fresh_folder/"npusim.first-fault.txt").write_text(str(error),encoding="utf-8")
            raise
        (fresh_folder/"npusim.stdout.txt").write_text(stdout,encoding="utf-8")
        if hashlib.sha256(args.npusim.resolve().read_bytes()).hexdigest() != binary_sha:
            raise RuntimeError("two MoE fresh processes used different binaries")
        actual.append(_observe(stdout,sidecar,expected_flows=flow_count,
                               outward=outward,inward=inward,
                               remote_expert_matmuls=remote_expert_matmuls))
    if actual[0] != actual[1]:
        raise RuntimeError(f"two independent full MoE run digests drifted: {actual}")
    report = {
        **actual[0],
        "paired_fresh_runs":2,
        "mesh":"1x2","ep":2,"active_die_ids":[0,1],
        "frontend_core_grid":list(source_fabric.dies[0].noc_grid),
        "native_core_grid":list(native_core_grid),
        "resident_rejection_code":resident_rejection,
        "resident_rejection_detail":resident_rejection_detail,
        "hbm_capacity_bytes_per_die":1024,
        "p3_workspace_end_bytes_per_die":464,
        "highest_relative_state_end_bytes":960,
        "external_capacity_bytes":2304,
        "p3_external_parameter_reserved_bytes":1952,
        "physical_parameter_bytes":1384,
        "physical_kv_bytes":256,
        "unproduced_rank1_shared_replica_bytes":552,
        "model_digest":model_digest,
        "resident_useful_graph_digest":useful_resident,
        "offload_useful_graph_digest":useful_offload,
        "offload_memory_plan_digest":canonical_digest(offload.memory_plan),
        "source_manifest_digests":source_digests,
        "paged_manifest_digests":sidecar["linked_manifest_digests"],
        "paged_runtime_contract_id":sidecar["id"],
        "npusim_sha256":binary_sha,
        "finalizer_sha256":finalizer_sha,
        "hardware_sha256":hardware_sha,
        "simulation_sha256":simulation_sha,
        "operator_record_coverage":dict(sorted(operator_counts.items())),
    }
    (output/"moe-inference-paged-runtime-evidence.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8",
    )
    print("Full MoE inference bounded external expert/shared weight+KV offload PASS "
          "mesh=1x2 ep=2 HBM_per_die=1024 resident=memory_capacity_exceeded "
          "peak_relative_end=960 DMA=89 KV_versions=0,1,2,3 "
          f"fresh_runs=2 makespan={report['makespan_cycles']} functional=0")
    return report


def _args() -> argparse.Namespace:
    build = _ROOT/"build-debug-dense-infer-pager"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,
                        default=build/"moe-inference-paged-offload-canary")
    parser.add_argument("--npusim",type=Path,default=build/"npusim")
    parser.add_argument("--finalizer",type=Path,
                        default=_ROOT/"build-debug-final/npusim_program_finalizer")
    parser.add_argument("--simulation",type=Path,
                        default=_ROOT/"llm/test/program/p5_behavioral_simulation.json")
    parser.add_argument("--timeout",type=int,default=900)
    parser.add_argument("--expected-npusim-sha256")
    parser.add_argument("--expected-finalizer-sha256")
    args = parser.parse_args()
    for key in ("expected_npusim_sha256", "expected_finalizer_sha256"):
        value = getattr(args, key)
        if value is not None and (len(value) != 64 or
                                  any(ch not in "0123456789abcdef" for ch in value)):
            parser.error(f"{key} must be a lowercase SHA-256 hex digest")
    if args.timeout <= 0 or not all(getattr(args,key).is_file()
                                    for key in ("npusim","finalizer","simulation")):
        parser.error("npusim/finalizer/simulation must exist and timeout >0")
    return args


if __name__ == "__main__":
    run(_args())
