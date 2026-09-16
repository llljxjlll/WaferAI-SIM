"""Frozen EP6 low-HBM source, capacity and first pager-blocker audit.

This is a compile/finalizer/ProgramIO probe. It does not claim EP6 paging or
native external-memory execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import (
    compile_moe_full_model_inference_sequence,
)
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import StateKind
from llm.frontend.wafer_frontend.schema.flexible_moe import MoeRectFlowStage
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryTier, MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily, WorkloadMemoryMode, WorkloadMemoryPolicy, WorkloadRunRequest,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.integration.run_dense_inference_paged_offload_runtime_canary import (
    _useful_graph_digest,
)
from llm.test.frontend.integration.run_moe_full_model_sequence_runtime_canary import (
    _flow_link_expectations, build_full_model_program_io,
    prove_full_model_dataflow,
)
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_moe_compile_sequence import _manifest, _request
from llm.test.frontend.unit.test_moe_full_model_compile_sequence import _legacy_template
from llm.test.frontend.unit.test_workload_materialization import _capability


_ROOT = Path(__file__).resolve().parents[4]
_RELINK = _ROOT / "llm/frontend/wafer_frontend/passes/moe_inference_paged_compile_sequence_ep4.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ep4_relinker():
    name = "llm.frontend.wafer_frontend.passes.moe_inference_paged_compile_sequence_ep4"
    spec = importlib.util.spec_from_file_location(name, _RELINK)
    if spec is None or spec.loader is None:
        raise ValueError("EP4 relinker not found")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _case(rows: int, columns: int, output: Path, finalizer: Path, relinker) -> dict:
    die_count = rows * columns
    request = _request(WorkloadFamily.MOE_INFERENCE, rows=rows, columns=columns)
    if (die_count != 6 or request.model.num_experts != 6
            or request.model.num_layers != 2 or request.steps.inference is None
            or (request.steps.inference.prefill_tokens,
                request.steps.inference.decode_steps,
                request.steps.inference.request_count) != (2, 2, 2)):
        raise ValueError("official fixed six-expert, two-layer, three-step profile changed")
    hbm = tuple(MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref=f"die:{die}",
        base_address=die << 30, capacity_bytes=1024, alignment_bytes=16,
    ) for die in range(die_count))
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL, location_ref="host:0",
        base_address=0, capacity_bytes=8192, alignment_bytes=16,
    )
    try:
        materialize_workload_preflight(request, _capability(supported=True),
                                       capacities=hbm)
    except SchemaError as error:
        if error.code != "memory_capacity_exceeded":
            raise
        resident_rejection = str(error)
    else:
        raise ValueError("EP6 same-model resident unexpectedly fits 1024B/Die")
    offload_request = WorkloadRunRequest.create(
        family=request.family, model=request.model, steps=request.steps,
        mesh=request.mesh, parallel=request.parallel,
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
            external_tier_ref="host:0",
        ), optimizer=request.optimizer, execution=request.execution,
    )
    offload = materialize_workload_preflight(
        offload_request, _capability(supported=True), capacities=(external, *hbm),
    )
    resident = _manifest(WorkloadFamily.MOE_INFERENCE, rows=rows, columns=columns)
    if (resident.request.model != offload.request.model
            or resident.request.steps != offload.request.steps
            or _useful_graph_digest(resident) != _useful_graph_digest(offload)):
        raise ValueError("EP6 offload changed the fixed useful model/graph")
    fabric = physical_fabric_from_data(minimal_hardware(columns, rows, sram_bytes=65536))
    sequence = compile_moe_full_model_inference_sequence(
        resident, _legacy_template(), fabric,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    if len(sequence.segments) != 3:
        raise ValueError("EP6 full-model source lacks all three segments")
    output.mkdir(parents=True)
    units = {item.id: item for item in sequence.moe_blocks.units}
    segment_receipts = []
    expected_flows = []
    first_blocker = None
    for step, segment in enumerate(sequence.segments):
        segment_units = tuple(units[ref] for ref in segment.moe_unit_refs)
        prove_full_model_dataflow(segment, segment_units)
        expected_flows.extend(
            flow for unit in segment_units for flow in unit.plan.flows
            if flow.stage in (MoeRectFlowStage.DISPATCH, MoeRectFlowStage.COMBINE)
            and flow.source_rank != flow.destination_rank
        )
        manifest = segment.executable_manifest
        manifest.validate("ep6.source")
        streams = [(item.runtime_core_id, len(item.records))
                   for item in manifest.core_streams]
        expected_streams = ([(0, 207), (4, 52), (8, 52), (12, 52), (16, 30), (20, 30)]
                            if step == 0 else
                            [(0, 199), (4, 52), (8, 30), (12, 30), (16, 30), (20, 30)])
        abis = relinker.unique_state_abis(manifest)
        if (streams != expected_streams or len(manifest.fragments) != (39 if step == 0 else 43)
                or len(manifest.state_operand_bindings) != (51 if step == 0 else 55)
                or len(abis) != 39):
            raise ValueError("true EP6 source core, fragment or StateABI inventory changed")
        abi_inventory = [{
            "die": die,
            "count": sum(abi.die_id == die for abi in abis),
            "parameter_bytes": sum(abi.size_bytes for abi in abis
                                   if abi.die_id == die and abi.kind in
                                   (StateKind.PARAMETER, StateKind.TRAINABLE_PARAMETER)),
            "max_parameter_page_bytes": max(abi.size_bytes for abi in abis
                                             if abi.die_id == die and abi.kind in
                                             (StateKind.PARAMETER, StateKind.TRAINABLE_PARAMETER)),
            "parameter_page_sizes": sorted(abi.size_bytes for abi in abis
                                           if abi.die_id == die and abi.kind in
                                           (StateKind.PARAMETER, StateKind.TRAINABLE_PARAMETER)),
        } for die in range(die_count)]
        manifest_path = output / f"segment_{step}.source.linked.json"
        artifact_path = output / f"segment_{step}.source.npup"
        report_path = output / f"segment_{step}.source.finalizer.json"
        manifest_path.write_text(canonical_json(manifest), encoding="utf-8")
        done = subprocess.run(
            [str(finalizer), "--input", str(manifest_path),
             "--output", str(artifact_path), "--report", str(report_path)],
            cwd=output, capture_output=True, text=True, timeout=180, check=False,
        )
        if done.returncode:
            raise RuntimeError(f"EP6 production source finalizer failed step={step}: {done.stdout}{done.stderr}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["artifact_sha256"] != _sha(artifact_path)
                or report["linked_manifest_id"] != manifest.id
                or report["linked_manifest_digest"] != canonical_digest(manifest)):
            raise ValueError("EP6 production source finalizer closure changed")
        program_io = build_full_model_program_io(segment, _sha(artifact_path))
        program_io.validate_against(manifest)
        io_path = output / f"segment_{step}.source.program_io.json"
        io_path.write_text(canonical_json(program_io), encoding="utf-8")
        try:
            relinker.relink_moe_inference_paged_segment_ep4(manifest, step)
        except SchemaError as error:
            blocker = {"code": error.code, "message": str(error)}
            if first_blocker is None:
                first_blocker = blocker
            elif blocker != first_blocker:
                raise ValueError("EP4 relinker blocker differs across real EP6 segments")
        else:
            raise ValueError("EP4 four-core relinker unexpectedly accepted EP6 source")
        segment_receipts.append({
            "step": step, "manifest_id": manifest.id,
            "fragment_count": len(manifest.fragments), "core_streams": streams,
            "state_binding_count": len(manifest.state_operand_bindings),
            "state_abi_count": len(abis), "abi_inventory": abi_inventory,
            "manifest_sha256": _sha(manifest_path),
            "artifact_sha256": _sha(artifact_path),
            "finalizer_report_sha256": _sha(report_path),
            "program_io_sha256": _sha(io_path),
            "program_io_initializations": len(program_io.initializations),
            "program_io_probes": len(program_io.output_probes),
        })
        print(f"EP6 {rows}x{columns} source/finalizer/ProgramIO PASS step={step}", flush=True)
    links = _flow_link_expectations(expected_flows, rows, columns)
    requests = {item.id: item for item in offload.memory_plan.requests}
    hbm_workspace_end = [max(
        item.address + requests[item.request_ref].size_bytes - (die << 30)
        for item in offload.memory_plan.allocations
        if requests[item.request_ref].tier is MemoryTier.HBM
        and requests[item.request_ref].location_ref == f"die:{die}"
    ) for die in range(die_count)]
    external_allocations = sorted((
        {"address": item.address, "size_bytes": requests[item.request_ref].size_bytes,
         "request_ref": item.request_ref, "allocation_ref": item.id}
        for item in offload.memory_plan.allocations
        if requests[item.request_ref].tier is MemoryTier.EXTERNAL
    ), key=lambda item: (item["address"], item["allocation_ref"]))
    return {
        "mesh": f"{rows}x{columns}", "model_sha256": canonical_digest(request.model),
        "resident_rejection": resident_rejection,
        "offload_manifest_id": offload.id,
        "useful_graph_digest": _useful_graph_digest(offload),
        "external_allocations": external_allocations,
        "hbm_workspace_end_bytes_by_die": hbm_workspace_end,
        "source_segment_receipts": segment_receipts,
        "expected_d2d_links": [
            {"source_die": source, "destination_die": destination,
             "direction": direction, "request_hops": counts[0],
             "packet_hops": counts[1]}
            for (source, destination, direction), counts in sorted(links.items())
        ],
        "first_existing_pager_blocker": first_blocker,
        "status": "source_compile_finalizer_program_io_only",
    }


def run(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    if not Path(sys.modules[_request.__module__].__file__).resolve().is_relative_to(source_root):
        raise ValueError("MoE production source not imported from frozen source root")
    finalizer = args.finalizer.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("EP6 gap probe needs a new output root")
    if not finalizer.is_file():
        raise ValueError("production finalizer missing")
    relinker = _ep4_relinker()
    binding = {
        "source_root": str(source_root),
        "driver_sha256": _sha(Path(__file__).resolve()),
        "ep4_relinker_sha256": _sha(_RELINK),
        "finalizer_sha256": _sha(finalizer),
    }
    output.mkdir(parents=True)
    cases = [_case(rows, columns, output / f"{rows}x{columns}", finalizer, relinker)
             for rows, columns in ((2, 3), (3, 2))]
    if cases[0]["model_sha256"] != cases[1]["model_sha256"]:
        raise ValueError("2x3 and 3x2 did not use the same six-expert model")
    imported = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if type(name) is not str or not name.endswith(".py"):
            continue
        path = Path(name).resolve()
        if path.is_relative_to(source_root):
            imported[str(path.relative_to(source_root))] = _sha(path)
    binding["frozen_imported_source_sha256"] = dict(sorted(imported.items()))
    receipt = {
        "schema_version": "moe-ep6-low-hbm-source-blocker-v1",
        "status": "source_compile_finalizer_program_io_only",
        "binding": binding, "cases": cases,
    }
    (output / "ep6_gap_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print("EP6 2x3/3x2 source/finalizer/ProgramIO PASS; native offload BLOCKED", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--finalizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
