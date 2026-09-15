"""True low-HBM 1x1 Dense Prefill+2Decode resident reject/paged DMA pair."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    compile_dense_e2e_sequence_runtime_profiles,
)
from llm.frontend.wafer_frontend.passes.dense_inference_paged_compile_sequence import (
    relink_dense_inference_paged_segment,
)
from llm.frontend.wafer_frontend.passes.dense_inference_paged_program_io import (
    retarget_dense_inference_paged_sram_program_io,
)
from llm.frontend.wafer_frontend.passes.dense_inference_paged_runtime import (
    build_dense_inference_paged_runtime,
)
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.external_memory import (
    ExternalMemoryConnection,
    ExternalMemoryFabric,
    ExternalMemoryLink,
)
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryTier,
    MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.serde import (
    canonical_digest,
    canonical_json,
)
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadMemoryMode,
    WorkloadMemoryPolicy,
    WorkloadRunRequest,
)
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_dense_compile_sequence import _one_die_case

from .flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)


_ROOT = Path(__file__).resolve().parents[4]


def _source_tool_snapshot(args: argparse.Namespace) -> dict[str, object]:
    imported: dict[str, str] = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py" or not path.is_relative_to(_ROOT):
            continue
        imported[str(path.relative_to(_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    imported[str(Path(__file__).resolve().relative_to(_ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes(),
    ).hexdigest()
    tools = {
        name: hashlib.sha256(getattr(args, name).resolve().read_bytes()).hexdigest()
        for name in ("npusim", "finalizer", "resolver", "simulation")
    }
    return {"imported_python_sha256": dict(sorted(imported.items())),
            "tool_sha256": dict(sorted(tools.items()))}


def _run(command: tuple[str, ...], *, cwd: Path, timeout: int) -> str:
    finished = subprocess.run(
        command, cwd=cwd, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout,
    )
    if finished.returncode != 0:
        raise RuntimeError(
            f"returncode={finished.returncode}: {' '.join(command)}\n"
            f"{finished.stdout}"
        )
    return finished.stdout


def _useful_graph_digest(manifest) -> str:
    states = {item.id: item for item in manifest.logical_graph.state_versions}
    values = {item.id: item for item in manifest.logical_graph.tensor_values}
    order = {
        item.id: item.sequence_index
        for item in manifest.logical_graph.operations
    }

    def state(ref: str):
        item = states[ref]
        return (item.logical_name, item.kind.value, item.version,
                item.layer, item.expert)

    def value(ref: str):
        item = values[ref]
        return (item.logical_name, item.shape, item.dtype.value,
                item.size_bytes, state(item.state_ref),
                item.logical_rank, item.tp_shard)

    return canonical_digest(tuple(
        (item.sequence_index, item.kind.value, item.phase, item.step,
         item.layer, item.expert, item.parameter_ref,
         tuple(state(ref) for ref in item.reads),
         tuple(state(ref) for ref in item.writes),
         tuple(value(ref) for ref in item.input_value_refs),
         tuple(value(ref) for ref in item.output_value_refs),
         tuple(order[ref] for ref in item.deps))
        for item in manifest.logical_graph.operations
    ))


def _observe(stdout: str, sidecar: dict[str, object]) -> dict[str, object]:
    events = re.findall(
        r"\[DENSE_INFERENCE_PAGED_DMA_EVENT\] index=(\d+) "
        r"segment=(\d+) linked_record=(\d+) kind=(\w+) "
        r"state_ref=(\S+) lsu_bytes=(\d+) dma_bytes=(\d+) "
        r"issue_cycle=(\d+) completed_at_ticks=(\d+) "
        r"lsu_dependency_complete=(\d+) pass=(\d+)", stdout,
    )
    signed = sidecar["events"]
    if len(events) != 65 or any(
        int(actual[0]) != index
        or int(actual[1]) != item["segment_index"]
        or int(actual[2]) != item["linked_record_index"]
        or actual[3] != item["kind"]
        or actual[4] != item["state_ref"]
        or int(actual[5]) != item["lsu_size_bytes"]
        or int(actual[6]) != item["dma_size_bytes"]
        or actual[9:] != ("1", "1")
        for index, (actual, item) in enumerate(zip(events, signed))
    ):
        raise RuntimeError("65 actual DMA completions drifted from 65 signed linked LSU gates")
    if any(
        int(actual[8]) <= 0 or
        (index and int(actual[8]) <= int(events[index - 1][8]))
        for index, actual in enumerate(events)
    ):
        raise RuntimeError("shared external link did not advance in real service time")
    per_segment = tuple(
        (
            sum(int(item[6]) for item in events
                if item[1] == str(segment) and item[3] != "kv_writeback_after_store"),
            sum(int(item[6]) for item in events
                if item[1] == str(segment) and item[3] == "kv_writeback_after_store"),
        ) for segment in range(3)
    )
    if per_segment != ((53568, 512), (54208, 640), (54336, 768)):
        raise RuntimeError(f"actual parameter/full-page KV traffic changed: {per_segment}")
    kv = re.findall(
        r"\[DENSE_INFERENCE_PAGED_KV\] version=(\d+) bytes=(\d+) "
        r"digest=([0-9a-f]{64}) authority=external functional=(\d+) pass=(\d+)",
        stdout,
    )
    if (
        len(kv) != 4
        or tuple(item[:2] for item in kv) != (
            ("0", "0"), ("1", "512"), ("2", "640"), ("3", "768"),
        )
        or any(item[3:] != ("0", "1") for item in kv)
    ):
        raise RuntimeError("external KV lifecycle did not advance version0→1→2→3")
    io = re.findall(
        r"\[DENSE_INFERENCE_PAGED_EXTERNAL_PROGRAM_IO\] index=(\d+) "
        r"kv_probes=(\d+) kv_bytes=(\d+) pending=(\d+) "
        r"functional=(\d+) pass=(\d+)", stdout,
    )
    if io != [
        ("0", "4", "512", "0", "0", "1"),
        ("1", "4", "640", "0", "0", "1"),
        ("2", "4", "768", "0", "0", "1"),
    ]:
        raise RuntimeError(f"actual SRAM ProgramIO/external four-page probes failed: {io}")
    drain = re.findall(
        r"\[DENSE_INFERENCE_PAGED_DMA_DRAIN\] events=(\d+) "
        r"kv_probes=(\d+) submitted=(\d+) completed=(\d+) "
        r"external_read_bytes=(\d+) external_write_bytes=(\d+) "
        r"hbm_read_bytes=(\d+) hbm_write_bytes=(\d+) "
        r"pending=(\d+) dirty=(\d+) pinned=(\d+) pass=(\d+)",
        stdout,
    )
    if drain != [(
        "65", "12", "65", "65", "162112", "1920", "1920",
        "162112", "0", "0", "0", "1",
    )]:
        raise RuntimeError(f"actual shared DMA byte/drain oracle changed: {drain}")
    segments = re.findall(
        r"\[DENSE_SEQUENCE_SEGMENT\] index=(\d+) status=done final=(\d+)",
        stdout,
    )
    one_shot = re.findall(
        r"\[DENSE_SEQUENCE_DRAIN\] segments=(\d+) one_shot=(\d+)",
        stdout,
    )
    makespan = re.findall(r"\[SIM_RESULT\] makespan_cycles=(\d+)", stdout)
    if (
        segments != [("0", "0"), ("1", "0"), ("2", "1")]
        or one_shot != [("3", "1")]
        or len(makespan) != 1
        or stdout.count("[DENSE_INFERENCE_PAGED_BINDING]") != 1
    ):
        raise RuntimeError(
            f"true one-shot three-segment timeline failed: "
            f"segments={segments} one_shot={one_shot} makespan={makespan}"
        )
    return {
        "kv_versions": [0, 1, 2, 3],
        "kv_page_bytes": [0, 512, 640, 768],
        "kv_authority_digests": [item[2] for item in kv],
        "external_dma_events": 65,
        "external_kv_probes": 12,
        "first_dma_issue_cycle": int(events[0][7]),
        "last_dma_completed_at_ticks": int(events[-1][8]),
        "makespan_cycles": int(makespan[0]),
        "functional": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    source_tool_at_entry = _source_tool_snapshot(args)
    resident, template, fabric = _one_die_case()
    model_digest = canonical_digest(resident.request.model)
    hbm = MemoryTierCapacity.create(
        tier=MemoryTier.HBM, location_ref="die:0",
        base_address=0, capacity_bytes=12288, alignment_bytes=64,
    )
    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL, location_ref="host:0",
        base_address=0, capacity_bytes=54400, alignment_bytes=64,
    )
    try:
        materialize_workload_preflight(
            resident.request, resident.capability, capacities=(hbm,),
        )
    except SchemaError as error:
        resident_rejection = error.code
        resident_rejection_detail = str(error)
    else:
        raise RuntimeError("same-model resident-only unexpectedly fits 12,288B HBM")
    if resident_rejection != "memory_capacity_exceeded":
        raise RuntimeError(f"resident-only rejected for wrong reason: {resident_rejection_detail}")
    source = resident.request
    request = WorkloadRunRequest.create(
        family=source.family, model=source.model,
        steps=source.steps, mesh=source.mesh,
        parallel=source.parallel,
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
            external_tier_ref="host:0",
        ),
        optimizer=source.optimizer,
        execution=source.execution,
    )
    offload = materialize_workload_preflight(
        request, resident.capability, capacities=(external, hbm),
    )
    resident_graph = _useful_graph_digest(resident)
    offload_graph = _useful_graph_digest(offload)
    if resident_graph != offload_graph:
        raise RuntimeError("same-model memory policy changed the useful logical graph")
    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id, ingress_die_id=0,
        bytes_per_cycle=256, latency_cycles=2,
        queue_depth=2, max_outstanding=2,
    )
    connection = ExternalMemoryConnection.create(
        link_ref=link.id, hbm_capacity_ref=hbm.id,
        target_die_id=0, route_die_ids=(0,),
        route_latency_cycles=0, route_bytes_per_cycle=None,
    )
    external_fabric = ExternalMemoryFabric.create(
        external_capacities=(external,), hbm_capacities=(hbm,),
        links=(link,), connections=(connection,),
    )
    sequence, linked_profiles = compile_dense_e2e_sequence_runtime_profiles(
        resident, template, fabric,
        hbm_address_spaces=valid_hbm_address_spaces(fabric),
    )
    sequence.validate()
    source_manifests = tuple(
        segment.linked_manifest for segment in sequence.segments
    )
    paged_manifests = tuple(
        relink_dense_inference_paged_segment(item, index)
        for index, item in enumerate(source_manifests)
    )
    sidecar = build_dense_inference_paged_runtime(
        resident=resident, offload=offload,
        source_manifests=source_manifests,
        paged_manifests=paged_manifests,
        fabric=external_fabric,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sidecar_path = output / "dense_inference_paged_runtime.json"
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    paged_manifest_paths: list[Path] = []
    paged_artifact_paths: list[Path] = []
    paged_io_paths: list[Path] = []
    operator_counts: Counter[str] = Counter()
    for index, (original, paged, profile) in enumerate(zip(
        source_manifests, paged_manifests, linked_profiles,
    )):
        for fragment in original.fragments:
            for stream in fragment.core_streams:
                operator_counts.update(record.opcode.name for record in stream.records)
        source_manifest_path = output / f"segment_{index}.source.linked.json"
        source_artifact_path = output / f"segment_{index}.source.npup"
        paged_manifest_path = output / f"segment_{index}.linked.json"
        paged_artifact_path = output / f"segment_{index}.npup"
        source_manifest_path.write_text(canonical_json(original), encoding="utf-8")
        paged_manifest_path.write_text(canonical_json(paged), encoding="utf-8")
        for manifest_path, artifact_path, manifest in (
            (source_manifest_path, source_artifact_path, original),
            (paged_manifest_path, paged_artifact_path, paged),
        ):
            report_path = output / f"{artifact_path.stem}.finalizer.json"
            _run((
                str(args.finalizer.resolve()), "--input", str(manifest_path),
                "--output", str(artifact_path), "--report", str(report_path),
            ), cwd=output, timeout=120)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report["artifact_sha256"] !=
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                or report["linked_manifest_id"] != manifest.id
                or report["linked_manifest_digest"] != canonical_digest(manifest)
            ):
                raise RuntimeError("actual source/paged production finalizer closure drifted")
        first_access: dict[str, StateUseAccess] = {}
        abi_by_binding = {
            abi.hbm_binding_ref: abi
            for fragment in original.fragments
            for abi in fragment.state_abi
        }
        for action in profile.lowering_context.global_dag.actions:
            for use in action.state_uses:
                first_access.setdefault(use.hbm_binding_ref, use.access)
        state_seeds = {
            abi_by_binding[binding].state_ref:
                bytes(abi_by_binding[binding].size_bytes)
            for binding, access in first_access.items()
            if access is StateUseAccess.READ
        }
        production_io = build_timing_program_io(
            profile,
            hashlib.sha256(source_artifact_path.read_bytes()).hexdigest(),
            state_seed_overrides=state_seeds,
        )
        production_io.validate_against(original)
        paged_io = retarget_dense_inference_paged_sram_program_io(
            production_io, paged,
            hashlib.sha256(paged_artifact_path.read_bytes()).hexdigest(),
        )
        paged_io_path = output / f"segment_{index}.program_io.json"
        paged_io_path.write_text(canonical_json(paged_io), encoding="utf-8")
        resolver_stdout = _run((
            str(args.resolver.resolve()), "--resolve",
            str(paged_manifest_path), str(paged_artifact_path), str(paged_io_path),
        ), cwd=args.resolver.resolve().parent, timeout=120)
        (output / f"segment_{index}.resolver.stdout.txt").write_text(
            resolver_stdout, encoding="utf-8",
        )
        if (f"initializations={len(paged_io.initializations)}" not in resolver_stdout
                or f"probes={len(paged_io.output_probes)}" not in resolver_stdout):
            raise RuntimeError("native paged ProgramIO resolver closure failed")
        paged_manifest_paths.append(paged_manifest_path)
        paged_artifact_paths.append(paged_artifact_path)
        paged_io_paths.append(paged_io_path)
    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    core_grid = fabric.dies[0].noc_grid
    if core_grid[0] <= 0 or core_grid[1] <= 0 or len(fabric.dies) != 1:
        raise RuntimeError("single-Die Dense inference core geometry is invalid")
    hardware["x"], hardware["y"] = core_grid
    if hardware["die"] != {"x": 1, "y": 1}:
        raise RuntimeError("native physical Die geometry drifted from frontend fabric")
    hardware["memory"]["sram_size"] = 65536
    hardware["memory"]["sram"]["capacity_bytes"] = 65536
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 65536
    physical = hardware["memory_system"]
    if (
        len(physical["hbm_stacks"]) != 1
        or physical["hbm_stacks"][0]["backend"] != "behavioral"
        or len(physical["address_policy"]["home_ranges"]) != 1
    ):
        raise RuntimeError("same-hardware behavioral die0 backend identity drifted")
    physical["hbm_stacks"][0]["capacity_bytes"] = 12288
    physical["address_policy"]["home_ranges"][0]["size_bytes"] = 12288
    physical["address_policy"]["stack_interleave_bytes"] = 12288
    hardware_path = output / "hardware.json"
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    mapping_path = output / "mapping.spec"
    mapping_path.write_text("0:0\n", encoding="utf-8")
    binary_sha = hashlib.sha256(args.npusim.resolve().read_bytes()).hexdigest()
    finalizer_sha = hashlib.sha256(args.finalizer.resolve().read_bytes()).hexdigest()
    resolver_sha = hashlib.sha256(args.resolver.resolve().read_bytes()).hexdigest()
    hardware_sha = hashlib.sha256(hardware_path.read_bytes()).hexdigest()
    simulation_sha = hashlib.sha256(args.simulation.resolve().read_bytes()).hexdigest()
    observations = []
    for fresh in (0, 1):
        run_folder = output / f"fresh_{fresh}"
        run_folder.mkdir(parents=True, exist_ok=True)
        command = (
            str(args.npusim.resolve()),
            "--program-sequence", ",".join(map(str, paged_artifact_paths)),
            "--linked-manifest-sequence", ",".join(map(str, paged_manifest_paths)),
            "--program-io-sequence", ",".join(map(str, paged_io_paths)),
            "--dense-inference-paged-runtime", str(sidecar_path),
            "--hardware-config", str(hardware_path),
            "--simulation-config", str(args.simulation.resolve()),
            "--mapping-config", str(mapping_path),
            "--trace-window", "1000000",
        )
        try:
            stdout = _run(command, cwd=args.npusim.resolve().parent,
                          timeout=args.timeout)
        except RuntimeError as error:
            (run_folder / "npusim.first-fault.txt").write_text(
                str(error), encoding="utf-8",
            )
            raise
        (run_folder / "npusim.stdout.txt").write_text(stdout, encoding="utf-8")
        if hashlib.sha256(args.npusim.resolve().read_bytes()).hexdigest() != binary_sha:
            raise RuntimeError("two fresh NpuSim processes used different binaries")
        observations.append(_observe(stdout, sidecar))
    if observations[0] != observations[1]:
        raise RuntimeError(f"two independent fresh run digests drifted: {observations}")
    source_tool_at_exit = _source_tool_snapshot(args)
    if any(source_tool_at_exit["imported_python_sha256"].get(name) != digest
           for name, digest in source_tool_at_entry["imported_python_sha256"].items()) or source_tool_at_exit["tool_sha256"] != source_tool_at_entry["tool_sha256"]:
        raise RuntimeError("imported source or native tool bytes drifted during paged inference run")
    source_binding = json.dumps(source_tool_at_exit, sort_keys=True, separators=(",", ":"))
    report = {
        **observations[0],
        "paired_fresh_runs": 2,
        "source_tool_binding_sha256": hashlib.sha256(source_binding.encode()).hexdigest(),
        "source_tool_at_entry": source_tool_at_entry,
        "source_tool_at_exit": source_tool_at_exit,
        "resident_rejection_code": resident_rejection,
        "resident_rejection_detail": resident_rejection_detail,
        "hbm_capacity_bytes": 12288,
        "p3_workspace_end_bytes": 1600,
        "highest_paged_state_end_bytes": 10560,
        "parameter_state_bytes": 53568,
        "kv_final_capacity_bytes": 768,
        "model_digest": model_digest,
        "resident_logical_graph_digest": resident_graph,
        "offload_logical_graph_digest": offload_graph,
        "offload_memory_plan_digest": canonical_digest(offload.memory_plan),
        "paged_runtime_contract_id": sidecar["id"],
        "source_manifest_digests": [canonical_digest(item) for item in source_manifests],
        "paged_manifest_digests": sidecar["linked_manifest_digests"],
        "npusim_sha256": binary_sha,
        "finalizer_sha256": finalizer_sha,
        "resolver_sha256": resolver_sha,
        "frontend_core_grid": core_grid,
        "native_core_grid": (hardware["x"], hardware["y"]),
        "hardware_sha256": hardware_sha,
        "simulation_sha256": simulation_sha,
        "operator_record_coverage": dict(sorted(operator_counts.items())),
    }
    (output / "dense-inference-paged-runtime-evidence.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    print(
        "Dense inference bounded external parameter+KV offload PASS "
        "resident=memory_capacity_exceeded HBM=12288 peak_end=10560 "
        "KV_versions=0,1,2,3 DMA=65 fresh_runs=2 "
        f"makespan={report['makespan_cycles']} functional=0"
    )
    return report


def _args() -> argparse.Namespace:
    build = _ROOT / "build-debug-dense-infer-pager"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=build / "dense-inference-paged-offload-canary")
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument("--finalizer", type=Path,
                        default=_ROOT / "build-debug-final/npusim_program_finalizer")
    parser.add_argument("--resolver", type=Path,
                        default=_ROOT / "build-debug-final/npusim_program_io_selftest")
    parser.add_argument("--simulation", type=Path,
                        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.timeout <= 0 or not all(
        getattr(args, key).is_file()
        for key in ("npusim", "finalizer", "resolver", "simulation")
    ):
        parser.error("npusim/finalizer/simulation must exist and timeout >0")
    return args


if __name__ == "__main__":
    run(_args())
