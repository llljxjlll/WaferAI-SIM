"""Run Dense inference after a typed external-to-HBM parameter bring-in."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_compile_sequence import (
    compile_dense_e2e_sequence_runtime_profiles,
)
from llm.frontend.wafer_frontend.passes.external_dma_action_graph import (
    build_external_dma_action_graph,
)
from llm.frontend.wafer_frontend.passes.external_dma_program import (
    finalize_external_dma_program,
)
from llm.frontend.wafer_frontend.passes.offload import plan_offload_blocking
from llm.frontend.wafer_frontend.passes.program_io import build_timing_program_io
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.artifact_manifest import RegionManifest
from llm.frontend.wafer_frontend.schema.external_dma_action_graph import (
    ExternalDmaRuntimeBinding,
)
from llm.frontend.wafer_frontend.schema.external_dma_program import (
    ExternalDmaBackendBinding,
    ExternalDmaProbe,
    ExternalDmaSeed,
)
from llm.frontend.wafer_frontend.schema.external_memory import (
    ExternalMemoryConnection,
    ExternalMemoryFabric,
    ExternalMemoryLink,
)
from llm.frontend.wafer_frontend.schema.ir2 import StateUseAccess
from llm.frontend.wafer_frontend.schema.memory_plan import (
    MemoryObjectKind,
    MemoryTier,
    MemoryTierCapacity,
)
from llm.frontend.wafer_frontend.schema.offload import (
    OffloadChunk,
    OffloadEventKind,
    OffloadStateMapping,
    OffloadTraceEvent,
)
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
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
from .run_dense_sequence_runtime_canary import _run


_ROOT = Path(__file__).resolve().parents[4]


def _useful_graph_projection_digest(graph) -> str:
    """Remove request identity while retaining every operation/value edge."""

    states = {item.id: item for item in graph.state_versions}
    values = {item.id: item for item in graph.tensor_values}
    groups = {item.id: item for item in graph.placement.groups}
    operation_index = {
        item.id: item.sequence_index for item in graph.operations
    }

    def state(ref: str):
        item = states[ref]
        return (
            item.logical_name,
            item.kind.value,
            item.version,
            item.layer,
            item.expert,
        )

    def value(ref: str):
        item = values[ref]
        return (
            item.logical_name,
            item.shape,
            item.dtype.value,
            item.size_bytes,
            state(item.state_ref),
            item.logical_rank,
            item.tp_shard,
        )

    operations = tuple(
        {
            "sequence": item.sequence_index,
            "kind": item.kind.value,
            "phase": item.phase,
            "step": item.step,
            "layer": item.layer,
            "expert": item.expert,
            "parameter": item.parameter_ref,
            "reads": tuple(state(ref) for ref in item.reads),
            "writes": tuple(state(ref) for ref in item.writes),
            "inputs": tuple(value(ref) for ref in item.input_value_refs),
            "outputs": tuple(value(ref) for ref in item.output_value_refs),
            "groups": tuple(
                (groups[ref].kind.value, groups[ref].index, groups[ref].ranks)
                for ref in item.group_refs
            ),
            "reduce": None if item.reduce_op is None else item.reduce_op.value,
            "normalization": item.normalization_denominator,
            "deps": tuple(operation_index[ref] for ref in item.deps),
        }
        for item in graph.operations
    )
    routes = tuple(
        {
            "phase": item.phase,
            "step": item.step,
            "layer": item.layer,
            "token_count": item.token_count,
            "expert_by_token": item.expert_by_token,
            "expert_token_counts": item.expert_token_counts,
            "route_values": tuple(value(ref) for ref in item.route_value_refs),
            "dispatch_values": tuple(
                value(ref) for ref in item.dispatch_value_refs
            ),
            "expert_outputs": tuple(
                value(ref) for ref in item.expert_output_value_refs
            ),
            "combine_values": tuple(
                value(ref) for ref in item.combine_value_refs
            ),
        }
        for item in graph.route_traces
    )
    return canonical_digest({"operations": operations, "routes": routes})


def _build_offload_case(base_manifest):
    parameter_bytes = sum(
        item.size_bytes
        for item in base_manifest.state_inventory
        if item.object_kind is MemoryObjectKind.PARAMETER
    )
    kv_bytes = sum(
        item.size_bytes
        for item in base_manifest.state_inventory
        if item.object_kind is MemoryObjectKind.KV
    )
    hbm_bytes = parameter_bytes + kv_bytes
    resident_peak_bytes = max(
        item.peak_bytes for item in base_manifest.memory_plan.peaks
    )
    if resident_peak_bytes <= hbm_bytes:
        raise RuntimeError("resident peak must exceed the offload HBM window")
    hbm = MemoryTierCapacity.create(
        tier=MemoryTier.HBM,
        location_ref="die:0",
        base_address=0,
        capacity_bytes=hbm_bytes,
        alignment_bytes=64,
    )
    try:
        materialize_workload_preflight(
            base_manifest.request,
            base_manifest.capability,
            capacities=(hbm,),
        )
    except SchemaError as error:
        resident_oom = str(error)
    else:
        raise RuntimeError(
            "resident-HBM baseline unexpectedly fits the bounded hardware"
        )

    external = MemoryTierCapacity.create(
        tier=MemoryTier.EXTERNAL,
        location_ref="host:0",
        base_address=0,
        capacity_bytes=parameter_bytes,
        alignment_bytes=64,
    )
    source = base_manifest.request
    request = WorkloadRunRequest.create(
        family=source.family,
        model=source.model,
        steps=source.steps,
        mesh=source.mesh,
        parallel=source.parallel,
        memory=WorkloadMemoryPolicy(
            mode=WorkloadMemoryMode.EXTERNAL_OFFLOAD,
            external_tier_ref="host:0",
        ),
        optimizer=source.optimizer,
        execution=source.execution,
    )
    manifest = materialize_workload_preflight(
        request,
        base_manifest.capability,
        capacities=(external, hbm),
    )
    base_useful_graph_digest = _useful_graph_projection_digest(
        base_manifest.logical_graph
    )
    offload_useful_graph_digest = _useful_graph_projection_digest(
        manifest.logical_graph
    )
    if base_useful_graph_digest != offload_useful_graph_digest:
        raise RuntimeError("offload memory policy changed the useful logical graph")
    requests = {item.id: item for item in manifest.memory_plan.requests}
    versions = {item.id: item for item in manifest.memory_plan.state_versions}
    parameter_allocations = tuple(
        item
        for item in manifest.memory_plan.allocations
        if requests[item.request_ref].object_kind is MemoryObjectKind.PARAMETER
    )
    if len(parameter_allocations) != 1:
        raise RuntimeError("offload canary requires one aggregate parameter shard")
    source_allocation = parameter_allocations[0]
    source_request = requests[source_allocation.request_ref]
    if (
        source_request.tier is not MemoryTier.EXTERNAL
        or source_request.size_bytes != parameter_bytes
        or source_allocation.address != 0
    ):
        raise RuntimeError("parameter source allocation is not exact external backing")
    version = versions[source_request.state_version_ref]

    link = ExternalMemoryLink.create(
        external_capacity_ref=external.id,
        ingress_die_id=0,
        bytes_per_cycle=256,
        latency_cycles=2,
        queue_depth=2,
        max_outstanding=2,
    )
    connection = ExternalMemoryConnection.create(
        link_ref=link.id,
        hbm_capacity_ref=hbm.id,
        target_die_id=0,
        route_die_ids=(0,),
        route_latency_cycles=0,
        route_bytes_per_cycle=None,
    )
    fabric = ExternalMemoryFabric.create(
        external_capacities=(external,),
        hbm_capacities=(hbm,),
        links=(link,),
        connections=(connection,),
    )
    chunk = OffloadChunk.create(
        initial_version=version,
        object_kind=MemoryObjectKind.PARAMETER,
        size_bytes=parameter_bytes,
        alignment_bytes=source_request.alignment_bytes,
        external_capacity_ref=external.id,
        external_address=source_allocation.address,
        connection_ref=connection.id,
    )
    mapping = OffloadStateMapping.create(
        chunk_ref=chunk.id,
        source_state_version_ref=version.id,
        source_allocation_ref=source_allocation.id,
    )
    plan = plan_offload_blocking(
        request_digest=request.digest,
        logical_graph_digest=manifest.logical_graph_digest,
        source_memory_plan_digest=canonical_digest(manifest.memory_plan),
        source_memory_plan=manifest.memory_plan,
        state_mappings=(mapping,),
        fabric=fabric,
        chunks=(chunk,),
        events=(
            OffloadTraceEvent.create(
                ordinal=0,
                kind=OffloadEventKind.READ,
                chunk_ref=chunk.id,
            ),
        ),
    )
    payload = bytes((index % 251) + 1 for index in range(parameter_bytes))
    program = finalize_external_dma_program(
        plan=plan,
        case_digest=canonical_digest(request.case_id),
        backend_bindings=(
            ExternalDmaBackendBinding.create(
                hbm_capacity_ref=hbm.id,
                owner_die_id=0,
                stack_id=0,
                channel_id=0,
            ),
        ),
        external_seeds=(
            ExternalDmaSeed.create(
                external_capacity_ref=external.id,
                address=source_allocation.address,
                payload=payload,
            ),
        ),
        external_probes=(
            ExternalDmaProbe.create(
                external_capacity_ref=external.id,
                address=source_allocation.address,
                expected_payload=payload,
            ),
        ),
    )
    action_graph = build_external_dma_action_graph(
        manifest=manifest,
        plan=plan,
        program=program,
    )
    binding = ExternalDmaRuntimeBinding.create(
        action_graph_digest=action_graph.digest,
        program_relative_path="artifacts/external_dma_program.json",
        case_digest=program.case_digest,
        request_digest=program.request_digest,
        logical_graph_digest=program.logical_graph_digest,
        source_memory_plan_digest=program.source_memory_plan_digest,
        blocking_offload_plan_digest=program.blocking_offload_plan_digest,
    )
    return (
        manifest,
        plan,
        program,
        action_graph,
        binding,
        hbm_bytes,
        parameter_bytes,
        resident_oom,
        base_useful_graph_digest,
        resident_peak_bytes,
    )


def run(args: argparse.Namespace) -> None:
    base_manifest, template, physical_fabric = _one_die_case()
    (
        manifest,
        offload_plan,
        dma_program,
        action_graph,
        runtime_binding,
        hbm_bytes,
        parameter_bytes,
        resident_oom,
        useful_graph_digest,
        resident_peak_bytes,
    ) = _build_offload_case(base_manifest)
    sequence, linked_profiles = compile_dense_e2e_sequence_runtime_profiles(
        base_manifest,
        template,
        physical_fabric,
        hbm_address_spaces=valid_hbm_address_spaces(physical_fabric),
    )
    sequence.validate()
    state_abis = tuple(
        abi
        for profile in linked_profiles
        for fragment in profile.manifest.fragments
        for abi in (
            fragment.fragment.state_abi
            if isinstance(fragment, RegionManifest)
            else fragment.state_abi
        )
    )
    state_end = max(abi.address + abi.size_bytes for abi in state_abis)
    if state_end != hbm_bytes:
        raise RuntimeError(
            f"linked HBM boundary {state_end} differs from capacity {hbm_bytes}"
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (output / "workload_manifest.json").write_text(
        canonical_json(manifest), encoding="utf-8"
    )
    (output / "blocking_offload_plan.json").write_text(
        canonical_json(offload_plan), encoding="utf-8"
    )
    (artifacts / "external_dma_action_graph.json").write_text(
        canonical_json(action_graph), encoding="utf-8"
    )
    (artifacts / "external_dma_program.json").write_text(
        canonical_json(dma_program), encoding="utf-8"
    )
    binding_path = output / "external_dma_runtime_binding.json"
    binding_path.write_text(canonical_json(runtime_binding), encoding="utf-8")

    manifests: list[Path] = []
    programs: list[Path] = []
    sidecars: list[Path] = []
    for index, segment in enumerate(sequence.segments):
        manifest_path = output / f"segment_{index}.linked.json"
        artifact_path = output / f"segment_{index}.npup"
        report_path = output / f"segment_{index}.finalizer.json"
        manifest_path.write_text(
            canonical_json(segment.linked_manifest), encoding="utf-8"
        )
        _run(
            (
                str(args.finalizer.resolve()),
                "--input",
                str(manifest_path),
                "--output",
                str(artifact_path),
                "--report",
                str(report_path),
            ),
            cwd=output,
            timeout=120,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if (
            report.get("artifact_sha256") != artifact_digest
            or report.get("linked_manifest_id") != segment.linked_manifest.id
            or report.get("linked_manifest_digest")
            != canonical_digest(segment.linked_manifest)
        ):
            raise RuntimeError(f"segment {index} finalizer closure failed")
        manifests.append(manifest_path)
        programs.append(artifact_path)

        abi_by_binding = {
            abi.hbm_binding_ref: abi
            for fragment in linked_profiles[index].manifest.fragments
            for abi in (
                fragment.fragment.state_abi
                if isinstance(fragment, RegionManifest)
                else fragment.state_abi
            )
        }
        first_access: dict[str, StateUseAccess] = {}
        for action in linked_profiles[index].lowering_context.global_dag.actions:
            for use in action.state_uses:
                first_access.setdefault(use.hbm_binding_ref, use.access)
        state_seeds = {
            abi_by_binding[binding].state_ref: bytes(
                abi_by_binding[binding].size_bytes
            )
            for binding, access in first_access.items()
            if access is StateUseAccess.READ
        }
        contract = build_timing_program_io(
            linked_profiles[index],
            artifact_digest,
            state_seed_overrides=state_seeds,
        )
        contract.validate_against(segment.linked_manifest)
        sidecar_path = output / f"segment_{index}.program_io.json"
        sidecar_path.write_text(canonical_json(contract), encoding="utf-8")
        sidecars.append(sidecar_path)

    hardware = json.loads(specialize_p5_large_release_hardware(1, 1))
    hardware["memory"]["sram_size"] = 65536
    hardware["memory"]["sram"]["capacity_bytes"] = 65536
    hardware["memory"]["sram"]["regions"][0]["name"] = "sram"
    hardware["memory"]["sram"]["regions"][0]["size_bytes"] = 65536
    hardware["memory_system"]["hbm_stacks"][0]["capacity_bytes"] = hbm_bytes
    hardware["memory_system"]["address_policy"]["home_ranges"][0][
        "size_bytes"
    ] = hbm_bytes
    hardware["memory_system"]["address_policy"][
        "stack_interleave_bytes"
    ] = hbm_bytes
    hardware_path = output / "hardware.json"
    hardware_path.write_text(
        json.dumps(hardware, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    mapping_path = output / "mapping.spec"
    mapping_path.write_text("0:0\n", encoding="utf-8")

    runtime_output = _run(
        (
            str(args.npusim.resolve()),
            "--program-sequence",
            ",".join(str(path) for path in programs),
            "--linked-manifest-sequence",
            ",".join(str(path) for path in manifests),
            "--program-io-sequence",
            ",".join(str(path) for path in sidecars),
            "--external-dma-binding",
            str(binding_path),
            "--hardware-config",
            str(hardware_path),
            "--simulation-config",
            str(args.simulation.resolve()),
            "--mapping-config",
            str(mapping_path),
            "--trace-window",
            "1000000",
        ),
        cwd=output,
        timeout=args.timeout,
    )
    (output / "npusim.stdout.txt").write_text(runtime_output, encoding="utf-8")

    ready = re.findall(
        r"\[EXTERNAL_DMA_READY\].*completed=(\d+).*"
        r"external_read_bytes=(\d+).*hbm_write_bytes=(\d+).*pending=0",
        runtime_output,
    )
    drain = re.findall(
        r"\[EXTERNAL_DMA_DRAIN\] probes=(\d+).*"
        r"external_read_bytes=(\d+).*external_write_bytes=(\d+).*"
        r"hbm_read_bytes=(\d+).*hbm_write_bytes=(\d+).*pending=0",
        runtime_output,
    )
    compute = re.findall(
        r"\[DENSE_SEQUENCE_COMPUTE\] index=(\d+) records=(\d+) "
        r"lsu_loads=(\d+) status=done",
        runtime_output,
    )
    if ready != [("1", str(parameter_bytes), str(parameter_bytes))]:
        raise RuntimeError(f"external DMA ready closure failed: {ready}")
    if drain != [("1", str(parameter_bytes), "0", "0", str(parameter_bytes))]:
        raise RuntimeError(f"external DMA drain closure failed: {drain}")
    if (
        len(compute) != 3
        or [item[0] for item in compute] != ["0", "1", "2"]
        or any(int(item[1]) <= 0 or int(item[2]) <= 0 for item in compute)
    ):
        raise RuntimeError(f"Dense compute/load closure failed: {compute}")
    if runtime_output.find("[EXTERNAL_DMA_READY]") > runtime_output.find(
        "[DENSE_SEQUENCE_COMPUTE] index=0"
    ):
        raise RuntimeError("Dense compute completed before external DMA readiness")
    if runtime_output.count("[SIM_RESULT]") != 1:
        raise RuntimeError("runtime did not emit exactly one SIM_RESULT")

    evidence = {
        "schema_version": "npusim.dense_external_offload_canary/v1alpha1",
        "request_digest": manifest.request_digest,
        "resident_logical_graph_digest": base_manifest.logical_graph_digest,
        "offload_logical_graph_digest": manifest.logical_graph_digest,
        "useful_graph_projection_digest": useful_graph_digest,
        "resident_hbm_error": resident_oom,
        "hbm_capacity_bytes": hbm_bytes,
        "resident_peak_bytes": resident_peak_bytes,
        "parameter_bytes": parameter_bytes,
        "external_dma_action_graph_digest": action_graph.digest,
        "segments": len(sequence.segments),
        "compute_records": [int(item[1]) for item in compute],
        "lsu_load_records": [int(item[2]) for item in compute],
        "sim_result_count": 1,
    }
    (output / "evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "Dense external offload runtime canary PASS "
        f"resident_peak={resident_peak_bytes} hbm={hbm_bytes} "
        f"parameter={parameter_bytes} "
        f"action_graph={action_graph.digest}"
    )


def _parse_args() -> argparse.Namespace:
    build = _ROOT / "build-debug-final"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=build / "dense-external-offload-runtime-canary",
    )
    parser.add_argument(
        "--finalizer", type=Path, default=build / "npusim_program_finalizer"
    )
    parser.add_argument("--npusim", type=Path, default=build / "npusim")
    parser.add_argument(
        "--simulation",
        type=Path,
        default=_ROOT / "llm/test/program/p5_behavioral_simulation.json",
    )
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    for name in ("finalizer", "npusim", "simulation"):
        if not getattr(args, name).is_file():
            parser.error(f"--{name} must name an existing file")
    return args


if __name__ == "__main__":
    run(_parse_args())
