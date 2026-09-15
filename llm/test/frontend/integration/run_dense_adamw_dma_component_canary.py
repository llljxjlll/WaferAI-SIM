"""Exercise true 5-group Dense AdamW DMA traffic, not an end-to-end offload run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.frontend.wafer_frontend.passes.dense_adamw_compile_sequence import (
    compile_dense_adamw_step,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_offload_plan import (
    plan_dense_adamw_source_offload,
)
from llm.frontend.wafer_frontend.passes.dense_adamw_offload_preflight import (
    preflight_dense_adamw_offload_window,
)
from llm.frontend.wafer_frontend.passes.external_dma_action_graph import (
    build_external_dma_action_graph,
)
from llm.frontend.wafer_frontend.passes.external_dma_program import (
    finalize_external_dma_program,
)
from llm.frontend.wafer_frontend.passes.program_io import (
    build_deterministic_timing_state_overrides,
)
from llm.frontend.wafer_frontend.schema.external_dma_program import (
    ExternalDmaBackendBinding, ExternalDmaProbe, ExternalDmaSeed,
)
from llm.frontend.wafer_frontend.schema.persistent_state import StateKind
from llm.frontend.wafer_frontend.schema.serde import canonical_digest, canonical_json
from llm.test.frontend.unit.test_dense_adamw_compile_sequence import _adamw_case

from .run_dense_training_sequence_runtime_canary import _run


_ROOT = Path(__file__).resolve().parents[4]
_GROUP_KINDS = {
    "parameter": StateKind.TRAINABLE_PARAMETER,
    "master": StateKind.OPTIMIZER_MASTER,
    "m": StateKind.OPTIMIZER_MOMENT1,
    "v": StateKind.OPTIMIZER_MOMENT2,
    "step": StateKind.OPTIMIZER_STEP,
}


def run(args: argparse.Namespace) -> dict[str, object]:
    source, physical = _adamw_case()
    window = preflight_dense_adamw_offload_window(source)
    if window.startup_window_sufficient:
        raise RuntimeError("low-HBM full-model startup phase unexpectedly fits")
    offload = window.materialization
    plan = plan_dense_adamw_source_offload(offload)
    linked = compile_dense_adamw_step(source, physical, 0)
    state_abi = linked.manifest.fragments[0].state_abi
    seeds, _expected = build_deterministic_timing_state_overrides(linked)
    grouped = {}
    for name, kind in _GROUP_KINDS.items():
        states = tuple(sorted(
            (abi for abi in state_abi if abi.kind is kind),
            key=lambda item: item.address,
        ))
        if len(states) != (15 if name == "parameter" else 17):
            raise RuntimeError(f"physical {name} ABI state count drifted")
        grouped[name] = b"".join(
            bytes(4) if kind is StateKind.OPTIMIZER_STEP
            else seeds[abi.state_ref]
            for abi in states
        )
    if sum(map(len, grouped.values())) != 32100 or len(grouped["step"]) != 68:
        raise RuntimeError("five DMA seed groups must close all 83 true ABI bytes")
    source_requests = {item.id: item for item in offload.memory_plan.requests}
    source_versions = {item.id: item for item in offload.memory_plan.state_versions}
    inventories = {item.id: item for item in offload.state_inventory}
    seeds_dma = []
    probes_dma = []
    for allocation in offload.memory_plan.allocations:
        request = source_requests[allocation.request_ref]
        if request.tier.value != "external":
            continue
        version = source_versions[request.state_version_ref]
        item = inventories[version.state_ref]
        if item.logical_name.startswith("parameter."):
            group = "parameter"
        elif item.logical_name.startswith("optimizer.adamw."):
            group = item.logical_name.split(".")[2]
        else:
            continue
        payload = grouped[group]
        if len(payload) != request.size_bytes:
            raise RuntimeError("P3 external aggregate differs from true ABI payload")
        external_capacity = plan.fabric.external_capacities[0]
        seeds_dma.append(ExternalDmaSeed.create(
            external_capacity_ref=external_capacity.id,
            address=allocation.address, payload=payload,
        ))
        probes_dma.append(ExternalDmaProbe.create(
            external_capacity_ref=external_capacity.id,
            address=allocation.address, expected_payload=payload,
        ))
    if len(seeds_dma) != 5 or len(probes_dma) != 5:
        raise RuntimeError("83 real ABI bytes must have five exact external seed/probe groups")
    hbm = plan.fabric.hbm_capacities[0]
    program = finalize_external_dma_program(
        plan=plan, case_digest=canonical_digest(offload.request.case_id),
        backend_bindings=(ExternalDmaBackendBinding.create(
            hbm_capacity_ref=hbm.id, owner_die_id=0,
            stack_id=0, channel_id=0,
        ),),
        external_seeds=tuple(seeds_dma), external_probes=tuple(probes_dma),
    )
    if len(program.descriptors) != 10 or sum(
        descriptor.size_bytes for descriptor in program.descriptors
    ) != 64200:
        raise RuntimeError("5 restore + 5 dirty writeback source DMA missing")
    action_graph = build_external_dma_action_graph(
        manifest=offload, plan=plan, program=program,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    program_path = output / "external_dma_program.json"
    report_path = output / "external_dma_component_report.json"
    program_path.write_text(canonical_json(program), encoding="utf-8")
    _run((
        str(args.runner.resolve()), str(program_path), str(report_path),
        action_graph.digest, program.case_digest,
        program.request_digest, program.logical_graph_digest,
        program.source_memory_plan_digest,
        program.blocking_offload_plan_digest,
    ), cwd=output, timeout=args.timeout)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not report["completed"] or report["pending_requests"] != 0
        or report["completed_requests"] != 10
        or report["external_read_bytes"] != 32100
        or report["external_write_bytes"] != 32100
        or report["hbm_read_bytes"] != 32100
        or report["hbm_write_bytes"] != 32100
        or not report["all_probes_matched"]
    ):
        raise RuntimeError(f"true C++ external DMA five-group traffic failed: {report}")
    report["full_model_bounded_offload_runtime"] = False
    report["startup_combined_hbm_bytes"] = window.startup_combined_peak_bytes
    report["bounded_hbm_capacity_bytes"] = window.resident_hbm_capacity_bytes
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Dense AdamW DMA component PASS restore=32100B dirty=32100B "
          "probes=5 pending=0 bounded-full-model=BLOCKED")
    return report


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path,
        default=_ROOT / "build-debug-final/npusim_external_dma_program_runner")
    parser.add_argument("--output", type=Path,
        default=_ROOT / "build-debug-final/dense-adamw-dma-component")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if not args.runner.is_file() or args.timeout <= 0:
        parser.error("--runner must exist and --timeout must be positive")
    return args


if __name__ == "__main__":
    run(_args())
