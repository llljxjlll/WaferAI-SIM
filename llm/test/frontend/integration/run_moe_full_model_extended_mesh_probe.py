"""Fail-closed 11x11 probe for one unchanged 100-expert MoE inference model.

This is an implementation-envelope diagnostic, not a release runner. It keeps
all canonical 1..10 parsers and hardware specialization untouched and never
turns a successful compile into a native execution claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

from llm.frontend.wafer_frontend.errors import UnsupportedFeatureError
from llm.frontend.wafer_frontend.passes.load_fabric import physical_fabric_from_data
from llm.frontend.wafer_frontend.passes.moe_full_model_compile_sequence import (
    compile_moe_full_model_inference_sequence,
)
from llm.frontend.wafer_frontend.passes.moe_compile_sequence import compile_moe_sequence
from llm.frontend.wafer_frontend.passes.workload_materialization import (
    materialize_workload_preflight,
)
from llm.frontend.wafer_frontend.schema.memory_plan import MemoryTier, MemoryTierCapacity
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest
from llm.frontend.wafer_frontend.schema.workload_run import (
    WorkloadFamily,
    WorkloadMeshSpec,
    WorkloadParallelSpec,
    WorkloadRunCapability,
    WorkloadRunRequest,
)
from llm.test.frontend.flexible_mesh_fixtures import minimal_hardware
from llm.test.frontend.unit._fixtures import valid_hbm_address_spaces
from llm.test.frontend.unit.test_moe_compile_sequence import _request
from llm.test.frontend.unit.test_moe_full_model_compile_sequence import _legacy_template
from llm.test.frontend.unit.test_workload_materialization import _capability


_SHAPE = (11, 11)
_MODEL_SHAPE = (10, 10)
_HBM_CAPACITY_BYTES = 1 << 24
_EXPECTED_REASONS = (
    "manifest.full_mesh_required",
    "request.ep_must_cover_mesh",
    "request.experts_must_cover_mesh",
    "request.full_mesh_required",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe() -> dict[str, object]:
    """Invoke the real materializer and compiler with 21 legal idle Die."""
    start = time.monotonic()
    physical = RectMeshSpec(*_SHAPE)
    physical.validate()
    if physical.within_release_envelope:
        raise RuntimeError("extended probe accidentally entered the 1..10 release envelope")
    baseline = _request(WorkloadFamily.MOE_INFERENCE, rows=10, columns=10)
    active = tuple(row * 11 + column for row in range(10) for column in range(10))
    idle = tuple(sorted(set(range(121)) - set(active)))
    request = WorkloadRunRequest.create(
        family=baseline.family,
        model=baseline.model,
        steps=baseline.steps,
        mesh=WorkloadMeshSpec(*_SHAPE),
        parallel=WorkloadParallelSpec(ep=100, active_die_ids=active),
        memory=baseline.memory,
        optimizer=baseline.optimizer,
        execution=baseline.execution,
    )
    capability = WorkloadRunCapability.create(
        max_mesh_rows=11, max_mesh_columns=11, max_mesh_ranks=121,
        families=_capability(supported=True).families,
    )
    capacities = tuple(
        MemoryTierCapacity.create(
            tier=MemoryTier.HBM,
            location_ref=f"die:{die}",
            base_address=0,
            capacity_bytes=_HBM_CAPACITY_BYTES,
            alignment_bytes=16,
        )
        for die in range(121)
    )
    manifest = materialize_workload_preflight(
        request, capability, capacities=capacities,
    )
    fabric = physical_fabric_from_data(
        minimal_hardware(11, 11, sram_bytes=65536)
    )
    spaces = valid_hbm_address_spaces(fabric)
    if (
        tuple(manifest.placement.active_die_ids) != active
        or len(fabric.dies) != 121
        or fabric.die_grid != (11, 11)
        or len(spaces) != 121
        or len(fabric.links) != physical.directed_link_count
    ):
        raise RuntimeError("11x11 materialization or physical Fabric is incomplete")

    from llm.frontend.wafer_frontend.passes import moe_full_model_compile_sequence
    from llm.frontend.wafer_frontend.schema import rect_mesh
    from llm.test.frontend.integration import (
        run_moe_full_model_native_mesh_matrix,
        run_moe_full_model_sequence_runtime_canary,
    )
    sources = {
        "extended_probe": Path(__file__).resolve(),
        "full_model_compiler": Path(moe_full_model_compile_sequence.__file__).resolve(),
        "rect_mesh": Path(rect_mesh.__file__).resolve(),
        "canonical_matrix": Path(run_moe_full_model_native_mesh_matrix.__file__).resolve(),
        "canonical_runner": Path(run_moe_full_model_sequence_runtime_canary.__file__).resolve(),
    }
    result: dict[str, object] = {
        "schema_version": "moe-full-model-extended-mesh-probe-v2",
        "status": "compile_not_verified",
        "physical_mesh": "11x11",
        "physical_die_count": 121,
        "physical_directed_link_budget": physical.directed_link_count,
        "physical_max_hops": physical.max_hop_count,
        "route_traffic_status": "not_compiled",
        "active_die_ids": list(active),
        "idle_die_ids": list(idle),
        "active_die_count": len(active),
        "idle_die_count": len(idle),
        "request_case_id": request.case_id,
        "baseline_model_digest": canonical_digest(baseline.model),
        "extended_model_digest": canonical_digest(request.model),
        "baseline_steps_digest": canonical_digest(baseline.steps),
        "extended_steps_digest": canonical_digest(request.steps),
        "model_expert_count": request.model.num_experts,
        "ep_rank_count": request.parallel.ep,
        "fabric_die_count": len(fabric.dies),
        "fabric_directed_link_count": len(fabric.links),
        "hbm_space_count": len(spaces),
        "source_sha256": {name: _sha(path) for name, path in sources.items()},
        "source_paths": {name: str(path) for name, path in sources.items()},
    }
    try:
        sequence = compile_moe_full_model_inference_sequence(
            manifest, _legacy_template(), fabric, hbm_address_spaces=spaces,
        )
    except UnsupportedFeatureError as error:
        if (
            error.code != "moe_full_model_compile_sequence_unsupported"
            or error.path != "moe_full_model_compile_sequence"
            or any(reason not in error.message for reason in _EXPECTED_REASONS)
        ):
            raise
        result["status"] = "blocked_by_full_mesh_compiler_contract"
        result["error_code"] = error.code
        result["error_path"] = error.path
        result["error_message"] = error.message
        result["blocking_reasons"] = list(_EXPECTED_REASONS)
    else:
        result["status"] = "compiled_without_native_verification"
        result["sequence_digest"] = sequence.digest
    if result["status"] == "blocked_by_full_mesh_compiler_contract":
        try:
            compile_moe_sequence(
                manifest,
                source_rank_policy="rank0_shared_spine",
                runtime_core_ids=tuple(
                    next(core.runtime_core_id for core in die.cores
                         if core.local_core_id == 0)
                    for die in fabric.dies
                ),
            )
        except UnsupportedFeatureError as error:
            expected = (
                "request.ep_experts_must_cover_mesh",
                "request.full_row_major_mesh_required",
                "manifest.full_row_major_mesh_required",
            )
            if (error.code != "moe_compile_sequence_unsupported"
                    or any(reason not in error.message for reason in expected)):
                raise
            result["downstream_error_code"] = error.code
            result["downstream_error_message"] = error.message
            result["downstream_blocking_reasons"] = list(expected)
        else:
            raise RuntimeError("MoE block compiler unexpectedly accepted idle Die")
    result["wall_seconds"] = round(time.monotonic() - start, 3)
    result["python_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"MoE 11x11 probe status={report['status']} output={output}")
    if report["status"] != "blocked_by_full_mesh_compiler_contract":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
