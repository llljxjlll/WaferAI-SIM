"""Production lower/link adapter for flexible-Mesh MeshSlice runtime cases."""

from __future__ import annotations

import json

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.flexible_mesh_compiler import (
    compile_flexible_mesh_workload,
)
from llm.frontend.wafer_frontend.lowering.swizzle_meshslice_fallback import (
    link_meshslice_unfused_fallback_program,
    MeshSliceFallbackReason,
    MeshSliceSelectedPath,
)
from llm.frontend.wafer_frontend.lowering.swizzle_meshslice_standard import (
    decide_meshslice_2d_standard,
    link_meshslice_2d_standard_program,
)
from llm.frontend.wafer_frontend.passes.load_fabric import (
    hbm_address_spaces_from_data,
    physical_fabric_from_data,
)
from llm.frontend.wafer_frontend.passes.meshslice_2d_placement import (
    place_meshslice_2d_ir1,
)
from llm.frontend.wafer_frontend.schema.experiment import (
    PlacementSpec,
    PlacementStrategy,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_runtime import (
    FlexibleMeshRuntimeBaseline,
    FlexibleMeshRuntimeCase,
)
from llm.frontend.wafer_frontend.schema.flexible_mesh_workload import (
    FlexibleMeshSliceOperation,
)
from llm.frontend.wafer_frontend.schema.ir0 import FusionPattern
from llm.frontend.wafer_frontend.schema.swizzle import SwizzleAlgorithm
from llm.frontend.wafer_frontend.schema.placement import PlacementContext

from flexible_mesh_runtime_provider import FlexibleMeshRuntimeExecutable
from flexible_mesh_release_hardware import (
    specialize_p5_large_release_hardware,
)
from swizzle_cases import build_swizzle_integration_cases
from test_swizzle_meshslice_fallback import _fallback_problem
from test_swizzle_meshslice_standard import _planner, _source

_MAX_10X10_MESHSLICE_SRAM_BYTES = 417984

def _runtime_hardware(rows: int, columns: int) -> dict[str, object]:
    return json.loads(specialize_p5_large_release_hardware(rows, columns))


def _runtime_case(rows: int, columns: int):
    hardware = _runtime_hardware(rows, columns)
    context = PlacementContext.create(
        producer_pass="flexible_mesh_runtime_meshslice",
        fabric=physical_fabric_from_data(
            hardware,
            path="runtime.hardware",
        ),
        placement=PlacementSpec(PlacementStrategy.COMPACT, ()),
        hbm_address_spaces=hbm_address_spaces_from_data(
            hardware,
            path="runtime.hardware",
        ),
    )
    ranks = rows * columns
    ir1 = place_meshslice_2d_ir1(
        _source(
            rows=rows,
            columns=columns,
            m_factor=4,
            n_factor=4,
            k_factor=4 if ranks == 1 else 2,
        ),
        context,
    )
    expected_actions = ranks * (3 * (rows + columns - 2) + 1)
    planner = _planner(
        max_actions=max(expected_actions, 1),
        max_buffers=max(3 * ranks, 4),
        efficient_tile_floor=(4, 4, 2),
        max_chunk_count=1,
        sram_budget_bytes=_MAX_10X10_MESHSLICE_SRAM_BYTES,
    )
    decision = decide_meshslice_2d_standard(
        ir1,
        ir1.fused_op_skeletons[0],
        planner.hardware_profile,
        planner.constraints,
    )
    return hardware, ir1, decision


class ProductionMeshSliceRuntimeMaterializer:
    """Materialize only the already-validated MeshSlice executable baselines."""

    def __init__(self, *, mapping_text: str) -> None:
        if type(mapping_text) is not str or not mapping_text:
            raise SchemaError("mapping text must be non-empty", path="meshslice_runtime")
        self._mapping_text = mapping_text
        self._fallback_decisions = {
            case.pattern: case.decision
            for case in build_swizzle_integration_cases()
            if case.pattern in (FusionPattern.GEMM_RS, FusionPattern.GEMM_AR)
        }

    def materialize(
        self,
        case: FlexibleMeshRuntimeCase,
    ) -> FlexibleMeshRuntimeExecutable:
        case.validate()
        compilation = compile_flexible_mesh_workload(case.workload)
        rows = case.workload.mesh.rows
        columns = case.workload.mesh.columns
        hardware, ir1, decision = _runtime_case(rows, columns)
        if case.operation is FlexibleMeshSliceOperation.AG_GEMM:
            if case.selected_baseline is not FlexibleMeshRuntimeBaseline.MESHSLICE_STANDARD:
                raise SchemaError("AG requires MeshSlice standard", path="runtime.case")
            candidate = next(
                (
                    item
                    for item in decision.ranked_candidates
                    if item.algorithm is SwizzleAlgorithm.MESHSLICE_2D_OS
                    and item.chunk_count == 1
                ),
                None,
            )
            if candidate is None:
                raise SchemaError(
                    "MeshSlice standard candidate is absent",
                    path="runtime.lower_link",
                )
            source, _audit = link_meshslice_2d_standard_program(
                ir1,
                decision,
                candidate_ref=candidate.id,
            )
        else:
            pattern = {
                FlexibleMeshSliceOperation.GEMM_RS: FusionPattern.GEMM_RS,
                FlexibleMeshSliceOperation.GEMM_AR: FusionPattern.GEMM_AR,
            }.get(case.operation)
            if (
                pattern is None
                or case.selected_baseline
                is not FlexibleMeshRuntimeBaseline.UNFUSED_FALLBACK
                or case.fallback_reason != "STRICT_TWO_INPUT_REDUCE_ABI"
            ):
                raise SchemaError("fallback selection drifted", path="runtime.case")
            problem, baseline = _fallback_problem(
                ir1,
                self._fallback_decisions[pattern],
            )
            fallback = link_meshslice_unfused_fallback_program(
                ir1,
                problem,
                baseline,
            )
            if (
                fallback.selected_path is not MeshSliceSelectedPath.UNFUSED_FALLBACK
                or fallback.reason
                is not MeshSliceFallbackReason.STRICT_TWO_INPUT_REDUCE_ABI
            ):
                raise SchemaError("typed fallback metadata drifted", path="runtime.lower_link")
            source = fallback.linked_source
        result = FlexibleMeshRuntimeExecutable(
            case_id=case.id,
            compilation_id=compilation.id,
            linked_source_ref=source.id,
            linked_source=source,
            hardware_json=json.dumps(
                hardware,
                sort_keys=True,
                separators=(",", ":"),
            ),
            mapping_text=self._mapping_text,
        )
        # The source is the immediate immutable output of the exact standard
        # or fallback producer above. Public validate() still fully re-derives
        # it; this trusted builder only closes scalar references once.
        if (
            result.case_id != case.id
            or result.linked_source_ref != result.linked_source.id
        ):
            raise SchemaError(
                "trusted executable closure drifted", path="runtime.executable"
            )
        return result


__all__ = ["ProductionMeshSliceRuntimeMaterializer"]
