"""Typed executable fallback for MeshSlice post-GEMM collectives.

GEMM_RS and GEMM_AR remain fail-closed in the MeshSlice standard bridge until
their candidate DAG exposes the two typed REDUCE inputs required by the ISA
ABI.  This adapter records that selection explicitly and reuses the exact
UNFUSED lower/link path instead of weakening reduction validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from ..passes.project_unfused_comparison import (
    _project_unfused_comparison_prevalidated,
    build_unfused_comparison_plan,
)
from ..schema.common import stable_artifact_id
from ..schema._validation_session import builder_validation_session
from ..schema.ir0 import FusionPattern
from ..schema.ir1 import IR1
from ..schema.swizzle import SwizzleAlgorithm, SwizzleCandidate, SwizzleProblem
from ..schema.swizzle_unfused_standard import (
    UnfusedComparisonStandardLinkedProgram,
)
from .swizzle_unfused import (
    _build_unfused_comparison_abis_prevalidated,
)
from .swizzle_unfused_standard import (
    _link_unfused_comparison_program_prevalidated,
)


MESHSLICE_EXECUTABLE_FALLBACK_SCHEMA_VERSION = (
    "wafer_frontend.meshslice_executable_fallback/v1alpha1"
)


class MeshSliceSelectedPath(str, Enum):
    MESHSLICE_STANDARD = "MESHSLICE_STANDARD"
    UNFUSED_FALLBACK = "UNFUSED_FALLBACK"


class MeshSliceFallbackReason(str, Enum):
    STRICT_TWO_INPUT_REDUCE_ABI = (
        "MeshSlice RS/AR candidate lacks the exact two-input typed REDUCE "
        "storage contract; selected executable UNFUSED baseline"
    )


@dataclass(frozen=True, slots=True)
class MeshSliceExecutableFallback:
    schema_version: str
    producer_pass: str
    id: str
    selected_path: MeshSliceSelectedPath
    reason: MeshSliceFallbackReason
    pattern: FusionPattern
    linked_source: UnfusedComparisonStandardLinkedProgram

    @classmethod
    def create(
        cls,
        *,
        pattern: FusionPattern,
        linked_source: UnfusedComparisonStandardLinkedProgram,
    ) -> "MeshSliceExecutableFallback":
        semantic = {
            "selected_path": MeshSliceSelectedPath.UNFUSED_FALLBACK,
            "reason": MeshSliceFallbackReason.STRICT_TWO_INPUT_REDUCE_ABI,
            "pattern": pattern,
            "linked_source": linked_source,
        }
        result = cls(
            schema_version=MESHSLICE_EXECUTABLE_FALLBACK_SCHEMA_VERSION,
            producer_pass="meshslice_executable_fallback",
            id=stable_artifact_id(
                "meshslice_executable_fallback",
                semantic,
                schema_version=MESHSLICE_EXECUTABLE_FALLBACK_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "selected_path": self.selected_path,
            "reason": self.reason,
            "pattern": self.pattern,
            "linked_source": self.linked_source,
        }

    def validate(
        self,
        path: str = "meshslice_executable_fallback",
    ) -> None:
        if self.schema_version != MESHSLICE_EXECUTABLE_FALLBACK_SCHEMA_VERSION:
            raise SchemaError(
                "unsupported schema version",
                path=f"{path}.schema_version",
            )
        if self.producer_pass != "meshslice_executable_fallback":
            raise SchemaError(
                "requires exact fallback producer",
                path=f"{path}.producer_pass",
            )
        if (
            self.selected_path is not MeshSliceSelectedPath.UNFUSED_FALLBACK
            or self.reason
            is not MeshSliceFallbackReason.STRICT_TWO_INPUT_REDUCE_ABI
            or self.pattern not in (
                FusionPattern.GEMM_RS,
                FusionPattern.GEMM_AR,
            )
        ):
            raise SchemaError(
                "requires the strict RS/AR UNFUSED fallback selection",
                path=path,
            )
        self.linked_source.validate_against()
        if (
            self.linked_source.plan.pattern is not self.pattern
            or self.linked_source.plan.baseline.algorithm
            is not SwizzleAlgorithm.UNFUSED
        ):
            raise SchemaError(
                "linked source must be the exact pattern's UNFUSED baseline",
                path=f"{path}.linked_source",
            )
        expected = stable_artifact_id(
            "meshslice_executable_fallback",
            self._semantic_key(),
            schema_version=MESHSLICE_EXECUTABLE_FALLBACK_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}",
                path=f"{path}.id",
            )


def _create_prevalidated_fallback(
    *,
    pattern: FusionPattern,
    linked_source: UnfusedComparisonStandardLinkedProgram,
) -> MeshSliceExecutableFallback:
    semantic = {
        "selected_path": MeshSliceSelectedPath.UNFUSED_FALLBACK,
        "reason": MeshSliceFallbackReason.STRICT_TWO_INPUT_REDUCE_ABI,
        "pattern": pattern,
        "linked_source": linked_source,
    }
    result = MeshSliceExecutableFallback(
        schema_version=MESHSLICE_EXECUTABLE_FALLBACK_SCHEMA_VERSION,
        producer_pass="meshslice_executable_fallback",
        id=stable_artifact_id(
            "meshslice_executable_fallback",
            semantic,
            schema_version=MESHSLICE_EXECUTABLE_FALLBACK_SCHEMA_VERSION,
        ),
        **semantic,
    )
    if (
        linked_source.producer_pass != "unfused_comparison_standard_linker"
        or linked_source.plan.pattern is not pattern
        or linked_source.plan.baseline.algorithm is not SwizzleAlgorithm.UNFUSED
        or result.selected_path is not MeshSliceSelectedPath.UNFUSED_FALLBACK
        or result.reason
        is not MeshSliceFallbackReason.STRICT_TWO_INPUT_REDUCE_ABI
        or result.pattern not in (FusionPattern.GEMM_RS, FusionPattern.GEMM_AR)
        or result.linked_source is not linked_source
    ):
        raise SchemaError("prevalidated fallback carrier drifted", path="result")
    return result


def _validate_exact_ir1_group(ir1: IR1, problem: SwizzleProblem) -> None:
    physical_group = next(
        (item for item in ir1.groups if item.id == problem.group.group_ref),
        None,
    )
    if physical_group is None:
        raise SchemaError(
            "problem group is absent from the exact IR1",
            path="problem.group.group_ref",
        )
    die_by_id = {item.id: item for item in ir1.fabric.dies}
    expected_placements = tuple(
        (
            item.rank,
            die_by_id[item.die_id].coord[0],
            die_by_id[item.die_id].coord[1],
        )
        for item in physical_group.placements
    )
    actual_placements = tuple(
        (item.rank, item.x, item.y) for item in problem.group.placements
    )
    expected_routes = tuple(
        (
            item.id,
            item.source_rank,
            item.destination_rank,
            item.die_path,
            item.resource_ids,
        )
        for item in physical_group.embedding.routes
    )
    actual_routes = tuple(
        (
            item.id,
            item.source_rank,
            item.destination_rank,
            item.die_path,
            item.resource_ids,
        )
        for item in problem.group.routes
    )
    if (
        problem.group.logical_shape != physical_group.logical_shape
        or actual_placements != expected_placements
        or actual_routes != expected_routes
    ):
        raise SchemaError(
            "problem group must exactly match IR1 placement and routes",
            path="problem.group",
        )


@builder_validation_session()
def link_meshslice_unfused_fallback_program(
    ir1: IR1,
    problem: SwizzleProblem,
    baseline: SwizzleCandidate,
) -> MeshSliceExecutableFallback:
    """Build the exact typed UNFUSED fallback for MeshSlice RS or AR."""

    ir1.validate("ir1")
    problem.validate("problem")
    baseline.validate("baseline")
    if problem.pattern not in (
        FusionPattern.GEMM_RS,
        FusionPattern.GEMM_AR,
    ):
        raise SchemaError(
            "fallback adapter accepts GEMM_RS or GEMM_AR only",
            path="problem.pattern",
        )
    if (
        SwizzleAlgorithm.MESHSLICE_2D_OS
        not in problem.constraints.allowed_algorithms
        or baseline.algorithm is not SwizzleAlgorithm.UNFUSED
        or baseline.problem_ref != problem.id
        or problem.source_ir1_id != ir1.id
    ):
        raise SchemaError(
            "requires a MeshSlice-capable problem and its exact UNFUSED baseline",
            path="baseline",
        )
    _validate_exact_ir1_group(ir1, problem)
    plan = build_unfused_comparison_plan(ir1, problem, baseline)
    projection = _project_unfused_comparison_prevalidated(ir1, plan)
    core_abi, operand_abi, lowered = _build_unfused_comparison_abis_prevalidated(
        ir1, plan, projection
    )
    linked = _link_unfused_comparison_program_prevalidated(
        ir1,
        plan,
        projection,
        lowered,
        core_abi,
        operand_abi,
    )
    return _create_prevalidated_fallback(
        pattern=problem.pattern,
        linked_source=linked,
    )


__all__ = [
    "link_meshslice_unfused_fallback_program",
    "MeshSliceExecutableFallback",
    "MeshSliceFallbackReason",
    "MeshSliceSelectedPath",
    "MESHSLICE_EXECUTABLE_FALLBACK_SCHEMA_VERSION",
]
