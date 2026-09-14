"""Trusted trace profiles for supplemental MoE timing evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.flexible_moe import (
    build_round_robin_flexible_moe_spec,
)
from llm.frontend.wafer_frontend.schema.common import DType
from llm.frontend.wafer_frontend.schema.flexible_mesh_release import (
    FlexibleMeshReleaseFamily,
)
from llm.frontend.wafer_frontend.schema.flexible_moe import (
    FlexibleMoeLimits,
    FlexibleMoeMode,
    FlexibleMoeSpec,
    MoeRectStaticTrace,
    MoeRectTraceAssignment,
)
from llm.frontend.wafer_frontend.schema.rect_mesh import RectMeshSpec
from llm.frontend.wafer_frontend.schema.serde import canonical_digest

from flexible_mesh_release_supplemental_coverage import (
    REPRESENTATIVE_RELEASE_SHAPES,
)


class FlexibleMoeSupplementalTrace(str, Enum):
    ALL_LOCAL = "all-local"
    BALANCED = "balanced"
    HOT_EMPTY = "hot-empty"


@dataclass(frozen=True, slots=True)
class FlexibleMoeSupplementalProfile:
    family: FlexibleMeshReleaseFamily
    trace: FlexibleMoeSupplementalTrace
    profile_version: str
    adapter_inputs: tuple[str, ...]

    def validate(self, path: str = "moe_supplemental_profile") -> None:
        if self.family not in (
            FlexibleMeshReleaseFamily.MOE_INFERENCE,
            FlexibleMeshReleaseFamily.MOE_TRAIN,
        ):
            raise SchemaError("requires a MoE family", path=f"{path}.family")
        if type(self.trace) is not FlexibleMoeSupplementalTrace:
            raise SchemaError("requires a typed trace", path=f"{path}.trace")
        if not self.profile_version:
            raise SchemaError("profile version is empty", path=f"{path}.profile_version")
        if self.adapter_inputs != tuple(sorted(set(self.adapter_inputs))):
            raise SchemaError("adapter inputs must be sorted and unique", path=f"{path}.adapter_inputs")

    @property
    def digest(self) -> str:
        self.validate()
        return canonical_digest(self)


REPRESENTATIVE_SHAPES = REPRESENTATIVE_RELEASE_SHAPES


def _profiles() -> tuple[FlexibleMoeSupplementalProfile, ...]:
    result = []
    for family in (
        FlexibleMeshReleaseFamily.MOE_INFERENCE,
        FlexibleMeshReleaseFamily.MOE_TRAIN,
    ):
        for trace in FlexibleMoeSupplementalTrace:
            result.append(FlexibleMoeSupplementalProfile(
                family=family,
                trace=trace,
                profile_version="static_top1_supplemental/v1",
                adapter_inputs=tuple(sorted((
                    f"mode={family.value}",
                    f"trace={trace.value}",
                    "top_k=1",
                    "timing_only=true",
                ))),
            ))
    return tuple(result)


SUPPLEMENTAL_PROFILES = _profiles()


def supplemental_profile(
    family: FlexibleMeshReleaseFamily,
    trace: FlexibleMoeSupplementalTrace,
) -> FlexibleMoeSupplementalProfile:
    matches = tuple(
        item for item in SUPPLEMENTAL_PROFILES
        if item.family is family and item.trace is trace
    )
    if len(matches) != 1:
        raise SchemaError("unknown supplemental profile", path="supplemental_profile")
    return matches[0]


def build_supplemental_spec(
    mesh: RectMeshSpec,
    mode: FlexibleMoeMode,
    trace: FlexibleMoeSupplementalTrace,
) -> FlexibleMoeSpec:
    if trace is FlexibleMoeSupplementalTrace.ALL_LOCAL:
        return build_round_robin_flexible_moe_spec(mesh, mode, routing_shift=0)
    if trace is FlexibleMoeSupplementalTrace.BALANCED:
        return build_round_robin_flexible_moe_spec(
            mesh, mode, routing_shift=0 if mesh.rank_count == 1 else 1,
        )
    if trace is not FlexibleMoeSupplementalTrace.HOT_EMPTY:
        raise SchemaError("unknown supplemental trace", path="trace")
    assignments = tuple(
        MoeRectTraceAssignment(
            token_index=rank,
            source_rank=rank,
            expert_index=0,
            expert_home_rank=0,
            slot_index=rank,
        )
        for rank in range(mesh.rank_count)
    )
    static_trace = MoeRectStaticTrace.create(
        token_count=mesh.rank_count,
        expert_count=mesh.rank_count,
        capacity_per_expert=mesh.rank_count,
        assignments=assignments,
    )
    return FlexibleMoeSpec.create(
        mesh=mesh,
        mode=mode,
        hidden_size=16,
        intermediate_size=32,
        expert_count=mesh.rank_count,
        expert_parallel_degree=mesh.rank_count,
        top_k=1,
        trace_mode="static",
        trace=static_trace,
        limits=FlexibleMoeLimits(),
        expert_dtype=DType.FP16,
        combine_dtype=DType.FP32,
        token_drop=False,
    )


def trusted_spec_builders(family: FlexibleMeshReleaseFamily):
    expected_mode = (
        FlexibleMoeMode.INFERENCE
        if family is FlexibleMeshReleaseFamily.MOE_INFERENCE
        else FlexibleMoeMode.TRAIN
        if family is FlexibleMeshReleaseFamily.MOE_TRAIN
        else None
    )
    if expected_mode is None:
        raise SchemaError("requires a MoE family", path="family")
    result = []
    for trace in FlexibleMoeSupplementalTrace:
        profile = supplemental_profile(family, trace)

        def builder(mesh, mode, *, _trace=trace, _mode=expected_mode):
            if mode is not _mode:
                raise SchemaError("supplemental mode drifted", path="mode")
            return build_supplemental_spec(mesh, mode, _trace)

        result.append((profile.digest, builder))
    return tuple(result)


__all__ = [
    "FlexibleMoeSupplementalProfile",
    "FlexibleMoeSupplementalTrace",
    "REPRESENTATIVE_SHAPES",
    "SUPPLEMENTAL_PROFILES",
    "build_supplemental_spec",
    "supplemental_profile",
    "trusted_spec_builders",
]
