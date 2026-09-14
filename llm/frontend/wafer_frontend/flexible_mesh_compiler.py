"""Bounded v2 compiler for flexible rectangular-Mesh timing workloads.

This compiler intentionally materializes a compact typed execution plan, not
one action per model layer or token.  Workload-specific lowerers may replace a
stage, while the cyclic transport waves and capacity proof remain shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SchemaError
from .passes.flexible_moe import compile_flexible_moe_baseline
from .schema.common import stable_artifact_id, validate_nonempty, validate_uint64
from .schema.flexible_mesh_capacity import FlexibleMeshCapacityDemand
from .schema.flexible_mesh_groups import (
    FlexibleMeshGroup,
    FlexibleMeshGroupKind,
    FlexibleMeshGroupRegistry,
)
from .schema.flexible_mesh_workload import (
    FlexibleMeshSliceMode,
    FlexibleMeshWorkloadKind,
    FlexibleMeshWorkloadSpec,
)
from .schema.flexible_moe import FlexibleMoeExecutablePlan


FLEXIBLE_MESH_PLAN_SCHEMA_VERSION = "wafer_frontend.flexible_mesh_plan/v1alpha1"
FLEXIBLE_MESH_CAPABILITY_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_workload_capability/v1alpha1"
)
FLEXIBLE_MESH_COMPILATION_SCHEMA_VERSION = (
    "wafer_frontend.flexible_mesh_compilation/v1alpha1"
)


class FlexibleMeshStageKind(str, Enum):
    PARAMETER_LOAD = "parameter_load"
    FORWARD = "forward"
    MOE_DISPATCH = "moe_dispatch"
    EXPERT_FORWARD = "expert_forward"
    MOE_COMBINE = "moe_combine"
    LOSS = "loss"
    BACKWARD = "backward"
    MOE_GRAD_DISPATCH = "moe_grad_dispatch"
    EXPERT_BACKWARD = "expert_backward"
    MOE_DX_COMBINE = "moe_dx_combine"
    GRADIENT_SYNC = "gradient_sync"
    SGD = "sgd"
    STATE_STORE = "state_store"


class FlexibleMeshCapabilityStatus(str, Enum):
    VERIFIED = "verified"
    FALLBACK = "fallback"
    NOT_MEASURED = "not_measured"
    NOT_APPLICABLE = "not_applicable"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class FlexibleMeshTransportWave:
    stage: FlexibleMeshStageKind
    group_id: str
    wave_index: int
    sends: tuple[tuple[int, int], ...]
    peak_sessions_per_rank: int

    def validate(
        self,
        group: FlexibleMeshGroup,
        path: str = "flexible_mesh_transport_wave",
    ) -> None:
        if type(self.stage) is not FlexibleMeshStageKind:
            raise SchemaError("must be a stage kind", path=f"{path}.stage")
        if self.group_id != group.id:
            raise SchemaError("must reference exact group", path=f"{path}.group_id")
        if type(self.wave_index) is not int or self.wave_index < 0:
            raise SchemaError("must be non-negative", path=f"{path}.wave_index")
        if type(self.sends) is not tuple:
            raise SchemaError("must be a tuple", path=f"{path}.sends")
        participants = set(group.ranks)
        sources: set[int] = set()
        destinations: set[int] = set()
        for index, pair in enumerate(self.sends):
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or pair[0] == pair[1]
                or pair[0] not in participants
                or pair[1] not in participants
            ):
                raise SchemaError("invalid send pair", path=f"{path}.sends[{index}]")
            sources.add(pair[0])
            destinations.add(pair[1])
        if len(sources) != len(self.sends) or len(destinations) != len(self.sends):
            raise SchemaError(
                "a rank may send and receive at most once per wave",
                path=f"{path}.sends",
            )
        expected_sessions = 0 if not self.sends else 2
        if self.peak_sessions_per_rank != expected_sessions:
            raise SchemaError(
                "session demand must match one-send/one-receive waves",
                path=f"{path}.peak_sessions_per_rank",
            )


@dataclass(frozen=True, slots=True)
class FlexibleMeshExecutionPlan:
    schema_version: str
    id: str
    workload_digest: str
    groups: FlexibleMeshGroupRegistry
    stages: tuple[FlexibleMeshStageKind, ...]
    waves: tuple[FlexibleMeshTransportWave, ...]
    meshslice_mode: FlexibleMeshSliceMode
    demand: FlexibleMeshCapacityDemand

    @classmethod
    def create(
        cls,
        *,
        workload: FlexibleMeshWorkloadSpec,
        groups: FlexibleMeshGroupRegistry,
        stages: tuple[FlexibleMeshStageKind, ...],
        waves: tuple[FlexibleMeshTransportWave, ...],
        demand: FlexibleMeshCapacityDemand,
    ) -> "FlexibleMeshExecutionPlan":
        semantic = {
            "workload_digest": workload.digest,
            "groups": groups,
            "stages": stages,
            "waves": waves,
            "meshslice_mode": workload.meshslice_mode,
            "demand": demand,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_PLAN_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_plan",
                semantic,
                schema_version=FLEXIBLE_MESH_PLAN_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate(workload)
        return result

    def validate(
        self,
        workload: FlexibleMeshWorkloadSpec,
        path: str = "flexible_mesh_plan",
    ) -> None:
        if self.schema_version != FLEXIBLE_MESH_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.id, f"{path}.id")
        workload.validate(f"{path}.workload")
        if self.workload_digest != workload.digest:
            raise SchemaError("workload digest mismatch", path=f"{path}.workload_digest")
        self.groups.validate(f"{path}.groups")
        if (
            self.groups.mesh != workload.mesh
            or self.groups.axis_mapping != workload.axis_mapping
        ):
            raise SchemaError("group registry mismatch", path=f"{path}.groups")
        if (
            type(self.stages) is not tuple
            or not self.stages
            or any(type(stage) is not FlexibleMeshStageKind for stage in self.stages)
            or len(set(self.stages)) != len(self.stages)
        ):
            raise SchemaError("stages must be unique and ordered", path=f"{path}.stages")
        group_by_id = {group.id: group for group in self.groups.groups}
        for index, wave in enumerate(self.waves):
            if type(wave) is not FlexibleMeshTransportWave:
                raise SchemaError("must be a wave", path=f"{path}.waves[{index}]")
            try:
                group = group_by_id[wave.group_id]
            except KeyError as exc:
                raise SchemaError(
                    "unknown group", path=f"{path}.waves[{index}].group_id"
                ) from exc
            wave.validate(group, f"{path}.waves[{index}]")
            if wave.stage not in self.stages:
                raise SchemaError("wave stage is absent", path=f"{path}.waves[{index}].stage")
        if self.meshslice_mode is not workload.meshslice_mode:
            raise SchemaError("MeshSlice mode mismatch", path=f"{path}.meshslice_mode")
        self.demand.require_fits(workload.capacity, f"{path}.demand")
        expected = stable_artifact_id(
            "flexible_mesh_plan",
            {
                "workload_digest": self.workload_digest,
                "groups": self.groups,
                "stages": self.stages,
                "waves": self.waves,
                "meshslice_mode": self.meshslice_mode,
                "demand": self.demand,
            },
            schema_version=FLEXIBLE_MESH_PLAN_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable plan id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMeshWorkloadCapabilityReport:
    schema_version: str
    id: str
    workload_digest: str
    plan_id: str
    mesh_foundation: FlexibleMeshCapabilityStatus
    baseline_plan: FlexibleMeshCapabilityStatus
    optimized_plan: FlexibleMeshCapabilityStatus
    lower_link: FlexibleMeshCapabilityStatus
    program_io: FlexibleMeshCapabilityStatus
    finalizer: FlexibleMeshCapabilityStatus
    runtime: FlexibleMeshCapabilityStatus
    repeatability: FlexibleMeshCapabilityStatus
    performance_calibration: FlexibleMeshCapabilityStatus
    functional_execution: FlexibleMeshCapabilityStatus
    fallback_reasons: tuple[str, ...]
    demand: FlexibleMeshCapacityDemand

    @classmethod
    def create(
        cls,
        workload: FlexibleMeshWorkloadSpec,
        plan: FlexibleMeshExecutionPlan,
    ) -> "FlexibleMeshWorkloadCapabilityReport":
        semantic = {
            "workload_digest": workload.digest,
            "plan_id": plan.id,
            "mesh_foundation": FlexibleMeshCapabilityStatus.VERIFIED,
            "baseline_plan": FlexibleMeshCapabilityStatus.VERIFIED,
            "optimized_plan": FlexibleMeshCapabilityStatus.FALLBACK,
            "lower_link": FlexibleMeshCapabilityStatus.NOT_MEASURED,
            "program_io": FlexibleMeshCapabilityStatus.NOT_MEASURED,
            "finalizer": FlexibleMeshCapabilityStatus.NOT_MEASURED,
            "runtime": FlexibleMeshCapabilityStatus.NOT_MEASURED,
            "repeatability": FlexibleMeshCapabilityStatus.NOT_MEASURED,
            "performance_calibration": FlexibleMeshCapabilityStatus.NOT_MEASURED,
            "functional_execution": FlexibleMeshCapabilityStatus.OUT_OF_SCOPE,
            "fallback_reasons": ("optimized_candidate_not_selected",),
            "demand": plan.demand,
        }
        result = cls(
            schema_version=FLEXIBLE_MESH_CAPABILITY_SCHEMA_VERSION,
            id=stable_artifact_id(
                "flexible_mesh_workload_capability",
                semantic,
                schema_version=FLEXIBLE_MESH_CAPABILITY_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in ("schema_version", "id")
        }

    def validate(
        self, path: str = "flexible_mesh_workload_capability"
    ) -> None:
        if self.schema_version != FLEXIBLE_MESH_CAPABILITY_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.id, f"{path}.id")
        validate_nonempty(self.workload_digest, f"{path}.workload_digest")
        validate_nonempty(self.plan_id, f"{path}.plan_id")
        status_fields = (
            "mesh_foundation",
            "baseline_plan",
            "optimized_plan",
            "lower_link",
            "program_io",
            "finalizer",
            "runtime",
            "repeatability",
            "performance_calibration",
            "functional_execution",
        )
        for name in status_fields:
            if type(getattr(self, name)) is not FlexibleMeshCapabilityStatus:
                raise SchemaError("must be a capability status", path=f"{path}.{name}")
        if (
            self.mesh_foundation is not FlexibleMeshCapabilityStatus.VERIFIED
            or self.baseline_plan is not FlexibleMeshCapabilityStatus.VERIFIED
        ):
            raise SchemaError("foundation and baseline must be verified", path=path)
        if self.functional_execution is not FlexibleMeshCapabilityStatus.OUT_OF_SCOPE:
            raise SchemaError(
                "v1 is timing-only", path=f"{path}.functional_execution"
            )
        if (
            type(self.fallback_reasons) is not tuple
            or any(type(reason) is not str or not reason for reason in self.fallback_reasons)
            or self.fallback_reasons != tuple(sorted(set(self.fallback_reasons)))
        ):
            raise SchemaError(
                "fallback reasons must be unique and canonical",
                path=f"{path}.fallback_reasons",
            )
        self.demand.validate(f"{path}.demand")
        expected = stable_artifact_id(
            "flexible_mesh_workload_capability",
            self._semantic_key(),
            schema_version=FLEXIBLE_MESH_CAPABILITY_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable capability id", path=f"{path}.id")


@dataclass(frozen=True, slots=True)
class FlexibleMeshCompilation:
    schema_version: str
    id: str
    workload: FlexibleMeshWorkloadSpec
    plan: FlexibleMeshExecutionPlan
    workload_plan: FlexibleMoeExecutablePlan | None
    capability_report: FlexibleMeshWorkloadCapabilityReport

    def validate(self, path: str = "flexible_mesh_compilation") -> None:
        if self.schema_version != FLEXIBLE_MESH_COMPILATION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.workload.validate(f"{path}.workload")
        self.plan.validate(self.workload, f"{path}.plan")
        is_moe = self.workload.workload_kind in (
            FlexibleMeshWorkloadKind.MOE_INFER,
            FlexibleMeshWorkloadKind.MOE_TRAIN,
        )
        if is_moe:
            if type(self.workload_plan) is not FlexibleMoeExecutablePlan:
                raise SchemaError(
                    "MoE requires its executable workload plan",
                    path=f"{path}.workload_plan",
                )
            assert self.workload.moe is not None
            self.workload_plan.validate_against(
                self.workload.moe, f"{path}.workload_plan"
            )
        elif self.workload_plan is not None:
            raise SchemaError(
                "Dense common carrier does not attach a workload plan",
                path=f"{path}.workload_plan",
            )
        self.capability_report.validate(f"{path}.capability_report")
        if (
            self.capability_report.workload_digest != self.workload.digest
            or self.capability_report.plan_id != self.plan.id
            or self.capability_report.demand != self.plan.demand
        ):
            raise SchemaError("capability report mismatch", path=f"{path}.capability_report")
        semantic = {
            "workload": self.workload,
            "plan": self.plan,
            "workload_plan": self.workload_plan,
            "capability_report": self.capability_report,
        }
        expected = stable_artifact_id(
            "flexible_mesh_compilation",
            semantic,
            schema_version=FLEXIBLE_MESH_COMPILATION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError("unstable compilation id", path=f"{path}.id")


def _cyclic_waves(
    groups: tuple[FlexibleMeshGroup, ...],
    stage: FlexibleMeshStageKind,
    *,
    repetitions: int = 1,
) -> tuple[FlexibleMeshTransportWave, ...]:
    result: list[FlexibleMeshTransportWave] = []
    wave_index = 0
    for _ in range(repetitions):
        for group in groups:
            ranks = group.ranks
            for delta in range(1, len(ranks)):
                sends = tuple(
                    (rank, ranks[(index + delta) % len(ranks)])
                    for index, rank in enumerate(ranks)
                )
                result.append(
                    FlexibleMeshTransportWave(
                        stage=stage,
                        group_id=group.id,
                        wave_index=wave_index,
                        sends=sends,
                        peak_sessions_per_rank=2,
                    )
                )
                wave_index += 1
    return tuple(result)


def _moe_pair_waves(
    workload: FlexibleMeshWorkloadSpec,
    groups: FlexibleMeshGroupRegistry,
    stage: FlexibleMeshStageKind,
    *,
    reverse: bool,
) -> tuple[FlexibleMeshTransportWave, ...]:
    assert workload.moe is not None
    full = groups.select(FlexibleMeshGroupKind.EP)[0]
    pairs = {
        (
            assignment.expert_home_rank if reverse else assignment.source_rank,
            assignment.source_rank if reverse else assignment.expert_home_rank,
        )
        for assignment in workload.moe.trace.assignments
        if assignment.source_rank != assignment.expert_home_rank
    }
    result: list[FlexibleMeshTransportWave] = []
    for delta in range(1, workload.mesh.rank_count):
        sends = tuple(
            (source, destination)
            for source, destination in sorted(pairs)
            if (destination - source) % workload.mesh.rank_count == delta
        )
        if sends:
            result.append(
                FlexibleMeshTransportWave(
                    stage=stage,
                    group_id=full.id,
                    wave_index=delta - 1,
                    sends=sends,
                    peak_sessions_per_rank=2,
                )
            )
    return tuple(result)


def _stages_for(kind: FlexibleMeshWorkloadKind) -> tuple[FlexibleMeshStageKind, ...]:
    if kind is FlexibleMeshWorkloadKind.DENSE_INFER:
        return (
            FlexibleMeshStageKind.PARAMETER_LOAD,
            FlexibleMeshStageKind.FORWARD,
            FlexibleMeshStageKind.STATE_STORE,
        )
    if kind is FlexibleMeshWorkloadKind.DENSE_TRAIN:
        return (
            FlexibleMeshStageKind.PARAMETER_LOAD,
            FlexibleMeshStageKind.FORWARD,
            FlexibleMeshStageKind.LOSS,
            FlexibleMeshStageKind.BACKWARD,
            FlexibleMeshStageKind.GRADIENT_SYNC,
            FlexibleMeshStageKind.SGD,
            FlexibleMeshStageKind.STATE_STORE,
        )
    base = (
        FlexibleMeshStageKind.PARAMETER_LOAD,
        FlexibleMeshStageKind.FORWARD,
        FlexibleMeshStageKind.MOE_DISPATCH,
        FlexibleMeshStageKind.EXPERT_FORWARD,
        FlexibleMeshStageKind.MOE_COMBINE,
    )
    if kind is FlexibleMeshWorkloadKind.MOE_INFER:
        return base + (FlexibleMeshStageKind.STATE_STORE,)
    return base + (
        FlexibleMeshStageKind.LOSS,
        FlexibleMeshStageKind.MOE_GRAD_DISPATCH,
        FlexibleMeshStageKind.EXPERT_BACKWARD,
        FlexibleMeshStageKind.MOE_DX_COMBINE,
        FlexibleMeshStageKind.BACKWARD,
        FlexibleMeshStageKind.GRADIENT_SYNC,
        FlexibleMeshStageKind.SGD,
        FlexibleMeshStageKind.STATE_STORE,
    )


def compile_flexible_mesh_workload(
    workload: FlexibleMeshWorkloadSpec,
) -> FlexibleMeshCompilation:
    """Build one deterministic, capacity-safe timing execution plan."""

    if type(workload) is not FlexibleMeshWorkloadSpec:
        raise SchemaError("must be a workload spec", path="workload")
    workload.validate()
    groups = FlexibleMeshGroupRegistry.create(workload.mesh, workload.axis_mapping)
    stages = _stages_for(workload.workload_kind)
    waves: tuple[FlexibleMeshTransportWave, ...]
    if workload.workload_kind in (
        FlexibleMeshWorkloadKind.DENSE_INFER,
        FlexibleMeshWorkloadKind.DENSE_TRAIN,
    ):
        tp = groups.select(FlexibleMeshGroupKind.TP)
        waves = _cyclic_waves(tp, FlexibleMeshStageKind.FORWARD)
        if workload.workload_kind is FlexibleMeshWorkloadKind.DENSE_TRAIN:
            waves += _cyclic_waves(tp, FlexibleMeshStageKind.BACKWARD)
            waves += _cyclic_waves(
                groups.select(FlexibleMeshGroupKind.DP),
                FlexibleMeshStageKind.GRADIENT_SYNC,
                repetitions=2,
            )
    else:
        waves = _moe_pair_waves(
            workload, groups, FlexibleMeshStageKind.MOE_DISPATCH, reverse=False
        )
        waves += _moe_pair_waves(
            workload, groups, FlexibleMeshStageKind.MOE_COMBINE, reverse=True
        )
        if workload.workload_kind is FlexibleMeshWorkloadKind.MOE_TRAIN:
            waves += _moe_pair_waves(
                workload,
                groups,
                FlexibleMeshStageKind.MOE_GRAD_DISPATCH,
                reverse=False,
            )
            waves += _moe_pair_waves(
                workload,
                groups,
                FlexibleMeshStageKind.MOE_DX_COMBINE,
                reverse=True,
            )
            waves += _cyclic_waves(
                groups.select(FlexibleMeshGroupKind.EP),
                FlexibleMeshStageKind.GRADIENT_SYNC,
                repetitions=2,
            )

    send_count = sum(len(wave.sends) for wave in waves)
    rank_count = workload.mesh.rank_count
    state_multiplier = 4 if workload.training is not None else 1
    demand = FlexibleMeshCapacityDemand(
        ranks=rank_count,
        actions=rank_count * len(stages) + 2 * send_count,
        buffers=rank_count * (2 + state_multiplier),
        symbolic_records=rank_count * len(stages) + 2 * send_count,
        artifact_file_upper_bound_bytes=(
            rank_count * len(stages) + 2 * send_count
        ) * 256,
        peak_sessions_per_core_per_wave=max(
            (wave.peak_sessions_per_rank for wave in waves), default=0
        ),
        transport_tags=len(waves),
        state_bytes=rank_count * state_multiplier * 4096,
    )
    demand.require_fits(workload.capacity)
    plan = FlexibleMeshExecutionPlan.create(
        workload=workload,
        groups=groups,
        stages=stages,
        waves=waves,
        demand=demand,
    )
    report = FlexibleMeshWorkloadCapabilityReport.create(workload, plan)
    workload_plan = None
    if workload.workload_kind in (
        FlexibleMeshWorkloadKind.MOE_INFER,
        FlexibleMeshWorkloadKind.MOE_TRAIN,
    ):
        assert workload.moe is not None
        workload_plan = compile_flexible_moe_baseline(workload.moe)
    semantic = {
        "workload": workload,
        "plan": plan,
        "workload_plan": workload_plan,
        "capability_report": report,
    }
    result = FlexibleMeshCompilation(
        schema_version=FLEXIBLE_MESH_COMPILATION_SCHEMA_VERSION,
        id=stable_artifact_id(
            "flexible_mesh_compilation",
            semantic,
            schema_version=FLEXIBLE_MESH_COMPILATION_SCHEMA_VERSION,
        ),
        **semantic,
    )
    result.validate()
    return result


__all__ = [
    "FLEXIBLE_MESH_CAPABILITY_SCHEMA_VERSION",
    "FLEXIBLE_MESH_COMPILATION_SCHEMA_VERSION",
    "FLEXIBLE_MESH_PLAN_SCHEMA_VERSION",
    "FlexibleMeshCapabilityStatus",
    "FlexibleMeshCompilation",
    "FlexibleMeshExecutionPlan",
    "FlexibleMeshStageKind",
    "FlexibleMeshTransportWave",
    "FlexibleMeshWorkloadCapabilityReport",
    "compile_flexible_mesh_workload",
]
