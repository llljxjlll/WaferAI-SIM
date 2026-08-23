"""Pattern-aware N4 carrier for a selected inter-die Swizzle program.

The carrier is deliberately separate from the legacy DIRECT ``FusionPlan``.
Every executable witness is bound back to immutable IR-1 math, routing,
synchronization, logical slicing, and temporary-value provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING

from ..errors import SchemaError
from .action import BarrierContract, FusionActionKind, FusionPlan, SyncContract
from .common import DType, ProfileKey, stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import (
    CollectiveWorkload,
    FusionImpl,
    FusionPattern,
    GemmWorkload,
    NodeEffects,
    NodeMath,
    ReduceOp,
)
from .swizzle import (
    SwizzleActionKind,
    SwizzleActionWitness,
    SwizzleAlgorithm,
    SwizzleBufferRequirement,
    SwizzleCandidate,
    SwizzleDecision,
)

if TYPE_CHECKING:
    from .ir1 import IR1


SWIZZLE_DEPLOYMENT_SELECTION_SCHEMA_VERSION = (
    "wafer_frontend.swizzle_deployment_selection/v1alpha1"
)
SWIZZLE_FUSION_PLAN_SCHEMA_VERSION = "wafer_frontend.swizzle_fusion_plan/v1alpha1"


def _owner_profile(ir1: "IR1", instance_id: str, *, path: str) -> ProfileKey:
    if not ir1.instance_profiles:
        if len(ir1.instances) != 1 or ir1.instances[0].id != instance_id:
            raise SchemaError(
                "legacy profile requires the sole IR-1 instance",
                path=path,
            )
        return ir1.profile
    matches = tuple(
        binding.profile
        for binding in ir1.instance_profiles
        if binding.instance_ref == instance_id
    )
    if len(matches) != 1:
        raise SchemaError(
            "owner instance must have exactly one profile binding",
            path=path,
        )
    return matches[0]


class SwizzleDeploymentReason(str, Enum):
    ECONOMIC_DECISION = "economic_decision"
    FORCED_BY_POLICY = "forced_by_policy"


@dataclass(frozen=True, slots=True)
class SwizzleDeploymentSelection:
    """Select one ranked fused candidate without rewriting planner economics."""

    schema_version: str
    id: str
    economic_decision: SwizzleDecision
    candidate: SwizzleCandidate
    reason: SwizzleDeploymentReason

    @classmethod
    def create(
        cls,
        *,
        economic_decision: SwizzleDecision,
        candidate: SwizzleCandidate,
        reason: SwizzleDeploymentReason,
    ) -> "SwizzleDeploymentSelection":
        semantic = {
            "economic_decision": economic_decision,
            "candidate": candidate,
            "reason": reason,
        }
        result = cls(
            schema_version=SWIZZLE_DEPLOYMENT_SELECTION_SCHEMA_VERSION,
            id=stable_artifact_id(
                "swizzle_deployment_selection",
                semantic,
                schema_version=SWIZZLE_DEPLOYMENT_SELECTION_SCHEMA_VERSION,
            ),
            **semantic,
        )
        result.validate()
        return result

    @property
    def economic_decision_ref(self) -> str:
        return self.economic_decision.id

    def _semantic_key(self) -> dict[str, object]:
        return {
            "economic_decision": self.economic_decision,
            "candidate": self.candidate,
            "reason": self.reason,
        }

    def validate(self, path: str = "swizzle_deployment_selection") -> None:
        if self.schema_version != SWIZZLE_DEPLOYMENT_SELECTION_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        self.economic_decision.validate(f"{path}.economic_decision")
        self.candidate.validate(f"{path}.candidate")
        if type(self.reason) is not SwizzleDeploymentReason:
            raise SchemaError(
                "must be a SwizzleDeploymentReason", path=f"{path}.reason"
            )
        ranked = next(
            (
                item
                for item in self.economic_decision.ranked_candidates
                if item.id == self.candidate.id
            ),
            None,
        )
        if ranked is None or ranked != self.candidate:
            raise SchemaError(
                "candidate must exactly equal a ranked economic candidate",
                path=f"{path}.candidate",
            )
        if self.candidate.algorithm is SwizzleAlgorithm.UNFUSED:
            raise SchemaError(
                "deployment candidate must be fused", path=f"{path}.candidate.algorithm"
            )
        is_economic_selection = (
            self.candidate.id == self.economic_decision.selected_candidate_ref
        )
        if self.reason is SwizzleDeploymentReason.ECONOMIC_DECISION:
            if not is_economic_selection:
                raise SchemaError(
                    "economic deployment must preserve the selected candidate",
                    path=f"{path}.candidate",
                )
        elif is_economic_selection:
            raise SchemaError(
                "FORCED_BY_POLICY must select a non-economic candidate",
                path=f"{path}.reason",
            )
        expected = stable_artifact_id(
            "swizzle_deployment_selection",
            self._semantic_key(),
            schema_version=SWIZZLE_DEPLOYMENT_SELECTION_SCHEMA_VERSION,
        )
        if self.id != expected:
            raise SchemaError(
                f"unstable artifact id; expected {expected!r}", path=f"{path}.id"
            )


class SwizzleValueUse(str, Enum):
    READ = "read"
    WRITE = "write"


class SwizzleReductionOriginKind(str, Enum):
    GEMM_ACCUMULATION = "gemm_accumulation"
    COLLECTIVE_REDUCTION = "collective_reduction"


@dataclass(frozen=True, slots=True)
class SwizzleChunkOrigin:
    source_value_ref: str
    axis: int
    chunk_index: int
    logical_offset: tuple[int, ...]
    logical_shape: tuple[int, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.source_value_ref, f"{path}.source_value_ref")
        validate_uint64(self.axis, f"{path}.axis")
        validate_uint64(self.chunk_index, f"{path}.chunk_index")
        if (
            type(self.logical_offset) is not tuple
            or type(self.logical_shape) is not tuple
            or not self.logical_shape
            or len(self.logical_offset) != len(self.logical_shape)
            or self.axis >= len(self.logical_shape)
        ):
            raise SchemaError("must carry equal-rank non-empty slice tuples", path=path)
        for name in ("logical_offset", "logical_shape"):
            for index, extent in enumerate(getattr(self, name)):
                validate_uint64(extent, f"{path}.{name}[{index}]")
                if name == "logical_shape" and extent == 0:
                    raise SchemaError("must be positive", path=f"{path}.{name}[{index}]")


@dataclass(frozen=True, slots=True)
class SwizzleValueOrigin:
    value_ref: str
    use: SwizzleValueUse
    logical_source_ref: str | None
    producer_action_ref: str | None
    local_member_ref: str | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.value_ref, f"{path}.value_ref")
        if type(self.use) is not SwizzleValueUse:
            raise SchemaError("must be a SwizzleValueUse", path=f"{path}.use")
        origins = (
            self.logical_source_ref,
            self.producer_action_ref,
            self.local_member_ref,
        )
        if sum(item is not None for item in origins) != 1:
            raise SchemaError("must carry exactly one exact origin", path=path)
        for name, value in zip(
            ("logical_source_ref", "producer_action_ref", "local_member_ref"),
            origins,
            strict=True,
        ):
            if value is not None:
                validate_nonempty(value, f"{path}.{name}")


@dataclass(frozen=True, slots=True)
class SwizzleComputeOrigin:
    node_ref: str
    workload: GemmWorkload
    math: NodeMath
    effects: NodeEffects
    impl_ref: str
    ir1_input_refs: tuple[str, ...]
    ir1_output_refs: tuple[str, ...]

    def validate(self, path: str) -> None:
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        self.workload.validate(f"{path}.workload")
        self.math.validate(f"{path}.math")
        self.effects.validate(f"{path}.effects")
        validate_nonempty(self.impl_ref, f"{path}.impl_ref")
        for name in ("ir1_input_refs", "ir1_output_refs"):
            refs = getattr(self, name)
            if type(refs) is not tuple or not refs:
                raise SchemaError("must be a non-empty immutable tuple", path=f"{path}.{name}")
            for index, ref in enumerate(refs):
                validate_nonempty(ref, f"{path}.{name}[{index}]")


@dataclass(frozen=True, slots=True)
class SwizzleReductionOrigin:
    node_ref: str
    kind: SwizzleReductionOriginKind
    reduce_op: ReduceOp
    math: NodeMath
    collective_workload: CollectiveWorkload | None

    def validate(self, path: str) -> None:
        validate_nonempty(self.node_ref, f"{path}.node_ref")
        if type(self.kind) is not SwizzleReductionOriginKind:
            raise SchemaError("must be a SwizzleReductionOriginKind", path=f"{path}.kind")
        if self.reduce_op is not ReduceOp.SUM:
            raise SchemaError("Swizzle v1 supports SUM only", path=f"{path}.reduce_op")
        self.math.validate(f"{path}.math")
        if self.kind is SwizzleReductionOriginKind.COLLECTIVE_REDUCTION:
            if type(self.collective_workload) is not CollectiveWorkload:
                raise SchemaError("collective reduction requires its IR-1 workload", path=f"{path}.collective_workload")
            self.collective_workload.validate(f"{path}.collective_workload")
            if self.collective_workload.reduce_op is not self.reduce_op:
                raise SchemaError("reduce op disagrees with collective workload", path=path)
        elif self.collective_workload is not None:
            raise SchemaError("GEMM accumulation cannot claim a collective workload", path=f"{path}.collective_workload")


@dataclass(frozen=True, slots=True)
class SwizzleBoundAction:
    source_action: SwizzleActionWitness
    fusion_kind: FusionActionKind
    member_ref: str
    expected_route: tuple[int, ...]
    chunk_origin: SwizzleChunkOrigin | None
    value_origins: tuple[SwizzleValueOrigin, ...]
    compute_origin: SwizzleComputeOrigin | None
    reduction_origin: SwizzleReductionOrigin | None
    sync: SyncContract

    def validate(self, path: str) -> None:
        self.source_action.validate(f"{path}.source_action")
        expected_kind = FusionActionKind(self.source_action.kind.value)
        if self.fusion_kind is not expected_kind:
            raise SchemaError("fusion action kind must preserve witness kind", path=f"{path}.fusion_kind")
        validate_nonempty(self.member_ref, f"{path}.member_ref")
        self.sync.validate_for_kind(self.fusion_kind, f"{path}.sync")
        if self.source_action.chunk_index is None:
            if self.chunk_origin is not None:
                raise SchemaError("chunkless witness cannot carry chunk origin", path=f"{path}.chunk_origin")
        else:
            if self.chunk_origin is None:
                raise SchemaError("chunked witness requires chunk origin", path=f"{path}.chunk_origin")
            self.chunk_origin.validate(f"{path}.chunk_origin")
            if self.chunk_origin.chunk_index != self.source_action.chunk_index:
                raise SchemaError("chunk index must preserve witness", path=f"{path}.chunk_origin.chunk_index")
        expected_uses = (
            (SwizzleValueUse.READ,) * len(self.source_action.input_refs)
            + (SwizzleValueUse.WRITE,) * len(self.source_action.output_refs)
        )
        expected_refs = self.source_action.input_refs + self.source_action.output_refs
        if tuple(item.use for item in self.value_origins) != expected_uses or tuple(
            item.value_ref for item in self.value_origins
        ) != expected_refs:
            raise SchemaError("value origins must exactly preserve witness I/O order", path=f"{path}.value_origins")
        for index, origin in enumerate(self.value_origins):
            origin.validate(f"{path}.value_origins[{index}]")
        if self.fusion_kind is FusionActionKind.COMP:
            if self.compute_origin is None or self.reduction_origin is not None:
                raise SchemaError("COMP requires only compute origin", path=path)
            self.compute_origin.validate(f"{path}.compute_origin")
        elif self.compute_origin is not None:
            raise SchemaError("only COMP may carry compute origin", path=f"{path}.compute_origin")
        if self.fusion_kind is FusionActionKind.REDUCE:
            if self.reduction_origin is None:
                raise SchemaError("REDUCE requires reduction origin", path=f"{path}.reduction_origin")
            self.reduction_origin.validate(f"{path}.reduction_origin")
        elif self.reduction_origin is not None:
            raise SchemaError("only REDUCE may carry reduction origin", path=f"{path}.reduction_origin")
        if self.fusion_kind in (FusionActionKind.SEND, FusionActionKind.RECV):
            if len(self.expected_route) < 2:
                raise SchemaError("transport requires a physical route", path=f"{path}.expected_route")
        elif self.expected_route:
            raise SchemaError("non-transport action cannot carry a route", path=f"{path}.expected_route")


@dataclass(frozen=True, slots=True)
class SwizzleBoundRankProgram:
    rank: int
    actions: tuple[SwizzleBoundAction, ...]

    def validate(self, path: str) -> None:
        validate_uint64(self.rank, f"{path}.rank")
        if type(self.actions) is not tuple or not self.actions:
            raise SchemaError("must contain actions", path=f"{path}.actions")
        for index, action in enumerate(self.actions):
            if action.source_action.rank != self.rank:
                raise SchemaError("action rank disagrees with program", path=f"{path}.actions[{index}]")
            action.validate(f"{path}.actions[{index}]")


@dataclass(frozen=True, slots=True)
class SwizzleFusionPlan:
    schema_version: str
    producer_pass: str
    id: str
    source_ir1_id: str
    fused_op_id: str
    group_ref: str
    impl: FusionImpl
    profile_key: ProfileKey
    pattern: FusionPattern
    algorithm: SwizzleAlgorithm
    deployment_selection: SwizzleDeploymentSelection
    rank_programs: tuple[SwizzleBoundRankProgram, ...]
    buffer_requirements: tuple[SwizzleBufferRequirement, ...]

    @classmethod
    def create(cls, *, producer_pass: str, **semantic: object) -> "SwizzleFusionPlan":
        result = cls(
            schema_version=SWIZZLE_FUSION_PLAN_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id("swizzle_fusion_plan", semantic, schema_version=SWIZZLE_FUSION_PLAN_SCHEMA_VERSION),
            **semantic,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "source_ir1_id", "fused_op_id", "group_ref", "impl", "profile_key",
            "pattern", "algorithm", "deployment_selection", "rank_programs",
            "buffer_requirements",
        )}

    @property
    def decision(self) -> SwizzleDecision:
        return self.deployment_selection.economic_decision

    @property
    def candidate(self) -> SwizzleCandidate:
        return self.deployment_selection.candidate

    def validate(self, path: str = "swizzle_fusion_plan") -> None:
        if self.schema_version != SWIZZLE_FUSION_PLAN_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        for name in ("producer_pass", "source_ir1_id", "fused_op_id", "group_ref"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if self.impl is not FusionImpl.SWIZZLE_TOPO:
            raise SchemaError("must use impl=swizzle_topo", path=f"{path}.impl")
        self.profile_key.validate(f"{path}.profile_key")
        self.deployment_selection.validate(f"{path}.deployment_selection")
        if self.pattern is not self.candidate.pattern or self.algorithm is not self.candidate.algorithm:
            raise SchemaError("pattern/algorithm must preserve selected candidate", path=path)
        if self.source_ir1_id != self.decision.problem.source_ir1_id or self.fused_op_id != self.decision.problem.fused_op_id or self.group_ref != self.decision.problem.group.group_ref:
            raise SchemaError("plan provenance must exactly preserve decision problem", path=path)
        if self.buffer_requirements != self.candidate.buffer_requirements:
            raise SchemaError("buffer requirements must be lossless", path=f"{path}.buffer_requirements")
        if tuple(program.rank for program in self.rank_programs) != tuple(program.rank for program in self.candidate.rank_programs):
            raise SchemaError("rank programs must preserve candidate rank order", path=f"{path}.rank_programs")
        for index, (bound, source) in enumerate(zip(self.rank_programs, self.candidate.rank_programs, strict=True)):
            bound.validate(f"{path}.rank_programs[{index}]")
            if tuple(action.source_action for action in bound.actions) != source.actions:
                raise SchemaError("bound actions must preserve candidate actions", path=f"{path}.rank_programs[{index}]")
        expected = stable_artifact_id("swizzle_fusion_plan", self._semantic_key(), schema_version=SWIZZLE_FUSION_PLAN_SCHEMA_VERSION)
        if self.id != expected:
            raise SchemaError(f"unstable artifact id; expected {expected!r}", path=f"{path}.id")

    def validate_against(self, ir1: "IR1", path: str = "swizzle_fusion_plan") -> None:
        self.validate(path)
        ir1.validate("ir1")
        if self.source_ir1_id != ir1.id:
            raise SchemaError("plan references a different IR-1", path=f"{path}.source_ir1_id")
        skeleton = next((item for item in ir1.fused_op_skeletons if item.id == self.fused_op_id), None)
        group = next((item for item in ir1.groups if item.id == self.group_ref), None)
        if skeleton is None or group is None:
            raise SchemaError("plan references a missing skeleton/group", path=path)
        if skeleton.semantic_contract.pattern is not self.pattern:
            raise SchemaError("pattern disagrees with fused semantic contract", path=f"{path}.pattern")
        if group.instance_id != skeleton.instance_id:
            raise SchemaError("group and skeleton belong to different instances", path=path)
        if self.profile_key != _owner_profile(ir1, skeleton.instance_id, path=f"{path}.profile_key"):
            raise SchemaError("profile key disagrees with owner instance", path=f"{path}.profile_key")
        nodes = {node.id: node for node in ir1.nodes}
        values = {value.id for value in ir1.values}
        members = set(skeleton.member_node_ids)
        routes = {(route.source_rank, route.destination_rank, route.die_path) for route in group.embedding.routes}
        action_ids = {action.source_action.id for program in self.rank_programs for action in program.actions}
        for program in self.rank_programs:
            for action in program.actions:
                if action.member_ref not in members:
                    raise SchemaError("action origin is not a fused member", path=f"{path}.rank_programs")
                node = nodes[action.member_ref]
                if action.compute_origin is not None and (
                    action.compute_origin.node_ref,
                    action.compute_origin.workload,
                    action.compute_origin.math,
                    action.compute_origin.effects,
                    action.compute_origin.impl_ref,
                    action.compute_origin.ir1_input_refs,
                    action.compute_origin.ir1_output_refs,
                ) != (node.id, node.workload, node.math, node.effects, node.impl_ref, node.inputs, node.outputs):
                    raise SchemaError("compute origin must exactly equal IR-1 member", path=f"{path}.rank_programs")
                if action.reduction_origin is not None:
                    origin = action.reduction_origin
                    if origin.node_ref != node.id or origin.math != node.math:
                        raise SchemaError("reduction origin must exactly equal IR-1 member", path=f"{path}.rank_programs")
                    if origin.collective_workload is not None and origin.collective_workload != node.workload:
                        raise SchemaError("collective reduction workload disagrees with IR-1", path=f"{path}.rank_programs")
                witness = action.source_action
                if witness.peer_rank is not None:
                    endpoints = (program.rank, witness.peer_rank) if witness.kind is SwizzleActionKind.SEND else (witness.peer_rank, program.rank)
                    if (*endpoints, action.expected_route) not in routes:
                        raise SchemaError("action route disagrees with IR-1 group embedding", path=f"{path}.rank_programs")
                for origin in action.value_origins:
                    if origin.logical_source_ref is not None and origin.logical_source_ref not in values:
                        raise SchemaError("logical value origin is absent from IR-1", path=f"{path}.rank_programs")
                    if origin.producer_action_ref is not None and origin.producer_action_ref not in action_ids:
                        raise SchemaError("temporary producer action is absent from plan", path=f"{path}.rank_programs")
                    if origin.local_member_ref is not None and origin.local_member_ref not in members:
                        raise SchemaError("local operand origin is not a fused member", path=f"{path}.rank_programs")


FusedPlan = FusionPlan | SwizzleFusionPlan


__all__ = [
    "FusedPlan",
    "SWIZZLE_DEPLOYMENT_SELECTION_SCHEMA_VERSION",
    "SWIZZLE_FUSION_PLAN_SCHEMA_VERSION",
    "SwizzleBoundAction",
    "SwizzleBoundRankProgram",
    "SwizzleChunkOrigin",
    "SwizzleComputeOrigin",
    "SwizzleDeploymentReason",
    "SwizzleDeploymentSelection",
    "SwizzleFusionPlan",
    "SwizzleReductionOrigin",
    "SwizzleReductionOriginKind",
    "SwizzleValueOrigin",
    "SwizzleValueUse",
]
