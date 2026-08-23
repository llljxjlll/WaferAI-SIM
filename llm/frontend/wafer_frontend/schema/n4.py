"""Versioned N4 contexts and exact-provenance pass-output bundles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..errors import SchemaError
from .action import FusionPlan, StandaloneCollectivePlan
from .swizzle_plan import FusedPlan, SwizzleFusionPlan
from .common import stable_artifact_id, validate_nonempty, validate_uint64
from .ir0 import (
    CollectiveKind,
    CollectiveWorkload,
    EdgeKind,
    FusionImpl,
    FusionPattern,
    GemmPartition,
    GemmWorkload,
    OpKind,
    ReduceOp,
    TrainStructure,
)
from .ir1 import FusedOpSkeleton, IR1
from .logical import ProfileEntry
from .placed_ir1 import (
    PlacedIR1Bundle,
    PlacedProfileIR1,
    Stage4PlacedIR1,
    TrainPlacedIR1,
    _train_logical_signature,
)
from .policy import PolicySelection, RegistryKind
from .stage4_pd import Stage4PdPlan


FUSION_PARTITION_CONTEXT_SCHEMA_VERSION = (
    "wafer_frontend.fusion_partition_context/v1alpha1"
)
INTERDIE_PLANNING_CONTEXT_SCHEMA_VERSION = (
    "wafer_frontend.inter_die_planning_context/v1alpha2"
)
FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION = (
    "wafer_frontend.fusion_partitioned_ir1_bundle/v1alpha3"
)
INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION = (
    "wafer_frontend.inter_die_plan_bundle/v1alpha4"
)
FUSED_OP_SKELETON_SCHEMA_VERSION = (
    "wafer_frontend.fused_op_skeleton/v1alpha1"
)
STAGE4_FUSION_PARTITIONED_IR1_SCHEMA_VERSION = (
    "wafer_frontend.stage4_fusion_partitioned_ir1/v1alpha2"
)
STAGE4_INTERDIE_PLANNED_IR1_SCHEMA_VERSION = (
    "wafer_frontend.stage4_inter_die_planned_ir1/v1alpha2"
)
TRAIN_FUSION_PARTITIONED_IR1_SCHEMA_VERSION = (
    "wafer_frontend.train_fusion_partitioned_ir1/v1alpha1"
)
TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION = (
    "wafer_frontend.train_interdie_planned_ir1/v1alpha1"
)


class FusionPartitionContract(str, Enum):
    GEMM_RS_ALL_V1 = "gemm_rs_all/v1"


class FusedInterDieContract(str, Enum):
    DIRECT_NAIVE_V1 = "direct_naive/v1"
    SWIZZLE_TOPO_V1 = "swizzle_topo/v1"


class StandaloneInterDieContract(str, Enum):
    DIRECT_ALL_GATHER_V1 = "direct_all_gather/v1"


@dataclass(frozen=True, slots=True)
class FusionPartitionContext:
    schema_version: str
    producer_pass: str
    id: str
    contract: FusionPartitionContract

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        contract: FusionPartitionContract = FusionPartitionContract.GEMM_RS_ALL_V1,
    ) -> "FusionPartitionContext":
        semantic_key = {"contract": contract}
        return cls(
            schema_version=FUSION_PARTITION_CONTEXT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "fusion_partition_context",
                semantic_key,
                schema_version=FUSION_PARTITION_CONTEXT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str = "fusion_partition_context") -> None:
        if self.schema_version != FUSION_PARTITION_CONTEXT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.contract) is not FusionPartitionContract:
            raise SchemaError(
                "must be a FusionPartitionContract",
                path=f"{path}.contract",
            )
        if self.contract is not FusionPartitionContract.GEMM_RS_ALL_V1:
            raise SchemaError("unsupported partition contract", path=f"{path}.contract")
        expected_id = stable_artifact_id(
            "fusion_partition_context",
            {"contract": self.contract},
            schema_version=FUSION_PARTITION_CONTEXT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


@dataclass(frozen=True, slots=True)
class InterDiePlanningContext:
    schema_version: str
    producer_pass: str
    id: str
    fused_policy: PolicySelection
    standalone_policy: PolicySelection
    fused_contract: FusedInterDieContract
    standalone_contract: StandaloneInterDieContract

    @classmethod
    def create(
        cls,
        *,
        producer_pass: str,
        fused_policy: PolicySelection,
        standalone_policy: PolicySelection,
        fused_contract: FusedInterDieContract = FusedInterDieContract.DIRECT_NAIVE_V1,
        standalone_contract: StandaloneInterDieContract = (
            StandaloneInterDieContract.DIRECT_ALL_GATHER_V1
        ),
    ) -> "InterDiePlanningContext":
        semantic_key = {
            "fused_policy": fused_policy,
            "standalone_policy": standalone_policy,
            "fused_contract": fused_contract,
            "standalone_contract": standalone_contract,
        }
        return cls(
            schema_version=INTERDIE_PLANNING_CONTEXT_SCHEMA_VERSION,
            producer_pass=producer_pass,
            id=stable_artifact_id(
                "inter_die_planning_context",
                semantic_key,
                schema_version=INTERDIE_PLANNING_CONTEXT_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str = "inter_die_planning_context") -> None:
        if self.schema_version != INTERDIE_PLANNING_CONTEXT_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        validate_nonempty(self.producer_pass, f"{path}.producer_pass")
        if type(self.fused_policy) is not PolicySelection:
            raise SchemaError(
                "must be a PolicySelection", path=f"{path}.fused_policy"
            )
        self.fused_policy.validate(f"{path}.fused_policy")
        if (
            self.fused_policy.kind is not RegistryKind.INTER_DIE
            or self.fused_policy.name not in ("naive", "swizzle_topo")
        ):
            raise SchemaError(
                "must select a registered naive/swizzle_topo inter-die policy",
                path=f"{path}.fused_policy",
            )
        if type(self.standalone_policy) is not PolicySelection:
            raise SchemaError(
                "must be a PolicySelection", path=f"{path}.standalone_policy"
            )
        self.standalone_policy.validate(f"{path}.standalone_policy")
        if (
            self.standalone_policy.kind
            is not RegistryKind.STANDALONE_COLLECTIVE
            or self.standalone_policy.name != "direct_all_gather"
        ):
            raise SchemaError(
                "must select the registered direct_all_gather policy",
                path=f"{path}.standalone_policy",
            )
        if type(self.fused_contract) is not FusedInterDieContract:
            raise SchemaError(
                "must be a FusedInterDieContract",
                path=f"{path}.fused_contract",
            )
        expected_pairs = {
            ("naive", FusedInterDieContract.DIRECT_NAIVE_V1),
            ("swizzle_topo", FusedInterDieContract.SWIZZLE_TOPO_V1),
        }
        if (self.fused_policy.name, self.fused_contract) not in expected_pairs:
            raise SchemaError(
                "fused policy and contract must be an exact supported pair",
                path=f"{path}.fused_contract",
            )
        if type(self.standalone_contract) is not StandaloneInterDieContract:
            raise SchemaError(
                "must be a StandaloneInterDieContract",
                path=f"{path}.standalone_contract",
            )
        if (
            self.standalone_contract
            is not StandaloneInterDieContract.DIRECT_ALL_GATHER_V1
        ):
            raise SchemaError(
                "unsupported standalone contract",
                path=f"{path}.standalone_contract",
            )
        semantic_key = {
            "fused_policy": self.fused_policy,
            "standalone_policy": self.standalone_policy,
            "fused_contract": self.fused_contract,
            "standalone_contract": self.standalone_contract,
        }
        expected_id = stable_artifact_id(
            "inter_die_planning_context",
            semantic_key,
            schema_version=INTERDIE_PLANNING_CONTEXT_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable artifact id; expected {expected_id!r}",
                path=f"{path}.id",
            )


def _validate_profile_manifest(
    profiles: tuple[ProfileEntry, ...], *, path: str
) -> None:
    if type(profiles) is not tuple:
        raise SchemaError("must be an immutable tuple", path=path)
    if not profiles:
        raise SchemaError("must contain at least one profile", path=path)
    previous_id: str | None = None
    for index, profile in enumerate(profiles):
        profile_path = f"{path}[{index}]"
        if type(profile) is not ProfileEntry:
            raise SchemaError("must be a ProfileEntry", path=profile_path)
        profile.validate(profile_path)
        if previous_id is not None and profile.profile_id <= previous_id:
            raise SchemaError(
                "profile_id values must be strictly increasing",
                path=f"{profile_path}.profile_id",
            )
        previous_id = profile.profile_id
    if abs(math.fsum(profile.weight for profile in profiles) - 1.0) > 1e-12:
        raise SchemaError("weights must sum to 1 within 1e-12", path=path)


def _validate_gemm_rs_candidate(graph: IR1, skeleton: FusedOpSkeleton, *, path: str) -> None:
    nodes = {node.id: node for node in graph.nodes}
    if len(skeleton.member_node_ids) != 2:
        raise SchemaError(
            "gemm_rs_all/v1 requires exactly two members",
            path=f"{path}.member_node_ids",
        )
    gemm = nodes[skeleton.member_node_ids[0]]
    collective = nodes[skeleton.member_node_ids[1]]
    if (
        gemm.kind is not OpKind.GEMM
        or type(gemm.workload) is not GemmWorkload
        or gemm.workload.partition is not GemmPartition.ROW_PARALLEL
    ):
        raise SchemaError(
            "first member must be a row-parallel GEMM",
            path=f"{path}.member_node_ids[0]",
        )
    if (
        collective.kind is not OpKind.COLLECTIVE
        or type(collective.workload) is not CollectiveWorkload
        or collective.workload.collective is not CollectiveKind.REDUCE_SCATTER
        or collective.workload.reduce_op is not ReduceOp.SUM
    ):
        raise SchemaError(
            "second member must be SUM ReduceScatter",
            path=f"{path}.member_node_ids[1]",
        )
    if (
        gemm.instance_id != collective.instance_id
        or gemm.stage != collective.stage
        or gemm.phase is not collective.phase
        or gemm.execution_group_ref != collective.execution_group_ref
    ):
        raise SchemaError(
            "members must share instance, stage, phase, and execution group",
            path=f"{path}.member_node_ids",
        )
    direct_edges = tuple(
        edge
        for edge in graph.edges
        if edge.kind is EdgeKind.DATA
        and edge.source_node == gemm.id
        and edge.destination_node == collective.id
    )
    if len(direct_edges) != 1:
        raise SchemaError(
            "members must have exactly one direct edge",
            path=f"{path}.member_node_ids",
        )


def _validate_skeletons(graph: IR1, *, path: str) -> None:
    selected_candidates = tuple(
        candidate
        for candidate in graph.fusion_candidates
        if candidate.semantic_contract.pattern is FusionPattern.GEMM_RS
    )
    if len(graph.fused_op_skeletons) != len(selected_candidates):
        raise SchemaError(
            "gemm_rs_all/v1 must select every GEMM_RS candidate exactly once",
            path=f"{path}.fused_op_skeletons",
        )
    nodes = {node.id: node for node in graph.nodes}
    occupied: set[str] = set()
    for index, (candidate, skeleton) in enumerate(
        zip(selected_candidates, graph.fused_op_skeletons, strict=True)
    ):
        skeleton_path = f"{path}.fused_op_skeletons[{index}]"
        if skeleton.fusion_ref != candidate.id:
            raise SchemaError(
                "must preserve candidate order and fusion_ref",
                path=f"{skeleton_path}.fusion_ref",
            )
        comparisons = (
            ("member_node_ids", skeleton.member_node_ids, candidate.members),
            ("boundary_inputs", skeleton.boundary_inputs, candidate.boundary_inputs),
            ("boundary_outputs", skeleton.boundary_outputs, candidate.boundary_outputs),
            (
                "semantic_contract",
                skeleton.semantic_contract,
                candidate.semantic_contract,
            ),
        )
        for field_name, actual, expected in comparisons:
            if actual != expected:
                raise SchemaError(
                    "must exactly preserve the source candidate field",
                    path=f"{skeleton_path}.{field_name}",
                )
        if type(skeleton.impl) is not FusionImpl or skeleton.impl is not FusionImpl.NONE:
            raise SchemaError(
                "partition must leave inter-die implementation unplanned",
                path=f"{skeleton_path}.impl",
            )
        skeleton_semantic_key = {
            "fusion_ref": skeleton.fusion_ref,
            "member_node_ids": skeleton.member_node_ids,
            "boundary_inputs": skeleton.boundary_inputs,
            "boundary_outputs": skeleton.boundary_outputs,
            "semantic_contract": skeleton.semantic_contract,
            "impl": skeleton.impl,
        }
        expected_skeleton_id = stable_artifact_id(
            "fused_op_skeleton",
            skeleton_semantic_key,
            schema_version=FUSED_OP_SKELETON_SCHEMA_VERSION,
        )
        if skeleton.id != expected_skeleton_id:
            raise SchemaError(
                f"unstable skeleton id; expected {expected_skeleton_id!r}",
                path=f"{skeleton_path}.id",
            )
        member_instances = {nodes[node_id].instance_id for node_id in skeleton.member_node_ids}
        if member_instances != {skeleton.instance_id}:
            raise SchemaError(
                "instance_id must equal the member instance",
                path=f"{skeleton_path}.instance_id",
            )
        overlap = occupied.intersection(skeleton.member_node_ids)
        if overlap:
            raise SchemaError(
                "selected skeletons must not overlap",
                path=f"{skeleton_path}.member_node_ids",
            )
        occupied.update(skeleton.member_node_ids)
        _validate_gemm_rs_candidate(graph, skeleton, path=skeleton_path)


_PARTITION_PRESERVED_FIELDS = (
    "source_ir0_id",
    "profile",
    "fabric",
    "instances",
    "groups",
    "nodes",
    "values",
    "edges",
    "fusion_candidates",
    "cross_routes",
    "state_accesses",
    "persistent_state_manifest",
)


@dataclass(frozen=True, slots=True)
class FusionPartitionedProfileIR1:
    id: str
    source_placed_entry_id: str
    source_ir1_id: str
    partition_context_id: str
    profile_id: str
    weight: float
    graph: IR1

    @classmethod
    def create(
        cls,
        *,
        source: PlacedProfileIR1,
        context: FusionPartitionContext,
        graph: IR1,
    ) -> "FusionPartitionedProfileIR1":
        semantic_key = {
            "source_placed_entry_id": source.id,
            "source_ir1_id": source.graph.id,
            "partition_context_id": context.id,
            "profile_id": source.profile_id,
            "weight": source.weight,
            "graph_id": graph.id,
        }
        return cls(
            id=stable_artifact_id(
                "fusion_partitioned_profile_ir1",
                semantic_key,
                schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
            ),
            source_placed_entry_id=source.id,
            source_ir1_id=source.graph.id,
            partition_context_id=context.id,
            profile_id=source.profile_id,
            weight=source.weight,
            graph=graph,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_placed_entry_id": self.source_placed_entry_id,
            "source_ir1_id": self.source_ir1_id,
            "partition_context_id": self.partition_context_id,
            "profile_id": self.profile_id,
            "weight": self.weight,
            "graph_id": self.graph.id,
        }

    def validate(self, path: str = "fusion_partitioned_entry") -> None:
        for field_name in (
            "source_placed_entry_id",
            "source_ir1_id",
            "partition_context_id",
            "profile_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.weight) is not float or not math.isfinite(self.weight) or self.weight <= 0.0:
            raise SchemaError("must be a finite positive float", path=f"{path}.weight")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "fusion_partition":
            raise SchemaError(
                "must be produced by fusion_partition",
                path=f"{path}.graph.producer_pass",
            )
        if self.profile_id != self.graph.profile.stable_id():
            raise SchemaError("must equal graph profile id", path=f"{path}.profile_id")
        if self.graph.cross_routes:
            raise SchemaError(
                "must remain empty before inter_die_plan",
                path=f"{path}.graph.cross_routes",
            )
        _validate_skeletons(self.graph, path=f"{path}.graph")
        expected_id = stable_artifact_id(
            "fusion_partitioned_profile_ir1",
            self._semantic_key(),
            schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(
                f"unstable entry id; expected {expected_id!r}",
                path=f"{path}.id",
            )

    def validate_against(
        self,
        source: PlacedProfileIR1,
        context: FusionPartitionContext,
        path: str = "fusion_partitioned_entry",
    ) -> None:
        source.validate("placed_entry")
        context.validate("fusion_partition_context")
        self.validate(path)
        if self.source_placed_entry_id != source.id:
            raise SchemaError("must match source placed entry", path=f"{path}.source_placed_entry_id")
        if self.source_ir1_id != source.graph.id:
            raise SchemaError("must match source placed IR1", path=f"{path}.source_ir1_id")
        if self.partition_context_id != context.id:
            raise SchemaError("must match partition context", path=f"{path}.partition_context_id")
        if self.profile_id != source.profile_id or self.weight != source.weight:
            raise SchemaError("must preserve source profile and weight", path=path)
        for field_name in _PARTITION_PRESERVED_FIELDS:
            if getattr(self.graph, field_name) != getattr(source.graph, field_name):
                raise SchemaError(
                    "partition may only add fused_op_skeletons",
                    path=f"{path}.graph.{field_name}",
                )


@dataclass(frozen=True, slots=True)
class FusionPartitionedIR1Bundle:
    schema_version: str
    producer_pass: str
    id: str
    source_placed_bundle_id: str
    placement_context_id: str
    partition_context_id: str
    source_profiles: tuple[ProfileEntry, ...]
    entries: tuple[FusionPartitionedProfileIR1, ...]

    @classmethod
    def create(
        cls,
        *,
        source: PlacedIR1Bundle,
        context: FusionPartitionContext,
        entries: tuple[FusionPartitionedProfileIR1, ...],
    ) -> "FusionPartitionedIR1Bundle":
        semantic_key = {
            "source_placed_bundle_id": source.id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": context.id,
            "source_profiles": source.source_profiles,
            "entries": entries,
        }
        return cls(
            schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
            producer_pass="fusion_partition",
            id=stable_artifact_id(
                "fusion_partitioned_ir1_bundle",
                semantic_key,
                schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_placed_bundle_id": self.source_placed_bundle_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "source_profiles": self.source_profiles,
            "entries": self.entries,
        }

    def validate(self, path: str = "fusion_partitioned_ir1_bundle") -> None:
        if self.schema_version != FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "fusion_partition":
            raise SchemaError("must be 'fusion_partition'", path=f"{path}.producer_pass")
        for field_name in (
            "source_placed_bundle_id",
            "placement_context_id",
            "partition_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_profile_manifest(self.source_profiles, path=f"{path}.source_profiles")
        if type(self.entries) is not tuple:
            raise SchemaError("must be an immutable tuple", path=f"{path}.entries")
        if len(self.entries) != len(self.source_profiles):
            raise SchemaError("must contain one entry per profile", path=f"{path}.entries")
        source_ids: set[str] = set()
        reference_fabric = None
        reference_groups = None
        reference_skeletons = None
        for index, (profile, entry) in enumerate(zip(self.source_profiles, self.entries)):
            entry_path = f"{path}.entries[{index}]"
            if type(entry) is not FusionPartitionedProfileIR1:
                raise SchemaError("must be a FusionPartitionedProfileIR1", path=entry_path)
            entry.validate(entry_path)
            if entry.profile_id != profile.profile_id or entry.graph.profile != profile.key:
                raise SchemaError("must match source profile order", path=f"{entry_path}.profile_id")
            if entry.weight != profile.weight:
                raise SchemaError("must match source profile weight", path=f"{entry_path}.weight")
            if entry.partition_context_id != self.partition_context_id:
                raise SchemaError("must match bundle partition context", path=f"{entry_path}.partition_context_id")
            if entry.source_placed_entry_id in source_ids:
                raise SchemaError("duplicate source placed entry", path=f"{entry_path}.source_placed_entry_id")
            source_ids.add(entry.source_placed_entry_id)
            if reference_fabric is None:
                reference_fabric = entry.graph.fabric
                reference_groups = entry.graph.groups
                reference_skeletons = entry.graph.fused_op_skeletons
            elif (
                entry.graph.fabric != reference_fabric
                or entry.graph.groups != reference_groups
                or entry.graph.fused_op_skeletons != reference_skeletons
            ):
                raise SchemaError(
                    "all profiles must share fabric, groups, and skeleton partition",
                    path=f"{entry_path}.graph",
                )
        expected_id = stable_artifact_id(
            "fusion_partitioned_ir1_bundle",
            self._semantic_key(),
            schema_version=FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: PlacedIR1Bundle,
        context: FusionPartitionContext,
        path: str = "fusion_partitioned_ir1_bundle",
    ) -> None:
        source.validate("placed_ir1_bundle")
        context.validate("fusion_partition_context")
        self.validate(path)
        if self.source_placed_bundle_id != source.id:
            raise SchemaError("must match source placed bundle", path=f"{path}.source_placed_bundle_id")
        if self.placement_context_id != source.placement_context_id:
            raise SchemaError("must preserve placement context", path=f"{path}.placement_context_id")
        if self.partition_context_id != context.id:
            raise SchemaError("must match partition context", path=f"{path}.partition_context_id")
        if self.source_profiles != source.source_profiles or len(self.entries) != len(source.entries):
            raise SchemaError("must preserve complete profile manifest", path=f"{path}.source_profiles")
        for index, (entry, source_entry) in enumerate(zip(self.entries, source.entries)):
            entry.validate_against(source_entry, context, f"{path}.entries[{index}]")


def _unfused_collective_ids(graph: IR1) -> tuple[str, ...]:
    fused_members = {
        node_id
        for skeleton in graph.fused_op_skeletons
        for node_id in skeleton.member_node_ids
    }
    return tuple(
        node.id
        for node in graph.nodes
        if node.kind is OpKind.COLLECTIVE and node.id not in fused_members
    )


@dataclass(frozen=True, slots=True)
class InterDiePlannedProfile:
    id: str
    source_partitioned_entry_id: str
    source_ir1_id: str
    planning_context_id: str
    profile_id: str
    weight: float
    graph: IR1
    fusion_plans: tuple[FusedPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]

    @classmethod
    def create(
        cls,
        *,
        source: FusionPartitionedProfileIR1,
        context: InterDiePlanningContext,
        fusion_plans: tuple[FusedPlan, ...],
        standalone_plans: tuple[StandaloneCollectivePlan, ...],
    ) -> "InterDiePlannedProfile":
        semantic_key = {
            "source_partitioned_entry_id": source.id,
            "source_ir1_id": source.graph.id,
            "planning_context_id": context.id,
            "profile_id": source.profile_id,
            "weight": source.weight,
            "graph_id": source.graph.id,
            "fusion_plans": fusion_plans,
            "standalone_plans": standalone_plans,
        }
        return cls(
            id=stable_artifact_id(
                "inter_die_planned_profile",
                semantic_key,
                schema_version=INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
            ),
            source_partitioned_entry_id=source.id,
            source_ir1_id=source.graph.id,
            planning_context_id=context.id,
            profile_id=source.profile_id,
            weight=source.weight,
            graph=source.graph,
            fusion_plans=fusion_plans,
            standalone_plans=standalone_plans,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_partitioned_entry_id": self.source_partitioned_entry_id,
            "source_ir1_id": self.source_ir1_id,
            "planning_context_id": self.planning_context_id,
            "profile_id": self.profile_id,
            "weight": self.weight,
            "graph_id": self.graph.id,
            "fusion_plans": self.fusion_plans,
            "standalone_plans": self.standalone_plans,
        }

    def validate(self, path: str = "inter_die_planned_entry") -> None:
        for field_name in (
            "source_partitioned_entry_id",
            "source_ir1_id",
            "planning_context_id",
            "profile_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        if type(self.weight) is not float or not math.isfinite(self.weight) or self.weight <= 0.0:
            raise SchemaError("must be a finite positive float", path=f"{path}.weight")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "fusion_partition":
            raise SchemaError("must embed partition output IR1", path=f"{path}.graph.producer_pass")
        _validate_skeletons(self.graph, path=f"{path}.graph")
        if self.profile_id != self.graph.profile.stable_id():
            raise SchemaError("must equal graph profile id", path=f"{path}.profile_id")
        if self.source_ir1_id != self.graph.id:
            raise SchemaError("must equal embedded graph id", path=f"{path}.source_ir1_id")
        if type(self.fusion_plans) is not tuple or type(self.standalone_plans) is not tuple:
            raise SchemaError("plan collections must be immutable tuples", path=path)
        expected_fused_ids = tuple(item.id for item in self.graph.fused_op_skeletons)
        actual_fused_ids = tuple(plan.fused_op_id for plan in self.fusion_plans)
        if actual_fused_ids != expected_fused_ids:
            raise SchemaError(
                "fusion plans must exactly cover skeletons in source order",
                path=f"{path}.fusion_plans",
            )
        expected_standalone_ids = _unfused_collective_ids(self.graph)
        actual_standalone_ids = tuple(plan.op_id for plan in self.standalone_plans)
        if actual_standalone_ids != expected_standalone_ids:
            raise SchemaError(
                "standalone plans must exactly cover unfused collectives in node order",
                path=f"{path}.standalone_plans",
            )
        plan_ids: set[str] = set()
        for field_name, plans in (
            ("fusion_plans", self.fusion_plans),
            ("standalone_plans", self.standalone_plans),
        ):
            for index, plan in enumerate(plans):
                plan_path = f"{path}.{field_name}[{index}]"
                expected_types = (
                    (FusionPlan, SwizzleFusionPlan)
                    if field_name == "fusion_plans"
                    else (StandaloneCollectivePlan,)
                )
                if type(plan) not in expected_types:
                    raise SchemaError(
                        "must be a valid typed plan",
                        path=plan_path,
                    )
                if plan.id in plan_ids:
                    raise SchemaError("duplicate plan id", path=f"{plan_path}.id")
                plan_ids.add(plan.id)
                if plan.producer_pass != "inter_die_plan":
                    raise SchemaError(
                        "must be produced by inter_die_plan",
                        path=f"{plan_path}.producer_pass",
                    )
                if plan.source_ir1_id != self.graph.id:
                    raise SchemaError(
                        "must reference the embedded partition IR1",
                        path=f"{plan_path}.source_ir1_id",
                    )
                if plan.profile_key != self.graph.profile:
                    raise SchemaError(
                        "must preserve the profile key",
                        path=f"{plan_path}.profile_key",
                    )
                plan.validate_against(self.graph, plan_path)
        expected_id = stable_artifact_id(
            "inter_die_planned_profile",
            self._semantic_key(),
            schema_version=INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable entry id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: FusionPartitionedProfileIR1,
        context: InterDiePlanningContext,
        path: str = "inter_die_planned_entry",
    ) -> None:
        source.validate("fusion_partitioned_entry")
        context.validate("inter_die_planning_context")
        self.validate(path)
        if self.source_partitioned_entry_id != source.id:
            raise SchemaError("must match source partitioned entry", path=f"{path}.source_partitioned_entry_id")
        if self.source_ir1_id != source.graph.id or self.graph != source.graph:
            raise SchemaError("planning must preserve partition IR1 exactly", path=f"{path}.graph")
        if self.planning_context_id != context.id:
            raise SchemaError("must match planning context", path=f"{path}.planning_context_id")
        if self.profile_id != source.profile_id or self.weight != source.weight:
            raise SchemaError("must preserve source profile and weight", path=path)
        expected = (
            (FusionPlan, FusionImpl.NAIVE)
            if context.fused_contract is FusedInterDieContract.DIRECT_NAIVE_V1
            else (SwizzleFusionPlan, FusionImpl.SWIZZLE_TOPO)
        )
        for index, plan in enumerate(self.fusion_plans):
            if type(plan) is not expected[0] or plan.impl is not expected[1]:
                raise SchemaError(
                    "fusion plan type/impl disagrees with planning contract",
                    path=f"{path}.fusion_plans[{index}]",
                )



@dataclass(frozen=True, slots=True)
class InterDiePlanBundle:
    schema_version: str
    producer_pass: str
    id: str
    source_partitioned_bundle_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    source_profiles: tuple[ProfileEntry, ...]
    entries: tuple[InterDiePlannedProfile, ...]

    @classmethod
    def create(
        cls,
        *,
        source: FusionPartitionedIR1Bundle,
        context: InterDiePlanningContext,
        entries: tuple[InterDiePlannedProfile, ...],
    ) -> "InterDiePlanBundle":
        semantic_key = {
            "source_partitioned_bundle_id": source.id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": context.id,
            "source_profiles": source.source_profiles,
            "entries": entries,
        }
        return cls(
            schema_version=INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
            producer_pass="inter_die_plan",
            id=stable_artifact_id(
                "inter_die_plan_bundle",
                semantic_key,
                schema_version=INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_partitioned_bundle_id": self.source_partitioned_bundle_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "source_profiles": self.source_profiles,
            "entries": self.entries,
        }

    def validate(self, path: str = "inter_die_plan_bundle") -> None:
        if self.schema_version != INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "inter_die_plan":
            raise SchemaError("must be 'inter_die_plan'", path=f"{path}.producer_pass")
        for field_name in (
            "source_partitioned_bundle_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
        ):
            validate_nonempty(getattr(self, field_name), f"{path}.{field_name}")
        _validate_profile_manifest(self.source_profiles, path=f"{path}.source_profiles")
        if type(self.entries) is not tuple or len(self.entries) != len(self.source_profiles):
            raise SchemaError("must contain one immutable entry per profile", path=f"{path}.entries")
        source_ids: set[str] = set()
        reference_fabric = None
        reference_groups = None
        reference_skeletons = None
        for index, (profile, entry) in enumerate(zip(self.source_profiles, self.entries)):
            entry_path = f"{path}.entries[{index}]"
            if type(entry) is not InterDiePlannedProfile:
                raise SchemaError("must be an InterDiePlannedProfile", path=entry_path)
            entry.validate(entry_path)
            if entry.profile_id != profile.profile_id or entry.graph.profile != profile.key:
                raise SchemaError("must match source profile order", path=f"{entry_path}.profile_id")
            if entry.weight != profile.weight:
                raise SchemaError("must match source profile weight", path=f"{entry_path}.weight")
            if entry.planning_context_id != self.planning_context_id:
                raise SchemaError("must match bundle planning context", path=f"{entry_path}.planning_context_id")
            if entry.source_partitioned_entry_id in source_ids:
                raise SchemaError("duplicate source partitioned entry", path=f"{entry_path}.source_partitioned_entry_id")
            source_ids.add(entry.source_partitioned_entry_id)
            if reference_fabric is None:
                reference_fabric = entry.graph.fabric
                reference_groups = entry.graph.groups
                reference_skeletons = entry.graph.fused_op_skeletons
            elif (
                entry.graph.fabric != reference_fabric
                or entry.graph.groups != reference_groups
                or entry.graph.fused_op_skeletons != reference_skeletons
            ):
                raise SchemaError(
                    "all profiles must share fabric, groups, and skeleton partition",
                    path=f"{entry_path}.graph",
                )
        expected_id = stable_artifact_id(
            "inter_die_plan_bundle",
            self._semantic_key(),
            schema_version=INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: FusionPartitionedIR1Bundle,
        context: InterDiePlanningContext,
        path: str = "inter_die_plan_bundle",
    ) -> None:
        source.validate("fusion_partitioned_ir1_bundle")
        context.validate("inter_die_planning_context")
        self.validate(path)
        if self.source_partitioned_bundle_id != source.id:
            raise SchemaError("must match source partition bundle", path=f"{path}.source_partitioned_bundle_id")
        if (
            self.placement_context_id != source.placement_context_id
            or self.partition_context_id != source.partition_context_id
        ):
            raise SchemaError("must preserve upstream contexts", path=path)
        if self.planning_context_id != context.id:
            raise SchemaError("must match planning context", path=f"{path}.planning_context_id")
        if self.source_profiles != source.source_profiles or len(self.entries) != len(source.entries):
            raise SchemaError("must preserve complete profile manifest", path=f"{path}.source_profiles")
        for index, (entry, source_entry) in enumerate(zip(self.entries, source.entries)):
            entry.validate_against(source_entry, context, f"{path}.entries[{index}]")


_STAGE4_PARTITION_PRESERVED_FIELDS = (
    "source_ir0_id", "profile", "fabric", "instances", "groups", "nodes",
    "values", "edges", "fusion_candidates", "cross_routes", "state_accesses",
    "persistent_state_manifest", "instance_profiles", "pd_plan_id",
    "node_profiles",
)


@dataclass(frozen=True, slots=True)
class Stage4FusionPartitionedIR1:
    """Exact single-graph Stage 4 fusion-partition carrier."""

    schema_version: str
    producer_pass: str
    id: str
    source_placed_carrier_id: str
    placement_context_id: str
    partition_context_id: str
    pd_plan: Stage4PdPlan
    graph: IR1

    @classmethod
    def create(
        cls,
        *,
        source: Stage4PlacedIR1,
        context: FusionPartitionContext,
        graph: IR1,
    ) -> "Stage4FusionPartitionedIR1":
        semantic_key = {
            "source_placed_carrier_id": source.id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": context.id,
            "pd_plan": source.pd_plan,
            "graph_id": graph.id,
        }
        return cls(
            schema_version=STAGE4_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
            producer_pass="fusion_partition",
            id=stable_artifact_id(
                "stage4_fusion_partitioned_ir1",
                semantic_key,
                schema_version=STAGE4_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
            ),
            source_placed_carrier_id=source.id,
            placement_context_id=source.placement_context_id,
            partition_context_id=context.id,
            pd_plan=source.pd_plan,
            graph=graph,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_placed_carrier_id": self.source_placed_carrier_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "pd_plan": self.pd_plan,
            "graph_id": self.graph.id,
        }

    def validate(self, path: str = "stage4_fusion_partitioned_ir1") -> None:
        if self.schema_version != STAGE4_FUSION_PARTITIONED_IR1_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "fusion_partition":
            raise SchemaError("must be 'fusion_partition'", path=f"{path}.producer_pass")
        for name in ("source_placed_carrier_id", "placement_context_id", "partition_context_id"):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.pd_plan) is not Stage4PdPlan:
            raise SchemaError("must be a Stage4PdPlan", path=f"{path}.pd_plan")
        self.pd_plan.validate(f"{path}.pd_plan")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "fusion_partition":
            raise SchemaError("must embed fusion_partition output", path=f"{path}.graph.producer_pass")
        if self.graph.pd_plan_id != self.pd_plan.id:
            raise SchemaError("graph must reference embedded PD plan", path=f"{path}.graph.pd_plan_id")
        _validate_skeletons(self.graph, path=f"{path}.graph")
        expected_id = stable_artifact_id(
            "stage4_fusion_partitioned_ir1",
            self._semantic_key(),
            schema_version=STAGE4_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: Stage4PlacedIR1,
        context: FusionPartitionContext,
        path: str = "stage4_fusion_partitioned_ir1",
    ) -> None:
        if type(source) is not Stage4PlacedIR1:
            raise SchemaError("must be a Stage4PlacedIR1", path="source")
        if type(context) is not FusionPartitionContext:
            raise SchemaError("must be a FusionPartitionContext", path="fusion_partition_context")
        source.validate("source")
        context.validate("fusion_partition_context")
        self.validate(path)
        if self.source_placed_carrier_id != source.id:
            raise SchemaError("must match source placed carrier", path=f"{path}.source_placed_carrier_id")
        if self.placement_context_id != source.placement_context_id:
            raise SchemaError("must preserve placement context", path=f"{path}.placement_context_id")
        if self.partition_context_id != context.id:
            raise SchemaError("must match partition context", path=f"{path}.partition_context_id")
        if self.pd_plan != source.pd_plan:
            raise SchemaError("must preserve PD plan exactly", path=f"{path}.pd_plan")
        for name in _STAGE4_PARTITION_PRESERVED_FIELDS:
            if getattr(self.graph, name) != getattr(source.graph, name):
                raise SchemaError("partition may only add fused skeletons", path=f"{path}.graph.{name}")


@dataclass(frozen=True, slots=True)
class Stage4InterDiePlannedIR1:
    """Exact single-graph Stage 4 inter-die planning carrier."""

    schema_version: str
    producer_pass: str
    id: str
    source_partitioned_carrier_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    pd_plan: Stage4PdPlan
    graph: IR1
    fusion_plans: tuple[FusedPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]

    @classmethod
    def create(
        cls,
        *,
        source: Stage4FusionPartitionedIR1,
        context: InterDiePlanningContext,
        fusion_plans: tuple[FusedPlan, ...],
        standalone_plans: tuple[StandaloneCollectivePlan, ...],
    ) -> "Stage4InterDiePlannedIR1":
        semantic_key = {
            "source_partitioned_carrier_id": source.id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": context.id,
            "pd_plan": source.pd_plan,
            "graph_id": source.graph.id,
            "fusion_plans": fusion_plans,
            "standalone_plans": standalone_plans,
        }
        return cls(
            schema_version=STAGE4_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
            producer_pass="inter_die_plan",
            id=stable_artifact_id(
                "stage4_inter_die_planned_ir1",
                semantic_key,
                schema_version=STAGE4_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
            ),
            source_partitioned_carrier_id=source.id,
            placement_context_id=source.placement_context_id,
            partition_context_id=source.partition_context_id,
            planning_context_id=context.id,
            pd_plan=source.pd_plan,
            graph=source.graph,
            fusion_plans=fusion_plans,
            standalone_plans=standalone_plans,
        )

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_partitioned_carrier_id": self.source_partitioned_carrier_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "pd_plan": self.pd_plan,
            "graph_id": self.graph.id,
            "fusion_plans": self.fusion_plans,
            "standalone_plans": self.standalone_plans,
        }

    def validate(self, path: str = "stage4_inter_die_planned_ir1") -> None:
        if self.schema_version != STAGE4_INTERDIE_PLANNED_IR1_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "inter_die_plan":
            raise SchemaError("must be 'inter_die_plan'", path=f"{path}.producer_pass")
        for name in (
            "source_partitioned_carrier_id", "placement_context_id",
            "partition_context_id", "planning_context_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.pd_plan) is not Stage4PdPlan:
            raise SchemaError("must be a Stage4PdPlan", path=f"{path}.pd_plan")
        self.pd_plan.validate(f"{path}.pd_plan")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.graph.producer_pass != "fusion_partition":
            raise SchemaError("must preserve partition IR1", path=f"{path}.graph.producer_pass")
        if self.graph.pd_plan_id != self.pd_plan.id:
            raise SchemaError("graph must reference embedded PD plan", path=f"{path}.graph.pd_plan_id")
        _validate_skeletons(self.graph, path=f"{path}.graph")
        if type(self.fusion_plans) is not tuple or type(self.standalone_plans) is not tuple:
            raise SchemaError("plan collections must be immutable tuples", path=path)
        plan_ids: set[str] = set()
        for name, plans, expected_types in (
            ("fusion_plans", self.fusion_plans, (FusionPlan, SwizzleFusionPlan)),
            ("standalone_plans", self.standalone_plans, (StandaloneCollectivePlan,)),
        ):
            for index, plan in enumerate(plans):
                plan_path = f"{path}.{name}[{index}]"
                if type(plan) not in expected_types:
                    raise SchemaError("must be a valid typed plan", path=plan_path)
                if plan.id in plan_ids:
                    raise SchemaError("duplicate plan id", path=f"{plan_path}.id")
                plan_ids.add(plan.id)
                if plan.producer_pass != "inter_die_plan":
                    raise SchemaError("must be produced by inter_die_plan", path=f"{plan_path}.producer_pass")
                if plan.source_ir1_id != self.graph.id:
                    raise SchemaError("must reference embedded partition IR1", path=f"{plan_path}.source_ir1_id")
                plan.validate_against(self.graph, plan_path)
        if tuple(item.fused_op_id for item in self.fusion_plans) != tuple(item.id for item in self.graph.fused_op_skeletons):
            raise SchemaError("fusion plans must exactly cover skeletons", path=f"{path}.fusion_plans")
        if tuple(item.op_id for item in self.standalone_plans) != _unfused_collective_ids(self.graph):
            raise SchemaError("standalone plans must exactly cover unfused collectives", path=f"{path}.standalone_plans")
        expected_id = stable_artifact_id(
            "stage4_inter_die_planned_ir1",
            self._semantic_key(),
            schema_version=STAGE4_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: Stage4FusionPartitionedIR1,
        context: InterDiePlanningContext,
        path: str = "stage4_inter_die_planned_ir1",
    ) -> None:
        if type(source) is not Stage4FusionPartitionedIR1:
            raise SchemaError("must be a Stage4FusionPartitionedIR1", path="source")
        if type(context) is not InterDiePlanningContext:
            raise SchemaError("must be an InterDiePlanningContext", path="inter_die_planning_context")
        source.validate("source")
        context.validate("inter_die_planning_context")
        self.validate(path)
        if self.source_partitioned_carrier_id != source.id:
            raise SchemaError("must match source partition carrier", path=f"{path}.source_partitioned_carrier_id")
        if (
            self.placement_context_id != source.placement_context_id
            or self.partition_context_id != source.partition_context_id
        ):
            raise SchemaError("must preserve upstream contexts", path=path)
        if self.planning_context_id != context.id:
            raise SchemaError("must match planning context", path=f"{path}.planning_context_id")
        if self.pd_plan != source.pd_plan or self.graph != source.graph:
            raise SchemaError("must preserve partition graph and PD plan", path=f"{path}.graph")
        expected = (
            (FusionPlan, FusionImpl.NAIVE)
            if context.fused_contract is FusedInterDieContract.DIRECT_NAIVE_V1
            else (SwizzleFusionPlan, FusionImpl.SWIZZLE_TOPO)
        )
        for index, plan in enumerate(self.fusion_plans):
            if type(plan) is not expected[0] or plan.impl is not expected[1]:
                raise SchemaError(
                    "fusion plan type/impl disagrees with planning contract",
                    path=f"{path}.fusion_plans[{index}]",
                )


@dataclass(frozen=True, slots=True)
class TrainFusionPartitionedIR1:
    schema_version: str
    producer_pass: str
    id: str
    source_train_placed_id: str
    placement_context_id: str
    partition_context_id: str
    train_structure: TrainStructure
    dp_degree: int
    source_replica_ids: tuple[str, ...]
    replicas: tuple[IR1, ...]

    @classmethod
    def create(
        cls,
        *,
        source: TrainPlacedIR1,
        context: FusionPartitionContext,
        replicas: tuple[IR1, ...],
    ) -> "TrainFusionPartitionedIR1":
        semantic_key = {
            "source_train_placed_id": source.id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": context.id,
            "train_structure": source.train_structure,
            "dp_degree": source.dp_degree,
            "source_replica_ids": tuple(item.id for item in source.replicas),
            "replicas": replicas,
        }
        result = cls(
            schema_version=TRAIN_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
            producer_pass="train_fusion_partition",
            id=stable_artifact_id(
                "train_fusion_partitioned_ir1",
                semantic_key,
                schema_version=TRAIN_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate()
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_train_placed_id": self.source_train_placed_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "train_structure": self.train_structure,
            "dp_degree": self.dp_degree,
            "source_replica_ids": self.source_replica_ids,
            "replicas": self.replicas,
        }

    def validate(self, path: str = "train_fusion_partitioned_ir1") -> None:
        if self.schema_version != TRAIN_FUSION_PARTITIONED_IR1_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "train_fusion_partition":
            raise SchemaError("must be 'train_fusion_partition'", path=f"{path}.producer_pass")
        for name in (
            "source_train_placed_id",
            "placement_context_id",
            "partition_context_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.train_structure) is not TrainStructure:
            raise SchemaError("must be a TrainStructure", path=f"{path}.train_structure")
        self.train_structure.validate(f"{path}.train_structure")
        validate_uint64(self.dp_degree, f"{path}.dp_degree")
        if (
            self.dp_degree == 0
            or len(self.source_replica_ids) != self.dp_degree
            or len(self.replicas) != self.dp_degree
        ):
            raise SchemaError("must contain exact DP replica coverage", path=f"{path}.replicas")
        if len(set(self.source_replica_ids)) != self.dp_degree:
            raise SchemaError("source replica ids must be unique", path=f"{path}.source_replica_ids")
        group_ids: set[str] = set()
        node_ids: set[str] = set()
        reference_signature: tuple[object, ...] | None = None
        for index, graph in enumerate(self.replicas):
            graph_path = f"{path}.replicas[{index}]"
            if type(graph) is not IR1:
                raise SchemaError("must be an IR1", path=graph_path)
            graph.validate(graph_path)
            if graph.producer_pass != "fusion_partition":
                raise SchemaError("must be produced by fusion_partition", path=f"{graph_path}.producer_pass")
            _validate_skeletons(graph, path=graph_path)
            if len(graph.groups) != 1 or not graph.groups[0].id.endswith(f"__dp{index}"):
                raise SchemaError("partition graph must preserve canonical replica group", path=f"{graph_path}.groups")
            if graph.groups[0].id in group_ids:
                raise SchemaError("replica groups must be distinct", path=f"{graph_path}.groups[0].id")
            group_ids.add(graph.groups[0].id)
            local_node_ids = {node.id for node in graph.nodes}
            if node_ids.intersection(local_node_ids) or any(
                not node_id.endswith(f"__dp{index}") for node_id in local_node_ids
            ):
                raise SchemaError(
                    "physical node ids must be replica-distinct and canonical",
                    path=f"{graph_path}.nodes",
                )
            node_ids.update(local_node_ids)
            signature = _train_logical_signature(graph)
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise SchemaError(
                    "replicas must preserve one exact logical graph",
                    path=graph_path,
                )
        expected_id = stable_artifact_id(
            "train_fusion_partitioned_ir1",
            self._semantic_key(),
            schema_version=TRAIN_FUSION_PARTITIONED_IR1_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: TrainPlacedIR1,
        context: FusionPartitionContext,
        path: str = "train_fusion_partitioned_ir1",
    ) -> None:
        source.validate("source")
        context.validate("fusion_partition_context")
        self.validate(path)
        if (
            self.source_train_placed_id != source.id
            or self.placement_context_id != source.placement_context_id
            or self.partition_context_id != context.id
            or self.train_structure != source.train_structure
            or self.dp_degree != source.dp_degree
            or self.source_replica_ids != tuple(item.id for item in source.replicas)
        ):
            raise SchemaError("does not preserve train placement provenance", path=path)
        for index, (actual, source_replica) in enumerate(zip(self.replicas, source.replicas)):
            for field_name in _PARTITION_PRESERVED_FIELDS:
                if getattr(actual, field_name) != getattr(source_replica.graph, field_name):
                    raise SchemaError(
                        "partition may only add replica-local fusion skeletons",
                        path=f"{path}.replicas[{index}].{field_name}",
                    )


@dataclass(frozen=True, slots=True)
class TrainReplicaInterDiePlans:
    id: str
    replica_index: int
    source_ir1_id: str
    graph: IR1
    fusion_plans: tuple[FusedPlan, ...]
    standalone_plans: tuple[StandaloneCollectivePlan, ...]

    @classmethod
    def create(
        cls,
        *,
        replica_index: int,
        graph: IR1,
        fusion_plans: tuple[FusedPlan, ...],
        standalone_plans: tuple[StandaloneCollectivePlan, ...],
    ) -> "TrainReplicaInterDiePlans":
        semantic_key = {
            "replica_index": replica_index,
            "source_ir1_id": graph.id,
            "graph": graph,
            "fusion_plans": fusion_plans,
            "standalone_plans": standalone_plans,
        }
        return cls(
            id=stable_artifact_id(
                "train_replica_interdie_plans",
                semantic_key,
                schema_version=TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
            ),
            **semantic_key,
        )

    def validate(self, path: str) -> None:
        validate_uint64(self.replica_index, f"{path}.replica_index")
        validate_nonempty(self.source_ir1_id, f"{path}.source_ir1_id")
        if type(self.graph) is not IR1:
            raise SchemaError("must be an IR1", path=f"{path}.graph")
        self.graph.validate(f"{path}.graph")
        if self.source_ir1_id != self.graph.id:
            raise SchemaError("must reference the replica partition IR1", path=f"{path}.source_ir1_id")
        if tuple(plan.fused_op_id for plan in self.fusion_plans) != tuple(
            skeleton.id for skeleton in self.graph.fused_op_skeletons
        ):
            raise SchemaError("fusion plans must exactly cover replica skeletons", path=f"{path}.fusion_plans")
        if tuple(plan.op_id for plan in self.standalone_plans) != _unfused_collective_ids(self.graph):
            raise SchemaError("standalone plans must exactly cover replica collectives", path=f"{path}.standalone_plans")
        group_id = self.graph.groups[0].id
        plan_ids: set[str] = set()
        for name, plans in (
            ("fusion_plans", self.fusion_plans),
            ("standalone_plans", self.standalone_plans),
        ):
            for index, plan in enumerate(plans):
                plan_path = f"{path}.{name}[{index}]"
                if name == "fusion_plans" and type(plan) not in (FusionPlan, SwizzleFusionPlan):
                    raise SchemaError("must be a FusedPlan", path=plan_path)
                if name == "standalone_plans" and type(plan) is not StandaloneCollectivePlan:
                    raise SchemaError(
                        "must be a StandaloneCollectivePlan", path=plan_path
                    )
                if plan.id in plan_ids:
                    raise SchemaError("plan ids must be unique", path=f"{plan_path}.id")
                plan_ids.add(plan.id)
                if plan.group_ref != group_id:
                    raise SchemaError("plan cannot reference another DP replica group", path=f"{plan_path}.group_ref")
                plan.validate_against(self.graph, plan_path)
        expected_id = stable_artifact_id(
            "train_replica_interdie_plans",
            {
                "replica_index": self.replica_index,
                "source_ir1_id": self.source_ir1_id,
                "graph": self.graph,
                "fusion_plans": self.fusion_plans,
                "standalone_plans": self.standalone_plans,
            },
            schema_version=TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable replica plan id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(self, graph: IR1, path: str) -> None:
        self.validate(path)
        if self.graph != graph:
            raise SchemaError("must preserve the exact replica partition IR1", path=f"{path}.graph")


@dataclass(frozen=True, slots=True)
class TrainInterDiePlannedIR1:
    schema_version: str
    producer_pass: str
    id: str
    source_partitioned_id: str
    placement_context_id: str
    partition_context_id: str
    planning_context_id: str
    train_structure: TrainStructure
    dp_degree: int
    replicas: tuple[TrainReplicaInterDiePlans, ...]

    @classmethod
    def create(
        cls,
        *,
        source: TrainFusionPartitionedIR1,
        context: InterDiePlanningContext,
        replicas: tuple[TrainReplicaInterDiePlans, ...],
    ) -> "TrainInterDiePlannedIR1":
        semantic_key = {
            "source_partitioned_id": source.id,
            "placement_context_id": source.placement_context_id,
            "partition_context_id": source.partition_context_id,
            "planning_context_id": context.id,
            "train_structure": source.train_structure,
            "dp_degree": source.dp_degree,
            "replicas": replicas,
        }
        result = cls(
            schema_version=TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
            producer_pass="train_inter_die_plan",
            id=stable_artifact_id(
                "train_interdie_planned_ir1",
                semantic_key,
                schema_version=TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
            ),
            **semantic_key,
        )
        result.validate_against(source, context)
        return result

    def _semantic_key(self) -> dict[str, object]:
        return {
            "source_partitioned_id": self.source_partitioned_id,
            "placement_context_id": self.placement_context_id,
            "partition_context_id": self.partition_context_id,
            "planning_context_id": self.planning_context_id,
            "train_structure": self.train_structure,
            "dp_degree": self.dp_degree,
            "replicas": self.replicas,
        }

    def validate(self, path: str = "train_interdie_planned_ir1") -> None:
        if self.schema_version != TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION:
            raise SchemaError("unsupported schema version", path=f"{path}.schema_version")
        if self.producer_pass != "train_inter_die_plan":
            raise SchemaError("must be 'train_inter_die_plan'", path=f"{path}.producer_pass")
        for name in (
            "source_partitioned_id",
            "placement_context_id",
            "partition_context_id",
            "planning_context_id",
        ):
            validate_nonempty(getattr(self, name), f"{path}.{name}")
        if type(self.train_structure) is not TrainStructure:
            raise SchemaError("must be a TrainStructure", path=f"{path}.train_structure")
        self.train_structure.validate(f"{path}.train_structure")
        validate_uint64(self.dp_degree, f"{path}.dp_degree")
        if self.dp_degree == 0 or len(self.replicas) != self.dp_degree:
            raise SchemaError("must contain exact DP replica plan coverage", path=f"{path}.replicas")
        if tuple(item.replica_index for item in self.replicas) != tuple(range(self.dp_degree)):
            raise SchemaError("replica plans must use canonical DP order", path=f"{path}.replicas")
        graph_ids: set[str] = set()
        group_ids: set[str] = set()
        node_ids: set[str] = set()
        all_plan_ids: set[str] = set()
        reference_signature: tuple[object, ...] | None = None
        for index, replica in enumerate(self.replicas):
            replica_path = f"{path}.replicas[{index}]"
            if type(replica) is not TrainReplicaInterDiePlans:
                raise SchemaError("must be a TrainReplicaInterDiePlans", path=replica_path)
            replica.validate(replica_path)
            if replica.graph.id in graph_ids:
                raise SchemaError("replica graphs must be distinct", path=f"{replica_path}.graph.id")
            graph_ids.add(replica.graph.id)
            group_id = replica.graph.groups[0].id
            if group_id in group_ids or not group_id.endswith(f"__dp{index}"):
                raise SchemaError("replica plan groups must be distinct and canonical", path=f"{replica_path}.graph.groups")
            group_ids.add(group_id)
            local_node_ids = {node.id for node in replica.graph.nodes}
            if node_ids.intersection(local_node_ids) or any(
                not node_id.endswith(f"__dp{index}") for node_id in local_node_ids
            ):
                raise SchemaError(
                    "physical node ids must be replica-distinct and canonical",
                    path=f"{replica_path}.graph.nodes",
                )
            node_ids.update(local_node_ids)
            signature = _train_logical_signature(replica.graph)
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise SchemaError(
                    "replicas must preserve one exact logical graph",
                    path=f"{replica_path}.graph",
                )
            local_ids = {plan.id for plan in (*replica.fusion_plans, *replica.standalone_plans)}
            if all_plan_ids.intersection(local_ids):
                raise SchemaError("plan ids must be replica-distinct", path=replica_path)
            all_plan_ids.update(local_ids)
        expected_id = stable_artifact_id(
            "train_interdie_planned_ir1",
            self._semantic_key(),
            schema_version=TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION,
        )
        if self.id != expected_id:
            raise SchemaError(f"unstable artifact id; expected {expected_id!r}", path=f"{path}.id")

    def validate_against(
        self,
        source: TrainFusionPartitionedIR1,
        context: InterDiePlanningContext,
        path: str = "train_interdie_planned_ir1",
    ) -> None:
        source.validate("source")
        context.validate("inter_die_planning_context")
        self.validate(path)
        if (
            self.source_partitioned_id != source.id
            or self.placement_context_id != source.placement_context_id
            or self.partition_context_id != source.partition_context_id
            or self.planning_context_id != context.id
            or self.train_structure != source.train_structure
            or self.dp_degree != source.dp_degree
            or len(self.replicas) != source.dp_degree
        ):
            raise SchemaError("does not preserve train partition provenance", path=path)
        for index, (replica, graph) in enumerate(zip(self.replicas, source.replicas)):
            replica.validate_against(graph, f"{path}.replicas[{index}]")
        expected = (
            (FusionPlan, FusionImpl.NAIVE)
            if context.fused_contract is FusedInterDieContract.DIRECT_NAIVE_V1
            else (SwizzleFusionPlan, FusionImpl.SWIZZLE_TOPO)
        )
        for replica_index, replica in enumerate(self.replicas):
            for plan_index, plan in enumerate(replica.fusion_plans):
                if type(plan) is not expected[0] or plan.impl is not expected[1]:
                    raise SchemaError(
                        "fusion plan type/impl disagrees with planning contract",
                        path=f"{path}.replicas[{replica_index}].fusion_plans[{plan_index}]",
                    )


__all__ = [
    "FUSION_PARTITION_CONTEXT_SCHEMA_VERSION",
    "INTERDIE_PLANNING_CONTEXT_SCHEMA_VERSION",
    "FUSION_PARTITIONED_IR1_BUNDLE_SCHEMA_VERSION",
    "INTERDIE_PLAN_BUNDLE_SCHEMA_VERSION",
    "FUSED_OP_SKELETON_SCHEMA_VERSION",
    "STAGE4_FUSION_PARTITIONED_IR1_SCHEMA_VERSION",
    "STAGE4_INTERDIE_PLANNED_IR1_SCHEMA_VERSION",
    "TRAIN_FUSION_PARTITIONED_IR1_SCHEMA_VERSION",
    "TRAIN_INTERDIE_PLANNED_IR1_SCHEMA_VERSION",
    "FusionPartitionContract",
    "FusedInterDieContract",
    "StandaloneInterDieContract",
    "FusionPartitionContext",
    "InterDiePlanningContext",
    "FusionPartitionedProfileIR1",
    "FusionPartitionedIR1Bundle",
    "InterDiePlannedProfile",
    "InterDiePlanBundle",
    "Stage4FusionPartitionedIR1",
    "Stage4InterDiePlannedIR1",
    "TrainFusionPartitionedIR1",
    "TrainInterDiePlannedIR1",
    "TrainReplicaInterDiePlans",
]
